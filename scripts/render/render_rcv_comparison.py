"""
Generate side-by-side comparison videos of Layer R / Layer C / Layer V meshes
for the baseline test models.

Reads persisted .ply files from
    results/baseline_experiments/EXP6_corep_recon/work_{model}_{res}/
and writes one MP4 per model to
    results/baseline_experiments/videos/{model}_comparison.mp4

Animation phases (15 s @ 24 fps = 360 frames total):
    Phase 1 (0-3s,   frames 0-71):   full mesh,          yaw +120 deg
    Phase 2 (3-6s,   frames 72-143): slice reveal,       yaw +120 deg
    Phase 3 (6-12s,  frames 144-287): slice at center,   yaw +240 deg
    Phase 4 (12-15s, frames 288-359): slice retract,     yaw +120 deg

The slice plane has fixed normal (-1, +1, +1)/sqrt(3) and moves along that
axis from outer corner (-0.9, +0.9, +0.9) to center (0, 0, 0) and back.

Usage:
    .venv/bin/python scripts/render/render_rcv_comparison.py
    .venv/bin/python scripts/render/render_rcv_comparison.py --models icosphere
"""
import argparse
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass

import numpy as np

# Project root on sys.path so trellis2 and scripts.* imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


DEFAULT_MODELS = [
    'bowl', 'icosphere', 'parallel_planes', 'nested_spheres',
    'bugatti', 'spacesuit', 'helmet',
]

MESH_ROOT = 'results/baseline_experiments/EXP6_corep_recon'
DEFAULT_OUT_DIR = 'results/baseline_experiments/videos'
DEFAULT_CACHE_ROOT = 'tmp/rcv_cache'

LABELS = ['Layer R (O-Voxel)', 'Layer C (CoReP)', 'Layer V (O-Voxel + VAE)']

SLICE_NORMAL = np.array([-1.0, 1.0, 1.0]) / math.sqrt(3.0)
SLICE_OUTER = np.array([-0.9, 0.9, 0.9])
SLICE_CENTER = np.array([0.0, 0.0, 0.0])


# Module-level globals populated by worker init. Workers are forked child
# processes used by render_comparison_video_parallel to overlap CPU-bound
# mesh slicing across many cores while the main process serializes GPU
# rendering. Storing the meshes once per worker (via initializer) avoids
# paying the pickling cost on every task.
_WORKER_MESHES = None
_WORKER_CONFIG = None


@dataclass
class AnimConfig:
    fps: int = 24
    duration: float = 15.0
    resolution: int = 768       # per-layer render resolution
    mesh_resolution: int = 512  # which EXP6 work_dir to read from
    pitch_deg: float = 25.0
    fov_deg: float = 40.0
    label_band_px: int = 60

    @property
    def total_frames(self) -> int:
        return int(round(self.fps * self.duration))

    @property
    def phase_boundaries(self) -> list:
        """Return frame indices [p1_end, p2_end, p3_end, p4_end]."""
        f = self.fps
        return [
            int(round(3.0 * f)),   # end of phase 1
            int(round(6.0 * f)),   # end of phase 2
            int(round(12.0 * f)),  # end of phase 3
            int(round(15.0 * f)),  # end of phase 4 (== total_frames)
        ]


def ease_in_out_quad(t: float) -> float:
    """Smooth easing: 0 at t=0, 1 at t=1, zero derivative at both ends."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def compute_phase_state(frame_idx: int, config: AnimConfig) -> tuple:
    """Return (slice_origin: np.ndarray[3], camera_yaw_rad: float) for a frame.

    Yaw schedule (total 600 degrees over the whole video, linear per phase):
        Phase 1: 0   ->  120 deg   (72 frames)
        Phase 2: 120 ->  240 deg   (72 frames)
        Phase 3: 240 ->  480 deg   (144 frames)
        Phase 4: 480 ->  600 deg   (72 frames)
    """
    p1, p2, p3, p4 = config.phase_boundaries  # [72, 144, 288, 360] at 24fps/15s

    if frame_idx < p1:
        # Phase 1: full mesh, slice at outer corner (no-op).
        phase = 1
        progress = frame_idx / max(p1 - 1, 1)
        slice_origin = SLICE_OUTER.copy()
        yaw_deg = 0.0 + 120.0 * progress
    elif frame_idx < p2:
        # Phase 2: slice reveals from corner to center with easing.
        phase = 2
        local = frame_idx - p1
        progress = local / max(p2 - p1 - 1, 1)
        t = ease_in_out_quad(progress)
        slice_origin = SLICE_OUTER * (1.0 - t) + SLICE_CENTER * t
        yaw_deg = 120.0 + 120.0 * progress
    elif frame_idx < p3:
        # Phase 3: slice pinned at center, camera rotates.
        phase = 3
        local = frame_idx - p2
        progress = local / max(p3 - p2 - 1, 1)
        slice_origin = SLICE_CENTER.copy()
        yaw_deg = 240.0 + 240.0 * progress
    else:
        # Phase 4: slice retracts from center to corner with easing.
        phase = 4
        local = frame_idx - p3
        progress = local / max(p4 - p3 - 1, 1)
        t = ease_in_out_quad(progress)
        slice_origin = SLICE_CENTER * (1.0 - t) + SLICE_OUTER * t
        yaw_deg = 480.0 + 120.0 * progress

    return slice_origin, math.radians(yaw_deg)


def _align_to_reference(mesh, ref_center: np.ndarray, ref_extent: float):
    """Translate + scale `mesh` so its bbox center matches ref_center and its
    max bbox extent matches ref_extent. Returns a new Trimesh (does not mutate
    the input)."""
    import trimesh
    bounds = mesh.bounds  # (2, 3)
    center = (bounds[0] + bounds[1]) / 2.0
    extent = (bounds[1] - bounds[0]).max()
    if extent < 1e-9:
        # Degenerate mesh, keep it alone.
        return mesh
    scale = ref_extent / extent
    new_verts = (mesh.vertices - center) * scale + ref_center
    return trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)


def load_layer_meshes(model: str, mesh_resolution: int):
    """Load Layer R / C / V meshes for a model, aligned to shared bbox.

    Returns (mesh_r, mesh_c, mesh_v, radius) where:
        - all three meshes are trimesh.Trimesh objects in [-0.5, 0.5]^3
        - radius is a camera radius that fits the unit bbox with 20% margin
    Raises FileNotFoundError if any layer .ply is missing.
    """
    import trimesh
    work_dir = os.path.join(MESH_ROOT, f'work_{model}_{mesh_resolution}')
    paths = {
        'R': os.path.join(work_dir, 'layer_r.ply'),
        'C': os.path.join(work_dir, 'corep_recon.ply'),
        'V': os.path.join(work_dir, 'layer_v.ply'),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing Layer {k} mesh for {model}: {p}')

    raw_r = trimesh.load(paths['R'], process=False, force='mesh')
    raw_c = trimesh.load(paths['C'], process=False, force='mesh')
    raw_v = trimesh.load(paths['V'], process=False, force='mesh')

    # Use Layer C bounds as the reference frame (CoReP uses a well-defined
    # normalize_mesh step, so its bbox is most trustworthy).
    c_bounds = raw_c.bounds
    c_center = (c_bounds[0] + c_bounds[1]) / 2.0
    c_extent = (c_bounds[1] - c_bounds[0]).max()

    # Step 1: align R and V to C's bbox.
    aligned_r = _align_to_reference(raw_r, c_center, c_extent)
    aligned_c = raw_c  # already the reference
    aligned_v = _align_to_reference(raw_v, c_center, c_extent)

    # Step 2: normalize all three to [-0.5, 0.5]^3 (translate c_center -> 0,
    # scale c_extent -> 1.0).
    def _normalize(mesh):
        new_verts = (mesh.vertices - c_center) / c_extent
        return trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)

    mesh_r = _normalize(aligned_r)
    mesh_c = _normalize(aligned_c)
    mesh_v = _normalize(aligned_v)

    # Camera radius: unit bbox corner distance is sqrt(3)/2 ~= 0.866,
    # with 20% margin and FOV~40 degrees we want r such that tan(fov/2)*r > 0.6.
    # Empirically r=1.8 places everything comfortably in frame at fov=40.
    radius = 1.8
    return mesh_r, mesh_c, mesh_v, radius


def slice_mesh_for_frame(mesh, slice_origin: np.ndarray):
    """Slice `mesh` with a plane at `slice_origin` with normal SLICE_NORMAL,
    keeping the half-space opposite to the reveal direction.

    The reveal direction is from corner (-0.9, +0.9, +0.9) toward the origin,
    i.e. along SLICE_NORMAL = (-1, +1, +1)/sqrt(3). We want to keep the side
    that does NOT contain the corner (the "revealed" side), which is where
    (v - origin) . SLICE_NORMAL < 0. trimesh's slice_mesh_plane keeps the
    side where (v - origin) . plane_normal > 0, so we pass -SLICE_NORMAL.

    Uses cap=True to fill the exposed cross-section with a flat triangulated
    cap, producing the solid cross-section look the design calls for. This
    requires shapely + a polygon triangulation engine (mapbox-earcut).

    Returns (sliced_mesh, was_empty: bool). When was_empty is True, the
    caller receives the original mesh as a fallback.
    """
    import trimesh.intersections

    try:
        sliced = trimesh.intersections.slice_mesh_plane(
            mesh,
            plane_normal=-SLICE_NORMAL,
            plane_origin=slice_origin,
            cap=True,
        )
    except Exception:
        return mesh, True

    if sliced is None or len(sliced.vertices) == 0 or len(sliced.faces) == 0:
        return mesh, True
    return sliced, False


def render_single_mesh(mesh, yaw_rad: float, config: AnimConfig, radius: float) -> np.ndarray:
    """Render a single view of `mesh` and return a (H, W, 3) uint8 normal map.

    `yaw_rad` and `pitch` are in radians internally (render_utils expects rad).
    `radius` comes from load_layer_meshes.
    """
    import torch
    from trellis2.utils import render_utils
    from trellis2.representations import Mesh as TrellisMesh

    device = 'cuda'
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
    t_mesh = TrellisMesh(vertices=verts, faces=faces)

    pitch_rad = math.radians(config.pitch_deg)
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw_rad], [pitch_rad], radius, config.fov_deg,
    )
    result = render_utils.render_frames(
        t_mesh, extr, intr,
        options={'resolution': config.resolution, 'bg_color': (1.0, 1.0, 1.0)},
        verbose=False,
    )
    img = result['normal'][0]  # (H, W, 3) uint8
    return img


def compose_frame(img_r: np.ndarray, img_c: np.ndarray, img_v: np.ndarray,
                  config: AnimConfig) -> np.ndarray:
    """Horizontally composite the three per-layer renders with a bottom label
    band. Returns (H + band, W * 3, 3) uint8.

    Layout:
        [R image][C image][V image]   <- 768 x 768 each
        [Layer R] [Layer C] [Layer V]  <- 60 px label band
    """
    from PIL import Image, ImageDraw, ImageFont

    W = config.resolution
    H = config.resolution
    band = config.label_band_px

    canvas = Image.new('RGB', (W * 3, H + band), 'white')
    canvas.paste(Image.fromarray(img_r), (0, 0))
    canvas.paste(Image.fromarray(img_c), (W, 0))
    canvas.paste(Image.fromarray(img_v), (W * 2, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 28)
    except OSError:
        # Fallback to the default bitmap font if DejaVu is not installed.
        font = ImageFont.load_default()

    for i, label in enumerate(LABELS):
        # Measure with textbbox for precise centering (supported on Pillow >= 8.0).
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
        x = W * i + (W - text_w) // 2
        y = H + (band - text_h) // 2
        draw.text((x, y), label, fill='black', font=font)

    return np.asarray(canvas)


def render_comparison_video(model: str, out_path: str, config: AnimConfig) -> bool:
    """Render one comparison video for a single model. Returns True on success.
    On any per-frame error, skip that frame with a blank placeholder and keep
    going. On fatal error (e.g. missing meshes), log and return False."""
    import imageio.v2 as imageio

    try:
        mesh_r, mesh_c, mesh_v, radius = load_layer_meshes(model, config.mesh_resolution)
    except FileNotFoundError as e:
        print(f'  [SKIP] {model}: {e}')
        return False

    print(f'  Rendering {model}: {config.total_frames} frames '
          f'(R={len(mesh_r.faces)} C={len(mesh_c.faces)} V={len(mesh_v.faces)} faces)')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    writer = imageio.get_writer(
        out_path, fps=config.fps, codec='libx264', quality=8,
        macro_block_size=1,  # tolerate non-multiple-of-16 frame sizes
    )
    try:
        for frame_idx in range(config.total_frames):
            try:
                slice_origin, yaw_rad = compute_phase_state(frame_idx, config)
                sliced_r, _ = slice_mesh_for_frame(mesh_r, slice_origin)
                sliced_c, _ = slice_mesh_for_frame(mesh_c, slice_origin)
                sliced_v, _ = slice_mesh_for_frame(mesh_v, slice_origin)
                img_r = render_single_mesh(sliced_r, yaw_rad, config, radius)
                img_c = render_single_mesh(sliced_c, yaw_rad, config, radius)
                img_v = render_single_mesh(sliced_v, yaw_rad, config, radius)
                frame = compose_frame(img_r, img_c, img_v, config)
            except Exception:
                # Log the traceback but insert a blank frame so the video
                # stays time-synced and the loop continues.
                traceback.print_exc()
                h = config.resolution + config.label_band_px
                w = config.resolution * 3
                frame = np.full((h, w, 3), 255, dtype=np.uint8)

            writer.append_data(frame)

            if frame_idx % 30 == 0:
                print(f'    frame {frame_idx}/{config.total_frames}')
    finally:
        writer.close()

    print(f'  [OK] wrote {out_path}')
    return True


# -----------------------------------------------------------------------------
# Parallel variant: CPU-parallel slicing + serial GPU rendering + PNG cache for
# resume. See render_comparison_video_parallel for the main orchestration.
# -----------------------------------------------------------------------------

def _worker_init(mesh_data_tuple, config_dict):
    """Initializer for forked worker processes. Called once per worker. Sets
    module-level globals with the three trimesh objects (R, C, V) rebuilt from
    the vertex/face arrays and the AnimConfig rebuilt from its dict."""
    import trimesh
    global _WORKER_MESHES, _WORKER_CONFIG
    meshes = []
    for verts, faces in mesh_data_tuple:
        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
    _WORKER_MESHES = meshes  # [mesh_r, mesh_c, mesh_v]
    _WORKER_CONFIG = AnimConfig(**config_dict)


def _worker_slice_frame(frame_idx):
    """Compute the slice origin for `frame_idx` and slice all three layers.
    Returns lightweight numpy tuples so the result pickles cheaply back to the
    main process. Runs in worker (no CUDA)."""
    slice_origin, _ = compute_phase_state(frame_idx, _WORKER_CONFIG)
    sliced_r, _ = slice_mesh_for_frame(_WORKER_MESHES[0], slice_origin)
    sliced_c, _ = slice_mesh_for_frame(_WORKER_MESHES[1], slice_origin)
    sliced_v, _ = slice_mesh_for_frame(_WORKER_MESHES[2], slice_origin)
    return (
        frame_idx,
        (np.asarray(sliced_r.vertices, dtype=np.float32),
         np.asarray(sliced_r.faces, dtype=np.int32)),
        (np.asarray(sliced_c.vertices, dtype=np.float32),
         np.asarray(sliced_c.faces, dtype=np.int32)),
        (np.asarray(sliced_v.vertices, dtype=np.float32),
         np.asarray(sliced_v.faces, dtype=np.int32)),
    )


def _combine_pngs_to_mp4(cache_dir: str, out_path: str, config: AnimConfig):
    """Assemble a continuous PNG sequence 0000.png..NNNN.png into an MP4 via
    imageio+ffmpeg. Missing frames (should not happen post-render) are filled
    with white to keep the video time-synced."""
    import imageio.v2 as imageio
    from PIL import Image

    with imageio.get_writer(
        out_path, fps=config.fps, codec='libx264', quality=8,
        macro_block_size=1,
    ) as writer:
        for frame_idx in range(config.total_frames):
            png_path = os.path.join(cache_dir, f'{frame_idx:04d}.png')
            if os.path.exists(png_path):
                frame = np.asarray(Image.open(png_path).convert('RGB'))
            else:
                h = config.resolution + config.label_band_px
                w = config.resolution * 3
                frame = np.full((h, w, 3), 255, dtype=np.uint8)
            writer.append_data(frame)


def render_comparison_video_parallel(model: str, out_path: str, config: AnimConfig,
                                     num_workers: int = 16,
                                     resume: bool = True,
                                     cache_root: str = DEFAULT_CACHE_ROOT,
                                     frame_start: int = None,
                                     frame_end: int = None,
                                     assemble: bool = True) -> bool:
    """Parallel renderer: N worker processes slice frames concurrently (CPU),
    main process renders on GPU serially (avoids GPU contention), writes each
    completed frame as a PNG for resume. Final step assembles PNGs -> MP4.

    Args:
        model: baseline model name (e.g. 'bowl').
        out_path: final MP4 path.
        config: AnimConfig.
        num_workers: CPU worker processes for slicing.
        resume: if True, skip frames whose PNG is already cached.
        cache_root: directory under which per-model PNG caches live.
        frame_start: inclusive start frame for this call (default 0). Used by
            the multi-GPU driver to partition frames across GPU processes.
        frame_end: exclusive end frame (default total_frames). Combined with
            frame_start to restrict which frames this call renders.
        assemble: if False, skip the final PNG -> MP4 assembly. Used during
            multi-GPU partitioned runs where only the orchestrator assembles
            the MP4 after all partial renders finish.

    Returns True on success, False on fatal error (e.g. missing mesh files).
    """
    import multiprocessing as mp
    import trimesh
    from concurrent.futures import ProcessPoolExecutor
    from PIL import Image

    try:
        mesh_r, mesh_c, mesh_v, radius = load_layer_meshes(model, config.mesh_resolution)
    except FileNotFoundError as e:
        print(f'  [SKIP] {model}: {e}')
        return False

    cache_dir = os.path.join(
        cache_root,
        f'{model}_mesh{config.mesh_resolution}_render{config.resolution}',
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Resolve the frame range this call is responsible for.
    fs = 0 if frame_start is None else max(0, int(frame_start))
    fe = config.total_frames if frame_end is None else min(config.total_frames, int(frame_end))
    if fs >= fe:
        print(f'  [WARN] Empty frame range [{fs}, {fe}); nothing to do')
        todo_frames = []
    else:
        todo_frames = []
        for frame_idx in range(fs, fe):
            png_path = os.path.join(cache_dir, f'{frame_idx:04d}.png')
            if resume and os.path.exists(png_path):
                continue
            todo_frames.append(frame_idx)

    print(f'  Rendering {model} (parallel): {len(todo_frames)}/{fe - fs} frames '
          f'in range [{fs}, {fe}) '
          f'(R={len(mesh_r.faces)} C={len(mesh_c.faces)} V={len(mesh_v.faces)} faces, '
          f'workers={num_workers}, resume={resume}, assemble={assemble})')

    if todo_frames:
        # Pack mesh data as (verts, faces) numpy arrays for IPC-friendly pickling.
        mesh_data_tuple = (
            (np.asarray(mesh_r.vertices, dtype=np.float32),
             np.asarray(mesh_r.faces, dtype=np.int32)),
            (np.asarray(mesh_c.vertices, dtype=np.float32),
             np.asarray(mesh_c.faces, dtype=np.int32)),
            (np.asarray(mesh_v.vertices, dtype=np.float32),
             np.asarray(mesh_v.faces, dtype=np.int32)),
        )
        config_dict = asdict(config)

        # Use 'fork' context so workers inherit already-imported numpy/trimesh.
        # CRUCIAL: the main process MUST NOT have initialized CUDA yet when we
        # fork. trellis2 imports are lazy inside render_single_mesh, so as long
        # as we fork BEFORE the first render_single_mesh call, we are safe.
        ctx = mp.get_context('fork')

        t_start = time.time()
        completed = 0
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(mesh_data_tuple, config_dict),
        ) as pool:
            # pool.map preserves the input order, which gives us in-order slice
            # results without a manual reorder buffer. chunksize=1 maximizes
            # load balancing at a small scheduling cost.
            for result in pool.map(_worker_slice_frame, todo_frames, chunksize=1):
                frame_idx, r_data, c_data, v_data = result
                try:
                    sr = trimesh.Trimesh(vertices=r_data[0], faces=r_data[1], process=False)
                    sc = trimesh.Trimesh(vertices=c_data[0], faces=c_data[1], process=False)
                    sv = trimesh.Trimesh(vertices=v_data[0], faces=v_data[1], process=False)
                    _, yaw_rad = compute_phase_state(frame_idx, config)
                    img_r = render_single_mesh(sr, yaw_rad, config, radius)
                    img_c = render_single_mesh(sc, yaw_rad, config, radius)
                    img_v = render_single_mesh(sv, yaw_rad, config, radius)
                    frame = compose_frame(img_r, img_c, img_v, config)
                except Exception:
                    traceback.print_exc()
                    h = config.resolution + config.label_band_px
                    w = config.resolution * 3
                    frame = np.full((h, w, 3), 255, dtype=np.uint8)

                # Atomic-ish PNG write: write to a .tmp sibling first, then rename.
                # Pass format='PNG' explicitly so PIL does not try to infer
                # from the .tmp extension.
                png_path = os.path.join(cache_dir, f'{frame_idx:04d}.png')
                tmp_path = png_path + '.tmp'
                Image.fromarray(frame).save(tmp_path, format='PNG')
                os.replace(tmp_path, png_path)

                completed += 1
                if completed % 30 == 0 or completed == len(todo_frames):
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(f'    {completed}/{len(todo_frames)} '
                          f'({elapsed:.1f}s, {rate:.2f} fps)')

        elapsed = time.time() - t_start
        print(f'  Slice+render completed: {completed} frames in {elapsed:.1f}s '
              f'(avg {elapsed / max(completed, 1):.2f}s/frame)')
    else:
        if fe - fs == config.total_frames:
            print(f'  [RESUME] all {config.total_frames} frames already cached')
        else:
            print(f'  [RESUME] all frames in [{fs}, {fe}) already cached')

    # Final MP4 assembly from PNG cache. Skip when the caller is a partial
    # multi-GPU renderer; in that case the orchestrator assembles the MP4.
    if assemble:
        t_mp4 = time.time()
        _combine_pngs_to_mp4(cache_dir, out_path, config)
        print(f'  MP4 assembled in {time.time() - t_mp4:.1f}s')
        print(f'  [OK] wrote {out_path}')
    else:
        print(f'  [OK] partial render of [{fs}, {fe}) done; MP4 assembly skipped')
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                   help='Model names to render. Default: all 7 baseline models.')
    p.add_argument('--resolution', type=int, default=768,
                   help='Per-layer render resolution. Default: 768.')
    p.add_argument('--fps', type=int, default=24, help='Video framerate. Default: 24.')
    p.add_argument('--duration', type=float, default=15.0,
                   help='Video duration in seconds. Default: 15.0.')
    p.add_argument('--mesh-resolution', type=int, default=512,
                   help='Which EXP6 @res directory to read meshes from. Default: 512.')
    p.add_argument('--pitch', type=float, default=25.0, help='Camera pitch in degrees.')
    p.add_argument('--fov', type=float, default=40.0, help='Camera FOV in degrees.')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory for MP4 files.')
    p.add_argument('--overwrite', action='store_true', help='Re-render even if MP4 exists.')
    p.add_argument('--serial', action='store_true',
                   help='Use legacy serial renderer (no multiprocessing, no PNG cache, no resume). '
                        'Default is parallel + PNG cache + resume.')
    p.add_argument('--workers', type=int, default=16,
                   help='Number of CPU worker processes for parallel slicing. Default: 16.')
    p.add_argument('--no-resume', action='store_true',
                   help='Disable resume (regenerate every frame even if its PNG is cached).')
    p.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT,
                   help='Root directory for per-frame PNG caches.')
    p.add_argument('--frame-start', type=int, default=None,
                   help='Inclusive start frame index for this call (multi-GPU partitioning). '
                        'Default: 0 (process from the beginning).')
    p.add_argument('--frame-end', type=int, default=None,
                   help='Exclusive end frame index for this call (multi-GPU partitioning). '
                        'Default: total_frames (process through the end).')
    p.add_argument('--no-assemble', action='store_true',
                   help='Skip MP4 assembly; render the assigned frame range and exit. '
                        'Used by the multi-GPU driver; the orchestrator assembles the MP4 '
                        'after all partitions finish.')
    return p.parse_args()


def main():
    args = parse_args()
    config = AnimConfig(
        fps=args.fps,
        duration=args.duration,
        resolution=args.resolution,
        mesh_resolution=args.mesh_resolution,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
    )
    print(f'Models: {args.models}')
    print(f'Config: {config}')
    print(f'Total frames: {config.total_frames}')
    print(f'Phase boundaries (frame idx): {config.phase_boundaries}')
    print(f'Output dir: {args.out_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    use_parallel = not args.serial
    resume = not args.no_resume
    print(f'Renderer: {"parallel" if use_parallel else "serial"}'
          + (f' (workers={args.workers}, resume={resume}, cache={args.cache_root})'
             if use_parallel else ''))

    # Partial runs bypass the "MP4 already exists" early exit — the
    # orchestrator only needs the PNG cache populated, and the assembly
    # step is a cheap follow-up.
    partial_mode = (args.frame_start is not None) or (args.frame_end is not None) or args.no_assemble
    assemble = not args.no_assemble

    succeeded = 0
    for model in args.models:
        out_path = os.path.join(args.out_dir, f'{model}_comparison.mp4')
        if (not partial_mode) and os.path.exists(out_path) and not args.overwrite:
            print(f'  [EXISTS] {out_path} (use --overwrite to regenerate)')
            succeeded += 1
            continue
        try:
            if use_parallel:
                ok = render_comparison_video_parallel(
                    model, out_path, config,
                    num_workers=args.workers,
                    resume=resume,
                    cache_root=args.cache_root,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    assemble=assemble,
                )
            else:
                ok = render_comparison_video(model, out_path, config)
            if ok:
                succeeded += 1
        except Exception:
            traceback.print_exc()
            print(f'  [FAIL] {model}')
    print(f'\nDone: {succeeded}/{len(args.models)} videos rendered successfully')


def _self_test_phase_state():
    """Smoke test: verify key frames of compute_phase_state."""
    cfg = AnimConfig()
    assert cfg.total_frames == 360, f'expected 360 frames, got {cfg.total_frames}'
    assert cfg.phase_boundaries == [72, 144, 288, 360], cfg.phase_boundaries

    # Frame 0: phase 1 start, yaw 0, slice at outer.
    origin, yaw = compute_phase_state(0, cfg)
    assert np.allclose(origin, SLICE_OUTER), f'frame 0 origin: {origin}'
    assert abs(yaw) < 1e-9, f'frame 0 yaw: {yaw}'

    # Frame 72: phase 2 start (still outer, easing gives 0).
    origin, yaw = compute_phase_state(72, cfg)
    assert np.allclose(origin, SLICE_OUTER), f'frame 72 origin: {origin}'

    # Frame 107: phase 2 midpoint -> easing 0.5 -> origin halfway.
    origin, yaw = compute_phase_state(107, cfg)
    midpoint = 0.5 * SLICE_OUTER + 0.5 * SLICE_CENTER
    assert np.allclose(origin, midpoint, atol=0.05), f'frame 107 origin: {origin}, expected ~{midpoint}'

    # Frame 144: phase 3 start, slice at center.
    origin, yaw = compute_phase_state(144, cfg)
    assert np.allclose(origin, SLICE_CENTER), f'frame 144 origin: {origin}'

    # Frame 287: last frame of phase 3, still at center.
    origin, yaw = compute_phase_state(287, cfg)
    assert np.allclose(origin, SLICE_CENTER), f'frame 287 origin: {origin}'

    # Frame 288: phase 4 start, still at center (easing 0).
    origin, yaw = compute_phase_state(288, cfg)
    assert np.allclose(origin, SLICE_CENTER), f'frame 288 origin: {origin}'

    # Frame 359: last frame, back at outer corner.
    origin, yaw = compute_phase_state(359, cfg)
    assert np.allclose(origin, SLICE_OUTER, atol=1e-6), f'frame 359 origin: {origin}'
    assert abs(math.degrees(yaw) - 600.0) < 1e-6, f'frame 359 yaw deg: {math.degrees(yaw)}'

    # Easing sanity.
    assert ease_in_out_quad(0.0) == 0.0
    assert ease_in_out_quad(1.0) == 1.0
    assert abs(ease_in_out_quad(0.5) - 0.5) < 1e-9

    # load_layer_meshes end-to-end on icosphere.
    mesh_r, mesh_c, mesh_v, radius = load_layer_meshes('icosphere', 512)
    for name, m in [('R', mesh_r), ('C', mesh_c), ('V', mesh_v)]:
        bounds = m.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        extent = (bounds[1] - bounds[0]).max()
        assert np.all(np.abs(center) < 1e-3), f'{name} center {center}'
        assert abs(extent - 1.0) < 1e-3, f'{name} extent {extent}'
        assert len(m.vertices) > 0 and len(m.faces) > 0, f'{name} empty mesh'
    assert radius > 0

    # slice_mesh_for_frame at frame 0 (outer corner) and frame 216 (center).
    sliced_full, empty_full = slice_mesh_for_frame(mesh_c, SLICE_OUTER)
    assert not empty_full, 'frame 0 slice unexpectedly empty'
    # Outer corner slice keeps (almost) the entire mesh.
    assert len(sliced_full.faces) >= int(0.95 * len(mesh_c.faces)), (
        f'frame 0 sliced faces {len(sliced_full.faces)} vs full {len(mesh_c.faces)}')

    sliced_center, empty_center = slice_mesh_for_frame(mesh_c, SLICE_CENTER)
    assert not empty_center, 'center slice unexpectedly empty'
    # Center slice keeps noticeably fewer faces than the full mesh.
    assert len(sliced_center.faces) < len(mesh_c.faces), (
        f'center slice faces {len(sliced_center.faces)} vs full {len(mesh_c.faces)}')

    print('compute_phase_state self-test PASSED')
    print(f'load_layer_meshes: R={len(mesh_r.faces)} C={len(mesh_c.faces)} V={len(mesh_v.faces)} faces')
    print(f'slice test: full={len(sliced_full.faces)} center={len(sliced_center.faces)}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        _self_test_phase_state()
    else:
        main()
