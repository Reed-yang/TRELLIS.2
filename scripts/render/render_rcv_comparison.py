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
import traceback
from dataclasses import dataclass

import numpy as np

# Project root on sys.path so trellis2 and scripts.* imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


DEFAULT_MODELS = [
    'bowl', 'icosphere', 'parallel_planes', 'nested_spheres',
    'bugatti', 'spacesuit', 'helmet',
]

MESH_ROOT = 'results/baseline_experiments/EXP6_corep_recon'
DEFAULT_OUT_DIR = 'results/baseline_experiments/videos'

LABELS = ['Layer R (O-Voxel)', 'Layer C (CoReP)', 'Layer V (O-Voxel + VAE)']

SLICE_NORMAL = np.array([-1.0, 1.0, 1.0]) / math.sqrt(3.0)
SLICE_OUTER = np.array([-0.9, 0.9, 0.9])
SLICE_CENTER = np.array([0.0, 0.0, 0.0])


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
