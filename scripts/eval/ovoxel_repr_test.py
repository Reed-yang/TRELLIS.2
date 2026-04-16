"""
O-Voxel Representation Fidelity Test on Sketchfab-Hard Samples.

Layer A: Mesh -> O-Voxel -> Mesh (pure discretization loss)
Layer B: Mesh -> O-Voxel -> SC-VAE Encode -> Decode -> Mesh (+ compression loss)

Each (model, resolution) job runs in a subprocess with its own GPU to avoid
CUDA context corruption from O-Voxel internal GPU operations.

Usage:
    python scripts/eval/ovoxel_repr_test.py --layer a
    python scripts/eval/ovoxel_repr_test.py --layer b
    python scripts/eval/ovoxel_repr_test.py --layer all
    python scripts/eval/ovoxel_repr_test.py --layer a --num-gpus 8
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import csv
import argparse
import subprocess
import torch
import numpy as np
import trimesh
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOLUTIONS = [512, 1024, 1536, 2048]
NUM_SAMPLE_POINTS = 100_000
F_SCORE_THRESHOLDS = [0.005, 0.01, 0.05]
PREVIEW_VIEWS = 8
PREVIEW_RESOLUTION = 512
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

OUTPUT_ROOT = "experiments/ovoxel_repr_test"

# Sample registry
SAMPLES = {
    "helmet": {
        "source": "datasets/sketchfab_hard/early_medieval_nasal_helmet.glb",
        "description": "Chainmail ring topology, 362K verts, 324K faces",
    },
    "bugatti": {
        "source": "datasets/sketchfab_hard/bugatti-eb110-super-sport-1992-by-alexka.zip",
        "description": "Extreme aspect ratio (9:1), thin shell, 148K verts, 169K faces",
    },
    "spacesuit": {
        "source": "datasets/sketchfab_hard/franz-viehbocks-sokol-space-suit.zip",
        "description": "Fabric wrinkles, single mesh, 109K verts, 200K faces",
    },
}


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def _extract_mesh_from_zip(zip_path, model_id):
    """Extract mesh file from (possibly nested) zip/7z archive."""
    import zipfile
    import tempfile

    extract_dir = os.path.join(tempfile.gettempdir(), f"sketchfab_{model_id}")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    mesh_extensions = ('.obj', '.glb', '.gltf', '.ply', '.stl')
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(mesh_extensions):
                return os.path.join(root, f)

    # Nested archives
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if f.endswith('.7z'):
                inner_dir = os.path.join(extract_dir, "inner_extract")
                os.makedirs(inner_dir, exist_ok=True)
                subprocess.run(['7z', 'x', '-y', f'-o{inner_dir}', fpath],
                               capture_output=True)
                for r2, d2, f2s in os.walk(inner_dir):
                    for f2 in f2s:
                        if f2.lower().endswith(mesh_extensions):
                            return os.path.join(r2, f2)
            elif f.endswith('.zip'):
                inner_dir = os.path.join(extract_dir, "inner_extract")
                os.makedirs(inner_dir, exist_ok=True)
                import zipfile as zf_mod
                with zf_mod.ZipFile(fpath, 'r') as zf2:
                    zf2.extractall(inner_dir)
                for r2, d2, f2s in os.walk(inner_dir):
                    for f2 in f2s:
                        if f2.lower().endswith(mesh_extensions):
                            return os.path.join(r2, f2)

    raise FileNotFoundError(f"No mesh file found in {zip_path}")


def load_sample_mesh(model_id):
    """Load and normalize a sample mesh to [-0.5, 0.5]."""
    info = SAMPLES[model_id]
    source = info["source"]

    if source.endswith('.glb') or source.endswith('.obj'):
        mesh_path = source
    elif source.endswith('.zip'):
        mesh_path = _extract_mesh_from_zip(source, model_id)
    else:
        raise ValueError(f"Unknown source format: {source}")

    loaded = trimesh.load(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No Trimesh geometries in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    vertices = mesh.vertices.astype(np.float64)
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    mesh.vertices = (vertices - center) * scale
    return mesh


def ensure_gt_meshes():
    """Ensure all GT meshes are preprocessed and saved."""
    data_dir = os.path.join(OUTPUT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    for model_id in SAMPLES:
        out_path = os.path.join(data_dir, f"{model_id}.obj")
        if not os.path.exists(out_path):
            print(f"Preprocessing {model_id}...")
            mesh = load_sample_mesh(model_id)
            mesh.export(out_path)
            print(f"  Saved: {out_path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
        else:
            print(f"  Cached: {out_path}")


# ---------------------------------------------------------------------------
# Single-job worker (runs in subprocess with isolated CUDA context)
# ---------------------------------------------------------------------------

def _run_single_job(model_id, resolution, layer, gpu_id):
    """
    Run a single (model, resolution, layer) job in a subprocess.
    Returns a JSON string with results.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, __file__,
        "--worker",
        "--model-id", model_id,
        "--resolution", str(resolution),
        "--layer", layer,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=600, cwd=os.path.join(os.path.dirname(__file__), '..', '..'),
        )
        # Parse JSON result from stdout (last line)
        for line in reversed(result.stdout.strip().split('\n')):
            if line.startswith('{"'):
                return json.loads(line)
        # If no JSON found, report error
        return {
            "model_id": model_id,
            "resolution": resolution,
            "error": f"No result JSON. stderr: {result.stderr[-500:] if result.stderr else 'none'}",
        }
    except subprocess.TimeoutExpired:
        return {
            "model_id": model_id,
            "resolution": resolution,
            "error": "Timeout (600s)",
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "resolution": resolution,
            "error": str(e),
        }


def _worker_main(model_id, resolution, layer):
    """
    Worker entry point. Runs in isolated subprocess.
    Prints JSON result to stdout as the last line.
    """
    import warnings
    warnings.filterwarnings("ignore")

    gt_path = os.path.join(OUTPUT_ROOT, "data", f"{model_id}.obj")
    gt_mesh = trimesh.load(gt_path, process=False)

    out_dir = os.path.join(OUTPUT_ROOT, f"layer_{layer}", f"{model_id}_{resolution}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        if layer == "a":
            recon_mesh, meta = _worker_ovoxel_roundtrip(gt_mesh, resolution)
        else:
            recon_mesh, meta = _worker_scvae_roundtrip(gt_mesh, resolution)

        recon_mesh.export(os.path.join(out_dir, "recon.ply"))

        # Compute metrics on GPU (fresh CUDA context in this subprocess)
        metrics = _worker_compute_metrics(gt_mesh, recon_mesh)

        result = {
            "model_id": model_id,
            "resolution": resolution,
            **meta,
            **metrics,
            "error": "",
        }
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        result = {
            "model_id": model_id,
            "resolution": resolution,
            "error": str(e),
        }

    print(json.dumps(result), flush=True)


def _worker_ovoxel_roundtrip(gt_mesh, resolution):
    """O-Voxel roundtrip in worker subprocess."""
    import o_voxel

    vertices = torch.from_numpy(gt_mesh.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh.faces.copy()).long()

    t0 = time.time()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    encode_time = time.time() - t0

    # flexible_dual_grid_to_mesh requires CUDA tensors
    # (uses _C.hashmap_insert_3d_idx_as_val_cuda internally)
    t1 = time.time()
    out_verts, out_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
        voxel_indices.cuda(), dual_vertices.cuda(), intersected.cuda(),
        split_weight=None, grid_size=resolution, aabb=AABB,
    )
    decode_time = time.time() - t1

    # Post-processing: clean mesh using cumesh (same pipeline as to_glb)
    import cumesh
    t2 = time.time()
    cm = cumesh.CuMesh()
    cm.init(out_verts, out_faces)
    cm.fill_holes(max_hole_perimeter=3e-2)
    cm.remove_duplicate_faces()
    cm.repair_non_manifold_edges()
    cm.remove_small_connected_components(1e-5)
    cm.fill_holes(max_hole_perimeter=3e-2)
    cm.unify_face_orientations()
    clean_verts, clean_faces = cm.read()
    postproc_time = time.time() - t2

    recon_mesh = trimesh.Trimesh(
        vertices=clean_verts.detach().cpu().numpy(),
        faces=clean_faces.detach().cpu().numpy(),
        process=False,
    )

    meta = {
        "n_voxels": int(voxel_indices.shape[0]),
        "encode_time": round(encode_time, 2),
        "decode_time": round(decode_time, 2),
        "postproc_time": round(postproc_time, 2),
        "raw_verts": int(out_verts.shape[0]),
        "raw_faces": int(out_faces.shape[0]),
        "out_verts": int(clean_verts.shape[0]),
        "out_faces": int(clean_faces.shape[0]),
    }
    return recon_mesh, meta


def _worker_scvae_roundtrip(gt_mesh, resolution):
    """SC-VAE roundtrip in worker subprocess."""
    import o_voxel
    from trellis2.modules.sparse import SparseTensor
    from scripts.eval.eval_metrics import trellis_mesh_to_trimesh
    from scripts.eval.gap_measurement import load_vae_models

    vertices = torch.from_numpy(gt_mesh.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh.faces.copy()).long()

    t0 = time.time()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    ovoxel_time = time.time() - t0
    n_voxels = int(voxel_indices.shape[0])

    dv_local = dual_vertices * resolution - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)
    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices],
        dim=-1,
    )
    vertices_st = SparseTensor(feats=dv_local, coords=coords_with_batch)
    intersected_st = vertices_st.replace(intersected.float())

    encoder, decoder = load_vae_models()
    decoder.set_resolution(resolution)
    t1 = time.time()
    with torch.no_grad():
        z = encoder(vertices_st.cuda(), intersected_st.cuda())
        latent_tokens = z.feats.shape[0]
        latent_channels = z.feats.shape[1]
        recon_trellis = decoder(z)
    scvae_time = time.time() - t1

    if isinstance(recon_trellis, list):
        recon_trellis = recon_trellis[0]

    recon_mesh = trellis_mesh_to_trimesh(recon_trellis)

    meta = {
        "n_voxels": n_voxels,
        "ovoxel_time": round(ovoxel_time, 2),
        "scvae_time": round(scvae_time, 2),
        "latent_tokens": int(latent_tokens),
        "latent_channels": int(latent_channels),
        "out_verts": int(len(recon_mesh.vertices)),
        "out_faces": int(len(recon_mesh.faces)),
    }
    return recon_mesh, meta


def _compute_topo_metrics(mesh):
    """Compute topology-related metrics for a trimesh mesh.

    Uses numpy vectorized ops to handle multi-million-face meshes efficiently.
    """
    # Connected components via face adjacency graph (scipy sparse)
    try:
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        adj = mesh.face_adjacency
        n_faces = len(mesh.faces)
        if len(adj) > 0:
            data = np.ones(len(adj) * 2, dtype=np.int8)
            row = np.concatenate([adj[:, 0], adj[:, 1]])
            col = np.concatenate([adj[:, 1], adj[:, 0]])
            graph = csr_matrix((data, (row, col)), shape=(n_faces, n_faces))
            n_components = connected_components(graph, directed=False)[0]
        else:
            n_components = n_faces
    except Exception:
        n_components = -1

    # Boundary edges using numpy: edges with unique_count == 1
    edges_sorted = np.sort(mesh.edges, axis=1)
    # Encode edge pairs as single int64 for fast counting
    edge_keys = edges_sorted[:, 0].astype(np.int64) * (edges_sorted[:, 1].max() + 1) + edges_sorted[:, 1]
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    n_boundary_edges = int((counts == 1).sum())

    return {
        "euler_number": int(mesh.euler_number),
        "n_components": n_components,
        "is_watertight": bool(mesh.is_watertight),
        "n_boundary_edges": n_boundary_edges,
        "surface_area": float(mesh.area),
    }


def _worker_compute_metrics(gt_mesh, recon_mesh):
    """Compute geometric + topology metrics in worker subprocess."""
    from scripts.eval.eval_metrics import (
        sample_points_and_normals, chamfer_distance,
        f_score_multi, normal_consistency,
    )

    gt_pts, gt_nrm = sample_points_and_normals(gt_mesh, NUM_SAMPLE_POINTS)
    recon_pts, recon_nrm = sample_points_and_normals(recon_mesh, NUM_SAMPLE_POINTS)

    gt_pts, gt_nrm = gt_pts.cuda(), gt_nrm.cuda()
    recon_pts, recon_nrm = recon_pts.cuda(), recon_nrm.cuda()

    cd = chamfer_distance(recon_pts, gt_pts)
    nc = normal_consistency(recon_pts, recon_nrm, gt_pts, gt_nrm)
    fscores = f_score_multi(recon_pts, gt_pts, F_SCORE_THRESHOLDS)

    # Topology metrics
    gt_topo = _compute_topo_metrics(gt_mesh)
    recon_topo = _compute_topo_metrics(recon_mesh)

    return {
        "cd": cd,
        "nc": nc,
        **{f"fscore_{t}": v for t, v in fscores.items()},
        # GT topology
        "gt_euler": gt_topo["euler_number"],
        "gt_components": gt_topo["n_components"],
        "gt_watertight": gt_topo["is_watertight"],
        "gt_boundary_edges": gt_topo["n_boundary_edges"],
        "gt_area": gt_topo["surface_area"],
        # Recon topology
        "recon_euler": recon_topo["euler_number"],
        "recon_components": recon_topo["n_components"],
        "recon_watertight": recon_topo["is_watertight"],
        "recon_boundary_edges": recon_topo["n_boundary_edges"],
        "recon_area": recon_topo["surface_area"],
        # Deltas
        "delta_euler": recon_topo["euler_number"] - gt_topo["euler_number"],
        "delta_components": recon_topo["n_components"] - gt_topo["n_components"],
        "area_ratio": recon_topo["surface_area"] / gt_topo["surface_area"] if gt_topo["surface_area"] > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Parallel dispatcher
# ---------------------------------------------------------------------------

def run_parallel(layer, models, resolutions, num_gpus):
    """Dispatch all jobs in parallel across GPUs using subprocess.Popen."""
    jobs = []
    for model_id in models:
        for res in resolutions:
            jobs.append((model_id, res))

    print(f"\nDispatching {len(jobs)} jobs across {num_gpus} GPUs for layer {layer}...")

    # Launch all jobs in parallel batches of num_gpus
    results = []
    cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    for batch_start in range(0, len(jobs), num_gpus):
        batch = jobs[batch_start:batch_start + num_gpus]
        procs = []
        for i, (model_id, res) in enumerate(batch):
            gpu_id = i % num_gpus
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker",
                "--model-id", model_id,
                "--resolution", str(res),
                "--layer", layer,
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, cwd=cwd,
            )
            procs.append((model_id, res, proc))

        # Wait for all in batch
        for model_id, res, proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                results.append({"model_id": model_id, "resolution": res,
                                "error": "Timeout (600s)"})
                print(f"  [{model_id}@{res}] TIMEOUT")
                continue

            # Parse JSON from stdout
            result = None
            for line in reversed(stdout.strip().split('\n')):
                if line.startswith('{"'):
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        pass
                    break

            if result is None:
                err_msg = stderr[-300:].strip() if stderr else "no output"
                result = {"model_id": model_id, "resolution": res,
                          "error": f"No JSON result. {err_msg}"}

            if result.get("error"):
                print(f"  [{model_id}@{res}] FAILED: {result['error'][:80]}")
            else:
                cd = result.get('cd', '?')
                nc = result.get('nc', '?')
                nv = result.get('n_voxels', '?')
                cd_s = f"{cd:.6f}" if isinstance(cd, float) else cd
                nc_s = f"{nc:.4f}" if isinstance(nc, float) else nc
                nv_s = f"{nv:,}" if isinstance(nv, int) else nv
                print(f"  [{model_id}@{res}] OK — Voxels: {nv_s}, CD: {cd_s}, NC: {nc_s}")
            results.append(result)

    # Sort by model then resolution
    results.sort(key=lambda r: (r.get('model_id', ''), int(r.get('resolution', 0))))

    csv_path = os.path.join(OUTPUT_ROOT, f"results_{layer}.csv")
    _write_results_csv(csv_path, results)
    return results


# ---------------------------------------------------------------------------
# Preview rendering (runs in main process after all jobs complete)
# ---------------------------------------------------------------------------

MAX_RENDER_FACES = 500_000  # decimate meshes larger than this for rendering

def render_normal_maps_fixed_views(tm_mesh, nviews=PREVIEW_VIEWS, resolution=PREVIEW_RESOLUTION):
    """Render normal maps from fixed viewpoints using NVDiffRast."""
    from trellis2.representations import Mesh as TrellisMesh
    from trellis2.utils.render_utils import render_snapshot

    # Decimate very large meshes to avoid NVDiffRast OOM
    if len(tm_mesh.faces) > MAX_RENDER_FACES:
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(tm_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm_mesh.faces)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(MAX_RENDER_FACES)
        tm_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=False,
        )

    trellis_mesh = TrellisMesh(
        vertices=torch.from_numpy(tm_mesh.vertices.copy()).float().cuda(),
        faces=torch.from_numpy(tm_mesh.faces.copy()).int().cuda(),
    )

    result = render_snapshot(
        trellis_mesh,
        resolution=resolution,
        nviews=nviews,
        r=2, fov=40,
        offset=(0, 15 / 180 * np.pi),
        return_types=["normal"],
    )

    images = []
    for nmap in result["normal"]:
        images.append(Image.fromarray(nmap))
    return images


def render_side_by_side(gt_images, recon_images, out_dir, model_id, resolution):
    """Save side-by-side GT|Recon images for each view."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (gt_img, recon_img) in enumerate(zip(gt_images, recon_images)):
        w, h = gt_img.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(gt_img.convert("RGB"), (0, 0))
        combined.paste(recon_img.convert("RGB"), (w, 0))
        path = os.path.join(out_dir, f"{model_id}_{resolution}_view{i}.png")
        combined.save(path)


def render_model_grid(gt_images, recon_images_dict, out_path, resolutions):
    """Create a grid image: rows = [GT, res1, res2, ...], columns = viewpoints."""
    nviews = len(gt_images)
    cell_w, cell_h = gt_images[0].size

    valid_res = [r for r in resolutions if r in recon_images_dict]
    n_rows = 1 + len(valid_res)

    label_w = 100
    grid_w = label_w + cell_w * nviews
    grid_h = cell_h * n_rows

    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text((5, cell_h // 2 - 10), "GT", fill="white", font=font)
    for j, img in enumerate(gt_images):
        grid.paste(img.convert("RGB"), (label_w + j * cell_w, 0))

    for i, res in enumerate(valid_res):
        y = (i + 1) * cell_h
        draw.text((5, y + cell_h // 2 - 10), str(res), fill="white", font=font)
        for j, img in enumerate(recon_images_dict[res]):
            grid.paste(img.convert("RGB"), (label_w + j * cell_w, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"  Grid saved: {out_path}")


def render_previews(models, layer_name, resolutions):
    """Render preview images for all models in a given layer."""
    print(f"\nRendering previews for layer_{layer_name}...")

    for model_id in models:
        print(f"  {model_id}...")
        gt_path = os.path.join(OUTPUT_ROOT, "data", f"{model_id}.obj")
        gt_mesh = trimesh.load(gt_path, process=False)
        gt_images = render_normal_maps_fixed_views(gt_mesh)

        recon_images_dict = {}
        for res in resolutions:
            recon_path = os.path.join(OUTPUT_ROOT, f"layer_{layer_name}", f"{model_id}_{res}", "recon.ply")
            if not os.path.exists(recon_path):
                continue
            recon_mesh = trimesh.load(recon_path, process=False)
            try:
                recon_images = render_normal_maps_fixed_views(recon_mesh)
            except Exception as e:
                print(f"    Render failed for {model_id}@{res}: {e}")
                continue

            preview_dir = os.path.join(OUTPUT_ROOT, "previews", f"layer_{layer_name}")
            render_side_by_side(gt_images, recon_images, preview_dir, model_id, res)
            recon_images_dict[res] = recon_images

        if recon_images_dict:
            grid_path = os.path.join(OUTPUT_ROOT, "previews", f"layer_{layer_name}", f"{model_id}_grid.png")
            render_model_grid(gt_images, recon_images_dict, grid_path, resolutions)


# ---------------------------------------------------------------------------
# CSV + Summary utilities
# ---------------------------------------------------------------------------

def _write_results_csv(csv_path, results):
    """Write results list to CSV."""
    if not results:
        return
    all_fields = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                all_fields.append(k)
                seen.add(k)
    for r in results:
        for f in all_fields:
            r.setdefault(f, "")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")


def print_summary():
    """Print formatted summary tables from saved CSVs."""
    for layer, csv_name in [("Layer A (O-Voxel)", "results_a.csv"),
                            ("Layer B (SC-VAE)", "results_b.csv")]:
        csv_path = os.path.join(OUTPUT_ROOT, csv_name)
        if not os.path.exists(csv_path):
            continue

        rows = []
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        # Geometric metrics
        print(f"\n--- {layer}: Geometric ---")
        print(f"{'Model':<12} {'Res':>6} {'Voxels':>12} {'CD':>12} {'NC':>8} "
              f"{'F@0.005':>8} {'F@0.01':>8} {'F@0.05':>8} {'Error'}")
        print("-" * 95)
        for row in rows:
            if row.get("error"):
                print(f"{row['model_id']:<12} {row['resolution']:>6} "
                      f"{'':>12} {'':>12} {'':>8} {'':>8} {'':>8} {row['error'][:40]}")
            else:
                print(f"{row['model_id']:<12} {row['resolution']:>6} "
                      f"{int(float(row.get('n_voxels', 0))):>12,} "
                      f"{float(row.get('cd', 0)):>12.6f} "
                      f"{float(row.get('nc', 0)):>8.4f} "
                      f"{float(row.get('fscore_0.005', 0)):>8.4f} "
                      f"{float(row.get('fscore_0.01', 0)):>8.4f} "
                      f"{float(row.get('fscore_0.05', 0)):>8.4f} ")

        # Topology metrics (if present)
        if rows and 'recon_components' in rows[0]:
            print(f"\n--- {layer}: Topology ---")
            print(f"{'Model':<12} {'Res':>6} {'GT Comp':>8} {'Rec Comp':>9} "
                  f"{'GT Euler':>9} {'Rec Euler':>10} {'GT BndE':>8} {'Rec BndE':>9} "
                  f"{'Area Ratio':>11}")
            print("-" * 100)
            for row in rows:
                if row.get("error"):
                    continue
                print(f"{row['model_id']:<12} {row['resolution']:>6} "
                      f"{row.get('gt_components', ''):>8} "
                      f"{row.get('recon_components', ''):>9} "
                      f"{row.get('gt_euler', ''):>9} "
                      f"{row.get('recon_euler', ''):>10} "
                      f"{row.get('gt_boundary_edges', ''):>8} "
                      f"{row.get('recon_boundary_edges', ''):>9} "
                      f"{float(row.get('area_ratio', 0)):>11.4f} ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="O-Voxel Representation Fidelity Test")
    parser.add_argument("--layer", choices=["a", "b", "all"], default="all",
                        help="Which layer to test: a=O-Voxel, b=SC-VAE, all=both")
    parser.add_argument("--models", nargs="+", default=list(SAMPLES.keys()),
                        help="Which models to test (default: all)")
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS,
                        help="Resolutions to test (default: 512 1024 1536 2048)")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs for parallel execution (default: 8)")
    parser.add_argument("--skip-previews", action="store_true",
                        help="Skip preview rendering")
    # Internal worker mode
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-id", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--resolution", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode: run single job and exit
    if args.worker:
        _worker_main(args.model_id, args.resolution, args.layer)
        return

    # Dispatcher mode
    models = [m for m in args.models if m in SAMPLES]
    resolutions = args.resolutions

    # Preprocess
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    ensure_gt_meshes()

    # Layer A
    if args.layer in ("a", "all"):
        print("\n" + "=" * 60)
        print("LAYER A: O-Voxel Roundtrip")
        print("=" * 60)
        run_parallel("a", models, resolutions, args.num_gpus)
        if not args.skip_previews:
            render_previews(models, "a", resolutions)

    # Layer B
    if args.layer in ("b", "all"):
        print("\n" + "=" * 60)
        print("LAYER B: SC-VAE Roundtrip")
        print("=" * 60)
        run_parallel("b", models, resolutions, args.num_gpus)
        if not args.skip_previews:
            render_previews(models, "b", resolutions)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print_summary()


if __name__ == "__main__":
    main()
