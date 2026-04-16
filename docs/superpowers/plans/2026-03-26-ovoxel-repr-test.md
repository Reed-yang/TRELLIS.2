# O-Voxel Representation Fidelity Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify O-Voxel representation fidelity and SC-VAE compression loss on 3 hard Sketchfab models across resolutions 512–2048, with multi-view preview rendering for visual comparison.

**Architecture:** Single script `scripts/ovoxel_repr_test.py` handles all layers (A: O-Voxel roundtrip, B: SC-VAE roundtrip). Reuses existing `scripts/eval_metrics.py` for geometric metrics and `trellis2/utils/render_utils.py` for normal map rendering. Preview grid assembly uses PIL. No new modules needed.

**Tech Stack:** `o_voxel`, `trimesh`, `torch`, `trellis2`, `PIL`, existing `scripts/eval_metrics.py`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/ovoxel_repr_test.py` (create) | Main script: preprocessing, Layer A, Layer B, metrics, preview rendering, grid assembly |
| `scripts/eval_metrics.py` (read only) | Reuse: `sample_points_and_normals`, `chamfer_distance`, `f_score_multi`, `normal_consistency` |
| `scripts/gap_measurement.py` (read only) | Reuse: `load_and_normalize_mesh`, `load_vae_models`, `vae_reconstruct` pattern |
| `trellis2/utils/render_utils.py` (read only) | Reuse: `render_snapshot` for normal map rendering |

Output directories (created by script):
```
experiments/ovoxel_repr_test/
├── data/                          # Normalized GT meshes (.obj)
├── layer_a/{model}_{res}/         # O-Voxel roundtrip results (recon.obj)
├── layer_b/{model}_{res}/         # SC-VAE roundtrip results (recon.obj)
├── previews/layer_a/              # Preview images
├── previews/layer_b/
└── results_a.csv, results_b.csv   # Metrics
```

---

## Task 1: Data Preprocessing + Script Skeleton

**Files:**
- Create: `scripts/ovoxel_repr_test.py`

- [ ] **Step 1: Create script with sample registry and preprocessing**

```python
"""
O-Voxel Representation Fidelity Test on Sketchfab-Hard Samples.

Layer A: Mesh -> O-Voxel -> Mesh (pure discretization loss)
Layer B: Mesh -> O-Voxel -> SC-VAE Encode -> Decode -> Mesh (+ compression loss)

Usage:
    python scripts/ovoxel_repr_test.py --layer a
    python scripts/ovoxel_repr_test.py --layer b
    python scripts/ovoxel_repr_test.py --layer all
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import csv
import argparse
import torch
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from scripts.eval_metrics import (
    sample_points_and_normals,
    chamfer_distance,
    f_score_multi,
    normal_consistency,
)

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

# Sample registry: id -> (source_path, loader_type)
# loader_type: "glb" for direct load, "obj" for OBJ file needing extraction
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
    import subprocess
    import tempfile

    extract_dir = os.path.join(tempfile.gettempdir(), f"sketchfab_{model_id}")
    os.makedirs(extract_dir, exist_ok=True)

    # First level: unzip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # Look for mesh files directly
    mesh_extensions = ('.obj', '.glb', '.gltf', '.ply', '.stl')
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(mesh_extensions):
                return os.path.join(root, f)

    # Look for nested archives (.7z, .zip) in source/ directory
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
            elif f.endswith('.zip') and 'source' not in root.split(os.sep)[-1:]:
                pass  # already extracted top level
            elif f.endswith('.zip'):
                inner_dir = os.path.join(extract_dir, "inner_extract")
                os.makedirs(inner_dir, exist_ok=True)
                with zipfile.ZipFile(fpath, 'r') as zf2:
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

    print(f"  Loading mesh from: {mesh_path}")

    # Load with trimesh, merge all geometries
    loaded = trimesh.load(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No Trimesh geometries in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    # Normalize to [-0.5, 0.5] by max extent (preserve aspect ratio)
    vertices = mesh.vertices.astype(np.float64)
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    mesh.vertices = (vertices - center) * scale

    return mesh


def preprocess_all():
    """Load, normalize, and save all GT meshes."""
    data_dir = os.path.join(OUTPUT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    gt_meshes = {}
    for model_id in SAMPLES:
        out_path = os.path.join(data_dir, f"{model_id}.obj")
        print(f"Preprocessing {model_id}...")
        mesh = load_sample_mesh(model_id)
        mesh.export(out_path)
        print(f"  Saved: {out_path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
        print(f"  Extents: {mesh.bounding_box.extents}")
        gt_meshes[model_id] = mesh

    return gt_meshes
```

- [ ] **Step 2: Run preprocessing to verify all 3 models load correctly**

Run: `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from scripts.ovoxel_repr_test import preprocess_all
gt = preprocess_all()
for k, v in gt.items():
    print(f'{k}: {len(v.vertices)} verts, {len(v.faces)} faces, extents={v.bounding_box.extents}')
"`

Expected: 3 models loaded and saved as OBJ, all within [-0.5, 0.5].

- [ ] **Step 3: Commit**

```bash
git add scripts/ovoxel_repr_test.py
git commit -m "feat: add ovoxel repr test script skeleton with data preprocessing"
```

---

## Task 2: Layer A — O-Voxel Roundtrip

**Files:**
- Modify: `scripts/ovoxel_repr_test.py`

- [ ] **Step 1: Add O-Voxel roundtrip and metrics computation**

Append to `scripts/ovoxel_repr_test.py`:

```python
# ---------------------------------------------------------------------------
# Layer A: O-Voxel roundtrip (mesh -> voxel -> mesh)
# ---------------------------------------------------------------------------

def ovoxel_roundtrip(gt_mesh, resolution):
    """
    Convert mesh to O-Voxel and back. Returns reconstructed trimesh.Trimesh.
    Raises on failure (e.g., IndexError at low resolution).
    """
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

    t1 = time.time()
    out_verts, out_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
        voxel_indices, dual_vertices, intersected,
        split_weight=None, grid_size=resolution, aabb=AABB,
    )
    decode_time = time.time() - t1

    recon_mesh = trimesh.Trimesh(
        vertices=out_verts.cpu().numpy(),
        faces=out_faces.cpu().numpy(),
        process=False,
    )

    meta = {
        "n_voxels": int(voxel_indices.shape[0]),
        "encode_time": encode_time,
        "decode_time": decode_time,
        "out_verts": int(out_verts.shape[0]),
        "out_faces": int(out_faces.shape[0]),
    }
    return recon_mesh, meta


def compute_metrics(gt_mesh, recon_mesh):
    """Compute geometric metrics between GT and reconstructed mesh."""
    gt_pts, gt_nrm = sample_points_and_normals(gt_mesh, NUM_SAMPLE_POINTS)
    recon_pts, recon_nrm = sample_points_and_normals(recon_mesh, NUM_SAMPLE_POINTS)

    gt_pts = gt_pts.cuda()
    gt_nrm = gt_nrm.cuda()
    recon_pts = recon_pts.cuda()
    recon_nrm = recon_nrm.cuda()

    cd = chamfer_distance(recon_pts, gt_pts)
    nc = normal_consistency(recon_pts, recon_nrm, gt_pts, gt_nrm)
    fscores = f_score_multi(recon_pts, gt_pts, F_SCORE_THRESHOLDS)

    return {
        "cd": cd,
        "nc": nc,
        **{f"fscore_{t}": v for t, v in fscores.items()},
    }


def run_layer_a(gt_meshes):
    """Run Layer A (O-Voxel roundtrip) for all models × resolutions."""
    results = []

    for model_id, gt_mesh in gt_meshes.items():
        for res in RESOLUTIONS:
            print(f"\n[Layer A] {model_id} @ {res}...")
            out_dir = os.path.join(OUTPUT_ROOT, "layer_a", f"{model_id}_{res}")
            os.makedirs(out_dir, exist_ok=True)

            try:
                recon_mesh, meta = ovoxel_roundtrip(gt_mesh, res)
                recon_mesh.export(os.path.join(out_dir, "recon.obj"))
                print(f"  Voxels: {meta['n_voxels']:,}, Encode: {meta['encode_time']:.1f}s, "
                      f"Decode: {meta['decode_time']:.1f}s")

                metrics = compute_metrics(gt_mesh, recon_mesh)
                print(f"  CD: {metrics['cd']:.6f}, NC: {metrics['nc']:.4f}, "
                      f"F@0.005: {metrics['fscore_0.005']:.4f}")

                row = {
                    "model_id": model_id,
                    "resolution": res,
                    **meta,
                    **metrics,
                    "error": "",
                }
            except Exception as e:
                print(f"  FAILED: {e}")
                row = {
                    "model_id": model_id,
                    "resolution": res,
                    "error": str(e),
                }

            results.append(row)

    # Write CSV
    csv_path = os.path.join(OUTPUT_ROOT, "results_a.csv")
    if results:
        fieldnames = list(results[0].keys())
        # Ensure all rows have all fields
        for r in results:
            for f in fieldnames:
                r.setdefault(f, "")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    return results
```

- [ ] **Step 2: Add argparse main block**

Append to `scripts/ovoxel_repr_test.py`:

```python
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
    parser.add_argument("--skip-previews", action="store_true",
                        help="Skip preview rendering")
    args = parser.parse_args()

    # Override globals based on args
    global RESOLUTIONS
    RESOLUTIONS = args.resolutions

    # Filter samples
    samples_to_run = {k: v for k, v in SAMPLES.items() if k in args.models}
    if not samples_to_run:
        print(f"No matching models. Available: {list(SAMPLES.keys())}")
        return

    # Preprocess
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    gt_meshes = {}
    for model_id in samples_to_run:
        data_path = os.path.join(OUTPUT_ROOT, "data", f"{model_id}.obj")
        if os.path.exists(data_path):
            print(f"Loading cached GT: {data_path}")
            gt_meshes[model_id] = trimesh.load(data_path, process=False)
        else:
            gt_meshes[model_id] = load_sample_mesh(model_id)
            os.makedirs(os.path.join(OUTPUT_ROOT, "data"), exist_ok=True)
            gt_meshes[model_id].export(data_path)

    # Layer A
    if args.layer in ("a", "all"):
        print("\n" + "=" * 60)
        print("LAYER A: O-Voxel Roundtrip")
        print("=" * 60)
        results_a = run_layer_a(gt_meshes)
        if not args.skip_previews:
            render_previews(gt_meshes, "layer_a")

    # Layer B
    if args.layer in ("b", "all"):
        print("\n" + "=" * 60)
        print("LAYER B: SC-VAE Roundtrip")
        print("=" * 60)
        results_b = run_layer_b(gt_meshes)
        if not args.skip_previews:
            render_previews(gt_meshes, "layer_b")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print_summary()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run Layer A on a single model to verify**

Run: `.venv/bin/python scripts/ovoxel_repr_test.py --layer a --models helmet --resolutions 512 1024 --skip-previews`

Expected: 2 rows in `experiments/ovoxel_repr_test/results_a.csv`, recon.obj files saved.

- [ ] **Step 4: Run Layer A on all models**

Run: `.venv/bin/python scripts/ovoxel_repr_test.py --layer a --skip-previews`

Expected: 12 rows (3 models × 4 resolutions), some may fail at certain resolutions.

- [ ] **Step 5: Commit**

```bash
git add scripts/ovoxel_repr_test.py
git commit -m "feat: add Layer A (O-Voxel roundtrip) with metrics computation"
```

---

## Task 3: Layer B — SC-VAE Roundtrip

**Files:**
- Modify: `scripts/ovoxel_repr_test.py`

- [ ] **Step 1: Add SC-VAE roundtrip function**

Add after `run_layer_a` function in `scripts/ovoxel_repr_test.py`:

```python
# ---------------------------------------------------------------------------
# Layer B: SC-VAE roundtrip (mesh -> O-Voxel -> encode -> decode -> mesh)
# ---------------------------------------------------------------------------

def scvae_roundtrip(gt_mesh, resolution, encoder, decoder):
    """
    Full SC-VAE roundtrip. Returns reconstructed trimesh and metadata.
    """
    import o_voxel
    from trellis2.modules.sparse import SparseTensor
    from scripts.eval_metrics import trellis_mesh_to_trimesh

    vertices = torch.from_numpy(gt_mesh.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh.faces.copy()).long()

    # Mesh -> O-Voxel
    t0 = time.time()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    ovoxel_time = time.time() - t0
    n_voxels = int(voxel_indices.shape[0])

    # Prepare encoder input
    dv_local = dual_vertices * resolution - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)
    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices],
        dim=-1,
    )
    vertices_st = SparseTensor(feats=dv_local, coords=coords_with_batch)
    intersected_st = vertices_st.replace(intersected.float())

    # Encode -> decode
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
        "ovoxel_time": ovoxel_time,
        "scvae_time": scvae_time,
        "latent_tokens": int(latent_tokens),
        "latent_channels": int(latent_channels),
        "out_verts": int(len(recon_mesh.vertices)),
        "out_faces": int(len(recon_mesh.faces)),
    }
    return recon_mesh, meta


def run_layer_b(gt_meshes):
    """Run Layer B (SC-VAE roundtrip) for all models × resolutions."""
    from scripts.gap_measurement import load_vae_models

    print("Loading SC-VAE encoder/decoder...")
    encoder, decoder = load_vae_models()
    print("  Loaded.")

    results = []
    for model_id, gt_mesh in gt_meshes.items():
        for res in RESOLUTIONS:
            print(f"\n[Layer B] {model_id} @ {res}...")
            out_dir = os.path.join(OUTPUT_ROOT, "layer_b", f"{model_id}_{res}")
            os.makedirs(out_dir, exist_ok=True)

            try:
                recon_mesh, meta = scvae_roundtrip(gt_mesh, res, encoder, decoder)
                recon_mesh.export(os.path.join(out_dir, "recon.obj"))
                print(f"  Voxels: {meta['n_voxels']:,}, Latent tokens: {meta['latent_tokens']:,}, "
                      f"SC-VAE time: {meta['scvae_time']:.1f}s")

                metrics = compute_metrics(gt_mesh, recon_mesh)
                print(f"  CD: {metrics['cd']:.6f}, NC: {metrics['nc']:.4f}, "
                      f"F@0.005: {metrics['fscore_0.005']:.4f}")

                row = {
                    "model_id": model_id,
                    "resolution": res,
                    **meta,
                    **metrics,
                    "error": "",
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  FAILED: {e}")
                row = {
                    "model_id": model_id,
                    "resolution": res,
                    "error": str(e),
                }

            results.append(row)
            torch.cuda.empty_cache()

    # Write CSV
    csv_path = os.path.join(OUTPUT_ROOT, "results_b.csv")
    if results:
        fieldnames = list(results[0].keys())
        for r in results:
            for f in fieldnames:
                r.setdefault(f, "")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    return results
```

- [ ] **Step 2: Run Layer B on a single model at 512 to verify**

Run: `.venv/bin/python scripts/ovoxel_repr_test.py --layer b --models spacesuit --resolutions 512 --skip-previews`

Expected: 1 row in `results_b.csv`. Space suit is smallest (109K verts) so safest test.

- [ ] **Step 3: Run Layer B on all models (start with lower resolutions)**

Run: `.venv/bin/python scripts/ovoxel_repr_test.py --layer b --skip-previews`

Expected: Results for all feasible (model, resolution) combinations. Some high-res may OOM — recorded as errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/ovoxel_repr_test.py
git commit -m "feat: add Layer B (SC-VAE roundtrip) to ovoxel repr test"
```

---

## Task 4: Preview Rendering

**Files:**
- Modify: `scripts/ovoxel_repr_test.py`

- [ ] **Step 1: Add normal map rendering function**

Add after `compute_metrics` in `scripts/ovoxel_repr_test.py`:

```python
# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------

def render_normal_maps_fixed_views(tm_mesh, nviews=PREVIEW_VIEWS, resolution=PREVIEW_RESOLUTION):
    """
    Render normal maps from fixed viewpoints using NVDiffRast.
    Returns list of PIL Images (RGBA normal maps).
    """
    from trellis2.representations import Mesh as TrellisMesh
    from trellis2.utils.render_utils import render_snapshot

    trellis_mesh = TrellisMesh(
        vertices=torch.from_numpy(tm_mesh.vertices.copy()).float().cuda(),
        faces=torch.from_numpy(tm_mesh.faces.copy()).int().cuda(),
    )

    # Fixed viewpoints: evenly spaced azimuth, slight elevation
    result = render_snapshot(
        trellis_mesh,
        resolution=resolution,
        nviews=nviews,
        r=2, fov=40,
        offset=(0, 15 / 180 * np.pi),  # 0 yaw offset, 15deg elevation
        return_types=["normal"],
    )

    images = []
    for nmap in result["normal"]:
        images.append(Image.fromarray(nmap))  # uint8 [H, W, 3] or [H, W, 4]
    return images


def render_side_by_side(gt_images, recon_images, out_dir, model_id, resolution):
    """Save side-by-side GT|Recon images for each view."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (gt_img, recon_img) in enumerate(zip(gt_images, recon_images)):
        # Ensure same size
        w, h = gt_img.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(gt_img.convert("RGB"), (0, 0))
        combined.paste(recon_img.convert("RGB"), (w, 0))
        path = os.path.join(out_dir, f"{model_id}_{resolution}_view{i}.png")
        combined.save(path)


def render_model_grid(gt_images_dict, recon_images_dict, out_path, model_id, resolutions):
    """
    Create a grid image: rows = [GT, res1, res2, ...], columns = viewpoints.
    gt_images_dict: {res: [PIL images]} (all same GT, just pick first)
    recon_images_dict: {res: [PIL images]}
    """
    # Use first available GT images
    gt_imgs = None
    for res in resolutions:
        if res in gt_images_dict:
            gt_imgs = gt_images_dict[res]
            break
    if gt_imgs is None:
        print(f"  No GT images available for {model_id}, skipping grid")
        return

    nviews = len(gt_imgs)
    cell_w, cell_h = gt_imgs[0].size

    # Rows: GT + one per resolution
    valid_res = [r for r in resolutions if r in recon_images_dict]
    n_rows = 1 + len(valid_res)

    # Add row labels
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

    # GT row
    draw.text((5, cell_h // 2 - 10), "GT", fill="white", font=font)
    for j, img in enumerate(gt_imgs):
        grid.paste(img.convert("RGB"), (label_w + j * cell_w, 0))

    # Recon rows
    for i, res in enumerate(valid_res):
        y = (i + 1) * cell_h
        draw.text((5, y + cell_h // 2 - 10), str(res), fill="white", font=font)
        for j, img in enumerate(recon_images_dict[res]):
            grid.paste(img.convert("RGB"), (label_w + j * cell_w, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"  Grid saved: {out_path}")


def render_previews(gt_meshes, layer_name):
    """Render preview images for all models in a given layer."""
    print(f"\nRendering previews for {layer_name}...")

    for model_id, gt_mesh in gt_meshes.items():
        print(f"  {model_id}...")
        gt_images = render_normal_maps_fixed_views(gt_mesh)

        gt_images_dict = {}
        recon_images_dict = {}

        for res in RESOLUTIONS:
            recon_dir = os.path.join(OUTPUT_ROOT, layer_name, f"{model_id}_{res}")
            recon_path = os.path.join(recon_dir, "recon.obj")
            if not os.path.exists(recon_path):
                continue

            recon_mesh = trimesh.load(recon_path, process=False)
            recon_images = render_normal_maps_fixed_views(recon_mesh)

            # Save side-by-side
            preview_dir = os.path.join(OUTPUT_ROOT, "previews", layer_name)
            render_side_by_side(gt_images, recon_images, preview_dir, model_id, res)

            gt_images_dict[res] = gt_images
            recon_images_dict[res] = recon_images

        # Generate grid
        grid_path = os.path.join(OUTPUT_ROOT, "previews", layer_name, f"{model_id}_grid.png")
        render_model_grid(gt_images_dict, recon_images_dict, grid_path, model_id, RESOLUTIONS)
```

- [ ] **Step 2: Run preview rendering for Layer A**

Run: `.venv/bin/python scripts/ovoxel_repr_test.py --layer a --models helmet --resolutions 512 1024`

Expected: Side-by-side images and grid image generated in `experiments/ovoxel_repr_test/previews/layer_a/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/ovoxel_repr_test.py
git commit -m "feat: add preview rendering with side-by-side and grid views"
```

---

## Task 5: Summary Printer and Full Run

**Files:**
- Modify: `scripts/ovoxel_repr_test.py`

- [ ] **Step 1: Add summary table printer**

Add before `main()` in `scripts/ovoxel_repr_test.py`:

```python
def print_summary():
    """Print a formatted summary table from saved CSVs."""
    for layer, csv_name in [("Layer A (O-Voxel)", "results_a.csv"),
                            ("Layer B (SC-VAE)", "results_b.csv")]:
        csv_path = os.path.join(OUTPUT_ROOT, csv_name)
        if not os.path.exists(csv_path):
            continue

        print(f"\n--- {layer} ---")
        print(f"{'Model':<12} {'Res':>6} {'Voxels':>12} {'CD':>12} {'NC':>8} "
              f"{'F@0.005':>8} {'F@0.01':>8} {'F@0.05':>8} {'Error'}")
        print("-" * 95)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
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
```

- [ ] **Step 2: Run the complete pipeline (all layers, all models, all resolutions)**

Run: `.venv/bin/python scripts/ovoxel_repr_test.py --layer all`

Expected: Full results with metrics, preview images, and summary table.

- [ ] **Step 3: Commit all results**

```bash
git add scripts/ovoxel_repr_test.py
git commit -m "feat: add summary printer, complete ovoxel repr test script"
```

---

## Task 6: Review Results and Write Report

**Files:**
- Create: `experiments/ovoxel_repr_test/report.md`

- [ ] **Step 1: Inspect results**

Run: `.venv/bin/python -c "
import csv
for name in ['results_a.csv', 'results_b.csv']:
    path = f'experiments/ovoxel_repr_test/{name}'
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(row)
    except FileNotFoundError:
        pass
    print()
"`

- [ ] **Step 2: View preview grids**

Check the generated grid images in `experiments/ovoxel_repr_test/previews/`.

- [ ] **Step 3: Write report with findings**

Create `experiments/ovoxel_repr_test/report.md` with:
- Results tables for Layer A and Layer B
- Resolution-fidelity analysis per model
- Qualitative observations for each model's difficult structures
- Comparison of Layer A vs Layer B (compression overhead)
- Conclusions: what resolution is needed for these hard cases, where does O-Voxel break down

- [ ] **Step 4: Commit report**

```bash
git add experiments/ovoxel_repr_test/report.md
git commit -m "docs: add O-Voxel representation fidelity test report"
```
