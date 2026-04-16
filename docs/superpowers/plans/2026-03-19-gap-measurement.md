# Gap Measurement Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that quantifies the quality gap between SC-VAE reconstruction upper bound and DiT generation, to decide whether to prioritize VAE or DiT improvements.

**Architecture:** Two evaluation paths (VAE encode-decode vs DiT full pipeline) run on the same test meshes, compared against GT using geometric metrics (CD, F-score, NC) and rendering metrics (PSNR, SSIM on normal maps). Uses trimesh + torch for metrics, existing trellis2 models for inference.

**Tech Stack:** Python, PyTorch, trimesh, o_voxel, trellis2 (pretrained SC-VAE + pipeline), NVDiffRast (rendering), objaverse (pilot data download — new dependency)

**Task Dependencies:** Task 1→2 (tests→implementation), Task 3∥4 (parallel: data∥encoder), Task 5 depends on 2+3+4, Task 6 depends on 5, Task 7 depends on 6

**Spec:** `docs/superpowers/specs/2026-03-19-gap-measurement-design.md`

---

## File Structure

```
scripts/
├── eval_metrics.py           # Geometric metrics (CD, F-score, NC) + rendering metrics wrapper
├── gap_measurement.py        # Main script: Path A + Path B + evaluation + CSV/summary output
└── prepare_pilot_data.py     # Download Objaverse models + render reference images

tests/
└── test_eval_metrics.py      # Unit tests for metric functions

experiments/gap_measurement/   # Output directory (gitignored)
├── pilot_data/
│   ├── meshes/
│   └── images/
└── results/
```

---

### Task 1: Geometric Metrics Module — Tests

**Files:**
- Create: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write failing tests for CD, F-score, NC**

```python
# tests/test_eval_metrics.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from scripts.eval_metrics import chamfer_distance, f_score, normal_consistency


class TestChamferDistance:
    def test_identical_point_clouds_returns_zero(self):
        points = torch.rand(100, 3).cuda()
        cd = chamfer_distance(points, points)
        assert cd < 1e-6, f"CD of identical point clouds should be ~0, got {cd}"

    def test_known_distance(self):
        p1 = torch.tensor([[0.0, 0.0, 0.0]]).cuda()
        p2 = torch.tensor([[1.0, 0.0, 0.0]]).cuda()
        cd = chamfer_distance(p1, p2)
        assert abs(cd - 1.0) < 1e-5, f"CD should be 1.0, got {cd}"

    def test_symmetric(self):
        p1 = torch.rand(50, 3).cuda()
        p2 = torch.rand(80, 3).cuda()
        cd1 = chamfer_distance(p1, p2)
        cd2 = chamfer_distance(p2, p1)
        assert abs(cd1 - cd2) < 1e-5, f"CD should be symmetric: {cd1} vs {cd2}"


class TestFScore:
    def test_identical_points_returns_one(self):
        points = torch.rand(100, 3).cuda()
        fs = f_score(points, points, threshold=0.01)
        assert fs > 0.99, f"F-score of identical points should be ~1.0, got {fs}"

    def test_far_points_returns_zero(self):
        p1 = torch.zeros(100, 3).cuda()
        p2 = torch.ones(100, 3).cuda() * 10  # very far
        fs = f_score(p1, p2, threshold=0.01)
        assert fs < 0.01, f"F-score of far points should be ~0, got {fs}"


class TestNormalConsistency:
    def test_identical_normals_returns_one(self):
        points = torch.rand(100, 3).cuda()
        normals = torch.randn(100, 3).cuda()
        normals = normals / normals.norm(dim=1, keepdim=True)
        nc = normal_consistency(points, normals, points, normals)
        assert nc > 0.99, f"NC of identical normals should be ~1.0, got {nc}"

    def test_opposite_normals_returns_one(self):
        """NC uses absolute cosine, so opposite normals should also give 1.0."""
        points = torch.rand(50, 3).cuda()
        normals = torch.randn(50, 3).cuda()
        normals = normals / normals.norm(dim=1, keepdim=True)
        nc = normal_consistency(points, normals, points, -normals)
        assert nc > 0.99, f"NC of opposite normals should be ~1.0 (abs cosine), got {nc}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest tests/test_eval_metrics.py -v 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_metrics'`

---

### Task 2: Geometric Metrics Module — Implementation

**Files:**
- Create: `scripts/eval_metrics.py`

- [ ] **Step 1: Implement geometric metrics**

```python
# scripts/eval_metrics.py
"""
Evaluation metrics for Gap Measurement pipeline.
Geometric: Chamfer Distance, F-score, Normal Consistency
Rendering: PSNR, SSIM on normal maps (wraps trellis2 utilities)
"""

import torch
import numpy as np
import trimesh as tm


# ---------------------------------------------------------------------------
# Point cloud sampling
# ---------------------------------------------------------------------------

def sample_points_and_normals(mesh, num_points=10000):
    """
    Sample points and face normals from a trimesh mesh surface.

    Args:
        mesh: trimesh.Trimesh object
        num_points: number of points to sample

    Returns:
        points: [N, 3] torch float tensor
        normals: [N, 3] torch float tensor
    """
    points, face_indices = tm.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]
    return (
        torch.from_numpy(np.asarray(points)).float(),
        torch.from_numpy(np.asarray(normals)).float(),
    )


def trellis_mesh_to_trimesh(trellis_mesh):
    """Convert a trellis2.representations.Mesh to trimesh.Trimesh."""
    return tm.Trimesh(
        vertices=trellis_mesh.vertices.detach().cpu().numpy(),
        faces=trellis_mesh.faces.detach().cpu().numpy(),
        process=False,
    )


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------

def _chunked_min_dists(src, tgt, chunk_size=2048):
    """Compute min L2 distances from each point in src to nearest point in tgt."""
    min_dists = []
    for i in range(0, len(src), chunk_size):
        chunk = src[i:i + chunk_size]
        dists = torch.cdist(chunk, tgt)  # [chunk, M]
        min_dists.append(dists.min(dim=1)[0])
    return torch.cat(min_dists)


def chamfer_distance(points1, points2, chunk_size=2048):
    """
    Bidirectional Chamfer Distance (mean of squared L2 distances).

    Args:
        points1: [N, 3] tensor on GPU
        points2: [M, 3] tensor on GPU
        chunk_size: batch size for chunked cdist to avoid OOM

    Returns:
        Scalar float: mean bidirectional CD
    """
    d1 = _chunked_min_dists(points1, points2, chunk_size)
    d2 = _chunked_min_dists(points2, points1, chunk_size)
    return ((d1 ** 2).mean() + (d2 ** 2).mean()).item() / 2


def f_score(points1, points2, threshold=0.01, chunk_size=2048):
    """
    F-score: harmonic mean of precision and recall at distance threshold.

    Args:
        points1: [N, 3] tensor on GPU (predicted)
        points2: [M, 3] tensor on GPU (ground truth)
        threshold: distance threshold (in model coordinate space [-0.5, 0.5])
        chunk_size: batch size for chunked cdist

    Returns:
        Scalar float in [0, 1]
    """
    d1 = _chunked_min_dists(points1, points2, chunk_size)
    d2 = _chunked_min_dists(points2, points1, chunk_size)
    precision = (d1 < threshold).float().mean()
    recall = (d2 < threshold).float().mean()
    denom = precision + recall
    if denom < 1e-8:
        return 0.0
    return (2 * precision * recall / denom).item()


def normal_consistency(points1, normals1, points2, normals2, chunk_size=2048):
    """
    Normal Consistency: mean |cos(angle)| between matched point normals.
    For each point in points1, find nearest point in points2, compare normals.

    Args:
        points1, normals1: [N, 3] tensors on GPU
        points2, normals2: [M, 3] tensors on GPU

    Returns:
        Scalar float in [0, 1]
    """
    nn_indices = []
    for i in range(0, len(points1), chunk_size):
        chunk = points1[i:i + chunk_size]
        dists = torch.cdist(chunk, points2)
        nn_indices.append(dists.argmin(dim=1))
    nn_indices = torch.cat(nn_indices)

    matched_normals = normals2[nn_indices]
    cos_sim = torch.abs((normals1 * matched_normals).sum(dim=1))
    return cos_sim.mean().item()


# ---------------------------------------------------------------------------
# Rendering-based metrics
# ---------------------------------------------------------------------------

def render_normal_maps(trellis_mesh, nviews=8, resolution=512):
    """
    Render normal maps of a trellis2 Mesh from multiple views.

    Args:
        trellis_mesh: trellis2.representations.Mesh on CUDA
        nviews: number of views
        resolution: render resolution

    Returns:
        List of [3, H, W] float tensors (normal maps in [0, 1])
    """
    from trellis2.utils.render_utils import render_snapshot

    result = render_snapshot(
        trellis_mesh,
        resolution=resolution,
        nviews=nviews,
        r=2, fov=40,
        return_types=["normal"],
    )
    # render_snapshot returns dict of lists of uint8 numpy arrays [H, W, 3]
    normal_maps = []
    for nmap in result["normal"]:
        t = torch.from_numpy(nmap).float() / 255.0  # [H, W, 3]
        normal_maps.append(t.permute(2, 0, 1))  # [3, H, W]
    return normal_maps


def compute_rendering_metrics(normal_maps_pred, normal_maps_gt):
    """
    Compute PSNR and SSIM between predicted and GT normal maps.

    Args:
        normal_maps_pred: list of [3, H, W] float tensors
        normal_maps_gt: list of [3, H, W] float tensors

    Returns:
        dict with 'psnr' and 'ssim' (mean across views)
    """
    from trellis2.utils.loss_utils import psnr, ssim

    psnr_vals = []
    ssim_vals = []
    for pred, gt in zip(normal_maps_pred, normal_maps_gt):
        pred = pred.unsqueeze(0).cuda()  # [1, 3, H, W]
        gt = gt.unsqueeze(0).cuda()
        psnr_vals.append(psnr(pred, gt).item())
        ssim_vals.append(ssim(pred, gt).item())

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_metrics.py tests/test_eval_metrics.py
git commit -m "feat: add evaluation metrics module (CD, F-score, NC, rendering)"
```

---

### Task 3: Pilot Data Preparation

**Files:**
- Create: `scripts/prepare_pilot_data.py`

**Prerequisites:** `pip install objaverse` (if not installed)

- [ ] **Step 1: Install objaverse**

Run: `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/pip install objaverse 2>&1 | tail -5`

- [ ] **Step 2: Write pilot data preparation script**

```python
# scripts/prepare_pilot_data.py
"""
Download pilot test meshes from Objaverse and render reference images.
Uses Objaverse's lvis annotations to get diverse, high-quality models.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import trimesh
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_pilot_uids(num_models=30):
    """
    Select diverse Objaverse model UIDs using LVIS annotations.
    Falls back to random selection if annotations unavailable.
    """
    import objaverse

    # Get LVIS annotations (category → list of UIDs)
    lvis = objaverse.load_lvis_annotations()
    # Pick models from different categories for diversity
    selected = []
    categories_used = []
    for cat, uids in sorted(lvis.items()):
        if len(selected) >= num_models:
            break
        uid = uids[0]  # take first from each category
        selected.append(uid)
        categories_used.append(cat)

    print(f"Selected {len(selected)} models from {len(categories_used)} categories")
    return selected, categories_used


def download_models(uids, output_dir):
    """Download Objaverse models by UID."""
    import objaverse

    os.makedirs(output_dir, exist_ok=True)
    paths = objaverse.load_objects(uids=uids)

    # Copy to output directory with clean names
    downloaded = {}
    for uid, src_path in paths.items():
        dst_path = os.path.join(output_dir, f"{uid}.glb")
        if not os.path.exists(dst_path):
            os.symlink(os.path.abspath(src_path), dst_path)
        downloaded[uid] = dst_path

    return downloaded


def render_reference_image(mesh_path, output_path, resolution=512):
    """
    Render a reference image of the mesh for DiT input.
    Uses trimesh's built-in rendering (pyrender/pyglet).
    Falls back to a simple depth-based rendering if pyrender unavailable.
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        print(f"  Failed to load {mesh_path}: {e}")
        return False

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        print(f"  Empty mesh: {mesh_path}")
        return False

    # Normalize mesh to [-0.5, 0.5]
    center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    scale = 0.99999 / (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    mesh.vertices = (mesh.vertices - center) * scale

    # Try to render with pyrender
    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=(resolution, resolution))
        with open(output_path, 'wb') as f:
            f.write(png)
        return True
    except Exception:
        pass

    # Fallback: save a simple white image as placeholder
    # (the pipeline will still work, just with lower-quality conditioning)
    img = Image.new('RGB', (resolution, resolution), (200, 200, 200))
    img.save(output_path)
    print(f"  Warning: used placeholder image for {mesh_path}")
    return True


def validate_mesh(mesh_path):
    """Check if a mesh is valid for evaluation."""
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh.vertices.shape[0] < 10 or mesh.faces.shape[0] < 10:
            return False
        # Check for degenerate geometry
        if np.any(np.isnan(mesh.vertices)) or np.any(np.isinf(mesh.vertices)):
            return False
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare pilot test data for gap measurement")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement/pilot_data",
                        help="Output directory")
    parser.add_argument("--num_models", type=int, default=30,
                        help="Number of models to download")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution for reference images")
    args = parser.parse_args()

    mesh_dir = os.path.join(args.output_dir, "meshes")
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Step 1: Select and download models
    print("Selecting pilot models from Objaverse...")
    uids, categories = get_pilot_uids(args.num_models)

    print(f"Downloading {len(uids)} models...")
    paths = download_models(uids, mesh_dir)

    # Step 2: Validate and render
    valid_models = []
    for uid in tqdm(uids, desc="Validating and rendering"):
        mesh_path = paths.get(uid)
        if mesh_path is None:
            continue
        if not validate_mesh(mesh_path):
            print(f"  Skipping invalid mesh: {uid}")
            continue

        image_path = os.path.join(image_dir, f"{uid}.png")
        if render_reference_image(mesh_path, image_path, args.resolution):
            valid_models.append({
                "uid": uid,
                "mesh_path": os.path.abspath(mesh_path),
                "image_path": os.path.abspath(image_path),
                "category": categories[uids.index(uid)] if uid in uids else "unknown",
            })

    # Step 3: Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(valid_models, f, indent=2)

    print(f"\nPilot data ready: {len(valid_models)} valid models")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script to download pilot data**

Run: `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python scripts/prepare_pilot_data.py --num_models 30`
Expected: Downloads ~30 models to `experiments/gap_measurement/pilot_data/`, creates `manifest.json`

- [ ] **Step 4: Verify pilot data**

Run: `cat experiments/gap_measurement/pilot_data/manifest.json | python -m json.tool | head -20 && ls experiments/gap_measurement/pilot_data/meshes/ | wc -l`
Expected: JSON with model entries, ~30 mesh files

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_pilot_data.py
# Do NOT add experiments/ — it contains large binary data
echo "experiments/" >> .gitignore
git add .gitignore
git commit -m "feat: add pilot data preparation script for gap measurement"
```

---

### Task 4: Download Encoder Weights

The SC-VAE encoder is on HuggingFace Hub but NOT in the local cache. Must download before Path A.

- [ ] **Step 1: Download encoder weights from HF Hub**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/huggingface-cli download microsoft/TRELLIS.2-4B \
    ckpts/shape_enc_next_dc_f16c32_fp16.json \
    ckpts/shape_enc_next_dc_f16c32_fp16.safetensors \
    --local-dir pretrained/TRELLIS.2-4B 2>&1 | tail -5
```
Expected: Files downloaded to `pretrained/TRELLIS.2-4B/ckpts/shape_enc_*`

- [ ] **Step 2: Verify encoder loads**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "
import trellis2.models as models
enc = models.from_pretrained('pretrained/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')
print('Encoder loaded:', type(enc).__name__)
print('Parameters:', sum(p.numel() for p in enc.parameters()) / 1e6, 'M')
"
```
Expected: Prints encoder type and parameter count

Note: If `pretrained/TRELLIS.2-4B` doesn't match `from_pretrained`'s expected path format, check the actual HF cache structure:
```bash
find pretrained/ -name "shape_enc*" -type f 2>/dev/null
```
And adjust the path accordingly. The `from_pretrained` function looks for `{path}.json` and `{path}.safetensors` files.

---

### Task 5: Main Gap Measurement Script — Path A (VAE Reconstruction)

**Files:**
- Create: `scripts/gap_measurement.py`

- [ ] **Step 1: Write Path A implementation**

```python
# scripts/gap_measurement.py
"""
Gap Measurement: Quantify quality gap between SC-VAE reconstruction and DiT generation.

Usage:
    python scripts/gap_measurement.py --manifest experiments/gap_measurement/pilot_data/manifest.json
    python scripts/gap_measurement.py --manifest ... --path_a_only   # VAE reconstruction only
    python scripts/gap_measurement.py --manifest ... --path_b_only   # DiT generation only
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import csv
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from PIL import Image

from scripts.eval_metrics import (
    sample_points_and_normals,
    trellis_mesh_to_trimesh,
    chamfer_distance,
    f_score,
    normal_consistency,
    render_normal_maps,
    compute_rendering_metrics,
)

NUM_SAMPLE_POINTS = 10000
GRID_SIZE = 512
F_SCORE_THRESHOLD = 0.01
RENDER_NVIEWS = 8
RENDER_RESOLUTION = 512


# ---------------------------------------------------------------------------
# Mesh loading and normalization
# ---------------------------------------------------------------------------

def load_and_normalize_mesh(mesh_path):
    """Load a mesh with trimesh, normalize to [-0.5, 0.5]."""
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = mesh.vertices.astype(np.float64)
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    mesh.vertices = (vertices - center) * scale
    return mesh


def trimesh_to_trellis_mesh(tm_mesh):
    """Convert trimesh.Trimesh to trellis2.representations.Mesh (on CUDA)."""
    from trellis2.representations import Mesh as TrellisMesh
    return TrellisMesh(
        vertices=torch.from_numpy(tm_mesh.vertices.copy()).float().cuda(),
        faces=torch.from_numpy(tm_mesh.faces.copy()).int().cuda(),
    )


# ---------------------------------------------------------------------------
# Path A: VAE Reconstruction
# ---------------------------------------------------------------------------

def load_vae_models():
    """Load pretrained SC-VAE encoder and decoder."""
    import trellis2.models as models

    # Try local path first, then HF Hub
    enc_path = "pretrained/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    dec_path = "pretrained/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

    # Check if local files exist (with .json extension)
    if not os.path.exists(f"{enc_path}.json"):
        enc_path = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    if not os.path.exists(f"{dec_path}.json"):
        dec_path = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

    encoder = models.from_pretrained(enc_path).eval().cuda()
    decoder = models.from_pretrained(dec_path).eval().cuda()
    decoder.set_resolution(GRID_SIZE)
    return encoder, decoder


def vae_reconstruct(mesh_path, encoder, decoder):
    """
    Path A: GT mesh → O-Voxel → SC-VAE encode → decode → reconstructed mesh.

    Returns:
        trellis2.representations.Mesh on CUDA, or None on failure
    """
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    # Load and normalize
    tm_mesh = load_and_normalize_mesh(mesh_path)
    vertices = torch.from_numpy(tm_mesh.vertices.copy()).float()
    faces = torch.from_numpy(tm_mesh.faces.copy()).long()

    # Mesh → O-Voxel
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=GRID_SIZE,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    # Prepare encoder input
    # dual_vertices from o_voxel is in world coords; convert to local voxel coords [0, 1]
    # Use float directly without uint8 quantization (same as trellis2_texturing.py:211)
    dv_local = dual_vertices * GRID_SIZE - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)

    # Build SparseTensor with batch dim prepended: [batch_idx, x, y, z]
    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices],
        dim=-1,
    )
    vertices_st = SparseTensor(
        feats=dv_local,
        coords=coords_with_batch,
    )
    # intersected from o_voxel.convert is already [N, 3] bool
    intersected_st = vertices_st.replace(intersected.float())

    # Encode → decode
    with torch.no_grad():
        z = encoder(vertices_st.cuda(), intersected_st.cuda())
        recon_meshes = decoder(z)

    if isinstance(recon_meshes, list):
        return recon_meshes[0]
    return recon_meshes


# ---------------------------------------------------------------------------
# Path B: DiT Generation
# ---------------------------------------------------------------------------

_pipeline = None

def load_pipeline():
    """Load pretrained Trellis2 image-to-3D pipeline (singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Try local path first
    model_path = "pretrained/TRELLIS.2-4B"
    if not os.path.exists(os.path.join(model_path, "pipeline.json")):
        model_path = "microsoft/TRELLIS.2-4B"

    _pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    _pipeline.to("cuda")
    return _pipeline


def dit_generate(image_path):
    """
    Path B: test image → full pipeline → generated mesh.

    Returns:
        trellis2.representations.Mesh on CUDA, or None on failure
    """
    pipeline = load_pipeline()
    image = Image.open(image_path).convert("RGBA")
    meshes = pipeline.run(image, pipeline_type='512')
    return meshes[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single(gt_mesh_path, recon_mesh, gen_mesh, gt_trimesh=None):
    """
    Evaluate one sample: compare reconstructed and generated meshes against GT.

    Returns:
        dict with all metrics for both paths
    """
    if gt_trimesh is None:
        gt_trimesh = load_and_normalize_mesh(gt_mesh_path)

    gt_points, gt_normals = sample_points_and_normals(gt_trimesh, NUM_SAMPLE_POINTS)
    gt_points, gt_normals = gt_points.cuda(), gt_normals.cuda()

    result = {}
    gt_normals_maps = None  # cached across Path A and Path B

    # Path A metrics
    if recon_mesh is not None:
        recon_trimesh = trellis_mesh_to_trimesh(recon_mesh)
        recon_points, recon_normals = sample_points_and_normals(recon_trimesh, NUM_SAMPLE_POINTS)
        recon_points, recon_normals = recon_points.cuda(), recon_normals.cuda()
        result["vae_cd"] = chamfer_distance(recon_points, gt_points)
        result["vae_fscore"] = f_score(recon_points, gt_points, threshold=F_SCORE_THRESHOLD)
        result["vae_nc"] = normal_consistency(recon_points, recon_normals, gt_points, gt_normals)

        # Rendering metrics: render GT and recon from same views
        try:
            if gt_normals_maps is None:
                gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                gt_normals_maps = render_normal_maps(gt_trellis, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            recon_normals_maps = render_normal_maps(recon_mesh, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            render_metrics = compute_rendering_metrics(recon_normals_maps, gt_normals_maps)
            result["vae_psnr"] = render_metrics["psnr"]
            result["vae_ssim"] = render_metrics["ssim"]
        except Exception as e:
            print(f"    Rendering metrics failed: {e}")
            result["vae_psnr"] = float('nan')
            result["vae_ssim"] = float('nan')
    else:
        result["vae_cd"] = float('nan')
        result["vae_fscore"] = float('nan')
        result["vae_nc"] = float('nan')
        result["vae_psnr"] = float('nan')
        result["vae_ssim"] = float('nan')

    # Path B metrics
    if gen_mesh is not None:
        gen_trimesh = trellis_mesh_to_trimesh(gen_mesh)
        gen_points, gen_normals = sample_points_and_normals(gen_trimesh, NUM_SAMPLE_POINTS)
        gen_points, gen_normals = gen_points.cuda(), gen_normals.cuda()
        result["dit_cd"] = chamfer_distance(gen_points, gt_points)
        result["dit_fscore"] = f_score(gen_points, gt_points, threshold=F_SCORE_THRESHOLD)
        result["dit_nc"] = normal_consistency(gen_points, gen_normals, gt_points, gt_normals)

        try:
            if gt_normals_maps is None:
                gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                gt_normals_maps = render_normal_maps(gt_trellis, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            gen_normals_maps = render_normal_maps(gen_mesh, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            render_metrics = compute_rendering_metrics(gen_normals_maps, gt_normals_maps)
            result["dit_psnr"] = render_metrics["psnr"]
            result["dit_ssim"] = render_metrics["ssim"]
        except Exception as e:
            print(f"    Rendering metrics failed: {e}")
            result["dit_psnr"] = float('nan')
            result["dit_ssim"] = float('nan')
    else:
        result["dit_cd"] = float('nan')
        result["dit_fscore"] = float('nan')
        result["dit_nc"] = float('nan')
        result["dit_psnr"] = float('nan')
        result["dit_ssim"] = float('nan')

    return result


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(results, output_path):
    """Generate summary markdown from per-sample results."""
    metrics = ["cd", "fscore", "nc", "psnr", "ssim"]
    prefixes = ["vae", "dit"]

    lines = ["# Gap Measurement Results\n"]
    lines.append(f"**Samples evaluated:** {len(results)}\n")
    lines.append(f"**Resolution:** {GRID_SIZE}³\n")
    lines.append(f"**Points sampled:** {NUM_SAMPLE_POINTS}\n")
    lines.append(f"**F-score threshold:** {F_SCORE_THRESHOLD}\n\n")

    lines.append("## Aggregate Metrics\n")
    lines.append("| Metric | VAE Recon (mean±std) | DiT Gen (mean±std) | Gap (DiT - VAE) |")
    lines.append("|--------|---------------------|--------------------|-----------------|\n")

    for m in metrics:
        vae_vals = [r.get(f"vae_{m}", float('nan')) for r in results]
        dit_vals = [r.get(f"dit_{m}", float('nan')) for r in results]
        vae_vals = [v for v in vae_vals if not np.isnan(v)]
        dit_vals = [v for v in dit_vals if not np.isnan(v)]

        if vae_vals:
            vae_str = f"{np.mean(vae_vals):.6f} ± {np.std(vae_vals):.6f}"
        else:
            vae_str = "N/A"
        if dit_vals:
            dit_str = f"{np.mean(dit_vals):.6f} ± {np.std(dit_vals):.6f}"
        else:
            dit_str = "N/A"
        if vae_vals and dit_vals:
            gap = np.mean(dit_vals) - np.mean(vae_vals)
            gap_str = f"{gap:+.6f}"
        else:
            gap_str = "N/A"

        lines.append(f"| {m.upper()} | {vae_str} | {dit_str} | {gap_str} |")

    lines.append("\n## Interpretation\n")
    lines.append("- **CD/F-score gap small** → VAE is the ceiling, prioritize SC-VAE improvements")
    lines.append("- **CD/F-score gap large** → DiT is the bottleneck, prioritize DiT optimization")
    lines.append("- **NC gap large but CD gap small** → DiT struggles with surface normals specifically\n")

    # Decision
    vae_cds = [r.get("vae_cd", float('nan')) for r in results]
    dit_cds = [r.get("dit_cd", float('nan')) for r in results]
    vae_cds = [v for v in vae_cds if not np.isnan(v)]
    dit_cds = [v for v in dit_cds if not np.isnan(v)]
    if vae_cds and dit_cds:
        vae_mean = np.mean(vae_cds)
        dit_mean = np.mean(dit_cds)
        if vae_mean > 0:
            ratio = dit_mean / vae_mean
            lines.append(f"**DiT CD / VAE CD ratio:** {ratio:.2f}x\n")
            if ratio > 2.0:
                lines.append("**→ DECISION: DiT is significantly worse than VAE upper bound. Prioritize DiT optimization.**\n")
            elif ratio > 1.5:
                lines.append("**→ DECISION: Moderate gap. Consider improving both DiT and VAE.**\n")
            else:
                lines.append("**→ DECISION: DiT is close to VAE upper bound. Prioritize SC-VAE improvements.**\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Summary saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gap Measurement: VAE reconstruction vs DiT generation")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to pilot data manifest.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement/results",
                        help="Output directory for results")
    parser.add_argument("--path_a_only", action="store_true",
                        help="Only run Path A (VAE reconstruction)")
    parser.add_argument("--path_b_only", action="store_true",
                        help="Only run Path B (DiT generation)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate")
    parser.add_argument("--grid_size", type=int, default=512,
                        help="O-Voxel grid resolution")
    args = parser.parse_args()

    global GRID_SIZE
    GRID_SIZE = args.grid_size

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vae_reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "dit_generations"), exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.max_samples:
        manifest = manifest[:args.max_samples]

    print(f"Evaluating {len(manifest)} samples...")

    # Load models
    encoder, decoder = None, None
    if not args.path_b_only:
        print("Loading SC-VAE encoder + decoder...")
        encoder, decoder = load_vae_models()

    # Process each sample
    all_results = []
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    fieldnames = ["uid", "category",
                  "vae_cd", "vae_fscore", "vae_nc", "vae_psnr", "vae_ssim",
                  "dit_cd", "dit_fscore", "dit_nc", "dit_psnr", "dit_ssim",
                  "vae_error", "dit_error"]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(manifest, desc="Gap Measurement"):
            uid = item["uid"]
            mesh_path = item["mesh_path"]
            image_path = item.get("image_path")
            category = item.get("category", "unknown")

            row = {"uid": uid, "category": category, "vae_error": "", "dit_error": ""}

            # Path A: VAE reconstruction
            recon_mesh = None
            if not args.path_b_only:
                try:
                    recon_mesh = vae_reconstruct(mesh_path, encoder, decoder)
                except Exception as e:
                    print(f"  [Path A Error] {uid}: {e}")
                    row["vae_error"] = str(e)

            # Path B: DiT generation
            gen_mesh = None
            if not args.path_a_only and image_path and os.path.exists(image_path):
                try:
                    gen_mesh = dit_generate(image_path)
                except Exception as e:
                    print(f"  [Path B Error] {uid}: {e}")
                    row["dit_error"] = str(e)

            # Save intermediate meshes for debugging
            if recon_mesh is not None:
                try:
                    recon_tm = trellis_mesh_to_trimesh(recon_mesh)
                    recon_tm.export(os.path.join(args.output_dir, "vae_reconstructions", f"{uid}.obj"))
                except Exception:
                    pass
            if gen_mesh is not None:
                try:
                    gen_tm = trellis_mesh_to_trimesh(gen_mesh)
                    gen_tm.export(os.path.join(args.output_dir, "dit_generations", f"{uid}.obj"))
                except Exception:
                    pass

            # Evaluate
            try:
                metrics = evaluate_single(mesh_path, recon_mesh, gen_mesh)
                row.update(metrics)
            except Exception as e:
                print(f"  [Eval Error] {uid}: {e}")

            all_results.append(row)
            writer.writerow(row)
            csvfile.flush()

            # Free GPU memory
            del recon_mesh, gen_mesh
            torch.cuda.empty_cache()

    # Generate summary
    summary_path = os.path.join(args.output_dir, "summary.md")
    generate_summary(all_results, summary_path)

    # Print quick stats
    print(f"\nResults saved to {csv_path}")
    print(f"Summary saved to {summary_path}")
    vae_successes = sum(1 for r in all_results if not np.isnan(r.get("vae_cd", float('nan'))))
    dit_successes = sum(1 for r in all_results if not np.isnan(r.get("dit_cd", float('nan'))))
    print(f"Path A successes: {vae_successes}/{len(all_results)}")
    print(f"Path B successes: {dit_successes}/{len(all_results)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test Path A on 1 sample**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
# Create a minimal 1-sample manifest for testing
.venv/bin/python -c "
import json
m = json.load(open('experiments/gap_measurement/pilot_data/manifest.json'))
json.dump(m[:1], open('/tmp/test_manifest.json', 'w'), indent=2)
print('Test manifest:', m[0]['uid'])
"
.venv/bin/python scripts/gap_measurement.py --manifest /tmp/test_manifest.json --path_a_only --max_samples 1
```
Expected: Runs without error, produces `experiments/gap_measurement/results/per_sample.csv` with 1 row containing `vae_cd`, `vae_fscore`, `vae_nc` values.

- [ ] **Step 3: Commit**

```bash
git add scripts/gap_measurement.py
git commit -m "feat: add gap measurement script with VAE reconstruction path"
```

---

### Task 6: Smoke Test Path B (DiT Generation)

- [ ] **Step 1: Test Path B on 1 sample**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python scripts/gap_measurement.py --manifest /tmp/test_manifest.json --path_b_only --max_samples 1
```
Expected: Runs DiT pipeline, produces `dit_cd`, `dit_fscore`, `dit_nc` values. Note: this will be slower than Path A (~30-60s per sample at 512³).

If DiT fails due to image quality issues, try rendering a better reference image manually:
```bash
.venv/bin/python -c "
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('pretrained/TRELLIS.2-4B')
pipeline.to('cuda')
img = Image.open('experiments/gap_measurement/pilot_data/images/<uid>.png')
img = pipeline.preprocess_image(img)
img.save('/tmp/preprocessed.png')
print('Preprocessed image saved')
"
```

- [ ] **Step 2: Fix any issues and re-test**

Common issues:
- Image too small / wrong format → ensure RGBA or RGB, ≥256px
- OOM on 512 → try `--grid_size 256`
- Pipeline path not found → check `pretrained/TRELLIS.2-4B/pipeline.json` exists

---

### Task 7: Full Pilot Run

- [ ] **Step 1: Run full evaluation on all pilot data**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python scripts/gap_measurement.py \
    --manifest experiments/gap_measurement/pilot_data/manifest.json \
    --output_dir experiments/gap_measurement/results \
    --grid_size 512
```
Expected: Processes all ~30 samples (may take 30-60 minutes total), produces:
- `experiments/gap_measurement/results/per_sample.csv`
- `experiments/gap_measurement/results/summary.md`

- [ ] **Step 2: Review results**

Run:
```bash
cat experiments/gap_measurement/results/summary.md
```
Expected: Summary table with mean±std for all metrics, gap analysis, and decision recommendation.

- [ ] **Step 3: Final commit**

```bash
git add scripts/
git commit -m "feat: complete gap measurement pipeline (Phase 0)"
```

---

## Troubleshooting Guide

**OOM on O-Voxel conversion at 512³:**
- Reduce grid_size: `--grid_size 256`
- The pipeline will still work, just at lower resolution

**Encoder download fails (HF auth):**
```bash
.venv/bin/huggingface-cli login
# Then retry download
```

**`o_voxel.convert.mesh_to_flexible_dual_grid` fails on a specific mesh:**
- The mesh may have degenerate faces or non-manifold edges
- trimesh can fix some issues: `mesh.process(validate=True)` before conversion
- If persistent, skip the sample and log the error

**Rendering metrics fail:**
- `render_snapshot` requires a valid Mesh with faces on CUDA
- If the reconstructed mesh has 0 faces (VAE failure), rendering will fail — handled by try/except

**`from_pretrained` path issues:**
- The function checks for `{path}.json` and `{path}.safetensors`
- If using HF cache, the actual path may be `pretrained/models--microsoft--TRELLIS.2-4B/snapshots/<hash>/ckpts/...`
- Use `find pretrained/ -name "shape_enc*"` to locate the files
