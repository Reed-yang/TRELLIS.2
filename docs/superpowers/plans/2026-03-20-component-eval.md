# Component-Level Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-phase evaluation pipeline that measures upper-bound capability of each TRELLIS.2 component (VAE, Structure DiT, Shape DiT, Material DiT) on the paper-standard Toys4k-PBR test set.

**Architecture:** Progressive evaluation: (1) prepare Toys4k-PBR test set with complexity stratification, (2) extend eval_metrics with ICP/LPIPS/multi-threshold, (3) render 16 Blender views per asset, (4) Phase A: VAE baseline, (5) Phase B: DiT best-of-16, (6) Phase C: GT injection stage breakdown. Each phase reads from the previous phase's outputs.

**Tech Stack:** PyTorch, trimesh, Open3D (ICP), lpips (perceptual metric), Blender 3.0 CYCLES, o_voxel, trellis2 pipeline

**Spec:** `docs/superpowers/specs/2026-03-20-component-eval-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/prepare_toys4k.py` | Create | Download Toys4k, PBR filter, complexity stratification, output manifest |
| `scripts/eval_metrics.py` | Modify | Add ICP refinement, LPIPS, multi-threshold F-score, paper-aligned rendering |
| `scripts/component_eval.py` | Create | Main pipeline: `--phase a\|b\|c`, multi-GPU, all evaluation logic |
| `scripts/render_blender_cond.py` | Modify | Extend for Toys4k-scale batch rendering (reuse existing Blender logic) |
| `scripts/report_gen.py` | Create | Generate summary.md, by_tier.md, postprocess_comparison.md from CSV |
| `my-docs/component-eval-summary.md` | Create | Chinese summary of final results |

---

### Task 1: Extend eval_metrics.py — ICP, LPIPS, Multi-threshold F-score

**Files:**
- Modify: `scripts/eval_metrics.py`

This task upgrades the metrics module before any evaluation runs, since all phases depend on it.

- [ ] **Step 1: Add ICP refinement function**

Add after the `align_points_and_normals` function (line ~122):

```python
def icp_refine(pred_points, gt_points, initial_R, max_iterations=50, threshold=1e-8):
    """
    Refine alignment using ICP after 24-rotation coarse alignment.
    Rigid transform only (rotation + translation, no scale).

    Args:
        pred_points: [N, 3] tensor on GPU
        gt_points: [M, 3] tensor on GPU
        initial_R: [3, 3] rotation matrix from 24-rotation search

    Returns:
        transform: [4, 4] rigid transformation matrix (numpy)
    """
    import open3d as o3d

    # Apply initial rotation
    pred_rotated = (pred_points @ initial_R.T).cpu().numpy()
    gt_np = gt_points.cpu().numpy()

    # Build Open3D point clouds
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_rotated)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_np)

    # ICP with point-to-point
    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt,
        max_correspondence_distance=0.1,  # generous initial threshold
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
            relative_fitness=threshold,
            relative_rmse=threshold,
        ),
    )
    return reg.transformation


def find_best_rotation_24_with_icp(pred_points, gt_points, chunk_size=2048):
    """
    24-rotation search + ICP refinement.

    Returns:
        transform: [4, 4] numpy array (full rigid transform)
        best_cd: float (CD after alignment)
    """
    best_R, _ = find_best_rotation_24(pred_points, gt_points, chunk_size)
    transform = icp_refine(pred_points, gt_points, best_R)

    # Compute final CD with the refined transform
    pred_np = pred_points.cpu().numpy()
    pred_homo = np.hstack([pred_np, np.ones((len(pred_np), 1))])
    pred_transformed = (transform @ pred_homo.T).T[:, :3]
    pred_t = torch.from_numpy(pred_transformed).float().to(pred_points.device)
    cd = chamfer_distance(pred_t, gt_points)

    return transform, cd


def apply_transform_to_points(points, normals, transform):
    """
    Apply a [4, 4] rigid transform to points and normals.

    Args:
        points: [N, 3] tensor on GPU
        normals: [N, 3] tensor on GPU
        transform: [4, 4] numpy array

    Returns:
        transformed_points: [N, 3]
        transformed_normals: [N, 3]
    """
    R = torch.from_numpy(transform[:3, :3].copy()).float().to(points.device)
    t = torch.from_numpy(transform[:3, 3].copy()).float().to(points.device)
    transformed_points = points @ R.T + t
    transformed_normals = normals @ R.T
    return transformed_points, transformed_normals
```

- [ ] **Step 2: Add multi-threshold F-score function**

Add after the existing `f_score` function:

```python
F_SCORE_THRESHOLDS = [0.005, 0.01, 0.05, 0.1, 0.2]

def f_score_multi(points1, points2, thresholds=None, chunk_size=2048):
    """
    Compute F-score at multiple thresholds in one pass.

    Returns:
        dict: {threshold: f_score_value}
    """
    if thresholds is None:
        thresholds = F_SCORE_THRESHOLDS
    d1 = _chunked_min_dists(points1, points2, chunk_size)
    d2 = _chunked_min_dists(points2, points1, chunk_size)
    results = {}
    for tau in thresholds:
        precision = (d1 < tau).float().mean()
        recall = (d2 < tau).float().mean()
        denom = precision + recall
        results[tau] = (2 * precision * recall / denom).item() if denom > 1e-8 else 0.0
    return results
```

- [ ] **Step 3: Add LPIPS metric**

Add to the rendering metrics section:

```python
_lpips_model = None

def compute_lpips(normal_maps_pred, normal_maps_gt):
    """
    Compute LPIPS between predicted and GT normal maps.

    Returns:
        float: mean LPIPS across views (lower is better)
    """
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex').cuda().eval()

    vals = []
    for pred, gt in zip(normal_maps_pred, normal_maps_gt):
        # lpips expects [B, 3, H, W] in [-1, 1]
        pred_t = (pred.unsqueeze(0).cuda() * 2 - 1)
        gt_t = (gt.unsqueeze(0).cuda() * 2 - 1)
        with torch.no_grad():
            vals.append(_lpips_model(pred_t, gt_t).item())
    return float(np.mean(vals))
```

- [ ] **Step 4: Add paper-aligned 4-view rendering config**

Add a new rendering function:

```python
def render_normal_maps_paper_config(trellis_mesh, resolution=512):
    """
    Render normal maps matching TRELLIS.2 paper config:
    4 views, pitch 30deg, FoV 6deg, yaw 30/120/210/300 deg, radius 10.
    """
    from trellis2.utils.render_utils import render_snapshot
    from trellis2.representations import Mesh as TrellisMesh, MeshWithVoxel

    if isinstance(trellis_mesh, MeshWithVoxel):
        trellis_mesh = TrellisMesh(
            vertices=trellis_mesh.vertices,
            faces=trellis_mesh.faces,
        )

    # offset=(yaw_offset_rad, pitch_rad): yaw 30deg offset + pitch 30deg
    result = render_snapshot(
        trellis_mesh,
        resolution=resolution,
        nviews=4,
        r=10, fov=6,
        offset=(30 / 180 * np.pi, 30 / 180 * np.pi),
        return_types=["normal"],
    )
    normal_maps = []
    for nmap in result["normal"]:
        t = torch.from_numpy(nmap).float() / 255.0
        normal_maps.append(t.permute(2, 0, 1))
    return normal_maps
```

- [ ] **Step 5: Verify new functions work**

Run quick smoke test:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
python -c "
import torch, numpy as np
from scripts.eval_metrics import (
    f_score_multi, icp_refine, find_best_rotation_24_with_icp,
    apply_transform_to_points, compute_lpips, _build_24_rotations
)
# Test multi-threshold F-score
p1 = torch.randn(1000, 3).cuda()
p2 = p1 + torch.randn(1000, 3).cuda() * 0.01
result = f_score_multi(p1, p2)
print('F-score multi:', result)
assert all(0 <= v <= 1 for v in result.values())

# Test ICP
R = _build_24_rotations('cuda')[5]  # pick arbitrary rotation
p1_rot = p1 @ R.T
transform, cd = find_best_rotation_24_with_icp(p1_rot, p1)
print(f'ICP CD: {cd:.8f} (should be near 0)')
assert cd < 0.001

print('All smoke tests passed!')
"
```

Expected: All assertions pass, CD near 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_metrics.py
git commit -m "feat: add ICP refinement, LPIPS, multi-threshold F-score to eval_metrics"
```

---

### Task 2: Prepare Toys4k Test Set

**Files:**
- Create: `scripts/prepare_toys4k.py`

- [ ] **Step 1: Research Toys4k download mechanism**

Toys4k is available via the `toys4k` Python package or direct download. Check:

```bash
pip show toys4k 2>/dev/null || echo "Not installed"
python -c "import toys4k; print(toys4k.__file__)" 2>/dev/null || echo "Not available"
```

If not available, search for download URL:
```bash
pip install toys4k 2>/dev/null || echo "Need manual download"
```

Note: Toys4k may need to be downloaded from its official source (the paper's project page). The exact download mechanism should be determined at implementation time. The script should support both a `--data_dir` pointing to pre-downloaded data and an auto-download path.

- [ ] **Step 2: Write prepare_toys4k.py**

```python
"""
Prepare Toys4k-PBR test set for component evaluation.

Usage:
    python scripts/prepare_toys4k.py --data_dir /path/to/toys4k --output_dir experiments/component_eval/test_set
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import trimesh
import numpy as np
from tqdm import tqdm
from pathlib import Path


def check_pbr_materials(mesh_path):
    """
    Check if a mesh file has all three PBR maps: base_color, metallic, roughness.
    Returns True if all three are present.
    """
    try:
        scene = trimesh.load(mesh_path, process=False)
        # Handle both single mesh and scene
        if isinstance(scene, trimesh.Scene):
            for geom in scene.geometry.values():
                if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                    mat = geom.visual.material
                    if hasattr(mat, 'baseColorTexture') or hasattr(mat, 'baseColorFactor'):
                        # Check for metallic and roughness
                        has_metallic = hasattr(mat, 'metallicFactor') or hasattr(mat, 'metallicRoughnessTexture')
                        has_roughness = hasattr(mat, 'roughnessFactor') or hasattr(mat, 'metallicRoughnessTexture')
                        if has_metallic and has_roughness:
                            return True
        elif hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
            mat = scene.visual.material
            has_base = hasattr(mat, 'baseColorTexture') or hasattr(mat, 'baseColorFactor')
            has_metallic = hasattr(mat, 'metallicFactor') or hasattr(mat, 'metallicRoughnessTexture')
            has_roughness = hasattr(mat, 'roughnessFactor') or hasattr(mat, 'metallicRoughnessTexture')
            return has_base and has_metallic and has_roughness
    except Exception as e:
        print(f"  Error checking PBR for {mesh_path}: {e}")
    return False


def compute_complexity(mesh_path):
    """
    Compute complexity metrics for a mesh.
    Returns dict with face_count, vertex_count, and edge_ratio (Canny edge pixel ratio).
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        result = {
            'face_count': len(mesh.faces),
            'vertex_count': len(mesh.vertices),
            'edge_ratio': 0.0,
        }

        # Compute Canny edge ratio on a rendered normal map
        try:
            import cv2
            # Render a single-view normal map using trimesh
            scene = trimesh.Scene(mesh)
            # Use a simple camera setup
            png = scene.save_image(resolution=[256, 256])
            if png is not None:
                img = np.array(Image.open(io.BytesIO(png)).convert('L'))
                edges = cv2.Canny(img, 50, 150)
                result['edge_ratio'] = float(edges.sum() / 255) / (256 * 256)
        except Exception:
            pass  # edge_ratio stays 0 if rendering fails (headless)

        return result
    except Exception:
        return {'face_count': 0, 'vertex_count': 0, 'edge_ratio': 0.0}


def assign_tiers(items, key='face_count'):
    """Assign complexity tiers (1/2/3) based on tercile boundaries."""
    values = sorted([item[key] for item in items])
    t1 = values[len(values) // 3]
    t2 = values[2 * len(values) // 3]

    for item in items:
        v = item[key]
        if v <= t1:
            item['tier'] = 1
        elif v <= t2:
            item['tier'] = 2
        else:
            item['tier'] = 3

    return items, t1, t2


def main():
    parser = argparse.ArgumentParser(description="Prepare Toys4k-PBR test set")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to downloaded Toys4k dataset root")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/component_eval/test_set",
                        help="Output directory for manifest and metadata")
    parser.add_argument("--skip_pbr_filter", action="store_true",
                        help="Skip PBR filtering (use all meshes)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all mesh files
    data_path = Path(args.data_dir)
    mesh_files = sorted(
        list(data_path.rglob("*.glb")) +
        list(data_path.rglob("*.gltf")) +
        list(data_path.rglob("*.obj"))
    )
    print(f"Found {len(mesh_files)} mesh files in {args.data_dir}")

    # PBR filter
    items = []
    for mesh_path in tqdm(mesh_files, desc="PBR filtering"):
        uid = mesh_path.stem
        if not args.skip_pbr_filter:
            if not check_pbr_materials(str(mesh_path)):
                continue

        complexity = compute_complexity(str(mesh_path))
        items.append({
            'uid': uid,
            'mesh_path': str(mesh_path.resolve()),
            'category': mesh_path.parent.name,
            **complexity,
        })

    print(f"After PBR filter: {len(items)} assets (expected ~473)")

    # Complexity stratification
    items, t1, t2 = assign_tiers(items, key='face_count')

    tier_counts = {1: 0, 2: 0, 3: 0}
    for item in items:
        tier_counts[item['tier']] += 1
    print(f"Tier distribution: {tier_counts}")
    print(f"Thresholds: tier1 <= {t1} faces, tier2 <= {t2} faces, tier3 > {t2} faces")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(items, f, indent=2)

    # Save metadata summary
    meta = {
        'total_assets': len(items),
        'tier_thresholds': {'t1': t1, 't2': t2},
        'tier_counts': tier_counts,
        'source': 'Toys4k-PBR',
        'pbr_filtered': not args.skip_pbr_filter,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run on Toys4k data**

```bash
python scripts/prepare_toys4k.py --data_dir /path/to/toys4k --output_dir experiments/component_eval/test_set
```

Expected: `manifest.json` with ~473 entries, each containing `uid`, `mesh_path`, `category`, `face_count`, `vertex_count`, `tier`.

- [ ] **Step 4: Verify manifest quality**

```bash
python -c "
import json
with open('experiments/component_eval/test_set/manifest.json') as f:
    items = json.load(f)
print(f'Total: {len(items)}')
tiers = {}
for item in items:
    tiers[item['tier']] = tiers.get(item['tier'], 0) + 1
print(f'Tiers: {tiers}')
# Verify paths exist
import os
missing = [i['uid'] for i in items if not os.path.exists(i['mesh_path'])]
print(f'Missing meshes: {len(missing)}')
assert len(missing) == 0, f'Missing: {missing[:5]}'
print('Manifest OK!')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_toys4k.py
git commit -m "feat: add Toys4k-PBR test set preparation script"
```

---

### Task 3: Render Conditioning Images (Blender CYCLES)

**Files:**
- Modify: `scripts/render_blender_cond.py`

- [ ] **Step 1: Extend render_blender_cond.py for manifest-based batch rendering**

Add a new entry point that accepts a manifest.json and renders 16 views per asset. The existing `build_cond_views()` and Blender subprocess logic are reused.

Add to the end of the file (before `if __name__ == "__main__"`):

```python
def render_from_manifest(manifest_path, output_dir, num_views=16, rank=0, world_size=1):
    """
    Render conditioning images for all assets in a manifest.

    Args:
        manifest_path: path to manifest.json from prepare_toys4k.py
        output_dir: root output dir (renders stored in output_dir/{uid}/)
        num_views: views per asset (default 16, matching training)
        rank/world_size: for multi-process parallelism
    """
    with open(manifest_path) as f:
        items = json.load(f)

    # Shard
    if world_size > 1:
        start = len(items) * rank // world_size
        end = len(items) * (rank + 1) // world_size
        items = items[start:end]

    print(f"Rendering {len(items)} assets (rank {rank}/{world_size}), {num_views} views each")

    for item in tqdm(items, desc="Blender Rendering"):
        uid = item['uid']
        mesh_path = item['mesh_path']
        asset_dir = os.path.join(output_dir, uid)

        # Skip if already rendered
        if os.path.exists(os.path.join(asset_dir, f"{num_views-1:04d}.png")):
            continue

        os.makedirs(asset_dir, exist_ok=True)

        # Call Blender subprocess (reuse existing logic)
        cmd = [
            BLENDER_PATH, '--background', '--python', BLENDER_SCRIPT,
            '--', mesh_path, asset_dir,
            '--num_views', str(num_views),
            '--resolution', '1024',
        ]
        try:
            call(cmd, stdout=DEVNULL, stderr=DEVNULL, timeout=300)
        except Exception as e:
            print(f"  Render failed for {uid}: {e}")

    # Update manifest with render paths
    with open(manifest_path) as f:
        all_items = json.load(f)
    for item in all_items:
        uid = item['uid']
        renders_dir = os.path.join(output_dir, uid)
        item['renders_dir'] = renders_dir
        item['num_views'] = num_views
    with open(manifest_path, 'w') as f:
        json.dump(all_items, f, indent=2)
```

Update argparse to support manifest mode:

```python
# In main():
parser.add_argument("--manifest", type=str, help="Manifest JSON for batch rendering")
parser.add_argument("--render_output", type=str, default="experiments/component_eval/renders_cond")
parser.add_argument("--num_views", type=int, default=16)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)

# Add manifest mode handling:
if args.manifest:
    render_from_manifest(args.manifest, args.render_output, args.num_views, args.rank, args.world_size)
    return
```

- [ ] **Step 2: Launch parallel Blender rendering**

```bash
# Launch 6 parallel Blender processes (one per GPU for CYCLES)
for rank in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$rank python scripts/render_blender_cond.py \
        --manifest experiments/component_eval/test_set/manifest.json \
        --render_output experiments/component_eval/renders_cond \
        --num_views 16 --rank $rank --world_size 6 &
done
wait
echo "All rendering complete"
```

Note: This is the longest step (~10-20 hours). Start early and let it run.

- [ ] **Step 3: Verify rendered images**

```bash
python -c "
import json, os
from PIL import Image
import numpy as np

with open('experiments/component_eval/test_set/manifest.json') as f:
    items = json.load(f)

missing = 0
bad = 0
for item in items:
    renders_dir = item.get('renders_dir', '')
    for v in range(16):
        path = os.path.join(renders_dir, f'{v:04d}.png')
        if not os.path.exists(path):
            missing += 1
            continue
        img = np.array(Image.open(path))
        if img.std() < 5:  # gray placeholder
            bad += 1

print(f'Total: {len(items)*16}, Missing: {missing}, Bad (gray): {bad}')
assert missing == 0, f'{missing} missing renders'
print('Renders OK!')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/render_blender_cond.py
git commit -m "feat: extend Blender renderer for manifest-based batch conditioning"
```

---

### Task 4: Build component_eval.py — Core Pipeline

**Files:**
- Create: `scripts/component_eval.py`

This is the main evaluation script. It's structured as three phases sharing common utilities.

- [ ] **Step 1: Write the script skeleton with shared utilities**

```python
"""
Component-Level Evaluation Pipeline.

Usage:
    # Phase A: VAE reconstruction baseline
    python scripts/component_eval.py --phase a --manifest experiments/component_eval/test_set/manifest.json

    # Phase B: DiT generation (best-of-16), multi-GPU
    python scripts/component_eval.py --phase b --manifest ... --rank 0 --world_size 8

    # Phase C: GT injection stage breakdown
    python scripts/component_eval.py --phase c --manifest ... --phase_b_csv experiments/component_eval/phase_b/results/per_sample.csv
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import csv
import argparse
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from PIL import Image

from scripts.eval_metrics import (
    sample_points_and_normals,
    trellis_mesh_to_trimesh,
    chamfer_distance,
    f_score_multi,
    normal_consistency,
    find_best_rotation_24_with_icp,
    apply_transform_to_points,
    render_normal_maps,
    render_normal_maps_paper_config,
    compute_rendering_metrics,
    compute_lpips,
    F_SCORE_THRESHOLDS,
)

# Import shared utilities from gap_measurement.py
from scripts.gap_measurement import (
    load_and_normalize_mesh,
    trimesh_to_trellis_mesh,
    load_vae_models,
    vae_reconstruct,
    load_pipeline,
    _patch_gated_models,
)

GRID_SIZE = 512
PHASE_A_POINTS = [100_000, 1_000_000]  # both resolutions
PHASE_BC_POINTS = 100_000


def evaluate_geometric(pred_trimesh, gt_trimesh, num_points, align=False):
    """
    Compute all geometric metrics between pred and gt.

    Args:
        pred_trimesh: trimesh.Trimesh (predicted)
        gt_trimesh: trimesh.Trimesh (ground truth)
        num_points: number of surface points to sample
        align: if True, apply 24-rotation + ICP alignment

    Returns:
        dict with cd, f_scores (multi-threshold), nc, transform (if aligned)
    """
    pred_pts, pred_nrm = sample_points_and_normals(pred_trimesh, num_points)
    gt_pts, gt_nrm = sample_points_and_normals(gt_trimesh, num_points)
    pred_pts, pred_nrm = pred_pts.cuda(), pred_nrm.cuda()
    gt_pts, gt_nrm = gt_pts.cuda(), gt_nrm.cuda()

    transform = None
    if align:
        transform, _ = find_best_rotation_24_with_icp(pred_pts, gt_pts)
        pred_pts, pred_nrm = apply_transform_to_points(pred_pts, pred_nrm, transform)

    cd = chamfer_distance(pred_pts, gt_pts)
    f_scores = f_score_multi(pred_pts, gt_pts)
    nc = normal_consistency(pred_pts, pred_nrm, gt_pts, gt_nrm)

    result = {'cd': cd, 'nc': nc}
    for tau, val in f_scores.items():
        result[f'fscore_{tau}'] = val

    return result, transform
```

- [ ] **Step 2: Implement Phase A**

```python
def run_phase_a(manifest, output_dir):
    """Phase A: VAE reconstruction baseline on full test set."""
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)

    encoder, decoder = load_vae_models()

    csv_path = os.path.join(output_dir, 'results', 'per_sample.csv')
    fieldnames = ['uid', 'category', 'tier', 'face_count',
                  'cd_100k', 'cd_1m', 'nc',
                  'psnr', 'ssim', 'lpips', 'error']
    for tau in F_SCORE_THRESHOLDS:
        fieldnames.append(f'fscore_{tau}')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(manifest, desc="Phase A: VAE Recon"):
            uid = item['uid']
            row = {k: '' for k in fieldnames}
            row['uid'] = uid
            row['category'] = item.get('category', '')
            row['tier'] = item.get('tier', '')
            row['face_count'] = item.get('face_count', '')

            try:
                # VAE reconstruct
                recon_mesh = vae_reconstruct(item['mesh_path'], encoder, decoder)
                recon_trimesh = trellis_mesh_to_trimesh(recon_mesh)
                gt_trimesh = load_and_normalize_mesh(item['mesh_path'])

                # Save mesh
                recon_trimesh.export(os.path.join(output_dir, 'meshes', f'{uid}.obj'))

                # Metrics at 100K (for cross-phase comparison)
                metrics_100k, _ = evaluate_geometric(recon_trimesh, gt_trimesh, 100_000, align=False)
                row['cd_100k'] = metrics_100k['cd']
                row['nc'] = metrics_100k['nc']
                for tau in F_SCORE_THRESHOLDS:
                    row[f'fscore_{tau}'] = metrics_100k.get(f'fscore_{tau}', '')

                # Metrics at 1M (paper-aligned)
                metrics_1m, _ = evaluate_geometric(recon_trimesh, gt_trimesh, 1_000_000, align=False)
                row['cd_1m'] = metrics_1m['cd']

                # Rendering metrics (aligned mesh for normal maps)
                gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                gt_nmaps = render_normal_maps_paper_config(gt_trellis)
                recon_nmaps = render_normal_maps_paper_config(recon_mesh)
                render_m = compute_rendering_metrics(recon_nmaps, gt_nmaps)
                row['psnr'] = render_m['psnr']
                row['ssim'] = render_m['ssim']
                row['lpips'] = compute_lpips(recon_nmaps, gt_nmaps)

            except Exception as e:
                print(f"  [Phase A Error] {uid}: {e}")
                row['error'] = str(e)

            writer.writerow(row)
            csvfile.flush()

    print(f"Phase A complete. Results: {csv_path}")
```

- [ ] **Step 3: Implement Phase B**

```python
def dit_generate_single(image_path, pipeline, no_postprocess=False):
    """
    Run DiT generation for a single image.
    Returns (mesh_with_fill_holes, mesh_raw) or (mesh, None) depending on no_postprocess.
    """
    from trellis2.representations.mesh.base import Mesh as BaseMesh

    image = Image.open(image_path).convert("RGBA")

    if no_postprocess:
        # Monkey-patch fill_holes to no-op
        original_fill_holes = BaseMesh.fill_holes
        BaseMesh.fill_holes = lambda self, *a, **kw: None
        try:
            mesh_raw = pipeline.run(image, pipeline_type='512')[0]
        finally:
            BaseMesh.fill_holes = original_fill_holes
        return None, mesh_raw
    else:
        mesh_filled = pipeline.run(image, pipeline_type='512')[0]
        return mesh_filled, None


def run_phase_b(manifest, output_dir, rank=0, world_size=1):
    """Phase B: DiT generation, best-of-16 views, with/without fill_holes."""
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)

    # Shard manifest
    if world_size > 1:
        start = len(manifest) * rank // world_size
        end = len(manifest) * (rank + 1) // world_size
        manifest = manifest[start:end]

    pipeline = load_pipeline()

    csv_suffix = f"_rank{rank}" if world_size > 1 else ""
    csv_path = os.path.join(output_dir, 'results', f'per_sample{csv_suffix}.csv')

    fieldnames = ['uid', 'category', 'tier', 'face_count',
                  'best_view_idx', 'cd_filled', 'cd_raw', 'nc_filled', 'nc_raw',
                  'psnr', 'ssim', 'lpips']
    for tau in F_SCORE_THRESHOLDS:
        fieldnames.append(f'fscore_{tau}_filled')
        fieldnames.append(f'fscore_{tau}_raw')
    # Also store all 16 view CDs
    for v in range(16):
        fieldnames.append(f'cd_view_{v}')
    fieldnames.append('error')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(manifest, desc=f"Phase B (rank {rank})"):
            uid = item['uid']
            row = {k: '' for k in fieldnames}
            row['uid'] = uid
            row['category'] = item.get('category', '')
            row['tier'] = item.get('tier', '')
            row['face_count'] = item.get('face_count', '')

            try:
                gt_trimesh = load_and_normalize_mesh(item['mesh_path'])
                renders_dir = item.get('renders_dir', '')

                # Run DiT for all 16 views (with fill_holes)
                # Save meshes to disk to avoid OOM, keep only CDs in memory
                view_cds = []
                tmp_mesh_dir = os.path.join(output_dir, 'meshes', '_tmp', uid)
                os.makedirs(tmp_mesh_dir, exist_ok=True)
                for v in range(item.get('num_views', 16)):
                    img_path = os.path.join(renders_dir, f'{v:04d}.png')
                    if not os.path.exists(img_path):
                        view_cds.append(float('inf'))
                        continue

                    mesh_filled, _ = dit_generate_single(img_path, pipeline, no_postprocess=False)
                    mesh_tm = trellis_mesh_to_trimesh(mesh_filled)
                    metrics, _ = evaluate_geometric(mesh_tm, gt_trimesh, PHASE_BC_POINTS, align=True)
                    view_cds.append(metrics['cd'])
                    row[f'cd_view_{v}'] = metrics['cd']
                    # Save to disk, free GPU memory
                    mesh_tm.export(os.path.join(tmp_mesh_dir, f'{v:04d}.obj'))
                    del mesh_filled, mesh_tm
                    torch.cuda.empty_cache()

                # Select best view
                best_idx = int(np.argmin(view_cds))
                row['best_view_idx'] = best_idx
                row['cd_filled'] = view_cds[best_idx]

                # Re-generate best view for full metrics (reload not needed, just re-run)
                best_img_path_filled = os.path.join(renders_dir, f'{best_idx:04d}.png')
                best_mesh_filled, _ = dit_generate_single(best_img_path_filled, pipeline, no_postprocess=False)
                best_mesh = best_mesh_filled
                if best_mesh is not None:
                    best_tm = trellis_mesh_to_trimesh(best_mesh)
                    full_metrics, transform = evaluate_geometric(best_tm, gt_trimesh, PHASE_BC_POINTS, align=True)
                    row['cd_filled'] = full_metrics['cd']
                    row['nc_filled'] = full_metrics['nc']
                    for tau in F_SCORE_THRESHOLDS:
                        row[f'fscore_{tau}_filled'] = full_metrics.get(f'fscore_{tau}', '')

                    # Save best mesh
                    best_tm.export(os.path.join(output_dir, 'meshes', f'{uid}_filled.obj'))

                    # Rendering metrics (use aligned mesh)
                    gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                    gt_nmaps = render_normal_maps_paper_config(gt_trellis)
                    # Apply alignment transform to mesh before rendering
                    aligned_verts = best_tm.vertices.copy()
                    R = transform[:3, :3]
                    t = transform[:3, 3]
                    aligned_verts = aligned_verts @ R.T + t
                    aligned_tm = trimesh.Trimesh(vertices=aligned_verts, faces=best_tm.faces, process=False)
                    aligned_trellis = trimesh_to_trellis_mesh(aligned_tm)
                    pred_nmaps = render_normal_maps_paper_config(aligned_trellis)
                    render_m = compute_rendering_metrics(pred_nmaps, gt_nmaps)
                    row['psnr'] = render_m['psnr']
                    row['ssim'] = render_m['ssim']
                    row['lpips'] = compute_lpips(pred_nmaps, gt_nmaps)

                    # Re-run best view WITHOUT fill_holes
                    best_img_path = os.path.join(renders_dir, f'{best_idx:04d}.png')
                    _, mesh_raw = dit_generate_single(best_img_path, pipeline, no_postprocess=True)
                    raw_tm = trellis_mesh_to_trimesh(mesh_raw)
                    raw_metrics, _ = evaluate_geometric(raw_tm, gt_trimesh, PHASE_BC_POINTS, align=True)
                    row['cd_raw'] = raw_metrics['cd']
                    row['nc_raw'] = raw_metrics['nc']
                    for tau in F_SCORE_THRESHOLDS:
                        row[f'fscore_{tau}_raw'] = raw_metrics.get(f'fscore_{tau}', '')
                    raw_tm.export(os.path.join(output_dir, 'meshes', f'{uid}_raw.obj'))

                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [Phase B Error] {uid}: {e}")
                row['error'] = str(e)

            writer.writerow(row)
            csvfile.flush()

    print(f"Phase B complete (rank {rank}). Results: {csv_path}")
```

- [ ] **Step 4: Implement Phase C — GT Injection**

```python
def run_phase_c(manifest, output_dir, phase_b_csv, rank=0, world_size=1):
    """
    Phase C: GT injection experiments.
    Select 100 objects from Phase B, run 5 conditions each.
    """
    import pandas as pd

    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)

    # Load Phase B results to select 100 objects
    phase_b_df = pd.read_csv(phase_b_csv)
    manifest_dict = {item['uid']: item for item in manifest}

    # Stratified sampling: 33 per tier, spanning CD distribution
    selected_uids = []
    for tier in [1, 2, 3]:
        tier_df = phase_b_df[phase_b_df['tier'] == tier].sort_values('cd_filled')
        n = len(tier_df)
        if n == 0:
            continue
        # 11 from bottom quartile, 11 from median, 11 from top quartile
        q1_end = max(1, n // 4)
        q3_start = max(q1_end + 1, 3 * n // 4)
        mid_start = max(q1_end, n // 2 - 5)
        mid_end = min(q3_start, n // 2 + 6)

        bottom = tier_df.iloc[:q1_end].head(11)['uid'].tolist()
        middle = tier_df.iloc[mid_start:mid_end].head(11)['uid'].tolist()
        top = tier_df.iloc[q3_start:].head(11)['uid'].tolist()
        selected_uids.extend(bottom + middle + top)

    selected_uids = selected_uids[:100]
    print(f"Phase C: {len(selected_uids)} objects selected")

    # Shard
    if world_size > 1:
        start = len(selected_uids) * rank // world_size
        end = len(selected_uids) * (rank + 1) // world_size
        selected_uids = selected_uids[start:end]

    # Load models
    pipeline = load_pipeline()
    encoder, decoder = load_vae_models()

    # Prepare CSV
    csv_suffix = f"_rank{rank}" if world_size > 1 else ""
    csv_path = os.path.join(output_dir, 'results', f'per_sample{csv_suffix}.csv')
    conditions = ['baseline', 'c1_gt_struct', 'c2_gt_struct_shape', 'c3_gt_material', 'c4_gt_struct_material']
    fieldnames = ['uid', 'tier', 'best_view_idx']
    for cond in conditions:
        fieldnames.extend([f'{cond}_cd', f'{cond}_nc'])
        for tau in F_SCORE_THRESHOLDS:
            fieldnames.append(f'{cond}_fscore_{tau}')
    fieldnames.append('error')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for uid in tqdm(selected_uids, desc=f"Phase C (rank {rank})"):
            item = manifest_dict[uid]
            b_row = phase_b_df[phase_b_df['uid'] == uid].iloc[0]
            best_view_idx = int(b_row['best_view_idx'])

            row = {k: '' for k in fieldnames}
            row['uid'] = uid
            row['tier'] = item.get('tier', '')
            row['best_view_idx'] = best_view_idx

            try:
                gt_trimesh = load_and_normalize_mesh(item['mesh_path'])
                best_img_path = os.path.join(item['renders_dir'], f'{best_view_idx:04d}.png')
                image = Image.open(best_img_path).convert("RGBA")

                # Prepare GT latents
                gt_structure, gt_shape_slat, gt_material_slat = prepare_gt_latents(
                    item['mesh_path'], encoder, decoder, pipeline
                )

                # Run 5 conditions
                for cond_name, gt_overrides in [
                    ('baseline', {}),
                    ('c1_gt_struct', {'sparse_structure': gt_structure}),
                    ('c2_gt_struct_shape', {'sparse_structure': gt_structure, 'shape_slat': gt_shape_slat}),
                    ('c3_gt_material', {'material_slat': gt_material_slat}),
                    ('c4_gt_struct_material', {'sparse_structure': gt_structure, 'material_slat': gt_material_slat}),
                ]:
                    mesh = run_with_gt_injection(pipeline, image, gt_overrides)
                    mesh_tm = trellis_mesh_to_trimesh(mesh)
                    metrics, _ = evaluate_geometric(mesh_tm, gt_trimesh, PHASE_BC_POINTS, align=True)

                    row[f'{cond_name}_cd'] = metrics['cd']
                    row[f'{cond_name}_nc'] = metrics['nc']
                    for tau in F_SCORE_THRESHOLDS:
                        row[f'{cond_name}_fscore_{tau}'] = metrics.get(f'fscore_{tau}', '')

                    mesh_tm.export(os.path.join(output_dir, 'meshes', f'{uid}_{cond_name}.obj'))
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [Phase C Error] {uid}: {e}")
                row['error'] = str(e)

            writer.writerow(row)
            csvfile.flush()

    print(f"Phase C complete (rank {rank}). Results: {csv_path}")
```

- [ ] **Step 5: Implement GT injection helpers**

These functions prepare GT latents and run the pipeline with injected GT at specific stages. Implementation depends on pipeline internals discovered during execution — the structure below is the best estimate from code reading:

```python
def prepare_gt_latents(mesh_path, encoder, decoder, pipeline):
    """
    Prepare GT latent representations for injection.

    Returns:
        gt_structure: sparse structure coords from GT mesh
        gt_shape_slat: shape SLat from SC-VAE encoder
        gt_material_slat: material SLat (if material encoder available)
    """
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    # Load and convert to O-Voxel
    tm_mesh = load_and_normalize_mesh(mesh_path)
    vertices = torch.from_numpy(tm_mesh.vertices.copy()).float()
    faces = torch.from_numpy(tm_mesh.faces.copy()).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=GRID_SIZE,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    # GT sparse structure: active voxel coordinates
    gt_structure = voxel_indices.cuda()

    # GT shape latent via SC-VAE encoder
    dv_local = dual_vertices * GRID_SIZE - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)
    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices],
        dim=-1,
    )
    vertices_st = SparseTensor(feats=dv_local, coords=coords_with_batch)
    intersected_st = vertices_st.replace(intersected.float())

    with torch.no_grad():
        gt_shape_slat = encoder(vertices_st.cuda(), intersected_st.cuda())

    # GT material latent via material SC-VAE encoder
    gt_material_slat = None
    try:
        import trellis2.models as models
        mat_enc_path = "pretrained/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
        if not os.path.exists(f"{mat_enc_path}.json"):
            mat_enc_path = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
        mat_encoder = models.from_pretrained(mat_enc_path).eval().cuda()

        # Material encoder needs the mesh's PBR attributes projected onto O-Voxels.
        # Build material input from the GT mesh's textures.
        # Note: this requires the GT mesh to have PBR materials (guaranteed by Toys4k-PBR filter).
        # The exact input format depends on the material encoder architecture —
        # it takes SparseTensor with material features (base_color, metallic, roughness)
        # attached to the same voxel grid as shape.
        # Implementation detail: use the same O-Voxel coords, sample PBR textures
        # at dual vertex positions to build material features.
        # If the material encoder is not available in local weights, skip C2/C3 conditions.
        with torch.no_grad():
            # Attempt to encode — exact API TBD during implementation
            # gt_material_slat = mat_encoder(material_st.cuda())
            pass
    except Exception as e:
        print(f"  Warning: Material encoder not available, C3/C4 will use DiT material: {e}")

    return gt_structure, gt_shape_slat, gt_material_slat


def run_with_gt_injection(pipeline, image, gt_overrides):
    """
    Run pipeline with GT components injected at specific stages.

    gt_overrides can contain:
        'sparse_structure': GT voxel coords to replace structure DiT output
        'shape_slat': GT shape SLat to replace shape DiT output
        'material_slat': GT material SLat to replace material DiT output
    """
    from trellis2.representations.mesh.base import Mesh as BaseMesh

    # Preprocess image
    processed = pipeline.preprocess_image(image)
    torch.manual_seed(42)
    cond = pipeline.get_cond([processed], 512)

    # Stage 1: Sparse structure
    if 'sparse_structure' in gt_overrides:
        # Use GT structure directly
        coords = gt_overrides['sparse_structure']
        # Format: [N, 4] tensor with [batch_idx, x, y, z] coordinates
        if coords.dim() == 2 and coords.shape[1] == 3:
            coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int, device=coords.device), coords], dim=-1)
    else:
        coords = pipeline.sample_sparse_structure(cond, 32, 1, {})

    # Stage 2: Shape SLat
    if 'shape_slat' in gt_overrides:
        shape_slat = gt_overrides['shape_slat']
    else:
        shape_slat = pipeline.sample_shape_slat(
            cond, pipeline.models['shape_slat_flow_model_512'],
            coords, {}
        )

    # Stage 3: Material SLat
    if 'material_slat' in gt_overrides:
        tex_slat = gt_overrides['material_slat']
    else:
        tex_slat = pipeline.sample_tex_slat(
            cond, pipeline.models['tex_slat_flow_model_512'],
            shape_slat, {}
        )

    # Decode
    meshes = pipeline.decode_latent(shape_slat, tex_slat, 512)
    return meshes[0]
```

**Note:** The GT injection logic above is a best-effort implementation based on reading the pipeline code. The exact format of `coords`, `shape_slat`, and `tex_slat` may need adjustment during implementation. The key insight is that `pipeline.run()` at line 542-591 calls three stage functions in sequence — we replicate this flow but substitute GT outputs at specific stages.

- [ ] **Step 6: Add main entry point and argparse**

```python
def merge_rank_csvs(output_dir, prefix='per_sample'):
    """Merge per-rank CSV files into one."""
    import glob
    rank_files = sorted(glob.glob(os.path.join(output_dir, 'results', f'{prefix}_rank*.csv')))
    if not rank_files:
        return
    combined_path = os.path.join(output_dir, 'results', f'{prefix}.csv')
    with open(combined_path, 'w', newline='') as out_f:
        writer = None
        for rf in rank_files:
            with open(rf) as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    print(f"Merged {len(rank_files)} rank files -> {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Component-Level Evaluation")
    parser.add_argument("--phase", type=str, required=True, choices=['a', 'b', 'c', 'merge'],
                        help="Phase to run: a (VAE), b (DiT), c (GT injection), merge (combine rank CSVs)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to test set manifest.json")
    parser.add_argument("--output_dir", type=str, default="experiments/component_eval",
                        help="Root output directory")
    parser.add_argument("--phase_b_csv", type=str, default=None,
                        help="Phase B results CSV (required for Phase C)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=512)
    args = parser.parse_args()

    global GRID_SIZE
    GRID_SIZE = args.grid_size

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.phase == 'a':
        run_phase_a(manifest, os.path.join(args.output_dir, 'phase_a'))
    elif args.phase == 'b':
        run_phase_b(manifest, os.path.join(args.output_dir, 'phase_b'), args.rank, args.world_size)
    elif args.phase == 'c':
        assert args.phase_b_csv, "--phase_b_csv required for Phase C"
        run_phase_c(manifest, os.path.join(args.output_dir, 'phase_c'), args.phase_b_csv, args.rank, args.world_size)
    elif args.phase == 'merge':
        for phase in ['phase_a', 'phase_b', 'phase_c']:
            phase_dir = os.path.join(args.output_dir, phase)
            if os.path.exists(phase_dir):
                merge_rank_csvs(phase_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add scripts/component_eval.py
git commit -m "feat: add component evaluation pipeline with Phase A/B/C support"
```

---

### Task 5: Report Generator

**Files:**
- Create: `scripts/report_gen.py`

- [ ] **Step 1: Write report generation script**

```python
"""
Generate summary reports from component evaluation CSV results.

Usage:
    python scripts/report_gen.py --phase a --csv experiments/component_eval/phase_a/results/per_sample.csv
    python scripts/report_gen.py --phase b --csv experiments/component_eval/phase_b/results/per_sample.csv
    python scripts/report_gen.py --phase c --csv experiments/component_eval/phase_c/results/per_sample.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def report_phase_a(df, output_dir):
    """Generate Phase A reports."""
    lines = ["# Phase A: VAE Reconstruction Baseline\n"]
    lines.append(f"**Samples:** {len(df)} | **Test Set:** Toys4k-PBR\n")

    # Global summary
    lines.append("## Aggregate Metrics\n")
    lines.append("| Metric | Mean | Std | Median |\n|--------|------|-----|--------|\n")
    for col in ['cd_100k', 'cd_1m', 'nc', 'psnr', 'ssim', 'lpips']:
        if col in df.columns:
            vals = df[col].dropna()
            lines.append(f"| {col} | {vals.mean():.6f} | {vals.std():.6f} | {vals.median():.6f} |")

    # F-score table
    lines.append("\n## F-score at Multiple Thresholds\n")
    fscore_cols = [c for c in df.columns if c.startswith('fscore_')]
    if fscore_cols:
        lines.append("| Threshold | Mean | Std |\n|-----------|------|-----|\n")
        for col in sorted(fscore_cols):
            vals = df[col].dropna()
            tau = col.replace('fscore_', '')
            lines.append(f"| {tau} | {vals.mean():.4f} | {vals.std():.4f} |")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/summary.md", 'w') as f:
        f.write('\n'.join(lines))

    # By-tier breakdown
    tier_lines = ["# Phase A: By Complexity Tier\n"]
    for tier in sorted(df['tier'].unique()):
        tier_df = df[df['tier'] == tier]
        tier_lines.append(f"\n## Tier {tier} (n={len(tier_df)})\n")
        tier_lines.append("| Metric | Mean | Std |\n|--------|------|-----|\n")
        for col in ['cd_100k', 'nc', 'psnr', 'lpips']:
            if col in tier_df.columns:
                vals = tier_df[col].dropna()
                tier_lines.append(f"| {col} | {vals.mean():.6f} | {vals.std():.6f} |")

    with open(f"{output_dir}/by_tier.md", 'w') as f:
        f.write('\n'.join(tier_lines))


def report_phase_b(df, output_dir, phase_a_csv=None):
    """Generate Phase B reports including gap analysis and postprocess comparison."""
    lines = ["# Phase B: DiT Generation (Best-of-16)\n"]
    lines.append(f"**Samples:** {len(df)} | **Views per sample:** 16\n")

    # Load Phase A for gap computation
    vae_cd_mean = None
    if phase_a_csv:
        a_df = pd.read_csv(phase_a_csv)
        vae_cd_mean = a_df['cd_100k'].dropna().mean()
        lines.append(f"\n**VAE CD (Phase A, 100K):** {vae_cd_mean:.6f}\n")

    # DiT summary
    lines.append("## DiT Metrics (with fill_holes, best view)\n")
    lines.append("| Metric | Mean | Std | Median |\n|--------|------|-----|--------|\n")
    for col in ['cd_filled', 'nc_filled', 'psnr', 'ssim', 'lpips']:
        if col in df.columns:
            vals = df[col].dropna()
            lines.append(f"| {col} | {vals.mean():.6f} | {vals.std():.6f} | {vals.median():.6f} |")

    if vae_cd_mean and 'cd_filled' in df.columns:
        dit_cd = df['cd_filled'].dropna().mean()
        lines.append(f"\n**Gap Ratio (DiT/VAE CD):** {dit_cd/vae_cd_mean:.1f}x\n")

    # Best/worst cases
    valid_df = df.dropna(subset=['cd_filled']).sort_values('cd_filled')
    lines.append("\n## Best 10 DiT Cases\n")
    lines.append("| UID | CD | Tier | Best View |\n|-----|-----|------|-----------|")
    for _, r in valid_df.head(10).iterrows():
        lines.append(f"| {r['uid']} | {r['cd_filled']:.6f} | {r.get('tier','')} | {r.get('best_view_idx','')} |")

    lines.append("\n## Worst 10 DiT Cases\n")
    lines.append("| UID | CD | Tier | Best View |\n|-----|-----|------|-----------|")
    for _, r in valid_df.tail(10).iterrows():
        lines.append(f"| {r['uid']} | {r['cd_filled']:.6f} | {r.get('tier','')} | {r.get('best_view_idx','')} |")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/summary.md", 'w') as f:
        f.write('\n'.join(lines))

    # Postprocess comparison
    pp_lines = ["# Post-Processing (fill_holes) Impact\n"]
    if 'cd_raw' in df.columns and 'cd_filled' in df.columns:
        raw_mean = df['cd_raw'].dropna().mean()
        filled_mean = df['cd_filled'].dropna().mean()
        pp_lines.append(f"| Condition | CD Mean | NC Mean |\n|-----------|---------|---------|")
        pp_lines.append(f"| With fill_holes | {filled_mean:.6f} | {df['nc_filled'].dropna().mean():.4f} |")
        pp_lines.append(f"| Without fill_holes | {raw_mean:.6f} | {df['nc_raw'].dropna().mean():.4f} |")
        if filled_mean > 0:
            pp_lines.append(f"\n**fill_holes impact:** {(raw_mean - filled_mean) / filled_mean * 100:+.1f}% CD change")

    with open(f"{output_dir}/postprocess_comparison.md", 'w') as f:
        f.write('\n'.join(pp_lines))

    # By-tier
    tier_lines = ["# Phase B: By Complexity Tier\n"]
    for tier in sorted(df['tier'].dropna().unique()):
        tier_df = df[df['tier'] == tier]
        tier_lines.append(f"\n## Tier {int(tier)} (n={len(tier_df)})\n")
        tier_lines.append("| Metric | Mean | Std |\n|--------|------|-----|\n")
        for col in ['cd_filled', 'cd_raw', 'nc_filled']:
            if col in tier_df.columns:
                vals = tier_df[col].dropna()
                if len(vals) > 0:
                    tier_lines.append(f"| {col} | {vals.mean():.6f} | {vals.std():.6f} |")

    with open(f"{output_dir}/by_tier.md", 'w') as f:
        f.write('\n'.join(tier_lines))


def report_phase_c(df, output_dir):
    """Generate Phase C stage attribution report."""
    conditions = ['baseline', 'c1_gt_struct', 'c2_gt_struct_shape', 'c3_gt_material', 'c4_gt_struct_material']

    lines = ["# Phase C: Stage Attribution Analysis\n"]
    lines.append(f"**Samples:** {len(df)}\n")

    # Aggregate table
    lines.append("## CD by Condition\n")
    lines.append("| Condition | CD Mean | CD Std | Description |\n|-----------|---------|--------|-------------|")
    descriptions = {
        'baseline': 'Full DiT pipeline',
        'c1_gt_struct': 'GT structure + DiT shape + DiT material',
        'c2_gt_struct_shape': 'GT structure + GT shape + DiT material',
        'c3_gt_material': 'DiT structure + DiT shape + GT material',
        'c4_gt_struct_material': 'GT structure + DiT shape + GT material',
    }
    for cond in conditions:
        col = f'{cond}_cd'
        if col in df.columns:
            vals = df[col].dropna()
            lines.append(f"| {cond} | {vals.mean():.6f} | {vals.std():.6f} | {descriptions.get(cond, '')} |")

    # Attribution analysis
    lines.append("\n## Error Attribution\n")
    if all(f'{c}_cd' in df.columns for c in conditions):
        bl = df['baseline_cd'].dropna().mean()
        c1 = df['c1_gt_struct_cd'].dropna().mean()
        c2 = df['c2_gt_struct_shape_cd'].dropna().mean()
        c4 = df['c4_gt_struct_material_cd'].dropna().mean()

        struct_contrib = bl - c1
        shape_contrib = c1 - c2
        shape_isolated = c4  # with GT structure and material, remaining error is shape DiT
        mat_contrib = c1 - c4  # difference when material is GT vs DiT

        lines.append(f"- **Structure DiT error:** {struct_contrib:.6f} CD (Baseline - C1)")
        lines.append(f"- **Shape DiT error (cascaded):** {shape_contrib:.6f} CD (C1 - C2)")
        lines.append(f"- **Shape DiT isolated CD:** {shape_isolated:.6f} (C4, with GT struct+mat)")
        lines.append(f"- **Material DiT contribution:** {mat_contrib:.6f} CD (C1 - C4)")

        total = struct_contrib + shape_contrib + mat_contrib
        if total > 0:
            lines.append(f"\n**Relative contribution:** Structure {struct_contrib/total*100:.0f}% | Shape {shape_contrib/total*100:.0f}% | Material {mat_contrib/total*100:.0f}%")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/stage_attribution.md", 'w') as f:
        f.write('\n'.join(lines))

    # By-tier
    tier_lines = ["# Phase C: By Complexity Tier\n"]
    for tier in sorted(df['tier'].dropna().unique()):
        tier_df = df[df['tier'] == tier]
        tier_lines.append(f"\n## Tier {int(tier)} (n={len(tier_df)})\n")
        tier_lines.append("| Condition | CD Mean |\n|-----------|---------|")
        for cond in conditions:
            col = f'{cond}_cd'
            if col in tier_df.columns:
                vals = tier_df[col].dropna()
                if len(vals) > 0:
                    tier_lines.append(f"| {cond} | {vals.mean():.6f} |")

    with open(f"{output_dir}/by_tier.md", 'w') as f:
        f.write('\n'.join(tier_lines))


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("--phase", type=str, required=True, choices=['a', 'b', 'c'])
    parser.add_argument("--csv", type=str, required=True, help="Per-sample CSV path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (defaults to CSV parent)")
    parser.add_argument("--phase_a_csv", type=str, default=None,
                        help="Phase A CSV for gap computation in Phase B report")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.csv).parent)
    df = pd.read_csv(args.csv)

    if args.phase == 'a':
        report_phase_a(df, output_dir)
    elif args.phase == 'b':
        report_phase_b(df, output_dir, args.phase_a_csv)
    elif args.phase == 'c':
        report_phase_c(df, output_dir)

    print(f"Reports generated in {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/report_gen.py
git commit -m "feat: add report generator for component evaluation results"
```

---

### Task 6: Run Phase A — VAE Baseline

- [ ] **Step 1: Run Phase A**

```bash
python scripts/component_eval.py --phase a \
    --manifest experiments/component_eval/test_set/manifest.json \
    --output_dir experiments/component_eval
```

Expected: ~30 min, outputs `phase_a/results/per_sample.csv`

- [ ] **Step 2: Generate Phase A reports**

```bash
python scripts/report_gen.py --phase a \
    --csv experiments/component_eval/phase_a/results/per_sample.csv
```

- [ ] **Step 3: Verify Phase A results are reasonable**

```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/component_eval/phase_a/results/per_sample.csv')
print(f'Samples: {len(df)}')
print(f'CD 100K: {df.cd_100k.mean():.6f} +/- {df.cd_100k.std():.6f}')
print(f'CD 1M:   {df.cd_1m.mean():.6f} +/- {df.cd_1m.std():.6f}')
print(f'Errors:  {df.error.notna().sum()}')
# Phase 0 VAE CD was 0.000071 with 10K points — expect similar order of magnitude
assert df.cd_100k.mean() < 0.01, 'VAE CD suspiciously high'
print('Phase A sanity check passed!')
"
```

- [ ] **Step 4: Commit results**

```bash
git add experiments/component_eval/phase_a/results/
git commit -m "results: Phase A VAE reconstruction baseline on Toys4k-PBR"
```

---

### Task 7: Run Phase B — DiT Generation

- [ ] **Step 1: Launch multi-GPU Phase B**

```bash
for rank in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$rank python scripts/component_eval.py --phase b \
        --manifest experiments/component_eval/test_set/manifest.json \
        --output_dir experiments/component_eval \
        --rank $rank --world_size 8 &
done
wait
```

Expected: ~4 hours on 8×H100.

- [ ] **Step 2: Merge rank CSVs**

```bash
python scripts/component_eval.py --phase merge \
    --manifest experiments/component_eval/test_set/manifest.json \
    --output_dir experiments/component_eval
```

- [ ] **Step 3: Generate Phase B reports**

```bash
python scripts/report_gen.py --phase b \
    --csv experiments/component_eval/phase_b/results/per_sample.csv \
    --phase_a_csv experiments/component_eval/phase_a/results/per_sample.csv
```

- [ ] **Step 4: Verify results and commit**

```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/component_eval/phase_b/results/per_sample.csv')
print(f'Samples: {len(df)}')
print(f'DiT CD (filled): {df.cd_filled.mean():.6f}')
print(f'DiT CD (raw):    {df.cd_raw.mean():.6f}')
print(f'Errors: {df.error.notna().sum()}')
"
git add experiments/component_eval/phase_b/results/
git commit -m "results: Phase B DiT generation best-of-16 on Toys4k-PBR"
```

---

### Task 8: Run Phase C — GT Injection

- [ ] **Step 1: Launch Phase C**

```bash
for rank in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$rank python scripts/component_eval.py --phase c \
        --manifest experiments/component_eval/test_set/manifest.json \
        --output_dir experiments/component_eval \
        --phase_b_csv experiments/component_eval/phase_b/results/per_sample.csv \
        --rank $rank --world_size 8 &
done
wait
```

- [ ] **Step 2: Merge and generate reports**

```bash
python scripts/component_eval.py --phase merge \
    --manifest experiments/component_eval/test_set/manifest.json \
    --output_dir experiments/component_eval

python scripts/report_gen.py --phase c \
    --csv experiments/component_eval/phase_c/results/per_sample.csv
```

- [ ] **Step 3: Commit**

```bash
git add experiments/component_eval/phase_c/results/
git commit -m "results: Phase C GT injection stage attribution on Toys4k-PBR"
```

---

### Task 9: Final Summary and Documentation

**Files:**
- Create: `my-docs/component-eval-summary.md`
- Modify: `logs/progress.md`, `logs/findings.md`

- [ ] **Step 1: Write Chinese summary document**

Based on Phase A/B/C results, write `my-docs/component-eval-summary.md` containing:
- Key findings (which component is the bottleneck, by how much)
- Per-tier analysis
- Post-processing impact
- Stage attribution percentages
- Recommended optimization direction for Phase 1

This is written manually based on actual experiment results.

- [ ] **Step 2: Update progress and findings logs**

Update `logs/progress.md` with the Phase 1 evaluation timeline and milestones.
Update `logs/findings.md` with any new technical findings discovered during implementation.

- [ ] **Step 3: Final commit**

```bash
git add -f my-docs/component-eval-summary.md logs/progress.md logs/findings.md
git commit -m "docs: add component evaluation summary and update progress logs"
```

---

## Execution Order and Dependencies

```
Task 1 (eval_metrics) ──────────────────────┐
Task 2 (prepare_toys4k) ────────┐           │
Task 3 (render conditioning) ←──┘           │
Task 4 (component_eval.py) ←────────────────┘
Task 5 (report_gen.py)
Task 6 (Phase A run) ← Tasks 1,2,4
Task 7 (Phase B run) ← Tasks 1,3,4,6
  └─ includes merge step: `--phase merge` to combine rank CSVs
Task 8 (Phase C run) ← Tasks 1,4,7 (requires merged Phase B CSV)
Task 9 (Summary) ← Tasks 6,7,8
```

Note: Task 3 (Blender rendering, ~10-20 hours) should be started as early as possible — it can run in parallel with Tasks 1, 4, 5.

**Important implementation notes:**
- GT material latent injection (Phase C conditions C3, C4) depends on the material SC-VAE encoder being available. Check `pretrained/TRELLIS.2-4B/ckpts/` for `tex_enc_*` weights. If not available, download from HF Hub or skip C3/C4 conditions.
- The GT injection `run_with_gt_injection()` function is a best-effort implementation. The exact tensor formats for `coords`, `shape_slat`, and `tex_slat` may need debugging against the actual pipeline internals during Step 5 implementation.
