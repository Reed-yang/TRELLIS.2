# Gap Measurement Fix: Aligned Metrics + Better Conditioning + Aligned Previews

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three critical issues in the Phase 0 gap measurement: (1) recompute metrics with 24-rotation alignment, (2) select better front-facing conditioning images, (3) generate pose-aligned preview images. Then re-run the full experiment with better conditioning on 8×H100.

**Architecture:** A new unified script `scripts/reeval_gap.py` recomputes aligned metrics from saved OBJ files. Conditioning image selection is fixed in `render_blender_cond.py` to pick the best front-facing view via a scoring function on `transforms.json` camera parameters. Preview generation is added to `reeval_gap.py` using `eval_metrics.render_normal_maps()` with rotation alignment applied to DiT meshes before rendering. After fixing conditioning, a full re-run of DiT generation (Path B only) is launched on 8 GPUs in parallel using the existing `--rank/--world_size` mechanism.

**Tech Stack:** Python, PyTorch, trimesh, existing trellis2 pipeline, 8×H100 GPUs

**Spec:** Audit findings from current conversation (no separate spec doc)

**Task Dependencies:** Task 1 independent, Task 2 independent, Task 3 depends on Task 2 (needs fixed manifest), Task 4 depends on Tasks 1 + 3 (needs reeval script + re-generated DiT OBJs)

---

## File Structure

```
scripts/
├── eval_metrics.py              # NO CHANGE (already has find_best_rotation_24, align_points_and_normals)
├── reeval_gap.py                # CREATE: recompute aligned metrics + generate aligned previews from saved OBJs
├── render_blender_cond.py       # MODIFY: fix pick_best_view() to select front-facing view
└── gap_measurement.py           # NO CHANGE (re-used as-is for Path B re-run)

experiments/gap_measurement_v2/  # NEW output directory for fixed experiment
├── pilot_data/
│   └── manifest.json            # Updated manifest with best-view image paths
└── results/
    ├── per_sample.csv
    ├── summary.md
    ├── previews/
    ├── vae_reconstructions/     # Symlinked from gap_measurement_blender
    └── dit_generations/
```

---

### Task 1: Recompute Aligned Metrics from Saved OBJs

**Files:**
- Create: `scripts/reeval_gap.py`

This script loads the 30 saved GT meshes, VAE OBJs, and DiT OBJs, applies 24-rotation alignment, recomputes all geometric metrics, and outputs a corrected CSV + summary.

- [ ] **Step 1: Write the reeval script**

```python
# scripts/reeval_gap.py
"""
Recompute gap measurement metrics with 24-rotation alignment from saved OBJ files.
Also generates aligned preview images.

Usage:
    python scripts/reeval_gap.py \
        --manifest experiments/gap_measurement/pilot_data/manifest.json \
        --vae_dir experiments/gap_measurement_blender/results/vae_reconstructions \
        --dit_dir experiments/gap_measurement_blender/results/dit_generations \
        --cond_dir experiments/gap_measurement_blender/renders_cond \
        --output_dir experiments/gap_measurement_v2/results
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import csv
import math
import argparse
import torch
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from scripts.eval_metrics import (
    sample_points_and_normals,
    trellis_mesh_to_trimesh,
    chamfer_distance,
    f_score,
    normal_consistency,
    find_best_rotation_24,
    render_normal_maps,
)

NUM_SAMPLE_POINTS = 10000
F_SCORE_THRESHOLD = 0.01


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


def apply_rotation_to_trimesh(tm_mesh, R):
    """Apply a 3x3 rotation matrix to a trimesh mesh (in-place)."""
    R_np = R.cpu().numpy()
    tm_mesh.vertices = tm_mesh.vertices @ R_np.T
    return tm_mesh


def evaluate_aligned(gt_trimesh, vae_trimesh, dit_trimesh):
    """
    Evaluate one sample with 24-rotation alignment for DiT.
    Returns dict of metrics + best rotation matrix for DiT.
    """
    gt_pts, gt_norms = sample_points_and_normals(gt_trimesh, NUM_SAMPLE_POINTS)
    gt_pts, gt_norms = gt_pts.cuda(), gt_norms.cuda()

    result = {}
    best_R = None

    # Path A: VAE (no alignment needed - same coordinate system as GT)
    if vae_trimesh is not None and len(vae_trimesh.faces) > 0:
        vae_pts, vae_norms = sample_points_and_normals(vae_trimesh, NUM_SAMPLE_POINTS)
        vae_pts, vae_norms = vae_pts.cuda(), vae_norms.cuda()
        result["vae_cd"] = chamfer_distance(vae_pts, gt_pts)
        result["vae_fscore"] = f_score(vae_pts, gt_pts, threshold=F_SCORE_THRESHOLD)
        result["vae_nc"] = normal_consistency(vae_pts, vae_norms, gt_pts, gt_norms)

    # Path B: DiT (with 24-rotation alignment)
    if dit_trimesh is not None and len(dit_trimesh.faces) > 0:
        dit_pts, dit_norms = sample_points_and_normals(dit_trimesh, NUM_SAMPLE_POINTS)
        dit_pts, dit_norms = dit_pts.cuda(), dit_norms.cuda()

        # Find best rotation
        best_R, _ = find_best_rotation_24(dit_pts, gt_pts)
        aligned_pts = dit_pts @ best_R.T
        aligned_norms = dit_norms @ best_R.T

        result["dit_cd"] = chamfer_distance(aligned_pts, gt_pts)
        result["dit_fscore"] = f_score(aligned_pts, gt_pts, threshold=F_SCORE_THRESHOLD)
        result["dit_nc"] = normal_consistency(aligned_pts, aligned_norms, gt_pts, gt_norms)

    return result, best_R


def render_preview(gt_trimesh, vae_trimesh, dit_trimesh, best_R, cond_image_path, output_path):
    """
    Render a 4-row preview image:
      Row 0: conditioning image (padded to 4-column width)
      Row 1: GT normal maps (4 views)
      Row 2: VAE normal maps (4 views)
      Row 3: DiT normal maps (4 views, rotation-aligned)
    """
    nviews = 4
    resolution = 512

    rows = []

    # Row 0: conditioning image
    if cond_image_path and os.path.exists(cond_image_path):
        cond = np.array(Image.open(cond_image_path).convert("RGB").resize((resolution, resolution)))
    else:
        cond = np.full((resolution, resolution, 3), 128, dtype=np.uint8)
    # Pad to 4-column width: cond image on left, dark gray fill
    row0 = np.full((resolution, resolution * nviews, 3), 48, dtype=np.uint8)
    row0[:, :resolution] = cond
    rows.append(row0)

    # Row 1: GT
    bg_color = np.array([160, 160, 160], dtype=np.uint8)
    try:
        gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
        gt_nmaps = render_normal_maps(gt_trellis, nviews=nviews, resolution=resolution)
        row1_imgs = [(nm.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for nm in gt_nmaps]
        rows.append(np.concatenate(row1_imgs, axis=1))
    except Exception:
        rows.append(np.full((resolution, resolution * nviews, 3), 160, dtype=np.uint8))

    # Row 2: VAE
    try:
        vae_trellis = trimesh_to_trellis_mesh(vae_trimesh)
        vae_nmaps = render_normal_maps(vae_trellis, nviews=nviews, resolution=resolution)
        row2_imgs = [(nm.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for nm in vae_nmaps]
        rows.append(np.concatenate(row2_imgs, axis=1))
    except Exception:
        rows.append(np.full((resolution, resolution * nviews, 3), 160, dtype=np.uint8))

    # Row 3: DiT (rotation-aligned)
    try:
        if best_R is not None:
            dit_aligned = apply_rotation_to_trimesh(
                trimesh.Trimesh(vertices=dit_trimesh.vertices.copy(),
                                faces=dit_trimesh.faces.copy(), process=False),
                best_R
            )
        else:
            dit_aligned = dit_trimesh
        dit_trellis = trimesh_to_trellis_mesh(dit_aligned)
        dit_nmaps = render_normal_maps(dit_trellis, nviews=nviews, resolution=resolution)
        row3_imgs = [(nm.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for nm in dit_nmaps]
        rows.append(np.concatenate(row3_imgs, axis=1))
    except Exception:
        rows.append(np.full((resolution, resolution * nviews, 3), 160, dtype=np.uint8))

    # Composite
    composite = np.concatenate(rows, axis=0)
    Image.fromarray(composite).save(output_path)


def generate_summary(results, output_path, extra_info=""):
    """Generate summary markdown from per-sample results."""
    metrics = ["cd", "fscore", "nc"]

    lines = ["# Gap Measurement Results (Aligned)\n"]
    lines.append(f"**Samples evaluated:** {len(results)}\n")
    lines.append(f"**Points sampled:** {NUM_SAMPLE_POINTS}\n")
    lines.append(f"**F-score threshold:** {F_SCORE_THRESHOLD}\n")
    lines.append(f"**Alignment:** 24 axis-aligned rotations applied to DiT before metrics\n")
    if extra_info:
        lines.append(f"\n{extra_info}\n")

    lines.append("\n## Aggregate Metrics\n")
    lines.append("| Metric | VAE Recon (mean +- std) | DiT Gen (mean +- std) | Ratio (DiT/VAE) |")
    lines.append("|--------|------------------------|----------------------|-----------------|")

    for m in metrics:
        vae_vals = [r[f"vae_{m}"] for r in results if f"vae_{m}" in r and not np.isnan(r[f"vae_{m}"])]
        dit_vals = [r[f"dit_{m}"] for r in results if f"dit_{m}" in r and not np.isnan(r[f"dit_{m}"])]
        if vae_vals:
            vae_str = f"{np.mean(vae_vals):.6f} +- {np.std(vae_vals):.6f}"
        else:
            vae_str = "N/A"
        if dit_vals:
            dit_str = f"{np.mean(dit_vals):.6f} +- {np.std(dit_vals):.6f}"
        else:
            dit_str = "N/A"
        if vae_vals and dit_vals and np.mean(vae_vals) > 0:
            if m == "cd":
                ratio_str = f"{np.mean(dit_vals)/np.mean(vae_vals):.1f}x"
            else:
                ratio_str = f"{np.mean(dit_vals)/np.mean(vae_vals):.3f}"
        else:
            ratio_str = "N/A"
        lines.append(f"| {m.upper()} | {vae_str} | {dit_str} | {ratio_str} |")

    lines.append("")

    # Per-sample detail for worst/best DiT cases
    dit_cds = [(r.get("dit_cd", float('inf')), r.get("uid", "?"), r.get("category", "?")) for r in results]
    dit_cds = [(cd, uid, cat) for cd, uid, cat in dit_cds if not np.isnan(cd) and cd < float('inf')]
    if dit_cds:
        dit_cds.sort()
        lines.append("## Best DiT Cases (lowest CD)")
        lines.append("| Category | DiT CD | VAE CD | Ratio |")
        lines.append("|----------|--------|--------|-------|")
        for cd, uid, cat in dit_cds[:5]:
            vae_cd = next((r["vae_cd"] for r in results if r.get("uid") == uid), float('nan'))
            ratio = cd / vae_cd if vae_cd > 0 else float('inf')
            lines.append(f"| {cat} | {cd:.6f} | {vae_cd:.6f} | {ratio:.1f}x |")

        lines.append("\n## Worst DiT Cases (highest CD)")
        lines.append("| Category | DiT CD | VAE CD | Ratio |")
        lines.append("|----------|--------|--------|-------|")
        for cd, uid, cat in dit_cds[-5:]:
            vae_cd = next((r["vae_cd"] for r in results if r.get("uid") == uid), float('nan'))
            ratio = cd / vae_cd if vae_cd > 0 else float('inf')
            lines.append(f"| {cat} | {cd:.6f} | {vae_cd:.6f} | {ratio:.1f}x |")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Recompute aligned metrics from saved OBJ files")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--vae_dir", type=str, required=True)
    parser.add_argument("--dit_dir", type=str, required=True)
    parser.add_argument("--cond_dir", type=str, default=None,
                        help="Directory with renders_cond/{uid}/ for conditioning images")
    parser.add_argument("--cond_view", type=str, default="000.png",
                        help="Which view file to use as conditioning image in previews")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--previews", action="store_true", help="Generate preview images")
    parser.add_argument("--extra_info", type=str, default="", help="Extra info line for summary")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.previews:
        os.makedirs(os.path.join(args.output_dir, "previews"), exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_results = []
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    fieldnames = ["uid", "category", "vae_cd", "vae_fscore", "vae_nc",
                  "dit_cd", "dit_fscore", "dit_nc"]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(manifest, desc="Recomputing aligned metrics"):
            uid = item["uid"]
            category = item.get("category", "unknown")
            mesh_path = item["mesh_path"]

            vae_path = os.path.join(args.vae_dir, f"{uid}.obj")
            dit_path = os.path.join(args.dit_dir, f"{uid}.obj")

            # Load meshes
            gt_tm = load_and_normalize_mesh(mesh_path)
            vae_tm = trimesh.load(vae_path, force="mesh") if os.path.exists(vae_path) else None
            dit_tm = trimesh.load(dit_path, force="mesh") if os.path.exists(dit_path) else None

            # Evaluate with alignment
            metrics, best_R = evaluate_aligned(gt_tm, vae_tm, dit_tm)
            row = {"uid": uid, "category": category}
            row.update({k: v for k, v in metrics.items()})
            for k in fieldnames:
                if k not in row:
                    row[k] = float('nan')

            all_results.append(row)
            writer.writerow(row)
            csvfile.flush()

            # Preview
            if args.previews:
                cond_path = None
                if args.cond_dir:
                    cond_path = os.path.join(args.cond_dir, uid, args.cond_view)
                elif item.get("image_path"):
                    cond_path = item["image_path"]
                preview_path = os.path.join(args.output_dir, "previews",
                                            f"{category}_{uid}.png")
                try:
                    render_preview(gt_tm, vae_tm, dit_tm, best_R, cond_path, preview_path)
                except Exception as e:
                    print(f"  Preview failed for {uid}: {e}")

            torch.cuda.empty_cache()

    generate_summary(all_results, os.path.join(args.output_dir, "summary.md"), args.extra_info)
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run reeval on existing Blender experiment data**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/reeval_gap.py \
    --manifest experiments/gap_measurement/pilot_data/manifest.json \
    --vae_dir experiments/gap_measurement_blender/results/vae_reconstructions \
    --dit_dir experiments/gap_measurement_blender/results/dit_generations \
    --cond_dir experiments/gap_measurement_blender/renders_cond \
    --output_dir experiments/gap_measurement_v2/results_baseline \
    --previews \
    --extra_info "Baseline: random Blender view 000.png conditioning"
```
Expected: ~5 minutes (no generation, just metric recompute + rendering previews). Produces corrected per_sample.csv and summary.md with aligned metrics.

- [ ] **Step 3: Review corrected baseline numbers**

Run: `cat experiments/gap_measurement_v2/results_baseline/summary.md`
Expected: CD ratio should be significantly lower than 406x (likely ~20-50x based on 5-sample spot check).

- [ ] **Step 4: Commit**

```bash
git add scripts/reeval_gap.py
git commit -m "feat: add reeval script for aligned metrics + preview generation"
```

---

### Task 2: Fix Conditioning Image Selection

**Files:**
- Modify: `scripts/render_blender_cond.py` (function `pick_best_view`)

Replace the naive `000.png` selection with a scoring function that picks the most front-facing view with reasonable FOV.

- [ ] **Step 1: Update pick_best_view() in render_blender_cond.py**

Replace the existing `pick_best_view` function (around line 223-231) with:

```python
def pick_best_view(renders_dir, uid):
    """
    Pick the best front-facing view from rendered views using camera parameters.
    Scoring: prefer low |elevation|, azimuth near 0° or 180°, FOV in [30°, 50°].
    Falls back to 000.png if transforms.json is missing.
    """
    transforms_path = os.path.join(renders_dir, uid, 'transforms.json')
    if not os.path.exists(transforms_path):
        fallback = os.path.join(renders_dir, uid, '000.png')
        return fallback if os.path.exists(fallback) else None

    with open(transforms_path) as f:
        data = json.load(f)

    best_view = None
    best_score = float('inf')

    for frame in data.get('frames', []):
        m = frame['transform_matrix']
        cam_pos = [m[0][3], m[1][3], m[2][3]]
        r = math.sqrt(sum(c**2 for c in cam_pos))
        if r < 1e-6:
            continue

        elev_deg = math.degrees(math.asin(cam_pos[2] / r))
        azim_deg = math.degrees(math.atan2(cam_pos[1], cam_pos[0]))
        fov_deg = math.degrees(frame.get('camera_angle_x', 0.7))

        # Prefer: |elevation| small, azimuth near 0° or ±180°, FOV near 40°
        azim_front = min(abs(azim_deg), abs(abs(azim_deg) - 180))
        score = abs(elev_deg) * 1.0 + azim_front * 0.5 + abs(fov_deg - 40) * 0.3

        if score < best_score:
            best_score = score
            best_view = frame['file_path']

    if best_view:
        path = os.path.join(renders_dir, uid, best_view)
        if os.path.exists(path):
            return path

    fallback = os.path.join(renders_dir, uid, '000.png')
    return fallback if os.path.exists(fallback) else None
```

Also add `import math` at the top of the file if not already present.

- [ ] **Step 2: Create updated manifest with best-view images**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "
import sys, os, json, math
sys.path.insert(0, '.')
from scripts.render_blender_cond import pick_best_view

renders_dir = 'experiments/gap_measurement_blender/renders_cond'
manifest = json.load(open('experiments/gap_measurement/pilot_data/manifest.json'))

updated = []
for item in manifest:
    uid = item['uid']
    best = pick_best_view(renders_dir, uid)
    if best:
        new_item = dict(item)
        new_item['image_path'] = os.path.abspath(best)
        new_item['image_source'] = 'blender_cycles_best_view'
        updated.append(new_item)
        view_name = os.path.basename(best)
        print(f'  {item.get(\"category\",\"?\"):20s} -> {view_name}')
    else:
        print(f'  {item.get(\"category\",\"?\"):20s} -> MISSING')

os.makedirs('experiments/gap_measurement_v2/pilot_data', exist_ok=True)
json.dump(updated, open('experiments/gap_measurement_v2/pilot_data/manifest.json', 'w'), indent=2)
print(f'\nManifest: {len(updated)} models -> experiments/gap_measurement_v2/pilot_data/manifest.json')
"
```
Expected: Each model maps to a different view (not all 000.png). Views should be near elevation 0° with moderate FOV.

- [ ] **Step 3: Commit**

```bash
git add scripts/render_blender_cond.py
git commit -m "fix: select best front-facing Blender view instead of arbitrary view 000"
```

---

### Task 3: Re-run DiT Generation with Better Conditioning (8-GPU Parallel)

**Files:**
- No new files; uses existing `scripts/gap_measurement.py` with `--path_b_only --rank/--world_size`

Re-run only Path B (DiT generation) with the updated best-view manifest. Path A (VAE) results are unchanged — symlink from previous run.

- [ ] **Step 1: Prepare output directory and symlink VAE results**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p experiments/gap_measurement_v2/results/vae_reconstructions
mkdir -p experiments/gap_measurement_v2/results/dit_generations
mkdir -p experiments/gap_measurement_v2/results/rendered_refs

# Symlink VAE reconstructions from previous run (unchanged)
for f in experiments/gap_measurement_blender/results/vae_reconstructions/*.obj; do
    ln -sf "$(realpath "$f")" "experiments/gap_measurement_v2/results/vae_reconstructions/$(basename "$f")"
done
echo "Symlinked $(ls experiments/gap_measurement_v2/results/vae_reconstructions/ | wc -l) VAE OBJs"
```

- [ ] **Step 2: Launch 8-GPU parallel DiT generation**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2

# Launch 8 parallel workers (one per GPU)
for rank in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$rank .venv/bin/python scripts/gap_measurement.py \
        --manifest experiments/gap_measurement_v2/pilot_data/manifest.json \
        --output_dir experiments/gap_measurement_v2/results \
        --path_b_only \
        --grid_size 512 \
        --rank $rank \
        --world_size 8 \
        > "experiments/gap_measurement_v2/log_rank${rank}.txt" 2>&1 &
    echo "Launched rank $rank on GPU $rank"
done
echo "All 8 workers launched. Waiting..."
wait
echo "All workers done."
```
Expected: 30 models / 8 GPUs ≈ 3-4 models per GPU. ~5-10 min per model at 512³. Total ~15-40 minutes.

- [ ] **Step 3: Merge per-rank CSVs**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "
import csv, glob

# Read header from first file
files = sorted(glob.glob('experiments/gap_measurement_v2/results/per_sample_rank*.csv'))
if not files:
    print('ERROR: no per_sample_rank*.csv found')
    exit(1)

all_rows = []
with open(files[0]) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    all_rows.extend(list(reader))
for fpath in files[1:]:
    with open(fpath) as f:
        all_rows.extend(list(csv.DictReader(f)))

with open('experiments/gap_measurement_v2/results/per_sample_dit_only.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f'Merged {len(all_rows)} rows from {len(files)} rank files')
"
```
Expected: Single merged CSV with 30 rows of DiT results.

- [ ] **Step 4: Verify all 30 DiT OBJs were generated**

Run:
```bash
ls experiments/gap_measurement_v2/results/dit_generations/*.obj | wc -l
```
Expected: 30

---

### Task 4: Final Reeval with Aligned Metrics + Previews

**Files:**
- Uses `scripts/reeval_gap.py` from Task 1

- [ ] **Step 1: Run final reeval on v2 experiment data**

Note: omit `--cond_dir` so the preview uses `image_path` from the manifest (the actual best-view conditioning image), not the arbitrary 000.png.

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/reeval_gap.py \
    --manifest experiments/gap_measurement_v2/pilot_data/manifest.json \
    --vae_dir experiments/gap_measurement_v2/results/vae_reconstructions \
    --dit_dir experiments/gap_measurement_v2/results/dit_generations \
    --output_dir experiments/gap_measurement_v2/results \
    --previews \
    --extra_info "Best front-facing Blender view conditioning + 24-rotation alignment"
```
Expected: ~5 minutes. Produces final per_sample.csv, summary.md, and preview images.

- [ ] **Step 2: Review final results**

Run:
```bash
echo "=== BASELINE (random view 000, aligned metrics) ==="
cat experiments/gap_measurement_v2/results_baseline/summary.md
echo ""
echo "=== FIXED (best front-facing view, aligned metrics) ==="
cat experiments/gap_measurement_v2/results/summary.md
```
Expected: Both should show much lower CD ratios than 406x. The best-view version should show further improvement over baseline.

- [ ] **Step 3: Spot-check preview images**

Visually inspect a few preview PNGs in `experiments/gap_measurement_v2/results/previews/` to confirm:
1. Row 3 (DiT) is now pose-aligned with Rows 1-2 (GT/VAE)
2. Conditioning images (Row 0) look like reasonable front-facing views

- [ ] **Step 4: Update phase0-summary.md with corrected numbers**

Update `my-docs/phase0-summary.md` with the corrected metrics from the final summary.

- [ ] **Step 5: Final commit**

```bash
git add scripts/reeval_gap.py scripts/render_blender_cond.py my-docs/phase0-summary.md
git commit -m "fix: correct gap measurement with alignment, better conditioning, and aligned previews"
```

---

## Troubleshooting

**OBJ load fails for some models:**
- trimesh may warn about degenerate faces; use `force="mesh"` and `process=False`
- If a model is missing from dit_generations/, it was likely a DiT failure in the original run

**8-GPU launch: some ranks fail:**
- Check `log_rank*.txt` for errors
- Common: OOM on one GPU if another process is using it
- Fix: ensure `CUDA_VISIBLE_DEVICES` is set correctly per rank

**Preview rendering fails:**
- `render_normal_maps` requires CUDA; ensure one GPU is available
- If MeshWithVoxel issue reappears, `trellis_mesh_to_trimesh` + `trimesh_to_trellis_mesh` roundtrip avoids it

**Reeval script takes too long:**
- The bottleneck is preview rendering (8 normal map renders per model × 3 meshes)
- Use `--no-previews` flag (omit `--previews`) for metrics-only run (~1 min total)
