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
