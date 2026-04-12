"""
Render Phase B preview images: conditioning image + GT vs DiT normal maps.

Layout (3 rows × 4 columns):
  Row 0: conditioning image (best view) | CD / NC / PSNR text
  Row 1: GT normal maps (4 views)
  Row 2: DiT best-view normal maps (4 views, rotation-aligned)

Usage:
    python scripts/render/render_phase_b_previews.py \
        --manifest experiments/component_eval/test_set/manifest_pbr.json \
        --phase_b_csv experiments/component_eval/phase_b/results/per_sample_pbr.csv \
        --meshes_dir experiments/component_eval/phase_b/meshes \
        --output_dir experiments/component_eval/phase_b/previews \
        --mode best_worst  # or 'all' for all 590, 'sample N' for N random
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import argparse
import numpy as np
import pandas as pd
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

from scripts.eval.eval_metrics import (
    sample_points_and_normals,
    find_best_rotation_24_with_icp,
    render_normal_maps,
)
from scripts.eval.gap_measurement import load_and_normalize_mesh


def trimesh_to_trellis_mesh(tm_mesh):
    from trellis2.representations import Mesh as TrellisMesh
    return TrellisMesh(
        vertices=torch.from_numpy(tm_mesh.vertices.copy()).float().cuda(),
        faces=torch.from_numpy(tm_mesh.faces.copy()).int().cuda(),
    )


def apply_rotation_to_trimesh(tm_mesh, R):
    R_np = R.cpu().numpy() if torch.is_tensor(R) else R
    tm_mesh.vertices = tm_mesh.vertices @ R_np.T
    return tm_mesh


def render_preview(gt_trimesh, dit_trimesh, cond_image_path, output_path,
                   uid="", cd=None, nc=None, psnr=None, best_view_idx=None):
    """
    Render a 3-row preview image.
    """
    nviews = 4
    resolution = 512

    rows = []

    # Row 0: conditioning image + metrics text
    if cond_image_path and os.path.exists(cond_image_path):
        cond = np.array(Image.open(cond_image_path).convert("RGB").resize((resolution, resolution)))
    else:
        cond = np.full((resolution, resolution, 3), 128, dtype=np.uint8)

    row0 = np.full((resolution, resolution * nviews, 3), 32, dtype=np.uint8)
    row0[:, :resolution] = cond

    # Draw metrics text on the right side
    row0_img = Image.fromarray(row0)
    draw = ImageDraw.Draw(row0_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
        font_title = font

    text_x = resolution + 40
    text_y = 40
    draw.text((text_x, text_y), uid, fill=(255, 255, 255), font=font_title)
    text_y += 60
    if best_view_idx is not None:
        draw.text((text_x, text_y), f"Best view: {best_view_idx}", fill=(200, 200, 200), font=font)
        text_y += 45
    if cd is not None:
        draw.text((text_x, text_y), f"CD:   {cd:.2e}", fill=(200, 200, 200), font=font)
        text_y += 45
    if nc is not None:
        draw.text((text_x, text_y), f"NC:   {nc:.4f}", fill=(200, 200, 200), font=font)
        text_y += 45
    if psnr is not None:
        draw.text((text_x, text_y), f"PSNR: {psnr:.1f} dB", fill=(200, 200, 200), font=font)
        text_y += 45

    # Row labels
    draw.text((text_x, resolution - 80), "Row 1: GT", fill=(150, 255, 150), font=font)
    draw.text((text_x, resolution - 40), "Row 2: DiT (aligned)", fill=(150, 150, 255), font=font)

    rows.append(np.array(row0_img))

    # Row 1: GT normal maps
    try:
        gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
        gt_nmaps = render_normal_maps(gt_trellis, nviews=nviews, resolution=resolution)
        row1_imgs = [(nm.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for nm in gt_nmaps]
        rows.append(np.concatenate(row1_imgs, axis=1))
    except Exception as e:
        print(f"    GT render failed: {e}")
        rows.append(np.full((resolution, resolution * nviews, 3), 160, dtype=np.uint8))

    # Row 2: DiT (rotation-aligned via ICP)
    try:
        # Align DiT mesh to GT
        gt_pts, _ = sample_points_and_normals(gt_trimesh, 10000)
        dit_pts, _ = sample_points_and_normals(dit_trimesh, 10000)
        gt_pts, dit_pts = gt_pts.cuda(), dit_pts.cuda()
        transform, _ = find_best_rotation_24_with_icp(dit_pts, gt_pts)

        # Apply transform (numpy 4x4) to dit mesh
        R = transform[:3, :3]
        t = transform[:3, 3]
        dit_aligned = trimesh.Trimesh(
            vertices=dit_trimesh.vertices.copy(), faces=dit_trimesh.faces.copy(), process=False
        )
        dit_aligned.vertices = dit_aligned.vertices @ R.T + t

        dit_trellis = trimesh_to_trellis_mesh(dit_aligned)
        dit_nmaps = render_normal_maps(dit_trellis, nviews=nviews, resolution=resolution)
        row2_imgs = [(nm.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for nm in dit_nmaps]
        rows.append(np.concatenate(row2_imgs, axis=1))
    except Exception as e:
        print(f"    DiT render failed: {e}")
        rows.append(np.full((resolution, resolution * nviews, 3), 160, dtype=np.uint8))

    # Composite
    composite = np.concatenate(rows, axis=0)
    Image.fromarray(composite).save(output_path)
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Render Phase B preview images")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase_b_csv", required=True)
    parser.add_argument("--meshes_dir", default="experiments/component_eval/phase_b/meshes")
    parser.add_argument("--output_dir", default="experiments/component_eval/phase_b/previews")
    parser.add_argument("--mode", default="best_worst",
                        help="'best_worst' (top/bottom 10), 'all', 'sample N', or 'uids'")
    parser.add_argument("--uids_json", default=None,
                        help="JSON file with list of UIDs to render (used with --mode uids)")
    parser.add_argument("--n_best", type=int, default=10)
    parser.add_argument("--n_worst", type=int, default=10)
    parser.add_argument("--n_median", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    uid_to_item = {item["uid"]: item for item in manifest}

    # Load Phase B results
    df = pd.read_csv(args.phase_b_csv)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["cd_filled"] = pd.to_numeric(df["cd_filled"], errors="coerce")
    df = df.dropna(subset=["cd_filled"])
    df = df.sort_values("cd_filled")

    # Select samples based on mode
    if args.mode == "uids" and args.uids_json:
        with open(args.uids_json) as f:
            uid_list = set(json.load(f))
        selected = df[df["uid"].isin(uid_list)].sort_values("cd_filled", ascending=False)
    elif args.mode == "all":
        selected = df
    elif args.mode == "best_worst":
        best = df.head(args.n_best)
        worst = df.tail(args.n_worst)
        n_med = args.n_median
        mid = len(df) // 2
        median_region = df.iloc[mid - n_med // 2: mid + n_med // 2 + 1].head(n_med)
        selected = pd.concat([best, median_region, worst]).drop_duplicates(subset=["uid"])
    elif args.mode.startswith("sample"):
        n = int(args.mode.split()[1]) if " " in args.mode else 30
        selected = df.sample(n=min(n, len(df)), random_state=42)
    else:
        selected = df.head(20)

    print(f"Rendering {len(selected)} previews -> {args.output_dir}")

    for idx, (_, row) in enumerate(selected.iterrows()):
        uid = row["uid"]
        item = uid_to_item.get(uid)
        if item is None:
            print(f"  [{idx+1}] {uid} not in manifest, skipping")
            continue

        best_view = int(row.get("best_view_idx", 0))
        cd = float(row.get("cd_filled", 0))
        nc = float(row.get("nc_filled", 0)) if pd.notna(row.get("nc_filled")) else None
        psnr_val = float(row.get("psnr", 0)) if pd.notna(row.get("psnr")) else None

        # Paths
        mesh_path = item["mesh_path"]
        renders_dir = item.get("renders_dir", "")
        cond_path = os.path.join(renders_dir, uid, f"{best_view:03d}.png") if renders_dir else None
        dit_mesh_path = os.path.join(args.meshes_dir, f"{uid}_view{best_view:03d}.obj")

        if not os.path.exists(dit_mesh_path):
            print(f"  [{idx+1}] {uid} DiT mesh not found: {dit_mesh_path}")
            continue

        output_path = os.path.join(args.output_dir, f"{uid}.png")
        nc_str = f"{nc:.4f}" if nc else "N/A"
        print(f"  [{idx+1}/{len(selected)}] {uid}  CD={cd:.2e}  NC={nc_str}")

        try:
            gt_tm = load_and_normalize_mesh(mesh_path)
            dit_tm = trimesh.load(dit_mesh_path, force="mesh", process=False)

            render_preview(
                gt_tm, dit_tm, cond_path, output_path,
                uid=uid, cd=cd, nc=nc, psnr=psnr_val, best_view_idx=best_view,
            )
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"Done. Previews saved to {args.output_dir}")


if __name__ == "__main__":
    main()
