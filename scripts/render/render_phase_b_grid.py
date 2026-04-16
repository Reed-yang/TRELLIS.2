"""
Render a grid comparison image for Phase B hard cases.
Each cell: conditioning image | GT normal (1 view) | DiT normal (1 view, aligned)
Arranged in a grid with UID + CD/NC labels.

Usage:
    python scripts/render/render_phase_b_grid.py \
        --manifest experiments/component_eval/test_set/manifest_pbr.json \
        --phase_b_csv experiments/component_eval/phase_b/results/per_sample_pbr.csv \
        --meshes_dir experiments/component_eval/phase_b/meshes \
        --output experiments/component_eval/phase_b/previews/hard_cases_grid.png \
        --n_worst 20 --cols 4
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


def render_one_normal(tm_mesh, resolution=256):
    """Render a single front-view normal map."""
    trellis_mesh = trimesh_to_trellis_mesh(tm_mesh)
    nmaps = render_normal_maps(trellis_mesh, nviews=1, resolution=resolution)
    img = (nmaps[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase_b_csv", required=True)
    parser.add_argument("--meshes_dir", default="experiments/component_eval/phase_b/meshes")
    parser.add_argument("--output", default="experiments/component_eval/phase_b/previews/hard_cases_grid.png")
    parser.add_argument("--n_worst", type=int, default=20)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--cell_size", type=int, default=256)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    uid_to_item = {item["uid"]: item for item in manifest}

    df = pd.read_csv(args.phase_b_csv)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["cd_filled"] = pd.to_numeric(df["cd_filled"], errors="coerce")
    df["nc_filled"] = pd.to_numeric(df["nc_filled"], errors="coerce")
    df = df.dropna(subset=["cd_filled"])
    worst = df.nlargest(args.n_worst, "cd_filled")

    S = args.cell_size
    COLS = args.cols
    ROWS = (len(worst) + COLS - 1) // COLS
    # Each cell: 3 images side by side (cond | GT | DiT) + label bar
    LABEL_H = 40
    CELL_W = S * 3
    CELL_H = S + LABEL_H

    canvas = np.full((ROWS * CELL_H, COLS * CELL_W, 3), 40, dtype=np.uint8)
    canvas_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (_, row) in enumerate(worst.iterrows()):
        uid = row["uid"]
        best_view = int(row.get("best_view_idx", 0))
        cd = float(row["cd_filled"])
        nc = float(row["nc_filled"]) if pd.notna(row.get("nc_filled")) else 0

        col = idx % COLS
        r = idx // COLS
        x0 = col * CELL_W
        y0 = r * CELL_H

        item = uid_to_item.get(uid)
        if not item:
            continue

        print(f"  [{idx+1}/{len(worst)}] {uid}")

        # Label
        label = f"{uid}  CD={cd:.1e} NC={nc:.3f}"
        draw.text((x0 + 4, y0 + 2), label, fill=(255, 255, 255), font=font)

        img_y = y0 + LABEL_H

        # 1. Conditioning image
        renders_dir = item.get("renders_dir", "")
        cond_path = os.path.join(renders_dir, uid, f"{best_view:03d}.png") if renders_dir else None
        if cond_path and os.path.exists(cond_path):
            cond = np.array(Image.open(cond_path).convert("RGB").resize((S, S)))
        else:
            cond = np.full((S, S, 3), 80, dtype=np.uint8)
        canvas_img.paste(Image.fromarray(cond), (x0, img_y))

        # 2. GT normal map
        try:
            gt_tm = load_and_normalize_mesh(item["mesh_path"])
            gt_nmap = render_one_normal(gt_tm, S)
            canvas_img.paste(Image.fromarray(gt_nmap), (x0 + S, img_y))
        except Exception as e:
            print(f"    GT failed: {e}")

        # 3. DiT normal map (aligned)
        dit_mesh_path = os.path.join(args.meshes_dir, f"{uid}_view{best_view:03d}.obj")
        if os.path.exists(dit_mesh_path):
            try:
                dit_tm = trimesh.load(dit_mesh_path, force="mesh", process=False)
                gt_pts, _ = sample_points_and_normals(gt_tm, 10000)
                dit_pts, _ = sample_points_and_normals(dit_tm, 10000)
                transform, _ = find_best_rotation_24_with_icp(dit_pts.cuda(), gt_pts.cuda())
                R, t = transform[:3, :3], transform[:3, 3]
                dit_tm.vertices = dit_tm.vertices @ R.T + t
                dit_nmap = render_one_normal(dit_tm, S)
                canvas_img.paste(Image.fromarray(dit_nmap), (x0 + S * 2, img_y))
            except Exception as e:
                print(f"    DiT failed: {e}")

        torch.cuda.empty_cache()

    # Add column headers
    header = Image.new("RGB", (COLS * CELL_W, 24), (20, 20, 20))
    hdraw = ImageDraw.Draw(header)
    for c in range(COLS):
        cx = c * CELL_W
        hdraw.text((cx + S // 2 - 15, 4), "Cond", fill=(200, 200, 100), font=font)
        hdraw.text((cx + S + S // 2 - 8, 4), "GT", fill=(100, 255, 100), font=font)
        hdraw.text((cx + S * 2 + S // 2 - 12, 4), "DiT", fill=(100, 100, 255), font=font)

    final = Image.new("RGB", (COLS * CELL_W, 24 + ROWS * CELL_H))
    final.paste(header, (0, 0))
    final.paste(canvas_img, (0, 24))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final.save(args.output)
    print(f"Saved: {args.output} ({final.size[0]}x{final.size[1]})")


if __name__ == "__main__":
    main()
