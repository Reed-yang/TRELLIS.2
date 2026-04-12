"""
Export Phase B pairwise comparison meshes as GLB files for remote preview.

For each PBR UID, exports:
  {uid}_gt.glb       — Ground truth mesh (normalized)
  {uid}_vae.glb      — VAE reconstruction (Phase A, if available)
  {uid}_dit.glb      — DiT best-view generation (Phase B, ICP-aligned to GT)

Usage:
    python scripts/pipeline/export_phase_b_glb.py \
        --manifest experiments/component_eval/test_set/manifest_pbr.json \
        --phase_b_csv experiments/component_eval/phase_b/results/per_sample_pbr.csv \
        --phase_a_meshes experiments/component_eval/phase_a/meshes \
        --phase_b_meshes experiments/component_eval/phase_b/meshes \
        --output_dir experiments/component_eval/phase_b/glb_preview \
        --mode all  # or 'worst 20', 'best 20', 'tier3', 'uids uid1,uid2,...'
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import argparse
import numpy as np
import pandas as pd
import trimesh
import torch

from scripts.eval.gap_measurement import load_and_normalize_mesh
from scripts.eval.eval_metrics import sample_points_and_normals, find_best_rotation_24_with_icp


def align_dit_to_gt(dit_tm, gt_tm, n_points=10000):
    """Align DiT mesh to GT using 24-rotation + ICP. Returns aligned trimesh."""
    gt_pts, _ = sample_points_and_normals(gt_tm, n_points)
    dit_pts, _ = sample_points_and_normals(dit_tm, n_points)
    transform, _ = find_best_rotation_24_with_icp(dit_pts.cuda(), gt_pts.cuda())
    # transform is numpy 4x4
    R = transform[:3, :3]
    t = transform[:3, 3]
    aligned = trimesh.Trimesh(
        vertices=dit_tm.vertices.copy() @ R.T + t,
        faces=dit_tm.faces.copy(),
        process=False,
    )
    return aligned


def export_one(uid, gt_mesh_path, dit_mesh_path, vae_mesh_path, output_dir):
    """Export GT, VAE, DiT GLBs for one UID."""
    os.makedirs(output_dir, exist_ok=True)

    # GT
    gt_tm = load_and_normalize_mesh(gt_mesh_path)
    gt_out = os.path.join(output_dir, f"{uid}_gt.glb")
    gt_tm.export(gt_out)

    # VAE (Phase A)
    if vae_mesh_path and os.path.exists(vae_mesh_path):
        vae_tm = trimesh.load(vae_mesh_path, force="mesh", process=False)
        vae_out = os.path.join(output_dir, f"{uid}_vae.glb")
        vae_tm.export(vae_out)

    # DiT (Phase B, aligned)
    if dit_mesh_path and os.path.exists(dit_mesh_path):
        dit_tm = trimesh.load(dit_mesh_path, force="mesh", process=False)
        dit_aligned = align_dit_to_gt(dit_tm, gt_tm)
        dit_out = os.path.join(output_dir, f"{uid}_dit.glb")
        dit_aligned.export(dit_out)

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase_b_csv", required=True)
    parser.add_argument("--phase_a_meshes", default="experiments/component_eval/phase_a/meshes")
    parser.add_argument("--phase_b_meshes", default="experiments/component_eval/phase_b/meshes")
    parser.add_argument("--output_dir", default="experiments/component_eval/phase_b/glb_preview")
    parser.add_argument("--mode", default="all",
                        help="'all', 'worst N', 'best N', 'tier3', 'uids uid1,uid2,...'")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    uid_to_item = {item["uid"]: item for item in manifest}

    df = pd.read_csv(args.phase_b_csv)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["cd_filled"] = pd.to_numeric(df["cd_filled"], errors="coerce")
    df = df.dropna(subset=["cd_filled"])
    df = df.sort_values("cd_filled")

    # Select UIDs based on mode
    if args.mode == "all":
        selected = df
    elif args.mode.startswith("worst"):
        n = int(args.mode.split()[1]) if " " in args.mode else 20
        selected = df.nlargest(n, "cd_filled")
    elif args.mode.startswith("best"):
        n = int(args.mode.split()[1]) if " " in args.mode else 20
        selected = df.nsmallest(n, "cd_filled")
    elif args.mode == "tier3":
        selected = df[df["tier"].astype(int) == 3]
    elif args.mode.startswith("uids"):
        uid_list = args.mode.split(maxsplit=1)[1].split(",")
        selected = df[df["uid"].isin(uid_list)]
    else:
        selected = df

    # Shard for multi-GPU
    if args.world_size > 1:
        selected = selected.reset_index(drop=True)
        start = len(selected) * args.rank // args.world_size
        end = len(selected) * (args.rank + 1) // args.world_size
        selected = selected.iloc[start:end]

    print(f"[Rank {args.rank}/{args.world_size}] Exporting {len(selected)} UIDs as GLB -> {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, (_, row) in enumerate(selected.iterrows()):
        uid = row["uid"]
        item = uid_to_item.get(uid)
        if not item:
            continue

        best_view = int(row.get("best_view_idx", 0))
        gt_path = item["mesh_path"]
        dit_path = os.path.join(args.phase_b_meshes, f"{uid}_view{best_view:03d}.obj")
        vae_path = os.path.join(args.phase_a_meshes, f"{uid}.obj")

        cd = row["cd_filled"]
        nc = row.get("nc_filled", 0)
        print(f"  [{idx+1}/{len(selected)}] {uid}  CD={cd:.2e}  NC={nc:.4f}")

        try:
            export_one(uid, gt_path, dit_path, vae_path, args.output_dir)
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"Done. GLBs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
