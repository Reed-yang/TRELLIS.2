"""
Sanity-check `precompute_feat18.py` outputs by reconstructing meshes from the
precomputed (cube_indices, feats_18) files and comparing them to the ground
truth GLB files.

For each of 5 randomly sampled, successfully-precomputed sha256 ids under the
dataset's feat18_<resolution>/data/ directory, this script:

    1. Loads cube_indices, feats, num_boundary, resolution from <sha>.npz.
    2. Runs feats_to_param (from train_overfit_feat18.py) → param_to_mesh
       (from corep_fast.pipeline) → reconstructed trimesh.
    3. Loads the original GLB from metadata_first1k.csv local_path as GT.
    4. Saves GT and reconstructed meshes side-by-side under
       tmp/test_preprocess/.

Usage:
    python test_preprocess.py
    python test_preprocess.py --num_samples 10 --seed 123
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import trimesh

from corep_fast.pipeline import param_to_mesh
from train_overfit_feat18 import feats_to_param


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset_root",
        default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab",
    )
    p.add_argument("--metadata_csv", default="metadata_first1k.csv")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument(
        "--feat_dir",
        default=None,
        help="Directory containing precomputed .npz files (default: <dataset_root>/feat18_<resolution>/data).",
    )
    p.add_argument(
        "--out_dir",
        default="tmp/test_preprocess",
        help="Where to save the reconstructed and GT meshes.",
    )
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=16)
    return p.parse_args()


def _export_gt(gt_path: str, out_path: str) -> tuple[int, int]:
    """Load a GLB/OBJ/PLY as a single mesh and export to PLY.

    Returns (num_vertices, num_faces).
    """
    scene_or_mesh = trimesh.load(gt_path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.geometry.values()]
        )
    mesh.export(out_path)
    return int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])


def main():
    args = parse_args()

    feat_dir = args.feat_dir or os.path.join(
        args.dataset_root, f"feat18_{args.resolution}", "data"
    )
    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(f"feat_dir not found: {feat_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    metadata = pd.read_csv(os.path.join(args.dataset_root, args.metadata_csv))
    sha_to_local = dict(zip(metadata["sha256"], metadata["local_path"]))

    all_files = [
        f for f in os.listdir(feat_dir)
        if f.endswith(".npz") and not f.endswith(".failed")
    ]
    if len(all_files) == 0:
        raise RuntimeError(f"No precomputed .npz files found in {feat_dir}")

    rng = random.Random(args.seed)
    sampled = rng.sample(all_files, min(args.num_samples, len(all_files)))
    print(f"[info] sampled {len(sampled)} / {len(all_files)} from {feat_dir}")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    for i, fname in enumerate(sampled):
        sha = fname[: -len(".npz")]
        npz_path = os.path.join(feat_dir, fname)
        print(f"\n[{i + 1}/{len(sampled)}] {sha}")

        d = np.load(npz_path)
        cube_indices = d["cube_indices"].astype(np.int32)
        feats = d["feats"].astype(np.float32)
        num_boundary = d["num_boundary"].astype(np.int32)
        resolution = int(d["resolution"])
        print(
            f"    cube_indices={cube_indices.shape} feats={feats.shape} "
            f"resolution={resolution}"
        )

        # ── reconstruction ──
        param = feats_to_param(feats, cube_indices, resolution, num_boundary=num_boundary)
        vertices, faces = param_to_mesh(
            param,
            device=device,
            merge_decimals=5,
            num_workers=args.num_workers,
        )
        v_np = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else np.asarray(vertices)
        f_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)

        recon_path = os.path.join(args.out_dir, f"{sha}_recon.ply")
        if f_np.shape[0] == 0:
            print(f"    [warn] reconstruction produced no faces; skipping save")
        else:
            recon_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
            recon_mesh.export(recon_path)
            print(f"    recon: V={v_np.shape[0]} F={f_np.shape[0]} → {recon_path}")

        # ── ground truth ──
        rel = sha_to_local.get(sha)
        if rel is None:
            print(f"    [warn] sha not in metadata, cannot save GT")
            continue
        gt_src = os.path.join(args.dataset_root, rel)
        if not os.path.exists(gt_src):
            print(f"    [warn] GT source not found: {gt_src}")
            continue
        gt_out = os.path.join(args.out_dir, f"{sha}_gt.ply")
        try:
            gv, gf = _export_gt(gt_src, gt_out)
            print(f"    gt:    V={gv} F={gf} → {gt_out}")
        except Exception as e:
            print(f"    [warn] failed to export GT ({type(e).__name__}: {e})")

    print(f"\n[done] outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
