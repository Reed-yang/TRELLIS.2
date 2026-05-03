"""Reconstruct a mesh through the coart.vae encoder + decoder.

Pipeline (single GPU, no DDP):
  1. mesh (.glb / .ply / .obj)
       → corep_fast.mesh_to_param + param_to_feats
       → (cube_indices: (N,3) int, feats: (N,18) float)
  2. Build encoder/decoder via coart.vae.build, load the online ckpt for
     architecture + buffers, then overlay the EMA shadow params on top.
  3. Encode (sample_posterior=False) → Decode → predicted feats.
  4. feats_to_param → param_to_mesh → corep_to_exp5 → write .ply

Usage (run on the 119 node with an idle GPU):

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/coart_reconstruct_mesh.py \\
        --mesh /mnt/novita2/siyuan/test2/TRELLIS.2/tmp/test_mesh/underwater_plant_pack.glb \\
        --output_dir results/coart_feat18_20260423_three_branch_ws_v0 \\
        --ema_step 155000 \\
        --out_ply /tmp/underwater_plant_pack_recon.ply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
sys.path.insert(0, str(REPO))

from coart.data.stats import denormalize, load_stats, normalize
from coart.eval.normalization import corep_to_exp5_vertices
from coart.vae.build import build_models


def _overlay_ema_params(model, ema_sd):
    shadow = ema_sd["shadow"]
    live_params = list(model.parameters())
    if len(shadow) != len(live_params):
        raise RuntimeError(
            f"EMA shadow count mismatch: shadow={len(shadow)} live={len(live_params)}"
        )
    with torch.no_grad():
        for s, p in zip(shadow, live_params):
            p.data.copy_(s.data.to(p.device, dtype=p.dtype))
    return len(shadow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Input mesh path (glb/ply/obj).")
    ap.add_argument("--output_dir", required=True,
                    help="Run directory containing config.json + ckpt + ema_*.pt files.")
    ap.add_argument("--online_step", type=int, default=None,
                    help="Online ckpt step (default: same as --ema_step).")
    ap.add_argument("--ema_step", type=int, required=True)
    ap.add_argument("--ema_rate", default="0.9999")
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--num_workers", type=int, default=None,
                    help="corep_fast multiprocessing workers (default: heuristic).")
    args = ap.parse_args()

    online_step = args.online_step if args.online_step is not None else args.ema_step
    online_ckpt = os.path.join(args.output_dir, f"ckpt_step{online_step:07d}.pt")
    ema_enc = os.path.join(args.output_dir, f"ema_{args.ema_rate}_enc_step{args.ema_step:07d}.pt")
    ema_dec = os.path.join(args.output_dir, f"ema_{args.ema_rate}_dec_step{args.ema_step:07d}.pt")
    cfg_json = os.path.join(args.output_dir, "config.json")

    for p in (args.mesh, online_ckpt, ema_enc, ema_dec, cfg_json):
        if not os.path.exists(p):
            print(f"[fatal] missing {p}", file=sys.stderr)
            sys.exit(1)

    cfg_dict = json.loads(Path(cfg_json).read_text())
    cfg = argparse.Namespace(**cfg_dict)

    # ── 1. mesh → (cube_indices, feats_18) via corep_fast ──────────────────────
    from precompute_feat18 import encode_one
    print(f"[corep] encoding mesh @ res={cfg.resolution}: {args.mesh}")
    t0 = time.time()
    cube_indices_np, feats_raw_np, num_boundary = encode_one(
        args.mesh,
        resolution=cfg.resolution,
        device=torch.device("cuda"),
        num_workers=args.num_workers,
        verbose=True,
    )
    print(f"[corep] N_cubes={len(cube_indices_np):,}  num_boundary={num_boundary}  "
          f"({time.time() - t0:.1f}s)")

    # ── 2. build encoder/decoder, load online ckpt, overlay EMA ────────────────
    print(f"[build] io_arch={cfg.io_arch} latent_channels={cfg.latent_channels}")
    enc, dec = build_models(
        latent_channels=cfg.latent_channels,
        io_arch=cfg.io_arch,
    )
    print(f"[load] online ckpt: {online_ckpt}")
    ck = torch.load(online_ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"])
    dec.load_state_dict(ck["decoder"])
    print(f"[load] EMA enc overlay: {ema_enc}")
    n1 = _overlay_ema_params(enc, torch.load(ema_enc, map_location="cuda", weights_only=False))
    print(f"[load] EMA dec overlay: {ema_dec}")
    n2 = _overlay_ema_params(dec, torch.load(ema_dec, map_location="cuda", weights_only=False))
    print(f"[load] overlaid {n1} enc / {n2} dec params")

    # ── 3. normalize → encode → decode → denormalize ───────────────────────────
    stats_path = cfg.stats_path or os.path.join(cfg.data_root, "stats_global.npz")
    print(f"[stats] {stats_path}")
    mean_t, std_t = load_stats(stats_path, device=torch.device("cuda"))

    cube_indices = torch.from_numpy(cube_indices_np.astype(np.int32)).cuda()
    feats_raw = torch.from_numpy(feats_raw_np.astype(np.float32)).cuda()
    feats_n = normalize(feats_raw, mean_t, std_t)

    from trellis2.modules import sparse as sp
    N = cube_indices.shape[0]
    batch_col = torch.zeros((N, 1), dtype=torch.int32, device="cuda")
    coords_bn = torch.cat([batch_col, cube_indices], dim=1)
    x = sp.SparseTensor(feats=feats_n, coords=coords_bn)

    enc.eval(); dec.eval()
    print(f"[infer] encode + decode (bf16 autocast, sample_posterior=False)")
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = enc(x, sample_posterior=False)
        pred = dec(z)
        pred = pred[0] if isinstance(pred, tuple) else pred
    print(f"[infer] z.feats.shape={tuple(z.feats.shape)}  pred.feats.shape={tuple(pred.feats.shape)}  "
          f"({time.time() - t0:.1f}s)")

    feats_pred_raw = denormalize(
        pred.feats.float().cpu(),
        mean_t.cpu(),
        std_t.cpu(),
    ).numpy()
    # Predicted cube grid coords come from the decoder (subdivision predicted),
    # NOT from the encoder input — read them off pred.coords.
    pred_cubes = pred.coords[:, 1:].cpu().numpy().astype(np.int32)

    # ── 4. feature_to_mesh → corep_to_exp5 → save ──────────────────────────────
    from train_overfit_feat18 import feature_to_mesh
    print(f"[mesh ] feature_to_mesh on {len(pred_cubes):,} predicted cubes")
    t0 = time.time()
    mesh = feature_to_mesh(feats_pred_raw, pred_cubes, cfg.resolution)
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        print("[fatal] feature_to_mesh returned empty", file=sys.stderr)
        sys.exit(2)
    mesh.vertices = corep_to_exp5_vertices(np.asarray(mesh.vertices, dtype=np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)) or ".", exist_ok=True)
    mesh.export(args.out_ply)
    print(f"[mesh ] V={len(mesh.vertices):,}  F={len(mesh.faces):,}  "
          f"→ {args.out_ply}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
