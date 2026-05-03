"""EMA checkpoint evaluation + helmet pred mesh export.

Why: during the main training run we only logged deep_eval on the ONLINE
branch (`sample_posterior=False` over the non-EMA weights). The EMA
checkpoint (`ema_0.9999_{enc,dec}_step*.pt`) was never eval'd, yet it's
usually the preferred artifact for downstream DiT because the decay=0.9999
averaging smooths over KL-divergence spikes that the online weights suffer
in late training.

What this script does (single GPU, no DDP):
  1. Load encoder + decoder architecture via coart.vae.build.
  2. Load online ckpt (architecture + bias/buffers) then overwrite its
     trainable params with the EMA shadow values.
  3. Run `run_deep_eval` on all 8 golden assets in EXP-5 [-0.5, 0.5]^3
     space (the same pipeline the live training was hitting post-fix).
  4. Additionally dump the helmet pred mesh to `.ply` for visual inspection.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/coart_ema_eval.py \
        --output_dir results/coart_feat18_20260423_three_branch_ws_v0 \
        --online_step 155000
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
import trimesh

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
sys.path.insert(0, str(REPO))

from coart.common.dist_utils import unwrap
from coart.data.stats import denormalize, load_stats, normalize
from coart.eval.deep_eval import _one_asset
from coart.eval.metrics import sample_surface
from coart.eval.normalization import corep_to_exp5_vertices
from coart.vae.build import build_models
from coart.vae.config import VaeTrainConfig, parse_args


def _overlay_ema_params(model, ema_sd):
    """Overlay EMA shadow params onto model.parameters() in place.

    EMAModel.state_dict() returns {"decay": float, "shadow": [tensor_list]}
    where shadow is ordered to match model.parameters() at construction time.
    """
    shadow = ema_sd["shadow"]
    live_params = list(model.parameters())
    if len(shadow) != len(live_params):
        raise RuntimeError(
            f"EMA shadow count mismatch: shadow={len(shadow)} live={len(live_params)}"
        )
    with torch.no_grad():
        for s, p in zip(shadow, live_params):
            p.data.copy_(s.data.to(p.device, dtype=p.dtype))
    return len(shadow), 0


class _MockLogger:
    """Minimal logger compatible with deep_eval's scalar/image/object3d API."""
    def __init__(self):
        self.is_master = True
        self.scalars: dict[tuple[str, int], float] = {}
        self.images: dict[tuple[str, int], np.ndarray] = {}

    def scalar(self, tag, value, step):
        self.scalars[(tag, step)] = float(value)

    def image(self, tag, np_img, step):
        self.images[(tag, step)] = np_img

    def object3d(self, *args, **kwargs):
        pass  # wandb-only; silently ignore


def _export_helmet_pred_mesh(encoder, decoder, stats, cfg, out_ply_path: str):
    """Run the same encode→decode→feature_to_mesh path as deep_eval, but
    instead of computing metrics, save the predicted mesh to disk."""
    asset_list = json.loads(
        (REPO / "coart" / "eval" / "golden_assets.json").read_text()
    )
    helmet = next(a for a in asset_list if a["name"] == "helmet")
    npz_path = REPO / helmet["npz_path"]
    d = np.load(npz_path, allow_pickle=True)

    cube_indices = torch.from_numpy(d["cube_indices"].astype(np.int32)).cuda()
    feats_raw = torch.from_numpy(d["feats"].astype(np.float32)).cuda()
    feats_n = normalize(feats_raw, stats["mean"], stats["std"])

    from trellis2.modules import sparse as sp
    N = cube_indices.shape[0]
    batch_col = torch.zeros((N, 1), dtype=torch.int32, device="cuda")
    coords_bn = torch.cat([batch_col, cube_indices], dim=1)
    x = sp.SparseTensor(feats=feats_n, coords=coords_bn)

    encoder.eval(); decoder.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = encoder(x, sample_posterior=False)
        pred = decoder(z)
        pred = pred[0] if isinstance(pred, tuple) else pred

    feats_pred_raw = denormalize(
        pred.feats.float().cpu(),
        stats["mean"].cpu(),
        stats["std"].cpu(),
    ).numpy()
    cube_np = cube_indices.cpu().numpy().astype(np.int32)

    sys.path.insert(0, str(REPO))
    from train_overfit_feat18 import feature_to_mesh
    mesh = feature_to_mesh(feats_pred_raw, cube_np, cfg.resolution)
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        print(f"[helmet] feature_to_mesh returned empty")
        return

    # Apply the same CoReP→EXP-5 affine that deep_eval applies so the saved
    # mesh shares the coordinate frame with the GT point cloud.
    mesh.vertices = corep_to_exp5_vertices(
        np.asarray(mesh.vertices, dtype=np.float32)
    )
    mesh.export(out_ply_path)
    print(f"[helmet] pred mesh: V={len(mesh.vertices):,}  F={len(mesh.faces):,}  "
          f"→ {out_ply_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--online_step", type=int, required=True,
                    help="Step number of online ckpt (source of architecture + buffers)")
    ap.add_argument("--ema_step", type=int, default=None,
                    help="Step number of EMA ckpt; defaults to online_step")
    ap.add_argument("--ema_rate", default="0.9999")
    ap.add_argument("--ply_out", default=None,
                    help="Output path for helmet pred .ply (default under output_dir)")
    args = ap.parse_args()

    ema_step = args.ema_step or args.online_step
    online_ckpt = os.path.join(args.output_dir, f"ckpt_step{args.online_step:07d}.pt")
    ema_enc = os.path.join(args.output_dir, f"ema_{args.ema_rate}_enc_step{ema_step:07d}.pt")
    ema_dec = os.path.join(args.output_dir, f"ema_{args.ema_rate}_dec_step{ema_step:07d}.pt")
    cfg_json = os.path.join(args.output_dir, "config.json")

    for p in (online_ckpt, ema_enc, ema_dec, cfg_json):
        if not os.path.exists(p):
            print(f"[fatal] missing {p}", file=sys.stderr)
            sys.exit(1)

    cfg_dict = json.loads(Path(cfg_json).read_text())
    # Build a VaeTrainConfig-like namespace; we only need the fields deep_eval touches.
    cfg = argparse.Namespace(**cfg_dict)
    # These might not exist in older config.json dumps.
    if not hasattr(cfg, "n_dump_names"):
        cfg.n_dump_names = ["helmet"]
    if not hasattr(cfg, "first_deep_eval_step"):
        cfg.first_deep_eval_step = 0
    cfg.output_dir = args.output_dir

    # Build model, load online ckpt, then overlay EMA
    print(f"[build] building encoder + decoder (io_arch={cfg.io_arch}, res={cfg.resolution})")
    enc, dec = build_models(
        latent_channels=cfg.latent_channels,
        io_arch=cfg.io_arch,
    )

    print(f"[load] online ckpt: {online_ckpt}")
    ck = torch.load(online_ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"])
    dec.load_state_dict(ck["decoder"])

    print(f"[load] EMA overlay: {ema_enc}")
    ema_enc_sd = torch.load(ema_enc, map_location="cuda", weights_only=False)
    r1, s1 = _overlay_ema_params(enc, ema_enc_sd)
    print(f"  encoder: replaced {r1} params; skipped {s1}")

    print(f"[load] EMA overlay: {ema_dec}")
    ema_dec_sd = torch.load(ema_dec, map_location="cuda", weights_only=False)
    r2, s2 = _overlay_ema_params(dec, ema_dec_sd)
    print(f"  decoder: replaced {r2} params; skipped {s2}")

    # Stats (match train.py:146 resolution logic)
    stats_path = cfg.stats_path or os.path.join(cfg.data_root, "stats_global.npz")
    mean_t, std_t = load_stats(stats_path, device=torch.device("cuda"))

    # Deep-eval all golden assets
    logger = _MockLogger()
    asset_list = json.loads(
        (REPO / "coart" / "eval" / "golden_assets.json").read_text()
    )
    results = {}
    step_pretty = args.ema_step or args.online_step
    print(f"\n=== Running deep_eval on {len(asset_list)} golden assets ===")
    for a in asset_list:
        t0 = time.time()
        m = _one_asset(
            a, enc, dec, mean_t, std_t,
            step=step_pretty,
            resolution=cfg.resolution,
            n_dump_names=[],  # skip renders; handled separately
            logger=logger,
        )
        dt = time.time() - t0
        results[a["name"]] = m
        if m:
            print(f"  {a['name']:<15s} cd={m.get('cd', 0):.3e} nc={m.get('nc', 0):.4f} "
                  f"ncomp={int(m.get('n_components', 0)):>8d} f005={m.get('f005', 0):.4f} "
                  f"bdry={int(m.get('n_boundary_edges', 0)):>8d} area={m.get('surface_area', 0):.3f} "
                  f"({dt:.1f}s)")
        else:
            print(f"  {a['name']:<15s} FAILED ({dt:.1f}s)")

    # Aggregate
    mean_cd = np.mean([m["cd"] for m in results.values() if m])
    mean_nc = np.mean([m["nc"] for m in results.values() if m])
    print(f"\n=== EMA deep_eval mean ===")
    print(f"  mean/cd = {mean_cd:.3e}")
    print(f"  mean/nc = {mean_nc:.4f}")

    # Dump helmet pred mesh
    ply_out = args.ply_out or os.path.join(
        args.output_dir, f"helmet_pred_ema_step{ema_step:07d}.ply",
    )
    print(f"\n=== Exporting helmet pred mesh (EMA) ===")
    stats = {"mean": mean_t, "std": std_t}
    _export_helmet_pred_mesh(enc, dec, stats, cfg, ply_out)

    # Also save JSON summary
    summary_path = os.path.join(
        args.output_dir, f"ema_eval_step{ema_step:07d}.json",
    )
    with open(summary_path, "w") as fh:
        json.dump({
            "online_ckpt": online_ckpt,
            "ema_enc": ema_enc,
            "ema_dec": ema_dec,
            "step": step_pretty,
            "per_asset": results,
            "mean_cd": float(mean_cd),
            "mean_nc": float(mean_nc),
            "helmet_pred_ply": ply_out,
        }, fh, indent=2, default=str)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
