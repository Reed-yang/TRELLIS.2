"""CLI entrypoint for VAE finetune effectiveness analysis.

Usage:
    .venv/bin/python scripts/coart_analyze_finetune.py \\
        --ckpt_dir results/coart_feat18_20260423_three_branch_ws_v0 \\
        --step 155000 --use_ema --n_val 200 --ablate_assets golden

Output: <ckpt_dir>/analysis_step<STEP>/{report.md, tables/*.csv, figures/*.png}.
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

from coart.analysis.activation_stats import (
    analyze_activation_stats,
    plot_pred_vs_target_scatter,
    plot_zero_rate_bars,
    run_val_forward_pass,
)
from coart.analysis.head_ablation import (
    analyze_head_ablation,
    plot_ablation_delta_per_asset,
)
from coart.analysis.report import write_report
from coart.analysis.weight_drift import (
    TARGET_LINEARS,
    analyze_weight_drift,
    extract_io_linear_weights,
    plot_perch_norm_hist,
    plot_sv_spectrum,
    reconstruct_step0_state_dicts,
    _load_ckpt_state_dicts,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--use_ema", action="store_true", default=True)
    grp.add_argument("--use_online", action="store_false", dest="use_ema")
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--ablate_assets", choices=["golden", "all"], default="golden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_drift", action="store_true")
    p.add_argument("--skip_activation", action="store_true")
    p.add_argument("--skip_ablation", action="store_true")
    return p.parse_args()


def _load_models(ckpt_dir: str, step: int, use_ema: bool, device):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    from coart.vae.build import build_models, load_pretrained_into

    enc, dec = build_models(
        latent_channels=cfg.get("latent_channels", 32),
        device=device,
        io_arch=cfg.get("io_arch", "three_branch"),
    )
    if cfg.get("from_pretrained", True):
        load_pretrained_into(
            enc, dec,
            enc_path=cfg["enc_pretrained"],
            dec_path=cfg["dec_pretrained"],
            io_arch=cfg.get("io_arch", "three_branch"),
            warmstart_io=cfg.get("warmstart_io", True),
            verbose=False,
        )
    enc_sd, dec_sd = _load_ckpt_state_dicts(
        ckpt_dir, use_ema=use_ema, ckpt_dir=ckpt_dir, step=step,
    )
    enc.load_state_dict(enc_sd, strict=True)
    dec.load_state_dict(dec_sd, strict=True)
    enc.to(device).eval()
    dec.to(device).eval()
    return enc, dec, cfg


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.ckpt_dir) / f"analysis_step{args.step}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] output -> {out_dir}", flush=True)

    # ---- Section A: weight drift ----------------------------------------
    if not args.skip_drift:
        t0 = time.time()
        drift_df = analyze_weight_drift(
            ckpt_dir=args.ckpt_dir, step=args.step,
            use_ema=args.use_ema, seed=args.seed,
        )
        drift_df.to_csv(tab_dir / "weight_drift_summary.csv", index=False)
        # Per-Linear figures using only the (now, step0) pair.
        cfg_path = os.path.join(args.ckpt_dir, "config.json")
        enc_now, dec_now = _load_ckpt_state_dicts(
            args.ckpt_dir, use_ema=args.use_ema,
            ckpt_dir=args.ckpt_dir, step=args.step,
        )
        enc0, dec0 = reconstruct_step0_state_dicts(cfg_path, seed=args.seed)
        for side, branch in TARGET_LINEARS:
            sd_now = enc_now if side == "encoder" else dec_now
            sd_ref = enc0 if side == "encoder" else dec0
            Wnow, _ = extract_io_linear_weights(sd_now, side=side)[branch]
            Wref, _ = extract_io_linear_weights(sd_ref, side=side)[branch]
            title = f"{side}.{branch}"
            plot_sv_spectrum(
                Wnow, Wref, title=title,
                out_path=str(fig_dir / f"sv_spectrum_{branch}.png"),
            )
            plot_perch_norm_hist(
                Wnow, Wref, title=title,
                out_path=str(fig_dir / f"perch_norm_hist_{branch}.png"),
            )
        print(f"[analyze] section A weight drift done in {time.time()-t0:.1f}s")
    else:
        drift_df = None

    # ---- Section B: activation stats ------------------------------------
    if not args.skip_activation:
        t0 = time.time()
        enc, dec, cfg = _load_models(args.ckpt_dir, args.step, args.use_ema, device)
        from coart.data.feat18_dataset import Feat18Dataset
        from coart.data.stats import load_stats

        stats_path = cfg.get("stats_path") or os.path.join(
            cfg["data_root"], "stats_global.npz"
        )
        stats_mean, stats_std = load_stats(stats_path, device, verbose=False)

        val_ds = Feat18Dataset(
            data_dir=cfg.get("data_dir") or os.path.join(cfg["data_root"], "data"),
            resolution=cfg["resolution"],
            max_translate=0,           # no augmentation in analysis
            augment=False,
            precompute_voxel_counts=False,
            max_voxels=cfg.get("max_voxels", 0),
            val_split_mod=cfg["val_split_mod"],
            split="val",
        )
        captures = run_val_forward_pass(
            enc, dec, val_ds,
            stats_mean=stats_mean, stats_std=stats_std,
            n_items=args.n_val, device=device, seed=args.seed,
        )
        dfs = analyze_activation_stats(captures, stats_mean, stats_std)
        dfs["contributions"].to_csv(
            tab_dir / "activation_contributions.csv", index=False,
        )
        dfs["per_channel"].to_csv(
            tab_dir / "per_channel_pred_vs_target.csv", index=False,
        )
        plot_zero_rate_bars(
            dfs["per_channel"], out_path=str(fig_dir / "zero_rate_bars.png"),
        )
        from coart.data.stats import denormalize
        pred_raw = denormalize(captures["pred_norm"], stats_mean.cpu(), stats_std.cpu())
        target_raw = denormalize(captures["target_norm"], stats_mean.cpu(), stats_std.cpu())
        plot_pred_vs_target_scatter(
            pred_raw, target_raw,
            channels=[3, 5, 6, 7, 12, 13],
            out_dir=str(fig_dir),
            seed=args.seed,
        )
        per_channel_df = dfs["per_channel"]
        contrib_df = dfs["contributions"]
        print(f"[analyze] section B activation stats done in {time.time()-t0:.1f}s")
    else:
        per_channel_df = None
        contrib_df = None

    # ---- Section C: head ablation ---------------------------------------
    if not args.skip_ablation:
        t0 = time.time()
        if "enc" not in locals():
            enc, dec, cfg = _load_models(args.ckpt_dir, args.step, args.use_ema, device)
            from coart.data.stats import load_stats
            stats_path = cfg.get("stats_path") or os.path.join(
                cfg["data_root"], "stats_global.npz"
            )
            stats_mean, stats_std = load_stats(stats_path, device, verbose=False)
        manifest_path = REPO / "coart" / "eval" / "golden_assets.json"
        with open(manifest_path) as fh:
            assets = json.load(fh)
        if args.ablate_assets == "golden":
            asset_list = assets
        else:
            asset_list = assets  # spec keeps "all" room; for now == golden
        ablation_df = analyze_head_ablation(
            enc, dec, stats_mean, stats_std,
            asset_list=asset_list, resolution=cfg["resolution"], device=device,
        )
        ablation_df.to_csv(tab_dir / "ablation_metrics.csv", index=False)
        plot_ablation_delta_per_asset(
            ablation_df, metric="cd",
            out_path=str(fig_dir / "ablation_delta_per_asset.png"),
        )
        print(f"[analyze] section C head ablation done in {time.time()-t0:.1f}s")
    else:
        ablation_df = None

    # ---- report.md ------------------------------------------------------
    import pandas as pd
    write_report(
        drift_df=drift_df if drift_df is not None else pd.DataFrame(),
        per_channel_df=per_channel_df if per_channel_df is not None else pd.DataFrame(),
        contrib_df=contrib_df if contrib_df is not None else pd.DataFrame(),
        ablation_df=ablation_df if ablation_df is not None else pd.DataFrame(),
        figures_dir="figures",
        out_path=str(out_dir / "report.md"),
        ckpt_dir=args.ckpt_dir, step=args.step,
        use_ema=args.use_ema, n_val=args.n_val,
    )
    print(f"[analyze] report -> {out_dir}/report.md")


if __name__ == "__main__":
    main()
