"""CLI entrypoint for VAE finetune effectiveness analysis.

Usage (single-process):
    .venv/bin/python scripts/coart_analyze_finetune.py \\
        --ckpt_dir results/coart_feat18_20260423_three_branch_ws_v0 \\
        --step 155000 --use_ema --n_val 200 --ablate_assets golden

Usage (8-GPU sharded sweep — invoke once per rank, then once for merge):
    # Per rank (run 8 in parallel, one per GPU):
    CUDA_VISIBLE_DEVICES=$R .venv/bin/python scripts/coart_analyze_finetune.py \\
        --ckpt_dir <DIR> --step <S> --use_ema --n_val 200 \\
        --rank $R --world_size 8 --mode shard
    # After all ranks done, run merge once:
    .venv/bin/python scripts/coart_analyze_finetune.py \\
        --ckpt_dir <DIR> --step <S> --use_ema --n_val 200 --mode merge
    # Wrapper script: scripts/run_analyze_finetune_8gpu.sh

Output: <ckpt_dir>/analysis_step<STEP>/{report.md, tables/*.csv, figures/*.png}.
Per-rank intermediates land under <out_dir>/shards/.
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


# Capture keys produced by run_val_forward_pass and consumed by
# analyze_activation_stats. Defined here so shard/merge agree.
_CAPTURE_KEYS = (
    "pred_norm", "target_norm",
    "p1_contrib", "p2_contrib", "ef_contrib",
    "is_p2_zero", "has_ef_signal",
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
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument(
        "--mode", choices=["all", "shard", "merge"], default="all",
        help="all=single-process; shard=this rank's slice only; merge=consolidate shards.",
    )
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


def _load_dataset_and_stats(cfg, device):
    from coart.data.feat18_dataset import Feat18Dataset
    from coart.data.stats import load_stats

    stats_path = cfg.get("stats_path") or os.path.join(
        cfg["data_root"], "stats_global.npz"
    )
    stats_mean, stats_std = load_stats(stats_path, device, verbose=False)
    val_ds = Feat18Dataset(
        data_dir=cfg.get("data_dir") or os.path.join(cfg["data_root"], "data"),
        resolution=cfg["resolution"],
        max_translate=0,
        augment=False,
        precompute_voxel_counts=False,
        max_voxels=cfg.get("max_voxels", 0),
        val_split_mod=cfg["val_split_mod"],
        split="val",
    )
    return val_ds, stats_mean, stats_std


def _global_val_indices(n_val: int, n_total: int, seed: int) -> np.ndarray:
    """Deterministic permutation-then-truncate selection, identical across ranks."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_total)[: min(n_val, n_total)]


def _section_a_drift(args, out_dir: Path, fig_dir: Path, tab_dir: Path):
    """Weight drift CSV + per-Linear figures. CPU-only. Run once on rank 0."""
    t0 = time.time()
    drift_df = analyze_weight_drift(
        ckpt_dir=args.ckpt_dir, step=args.step,
        use_ema=args.use_ema, seed=args.seed,
    )
    drift_df.to_csv(tab_dir / "weight_drift_summary.csv", index=False)
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
    print(f"[analyze] section A weight drift done in {time.time()-t0:.1f}s",
          flush=True)
    return drift_df


def _section_b_shard(args, shard_dir: Path, device) -> str:
    """Compute this rank's slice of activation captures; save shard .pt."""
    t0 = time.time()
    enc, dec, cfg = _load_models(args.ckpt_dir, args.step, args.use_ema, device)
    val_ds, stats_mean, stats_std = _load_dataset_and_stats(cfg, device)

    all_idx = _global_val_indices(args.n_val, len(val_ds), args.seed)
    my_idx = all_idx[args.rank :: args.world_size]
    captures = run_val_forward_pass(
        enc, dec, val_ds,
        stats_mean=stats_mean, stats_std=stats_std,
        n_items=len(my_idx), device=device, seed=args.seed,
        indices=my_idx,
    )
    cpu_captures = {k: captures[k] for k in _CAPTURE_KEYS}
    out_path = shard_dir / f"rank{args.rank}_b_captures.pt"
    torch.save(cpu_captures, out_path)
    print(
        f"[analyze][rank{args.rank}] section B shard ({len(my_idx)} items) "
        f"done in {time.time()-t0:.1f}s -> {out_path}",
        flush=True,
    )
    # Free GPU memory before C section.
    del enc, dec, captures
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(out_path)


def _section_c_shard(args, shard_dir: Path, device) -> str:
    """Compute this rank's slice of head ablation; save shard CSV."""
    t0 = time.time()
    enc, dec, cfg = _load_models(args.ckpt_dir, args.step, args.use_ema, device)
    _, stats_mean, stats_std = _load_dataset_and_stats(cfg, device)

    manifest_path = REPO / "coart" / "eval" / "golden_assets.json"
    with open(manifest_path) as fh:
        assets = json.load(fh)
    asset_list_full = assets  # "golden" and "all" both point here today
    my_assets = asset_list_full[args.rank :: args.world_size]

    df = analyze_head_ablation(
        enc, dec, stats_mean, stats_std,
        asset_list=my_assets, resolution=cfg["resolution"], device=device,
    )
    out_path = shard_dir / f"rank{args.rank}_c_ablation.csv"
    df.to_csv(out_path, index=False)
    print(
        f"[analyze][rank{args.rank}] section C shard ({len(my_assets)} assets) "
        f"done in {time.time()-t0:.1f}s -> {out_path}",
        flush=True,
    )
    return str(out_path)


def _merge_b_captures(shard_dir: Path, world_size: int):
    """Concatenate rank{r}_b_captures.pt files along voxel dim."""
    files = [
        shard_dir / f"rank{r}_b_captures.pt" for r in range(world_size)
    ]
    blobs = []
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(
                f"missing shard {fp}; cannot merge until all ranks finish"
            )
        blobs.append(torch.load(fp, map_location="cpu", weights_only=True))
    merged = {}
    for k in _CAPTURE_KEYS:
        merged[k] = torch.cat([b[k] for b in blobs], dim=0)
    return merged


def _merge_c_ablation(shard_dir: Path, world_size: int):
    import pandas as pd
    files = [
        shard_dir / f"rank{r}_c_ablation.csv" for r in range(world_size)
    ]
    frames = []
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(
                f"missing shard {fp}; cannot merge until all ranks finish"
            )
        frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True)


def _finalize_b(captures, args, fig_dir: Path, tab_dir: Path, device):
    """Run analyze_activation_stats on merged captures; emit CSVs + plots."""
    from coart.data.stats import denormalize, load_stats

    cfg_path = os.path.join(args.ckpt_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    stats_path = cfg.get("stats_path") or os.path.join(
        cfg["data_root"], "stats_global.npz"
    )
    stats_mean, stats_std = load_stats(stats_path, device, verbose=False)

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
    pred_raw = denormalize(captures["pred_norm"], stats_mean.cpu(), stats_std.cpu())
    target_raw = denormalize(captures["target_norm"], stats_mean.cpu(), stats_std.cpu())
    plot_pred_vs_target_scatter(
        pred_raw, target_raw,
        channels=[3, 5, 6, 7, 12, 13],
        out_dir=str(fig_dir),
        seed=args.seed,
    )
    return dfs["per_channel"], dfs["contributions"]


def _finalize_c(ablation_df, fig_dir: Path, tab_dir: Path):
    ablation_df.to_csv(tab_dir / "ablation_metrics.csv", index=False)
    plot_ablation_delta_per_asset(
        ablation_df, metric="cd",
        out_path=str(fig_dir / "ablation_delta_per_asset.png"),
    )


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.ckpt_dir) / f"analysis_step{args.step}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    shard_dir = out_dir / "shards"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[analyze][rank{args.rank}/{args.world_size}][mode={args.mode}] "
        f"output -> {out_dir}",
        flush=True,
    )

    # ── Single-process path: do everything inline (legacy "all" mode). ────
    if args.mode == "all":
        drift_df = _section_a_drift(args, out_dir, fig_dir, tab_dir) \
            if not args.skip_drift else None

        per_channel_df = contrib_df = None
        if not args.skip_activation:
            t0 = time.time()
            enc, dec, cfg = _load_models(args.ckpt_dir, args.step, args.use_ema, device)
            val_ds, stats_mean, stats_std = _load_dataset_and_stats(cfg, device)
            all_idx = _global_val_indices(args.n_val, len(val_ds), args.seed)
            captures = run_val_forward_pass(
                enc, dec, val_ds,
                stats_mean=stats_mean, stats_std=stats_std,
                n_items=len(all_idx), device=device, seed=args.seed,
                indices=all_idx,
            )
            per_channel_df, contrib_df = _finalize_b(
                captures, args, fig_dir, tab_dir, device,
            )
            print(f"[analyze] section B activation stats done in {time.time()-t0:.1f}s",
                  flush=True)

        ablation_df = None
        if not args.skip_ablation:
            t0 = time.time()
            if "enc" not in locals():
                enc, dec, cfg = _load_models(args.ckpt_dir, args.step, args.use_ema, device)
                _, stats_mean, stats_std = _load_dataset_and_stats(cfg, device)
            manifest_path = REPO / "coart" / "eval" / "golden_assets.json"
            with open(manifest_path) as fh:
                assets = json.load(fh)
            ablation_df = analyze_head_ablation(
                enc, dec, stats_mean, stats_std,
                asset_list=assets, resolution=cfg["resolution"], device=device,
            )
            _finalize_c(ablation_df, fig_dir, tab_dir)
            print(f"[analyze] section C head ablation done in {time.time()-t0:.1f}s",
                  flush=True)

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
        print(f"[analyze] report -> {out_dir}/report.md", flush=True)
        return

    # ── Sharded path: each rank writes its slice; merge step builds report.
    if args.mode == "shard":
        # Section A only on rank 0 (CPU-only, ~50s; cheap and avoids dup work).
        if args.rank == 0 and not args.skip_drift:
            _section_a_drift(args, out_dir, fig_dir, tab_dir)
        if not args.skip_activation:
            _section_b_shard(args, shard_dir, device)
        if not args.skip_ablation:
            _section_c_shard(args, shard_dir, device)
        return

    if args.mode == "merge":
        import pandas as pd

        # Section A may not have run yet if the caller invoked merge first; do
        # it here as a safety net (CPU-only, idempotent overwrite).
        if not args.skip_drift:
            drift_df = _section_a_drift(args, out_dir, fig_dir, tab_dir)
        else:
            drift_df = pd.DataFrame()

        per_channel_df = contrib_df = pd.DataFrame()
        if not args.skip_activation:
            captures = _merge_b_captures(shard_dir, args.world_size)
            per_channel_df, contrib_df = _finalize_b(
                captures, args, fig_dir, tab_dir, device,
            )

        ablation_df = pd.DataFrame()
        if not args.skip_ablation:
            ablation_df = _merge_c_ablation(shard_dir, args.world_size)
            _finalize_c(ablation_df, fig_dir, tab_dir)

        write_report(
            drift_df=drift_df, per_channel_df=per_channel_df,
            contrib_df=contrib_df, ablation_df=ablation_df,
            figures_dir="figures",
            out_path=str(out_dir / "report.md"),
            ckpt_dir=args.ckpt_dir, step=args.step,
            use_ema=args.use_ema, n_val=args.n_val,
        )
        print(f"[analyze][merge] report -> {out_dir}/report.md", flush=True)
        return


if __name__ == "__main__":
    main()
