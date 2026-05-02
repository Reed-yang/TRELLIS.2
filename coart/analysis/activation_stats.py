"""See spec §6 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def compute_per_channel_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_eps: float = 0.05,
) -> pd.DataFrame:
    """Per-channel pred/target stats over voxels.

    Args:
        pred, target: (N, C) tensors in raw (denormalised) space.
        zero_eps: |x| < zero_eps counts as "zero" for `pred_zero_rate` and
            `target_zero_rate`. Default 0.05 matches the spec.

    Returns: DataFrame with one row per channel.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    p = pred.detach().to(torch.float64).cpu()
    t = target.detach().to(torch.float64).cpu()
    N, C = p.shape

    rows: List[Dict[str, object]] = []
    for c in range(C):
        pc = p[:, c]
        tc = t[:, c]
        nz_mask = tc.abs() > zero_eps
        n_nz = int(nz_mask.sum().item())

        if n_nz >= 2:
            mse_cond = ((pc[nz_mask] - tc[nz_mask]) ** 2).mean().item()
        else:
            mse_cond = float("nan")

        pc_centred = pc - pc.mean()
        tc_centred = tc - tc.mean()
        denom = pc_centred.norm() * tc_centred.norm()
        pearson = (
            (pc_centred * tc_centred).sum() / denom
        ).item() if denom.item() > 1e-12 else 0.0

        rows.append({
            "channel": c,
            "pred_mean": pc.mean().item(),
            "pred_std": pc.std(unbiased=False).item(),
            "target_mean": tc.mean().item(),
            "target_std": tc.std(unbiased=False).item(),
            "pearson_r": pearson,
            "mse_overall": ((pc - tc) ** 2).mean().item(),
            "mse_conditional": mse_cond,
            "pred_zero_rate": (pc.abs() < zero_eps).float().mean().item(),
            "target_zero_rate": (tc.abs() < zero_eps).float().mean().item(),
            "n_target_nonzero": n_nz,
        })
    return pd.DataFrame(rows)


def compute_branch_norms(
    contributions: torch.Tensor,
    signal_mask: torch.Tensor,
) -> Dict[str, float]:
    """Per-voxel L2 norms of a branch's contribution, partitioned by signal.

    Args:
        contributions: (N, c_model) tensor of one branch's output before sum.
        signal_mask: (N,) bool tensor; True = voxel is "signal-bearing"
            (e.g. p2 input non-zero, or any ef channel non-zero in the GT).
    """
    norms = contributions.detach().to(torch.float64).norm(dim=1).cpu()
    sig = signal_mask.bool().cpu()
    n_sig = int(sig.sum().item())
    n_zero = int((~sig).sum().item())

    return {
        "norm_all_mean": norms.mean().item(),
        "norm_all_std": norms.std(unbiased=False).item(),
        "norm_signal_mean": (
            norms[sig].mean().item() if n_sig > 0 else float("nan")
        ),
        "norm_signal_std": (
            norms[sig].std(unbiased=False).item() if n_sig > 1 else float("nan")
        ),
        "norm_zero_mean": (
            norms[~sig].mean().item() if n_zero > 0 else float("nan")
        ),
        "norm_zero_std": (
            norms[~sig].std(unbiased=False).item() if n_zero > 1 else float("nan")
        ),
        "n_signal": n_sig,
        "n_zero": n_zero,
    }
