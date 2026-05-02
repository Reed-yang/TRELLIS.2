"""See spec §5 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch


_TOP_K_SV = 5


def compute_drift_metrics(
    W_now: torch.Tensor,
    b_now: torch.Tensor,
    W_ref: torch.Tensor,
    b_ref: torch.Tensor,
) -> Dict[str, object]:
    """Static drift metrics between two `nn.Linear` weight/bias pairs.

    Both tensors must have matching shapes. Computation is fp64 internally
    to avoid catastrophic cancellation on near-zero drift.
    """
    if W_now.shape != W_ref.shape:
        raise ValueError(f"weight shape mismatch: {W_now.shape} vs {W_ref.shape}")
    if b_now.shape != b_ref.shape:
        raise ValueError(f"bias shape mismatch: {b_now.shape} vs {b_ref.shape}")

    W_now64 = W_now.detach().to(torch.float64)
    W_ref64 = W_ref.detach().to(torch.float64)
    b_now64 = b_now.detach().to(torch.float64)
    b_ref64 = b_ref.detach().to(torch.float64)

    diff = W_now64 - W_ref64
    rel_frob = (diff.norm() / W_ref64.norm().clamp_min(1e-12)).item()
    rms = (diff.norm() / math.sqrt(W_ref64.numel())).item()
    bias_drift = (
        (b_now64 - b_ref64).norm() / math.sqrt(max(b_ref64.numel(), 1))
    ).item()

    sv = torch.linalg.svdvals(W_now64)
    top = sv[:_TOP_K_SV].tolist()
    if len(top) < _TOP_K_SV:
        top = top + [0.0] * (_TOP_K_SV - len(top))

    s_sum = sv.sum().clamp_min(1e-12)
    p = (sv / s_sum).clamp_min(1e-30)
    H = -(p * p.log()).sum().item()
    eff_rank = math.exp(H)

    nonzero = sv[sv > 1e-9]
    cond = (sv[0] / nonzero[-1]).item() if nonzero.numel() > 0 else float("inf")

    cos = torch.nn.functional.cosine_similarity(
        W_now64, W_ref64, dim=1
    )
    mean_row_cos = cos.mean().item()

    return {
        "rel_frob_drift": rel_frob,
        "rms_elem_drift": rms,
        "bias_drift_per_dim": bias_drift,
        "singular_top5": top,
        "effective_rank": eff_rank,
        "cond_number": cond,
        "mean_row_cosine": mean_row_cos,
    }
