"""See spec §7 in
docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md.
"""
from __future__ import annotations

from typing import List

import torch


ABLATION_CONDITIONS: List[str] = [
    "full", "zero_ef", "zero_p2", "oracle_ef", "oracle_p2",
]


def apply_head_substitution(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    condition: str,
) -> torch.Tensor:
    """Return a new tensor with the requested head substitution applied.

    All inputs/outputs live in *normalised* feature space (the same space
    seen at the decoder output). Caller is responsible for `denormalize`
    afterwards. The returned tensor is always a fresh clone — never aliases
    the input.
    """
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; choose from {ABLATION_CONDITIONS}"
        )
    if pred_norm.shape != target_norm.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred_norm.shape} vs {target_norm.shape}"
        )

    out = pred_norm.clone()
    if condition == "full":
        return out
    if condition == "zero_ef":
        out[:, 6:18] = 0.0
    elif condition == "zero_p2":
        out[:, 3:6] = 0.0
    elif condition == "oracle_ef":
        out[:, 6:18] = target_norm[:, 6:18]
    elif condition == "oracle_p2":
        out[:, 3:6] = target_norm[:, 3:6]
    return out
