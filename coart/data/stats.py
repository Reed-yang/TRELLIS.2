"""Per-channel normalisation for 18-ch corep feats.

Layout of the 18 channels (from corep_fast param_to_feats):
    [point1_xyz(3), point2_xyz(3), edge_weights(6), face_weights(6)]

Normalisation convention (matches train_finetune_feat18.py:481-484):
    ch 0:6   — (x - 0.5) / 1.0       (point coords already in [0, 1] local cube)
    ch 6:18  — (x - mean) / std      (computed offline from 41k shards)

If the stats file is missing, fall back to identity defaults (mean=[0.5]*6+[0]*12,
std=ones), matching `load_stats` in the original script.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


def load_stats(
    stats_path: str | None,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) tensors of shape (18,) on `device`."""
    if stats_path is None or not os.path.exists(stats_path):
        if verbose:
            print(f"[stats] no file at {stats_path!r}, using identity defaults")
        mean = np.zeros(18, dtype=np.float32)
        mean[:6] = 0.5
        std = np.ones(18, dtype=np.float32)
    else:
        s = np.load(stats_path)
        mean = s["mean"].astype(np.float32)
        std = np.maximum(s["std"].astype(np.float32), 1e-3)  # hard-clamp safety
    if verbose:
        print(f"[stats] mean = {np.round(mean, 4).tolist()}")
        print(f"[stats] std  = {np.round(std, 4).tolist()}")
    return (
        torch.from_numpy(mean).to(device),
        torch.from_numpy(std).to(device),
    )


def normalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply per-channel (x - mean) / std. Shapes: feats (N, 18), mean/std (18,)."""
    return (feats - mean) / std


def denormalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert `normalize`. For logging/visualisation only (training uses normalised space)."""
    return feats * std + mean
