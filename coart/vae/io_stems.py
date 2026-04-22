"""Three-branch IO stems for feat18 SC-VAE encoder/decoder.

The 18-channel corep payload has three semantically distinct blocks:
    ch 0:3   - point1 xyz (local cube coords, lower-z representative)
    ch 3:6   - point2 xyz (higher-z; zero in 97.8% of single-point cubes)
    ch 6:18  - edge/face ordinal counts (integers 0..~22, 99% in {0,1})

Feat18EncIO splits the 18 -> C_model projection into three independent nn.Linear
branches summed together. This preserves the z-sort signal (p1 and p2 not
shared -> no permutation ambiguity) and allows warm-starting each branch
from a different pretrained source.

Feat18DecIO mirrors the structure on the decoder side: three independent heads
(p1_head, p2_head, ef_head), concatenated to produce the 18-channel output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Feat18EncIO(nn.Module):
    """Three-branch input stem for 18-ch corep feats.

    Forward:
        h = p1_branch(f[:, 0:3]) + p2_branch(f[:, 3:6]) + ef_branch(f[:, 6:18])
    Output shape: (N, c_model).

    Input must be a SparseTensor-like object with attributes `.feats` (N, 18)
    and method `.replace(new_feats) -> same-type`. Does NOT modify the original
    tensor metadata (coords / stride / etc.) — only swaps feats.
    """

    def __init__(self, c_model: int):
        super().__init__()
        self.p1_branch = nn.Linear(3, c_model)
        self.p2_branch = nn.Linear(3, c_model)
        self.ef_branch = nn.Linear(12, c_model)

    def forward(self, x):
        f = x.feats
        h = (
            self.p1_branch(f[:, 0:3])
            + self.p2_branch(f[:, 3:6])
            + self.ef_branch(f[:, 6:18])
        )
        return x.replace(h)


class Feat18DecIO(nn.Module):
    """Three-head output stem producing 18 channels from c_model features.

    Forward:
        out = concat[p1_head(f), p2_head(f), ef_head(f)]   # -> (N, 18)

    Matches the feat18 channel layout expected by `feats_to_param` /
    `feature_to_mesh` downstream: [p1(3), p2(3), ef(12)].
    """

    def __init__(self, c_model: int):
        super().__init__()
        self.p1_head = nn.Linear(c_model, 3)
        self.p2_head = nn.Linear(c_model, 3)
        self.ef_head = nn.Linear(c_model, 12)

    def forward(self, x):
        f = x.feats
        out = torch.cat([
            self.p1_head(f),
            self.p2_head(f),
            self.ef_head(f),
        ], dim=-1)
        return x.replace(out)
