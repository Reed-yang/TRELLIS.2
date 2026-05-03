"""CoartSparseMultiHeadRMSNorm.

Adapted from trellis2/modules/sparse/attention/modules.py:11-24 @ de38fdd.

Changes vs upstream:
- C1 (W1): self.scale is a fp32 buffer (was Python float). Eliminates the
  (float, double) scalar mul slow-path dispatch in autocast bf16 ctx.
- C7 (W5): forward will be replaced with flash_attn fused rms_norm_fn.
  Currently same math as upstream so this file is bit-equivalent vs the
  W1 baseline (which had C1 monkey-patched and the original forward intact).

NOTE on bit-exact equivalence: we MUST keep upstream's ``x.float()`` /
``x.to(x_type)`` round-trip — F.normalize on bf16 vs fp32 differs in low
bits. The W1 baseline used the upstream forward verbatim (only ``__init__``
was patched), so deviating here would break the W2.0 "0 step time Δ +
bit-exact" gate.
"""
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules.sparse import VarLenTensor


class CoartSparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        # C1: scale as fp32 buffer (vs Python float) — keeps the eltwise mul
        # on the hot vectorized fp32 kernel instead of the (float, double)
        # slow path. Bit-equivalent vs the W1 monkey-patched __init__.
        self.register_buffer(
            "scale",
            torch.tensor(dim ** 0.5, dtype=torch.float32),
            persistent=False,
        )
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(
        self, x: Union[VarLenTensor, torch.Tensor]
    ) -> Union[VarLenTensor, torch.Tensor]:
        # Bit-equivalent to upstream (only __init__ differs from upstream).
        # W5 will swap this body for flash_attn's fused rms_norm_fn.
        x_type = x.dtype
        x = x.float()
        if isinstance(x, VarLenTensor):
            x = x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        else:
            x = F.normalize(x, dim=-1) * self.gamma * self.scale
        return x.to(x_type)
