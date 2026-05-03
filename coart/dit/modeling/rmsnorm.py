"""CoartSparseMultiHeadRMSNorm.

Adapted from trellis2/modules/sparse/attention/modules.py:11-24 @ de38fdd.

Changes vs upstream:
- C1 (W1): self.scale is a fp32 buffer (was Python float). Eliminates the
  (float, double) scalar mul slow-path dispatch in autocast bf16 ctx.
- C7 (W5): forward dispatches to flash_attn fused rms_norm_fn (Triton impl)
  when available. The C++ ``dropout_layer_norm`` extension is not built in
  this venv, so we use ``flash_attn.ops.triton.layer_norm.rms_norm_fn``
  which works without the extension.

  flash_attn requires a 1D weight matching the last dim, but our gamma is
  per-head ``[num_heads, head_dim]``. Workaround: reshape ``[..., H, D]``
  -> ``[N*H, D]``, run fused RMS-norm with ``weight=ones`` (so it's a pure
  RMS-norm without per-head gamma), reshape back, then apply per-head
  gamma + scalar scale separately as a single fused bf16 mul (gamma * scale
  pre-computed once when both are constants — but here gamma is a
  Parameter so we just do two muls; PyTorch fuses adjacent elementwise).

NOTE on bit-exact equivalence vs upstream: the unfused path keeps
upstream's ``x.float()`` / ``x.to(x_type)`` round-trip — F.normalize on
bf16 vs fp32 differs in low bits. The W1 baseline used the upstream
forward verbatim (only ``__init__`` was patched), so the unfused path
remains bit-equivalent vs W1 / W2.0.

Set ``COART_DISABLE_FUSED_RMSNORM=1`` to revert to the unfused path
(useful for ablation / numerical regression checks).
"""
from __future__ import annotations
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules.sparse import VarLenTensor

try:
    # Triton-backed implementation works without the dropout_layer_norm
    # C++ extension; the top-level flash_attn.ops.rms_norm path tries to
    # import that extension and fails on this venv.
    from flash_attn.ops.triton.layer_norm import rms_norm_fn as _flash_rms_norm
    _HAS_FLASH_RMS = True
except ImportError:
    _HAS_FLASH_RMS = False


class CoartSparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        # C1: scale as fp32 buffer (vs Python float) — keeps the eltwise mul
        # on the hot vectorized fp32 kernel instead of the (float, double)
        # slow path. Bit-equivalent vs the W1 monkey-patched __init__.
        self.register_buffer(
            "scale",
            torch.tensor(dim ** 0.5, dtype=torch.float32),
            persistent=False,
        )
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def _forward_unfused(self, feats: torch.Tensor) -> torch.Tensor:
        """Bit-equivalent to upstream — keep the float()/.to(dtype) round-trip."""
        x_type = feats.dtype
        x = feats.float()
        out = F.normalize(x, dim=-1) * self.gamma * self.scale
        return out.to(x_type)

    def _forward_fused(self, feats: torch.Tensor) -> torch.Tensor:
        """Fused RMS-norm via flash_attn (Triton). feats shape: ``[..., heads, dim]``.

        Math note: upstream computes ``F.normalize(x) * gamma * scale`` where
        ``scale = sqrt(D)``. ``F.normalize(x) = x / ||x||_2``, while
        ``rms_norm(x) = x / sqrt(mean(x^2)) = x / (||x||_2 / sqrt(D))
        = F.normalize(x) * sqrt(D)``. So ``F.normalize(x) * scale ==
        rms_norm(x)`` and the fused replacement is just
        ``rms_norm(x) * gamma`` — no extra ``* scale`` needed.
        """
        x_type = feats.dtype
        orig_shape = feats.shape
        feats_flat = feats.reshape(-1, self.dim).contiguous()
        ones_w = torch.ones(self.dim, device=feats.device, dtype=torch.float32)
        # Pure RMS-norm: weight=ones (per-head gamma applied below).
        # eps matches F.normalize default (1e-12).
        normed = _flash_rms_norm(feats_flat, ones_w, None, eps=1e-12)
        normed = normed.reshape(orig_shape)
        # Per-head gamma only (rms_norm already encodes the sqrt(D) scale);
        # cast back to input dtype to match the unfused path.
        return (normed * self.gamma).to(x_type)

    def forward(
        self, x: Union[VarLenTensor, torch.Tensor]
    ) -> Union[VarLenTensor, torch.Tensor]:
        # COART_DISABLE_FUSED_RMSNORM=1 reverts to unfused for ablation.
        use_fused = _HAS_FLASH_RMS and os.environ.get(
            "COART_DISABLE_FUSED_RMSNORM", "0"
        ) != "1"
        if isinstance(x, VarLenTensor):
            feats = x.feats
            out = self._forward_fused(feats) if use_fused else self._forward_unfused(feats)
            return x.replace(out)
        return self._forward_fused(x) if use_fused else self._forward_unfused(x)
