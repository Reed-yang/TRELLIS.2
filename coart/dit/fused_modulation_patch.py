"""Optional fused-modulation patch for ModulatedSparseTransformerCrossBlock._forward.

Chrome trace + autograd-graph deep-dive on the baseline shows that
``indexing_backward_kernel<BFloat16, 4>`` dominates GPU time at **55%**
(1476 ms / step). The kernel is the backward of `aten::index` on the
modulation broadcast inside :class:`SparseTensor.__elemwise__`:

    other = other[self.batch_boardcast_map]   # (B, C) -> (T, C)

`ModulatedSparseTransformerCrossBlock._forward` issues 6 such broadcasts
per block: ``h * (1+scale_msa) + shift_msa``, ``h * gate_msa``, then 3
more for the MLP. With 30 blocks that is 180 `aten::index` per fwd ⇒
180 `IndexBackward0` ⇒ 180 `_index_put_impl_(accumulate=True, bf16 atomic)`
per step. Atomics serialize on the small (B, C) accumulator (~7000 colliding
writes per row) → ~4 ms/call.

This patch overrides ``_forward`` to do **one** ``index_select`` per block
instead of six. The (B, 6C) modulation is expanded once into (T, 6C),
then chunked into 6 plain (T, C) tensors which are consumed by ordinary
mul+add on `.feats` (no SparseTensor.__elemwise__ broadcast at all).

Net effect (predicted):
  * fwd: 360 -> 30 `aten::index_select` calls / step (12× fewer)
  * bwd: 360 `_index_put_impl_` -> 30 `index_add_` calls (matched faster path)
  * ~1461 ms / step saved on H100 (≈ -54% step time)

Numerical equivalence: the new code computes ``feats * (1 + scale[bm]) +
shift[bm]`` exactly the same way, just batches the indexing. Outputs
should match the original to bf16 ULP.

Patch is gated by env ``COART_FUSE_MODULATION=1`` (default off — opt-in).
Apply by importing :mod:`coart.dit.fused_modulation_patch`. Composes with
the existing `coart.dit` import chain so it propagates to mp.spawn workers.
"""
from __future__ import annotations

import os

import torch


_PATCHED = False


def _make_fused_forward():
    """Build a replacement `_forward` for `ModulatedSparseTransformerCrossBlock`."""

    def _forward(self, x, mod, context):
        # Build (B, 6C) modulator (matches upstream choice between share_mod paths).
        if self.share_mod:
            mod_full = (self.modulation + mod).type(mod.dtype)  # (B, 6C)
        else:
            mod_full = self.adaLN_modulation(mod)  # (B, 6C)

        # Single fwd index_select to expand (B, 6C) -> (T, 6C).
        # Backward of `index_select` is `index_add_` on the small (B, 6C)
        # accumulator — much faster than the 6 individual `_index_put_impl_`
        # calls that the original SparseTensor `*` / `+` overloads issue.
        bm = x.batch_boardcast_map
        mod_t = mod_full.index_select(0, bm)  # (T, 6C)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_t.chunk(6, dim=1)

        # All operations below are plain elementwise on dense (T, C) tensors;
        # SparseTensor.__elemwise__ is bypassed because operands have the same
        # leading dim T, so we never hit the `other[batch_boardcast_map]` path.
        h_feats = self.norm1(x.feats)
        h_feats = h_feats * (1 + scale_msa) + shift_msa
        h = x.replace(h_feats)
        h = self.self_attn(h)
        h = h.replace(h.feats * gate_msa)
        x = x + h

        h_feats = self.norm2(x.feats)
        h = x.replace(h_feats)
        h = self.cross_attn(h, context)
        x = x + h

        h_feats = self.norm3(x.feats)
        h_feats = h_feats * (1 + scale_mlp) + shift_mlp
        h = x.replace(h_feats)
        h = self.mlp(h)
        h = h.replace(h.feats * gate_mlp)
        x = x + h
        return x

    return _forward


def install() -> bool:
    """Apply the monkey patch. Returns True if applied, False if skipped."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("COART_FUSE_MODULATION") != "1":
        return False
    if not torch.cuda.is_available():
        return False

    from trellis2.modules.sparse.transformer import modulated as _mod

    fused_forward = _make_fused_forward()
    _mod.ModulatedSparseTransformerCrossBlock._forward = fused_forward
    _PATCHED = True
    print("[fused_modulation_patch] ModulatedSparseTransformerCrossBlock._forward "
          "→ batched index_select + chunk (single broadcast/block)")
    return True


# Auto-install on import (gated by env var). Cheap when env unset.
install()
