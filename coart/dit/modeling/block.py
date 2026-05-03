"""CoartDitBlock — modulated cross-attention transformer block.

Adapted from trellis2/modules/sparse/transformer/modulated.py:81-167 @ de38fdd
(class ModulatedSparseTransformerCrossBlock).

Changes vs upstream:
- Folds in fused_modulation_patch (W1): a single ``index_select`` + ``chunk(6)``
  per block replaces six SparseTensor ``__elemwise__`` broadcasts. Eliminates
  the ``indexing_backward_kernel`` 55%-of-GPU-time bottleneck (~1476 ms/step
  on H100). Was env-gated in fused_modulation_patch.py; now baked in.
- Swaps qk_rms_norm modules to CoartSparseMultiHeadRMSNorm (C1 fp32 scale
  buffer; W5 will replace forward with flash_attn fused rms_norm_fn).

Bit-exactness:
- Fused-modulation arithmetic is algebraically identical to upstream's
  six-broadcast form (just batched indexing — verified bf16-ULP equivalent
  in W1 chrome traces).
- CoartSparseMultiHeadRMSNorm matches upstream forward verbatim (only
  __init__ differs: scale = fp32 buffer instead of Python float).
"""
from __future__ import annotations
from typing import Union

import torch

from trellis2.modules.sparse import VarLenTensor, SparseTensor
from trellis2.modules.sparse.transformer.modulated import (
    ModulatedSparseTransformerCrossBlock,
)

from .rmsnorm import CoartSparseMultiHeadRMSNorm


class CoartDitBlock(ModulatedSparseTransformerCrossBlock):
    """Modulated cross-attn block with fused-modulation + Coart RMSNorm baked in.

    Inherits all module construction (norms, attentions, mlp, modulation
    parameters) from :class:`ModulatedSparseTransformerCrossBlock`. After
    super().__init__ we:

    1. Replace any ``q_rms_norm`` / ``k_rms_norm`` instances inside the two
       attention sub-modules with :class:`CoartSparseMultiHeadRMSNorm`.
       Same shapes / params, only ``scale`` becomes a fp32 buffer.
    2. Override ``_forward`` (below) to use the batched index_select form
       that bypasses ``SparseTensor.__elemwise__`` modulation broadcasts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace upstream SparseMultiHeadRMSNorm with Coart variant (C1
        # fp32 scale buffer). Same dim/heads → state_dict loads cleanly.
        for attn in (self.self_attn, self.cross_attn):
            if getattr(attn, "qk_rms_norm", False):
                # head_dim and num_heads are stored on the attention module.
                head_dim = attn.head_dim
                num_heads = attn.num_heads
                new_q = CoartSparseMultiHeadRMSNorm(head_dim, num_heads)
                new_k = CoartSparseMultiHeadRMSNorm(head_dim, num_heads)
                # Carry over learned gamma weights from upstream init.
                new_q.gamma.data.copy_(attn.q_rms_norm.gamma.data)
                new_k.gamma.data.copy_(attn.k_rms_norm.gamma.data)
                attn.q_rms_norm = new_q
                attn.k_rms_norm = new_k

    def _forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: Union[torch.Tensor, VarLenTensor],
    ) -> SparseTensor:
        # 1) Build (B, 6C) modulator (matches upstream choice between share_mod paths).
        if self.share_mod:
            mod_full = (self.modulation + mod).type(mod.dtype)  # (B, 6C)
        else:
            mod_full = self.adaLN_modulation(mod)  # (B, 6C)

        # 2) Single index_select per block: (B, 6C) -> (T, 6C); chunk into 6 (T, C).
        # Backward of index_select is index_add_ on the small (B, 6C) accumulator
        # — much faster than the 6 individual _index_put_impl_ calls that the
        # original SparseTensor.__elemwise__ broadcast path issues.
        bm = x.batch_boardcast_map
        mod_t = mod_full.index_select(0, bm)  # (T, 6C)
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = mod_t.chunk(6, dim=1)

        # 3) MSA path. All ops below are plain elementwise on dense (T, C)
        # tensors; SparseTensor.__elemwise__ is bypassed because operands
        # share leading dim T (no other[batch_boardcast_map] path).
        h_feats = self.norm1(x.feats)
        h_feats = h_feats * (1 + scale_msa) + shift_msa
        h = x.replace(h_feats)
        h = self.self_attn(h)
        h = h.replace(h.feats * gate_msa)
        x = x + h

        # 4) Cross-attn path. norm2 has elementwise_affine=True per upstream.
        h_feats = self.norm2(x.feats)
        h = x.replace(h_feats)
        h = self.cross_attn(h, context)
        x = x + h

        # 5) MLP path.
        h_feats = self.norm3(x.feats)
        h_feats = h_feats * (1 + scale_mlp) + shift_mlp
        h = x.replace(h_feats)
        h = self.mlp(h)
        h = h.replace(h.feats * gate_mlp)
        x = x + h
        return x
