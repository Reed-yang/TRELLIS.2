"""Eltwise-fusion patch for ModulatedSparseTransformerCrossBlock._forward.

Composes on top of ``fused_modulation_patch`` (must apply AFTER it). When
both patches are active (``COART_FUSE_MODULATION=1`` + ``COART_FUSE_ELTWISE=1``),
this patch's _forward replaces the v1 batched-index-select forward with a
v2 that adds three pure-PyTorch eltwise optimizations:

1. **LayerNorm bf16-native** — bypass ``LayerNorm32``'s explicit
   ``manual_cast(x, fp32) → LN → manual_cast(x, bf16)`` round trip. Calling
   ``F.layer_norm`` directly on bf16 input lets the kernel cast in-register
   (SRAM) instead of doing 2 full-tensor DRAM copies of the (T, C) bf16
   feature map per LN call. With 3 LN per block × 30 blocks = 90 calls
   per fwd; saves ~120 ms / step (eltwise ``bfloat16_copy_kernel``).

2. **Modulation via ``addcmul``** — ``h * (1 + scale) + shift`` rewritten as
   ``addcmul(shift, h, scale) + h``: one fused mul-add kernel + one add,
   replacing the 3-kernel ``add(1+scale) → mul → add(shift)``. Saves 1
   kernel × 2 chains × 30 blocks = 60 launches/step.

3. **Gate + residual via ``addcmul``** — the post-attn / post-mlp
   ``x.feats + h.feats * gate`` collapses into a single
   ``addcmul(x.feats, h.feats, gate)``. Replaces 2 kernels (mul + add) with
   1, × 3 sites × 30 blocks = 90 launches/step.

Predicted ROI on top of v1 fused_modulation (current step = 1181.6 ms,
eltwise = 460.5 ms / 39 %):
  * eltwise: -200 to -300 ms / step → ~160-260 ms / step (-50% of eltwise)
  * total: 1181 → ~900-980 ms / step (-17 to -24%)

Numerical equivalence: bf16 ULP. F.layer_norm on bf16 still does fp32
internal reductions; ``addcmul`` is the algebraically-identical fused
expression. Outputs match v1 to within bf16 round-off.

Gated by ``COART_FUSE_ELTWISE=1`` (default off — opt-in for measurement).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


_PATCHED = False


def _ln_bf16(ln, x_feats: torch.Tensor) -> torch.Tensor:
    """Bypass LayerNorm32's manual_cast — let F.layer_norm internally
    handle bf16↔fp32 casts in-register. Affine weight/bias stay fp32 if
    the layer was constructed that way (norm2: elementwise_affine=True);
    F.layer_norm casts them on-the-fly."""
    return F.layer_norm(
        x_feats,
        ln.normalized_shape,
        weight=ln.weight,
        bias=ln.bias,
        eps=ln.eps,
    )


def _make_eltwise_forward():
    """Build the v2 _forward with LN-bf16 + addcmul eltwise fusion."""

    def _forward(self, x, mod, context):
        # 1) Build (B, 6C) modulator — same as fused_modulation v1.
        if self.share_mod:
            mod_full = (self.modulation + mod).type(mod.dtype)
        else:
            mod_full = self.adaLN_modulation(mod)

        # 2) Single index_select per block: (B, 6C) -> (T, 6C); chunk into 6 (T, C).
        bm = x.batch_boardcast_map
        mod_t = mod_full.index_select(0, bm)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_t.chunk(6, dim=1)

        # === MSA path ===
        # LN: bf16 native (saves 2 manual_cast DRAM copies vs LayerNorm32).
        h_feats = _ln_bf16(self.norm1, x.feats)
        # Mod: addcmul(shift, h, scale) + h ≡ h*(1+scale) + shift. Saves
        # 1 kernel vs (1+scale)*h+shift's 3-kernel chain.
        h_feats = torch.addcmul(shift_msa, h_feats, scale_msa) + h_feats
        h = x.replace(h_feats)
        h = self.self_attn(h)
        # Gate + residual fuse: addcmul(x.feats, h.feats, gate) = x + h*gate.
        # Saves 1 kernel vs separate (h*gate) + add.
        x = x.replace(torch.addcmul(x.feats, h.feats, gate_msa))

        # === Cross-attn path ===
        h_feats = _ln_bf16(self.norm2, x.feats)
        h = x.replace(h_feats)
        h = self.cross_attn(h, context)
        # No gate on cross-attn output; plain residual.
        x = x + h

        # === MLP path ===
        h_feats = _ln_bf16(self.norm3, x.feats)
        h_feats = torch.addcmul(shift_mlp, h_feats, scale_mlp) + h_feats
        h = x.replace(h_feats)
        h = self.mlp(h)
        x = x.replace(torch.addcmul(x.feats, h.feats, gate_mlp))
        return x

    return _forward


def install() -> bool:
    """Apply the v2 eltwise patch. Returns True if applied, False if skipped."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("COART_FUSE_ELTWISE") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    # Soft dependency: warn if v1 fused_modulation isn't enabled — we override
    # the same _forward, but our v2 depends on the (T, 6C) chunk shape that v1
    # established (semantically equivalent though, even on raw upstream).
    if os.environ.get("COART_FUSE_MODULATION") != "1":
        print("[eltwise_patch] note: COART_FUSE_MODULATION!=1; v2 still works "
              "(superset of v1's batched index_select).")

    from trellis2.modules.sparse.transformer import modulated as _mod
    _mod.ModulatedSparseTransformerCrossBlock._forward = _make_eltwise_forward()
    _PATCHED = True
    print("[eltwise_patch] ModulatedSparseTransformerCrossBlock._forward → "
          "v2 (LN bf16-native + addcmul mod + addcmul gate-residual)")
    return True


# Auto-install on import (gated by env var). Cheap when env unset.
install()
