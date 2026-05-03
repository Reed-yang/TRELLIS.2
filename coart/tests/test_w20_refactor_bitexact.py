"""W2.0 refactor must be bit-equivalent vs upstream block + fused_modulation logic.

Compares :class:`CoartDitBlock` output against an upstream
:class:`ModulatedSparseTransformerCrossBlock` initialized with the same
weights. Both must produce identical bf16 output (within bf16 noise
tolerance, < 1e-3) — the refactor only:
  * folds in the W1 fused_modulation patch (already verified bf16-ULP equiv)
  * swaps qk_rms_norm to CoartSparseMultiHeadRMSNorm (only __init__ differs
    — scale becomes fp32 buffer, mathematically identical scalar value)

C7 (W5) note: ``COART_DISABLE_FUSED_RMSNORM=1`` is set so this test still
compares the unfused path (bit-equivalent vs upstream). Numerical
equivalence of the fused path is covered by ``test_w5_fused_rmsnorm_ulp``.
"""
import os
import pytest
import torch


def test_coart_block_matches_upstream(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    # C7: force unfused RMSNorm so this regression test still compares
    # against the bit-equivalent path. Fused path is gated by its own
    # ULP test (test_w5_fused_rmsnorm_ulp).
    monkeypatch.setenv("COART_DISABLE_FUSED_RMSNORM", "1")
    torch.manual_seed(42)
    from trellis2.modules.sparse import SparseTensor
    from trellis2.modules.sparse.transformer.modulated import (
        ModulatedSparseTransformerCrossBlock,
    )
    from coart.dit.modeling.block import CoartDitBlock

    cfg = dict(
        channels=192, ctx_channels=1024, num_heads=3,
        mlp_ratio=4.0, attn_mode="full", use_checkpoint=False,
        use_rope=True, share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    upstream = ModulatedSparseTransformerCrossBlock(**cfg).cuda().eval()
    new = CoartDitBlock(**cfg).cuda().eval()
    new.load_state_dict(upstream.state_dict(), strict=False)

    B, T, C = 2, 64, 192
    coords = torch.randint(0, 16, (T, 4), device="cuda", dtype=torch.int32)
    coords[:, 0] = torch.randint(0, B, (T,), device="cuda", dtype=torch.int32)
    feats = torch.randn(T, C, device="cuda", dtype=torch.bfloat16)
    x = SparseTensor(feats=feats, coords=coords)

    mod = torch.randn(B, 6 * C, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(B, 32, 1024, device="cuda", dtype=torch.bfloat16)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        y_up = upstream(x, mod, context)
        y_new = new(x, mod, context)

    diff = (y_up.feats.float() - y_new.feats.float()).abs().max().item()
    print(f"[w20] max abs diff = {diff} (bit-equivalent up to bf16 noise)")
    assert diff < 1e-3, f"refactor introduced numerical diff {diff}"
