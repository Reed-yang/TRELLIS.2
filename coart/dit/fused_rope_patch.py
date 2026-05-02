"""Optional flash_attn-fused RoPE for SparseRotaryPositionEmbedder.

The vanilla `trellis2.modules.sparse.attention.rope.SparseRotaryPositionEmbedder._rotary_embedding`
goes via `view_as_complex / mul / view_as_real`, whose backward on Hopper
shows up as fragmented `indexing_backward_kernel<BF16>` calls in chrome
traces.

This module monkey-patches the embedder to dispatch to
`flash_attn.layers.rotary.apply_rotary_emb_func` when:
  * env `COART_FUSE_ROPE=1` (default off — must opt-in for measurement)
  * cuda is available + flash_attn is importable

Numerical equivalence vs vanilla is bf16-ULP (max abs diff 0.03125).
The trellis2 RoPE uses GPT-J-style interleaved pairs (view_as_complex
on consecutive 2-dim chunks), which maps to ``interleaved=True`` in
flash_attn. The whole packed sparse sequence is treated as one
``cu_seqlens=[0, T]`` segment so each token can carry its own phase.

The patch is a no-op if ``COART_FUSE_ROPE`` is not set, so importing
this module is always safe.
"""
from __future__ import annotations

import os
from typing import Any

import torch


_PATCHED = False


def _build_fused_rotary():
    """Return a function with same signature as
    `SparseRotaryPositionEmbedder._rotary_embedding(self, x, phases) -> tensor`,
    but routing through flash_attn fused triton kernel."""
    from flash_attn.layers.rotary import apply_rotary_emb_func  # type: ignore

    def fused(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        # x: (T, H, D) bf16/fp16 (sparse-packed token stream)
        # phases: (T, D/2) complex64 — e^{i*theta_t,d}
        # Cast cos/sin to x.dtype (flash_attn requirement).
        cos = phases.real.to(x.dtype).contiguous()
        sin = phases.imag.to(x.dtype).contiguous()
        T = x.shape[0]
        cu = torch.tensor([0, T], dtype=torch.int32, device=x.device)
        out = apply_rotary_emb_func(
            x,
            cos,
            sin,
            interleaved=True,         # GPT-J pair-of-2 layout (matches view_as_complex)
            inplace=False,
            cu_seqlens=cu,
            max_seqlen=T,
        )
        return out

    return fused


def install() -> bool:
    """Apply the monkey patch. Returns True if applied, False if skipped."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("COART_FUSE_ROPE") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func  # noqa: F401
    except ImportError:
        print("[fused_rope_patch] flash_attn not available, skipping")
        return False

    from trellis2.modules.sparse.attention import rope as sparse_rope_mod

    fused = _build_fused_rotary()
    sparse_rope_mod.SparseRotaryPositionEmbedder._rotary_embedding = fused
    _PATCHED = True
    print("[fused_rope_patch] SparseRotaryPositionEmbedder._rotary_embedding "
          "→ flash_attn.apply_rotary_emb_func")
    return True


# Auto-install on import (gated by env var). Cheap when env is unset.
install()
