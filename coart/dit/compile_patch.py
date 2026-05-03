"""Optional torch.compile patch for the modulation/LayerNorm dense subgraph
inside ``ModulatedSparseTransformerCrossBlock._forward``.

Context
-------
Chrome-trace deep-dive on the COART_FUSE_MODULATION=1 baseline shows that
``eltwise + LayerNorm`` kernels still account for ~460 ms / 1182 ms
(~39%) of GPU time per step. Each block runs three dense sequences:

    h = LN(feats);  h = h * (1 + scale) + shift                # pre-attn
    h = LN(feats)                                               # pre-cross
    h = LN(feats);  h = h * (1 + scale) + shift                # pre-mlp

plus three ``h * gate`` updates after each sub-module, all on dense
``(T, 1536)`` bf16 tensors. These are launch-overhead bound; inductor can
fuse the LN reduction with the (1+scale)*h+shift mul-add into a single
kernel and trim the launch tax.

This patch wraps the dense subgraph in ``torch.compile(dynamic=True)``.
SparseTensor never enters the compiled function — we pass only ``.feats``
plus the modulation chunks. Composes on top of ``fused_modulation_patch``
(must be installed AFTER it). Gated by ``COART_COMPILE=1``.

Lazy-compile design
-------------------
``torch.compile`` is **not** invoked at import time. We only swap in a
patched ``_forward`` that wraps each helper on first call. Why:

* DataLoader workers (`pt_data_worker` subprocesses) inherit the import
  side-effects of the parent. Eager ``torch.compile()`` at import would
  cause every worker to spin up the inductor compile-worker pool and
  cudaInit, blowing up to thousands of zombie compile workers and pinning
  GPU memory in dataloader processes. We saw 4400+ inductor processes
  and 60GB held by the dataloader pool before adding this guard.
* By deferring compile to first forward, only the actual GPU rank
  processes ever touch inductor. DataLoader workers never call forward,
  so they stay clean.

Design choices
--------------
* dynamic=True — sparse T varies per step; static shapes would trigger
  endless recompilation.
* mode="default" — reduce-overhead (cudagraphs) is incompatible with
  dynamic shapes here. Default backend still gets the kernel-fusion win.
* Compile only the LN + (1+scale)*h + shift / LN-only / h*gate kernels;
  attention sub-modules and SparseTensor ops stay outside the graph.
* Helpers are module-level singletons protected by a lock, so the very
  first concurrent forward across blocks doesn't double-compile.
"""
from __future__ import annotations

import os
import threading

import torch


_PATCHED = False
_LOCK = threading.Lock()
_HELPERS = None  # populated on first forward (in GPU rank only)


def _ln_modshift_eager(feats, ln_weight, ln_bias, eps, scale, shift):
    # bf16 -> fp32 LN -> bf16, then (1+scale)*h + shift, all dense.
    x_dtype = feats.dtype
    h = torch.nn.functional.layer_norm(
        feats.to(torch.float32),
        (feats.shape[-1],),
        weight=ln_weight.to(torch.float32) if ln_weight is not None else None,
        bias=ln_bias.to(torch.float32) if ln_bias is not None else None,
        eps=eps,
    ).to(x_dtype)
    return h * (1.0 + scale) + shift


def _ln_only_eager(feats, ln_weight, ln_bias, eps):
    x_dtype = feats.dtype
    h = torch.nn.functional.layer_norm(
        feats.to(torch.float32),
        (feats.shape[-1],),
        weight=ln_weight.to(torch.float32) if ln_weight is not None else None,
        bias=ln_bias.to(torch.float32) if ln_bias is not None else None,
        eps=eps,
    ).to(x_dtype)
    return h


def _gate_mul_eager(feats, gate):
    return feats * gate


def _get_helpers():
    """Return the (compiled) helpers, building them on first call.

    Called from ``_forward``, which only runs in GPU rank processes — so
    inductor never starts up in dataloader workers.
    """
    global _HELPERS
    if _HELPERS is not None:
        return _HELPERS
    with _LOCK:
        if _HELPERS is not None:
            return _HELPERS
        ln_modshift_c = torch.compile(_ln_modshift_eager, dynamic=True)
        ln_only_c = torch.compile(_ln_only_eager, dynamic=True)
        gate_mul_c = torch.compile(_gate_mul_eager, dynamic=True)
        _HELPERS = (ln_modshift_c, ln_only_c, gate_mul_c)
        # Print once, from the first forward in each rank.
        print("[compile_patch] lazy-compiled LN+mod-shift / LN / gate helpers "
              "(dynamic=True, mode=default)")
        return _HELPERS


def _make_compiled_forward():
    """Return a replacement _forward that funnels dense ops through compiled helpers."""

    def _forward(self, x, mod, context):
        ln_modshift_c, ln_only_c, gate_mul_c = _get_helpers()

        # 1) Build (B, 6C) modulator (unchanged from fused_modulation_patch).
        if self.share_mod:
            mod_full = (self.modulation + mod).type(mod.dtype)
        else:
            mod_full = self.adaLN_modulation(mod)

        # 2) Single index_select to expand (B, 6C) -> (T, 6C); chunk into 6 (T, C).
        bm = x.batch_boardcast_map
        mod_t = mod_full.index_select(0, bm)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_t.chunk(6, dim=1)

        # 3) MSA path: LN + mod-shift fused via compile.
        h_feats = ln_modshift_c(
            x.feats, self.norm1.weight, self.norm1.bias, self.norm1.eps,
            scale_msa, shift_msa,
        )
        h = x.replace(h_feats)
        h = self.self_attn(h)
        h = h.replace(gate_mul_c(h.feats, gate_msa))
        x = x + h

        # 4) Cross-attn path: LN only (norm2 has affine; helper handles both cases).
        h_feats = ln_only_c(x.feats, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        h = x.replace(h_feats)
        h = self.cross_attn(h, context)
        x = x + h

        # 5) MLP path: LN + mod-shift fused, then gate after mlp.
        h_feats = ln_modshift_c(
            x.feats, self.norm3.weight, self.norm3.bias, self.norm3.eps,
            scale_mlp, shift_mlp,
        )
        h = x.replace(h_feats)
        h = self.mlp(h)
        h = h.replace(gate_mul_c(h.feats, gate_mlp))
        x = x + h
        return x

    return _forward


def install() -> bool:
    """Apply the compile patch (monkey-patch only; compile happens lazily).

    Returns True if applied, False if skipped. Cheap when env unset because
    we only swap a method pointer — no inductor activity at import time.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("COART_COMPILE") != "1":
        return False
    if not torch.cuda.is_available():
        return False

    if os.environ.get("COART_FUSE_MODULATION") != "1":
        print("[compile_patch] warn: COART_FUSE_MODULATION!=1; compile patch "
              "will replace upstream _forward directly.")

    try:
        from trellis2.modules.sparse.transformer import modulated as _mod
    except Exception as e:  # pragma: no cover - defensive
        print(f"[compile_patch] skipped: import failed: {e}")
        return False

    _mod.ModulatedSparseTransformerCrossBlock._forward = _make_compiled_forward()
    _PATCHED = True
    print("[compile_patch] ModulatedSparseTransformerCrossBlock._forward "
          "→ lazy torch.compile wrapper installed (compile fires on first forward)")
    return True


# Auto-install on import (gated by env var). Cheap when env unset; even when
# set, this only swaps a method pointer — no torch.compile() yet.
install()
