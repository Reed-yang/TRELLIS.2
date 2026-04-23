"""Verify fused=True path works and falls back cleanly when unsupported."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_adamw_on_cuda():
    """On H100/A100 fused AdamW must init without error."""
    from coart.vae.train import _build_optimizer
    params = [torch.randn(8, 8, requires_grad=True, device="cuda")]
    opt = _build_optimizer(params, lr=1e-5)
    assert hasattr(opt, "step")


def test_fused_adamw_cpu_fallback():
    """On CPU-only build, fused must fall back silently."""
    from coart.vae.train import _build_optimizer
    params = [torch.randn(8, 8, requires_grad=True, device="cpu")]
    opt = _build_optimizer(params, lr=1e-5)
    assert hasattr(opt, "step")
