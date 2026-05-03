"""C1 — verify CoartSparseMultiHeadRMSNorm.scale is a fp32 0-dim buffer.

After W2.0 refactor, the C1 fp32 scale lives directly on the new Coart
RMSNorm class (``coart.dit.modeling.rmsnorm.CoartSparseMultiHeadRMSNorm``),
no longer applied via monkey-patch on the upstream class.
"""
import torch

from coart.dit.modeling.rmsnorm import CoartSparseMultiHeadRMSNorm


def test_scale_is_fp32_buffer():
    norm = CoartSparseMultiHeadRMSNorm(dim=128, heads=12)
    assert isinstance(norm.scale, torch.Tensor), (
        f"expected scale to be a buffer (Tensor), got {type(norm.scale)}"
    )
    assert norm.scale.dtype == torch.float32
    assert norm.scale.numel() == 1
    # gamma stays as the upstream-shaped (heads, dim) parameter.
    assert tuple(norm.gamma.shape) == (12, 128)
