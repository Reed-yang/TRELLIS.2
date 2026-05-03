"""C1 — verify SparseMultiHeadRMSNorm.scale becomes a fp32 buffer after patch.

The patch lives in :mod:`coart.dit.fused_modulation_patch` (W1 temporary site).
Importing that module installs the patch; afterwards constructing a
``SparseMultiHeadRMSNorm`` should yield a tensor (not Python float) ``.scale``.
"""
import torch

from trellis2.modules.sparse.attention.modules import SparseMultiHeadRMSNorm


def test_scale_is_fp32_buffer_after_patch():
    # Apply C1 patch.
    import coart.dit.fused_modulation_patch  # noqa: F401  (patch on import)

    norm = SparseMultiHeadRMSNorm(dim=128, heads=12)
    assert isinstance(norm.scale, torch.Tensor), (
        f"expected scale to be a buffer (Tensor), got {type(norm.scale)}"
    )
    assert norm.scale.dtype == torch.float32
    assert norm.scale.numel() == 1
