"""Cached-features trainer for shape DiT finetune.

Inherits from ``trellis2.trainers.ImageConditionedSparseFlowMatchingCFGTrainer``
and overrides:

* ``_init_image_cond_model``: NO-OP. Our dataset emits cached DINO features
  directly, so we never need to load a DINO weight blob into VRAM.
* ``encode_image``: identity. The "image" tensor passed in by ``training_losses``
  is already the (B, T, 1024) feature tensor the denoiser expects.

The CFG dropout from ``ClassifierFreeGuidanceMixin.get_cond`` is preserved
intact: we still call ``super().get_cond`` after constructing the
zeros-like ``neg_cond``, so ``p_uncond`` semantics behave exactly as in the
upstream trainer.

Registered into ``trellis2.trainers`` namespace at import time so ``train.py``
can resolve us by name from JSON config.
"""
from __future__ import annotations

from typing import Any

import torch

from trellis2.trainers import ImageConditionedSparseFlowMatchingCFGTrainer


__all__ = ["CachedImageConditionedSparseFlowMatchingCFGTrainer"]


class CachedImageConditionedSparseFlowMatchingCFGTrainer(
    ImageConditionedSparseFlowMatchingCFGTrainer
):
    """Image-conditioned sparse FM CFG trainer that consumes cached features.

    Accepts the same arguments as the parent except ``image_cond_model``,
    which we silently absorb (and ignore) so existing JSON configs that still
    declare it can be reused unchanged.
    """

    def __init__(self, *args, image_cond_model: Any = None, **kwargs):
        # Forward a dummy image_cond_model dict to satisfy the base mixin's
        # __init__ signature. The dict is never read because we override
        # ``_init_image_cond_model`` and ``encode_image`` below.
        super().__init__(*args, image_cond_model=image_cond_model or {"name": "_unused", "args": {}}, **kwargs)

    # -------------------------------------------------------------- model init
    def _init_image_cond_model(self) -> None:
        """No-op: features are already cached on disk + supplied by the dataset."""
        # Mark as initialised (non-None) so encode_image short-circuit holds.
        self.image_cond_model = "cached"

    # -------------------------------------------------------------- encode path
    @torch.no_grad()
    def encode_image(self, image):
        """Identity passthrough. ``image`` is already the cached (B, T, 1024) feats."""
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"CachedImageConditionedSparseFlowMatchingCFGTrainer expects the "
                f"dataset to emit pre-extracted feature tensors, got {type(image)}"
            )
        return image

    # ---------------------------------------------------------------- snapshot
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """No-op: trellis2 BasicTrainer.snapshot_dataset stacks all dataset
        samples with `torch.stack`, but our dataset emits SparseTensor for
        ``x_0`` which can't be stacked that way. Skip this debug snapshot —
        actual training step doesn't need it."""
        pass

    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """No-op: BasicTrainer.snapshot calls ``.contiguous()`` on sample
        outputs and then ``dist.gather`` of the dense tensor. Our sparse
        flow-matching sampler emits SparseTensor, which has no
        ``.contiguous()`` method. Skip the periodic image probe — the loss
        curves + EMA ckpts are sufficient training signal; quality eval
        runs offline against the saved EMA ckpt."""
        pass


# ---------------------------------------------------------------------- registration
def _register() -> None:
    import trellis2.trainers as tt

    setattr(
        tt,
        "CachedImageConditionedSparseFlowMatchingCFGTrainer",
        CachedImageConditionedSparseFlowMatchingCFGTrainer,
    )


_register()
