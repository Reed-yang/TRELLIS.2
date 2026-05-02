"""coart.dit -- finetune the 1.3B shape DiT on cached features.

This sub-package piggybacks on the upstream ``train.py`` + trellis2 trainer,
adding only the two glue classes that swap online DINO + online VAE encode for
disk-cached features:

* :class:`coart.dit.dataset.CachedImageConditionedSLatShape` -- reads
  per-asset ``.npz`` files for the SLat latent + DINOv3 ViT-L/16 features.
* :class:`coart.dit.trainer.CachedImageConditionedSparseFlowMatchingCFGTrainer`
  -- skips loading DINO into VRAM and treats ``cond`` as already-encoded
  features.

Importing this package side-effect-registers both classes into the
``trellis2.datasets`` and ``trellis2.trainers`` namespaces respectively, so
``train.py``'s generic ``getattr(datasets, cfg.dataset.name)`` resolution
works without modifying upstream code.

Pretrained weights and the default JSON config path live in
:mod:`coart.dit.config`.
"""
from __future__ import annotations

# Side-effect imports register classes into trellis2.{datasets,trainers}.
from . import dataset as _dataset  # noqa: F401
from . import trainer as _trainer  # noqa: F401

from .config import (
    COART_DIT_DATA_ROOT,
    DEFAULT_VAE_TAG,
    PRETRAINED_DIT_CKPT,
    PRETRAINED_DIT_META_JSON,
    WARMSTART_HINT,
    CoartDitFinetunePaths,
    default_config_path,
)
from .dataset import CachedImageConditionedSLatShape
from .trainer import CachedImageConditionedSparseFlowMatchingCFGTrainer

__all__ = [
    "CachedImageConditionedSLatShape",
    "CachedImageConditionedSparseFlowMatchingCFGTrainer",
    "CoartDitFinetunePaths",
    "default_config_path",
    "COART_DIT_DATA_ROOT",
    "DEFAULT_VAE_TAG",
    "PRETRAINED_DIT_CKPT",
    "PRETRAINED_DIT_META_JSON",
    "WARMSTART_HINT",
]
