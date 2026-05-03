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

# Optional flash_attn-fused RoPE backward (gated by env COART_FUSE_ROPE=1).
# No-op when the env is unset, so safe to always import.
from . import fused_rope_patch as _rope_patch  # noqa: F401

# Optional batched-index-select modulation patch (gated by COART_FUSE_MODULATION=1).
# Eliminates the indexing_backward 55%-of-GPU-time bottleneck. No-op when unset.
from . import fused_modulation_patch as _mod_patch  # noqa: F401

# Optional eltwise fusion patch (gated by COART_FUSE_ELTWISE=1).
# Layered on top of fused_modulation: LN bf16-native + addcmul mod/gate.
# Targets the residual eltwise 39% (460 ms) bottleneck. No-op when unset.
from . import eltwise_patch as _elt_patch  # noqa: F401

# DEPRECATED: torch.compile path — dynamic shape × inductor partitioner ×
# trellis2 gradient checkpointing causes min_cut_rematerialization_partition
# to hang for 30+ min per kernel. Kept as a no-op import for tooling that
# may still set COART_COMPILE=1; superseded by eltwise_patch (manual ops).
from . import compile_patch as _compile_patch  # noqa: F401

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
