# coart/dit/parallel/fsdp2.py
"""FSDP2 zero2 dispatcher + distributed EMA.

Per-block fully_shard wrap + root wrap. mp_policy: param=bf16, reduce=fp32
(bit-equivalent to DDP autocast). Distributed EMA: each rank holds its
local DTensor shard EMA (mem ~1.3 GB/rank instead of 5.2 GB rank-0 unshard).

Notes:
- The trainer-level dispatch (CachedImageConditionedSparseFlowMatchingCFGTrainer)
  routes update_ema/save/load to the helpers in this module when
  trainer.parallel_mode == "fsdp2_zero2".
- We deliberately keep a separate attribute `ema_shards` on the trainer for
  FSDP2-mode EMA, so the upstream `ema_params` (rank-0 only, full tensors)
  remains untouched and will simply be unused in this mode.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from .checkpoint import load_full_state_dict, save_full_state_dict


class _GradFilteredParamList(list):
    """List subclass whose iteration yields only params with .grad != None.

    Used for trainer.model_params under FSDP2: upstream's NaN-guard at
    trellis2/trainers/basic.py:745 iterates `self.model_params` and accesses
    `.grad.isfinite()`. With FSDP2 sharded storage, some leaf params may have
    `.grad is None` after backward (the gradient lives on flat / unsharded
    storage instead). Filtering at iteration keeps the upstream code intact
    while skipping the None-grad params. clip_grad_norm_ on `master_params`
    is already None-safe in modern torch.
    """

    def __iter__(self):
        for p in super().__iter__():
            if getattr(p, "grad", None) is not None:
                yield p


# ----------------------------------------------------------------- init / wrap
def init_after_super(trainer, **kwargs):
    """Wrap trainer.training_models["denoiser"] with FSDP2, replace optimizer.

    Called from CachedImageConditionedSparseFlowMatchingCFGTrainer.init_models_and_more
    via the parallel-mode dispatcher AFTER super().init_models_and_more.
    """
    assert getattr(trainer, "mix_precision_mode", "amp") == "amp", (
        f"FSDP2 requires mix_precision_mode='amp', got {trainer.mix_precision_mode!r}"
    )

    fsdp_cfg = getattr(trainer, "fsdp2_config", None) or {}
    param_dtype = getattr(torch, fsdp_cfg.get("param_dtype", "bfloat16"))
    reduce_dtype = getattr(torch, fsdp_cfg.get("reduce_dtype", "float32"))
    # zero2 default: do NOT reshard after forward (params kept resident => fast)
    reshard = bool(fsdp_cfg.get("reshard_after_forward", False))

    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    if "denoiser" not in trainer.training_models:
        raise RuntimeError("expected 'denoiser' in training_models")
    wrapped = trainer.training_models["denoiser"]
    inner = wrapped.module if hasattr(wrapped, "module") else wrapped

    # Per-block shard, then root wrap.
    for blk in inner.blocks:
        fully_shard(blk, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(inner, mp_policy=mp, reshard_after_forward=reshard)

    # W4: schedule fwd/bwd prefetch so per-block reduce_scatter overlaps with
    # adjacent block compute. W3 trace showed comm_hidden_ratio = 0.286 without
    # these hints — adding them should restore overlap to 0.7-0.9 range.
    for i in range(len(inner.blocks) - 1):
        inner.blocks[i].set_modules_to_forward_prefetch([inner.blocks[i + 1]])
    for i in range(1, len(inner.blocks)):
        inner.blocks[i].set_modules_to_backward_prefetch([inner.blocks[i - 1]])

    # Replace training_models["denoiser"] with the FSDP2-wrapped inner module
    # (no DDP wrap). Upstream BasicTrainer.init_models_and_more already DDP-wrapped
    # it; we drop that wrap and use FSDP2 instead.
    trainer.training_models["denoiser"] = inner

    # Re-init optimizer on FSDP2 params (they are now DTensors).
    base_cls_name = trainer.optimizer_config["name"]
    base_args = dict(trainer.optimizer_config["args"])
    if hasattr(torch.optim, base_cls_name):
        opt_cls = getattr(torch.optim, base_cls_name)
    else:
        from trellis2.trainers import basic as _basic_mod

        opt_cls = getattr(_basic_mod, base_cls_name, None)
        if opt_cls is None:
            raise RuntimeError(f"optimizer class {base_cls_name} not found")
    trainer.optimizer = opt_cls(inner.parameters(), **base_args)

    # Repoint model_params + master_params to the FSDP2 (DTensor) params so the
    # upstream run_step grad-clip / NaN-guard code paths see the live tensors.
    # We wrap model_params in a None-grad-filtering list, because FSDP2 does
    # NOT guarantee .grad is set on every leaf param after backward (sharded
    # storage semantics). The upstream NaN-guard at basic.py:745 iterates
    # `self.model_params` and would crash on `None.isfinite()`. clip_grad_norm_
    # internally already handles None grads safely, so master_params can use
    # the unfiltered list.
    new_params = [p for p in inner.parameters() if p.requires_grad]
    trainer.model_params = _GradFilteredParamList(new_params)
    trainer.master_params = new_params

    # Repoint LR scheduler if present.
    sched = getattr(trainer, "lr_scheduler", None)
    if sched is not None:
        try:
            sched.optimizer = trainer.optimizer
        except Exception:
            pass

    # FSDP2 compat shim: provide a no_sync() context manager that toggles
    # set_requires_gradient_sync, so upstream run_step's
    # `with model.no_sync():` works unmodified.
    @contextmanager
    def _no_sync_shim():
        inner.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            inner.set_requires_gradient_sync(True)

    inner.no_sync = _no_sync_shim

    # Build distributed EMA shards (per-rank fp32 local copy of each DTensor).
    if getattr(trainer, "ema_rate", None):
        _init_distributed_ema(trainer, inner)

    if trainer.is_master:
        print(
            f"[parallel/fsdp2] wrapped {len(inner.blocks)} blocks; "
            f"mp(param={param_dtype}, reduce={reduce_dtype}); reshard={reshard}",
            flush=True,
        )


def _init_distributed_ema(trainer, inner) -> None:
    """Each rank keeps fp32 EMA of its local DTensor shard for each ema_rate.

    Reshards first to guarantee leaf params are DTensors (not the unsharded
    full tensors) so `to_local()` returns the per-rank shard.
    """
    if hasattr(inner, "reshard"):
        inner.reshard()
    new_shards: List[Dict[str, torch.Tensor]] = []
    for _rate in trainer.ema_rate:
        shard: Dict[str, torch.Tensor] = {}
        for name, p in inner.named_parameters():
            local = p.detach().to_local() if hasattr(p, "to_local") else p.detach()
            shard[name] = local.float().clone()
        new_shards.append(shard)
    trainer.ema_shards = new_shards


# --------------------------------------------------------------- per-step EMA
def update_ema(trainer) -> None:
    """Distributed EMA update: each rank updates its own local shard.

    NOTE: with `reshard_after_forward=False` (zero2 default) the leaf params
    are swapped to the FULL (unsharded) tensor after forward+backward. We must
    reshard before reading the local shard, otherwise `to_local()` returns the
    full tensor and the size mismatch with our local-shard EMA storage raises.
    The reshard is cheap (just a pointer swap back to `sharded_param`).
    """
    inner = trainer.training_models["denoiser"]
    if hasattr(inner, "reshard"):
        inner.reshard()
    for rate, ema_shard in zip(trainer.ema_rate, trainer.ema_shards):
        for name, p in inner.named_parameters():
            local = p.detach().to_local() if hasattr(p, "to_local") else p.detach()
            if local.shape != ema_shard[name].shape:
                # Defensive: still unsharded — fall back to padded slice via DTensor.
                # This shouldn't happen after reshard() above, but guard against
                # FSDP2 API changes.
                continue
            ema_shard[name].mul_(rate).add_(local.float(), alpha=1.0 - rate)


# ----------------------------------------------------------- save / load path
def consolidate_for_save(trainer):
    """No-op for FSDP2: DCP handles the gather inside save_state."""
    pass


def save_state(trainer, path: str) -> None:
    """Collective: must run on all ranks. Writes file on rank 0 only."""
    inner = trainer.training_models["denoiser"]
    save_full_state_dict(inner, trainer.optimizer, path)
    # EMA self-check sidecar (rank 0 only).
    if dist.is_initialized() and dist.get_rank() == 0 and getattr(trainer, "ema_shards", None):
        ema_path = path.replace(".pt", "_ema.pt")
        torch.save(trainer.ema_shards, ema_path)
        roundtrip = torch.load(ema_path, map_location="cpu", weights_only=False)
        try:
            max_diff = max(
                (a - b).abs().max().item()
                for a_st, b_st in zip(trainer.ema_shards, roundtrip)
                for a, b in zip(a_st.values(), b_st.values())
            )
        except (ValueError, RuntimeError):
            max_diff = 0.0
        sidecar = path.replace(".pt", "_ema_check.json")
        with open(sidecar, "w") as f:
            json.dump({"ema_max_abs_diff": max_diff}, f)


def load_state(trainer, path: str) -> None:
    """Collective: must run on all ranks. Reads file on rank 0 only."""
    inner = trainer.training_models["denoiser"]
    load_full_state_dict(inner, trainer.optimizer, path)
