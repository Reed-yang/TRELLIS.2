# coart/dit/parallel/zro1.py
"""ZRO-1 - wrap optimizer with ZeroRedundancyOptimizer.

Saves ~10 GB/rank optimizer state. Step time ~0%.
"""
from torch.distributed.optim import ZeroRedundancyOptimizer


def init_after_super(trainer, **kwargs):
    """Replace trainer.optimizer with ZeroRedundancyOptimizer.

    trellis2 BasicTrainer stores params in `self.master_params` (a list) and
    builds the optimizer from it. We wrap that same parameter list so the
    bucketing matches DDP's grad bucket layout.
    """
    base_optimizer = trainer.optimizer
    base_cls = type(base_optimizer)
    base_kwargs = dict(base_optimizer.defaults)

    params = _gather_params(trainer)
    trainer.optimizer = ZeroRedundancyOptimizer(
        params,
        optimizer_class=base_cls,
        parameters_as_bucket_view=True,
        **base_kwargs,
    )

    # If a LR scheduler was attached to the original optimizer, repoint it.
    sched = getattr(trainer, "lr_scheduler", None)
    if sched is not None:
        try:
            sched.optimizer = trainer.optimizer
        except Exception:
            pass

    # Free the base optimizer's per-rank state (no steps have run yet, but
    # the empty state dict still holds the param-group ref). del to make GC easy.
    del base_optimizer

    if trainer.is_master:
        print(
            f"[parallel/zro1] wrapped {base_cls.__name__} with "
            f"ZeroRedundancyOptimizer",
            flush=True,
        )


def _gather_params(trainer):
    """Pull the same parameter list BasicTrainer used to build the optimizer.

    Prefer `master_params` (matches inflat_all path too); fall back to
    concatenating training_models parameters if absent (defensive).
    """
    mp = getattr(trainer, "master_params", None)
    if mp is not None:
        return mp
    params = []
    for m in trainer.training_models.values():
        params.extend(p for p in m.parameters() if p.requires_grad)
    return params


def consolidate_for_save(trainer):
    """ZRO requires consolidate_state_dict before .state_dict() returns full state.

    Must be called on ALL ranks (it's a collective). Caller is responsible
    for invoking this from a code path executed by every rank.
    """
    trainer.optimizer.consolidate_state_dict(to=0)
