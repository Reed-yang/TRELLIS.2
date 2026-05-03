# coart/dit/parallel/ddp.py
"""DDP - default parallel mode.

Applies C4 (gradient_as_bucket_view=True) to all DDP-wrapped training_models.
This logic was previously inline in trainer.init_models_and_more; W2.1
refactor moves it here as part of the parallel-mode dispatch architecture.
"""
import torch.nn.parallel as _tnp


def init_after_super(trainer, **kwargs):
    for name, m in getattr(trainer, "training_models", {}).items():
        if isinstance(m, _tnp.DistributedDataParallel):
            try:
                m.gradient_as_bucket_view = True
                if trainer.is_master:
                    print(
                        f"[parallel/ddp] gradient_as_bucket_view=True "
                        f"set on DDP({name})",
                        flush=True,
                    )
            except Exception as e:
                if trainer.is_master:
                    print(
                        f"[parallel/ddp] warn: cannot set "
                        f"gradient_as_bucket_view on {name} ({e})",
                        flush=True,
                    )


def consolidate_for_save(trainer):
    pass  # no-op; full state already on each rank
