# coart/dit/parallel/__init__.py
"""coart.dit.parallel - parallelism mode dispatch.

Modes: "ddp" (default), "zro1", "fsdp2_zero2".

Each mode is a module exposing:
- init_after_super(trainer, **kwargs) - called after super().init_models_and_more
- consolidate_for_save(trainer) - optional; called before trainer.save() writes
"""

VALID_MODES = ("ddp", "zro1", "fsdp2_zero2")


def get_dispatcher(mode: str):
    if mode == "ddp":
        from . import ddp as _m
    elif mode == "zro1":
        from . import zro1 as _m
    elif mode == "fsdp2_zero2":
        from . import fsdp2 as _m  # not yet created in W2.1; will be added in W2.2
    else:
        raise ValueError(f"unknown parallel_mode={mode!r}; valid: {VALID_MODES}")
    return _m
