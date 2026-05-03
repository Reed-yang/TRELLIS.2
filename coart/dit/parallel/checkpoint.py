# coart/dit/parallel/checkpoint.py
"""Distributed Checkpoint (DCP) helpers for FSDP2.

Used by parallel/fsdp2.py to save/load model + optimizer state in a way that
is compatible with both DDP-trained and FSDP2-trained checkpoints (the
broadcast_from_rank0 path handles dim-1 sharding rebalance).
"""
from __future__ import annotations
import os

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


def save_full_state_dict(model, optimizer, path: str) -> None:
    """Gather full state to rank 0 and torch.save. CPU-offloaded for low mem.

    Collective; must be called from every rank.
    """
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    msd = get_model_state_dict(model, options=opts)
    osd = get_optimizer_state_dict(model, optimizer, options=opts)
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": msd, "optim": osd}, path)
    if dist.is_initialized():
        dist.barrier()


def load_full_state_dict(model, optimizer, path: str) -> None:
    """Load + broadcast full state from rank 0; resharding handled by DCP.

    Collective; must be called from every rank. Path is read on rank 0 only.

    Accepts two on-disk formats and dispatches accordingly:
    1. DCP-style: ``{"model": <state_dict>, "optim": <opt_state_dict>}`` — the
       format that ``save_full_state_dict`` writes.
    2. Plain state_dict: ``OrderedDict`` of param-name → tensor (the legacy
       per-name file format that upstream BasicTrainer.save writes for DDP /
       ZRO runs). Optimizer state is absent in this format and skipped.

    The optimizer load is also skipped on the DCP path when ``optim`` is empty
    (e.g., a fresh ckpt that hasn't taken any steps yet).
    """
    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    is_dcp_format = None
    if dist.get_rank() == 0:
        full = torch.load(path, map_location="cpu", weights_only=False)
        is_dcp_format = (
            isinstance(full, dict) and "model" in full and "optim" in full
        )
    else:
        full = None
    # Broadcast format-flag so non-master ranks know which set_*_state_dict
    # collectives to participate in.
    flag = [is_dcp_format] if dist.get_rank() == 0 else [None]
    if dist.is_initialized():
        dist.broadcast_object_list(flag, src=0)
    is_dcp_format = bool(flag[0])

    if is_dcp_format:
        msd = full["model"] if full is not None else None
        osd = full["optim"] if full is not None else None
        set_model_state_dict(model, msd, options=opts)
        if osd:
            set_optimizer_state_dict(model, optimizer, osd, options=opts)
    else:
        # Legacy plain state_dict — model only.
        msd = full if full is not None else None
        set_model_state_dict(model, msd, options=opts)

    if dist.is_initialized():
        dist.barrier()
