"""Distributed training helpers for torchrun + DDP."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_dist():
    """Initialise torch.distributed if launched via torchrun.

    Returns (rank, world_size, local_rank, is_dist).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def unwrap(m):
    """Return underlying module if wrapped in DDP."""
    return m.module if isinstance(m, DDP) else m


def wrap_ddp(model, local_rank):
    """Wrap model in DDP with canonical settings.

    - bucket_cap_mb=128: coalesce grad all-reduce buckets at 128MB
    - find_unused_parameters=False: all params receive grad every step
    - gradient_as_bucket_view=True: bucket zero-copy (saves one grad alloc per step)
    - broadcast_buffers=False: model has no BN, skip per-step buffer sync
    """
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=128,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )


def worker_init_fn(worker_id: int):
    """Independent RNG per (rank, worker) so augmentation is not duplicated.

    Reads `RANK` and `_FT_NUM_WORKERS` from env. Callers must set
    `_FT_NUM_WORKERS` before DataLoader construction.
    """
    rank = int(os.environ.get("RANK", 0))
    num_workers = int(os.environ.get("_FT_NUM_WORKERS", 1))
    seed = (rank * max(num_workers, 1) + worker_id) * 9973 + 17
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)
