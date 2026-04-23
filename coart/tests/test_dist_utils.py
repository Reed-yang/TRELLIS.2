"""Smoke test that wrap_ddp builds DDP with the expected flags."""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
import torch.nn as nn


def _make_linear():
    return nn.Linear(8, 8).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrap_ddp_flags_applied():
    """wrap_ddp must pass gradient_as_bucket_view=True and broadcast_buffers=False."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    try:
        from coart.common.dist_utils import wrap_ddp

        with mock.patch(
            "coart.common.dist_utils.DDP",
            wraps=__import__(
                "torch.nn.parallel", fromlist=["DistributedDataParallel"],
            ).DistributedDataParallel,
        ) as spy:
            try:
                wrap_ddp(_make_linear(), local_rank=0)
            except Exception:
                pass  # DDP may fail to fully init under gloo+cuda; spy still records kwargs
            assert spy.called, "DDP constructor was not invoked"
            kwargs = spy.call_args.kwargs
            assert kwargs.get("gradient_as_bucket_view") is True
            assert kwargs.get("broadcast_buffers") is False
            assert kwargs.get("bucket_cap_mb") == 128
            assert kwargs.get("find_unused_parameters") is False
    finally:
        # Destroy process group so subsequent tests don't observe gloo state
        # (ReduceOp.AVG is unsupported on gloo → breaks logger.flush_if_due).
        if dist.is_initialized():
            dist.destroy_process_group()
