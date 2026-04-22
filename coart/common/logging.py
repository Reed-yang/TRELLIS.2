"""TensorBoard logging helpers with cadence control and DDP all-reduce.

Pattern:
    logger = CoartTBLogger(output_dir, is_master=(rank==0))
    logger.scalar("loss/total", float_value, step)  # buffered, written at i_log cadence
    logger.flush_if_due(step, i_log=100)
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


class CoartTBLogger:
    """Rank-0 TB writer with per-step scalar buffer and DDP-aware all-reduce.

    Non-master ranks accept scalar calls (no-op) to avoid caller branching.
    `flush_if_due` performs an all-reduce across all ranks over the buffered
    scalars and writes rank-0's averaged values to TB at cadence `i_log`.
    """

    def __init__(self, output_dir: str, is_master: bool):
        self.is_master = is_master
        self._buf: Dict[str, list[float]] = {}
        self._writer: Optional[SummaryWriter] = None
        if is_master:
            os.makedirs(os.path.join(output_dir, "tb_logs"), exist_ok=True)
            self._writer = SummaryWriter(os.path.join(output_dir, "tb_logs"))

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Buffer a scalar for this step. Actual write happens on flush_if_due."""
        self._buf.setdefault(tag, []).append(float(value))

    def flush_if_due(self, step: int, i_log: int) -> None:
        """If step % i_log == 0, all-reduce across ranks and write averaged scalars."""
        if step % i_log != 0:
            return
        if not self._buf:
            return

        tags = sorted(self._buf.keys())
        vals = torch.tensor(
            [sum(self._buf[t]) / len(self._buf[t]) for t in tags],
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)

        if self.is_master and self._writer is not None:
            for t, v in zip(tags, vals.tolist()):
                self._writer.add_scalar(t, v, step)

        self._buf.clear()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
