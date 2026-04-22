"""Exponential-moving-average shadow parameter tracking for training stabilisation.

Matches the semantics of trellis2/trainers/basic.py::BasicTrainer EMA:
    shadow = decay * shadow + (1 - decay) * live_params
updated once per optimizer step, stored in fp32 on the same device as the model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator

import torch
import torch.nn as nn


class EMAModel:
    """Maintains fp32 shadow copies of `model.parameters()`.

    Args:
        model: an `nn.Module`; shadow is created from `model.parameters()` at construction.
        decay: EMA decay rate in (0, 1). Typical value 0.9999 for long runs.

    Thread-safety: not thread-safe. Call `update()` exactly once per optimizer
    step in the same process that owns the model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self._shadow = [p.detach().clone().float() for p in model.parameters()]

    def shadow_params(self) -> Iterator[torch.Tensor]:
        """Iterator over fp32 shadow tensors (live references, not copies)."""
        return iter(self._shadow)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Perform one EMA step against the current online parameters."""
        d = self.decay
        one_minus_d = 1.0 - d
        for s, p in zip(self._shadow, model.parameters()):
            s.mul_(d).add_(p.detach().float(), alpha=one_minus_d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Write shadow values into `model.parameters()` (cast to param dtype)."""
        for s, p in zip(self._shadow, model.parameters()):
            p.data.copy_(s.to(p.dtype))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": [s.clone() for s in self._shadow],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        saved = state["shadow"]
        if len(saved) != len(self._shadow):
            raise ValueError(
                f"EMA shadow count mismatch: saved={len(saved)} current={len(self._shadow)}"
            )
        for dst, src in zip(self._shadow, saved):
            if dst.shape != src.shape:
                raise ValueError(f"EMA shape mismatch: {dst.shape} vs {src.shape}")
            dst.copy_(src)
