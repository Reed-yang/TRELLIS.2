"""Sanity test for W1 C2: PyTorch's AdamW(fused=True) initializes on H100.

This is a platform sanity check — it does not directly exercise our trainer.
The real wiring is in coart/dit/configs/coart_dit_shape_512_ft.json
(trainer.args.optimizer.args.fused = true), which flows through
trellis2 BasicTrainer's optimizer construction unchanged.

Note: PyTorch enforces fused XOR foreach (RuntimeError if both True).
We pick fused=True only — it implies a fused CUDA kernel that subsumes
the foreach perf benefit on H100.
"""
from __future__ import annotations

import torch
from torch.optim import AdamW


def test_adamw_fused_supported_h100():
    # Sanity: PyTorch 2.6 supports fused=True on H100 sm_90a.
    if not torch.cuda.is_available():
        return
    p = torch.nn.Parameter(torch.zeros(8, 8, device="cuda"))
    opt = AdamW([p], lr=1e-3, fused=True)
    p.grad = torch.ones_like(p)
    opt.step()  # should not raise
