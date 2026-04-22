"""Atomic checkpoint I/O with rolling-K eviction.

Naming convention:
    {prefix}_step{step:07d}.pt

Typical prefixes used by coart.vae.train:
    ckpt        — encoder + decoder + optimizer state
    ema_<rate>  — EMA shadow parameters (one file per rate)
    misc        — sampler epoch, RNG state, step counter, etc.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch


_STEP_PATTERN = re.compile(r"_step(\d{7})\.pt$")


def atomic_save(obj: Any, path: str) -> None:
    """Write `torch.save(obj, path)` atomically via tmp + os.replace.

    If the process is killed mid-write, `path` either contains the previous
    contents (or does not exist) — never a torn half-written file.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _list_ckpts(output_dir: str, prefix: str) -> list[Tuple[str, int]]:
    """Return [(path, step), ...] sorted ascending by step."""
    pattern = os.path.join(output_dir, f"{prefix}_step*.pt")
    out: list[Tuple[str, int]] = []
    for p in glob.glob(pattern):
        m = _STEP_PATTERN.search(os.path.basename(p))
        if m:
            out.append((p, int(m.group(1))))
    out.sort(key=lambda t: t[1])
    return out


def save_ckpt(
    state: Dict[str, Any],
    output_dir: str,
    step: int,
    keep_k: int = 3,
    prefix: str = "ckpt",
) -> str:
    """Atomically save `state` to `<output_dir>/{prefix}_step{step:07d}.pt`.

    After saving, prune older files with the same prefix so only the newest
    `keep_k` remain on disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, f"{prefix}_step{step:07d}.pt")
    atomic_save(state, target)

    existing = _list_ckpts(output_dir, prefix)
    if len(existing) > keep_k:
        to_remove = existing[: len(existing) - keep_k]
        for path, _ in to_remove:
            try:
                os.remove(path)
            except OSError:
                pass
    return target


def find_latest_ckpt(
    output_dir: str,
    prefix: str = "ckpt",
) -> Optional[Tuple[str, int]]:
    """Return (path, step) of the highest-step ckpt with `prefix`, or None."""
    existing = _list_ckpts(output_dir, prefix)
    if not existing:
        return None
    return existing[-1]
