"""Unit tests for coart.common.checkpoint.

Covers atomic_save (no torn writes), save_ckpt + rolling-K eviction, and
find_latest_ckpt (glob latest + metadata parsing).
"""
import os

import pytest
import torch

from coart.common.checkpoint import atomic_save, save_ckpt, find_latest_ckpt


def test_atomic_save_writes_target_file(tmp_path):
    target = tmp_path / "obj.pt"
    atomic_save({"x": 42}, str(target))
    assert target.exists()
    loaded = torch.load(target)
    assert loaded["x"] == 42


def test_atomic_save_no_tmp_leak(tmp_path):
    target = tmp_path / "obj.pt"
    atomic_save({"x": 42}, str(target))
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"unexpected tmp leftovers: {leftovers}"


def test_save_ckpt_creates_step_file(tmp_path):
    state = {"step": 100, "model": {"w": torch.zeros(3)}}
    save_ckpt(state, str(tmp_path), step=100, keep_k=3, prefix="ckpt")
    files = sorted(tmp_path.glob("ckpt_step*.pt"))
    assert len(files) == 1
    assert files[0].name == "ckpt_step0000100.pt"


def test_save_ckpt_rolling_eviction(tmp_path):
    for step in [100, 200, 300, 400, 500]:
        save_ckpt({"step": step}, str(tmp_path), step=step, keep_k=3, prefix="ckpt")
    files = sorted(tmp_path.glob("ckpt_step*.pt"))
    assert len(files) == 3
    steps = [int(f.name[len("ckpt_step"):len("ckpt_step") + 7]) for f in files]
    assert steps == [300, 400, 500], f"expected [300,400,500] after rolling, got {steps}"


def test_find_latest_ckpt_returns_none_when_empty(tmp_path):
    assert find_latest_ckpt(str(tmp_path), prefix="ckpt") is None


def test_find_latest_ckpt_returns_max_step(tmp_path):
    for step in [50, 150, 250]:
        save_ckpt({"step": step}, str(tmp_path), step=step, keep_k=10, prefix="ckpt")
    path, step = find_latest_ckpt(str(tmp_path), prefix="ckpt")
    assert step == 250
    assert path.endswith("ckpt_step0000250.pt")


def test_save_ckpt_preserves_multiple_prefixes(tmp_path):
    """EMA and misc ckpts share dir but use different prefixes; rolling is per-prefix."""
    save_ckpt({"step": 100}, str(tmp_path), step=100, keep_k=2, prefix="ckpt")
    save_ckpt({"step": 100}, str(tmp_path), step=100, keep_k=2, prefix="ema_0.9999")
    save_ckpt({"step": 100}, str(tmp_path), step=100, keep_k=2, prefix="misc")
    assert (tmp_path / "ckpt_step0000100.pt").exists()
    assert (tmp_path / "ema_0.9999_step0000100.pt").exists()
    assert (tmp_path / "misc_step0000100.pt").exists()
