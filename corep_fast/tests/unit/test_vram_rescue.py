"""Tests for the 2026-04-21 VRAM rescue quick-wins (QW1/QW3/QW5/QW6)."""
import os, importlib
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VRAM rescue tests require CUDA",
)


def _set_flag(monkeypatch, on: bool):
    monkeypatch.setenv("COREP_FAST_VRAM_RESCUE", "1" if on else "0")
    import corep_fast.config as cfg
    importlib.reload(cfg)


def _build_synthetic_group(G=4, P=256, seed=0):
    """Build synthetic input for _count_uturns_from_packed with predictable P."""
    rng = np.random.default_rng(seed)
    pts = rng.random((G, P, 3)).astype(np.float64)
    pts_valid = np.ones((G, P), dtype=bool)
    fv = rng.random((G, 3, 3)).astype(np.float64)
    ed = np.zeros((G, 3), dtype=np.int64)
    return pts, pts_valid, fv, ed


def test_qw1_frees_cdist_intermediates(monkeypatch):
    """With VRAM_RESCUE=1, peak VRAM during _count_uturns_from_packed drops
    by at least half of the d-tensor (G*P*P*8 bytes) footprint."""
    from corep_fast.stages.s4_face_point import _count_uturns_from_packed
    dev = torch.device("cuda")
    pts, pv, fv, ed = _build_synthetic_group(G=8, P=512)
    t_pts = torch.from_numpy(pts).to(dev)
    t_pv = torch.from_numpy(pv).to(dev)
    t_fv = torch.from_numpy(fv).to(dev)
    t_ed = torch.from_numpy(ed).to(dev)

    # Baseline (flag off)
    _set_flag(monkeypatch, on=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = _count_uturns_from_packed(t_pts, t_pv, t_fv, t_ed)
    peak_off = torch.cuda.max_memory_allocated()

    # With QW1 (flag on)
    _set_flag(monkeypatch, on=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = _count_uturns_from_packed(t_pts, t_pv, t_fv, t_ed)
    peak_on = torch.cuda.max_memory_allocated()

    G, P = 8, 512
    saving_bytes_target = int(0.5 * (G * P * P * 8))
    assert peak_off - peak_on >= saving_bytes_target, (
        f"QW1 saved only {peak_off - peak_on:,} B; expected >= {saving_bytes_target:,} B"
    )
