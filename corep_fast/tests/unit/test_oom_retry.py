"""Tests for the OOM retry wrapper introduced on 2026-04-21."""
import os
import importlib
import pytest
import torch


def _set_flag(monkeypatch, on: bool):
    monkeypatch.setenv("COREP_FAST_VRAM_RESCUE", "1" if on else "0")
    import corep_fast.config as cfg
    importlib.reload(cfg)


def test_retry_succeeds_on_second_attempt(monkeypatch):
    """If the wrapped callable OOMs once then succeeds, the wrapper returns
    the success value and halves the chunk-budget env var in between."""
    _set_flag(monkeypatch, on=True)
    from corep_fast.pipeline import _retry_with_shrinking_budget

    call_count = {"n": 0}
    budgets_seen = []

    def flaky_stage(batch):
        call_count["n"] += 1
        budgets_seen.append(os.environ.get(
            "COREP_FAST_S4_UTURN_CHUNK_ELEMS", "default"))
        if call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return "ok"

    monkeypatch.setenv("COREP_FAST_S4_UTURN_CHUNK_ELEMS", "400000000")
    result = _retry_with_shrinking_budget(flaky_stage, batch=None)
    assert result == "ok"
    assert call_count["n"] == 2, f"expected exactly one retry; got {call_count['n']} calls"
    # First attempt saw the original budget; second saw it reduced 4x.
    assert budgets_seen == ["400000000", "100000000"]


def test_retry_raises_after_limit(monkeypatch):
    """Third OOM propagates."""
    _set_flag(monkeypatch, on=True)
    from corep_fast.pipeline import _retry_with_shrinking_budget

    def always_oom(batch):
        raise torch.cuda.OutOfMemoryError("always")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _retry_with_shrinking_budget(always_oom, batch=None)


def test_retry_disabled_when_flag_off(monkeypatch):
    """With VRAM_RESCUE=0, the wrapper must be a direct passthrough."""
    _set_flag(monkeypatch, on=False)
    from corep_fast.pipeline import _retry_with_shrinking_budget

    def failing(batch):
        raise torch.cuda.OutOfMemoryError("no retry wanted")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _retry_with_shrinking_budget(failing, batch=None)
