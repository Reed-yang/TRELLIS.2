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


def test_corep_encode_wraps_stages():
    """Smoke: assert the retry wrapper is wired in at the corep_encode and
    mesh_to_param call sites by grep-checking the source. Runtime OOM
    injection at the pipeline level requires a real CUDA context and a
    full mesh; the unit suite at Task 6 covers the helper's behaviour.

    Post-wrap, the original ``s4_face_point(batch, ...)`` / ``s6_collapse(...)``
    / ``s7_rank_assign(...)`` literal call sites are replaced by passing the
    stage fn as the first arg to ``_retry_with_shrinking_budget``. The check
    below verifies (a) each stage name still appears as a *reference* in
    pipeline.py, (b) each reference is preceded by ``_retry_with_shrinking_budget(``,
    and (c) the helper is invoked at least once per stage.
    """
    import re
    from pathlib import Path
    src = Path("corep_fast/pipeline.py").read_text()
    assert "_retry_with_shrinking_budget" in src, (
        "_retry_with_shrinking_budget must be invoked in pipeline.py")
    # Each wrapped call looks like:
    #   _retry_with_shrinking_budget(\n            s4_face_point, batch, ...
    pattern = re.compile(
        r"_retry_with_shrinking_budget\s*\(\s*(s4_face_point|s6_collapse|s7_rank_assign)\b")
    wrapped = set(m.group(1) for m in pattern.finditer(src))
    for stage in ("s4_face_point", "s6_collapse", "s7_rank_assign"):
        assert stage in wrapped, (
            f"{stage} must be routed through _retry_with_shrinking_budget "
            f"in pipeline.py; wrapped stages found: {wrapped}")
