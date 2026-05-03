"""Resolution semantics of ``_build_output_dir``.

Regression test for the date-drift bug: before this fix, running
``--resume_from latest`` on a day different from the original training
start silently created a NEW directory under today's date, found no
ckpt there, fell back to base mode, and forked the wandb run. The
contract now is:

  * explicit ``--output_dir`` wins.
  * ``--resume_from != "none"`` with no explicit ``--output_dir`` must
    resolve to an existing dir that contains real checkpoints, else
    raise. Most-recent wins when multiple match.
  * base mode (``--resume_from=="none"``) uses today's date.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from coart.vae.config import _build_output_dir


def _mk_run(root: Path, name: str, *, with_ckpt: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if with_ckpt:
        (d / "ckpt_step0000001.pt").write_bytes(b"x")
    return d


def test_explicit_output_dir_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    explicit = str(tmp_path / "my_custom")
    assert _build_output_dir(explicit, "v1", "latest") == explicit
    assert _build_output_dir(explicit, "v1", "none") == explicit


def test_resume_matches_existing_with_ckpt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = _mk_run(tmp_path / "results", "coart_feat18_20260101_v1")
    got = _build_output_dir(None, "v1", "latest")
    assert os.path.abspath(got) == os.path.abspath(str(expected))


def test_resume_rejects_ckptless_dir(tmp_path, monkeypatch):
    """A bare dir without ckpt_step*.pt must NOT be auto-selected."""
    monkeypatch.chdir(tmp_path)
    _mk_run(tmp_path / "results", "coart_feat18_20260101_v1", with_ckpt=False)
    with pytest.raises(RuntimeError, match="no results"):
        _build_output_dir(None, "v1", "latest")


def test_resume_picks_most_recent_when_multiple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    older = _mk_run(tmp_path / "results", "coart_feat18_20260101_v1")
    newer = _mk_run(tmp_path / "results", "coart_feat18_20260424_v1")
    # Force older to have an earlier mtime.
    os.utime(older, (0, 0))
    got = _build_output_dir(None, "v1", "latest")
    assert os.path.abspath(got) == os.path.abspath(str(newer))


def test_resume_missing_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    with pytest.raises(RuntimeError, match="no results"):
        _build_output_dir(None, "ghost_tag", "latest")


def test_base_mode_uses_today(tmp_path, monkeypatch):
    """--resume_from=none must NOT accidentally match an existing dir."""
    monkeypatch.chdir(tmp_path)
    _mk_run(tmp_path / "results", "coart_feat18_20260101_v1")
    got = _build_output_dir(None, "v1", "none")
    # Contains today's date, NOT the old one.
    assert "20260101" not in got
    assert got.startswith(os.path.join("results", "coart_feat18_"))
    assert got.endswith("_v1")


def test_resume_unaffected_by_other_run_tags(tmp_path, monkeypatch):
    """Matching is strict: run_tag boundary must be exact."""
    monkeypatch.chdir(tmp_path)
    _mk_run(tmp_path / "results", "coart_feat18_20260101_v1_extra")
    with pytest.raises(RuntimeError, match="no results"):
        _build_output_dir(None, "v1", "latest")
