"""Tests for scripts/coart_data_v0/pick_instances.py.

Builds tiny synthetic metadata + feat18 dir, asserts filter + sample logic.
"""
import importlib.util
import os
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module(name: str):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_filter_and_sample(tmp_path: pathlib.Path):
    pi = _load_module("pick_instances")

    rows = []
    for i in range(50):
        rows.append({
            "sha256": f"{i:064x}",
            "file_identifier": f"https://example.com/{i}",
            "aesthetic_score": 5.5 if i < 30 else 3.0,
            "captions": f'["caption {i}"]',
        })
    meta_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)

    raw_meta = tmp_path / "raw_metadata.csv"
    raw_rows = [{"sha256": f"{i:064x}", "local_path": f"raw/x/{i}.glb"}
                for i in range(50) if i != 5]
    pd.DataFrame(raw_rows).to_csv(raw_meta, index=False)

    feat_dir = tmp_path / "feat18"
    feat_dir.mkdir()
    for i in range(40):
        (feat_dir / f"{i:064x}.npz").write_bytes(b"\x00")

    out_csv = tmp_path / "instances.csv"
    pi.run(
        metadata_csv=str(meta_csv),
        raw_metadata_csv=str(raw_meta),
        feat18_dir=str(feat_dir),
        aesthetic_min=4.5,
        n=10,
        seed=0,
        out=str(out_csv),
    )

    out = pd.read_csv(out_csv)
    assert len(out) == 10
    for sha in out["sha256"].astype(str):
        idx = int(sha, 16)
        assert idx < 30, f"sha {sha} has aesthetic < 4.5"
        assert idx < 40, f"sha {sha} has no feat18"
        assert idx != 5, f"sha {sha} has no local_path"

    out_csv2 = tmp_path / "instances2.csv"
    pi.run(metadata_csv=str(meta_csv), raw_metadata_csv=str(raw_meta),
           feat18_dir=str(feat_dir), aesthetic_min=4.5, n=10, seed=0,
           out=str(out_csv2))
    pd.testing.assert_frame_equal(pd.read_csv(out_csv), pd.read_csv(out_csv2))


def test_n_exceeds_available(tmp_path: pathlib.Path):
    pi = _load_module("pick_instances")
    meta_csv = tmp_path / "metadata.csv"
    pd.DataFrame([{"sha256": f"{i:064x}", "file_identifier": "x",
                   "aesthetic_score": 5.0, "captions": ""} for i in range(5)]).to_csv(meta_csv, index=False)
    raw_meta = tmp_path / "raw_metadata.csv"
    pd.DataFrame([{"sha256": f"{i:064x}", "local_path": "x"} for i in range(5)]).to_csv(raw_meta, index=False)
    feat_dir = tmp_path / "feat18"; feat_dir.mkdir()
    for i in range(5):
        (feat_dir / f"{i:064x}.npz").write_bytes(b"\x00")

    with pytest.raises(ValueError, match="only 5 candidates"):
        pi.run(metadata_csv=str(meta_csv), raw_metadata_csv=str(raw_meta),
               feat18_dir=str(feat_dir), aesthetic_min=4.5, n=10, seed=0,
               out=str(tmp_path / "out.csv"))
