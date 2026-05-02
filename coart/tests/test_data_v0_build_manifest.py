"""Tests for scripts/coart_data_v0/build_manifest.py."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module(name: str):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _make_synthetic_layout(root: pathlib.Path, sha_states: dict[str, dict]):
    """sha_states[sha] = {'render': bool, 'dino': bool, 'slat_tags': list[str]}.
    Layout matches spec §4.1.
    """
    instances = []
    (root / "renders_cond").mkdir(parents=True, exist_ok=True)
    (root / "dino_l16_s512").mkdir(parents=True, exist_ok=True)
    (root / "slat").mkdir(parents=True, exist_ok=True)
    for sha, st in sha_states.items():
        instances.append({"sha256": sha, "aesthetic_score": 5.0,
                          "file_identifier": "x", "local_path": "x",
                          "feat18_npz_size_bytes": 1000, "captions": ""})
        if st["render"]:
            d = root / "renders_cond" / sha; d.mkdir(parents=True, exist_ok=True)
            for v in range(16):
                (d / f"{v:03d}.png").write_bytes(b"\x89PNG")
            (d / "transforms.json").write_text(
                json.dumps({"frames": [{"file_path": f"{v:03d}.png",
                                       "yaw": 0, "pitch": 0, "radius": 2,
                                       "fov": 30, "transform_matrix": [[1]*4]*4}
                                       for v in range(16)]}))
        if st["dino"]:
            np.savez_compressed(
                root / "dino_l16_s512" / f"{sha}.npz",
                features=np.zeros((16, 1029, 1024), dtype=np.float16),
                view_idx=np.arange(16, dtype=np.uint8),
                n_tokens=np.int32(1029), model_id="x", image_size=np.int32(512),
            )
        for tag in st["slat_tags"]:
            (root / "slat" / tag).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                root / "slat" / tag / f"{sha}.npz",
                coords=np.zeros((123, 3), dtype=np.int16),
                feats=np.zeros((123, 32), dtype=np.float16),
                num_voxels=np.int32(123), vae_ckpt_rel="x", vae_io_arch="three_branch",
            )
    pd.DataFrame(instances).to_csv(root / "instances.csv", index=False)


def test_full_partial_missing(tmp_path: pathlib.Path):
    bm = _load_module("build_manifest")

    sha_a = "a" * 64; sha_b = "b" * 64; sha_c = "c" * 64; sha_d = "d" * 64
    states = {
        sha_a: {"render": True,  "dino": True,  "slat_tags": ["v1"]},
        sha_b: {"render": True,  "dino": False, "slat_tags": []},
        sha_c: {"render": False, "dino": False, "slat_tags": []},
        sha_d: {"render": True,  "dino": True,  "slat_tags": ["v0", "v1"]},
    }
    _make_synthetic_layout(tmp_path, states)

    out = tmp_path / "manifest.csv"
    bm.run(instances=str(tmp_path / "instances.csv"),
           renders_dir=str(tmp_path / "renders_cond"),
           dino_dir=str(tmp_path / "dino_l16_s512"),
           slat_root=str(tmp_path / "slat"),
           out=str(out))

    m = pd.read_csv(out, dtype={"sha256": str}).set_index("sha256")
    assert m.loc[sha_a, "render_done"] and m.loc[sha_a, "dino_done"] and m.loc[sha_a, "slat_done"]
    assert m.loc[sha_a, "slat_tag"] == "v1"
    assert m.loc[sha_a, "num_voxels"] == 123

    assert m.loc[sha_b, "render_done"] and not m.loc[sha_b, "dino_done"]
    assert not m.loc[sha_b, "slat_done"]

    assert not m.loc[sha_c, "render_done"]
    assert not m.loc[sha_c, "dino_done"]

    assert m.loc[sha_d, "slat_tag"] == "v1"
