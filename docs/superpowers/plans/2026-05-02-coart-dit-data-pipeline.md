# Coart DiT Data Pipeline (v0, 10K-asset milestone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ≤10 h 内为 10K assets 生产 renders + DINO cache + SLat cache + manifest，符合 spec `docs/superpowers/specs/2026-05-02-coart-dit-data-pipeline-design.md` 中冻结的 schema 契约，且 SS-flow IoU pre-flight gate 通过。

**Architecture:** 5 个新脚本放在 `scripts/coart_data_v0/`（pick_instances, coart_cache_dino, coart_cache_slat, build_manifest, run.sh）+ 1 个 SS-flow IoU 验证脚本 `scripts/coart_compare_occupancy.py`。三 stage 独立：render 复用 `data_toolkit/render_cond.py` 零修改；dino/slat 是 ~150 LOC 单文件 GPU 脚本。Sharding 用 `int(sha[:8], 16) % world_size` 稳定哈希；写文件用 `os.replace` atomic rename。Tests 放在 `coart/tests/test_data_v0_*.py`。

**Tech Stack:** Python 3.10+, PyTorch + bf16 autocast, HuggingFace `transformers` (DinoV3 ViT-L/16), `pandas`, `numpy`, Blender 3.0.1 CYCLES (via 现成 data_toolkit), pytest, `o_voxel` + `corep_fast` (for IoU validation).

**Repo branch:** 当前 `vae-finetune` 分支（所有改动均为新增文件，与现有 vae 训练不冲突）。每个 task 一次提交，commit message 用英文 Conventional Commits 风格。

**File map (all new):**
```
scripts/coart_data_v0/
├── README.md
├── run.sh
├── pick_instances.py
├── coart_cache_dino.py
├── coart_cache_slat.py
└── build_manifest.py
scripts/coart_compare_occupancy.py
coart/tests/
├── test_data_v0_pick_instances.py
├── test_data_v0_atomic_write.py
├── test_data_v0_stable_shard.py
├── test_data_v0_build_manifest.py
├── test_data_v0_cache_dino_shape.py
├── test_data_v0_cache_slat_shape.py
└── test_data_v0_compare_occupancy.py
```

**Shared design conventions** (apply to all stage scripts):
- Sharding helper（每个脚本顶部 inline 一份，DRY OK 因每个 ≤10 行）：
  ```python
  def stable_shard(sha: str, world_size: int, rank: int) -> bool:
      """Return True if this rank should process `sha`."""
      return (int(sha[:8], 16) % world_size) == rank
  ```
- Atomic save helper:
  ```python
  def atomic_savez(out_path: str, **arrays):
      tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time()*1e6)}"
      np.savez_compressed(tmp, **arrays)
      os.replace(tmp, out_path)  # atomic on same fs incl. NFS
  ```
- 每个脚本 `--limit N` 用于 dry-run/smoke
- 每个脚本 `--rank R --world_size W` 必填
- 每个脚本一开始打印：python ver, torch ver, CUDA dev, args dict（便于 log 重现）

---

## Task 1: `pick_instances.py` — pre-flight 选 N 个 sha

**Files:**
- Create: `scripts/coart_data_v0/pick_instances.py`
- Test: `coart/tests/test_data_v0_pick_instances.py`

**Why first:** 不依赖 GPU、不依赖外部模型；纯 pandas 过滤；为后续所有 stage 提供输入。

- [ ] **Step 1: Write the failing test**

`coart/tests/test_data_v0_pick_instances.py`:

```python
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

    # Synthetic top-level metadata.csv
    rows = []
    for i in range(50):
        rows.append({
            "sha256": f"{i:064x}",
            "file_identifier": f"https://example.com/{i}",
            "aesthetic_score": 5.5 if i < 30 else 3.0,  # 30 pass aesthetic gate
            "captions": f'["caption {i}"]',
        })
    meta_csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta_csv, index=False)

    # Synthetic raw/metadata.csv (sha256 + local_path)
    raw_meta = tmp_path / "raw_metadata.csv"
    raw_rows = [{"sha256": f"{i:064x}", "local_path": f"raw/x/{i}.glb"}
                for i in range(50) if i != 5]  # sha=5 missing local_path
    pd.DataFrame(raw_rows).to_csv(raw_meta, index=False)

    # Synthetic feat18 dir: only 0..39 have npz
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
    # All sha must satisfy: aesthetic >= 4.5 AND has feat18 AND has local_path
    for sha in out["sha256"].astype(str):
        idx = int(sha, 16)
        assert idx < 30, f"sha {sha} has aesthetic < 4.5"
        assert idx < 40, f"sha {sha} has no feat18"
        assert idx != 5, f"sha {sha} has no local_path"
    # Determinism with same seed
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_pick_instances.py -v 2>&1 | head -20
```
Expected: `ImportError` or `FileNotFoundError` (pick_instances.py doesn't exist).

- [ ] **Step 3: Write minimal implementation**

`scripts/coart_data_v0/pick_instances.py`:

```python
#!/usr/bin/env python
"""Pre-flight: pick N sha256 from sketchfab subset that have feat18 + local_path
+ aesthetic >= threshold; deterministic via --seed; write instances_{N}k.csv.

Usage:
  python scripts/coart_data_v0/pick_instances.py \
    --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
    --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
    --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
    --aesthetic_min 4.5 --n 10000 --seed 0 \
    --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_10k.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd


def run(
    metadata_csv: str,
    raw_metadata_csv: str,
    feat18_dir: str,
    aesthetic_min: float,
    n: int,
    seed: int,
    out: str,
) -> None:
    df = pd.read_csv(metadata_csv, dtype={"sha256": str})
    raw = pd.read_csv(raw_metadata_csv, dtype={"sha256": str})
    df = df.merge(raw[["sha256", "local_path"]], on="sha256", how="inner")
    df = df[df["local_path"].notna()]
    df = df[df["aesthetic_score"] >= aesthetic_min]

    feat18_set = {fn.removesuffix(".npz")
                  for fn in os.listdir(feat18_dir) if fn.endswith(".npz")}
    df = df[df["sha256"].isin(feat18_set)]

    if len(df) < n:
        raise ValueError(
            f"requested n={n} but only {len(df)} candidates pass filters "
            f"(aesthetic >= {aesthetic_min} AND has feat18 AND has local_path)"
        )

    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(df), size=n, replace=False)
    out_df = df.iloc[np.sort(pick_idx)].copy().reset_index(drop=True)

    out_df["feat18_npz_size_bytes"] = out_df["sha256"].apply(
        lambda s: os.path.getsize(os.path.join(feat18_dir, f"{s}.npz"))
    )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[pick] wrote {len(out_df)} rows to {out}")
    print(f"[pick] aesthetic_score: min={out_df['aesthetic_score'].min():.2f} "
          f"mean={out_df['aesthetic_score'].mean():.2f} "
          f"max={out_df['aesthetic_score'].max():.2f}")
    print(f"[pick] feat18 size bytes: p50={out_df['feat18_npz_size_bytes'].median():.0f} "
          f"p95={out_df['feat18_npz_size_bytes'].quantile(0.95):.0f}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--raw_metadata_csv", required=True)
    p.add_argument("--feat18_dir", required=True)
    p.add_argument("--aesthetic_min", type=float, default=4.5)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_pick_instances.py -v
```
Expected: `2 passed`.

- [ ] **Step 5: Smoke test against real data**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  mkdir -p /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0 && \
  .venv/bin/python scripts/coart_data_v0/pick_instances.py \
    --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
    --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
    --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
    --aesthetic_min 4.5 --n 100 --seed 0 \
    --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_smoke100.csv && \
  head -3 /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_smoke100.csv
```
Expected: prints 100 rows summary + first 3 rows of CSV. All sha 都满足 aesthetic ≥ 4.5 且 feat18 文件存在。

- [ ] **Step 6: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add scripts/coart_data_v0/pick_instances.py coart/tests/test_data_v0_pick_instances.py && \
  git commit -m "feat(coart_data_v0): add pick_instances pre-flight selector

Filters sketchfab metadata by aesthetic_score >= 4.5 AND feat18 npz
exists AND raw local_path exists; deterministic seeded sample of N sha.
Inherits all metadata.csv columns + adds feat18_npz_size_bytes.

Test: 2/2 passed; smoke run on real data produced instances_smoke100.csv."
```

---

## Task 2: `build_manifest.py` — 聚合 stage 状态到 manifest.csv

**Files:**
- Create: `scripts/coart_data_v0/build_manifest.py`
- Test: `coart/tests/test_data_v0_build_manifest.py`

**Why next:** 也是纯 pandas 逻辑、无 GPU；后续 stage 完成后就能跑。

- [ ] **Step 1: Write the failing test**

`coart/tests/test_data_v0_build_manifest.py`:

```python
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
        sha_a: {"render": True,  "dino": True,  "slat_tags": ["v1"]},   # full
        sha_b: {"render": True,  "dino": False, "slat_tags": []},        # render only
        sha_c: {"render": False, "dino": False, "slat_tags": []},        # nothing
        sha_d: {"render": True,  "dino": True,  "slat_tags": ["v0", "v1"]},  # multi-vae, picks lex max
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

    # multi-tag picks lex max
    assert m.loc[sha_d, "slat_tag"] == "v1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_build_manifest.py -v 2>&1 | head -10
```
Expected: `FileNotFoundError` (build_manifest.py doesn't exist).

- [ ] **Step 3: Write minimal implementation**

`scripts/coart_data_v0/build_manifest.py`:

```python
#!/usr/bin/env python
"""Aggregate per-stage state into single manifest.csv.

Single-writer (no rank sharding); ~10 sec for 10K assets.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd


def _check_render(renders_dir: str, sha: str) -> tuple[bool, int]:
    d = os.path.join(renders_dir, sha)
    if not os.path.isdir(d):
        return False, 0
    if not os.path.isfile(os.path.join(d, "transforms.json")):
        return False, 0
    n = sum(1 for fn in os.listdir(d) if fn.endswith(".png"))
    return n == 16, n


def _check_dino(dino_dir: str, sha: str) -> bool:
    p = os.path.join(dino_dir, f"{sha}.npz")
    if not os.path.isfile(p):
        return False
    try:
        with np.load(p) as z:
            return z["features"].shape[0] == 16
    except (KeyError, ValueError, OSError):
        return False


def _check_slat(slat_root: str, sha: str) -> tuple[Optional[str], int]:
    """Return (tag_lex_max, num_voxels) or (None, 0)."""
    if not os.path.isdir(slat_root):
        return None, 0
    candidates = []
    for tag in sorted(os.listdir(slat_root)):
        p = os.path.join(slat_root, tag, f"{sha}.npz")
        if os.path.isfile(p):
            candidates.append((tag, p))
    if not candidates:
        return None, 0
    tag, p = candidates[-1]  # lex max
    try:
        with np.load(p) as z:
            return tag, int(z["num_voxels"])
    except (KeyError, ValueError, OSError):
        return tag, 0


def run(
    instances: str,
    renders_dir: str,
    dino_dir: str,
    slat_root: str,
    out: str,
) -> None:
    df = pd.read_csv(instances, dtype={"sha256": str})
    rows = []
    for sha in df["sha256"]:
        rd, n_views = _check_render(renders_dir, sha)
        dd = _check_dino(dino_dir, sha)
        slat_tag, n_vox = _check_slat(slat_root, sha)
        rows.append({
            "sha256": sha,
            "n_views_rendered": n_views,
            "render_done": rd,
            "dino_done": dd,
            "slat_done": slat_tag is not None,
            "slat_tag": slat_tag or "",
            "num_voxels": n_vox,
        })
    state = pd.DataFrame(rows)
    merged = df.merge(state, on="sha256", how="left")
    merged["last_updated"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
    merged["failed_reason"] = ""

    tmp = f"{out}.tmp.{os.getpid()}"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, out)

    n = len(merged)
    print(f"[manifest] wrote {n} rows → {out}")
    for col in ("render_done", "dino_done", "slat_done"):
        c = int(merged[col].sum())
        print(f"[manifest] {col}: {c}/{n} = {c/n:.1%}")
    if merged["slat_tag"].notna().any():
        print(f"[manifest] slat_tag dist: {merged['slat_tag'].value_counts().to_dict()}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True)
    p.add_argument("--renders_dir", required=True)
    p.add_argument("--dino_dir", required=True)
    p.add_argument("--slat_root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_build_manifest.py -v
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add scripts/coart_data_v0/build_manifest.py coart/tests/test_data_v0_build_manifest.py && \
  git commit -m "feat(coart_data_v0): add build_manifest aggregator

Single-writer manifest.csv aggregating render/dino/slat state per sha256.
Multi-vae slat picks lex max tag. Atomic write via tmp + os.replace.

Test: 1/1 passed (synthetic 4-sha layout covering full/partial/missing/multi-vae)."
```

---

## Task 3: `coart_compare_occupancy.py` — SS-flow IoU pre-flight gate

**Files:**
- Create: `scripts/coart_compare_occupancy.py`
- Test: `coart/tests/test_data_v0_compare_occupancy.py`

**Why now (before the GPU stages):** SS-flow IoU 是 spec §6.2 的 pre-flight gate；如 IoU < 0.8 整个 pipeline scope 要重谈。先把这个验证脚本写好，后续 production launch 之前必跑一次。

**Strategy:** 在 8 golden assets 上跑 corep voxelize（res=64）vs 原生 voxelize（res=64）→ max_pool 到 32³/16³ → 算 IoU。失败回退：如 corep_fast / o_voxel 的具体 voxelize API 与 spec 假设有差异，writing-plans 后续 task 会发现并修正。

- [ ] **Step 1: Write the failing test**

`coart/tests/test_data_v0_compare_occupancy.py`:

```python
"""Tests for scripts/coart_compare_occupancy.py.

Synthetic sphere mesh: corep + o_voxel should agree to high IoU.
"""
import importlib.util
import pathlib

import numpy as np
import pytest
import trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module():
    p = ROOT / "scripts" / "coart_compare_occupancy.py"
    spec = importlib.util.spec_from_file_location("coart_compare_occupancy", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_iou_on_synthetic_sphere(tmp_path: pathlib.Path):
    co = _load_module()
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    glb_path = tmp_path / "sphere.glb"
    sphere.export(glb_path)

    metrics = co.compare_one(str(glb_path), resolutions=[64, 32, 16])
    # Synthetic sphere should agree well at all resolutions
    assert metrics["iou_64"] > 0.80
    assert metrics["iou_32"] > 0.85
    assert metrics["iou_16"] > 0.90


def test_max_pool_correctness():
    co = _load_module()
    # cubes at [0,0,0], [1,1,1], [2,2,2] in res=4 → after max_pool to res=2 → [0,0,0], [1,1,1]
    cubes = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=np.int32)
    pooled = co.downsample_cubes(cubes, factor=2)
    expected = np.array([[0,0,0], [0,0,0], [1,1,1], [1,1,1]], dtype=np.int32)
    expected_unique = np.unique(expected, axis=0)
    pooled_unique = np.unique(pooled, axis=0)
    np.testing.assert_array_equal(np.sort(pooled_unique, axis=0), np.sort(expected_unique, axis=0))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_compare_occupancy.py -v 2>&1 | head -10
```
Expected: `FileNotFoundError`.

- [ ] **Step 3: Inspect existing voxelize APIs（避免猜 API）**

```bash
grep -rn "def voxelize\|def mesh_to_flexible\|def s1_voxelize" /mnt/novita2/siyuan/workspace/TRELLIS.2/corep_fast /mnt/novita2/siyuan/workspace/TRELLIS.2/o-voxel 2>&1 | head -20
```
Expected: 找到 `corep_fast/stages/s1_voxelize.py` 中的 voxelize 函数 + `o-voxel/o_voxel/convert/flexible_dual_grid.py` 中的 `mesh_to_flexible_dual_grid`。**记下确切 import 路径与签名**，用到下一步实现里。

- [ ] **Step 4: Write minimal implementation**

`scripts/coart_compare_occupancy.py`:

```python
#!/usr/bin/env python
"""SS-flow occupancy IoU validation: corep voxelize vs original o_voxel,
on golden assets, at res=64/32/16. Decision gate per spec §6.2.

Usage:
  python scripts/coart_compare_occupancy.py \
    --golden_dir datasets/coart_golden \
    --resolutions 64,32,16 \
    --out logs/findings_ss_flow_iou.md

Step-3 must inspect the actual API and fill in the two TODO call sites below
before this script runs. Both APIs are documented in spec §1.3.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np
import trimesh


def downsample_cubes(cubes: np.ndarray, factor: int) -> np.ndarray:
    """Max-pool 3D occupancy to lower resolution. cubes: (N, 3) int. Returns
    deduplicated downsampled coords."""
    pooled = (cubes.astype(np.int64) // factor).astype(np.int32)
    return np.unique(pooled, axis=0)


def voxelize_corep(mesh_path: str, resolution: int) -> np.ndarray:
    """Return (N, 3) int corep cube indices in [0, resolution)^3."""
    # NOTE: import inside fn so test that mocks this can patch.
    from corep_fast.stages import s1_voxelize  # type: ignore[import]
    from corep_fast.containers import MeshTensors  # type: ignore[import]

    m = trimesh.load(mesh_path, force="mesh")
    mt = MeshTensors.from_trimesh(m, resolution=resolution)
    cubes = s1_voxelize.voxelize_mesh(mt)  # exact name verified at step 3
    return cubes.cpu().numpy().astype(np.int32) if hasattr(cubes, "cpu") else np.asarray(cubes, np.int32)


def voxelize_native(mesh_path: str, resolution: int) -> np.ndarray:
    """Return (N, 3) int original (o_voxel) surface cube indices."""
    from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid  # type: ignore[import]
    m = trimesh.load(mesh_path, force="mesh")
    # native: scale = 0.99999 / extent, mesh in [-0.5, 0.5]^3
    extent = (m.vertices.max(0) - m.vertices.min(0)).max()
    scale = 0.99999 / extent
    center = (m.vertices.max(0) + m.vertices.min(0)) / 2
    m.vertices = (m.vertices - center) * scale  # → [-0.5, 0.5]
    out = mesh_to_flexible_dual_grid(m, resolution=resolution)  # exact name verified step 3
    cubes = out["voxel_indices"] if isinstance(out, dict) else out[0]
    return np.asarray(cubes, np.int32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Set IoU on (N,3) int cube coords."""
    sa = {tuple(r) for r in a}; sb = {tuple(r) for r in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def compare_one(mesh_path: str, resolutions: list[int]) -> dict:
    out = {"asset": os.path.basename(mesh_path)}
    base_res = max(resolutions)
    cubes_corep = voxelize_corep(mesh_path, base_res)
    cubes_native = voxelize_native(mesh_path, base_res)
    for res in resolutions:
        factor = base_res // res
        a = downsample_cubes(cubes_corep, factor) if factor > 1 else cubes_corep
        b = downsample_cubes(cubes_native, factor) if factor > 1 else cubes_native
        out[f"iou_{res}"] = iou(a, b)
        out[f"a_minus_b_{res}"] = max(0, len(a) - len(set(map(tuple, a)) & set(map(tuple, b))))
        out[f"b_minus_a_{res}"] = max(0, len(b) - len(set(map(tuple, b)) & set(map(tuple, a))))
        out[f"n_corep_{res}"] = len(a)
        out[f"n_native_{res}"] = len(b)
    return out


def run(golden_dir: str, resolutions: list[int], out: str) -> None:
    paths = sorted(glob.glob(os.path.join(golden_dir, "*.glb"))
                  + glob.glob(os.path.join(golden_dir, "*", "*.glb")))
    if not paths:
        raise FileNotFoundError(f"no .glb in {golden_dir}")

    rows = []
    for p in paths:
        try:
            rows.append(compare_one(p, resolutions))
        except Exception as e:
            print(f"[iou] FAIL {p}: {e}", file=sys.stderr)
            rows.append({"asset": os.path.basename(p), "error": str(e)})

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("# SS-flow Occupancy IoU Report\n\n")
        cols = ["asset"] + [f"iou_{r}" for r in resolutions] + [f"n_corep_{r}" for r in resolutions] + [f"n_native_{r}" for r in resolutions]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |\n")

        # decision
        valid = [r for r in rows if "error" not in r]
        if valid:
            min_res = min(resolutions)
            mean_iou_min = sum(r[f"iou_{min_res}"] for r in valid) / len(valid)
            f.write(f"\n**Mean IoU @ {min_res}³: {mean_iou_min:.4f}**\n\n")
            if mean_iou_min >= 0.9:
                verdict = "PASS — 方案 1 直接走，仅推理时仿射量化对齐"
            elif mean_iou_min >= 0.8:
                verdict = "PASS WITH WARNING — 方案 1 + spec 增加 affine 对齐节"
            else:
                verdict = "FAIL — 阻塞，升级到方案 2 (SS-flow finetune)"
            f.write(f"**Verdict: {verdict}**\n")
            print(f"[iou] mean_iou_{min_res}={mean_iou_min:.4f} → {verdict}")
        else:
            print("[iou] no valid rows; check .glb paths and voxelize APIs", file=sys.stderr)
            sys.exit(1)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden_dir", required=True)
    p.add_argument("--resolutions", default="64,32,16")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    run(args.golden_dir, [int(x) for x in args.resolutions.split(",")], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 验证 voxelize 函数名 + 签名**

如果 `voxelize_corep` / `voxelize_native` 调用失败（API 与 spec 假设不同），用真实 import 路径修正。**这是 spec §14 open question #5 的实质 verify**：

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -c "
from corep_fast.stages import s1_voxelize
print(dir(s1_voxelize))
from corep_fast.containers import MeshTensors
print('MeshTensors.from_trimesh:', hasattr(MeshTensors, 'from_trimesh'))
"
```
若打印的函数名与代码不一致，**修正 `voxelize_corep` 函数体内的调用**（不要乱猜）。同样 verify o_voxel：

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -c "
from o_voxel.convert import flexible_dual_grid as fdg
print([x for x in dir(fdg) if 'mesh' in x.lower()])
"
```

- [ ] **Step 6: Run test to verify it passes**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_compare_occupancy.py -v
```
Expected: `2 passed`. 若 `test_iou_on_synthetic_sphere` 失败但 IoU 值合理（如 0.78），说明 sphere subdivision 太粗；提高 `subdivisions=4` 重试。若 IoU 严重低于 0.5，说明 voxelize 调用错了，回到 step 5。

- [ ] **Step 7: Smoke run on real golden assets**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python scripts/coart_compare_occupancy.py \
    --golden_dir datasets/coart_golden \
    --resolutions 64,32,16 \
    --out logs/findings_ss_flow_iou.md && \
  cat logs/findings_ss_flow_iou.md'"
```
Expected: 8 行 asset 表格 + verdict 行。**记录 verdict（PASS / PASS WITH WARNING / FAIL）**，这是 production launch 的 gate。

- [ ] **Step 8: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add scripts/coart_compare_occupancy.py coart/tests/test_data_v0_compare_occupancy.py logs/findings_ss_flow_iou.md && \
  git commit -m "feat(coart): add SS-flow occupancy IoU validation gate

Compares corep_fast vs original o_voxel surface cube sets at res=64/32/16
on 8 golden assets; writes markdown report + decision verdict (spec §6.2).

Test: 2/2 passed (synthetic sphere IoU > 0.85 at all res).
Real run: see logs/findings_ss_flow_iou.md verdict."
```

---

## Task 4: `coart_cache_dino.py` — DINOv3 ViT-L/16 features cache

**Files:**
- Create: `scripts/coart_data_v0/coart_cache_dino.py`
- Test: `coart/tests/test_data_v0_cache_dino_shape.py`
- Test: `coart/tests/test_data_v0_atomic_write.py`
- Test: `coart/tests/test_data_v0_stable_shard.py`

**Why now:** `pick_instances` 已有；本 task 要 GPU；可与 render 并行 launch（render 在 task 7-8 跑）。

- [ ] **Step 1: Write 3 failing tests**

`coart/tests/test_data_v0_stable_shard.py`:
```python
"""Test the inline stable_shard helper used by dino/slat scripts."""
import importlib.util, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_shard_partition_is_complete_and_disjoint():
    cache_dino = _load("coart_cache_dino")
    sha_pool = [f"{i:08x}" + "0" * 56 for i in range(1000)]
    world = 8
    assignments = [[s for s in sha_pool if cache_dino.stable_shard(s, world, r)] for r in range(world)]
    flat = [s for row in assignments for s in row]
    assert len(flat) == 1000  # complete
    assert len(set(flat)) == 1000  # disjoint
    # rough balance: each rank gets within 30% of mean
    sizes = [len(row) for row in assignments]
    assert max(sizes) - min(sizes) < 0.3 * 1000 / world
```

`coart/tests/test_data_v0_atomic_write.py`:
```python
"""Test the inline atomic_savez helper."""
import importlib.util, os, pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_atomic_savez_no_partial_visible(tmp_path: pathlib.Path):
    cache_dino = _load("coart_cache_dino")
    target = tmp_path / "out.npz"
    arr = np.arange(100, dtype=np.float32)
    cache_dino.atomic_savez(str(target), x=arr)
    # only the final file should exist; no .tmp leftovers
    assert target.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    with np.load(target) as z:
        np.testing.assert_array_equal(z["x"], arr)
```

`coart/tests/test_data_v0_cache_dino_shape.py`:
```python
"""Test cache_dino output schema using mocked DinoV3 model."""
import importlib.util, os, pathlib
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _make_dummy_renders(root: pathlib.Path, sha: str, n_views: int = 16):
    d = root / sha; d.mkdir(parents=True, exist_ok=True)
    for v in range(n_views):
        img = Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))
        img.save(d / f"{v:03d}.png")


def test_dino_output_schema_with_mock(tmp_path: pathlib.Path):
    cache_dino = _load("coart_cache_dino")
    sha = "a" * 64
    renders_dir = tmp_path / "renders"
    out_dir = tmp_path / "dino"; out_dir.mkdir()
    _make_dummy_renders(renders_dir, sha)

    # Mock the extractor to return a fixed (B, T, D) tensor
    import torch
    fake_T = 1029
    fake_D = 1024

    class FakeExtractor:
        def __init__(self, *a, **kw): pass
        def cuda(self): return self
        def __call__(self, image):
            return torch.zeros((image.shape[0], fake_T, fake_D), dtype=torch.bfloat16)

    with patch.object(cache_dino, "build_extractor", return_value=FakeExtractor()):
        cache_dino.process_one(
            sha=sha,
            renders_dir=str(renders_dir),
            out_dir=str(out_dir),
            extractor=FakeExtractor(),
            image_size=512,
        )

    out = np.load(out_dir / f"{sha}.npz")
    assert out["features"].shape == (16, fake_T, fake_D)
    assert out["features"].dtype == np.float16
    assert int(out["n_tokens"]) == fake_T
    assert int(out["image_size"]) == 512
    assert str(out["model_id"]) == "facebook/dinov3-vitl16-pretrain-lvd1689m"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_stable_shard.py coart/tests/test_data_v0_atomic_write.py coart/tests/test_data_v0_cache_dino_shape.py -v 2>&1 | head -15
```
Expected: 3 fail with FileNotFoundError on coart_cache_dino.py.

- [ ] **Step 3: Write minimal implementation**

`scripts/coart_data_v0/coart_cache_dino.py`:

```python
#!/usr/bin/env python
"""Cache DINOv3 ViT-L/16 features per asset (16 views × T tokens × 1024 dim, bf16).

Per-rank sharding via stable_shard; resume by skipping existing output.
Atomic write via os.replace. See spec §4.2.2 + §5.4.

Usage:
  CUDA_VISIBLE_DEVICES=$rank python scripts/coart_data_v0/coart_cache_dino.py \
    --instances <coart_data_root>/instances_10k.csv \
    --renders_dir <coart_data_root>/renders_cond \
    --out_dir <coart_data_root>/dino_l16_s512 \
    --rank R --world_size W
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def stable_shard(sha: str, world_size: int, rank: int) -> bool:
    """Stable hash via sha[:8] hex; PYTHONHASHSEED-independent."""
    return (int(sha[:8], 16) % world_size) == rank


def atomic_savez(out_path: str, **arrays) -> None:
    tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time()*1e6)}"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)


def build_extractor(image_size: int):
    """Return a callable that maps (B, 3, H, W) tensor → (B, T, D) bf16."""
    # Mirror trellis2/modules/image_feature_extractor.py:DinoV3FeatureExtractor.
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    extractor = DinoV3FeatureExtractor(model_name=DINO_MODEL_ID, image_size=image_size)
    extractor.cuda()
    return extractor


def _load_view(png_path: str, image_size: int) -> torch.Tensor:
    """Match ImageConditionedMixin.get_instance preprocessing
    (trellis2/datasets/components.py:113-129)."""
    img = Image.open(png_path)
    alpha = np.array(img.getchannel(3))
    bbox = alpha.nonzero()
    if bbox[0].size == 0:
        # all-transparent fallback: center crop then resize
        img = img.resize((image_size, image_size), Image.LANCZOS)
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        a = np.array(img.getchannel(3), dtype=np.float32) / 255.0
    else:
        bb = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        h = max(bb[2] - bb[0], bb[3] - bb[1]) / 2
        crop = [int(cx - h), int(cy - h), int(cx + h), int(cy + h)]
        img = img.crop(crop).resize((image_size, image_size), Image.LANCZOS)
        a = np.array(img.getchannel(3), dtype=np.float32) / 255.0
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    rgb = rgb * a[..., None]  # alpha composite over black
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return t  # (3, H, W) in [0, 1] before normalize


@torch.no_grad()
def process_one(
    sha: str,
    renders_dir: str,
    out_dir: str,
    extractor,
    image_size: int,
) -> tuple[bool, str]:
    out_path = os.path.join(out_dir, f"{sha}.npz")
    if os.path.exists(out_path):
        return True, "skip-exists"
    asset_dir = os.path.join(renders_dir, sha)
    pngs = sorted(p for p in os.listdir(asset_dir) if p.endswith(".png"))
    if len(pngs) != 16:
        return False, f"expected 16 views, got {len(pngs)}"

    views = torch.stack([_load_view(os.path.join(asset_dir, p), image_size) for p in pngs])  # (16, 3, H, W)
    views = views.cuda()
    # DinoV3FeatureExtractor.transform = Normalize(mean, std) only (no resize, since we did resize)
    feats = extractor(views)  # (16, T, D)
    # Store as fp16 (np.float16): same disk size as bf16, well-supported by numpy,
    # safe for DINO output (layer-normed, max abs ~50 < fp16 range 65504).
    feats_f16 = feats.to(torch.float16).cpu().numpy()
    T = feats_f16.shape[1]

    atomic_savez(
        out_path,
        features=feats_f16,
        view_idx=np.arange(16, dtype=np.uint8),
        n_tokens=np.int32(T),
        model_id=np.array(DINO_MODEL_ID),
        image_size=np.int32(image_size),
    )
    return True, f"ok T={T}"


def run(
    instances: str,
    renders_dir: str,
    out_dir: str,
    rank: int,
    world_size: int,
    image_size: int = 512,
    limit: Optional[int] = None,
) -> None:
    print(f"[dino] rank={rank}/{world_size} python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(instances, dtype={"sha256": str})
    todo = [s for s in df["sha256"] if stable_shard(s, world_size, rank)]
    if limit:
        todo = todo[:limit]
    print(f"[dino] rank={rank}: {len(todo)} sha to process")

    extractor = build_extractor(image_size)
    n_ok, n_fail, n_skip = 0, 0, 0
    for sha in tqdm(todo, desc=f"rank{rank}"):
        ok, msg = process_one(sha, renders_dir, out_dir, extractor, image_size)
        if not ok:
            n_fail += 1
            print(f"[dino] FAIL {sha}: {msg}", file=sys.stderr)
        elif msg == "skip-exists":
            n_skip += 1
        else:
            n_ok += 1
    print(f"[dino] rank={rank} done: ok={n_ok} skip={n_skip} fail={n_fail}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True)
    p.add_argument("--renders_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_stable_shard.py coart/tests/test_data_v0_atomic_write.py coart/tests/test_data_v0_cache_dino_shape.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Real DINO model load smoke test (1 rank, no asset)**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \"
from scripts.coart_data_v0 import coart_cache_dino as m
e = m.build_extractor(512)
import torch
x = torch.zeros((1, 3, 512, 512)).cuda()
y = e(x)
print(\\\"shape\\\", y.shape, \\\"dtype\\\", y.dtype)
\"'"
```
Expected: prints `shape torch.Size([1, T, 1024]) dtype torch.float32` (or bf16). T 期望 1029。**记录实际 T 值**，与 spec §4.2.2 假设比对；不一致则 spec 注释更新。

> 若提示 `from scripts.coart_data_v0 import ...` import 失败：scripts/ 不是包。临时改为 `import importlib.util` + `spec_from_file_location` 直接 load script。或者 `.venv/bin/python scripts/coart_data_v0/coart_cache_dino.py --help` 验证。

- [ ] **Step 6: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add scripts/coart_data_v0/coart_cache_dino.py \
          coart/tests/test_data_v0_stable_shard.py \
          coart/tests/test_data_v0_atomic_write.py \
          coart/tests/test_data_v0_cache_dino_shape.py && \
  git commit -m "feat(coart_data_v0): add coart_cache_dino with stable shard + atomic save

Loads DinoV3 ViT-L/16 (matches official trellis2 trainer config).
Per-asset: load 16 PNG → preprocess (alpha bbox crop, resize 512, alpha
composite, DINO normalize) → batch forward → bf16 npz with features
(16, T, 1024), view_idx, n_tokens, model_id, image_size.

Sharding via stable_shard (sha[:8] hex hash, PYTHONHASHSEED-safe).
Atomic write via os.replace, NFS-safe.

Tests: 3/3 passed (shard partition, atomic write, dino schema with mock)."
```

---

## Task 5: `coart_cache_slat.py` — coart EMA encoder mu cache

**Files:**
- Create: `scripts/coart_data_v0/coart_cache_slat.py`
- Test: `coart/tests/test_data_v0_cache_slat_shape.py`

- [ ] **Step 1: Write the failing test**

`coart/tests/test_data_v0_cache_slat_shape.py`:
```python
"""Test cache_slat output schema using a mock encoder."""
import importlib.util, pathlib
from unittest.mock import patch

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name):
    p = ROOT / "scripts" / "coart_data_v0" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_slat_output_schema_with_mock(tmp_path: pathlib.Path):
    slat = _load("coart_cache_slat")
    sha = "a" * 64

    feat18 = tmp_path / "feat18"; feat18.mkdir()
    N = 50
    np.savez(feat18 / f"{sha}.npz",
             cube_indices=np.random.randint(0, 32, (N, 3), dtype=np.int32),
             feats=np.random.randn(N, 18).astype(np.float32),
             num_boundary=np.zeros(N, dtype=np.int32))

    out_dir = tmp_path / "slat" / "vae_test_tag"
    out_dir.mkdir(parents=True)

    class FakeEncoder:
        def __call__(self, x, sample_posterior=False):
            # mock: return a SparseTensor-like with .feats of shape (N, 32)
            class Z:
                def __init__(self, f, c):
                    self.feats = f; self.coords = c
            return Z(torch.randn(x.feats.shape[0], 32), x.coords)
        def eval(self): return self

    with patch.object(slat, "load_encoder",
                      return_value=(FakeEncoder(), torch.zeros(18), torch.ones(18))):
        slat.process_one(
            sha=sha,
            feat18_dir=str(feat18),
            out_dir=str(out_dir),
            encoder=FakeEncoder(),
            mean=torch.zeros(18), std=torch.ones(18),
            vae_ckpt_rel="results/x.pt",
            vae_io_arch="three_branch",
        )

    z = np.load(out_dir / f"{sha}.npz")
    assert z["coords"].shape == (N, 3)
    assert z["coords"].dtype == np.int16
    assert z["feats"].shape == (N, 32)
    assert z["feats"].dtype == np.float16
    assert int(z["num_voxels"]) == N
    assert str(z["vae_ckpt_rel"]) == "results/x.pt"
    assert str(z["vae_io_arch"]) == "three_branch"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_cache_slat_shape.py -v 2>&1 | head -10
```
Expected: FileNotFoundError.

- [ ] **Step 3: Write minimal implementation**

`scripts/coart_data_v0/coart_cache_slat.py`:

```python
#!/usr/bin/env python
"""Cache coart shape SLat latent (encoder mu) per asset, versioned by VAE tag.

Per-rank sharding via stable_shard; resume by skipping existing output.
Atomic write via os.replace. See spec §4.2.3 + §5.5.

Usage:
  CUDA_VISIBLE_DEVICES=$rank python scripts/coart_data_v0/coart_cache_slat.py \
    --instances <coart_data_root>/instances_10k.csv \
    --feat18_dir <dataset_root>/feat18_512/data \
    --vae_ckpt   results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \
    --vae_tag    vae_three_branch_ws_v0_ema_s0155000 \
    --vae_io_arch three_branch \
    --stats_npz  /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz \
    --out_root <coart_data_root>/slat \
    --rank R --world_size W
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# These two helpers are duplicated across cache_dino / cache_slat for isolation
# (~10 lines each, no shared module to keep scripts self-contained).


def stable_shard(sha: str, world_size: int, rank: int) -> bool:
    return (int(sha[:8], 16) % world_size) == rank


def atomic_savez(out_path: str, **arrays) -> None:
    tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time()*1e6)}"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)


def load_encoder(vae_ckpt: str, vae_io_arch: str, stats_npz: str, device: str = "cuda"):
    """Build encoder via coart.vae.build, load EMA ckpt, return (encoder, mean, std)."""
    from coart.vae.build import build_models
    from coart.data.stats import load_stats

    encoder, _decoder = build_models(io_arch=vae_io_arch, device=device)
    state = torch.load(vae_ckpt, map_location=device, weights_only=True)
    # ema files store the encoder state dict directly (verified pattern from coart.vae.train)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[slat] encoder load: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    encoder.eval()
    mean, std = load_stats(stats_npz, torch.device(device), verbose=True)
    return encoder, mean, std


@torch.no_grad()
def process_one(
    sha: str,
    feat18_dir: str,
    out_dir: str,
    encoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    vae_ckpt_rel: str,
    vae_io_arch: str,
) -> tuple[bool, str]:
    out_path = os.path.join(out_dir, f"{sha}.npz")
    if os.path.exists(out_path):
        return True, "skip-exists"

    feat_path = os.path.join(feat18_dir, f"{sha}.npz")
    if not os.path.isfile(feat_path):
        return False, "no-feat18"

    with np.load(feat_path) as z:
        cube_indices = z["cube_indices"].astype(np.int32)  # (N, 3)
        feats_raw = z["feats"].astype(np.float32)          # (N, 18)
    N = cube_indices.shape[0]
    if N == 0:
        return False, "empty-feat18"

    # Build SparseTensor (matches train loop coart/vae/train.py:393-397)
    import trellis2.modules.sparse as sp
    from coart.data.stats import normalize
    coords_t = torch.from_numpy(cube_indices).int().cuda()
    # batch_idx column required by SparseTensor: prepend 0 for single sample
    coords_with_b = torch.cat([torch.zeros((N, 1), dtype=torch.int32, device=coords_t.device), coords_t], dim=1)
    feats_t = torch.from_numpy(feats_raw).cuda()
    feats_n = normalize(feats_t, mean, std)
    x = sp.SparseTensor(feats=feats_n, coords=coords_with_b)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        z_out = encoder(x, sample_posterior=False)  # returns mu when sample_posterior=False
    # Store as fp16 (see cache_dino note); coart latent is also normalized → safe range.
    mu = z_out.feats.detach().to(torch.float16).cpu().numpy()

    atomic_savez(
        out_path,
        coords=cube_indices.astype(np.int16),
        feats=mu,
        num_voxels=np.int32(N),
        vae_ckpt_rel=np.array(vae_ckpt_rel),
        vae_io_arch=np.array(vae_io_arch),
    )
    return True, f"ok N={N}"


def run(
    instances: str,
    feat18_dir: str,
    vae_ckpt: str,
    vae_tag: str,
    vae_io_arch: str,
    stats_npz: str,
    out_root: str,
    rank: int,
    world_size: int,
    limit: Optional[int] = None,
) -> None:
    print(f"[slat] rank={rank}/{world_size} vae_tag={vae_tag}")
    out_dir = os.path.join(out_root, vae_tag)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(instances, dtype={"sha256": str})
    todo = [s for s in df["sha256"] if stable_shard(s, world_size, rank)]
    if limit:
        todo = todo[:limit]
    print(f"[slat] rank={rank}: {len(todo)} sha to process")

    encoder, mean, std = load_encoder(vae_ckpt, vae_io_arch, stats_npz)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vae_ckpt_rel = os.path.relpath(os.path.abspath(vae_ckpt), repo_root)
    n_ok, n_fail, n_skip = 0, 0, 0
    for sha in tqdm(todo, desc=f"rank{rank}"):
        ok, msg = process_one(sha, feat18_dir, out_dir, encoder, mean, std, vae_ckpt_rel, vae_io_arch)
        if not ok:
            n_fail += 1
            print(f"[slat] FAIL {sha}: {msg}", file=sys.stderr)
        elif msg == "skip-exists":
            n_skip += 1
        else:
            n_ok += 1
    print(f"[slat] rank={rank} done: ok={n_ok} skip={n_skip} fail={n_fail}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True)
    p.add_argument("--feat18_dir", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--vae_tag", required=True)
    p.add_argument("--vae_io_arch", default="three_branch")
    p.add_argument("--stats_npz", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest coart/tests/test_data_v0_cache_slat_shape.py -v
```
Expected: `1 passed`.

- [ ] **Step 5: Real encoder load smoke test (no asset, just init)**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \"
import sys; sys.path.insert(0, \\\"scripts/coart_data_v0\\\")
import coart_cache_slat as m
enc, mean, std = m.load_encoder(
    vae_ckpt=\\\"results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt\\\",
    vae_io_arch=\\\"three_branch\\\",
    stats_npz=\\\"/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz\\\",
)
n_params = sum(p.numel() for p in enc.parameters())
print(f\\\"encoder loaded, n_params={n_params}, mean.shape={mean.shape}, std.shape={std.shape}\\\")
\"'"
```
Expected: 打印 encoder 参数数与 stats shape (18,)。**若 stats_global.npz 不存在**：检查 `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/` 实际文件，与 `coart/data/stats.py:load_stats` 对照（path 可能在其他位置）。

- [ ] **Step 6: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add scripts/coart_data_v0/coart_cache_slat.py coart/tests/test_data_v0_cache_slat_shape.py && \
  git commit -m "feat(coart_data_v0): add coart_cache_slat with VAE-tag versioning

Loads coart EMA encoder via coart.vae.build.build_models + load_state_dict.
Per-asset: load feat18 npz → normalize → SparseTensor → encoder forward
(sample_posterior=False) → mu → bf16 npz with coords (int16), feats (uint16=bf16),
num_voxels, vae_ckpt_rel, vae_io_arch.

Output goes to slat/{vae_tag}/{sha}.npz; tag enables A/B vs old VAE.

Test: 1/1 passed (mock encoder, schema contract)."
```

---

## Task 6: `run.sh` 编排 + `README.md`

**Files:**
- Create: `scripts/coart_data_v0/run.sh`
- Create: `scripts/coart_data_v0/README.md`

- [ ] **Step 1: Write run.sh**

`scripts/coart_data_v0/run.sh`:

```bash
#!/usr/bin/env bash
# Stage launcher for coart DiT data pipeline v0.
# Mirrors the env-var pattern of scripts/precompute_feat18_objaverse_sketchfab.sh.
#
# Required env vars:
#   STAGE                   render | dino | slat | manifest
#
# Common env vars (defaults shown):
#   COART_DATA_ROOT         /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0
#   DATASET_ROOT            /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
#   INSTANCES               instances_10k.csv (relative to COART_DATA_ROOT)
#   NODES                   "117 118 119"
#   NUM_GPUS_PER_NODE       8
#   LIMIT                   "" (smoke runs use --limit)
#   PYTHON                  .venv/bin/python
#   REPO_ROOT               (auto-detected from this script's location)
#
# STAGE=slat additionally requires:
#   VAE_CKPT                results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt
#   VAE_TAG                 vae_three_branch_ws_v0_ema_s0155000
#   VAE_IO_ARCH             three_branch
#   STATS_NPZ               <DATASET_ROOT>/feat18_512/stats_global.npz
#
# Examples:
#   STAGE=render NODES="117 118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
#   STAGE=dino NODES="118 119" bash scripts/coart_data_v0/run.sh
#   STAGE=slat VAE_CKPT=... VAE_TAG=... NODES="119" bash scripts/coart_data_v0/run.sh
#   STAGE=manifest bash scripts/coart_data_v0/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

STAGE="${STAGE:?STAGE env var required (render | dino | slat | manifest)}"
COART_DATA_ROOT="${COART_DATA_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab}"
INSTANCES_REL="${INSTANCES:-instances_10k.csv}"
INSTANCES="${COART_DATA_ROOT}/${INSTANCES_REL}"
NODES="${NODES:-117 118 119}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"
LIMIT="${LIMIT:-}"
LIMIT_ARG=""
[ -n "${LIMIT}" ] && LIMIT_ARG="--limit ${LIMIT}"

LOG_DIR="${COART_DATA_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# Compute world size
NODES_ARR=( ${NODES} )
WORLD_SIZE=$(( ${#NODES_ARR[@]} * NUM_GPUS_PER_NODE ))
echo "[run] STAGE=${STAGE} NODES=(${NODES}) NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} WORLD_SIZE=${WORLD_SIZE}"

dispatch_python_per_rank() {
    local cmd_template="$1"
    local stage_log="$2"
    local pids=()
    local rank=0
    for node in "${NODES_ARR[@]}"; do
        for gpu in $(seq 0 $((NUM_GPUS_PER_NODE - 1))); do
            local log_file="${LOG_DIR}/${stage_log}_rank$(printf '%02d' ${rank}).log"
            local cmd="$(printf "${cmd_template}" "${gpu}" "${rank}" "${WORLD_SIZE}")"
            local script="${REPO_ROOT}/tmp/run_${STAGE}_rank${rank}.sh"
            mkdir -p "${REPO_ROOT}/tmp"
            cat > "${script}" <<EOF
#!/usr/bin/env bash
cd "${REPO_ROOT}"
${cmd} > "${log_file}" 2>&1
EOF
            chmod +x "${script}"
            ssh "host-10-240-99-${node}" "bash ${script}" &
            pids+=($!)
            rank=$((rank + 1))
        done
    done
    echo "[run] launched ${#pids[@]} ranks; waiting..."
    local exits=()
    for pid in "${pids[@]}"; do
        wait ${pid} && exits+=(0) || exits+=($?)
    done
    echo "[run] per-rank exit codes: ${exits[@]}"
    local n_fail=0
    for code in "${exits[@]}"; do [ ${code} -ne 0 ] && n_fail=$((n_fail+1)); done
    [ ${n_fail} -ne 0 ] && echo "[run] WARN: ${n_fail}/${#exits[@]} ranks failed"
}

case "${STAGE}" in
    render)
        # Reuse data_toolkit/render_cond.py as-is (no modification).
        # Need cwd=data_toolkit so 'datasets.ObjaverseXL' import works.
        SHA_LIST="${COART_DATA_ROOT}/_render_sha_list.txt"
        cut -d, -f1 "${INSTANCES}" | tail -n +2 > "${SHA_LIST}"
        # comma-joined for --instances
        SHA_CSV=$(paste -sd, "${SHA_LIST}")
        TPL="CUDA_VISIBLE_DEVICES=%s ${PYTHON} render_cond.py ObjaverseXL --root ${DATASET_ROOT} --download_root ${DATASET_ROOT} --render_cond_root ${COART_DATA_ROOT} --rank %s --world_size %s --num_cond_views 16 --instances ${SHA_LIST}"
        # Note: render_cond.py expects cwd=data_toolkit; modify command:
        TPL="cd ${REPO_ROOT}/data_toolkit && ${TPL}"
        dispatch_python_per_rank "${TPL}" "render"
        ;;
    dino)
        TPL="CUDA_VISIBLE_DEVICES=%s ${PYTHON} scripts/coart_data_v0/coart_cache_dino.py --instances ${INSTANCES} --renders_dir ${COART_DATA_ROOT}/renders_cond --out_dir ${COART_DATA_ROOT}/dino_l16_s512 --rank %s --world_size %s ${LIMIT_ARG}"
        dispatch_python_per_rank "${TPL}" "dino"
        ;;
    slat)
        VAE_CKPT="${VAE_CKPT:?VAE_CKPT env var required for STAGE=slat}"
        VAE_TAG="${VAE_TAG:?VAE_TAG env var required for STAGE=slat}"
        VAE_IO_ARCH="${VAE_IO_ARCH:-three_branch}"
        STATS_NPZ="${STATS_NPZ:-${DATASET_ROOT}/feat18_512/stats_global.npz}"
        TPL="CUDA_VISIBLE_DEVICES=%s ${PYTHON} scripts/coart_data_v0/coart_cache_slat.py --instances ${INSTANCES} --feat18_dir ${DATASET_ROOT}/feat18_512/data --vae_ckpt ${VAE_CKPT} --vae_tag ${VAE_TAG} --vae_io_arch ${VAE_IO_ARCH} --stats_npz ${STATS_NPZ} --out_root ${COART_DATA_ROOT}/slat --rank %s --world_size %s ${LIMIT_ARG}"
        dispatch_python_per_rank "${TPL}" "slat_${VAE_TAG}"
        ;;
    manifest)
        cd "${REPO_ROOT}"
        ${PYTHON} scripts/coart_data_v0/build_manifest.py \
            --instances "${INSTANCES}" \
            --renders_dir "${COART_DATA_ROOT}/renders_cond" \
            --dino_dir "${COART_DATA_ROOT}/dino_l16_s512" \
            --slat_root "${COART_DATA_ROOT}/slat" \
            --out "${COART_DATA_ROOT}/manifest.csv"
        ;;
    *)
        echo "Unknown STAGE=${STAGE}"; exit 2
        ;;
esac

echo "[run] STAGE=${STAGE} complete"
```

- [ ] **Step 2: Write README**

`scripts/coart_data_v0/README.md`:

```markdown
# coart_data_v0 — DiT Data Pipeline (10K asset milestone)

See spec: `docs/superpowers/specs/2026-05-02-coart-dit-data-pipeline-design.md`
See plan: `docs/superpowers/plans/2026-05-02-coart-dit-data-pipeline.md`

## Quickstart

```bash
# 1. pre-flight: pick 10K sha
.venv/bin/python scripts/coart_data_v0/pick_instances.py \
  --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
  --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
  --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
  --aesthetic_min 4.5 --n 10000 --seed 0 \
  --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_10k.csv

# 2. SS-flow IoU validation gate (~15 min)
.venv/bin/python scripts/coart_compare_occupancy.py \
  --golden_dir datasets/coart_golden \
  --resolutions 64,32,16 \
  --out logs/findings_ss_flow_iou.md
# Read the verdict line; if FAIL, stop here.

# 3. render (5-7 h on 24 GPU)
STAGE=render NODES="117 118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh

# 4. dino (~10-30 min, can overlap with render)
STAGE=dino NODES="118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh

# 5. slat (~20 min)
STAGE=slat NODES="119" NUM_GPUS_PER_NODE=8 \
  VAE_CKPT=results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \
  VAE_TAG=vae_three_branch_ws_v0_ema_s0155000 \
  bash scripts/coart_data_v0/run.sh

# 6. manifest
STAGE=manifest bash scripts/coart_data_v0/run.sh
cat /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/manifest.csv | wc -l
```

## Smoke run (100 assets, < 30 min)

```bash
.venv/bin/python scripts/coart_data_v0/pick_instances.py \
  ...--n 100 --seed 1 --out .../instances_smoke100.csv

INSTANCES=instances_smoke100.csv NODES="119" NUM_GPUS_PER_NODE=1 LIMIT=100 \
  STAGE=render bash scripts/coart_data_v0/run.sh
# repeat for STAGE=dino, slat, manifest
```

## VAE re-encode (when coart.vae ckpt updates)

Only re-run STAGE=slat with a new VAE_TAG; render + dino caches are reusable.

## Output layout

See spec §4.1.

## Re-running on partial failure

All stages are idempotent (resume by checking output existence). Just re-launch
the same command.
```

- [ ] **Step 3: Verify run.sh syntax**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  bash -n scripts/coart_data_v0/run.sh && echo "syntax OK"
```
Expected: `syntax OK`.

- [ ] **Step 4: Dry-run STAGE=manifest（pure local, fastest sanity）**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  COART_DATA_ROOT=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0 \
  INSTANCES=instances_smoke100.csv \
  STAGE=manifest bash scripts/coart_data_v0/run.sh 2>&1 | tail -10
```
Expected: 写出 manifest.csv（即使所有 stage 都未跑，应得到 100 行 with `*_done=False`）。

- [ ] **Step 5: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  chmod +x scripts/coart_data_v0/run.sh && \
  git add scripts/coart_data_v0/run.sh scripts/coart_data_v0/README.md && \
  git commit -m "feat(coart_data_v0): add run.sh stage orchestrator + README

Mirrors precompute_feat18_objaverse_sketchfab.sh env-var pattern.
STAGE=render directly invokes data_toolkit/render_cond.py (no modification);
STAGE=dino/slat/manifest invoke the new scripts.

SSH dispatch via tmp/run_<stage>_rank<R>.sh pattern (per ~/.claude/CLAUDE.md
SSH override). Per-rank logs to <coart_data_root>/logs/.

Dry-run STAGE=manifest succeeds on smoke instances."
```

---

## Task 7: Pre-flight + Smoke run (100 assets end-to-end)

**No new code, but full plan tracking.** This task is the validation gate before production.

- [ ] **Step 1: Disk + ckpt + DinoV3 pre-flight**

```bash
ssh host-10-240-99-119 "bash -lc '
df -h /mnt/novita2/data/video_obj | tail -3
ls -la /mnt/novita2/siyuan/workspace/TRELLIS.2/results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt
ls /tmp/blender-3.0.1-linux-x64/blender 2>&1 | head
'"
```
Expected: ≥ 1.5 TB free; EMA ckpt exists; blender binary exists. **Anyone fails → halt and fix before proceeding**.

- [ ] **Step 2: DinoV3 ckpt download / verify**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -c \"
from transformers import AutoModel
m = AutoModel.from_pretrained(\\\"facebook/dinov3-vitl16-pretrain-lvd1689m\\\")
print(\\\"loaded\\\", sum(p.numel() for p in m.parameters())/1e6, \\\"M params\\\")
\"'"
```
Expected: prints `loaded ~300 M params`. 若网络失败：手动下载到 `pretrained/dinov3-vitl16-pretrain-lvd1689m/` 并设 `HF_HOME` 或 `local_files_only=True`。

- [ ] **Step 3: SS-flow IoU pre-flight gate**

Already done in Task 3 step 7. **Read `logs/findings_ss_flow_iou.md` 的 verdict 行**。
- PASS → proceed
- PASS WITH WARNING → spec 添加 affine 对齐节，本 plan 继续
- FAIL → halt; coart.dit shape DiT spec 需要重谈，把 SS-flow finetune 加入 scope

- [ ] **Step 4: Smoke pick 100**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python scripts/coart_data_v0/pick_instances.py \
    --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
    --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
    --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
    --aesthetic_min 4.5 --n 100 --seed 1 \
    --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_smoke100.csv'"
```

- [ ] **Step 5: Smoke render (1 rank, 1 GPU, ~15-20 min for 100 × 16 views)**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=render INSTANCES=instances_smoke100.csv NODES=119 NUM_GPUS_PER_NODE=1 \
    bash scripts/coart_data_v0/run.sh 2>&1 | tail -20'"
```
Expected: per-rank exit code 0；`renders_cond/<sha>/000.png` 存在。

- [ ] **Step 6: Smoke dino + slat**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=dino INSTANCES=instances_smoke100.csv NODES=119 NUM_GPUS_PER_NODE=1 \
    bash scripts/coart_data_v0/run.sh 2>&1 | tail -10 && \
  STAGE=slat INSTANCES=instances_smoke100.csv NODES=119 NUM_GPUS_PER_NODE=1 \
    VAE_CKPT=results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \
    VAE_TAG=vae_three_branch_ws_v0_ema_s0155000 \
    bash scripts/coart_data_v0/run.sh 2>&1 | tail -10'"
```

- [ ] **Step 7: Smoke manifest + end-to-end shape check**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=manifest INSTANCES=instances_smoke100.csv bash scripts/coart_data_v0/run.sh && \
  .venv/bin/python -c \"
import pandas as pd, numpy as np, json, os
root = \\\"/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0\\\"
m = pd.read_csv(os.path.join(root, \\\"manifest.csv\\\"))
done = m[m[\\\"render_done\\\"] & m[\\\"dino_done\\\"] & m[\\\"slat_done\\\"]]
print(f\\\"{len(done)}/{len(m)} fully cached\\\")
sha = done.iloc[0][\\\"sha256\\\"]
imgs = json.load(open(os.path.join(root, \\\"renders_cond\\\", sha, \\\"transforms.json\\\")))
assert len(imgs[\\\"frames\\\"]) == 16
dino = np.load(os.path.join(root, \\\"dino_l16_s512\\\", f\\\"{sha}.npz\\\"))
assert dino[\\\"features\\\"].shape[0] == 16 and dino[\\\"features\\\"].shape[2] == 1024
slat_dir = os.path.join(root, \\\"slat\\\", \\\"vae_three_branch_ws_v0_ema_s0155000\\\")
slat = np.load(os.path.join(slat_dir, f\\\"{sha}.npz\\\"))
assert slat[\\\"coords\\\"].shape[1] == 3 and slat[\\\"feats\\\"].shape[1] == 32
print(\\\"shape contract OK\\\")
\"'"
```
Expected: `≥ 95/100 fully cached`, `shape contract OK`.

- [ ] **Step 8: Document smoke results in plan progress notes**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  cat > logs/findings_data_v0_smoke.md <<EOF
# Smoke run findings — coart_dit_data_v0 (100 assets)

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Pre-flight verdict
- Disk: PASS / FAIL (... GB free)
- DinoV3 load: PASS / FAIL
- SS-flow IoU @16: ... (PASS / WARN / FAIL)

## Smoke results
- Render success: X/100
- Dino success: X/100
- Slat success: X/100
- Manifest: X fully cached
- Shape contract: PASS / FAIL

## Wall-clock breakdown
- Render: X min (1 rank)
- Dino: X min (1 rank)
- Slat: X min (1 rank)
- Extrapolated 10K production on 24 ranks: X h

## Issues
- ...

## Decision
- Proceed to production / Halt and fix [...]
EOF
echo "Edit logs/findings_data_v0_smoke.md with actual numbers"
```
**人工填写真实数字**，再继续。

- [ ] **Step 9: Commit smoke findings**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add logs/findings_data_v0_smoke.md && \
  git commit -m "docs(coart_data_v0): record smoke run findings (100 assets)

Pre-flight: disk/dino/SSflow-IoU verdicts; smoke success rates;
wall-clock breakdown; production extrapolation."
```

---

## Task 8: Production launch (10K assets) + validation gates

**No new code; this is the actual run + validation per spec §8 + §9.**

- [ ] **Step 1: Pick 10K instances**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python scripts/coart_data_v0/pick_instances.py \
    --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
    --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
    --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
    --aesthetic_min 4.5 --n 10000 --seed 0 \
    --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_10k.csv'"
```

- [ ] **Step 2: Launch render（5-7h, 24 ranks）**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=render INSTANCES=instances_10k.csv NODES=\"117 118 119\" NUM_GPUS_PER_NODE=8 \
    nohup bash scripts/coart_data_v0/run.sh > logs/render_dispatcher.log 2>&1 &
  echo \"dispatched, watch logs/render_dispatcher.log + per-rank logs in coart_dit_data_v0/logs/\"'"
```

监控（每 30 min check）：
```bash
ssh host-10-240-99-119 "ls /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/renders_cond/ | wc -l"
```

**4h 中检 gate**：若进度 < 30%（< 3000 sha），切 CYCLES samples=32 重启（per spec §10 R1）— 这需要 patch `data_toolkit/blender_script/render_cond.py` 的 samples 参数；视为本 plan 的 contingency Task 8.5。

- [ ] **Step 3: Launch dino（render 进度 60%+ 时）**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=dino INSTANCES=instances_10k.csv NODES=\"118 119\" NUM_GPUS_PER_NODE=8 \
    nohup bash scripts/coart_data_v0/run.sh > logs/dino_dispatcher.log 2>&1 &'"
```

- [ ] **Step 4: Launch slat**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=slat INSTANCES=instances_10k.csv NODES=\"119\" NUM_GPUS_PER_NODE=8 \
    VAE_CKPT=results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \
    VAE_TAG=vae_three_branch_ws_v0_ema_s0155000 \
    nohup bash scripts/coart_data_v0/run.sh > logs/slat_dispatcher.log 2>&1 &'"
```

- [ ] **Step 5: Final manifest + validation gates**

```bash
ssh host-10-240-99-119 "bash -lc 'cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  STAGE=manifest INSTANCES=instances_10k.csv bash scripts/coart_data_v0/run.sh && \
  .venv/bin/python -c \"
import pandas as pd, numpy as np, json, random, os
root = \\\"/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0\\\"
m = pd.read_csv(os.path.join(root, \\\"manifest.csv\\\"))
print(f\\\"Total: {len(m)}\\\")
for col in [\\\"render_done\\\", \\\"dino_done\\\", \\\"slat_done\\\"]:
    c = int(m[col].sum())
    print(f\\\"{col}: {c} ({c/len(m):.1%})\\\")
done = m[m[\\\"render_done\\\"] & m[\\\"dino_done\\\"] & m[\\\"slat_done\\\"]]
print(f\\\"All-stage done: {len(done)} ({len(done)/len(m):.1%}) — gate G1 needs ≥ 90%\\\")
random.seed(42)
sample_sha = random.sample(done[\\\"sha256\\\"].tolist(), min(5, len(done)))
for sha in sample_sha:
    dino = np.load(os.path.join(root, \\\"dino_l16_s512\\\", f\\\"{sha}.npz\\\"))
    assert dino[\\\"features\\\"].shape[0] == 16, f\\\"{sha} dino bad\\\"
    slat = np.load(os.path.join(root, \\\"slat\\\", \\\"vae_three_branch_ws_v0_ema_s0155000\\\", f\\\"{sha}.npz\\\"))
    assert slat[\\\"coords\\\"].shape[1] == 3 and slat[\\\"feats\\\"].shape[1] == 32
print(\\\"G2/G3/G4 sample assert PASSED\\\")
df_show = m[\\\"slat_tag\\\"].value_counts().head()
print(\\\"slat_tag dist:\\\", df_show.to_dict())
\"'"
```

通过条件：
- G1（completeness）: ≥ 9000 / 10000
- G2/G3/G4（shape contract）: 5/5 sample assert pass
- G5（SS-flow IoU）: 已在 Task 3 done
- G6（disk）: `df -h` 仍 ≥ 200 GB free

- [ ] **Step 6: Document production results**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  cat > logs/findings_data_v0_production.md <<EOF
# Production run findings — coart_dit_data_v0 (10K assets)

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Wall-clock
- Render: X h (24 ranks)
- Dino: X min (16 ranks)
- Slat: X min (8 ranks)
- Total: X h

## Validation gates
- G1 completeness: X/10000 = X% (target ≥ 90%)
- G2 dino shape: PASS
- G3 slat-vs-feat18 voxel count: PASS
- G4 end-to-end load: PASS
- G5 SS-flow IoU: see logs/findings_ss_flow_iou.md
- G6 disk free: X GB

## Failure breakdown
- Render fail: X (top reasons: ...)
- Dino fail: X
- Slat fail: X

## Disk usage
- renders_cond/: X GB
- dino_l16_s512/: X GB
- slat/: X GB

## Decision
- Production cache ready for shape DiT trainer
- Next: write coart.dit shape DiT spec + plan
EOF
echo "Edit logs/findings_data_v0_production.md with actual numbers"
```

- [ ] **Step 7: Commit production findings**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add logs/findings_data_v0_production.md && \
  git commit -m "docs(coart_data_v0): record production run findings (10K assets)

Wall-clock breakdown, validation gate results, failure analysis.
Cache layout ready for downstream DiT trainer (next spec)."
```

---

## Plan Summary

| Task | What | Wall-clock | Dependencies |
|---|---|---|---|
| 1 | pick_instances.py + tests | 30 min | none |
| 2 | build_manifest.py + tests | 30 min | none |
| 3 | coart_compare_occupancy.py + IoU validation | 1 h | none (uses corep_fast + o_voxel) |
| 4 | coart_cache_dino.py + tests | 1 h | DinoV3 ckpt loadable |
| 5 | coart_cache_slat.py + tests | 1 h | coart EMA ckpt + stats_global.npz |
| 6 | run.sh + README.md | 30 min | tasks 1-5 done |
| 7 | Pre-flight + smoke run | 1 h | tasks 1-6 + IoU PASS |
| 8 | Production 10K + validation | 8-10 h | task 7 PASS |

**Total dev time before launch: ~5 h. Production wall-clock: ~10 h (asynchronous after launch).**

After Task 8 PASS, the next spec is `docs/superpowers/specs/2026-05-XX-coart-dit-shape-finetune-design.md`（已 lock 的 defaults: warm-start from official 1.3B ckpt, lr=2e-5, no freeze, batch=8/GPU, max_steps placeholder 200K, dino cache → online-skip optimization）.
