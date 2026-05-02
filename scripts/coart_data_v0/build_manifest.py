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
    """Existence + lightweight integrity check.

    Avoid decompressing the (large) ``features`` array. Prior version called
    ``z['features'].shape`` which forces a full decompression of the 16x1029x1024
    fp16 tensor (~33MB). At 10K assets that's ~340GB of NFS read and runs for
    many minutes. ``n_tokens`` is a 0-d int32 scalar — cheap to decompress and
    its presence with value > 0 confirms the npz was written successfully.
    """
    p = os.path.join(dino_dir, f"{sha}.npz")
    if not os.path.isfile(p):
        return False
    try:
        with np.load(p) as z:
            return int(z["n_tokens"]) > 0
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
    tag, p = candidates[-1]
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
    print(f"[manifest] wrote {n} rows -> {out}")
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
