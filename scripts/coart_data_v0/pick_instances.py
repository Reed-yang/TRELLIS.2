#!/usr/bin/env python
"""Pre-flight: pick N sha256 from sketchfab subset that have feat18 + local_path
+ aesthetic >= threshold; deterministic via --seed; write instances_{N}k.csv.

Usage:
  python scripts/coart_data_v0/pick_instances.py \\
    --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \\
    --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \\
    --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \\
    --aesthetic_min 4.5 --n 10000 --seed 0 \\
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
