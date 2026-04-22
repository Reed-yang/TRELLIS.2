"""
Aggregate per-channel mean/std across all successfully-encoded npz files.

Run AFTER precompute_feat18.py with --skip_done_fast finishes. The fast
resume path skips the per-file stats accumulation that the original code
did inline during encode, so call this once at the end to produce a
single global stats_global.npz matching the schema of the per-rank
stats_rank<r>.npz the inline code would have written.

Uses multiprocessing to load npz files in parallel; tested at ~400 files/s
on a 128-core box.

Output: OUT_DIR/stats_global.npz with keys mean, std, n_voxels.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
from tqdm import tqdm


N_FEAT = 18


def _worker(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        d = np.load(path)
        f = d["feats"].astype(np.float64)
    except Exception:
        return (np.zeros(N_FEAT), np.zeros(N_FEAT), 0)
    return (f.sum(axis=0), (f * f).sum(axis=0), int(f.shape[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir",
                   default="/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out",
                   default=None,
                   help="stats output path (default: OUT_DIR/stats_global.npz)")
    args = p.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"[stats] no data dir at {data_dir}")
    out_path = args.out or os.path.join(args.out_dir, "stats_global.npz")

    print(f"[stats] scanning {data_dir} ...", flush=True)
    t0 = time.time()
    with os.scandir(data_dir) as it:
        paths = [e.path for e in it if e.name.endswith(".npz")]
    print(f"[stats] found {len(paths)} npz in {time.time() - t0:.1f}s", flush=True)

    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 4)
    sums = np.zeros(N_FEAT, dtype=np.float64)
    sumsqs = np.zeros(N_FEAT, dtype=np.float64)
    n_voxels = 0
    bad = 0
    chunksize = max(16, len(paths) // (n_workers * 32) or 16)
    t1 = time.time()
    with mp.Pool(n_workers) as pool:
        for s, sq, nv in tqdm(
            pool.imap_unordered(_worker, paths, chunksize=chunksize),
            total=len(paths),
            desc="agg",
            dynamic_ncols=True,
        ):
            if nv == 0:
                bad += 1
                continue
            sums += s
            sumsqs += sq
            n_voxels += nv

    if n_voxels == 0:
        raise SystemExit("[stats] no voxels aggregated, aborting")

    mean = sums / n_voxels
    var = sumsqs / n_voxels - mean * mean
    std = np.sqrt(np.maximum(var, 1e-6))

    # Match original finaliser: first 6 point-coord channels use (x-0.5)
    # not z-scoring, so the pretrained encoder's input layer still matches.
    mean[:6] = 0.5
    std[:6] = 1.0

    np.savez(out_path,
             mean=mean.astype(np.float32),
             std=std.astype(np.float32),
             n_voxels=np.int64(n_voxels))
    print(f"[stats] wrote {out_path}", flush=True)
    print(f"[stats] aggregated {len(paths) - bad} / {len(paths)} files in "
          f"{time.time() - t1:.1f}s (bad={bad})", flush=True)
    print(f"[stats] n_voxels = {n_voxels:,}", flush=True)
    print(f"[stats] mean = {np.round(mean, 4).tolist()}", flush=True)
    print(f"[stats] std  = {np.round(std, 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
