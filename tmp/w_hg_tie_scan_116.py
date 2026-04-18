"""W_HG tie scan: count cost-matrix ties in Phase 3 of s7_rank_assign on F2."""
import sys
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

# SerialPool monkeypatch (regression-gate determinism)
import multiprocessing as _mp, multiprocessing.pool as _mp_pool


class _SerialPool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, items, *a, **kw): return [fn(x) for x in items]
    def imap(self, fn, items, *a, **kw):
        for x in items: yield fn(x)
    def imap_unordered(self, fn, items, *a, **kw):
        for x in items: yield fn(x)
    def starmap(self, fn, items, *a, **kw): return [fn(*x) for x in items]
    def close(self): pass
    def terminate(self): pass
    def join(self): pass


_mp.Pool = _SerialPool
_mp_pool.Pool = _SerialPool

import numpy as np
import torch
import trimesh
from itertools import permutations

# Hook into Phase 3's linear_sum_assignment call via module-level patch
import corep_fast.stages.s7_rank_assign as s7
from scipy.optimize import linear_sum_assignment as _orig_lsa

stats = {
    "calls": 0,
    "with_ties": 0,
    "tie_count_total": 0,
    "matrix_shapes": [],
    "exact_min_collisions": [],   # for matrices with ties, how many other permutations tie with the optimum
    "skipped_oob": 0,
    "rect_reverse": 0,            # npts < nl (rectangular reverse) — must scipy-fallback
}

# Cap brute-force enumeration to keep wall time bounded.
MAX_BRUTE_CALLS = 100000
_brute_calls = [0]


def _hook(cost, *args, **kwargs):
    stats["calls"] += 1
    nl, npts = cost.shape
    stats["matrix_shapes"].append((int(nl), int(npts)))

    if npts < nl:
        stats["rect_reverse"] += 1

    if nl <= 5 and npts <= 8 and _brute_calls[0] < MAX_BRUTE_CALLS:
        _brute_calls[0] += 1
        # Brute-force enumerate all permutations and check ties at the optimum
        choices = list(permutations(range(npts), nl))
        costs_per_choice = []
        for c in choices:
            tot = 0.0
            for r, col in enumerate(c):
                tot += cost[r, col]
            costs_per_choice.append(tot)
        if costs_per_choice:
            best = min(costs_per_choice)
            tie_count = sum(1 for x in costs_per_choice if abs(x - best) < 1e-12)
            if tie_count > 1:
                stats["with_ties"] += 1
                stats["exact_min_collisions"].append(tie_count)
    elif nl > 5 or npts > 8:
        stats["skipped_oob"] += 1

    return _orig_lsa(cost, *args, **kwargs)


s7.linear_sum_assignment = _hook

def _triple_icosphere() -> trimesh.Trimesh:
    parts = []
    for r in (1.0, 0.7, 0.4):
        parts.append(trimesh.creation.icosphere(subdivisions=3, radius=r * 0.4))
    return trimesh.util.concatenate(parts)


def _reset_stats():
    stats["calls"] = 0
    stats["with_ties"] = 0
    stats["tie_count_total"] = 0
    stats["matrix_shapes"] = []
    stats["exact_min_collisions"] = []
    stats["skipped_oob"] = 0
    stats["rect_reverse"] = 0
    _brute_calls[0] = 0


def _summarize(label):
    print(f"\n=== {label} ===")
    print(f"Phase 3 calls: {stats['calls']}")
    print(f"Brute-force enumerated: {_brute_calls[0]} (cap={MAX_BRUTE_CALLS})")
    if _brute_calls[0] >= MAX_BRUTE_CALLS:
        print(f"[truncated_at_{MAX_BRUTE_CALLS}]")
    print(f"Skipped (out of brute-force scope nl>5 or npts>8): {stats['skipped_oob']}")
    print(f"Rect-reverse (npts < nl, must scipy-fallback): {stats['rect_reverse']}")
    print(f"With ties (multiple permutations achieve min cost): {stats['with_ties']}")
    denom = max(_brute_calls[0], 1)
    print(f"Tie pct (over brute-enumerated): {100.0 * stats['with_ties'] / denom:.4f}%")

    if stats["exact_min_collisions"]:
        arr = np.array(stats["exact_min_collisions"])
        print(f"Tie counts (when ties exist): min={arr.min()} max={arr.max()} mean={arr.mean():.2f} p99={np.percentile(arr, 99):.0f}")

    shapes = np.array(stats["matrix_shapes"])
    if shapes.size > 0:
        print(f"Matrix shape distribution:")
        print(f"  nl: min={shapes[:,0].min()} max={shapes[:,0].max()} p50={int(np.percentile(shapes[:,0],50))} p99={int(np.percentile(shapes[:,0],99))} p999={int(np.percentile(shapes[:,0],99.9))}")
        print(f"  npts: min={shapes[:,1].min()} max={shapes[:,1].max()} p50={int(np.percentile(shapes[:,1],50))} p99={int(np.percentile(shapes[:,1],99))} p999={int(np.percentile(shapes[:,1],99.9))}")
        print(f"  cubes with nl > 5 (out of brute-force scope): {(shapes[:,0] > 5).sum()}")
        print(f"  cubes with npts > 8: {(shapes[:,1] > 8).sum()}")
        # Distribution: count of (nl, npts) buckets up to (5, 8)
        counts_55 = {}
        for nl, np_ in shapes:
            if nl <= 5 and np_ <= 8:
                counts_55[(int(nl), int(np_))] = counts_55.get((int(nl), int(np_)), 0) + 1
        print("  shape histogram (nl<=5, npts<=8):")
        for k in sorted(counts_55.keys()):
            print(f"    nl={k[0]} npts={k[1]}: {counts_55[k]}  ({100.0*counts_55[k]/shapes.shape[0]:.4f}%)")


# F2: icosphere s3 @ res=256
mesh_f2 = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh_f2.export('/tmp/fix_tie_f2.ply')
from corep_fast.pipeline import corep_pipeline
_reset_stats()
_ = corep_pipeline('/tmp/fix_tie_f2.ply', 256, torch.device('cuda:0'))
_summarize('F2 (icosphere s3 @ res=256)')

# F3: triple icosphere @ res=128 (covers multi-loop s7 path)
mesh_f3 = _triple_icosphere()
mesh_f3.export('/tmp/fix_tie_f3.ply')
_reset_stats()
_ = corep_pipeline('/tmp/fix_tie_f3.ply', 128, torch.device('cuda:0'))
_summarize('F3 (triple icosphere @ res=128, multi-loop path)')
