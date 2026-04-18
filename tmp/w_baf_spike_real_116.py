"""W_BAF spike: Triton vs legacy using ACTUAL F2 pipeline (edge_weights, uturn).

Captures the (ew, uturn) args to _build_adjacency_gpu during pipeline run,
then bench both impls on those real tensors.
"""
import sys, time
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

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

import torch
import trimesh
from corep_fast.pipeline import corep_pipeline
import corep_fast.stages.s7_rank_assign as s7
from corep_fast.stages.s7_rank_assign import (
    _W_MAX, _NODES_PER_CUBE, _facet_pair_table, CUBE_FACETS,
)
from corep_fast.stages.s7_triton import build_adjacency_triton_fast_only

# Capture args
_orig = s7._build_adjacency_gpu
_captured = []


def _capture(*args, **kwargs):
    # Clone for later benchmarking (avoid aliasing)
    ew_clone = args[0].detach().clone()
    uturn_clone = args[1].detach().clone()
    _captured.append((ew_clone, uturn_clone))
    return _orig(*args, **kwargs)


s7._build_adjacency_gpu = _capture

mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_baf.ply')
_ = corep_pipeline('/tmp/fix_baf.ply', 256, torch.device('cuda:0'))

print(f"Captured {len(_captured)} calls")

if not _captured:
    raise RuntimeError("no capture")

# Use the first (main) call
ew, uturn = _captured[0]
N = int(ew.shape[0])
print(f"Call 0: N={N}")

# Fraction of fast-path cubes
is_fast = (uturn[:, 0, 0] == -1)
frac_fast = float(is_fast.float().mean())
print(f"Fast-path fraction: {frac_fast:.4f} ({int(is_fast.sum())}/{N})")

device = ew.device
eA_tab, eB_tab, a_v0, b_v0 = _facet_pair_table(device)
facets = CUBE_FACETS.to(device=device, dtype=torch.int64)
eC_tab = torch.stack([facets[:, 2], facets[:, 0], facets[:, 1]], dim=1).contiguous()


def _canonicalize(adj):
    valid = adj >= 0
    key = torch.where(valid, adj, torch.full_like(adj, _NODES_PER_CUBE))
    key, _ = key.sort(dim=-1)
    return torch.where(key == _NODES_PER_CUBE, torch.full_like(key, -1), key)


def time_call(fn, args, warmup=3, runs=10):
    for _ in range(warmup):
        fn(*args); torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        out = fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return out, sorted(times)[len(times) // 2]


def _t_call(ew_, uturn_):
    return build_adjacency_triton_fast_only(
        ew_, eA_tab, eB_tab, eC_tab, a_v0, b_v0,
        NODES=_NODES_PER_CUBE, W=_W_MAX,
    )[0]


# Bench legacy
adj_legacy, t_legacy = time_call(_orig, (ew, uturn))
# Bench triton (fast-path only impl; if slow-path cubes exist, triton
# results will be wrong for those -- but spike assumption is fast-only)
adj_triton, t_triton = time_call(_t_call, (ew, uturn))

print(f"\nReal F2 data @ N={N}, fast-frac={frac_fast:.3f}:")
print(f"  Legacy median wall: {t_legacy:.3f} ms")
print(f"  Triton median wall: {t_triton:.3f} ms")
print(f"  Speedup: {t_legacy / max(t_triton, 1e-6):.2f}x")
print(f"  Saved per call: {t_legacy - t_triton:.3f} ms")

# Parity only on fast-path cubes
fast_mask = is_fast
adj_legacy_fast = adj_legacy[fast_mask]
adj_triton_fast = adj_triton[fast_mask]
lc = _canonicalize(adj_legacy_fast)
tc = _canonicalize(adj_triton_fast)
mm_fast = int((lc != tc).sum())
n_fast = int(fast_mask.sum())
print(f"\nFast-only parity: {mm_fast} mismatches out of {n_fast} cubes "
      f"(frac {mm_fast / max(n_fast * _NODES_PER_CUBE * 2, 1):.6f})")

if mm_fast > 0:
    diffs = (lc != tc).nonzero()[:5]
    print(f"  First mismatches: {diffs.tolist()}")
    c0, n0, s0 = diffs[0].tolist()
    print(f"  At fast-cube[{c0}] node[{n0}]: legacy={lc[c0, n0].tolist()}, "
          f"triton={tc[c0, n0].tolist()}")
    print("  PARITY (fast only): FAIL")
else:
    print("  PARITY (fast only): OK")
