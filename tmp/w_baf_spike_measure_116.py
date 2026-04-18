"""W_BAF spike: measure actual _build_adjacency_gpu wall time on F2 fixture."""
import sys, time
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

# SerialPool monkeypatch (regression-gate determinism for measurement consistency)
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

# Wrap _build_adjacency_gpu to time it
_orig = s7._build_adjacency_gpu
_timings = {"calls": 0, "wall_ms_total": 0.0, "Ns": []}


def _timed(*args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    r = _orig(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    _timings["calls"] += 1
    _timings["wall_ms_total"] += (t1 - t0) / 1e6
    if len(args) > 0 and hasattr(args[0], "shape"):
        _timings["Ns"].append(int(args[0].shape[0]))
    return r


s7._build_adjacency_gpu = _timed

mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_baf.ply')
_ = corep_pipeline('/tmp/fix_baf.ply', 256, torch.device('cuda:0'))

print(f"_build_adjacency_gpu calls: {_timings['calls']}")
print(f"Total wall: {_timings['wall_ms_total']:.2f} ms")
if _timings['Ns']:
    print(f"N values: {_timings['Ns']}")
    print(f"Avg ms/call: {_timings['wall_ms_total'] / _timings['calls']:.2f}")
