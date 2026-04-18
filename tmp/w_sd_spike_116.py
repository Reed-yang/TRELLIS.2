"""W_SD spike: profile _count_uturns input sizes on F2 fixture.

Goal: determine max segments-per-group and max unique-nodes-per-group.
Drives (G, P, P) memory budget for the batched cdist in Task 8.
"""
import sys
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

# Monkeypatch SerialPool to match regression-gate determinism
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool


class _SerialPool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, items, *a, **kw): return [fn(x) for x in items]
    def imap(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)
    def imap_unordered(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)
    def starmap(self, fn, items, *a, **kw): return [fn(*x) for x in items]
    def close(self): pass
    def terminate(self): pass
    def join(self): pass


_mp.Pool = _SerialPool
_mp_pool.Pool = _SerialPool

import numpy as np
import torch
import trimesh
from corep_fast.pipeline import corep_pipeline
import corep_fast.stages.s4_face_point as s4

_group_stats = {"sizes": [], "nodes": []}

_orig = s4._count_uturns


def _instrumented(segments, V0, V1, V2, cube_verts, vert_ids, edge_ids):
    _group_stats["sizes"].append(len(segments))
    # Replicate _find_or_add_node logic to count unique nodes
    nodes = []
    for p1, p2 in segments:
        for pt in (p1, p2):
            found = False
            for n in nodes:
                if np.linalg.norm(pt - n) < 1e-8:
                    found = True
                    break
            if not found:
                nodes.append(pt)
    _group_stats["nodes"].append(len(nodes))
    return _orig(segments, V0, V1, V2, cube_verts, vert_ids, edge_ids)


s4._count_uturns = _instrumented

mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_spike.ply')
_ = corep_pipeline('/tmp/fix_spike.ply', 256, torch.device('cuda:0'))

sizes = np.asarray(_group_stats["sizes"], dtype=np.int64)
nodes = np.asarray(_group_stats["nodes"], dtype=np.int64)
print(f"groups = {len(sizes)}")
if len(sizes) > 0:
    print(f"segs/group: min={int(sizes.min())} max={int(sizes.max())} "
          f"mean={sizes.mean():.2f} p50={int(np.percentile(sizes,50))} "
          f"p95={int(np.percentile(sizes,95))} p99={int(np.percentile(sizes,99))}")
    print(f"nodes/group: min={int(nodes.min())} max={int(nodes.max())} "
          f"mean={nodes.mean():.2f} p50={int(np.percentile(nodes,50))} "
          f"p95={int(np.percentile(nodes,95))} p99={int(np.percentile(nodes,99))}")
    P_MAX = 2 * int(sizes.max())
    print(f"P_MAX (= 2 * max_segs) = {P_MAX}")
    print(f"(G, P, P) float64 bytes = {len(sizes) * P_MAX * P_MAX * 8:,} bytes "
          f"= {len(sizes) * P_MAX * P_MAX * 8 / 1024**3:.2f} GB")
