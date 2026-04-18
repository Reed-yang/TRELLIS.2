"""W_BAF spike: Triton fast-path adjacency vs legacy PyTorch."""
import sys, time
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

import numpy as np
import torch

from corep_fast.stages.s7_rank_assign import (
    _build_adjacency_gpu, _W_MAX, _NODES_PER_CUBE, _facet_pair_table, CUBE_FACETS,
)
from corep_fast.stages.s7_triton import build_adjacency_triton_fast_only, _TRITON_AVAILABLE

if not _TRITON_AVAILABLE:
    raise RuntimeError("Triton not installed")

device = torch.device('cuda:0')


def _canonicalize(adj):
    """Sort each (cube, src_node, :) so slot 0 < slot 1; -1 sentinels at end."""
    valid = adj >= 0
    key = torch.where(valid, adj, torch.full_like(adj, _NODES_PER_CUBE))
    key, _ = key.sort(dim=-1)
    return torch.where(key == _NODES_PER_CUBE, torch.full_like(key, -1), key)


def time_call(fn, args, kwargs, warmup=3, runs=10):
    for _ in range(warmup):
        fn(*args, **kwargs); torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    return out, sorted(times)[len(times) // 2]


# Tables (compute once)
eA_tab, eB_tab, a_v0, b_v0 = _facet_pair_table(device)
facets = CUBE_FACETS.to(device=device, dtype=torch.int64)
eC_tab = torch.stack([facets[:, 2], facets[:, 0], facets[:, 1]], dim=1).contiguous()


def _t_call(ew_, uturn_):
    return build_adjacency_triton_fast_only(
        ew_, eA_tab, eB_tab, eC_tab, a_v0, b_v0,
        NODES=_NODES_PER_CUBE, W=_W_MAX,
    )[0]


print("=" * 70)
print("Spike at multiple N to see scaling")
print("=" * 70)

for N_test in [1000, 10000, 100000, 275541]:
    gen = torch.Generator(device='cpu').manual_seed(42)
    ew = torch.randint(0, _W_MAX, (N_test, 18), generator=gen, dtype=torch.int64).to(device)
    uturn = torch.full((N_test, 12, 3), -1, dtype=torch.int64, device=device)  # all fast-path

    # Time legacy
    adj_legacy, t_legacy = time_call(_build_adjacency_gpu, (ew, uturn), {})
    # Time triton
    adj_triton, t_triton = time_call(_t_call, (ew, uturn), {})

    # Parity check
    adj_legacy_c = _canonicalize(adj_legacy)
    adj_triton_c = _canonicalize(adj_triton)
    mismatches = int((adj_legacy_c != adj_triton_c).sum())

    print(f"\nN={N_test}:")
    print(f"  Legacy median wall: {t_legacy:.3f} ms")
    print(f"  Triton median wall: {t_triton:.3f} ms")
    print(f"  Speedup: {t_legacy / max(t_triton, 1e-6):.2f}x")
    print(f"  Saved per call: {t_legacy - t_triton:.3f} ms")
    print(f"  Mismatched cells (after canonical sort): {mismatches}")

    if mismatches > 0:
        diffs = (adj_legacy_c != adj_triton_c).nonzero()[:5]
        print(f"  First mismatches: {diffs.tolist()}")
        # Show actual values for the first cube/node
        c0, n0, s0 = diffs[0].tolist()
        print(f"  At [{c0}, {n0}]: legacy={adj_legacy_c[c0, n0].tolist()}, "
              f"triton={adj_triton_c[c0, n0].tolist()}")
        print("  PARITY: FAIL")
    else:
        print("  PARITY: OK")
