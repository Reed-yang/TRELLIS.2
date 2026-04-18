# corep_fast/tests/unit/test_s6_fastpath_tracer.py
"""Equivalence test: numpy tracer output == GPU tracer output, per cube.

This test locks in the bit-exact contract for T5c/T5d (GPU rewrite). It
compares `_fastpath_trace_loops_gpu` against the numpy reference on a
small hand-crafted adjacency matrix. Loop start point is
implementation-defined; loops-as-multisets must match.

Design sketch: tmp/cpu_worker_optim_design/t5a_s6_fastpath_gpu_sketch.md
"""
import numpy as np
import pytest
import torch

from corep_fast.stages.s6_collapse import _fastpath_trace_loops_numpy


def _make_synthetic_cube(total_points: int, adjacency_edges: list[tuple[int, int]],
                          point_offset_row: np.ndarray):
    """Build adj_row from edge list. Each point has degree <= 2."""
    max_points = len(point_offset_row) - 1  # sentinel semantics
    adj = np.full((max_points, 2), -1, dtype=np.int32)
    slot = np.zeros(max_points, dtype=np.int32)
    for a, b in adjacency_edges:
        adj[a, slot[a]] = b; slot[a] += 1
        adj[b, slot[b]] = a; slot[b] += 1
    return adj


# Single closed 4-cycle: 0-1-2-3-0
SIMPLE_4CYCLE = {
    "point_offset_row": np.array([0, 1, 2, 3, 4] + [4]*14, dtype=np.int64),
    "adjacency_edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "total_points": 4,
}

# Two disjoint 2-cycles (edge doubled — not physically meaningful but
# exercises multi-loop path within a single cube).
TWO_LOOPS = {
    "point_offset_row": np.array([0, 1, 2, 3, 4] + [4]*14, dtype=np.int64),
    "adjacency_edges": [(0, 1), (1, 0), (2, 3), (3, 2)],
    "total_points": 4,
}


@pytest.mark.parametrize("case_name,case", [
    ("simple_4cycle", SIMPLE_4CYCLE),
    ("two_loops",     TWO_LOOPS),
])
def test_gpu_tracer_matches_numpy(case_name, case):
    """GPU tracer output must match numpy on multiset-of-loops basis."""
    adj = _make_synthetic_cube(
        case["total_points"], case["adjacency_edges"], case["point_offset_row"])

    # Reference
    expected = _fastpath_trace_loops_numpy(
        case["point_offset_row"], adj, case["total_points"])

    # GPU impl (does not exist yet — test should fail at import in T5b,
    # then pass once T5c ships _fastpath_trace_loops_gpu)
    from corep_fast.stages.s6_collapse import _fastpath_trace_loops_gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    po_gpu  = torch.from_numpy(case["point_offset_row"]).unsqueeze(0).to(device)
    adj_gpu = torch.from_numpy(adj).unsqueeze(0).to(device)
    tot_gpu = torch.tensor([case["total_points"]], dtype=torch.int64, device=device)

    loop_count, loop_offsets, edge_ids = _fastpath_trace_loops_gpu(
        po_gpu, adj_gpu, tot_gpu)

    # Reconstruct GPU CSR output to per-cube list of loops, compare as
    # multisets of multisets. Start point and edge order within a loop
    # are implementation-defined.
    cube0_count = int(loop_count[0].item())
    got = []
    for k in range(cube0_count):
        lo = int(loop_offsets[0, k].item())
        hi = int(loop_offsets[0, k + 1].item())
        got.append(sorted(edge_ids[lo:hi].cpu().tolist()))
    expected_sorted = [sorted(l) for l in expected]
    assert sorted(got) == sorted(expected_sorted), (
        f"GPU loops != numpy loops (case={case_name})\n"
        f"got={got}\nexpected={expected_sorted}"
    )


def test_empty_cube_produces_zero_loops():
    """Cube with 0 active points -> no loops."""
    from corep_fast.stages.s6_collapse import _fastpath_trace_loops_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    po  = torch.zeros((1, 19), dtype=torch.int64, device=device)
    adj = torch.full((1, 4, 2), -1, dtype=torch.int32, device=device)
    tot = torch.zeros((1,), dtype=torch.int64, device=device)
    loop_count, loop_offsets, edge_ids = _fastpath_trace_loops_gpu(po, adj, tot)
    assert int(loop_count[0].item()) == 0
    assert edge_ids.numel() == 0
