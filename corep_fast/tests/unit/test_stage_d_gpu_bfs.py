"""W_SD: _count_uturns_gpu_batched must match numpy _count_uturns per group."""
import numpy as np
import pytest
import torch

from corep_fast.stages.s4_face_point import (
    _count_uturns,
    _count_uturns_gpu_batched,  # NEW symbol — unresolved until Task 7 stub
)


def _fake_group(n_segs, seed=0):
    rng = np.random.RandomState(seed)
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    segs = []
    for _ in range(n_segs):
        a = rng.rand(3) * 0.5
        b = rng.rand(3) * 0.5
        a[2] = 0.0
        b[2] = 0.0
        segs.append((a, b))
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    return segs, V0, V1, V2, cube_verts


def test_matches_legacy_random_100_groups():
    rng = np.random.RandomState(0)
    groups_legacy = []
    groups_gpu_input = []
    for seed in range(100):
        n_segs = int(rng.randint(1, 10))
        segs, V0, V1, V2, cube_verts = _fake_group(n_segs, seed=int(rng.randint(1<<30)))
        vert_ids = (0, 1, 2)
        edge_ids = (0, 1, 2)
        legacy_uturn = _count_uturns(segs, V0, V1, V2, cube_verts, vert_ids, edge_ids)
        groups_legacy.append(legacy_uturn)
        groups_gpu_input.append((segs, V0, V1, V2, cube_verts, vert_ids, edge_ids))

    gpu_result = _count_uturns_gpu_batched(groups_gpu_input)
    assert gpu_result.shape == (100,)
    for i in range(100):
        assert int(gpu_result[i]) == groups_legacy[i], \
            f"group {i}: gpu={int(gpu_result[i])} vs legacy={groups_legacy[i]}"


def test_zero_segments_group_zero_uturn():
    empty = [([], np.zeros(3), np.array([1.,0,0]), np.array([0,1.,0]),
              np.zeros((8, 3)), (0,1,2), (0,1,2))]
    result = _count_uturns_gpu_batched(empty)
    assert int(result[0]) == 0


def test_single_edge_two_endpoints_same_edge_one_uturn():
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    p1 = np.array([0.2, 0.0, 0.0])
    p2 = np.array([0.7, 0.0, 0.0])
    segs = [(p1, p2)]
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    group = (segs, V0, V1, V2, cube_verts, (0,1,2), (10, 20, 30))
    result = _count_uturns_gpu_batched([group])
    legacy = _count_uturns(*group)
    assert int(result[0]) == legacy


def test_closed_loop_zero_uturn():
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    a = np.array([0.1, 0.1, 0.0])
    b = np.array([0.3, 0.1, 0.0])
    c = np.array([0.2, 0.3, 0.0])
    segs = [(a, b), (b, c), (c, a)]
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    group = (segs, V0, V1, V2, cube_verts, (0,1,2), (10, 20, 30))
    result = _count_uturns_gpu_batched([group])
    assert int(result[0]) == 0


def test_max_s_convergence():
    """n_segs=8 (P=16) chain of unique endpoints — graph diameter ~ P-1.

    Verifies the label-propagation loop cap (now P, not 16) handles groups
    where max_s >= 8. Constructs a chain of 8 segments laid along the V0-V1
    edge (so endpoints coalesce pairwise into a connected chain on the edge)
    and compares to the legacy reference.
    """
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    # 8 segments forming a chain along the V0-V1 edge, each sharing an
    # endpoint with the next (e.g. (p0,p1), (p1,p2), ..., (p7,p8)).
    xs = np.linspace(0.05, 0.95, 9)
    segs = [
        (np.array([xs[i], 0.0, 0.0]), np.array([xs[i + 1], 0.0, 0.0]))
        for i in range(8)
    ]
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    group = (segs, V0, V1, V2, cube_verts, (0, 1, 2), (10, 20, 30))
    legacy = _count_uturns(*group)
    result = _count_uturns_gpu_batched([group])
    assert result.shape == (1,)
    assert int(result[0]) == legacy, \
        f"max_s=8 chain: gpu={int(result[0])} vs legacy={legacy}"


def test_large_batch_50000():
    rng = np.random.RandomState(42)
    groups = []
    for _ in range(50_000):
        n_segs = int(rng.randint(1, 8))
        segs, V0, V1, V2, cube_verts = _fake_group(n_segs, seed=int(rng.randint(1<<30)))
        groups.append((segs, V0, V1, V2, cube_verts, (0,1,2), (0,1,2)))
    result = _count_uturns_gpu_batched(groups)
    assert result.shape == (50_000,)
    for i in range(0, 50_000, 5000):
        legacy = _count_uturns(*groups[i])
        assert int(result[i]) == legacy
