"""Equivalence test: _get_local_components_gpu vs _get_local_components_np.

Locks the bit-exact contract for T6c/T6d. Per T6a spike findings
(tmp/cpu_worker_optim_design/t6a_s4_uf_gpu_sketch.md), the numpy
version's externally-visible outputs are:
  (a) component ordering (first-occurrence by face slot)
  (b) face ordering within each component (slot-ascending)
Both can be reproduced by a GPU label-propagation seeded with
labels[j] = j + stable-sort-by-label output step. This test compares
as SETS of SETS first (fastest failure signal); T6d adds a stronger
ordering check against the full pipeline golden gate.

NOTE ON face_adj SEMANTICS:
`face_adj` is a GLOBAL face->neighbor table (audit-confirmed per T6a
spike + numpy impl at corep_fast/stages/s4_face_point.py:764-769 —
neighbors stored as GLOBAL face ids, not local indices). Shape is
`(max_face_id + 1, 3)` and rows are indexed by global face id. The GPU
signature `_get_local_components_gpu(face_ids, face_adj)` receives the
SAME global table as the numpy implementation.
"""
import numpy as np
import pytest
import torch

from corep_fast.stages.s4_face_point import _get_local_components_np


@pytest.fixture
def tiny_triangle_adj():
    """Three triangles with global face ids {10,11,12}.
    10 and 11 share an edge; 12 is isolated. Expected: 2 components.
    """
    face_ids = np.array([10, 11, 12], dtype=np.int32)
    # mesh_faces is a dead arg in numpy impl (per T6a); shape doesn't matter.
    mesh_faces = np.zeros((13, 3), dtype=np.int32)
    # Global face->3-neighbor table, sized to cover max face id.
    face_adj = np.full((13, 3), -1, dtype=np.int32)
    face_adj[10] = [11, -1, -1]
    face_adj[11] = [10, -1, -1]
    face_adj[12] = [-1, -1, -1]
    return face_ids, mesh_faces, face_adj


@pytest.fixture
def four_face_chain():
    """Linear chain 100-101-102-103 (global ids). One component."""
    face_ids = np.array([100, 101, 102, 103], dtype=np.int32)
    mesh_faces = np.zeros((104, 3), dtype=np.int32)
    face_adj = np.full((104, 3), -1, dtype=np.int32)
    face_adj[100] = [101, -1, -1]
    face_adj[101] = [100, 102, -1]
    face_adj[102] = [101, 103, -1]
    face_adj[103] = [102, -1, -1]
    return face_ids, mesh_faces, face_adj


@pytest.fixture
def empty_input():
    return (np.array([], dtype=np.int32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.int32))


def _as_set_of_sets(components):
    return {frozenset(int(x) for x in comp) for comp in components}


def test_gpu_uf_matches_numpy_two_components(tiny_triangle_adj):
    face_ids, mesh_faces, face_adj = tiny_triangle_adj
    expected = _get_local_components_np(face_ids, mesh_faces, face_adj)

    from corep_fast.stages.s4_face_point import _get_local_components_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fids_gpu = torch.from_numpy(face_ids).to(device)
    fadj_gpu = torch.from_numpy(face_adj).to(device)

    got = _get_local_components_gpu(fids_gpu, fadj_gpu)
    assert _as_set_of_sets(got) == _as_set_of_sets(expected)


def test_gpu_uf_single_chain(four_face_chain):
    face_ids, mesh_faces, face_adj = four_face_chain
    expected = _get_local_components_np(face_ids, mesh_faces, face_adj)

    from corep_fast.stages.s4_face_point import _get_local_components_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fids_gpu = torch.from_numpy(face_ids).to(device)
    fadj_gpu = torch.from_numpy(face_adj).to(device)

    got = _get_local_components_gpu(fids_gpu, fadj_gpu)
    assert _as_set_of_sets(got) == _as_set_of_sets(expected)
    # Should produce exactly one component of size 4
    assert len(got) == 1
    assert sum(len(c) for c in got) == 4


def test_gpu_uf_empty(empty_input):
    face_ids, mesh_faces, face_adj = empty_input

    from corep_fast.stages.s4_face_point import _get_local_components_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fids_gpu = torch.from_numpy(face_ids).to(device)
    fadj_gpu = torch.from_numpy(face_adj).to(device)

    got = _get_local_components_gpu(fids_gpu, fadj_gpu)
    assert got == []


def test_gpu_uf_signature_accepts_tensors():
    """Contract: input is GPU tensors (not numpy)."""
    from corep_fast.stages.s4_face_point import _get_local_components_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Global table with 2 isolated faces
    fids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    fadj = torch.full((2, 3), -1, dtype=torch.int32, device=device)
    got = _get_local_components_gpu(fids, fadj)
    assert _as_set_of_sets(got) == {frozenset([0]), frozenset([1])}
