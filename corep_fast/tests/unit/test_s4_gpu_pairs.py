"""Tests for s4 GPU pair expansion used in face_weights optimization."""
import torch
import pytest
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import _expand_pairs_gpu


@pytest.fixture
def pipeline_to_s3():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, 64, device=device)
    b = s1_voxelize(mt, 64, device)
    b = s2_components(b, mt)
    b = s3_edge_weights(b, mt)
    return b, mt


def test_expand_pairs_shapes(pipeline_to_s3):
    batch, mesh = pipeline_to_s3
    pairs = _expand_pairs_gpu(batch, mesh)
    P = pairs['cube_id'].shape[0]
    assert P % 12 == 0, "Pair count must be multiple of 12 (facets)"
    assert pairs['facet_id'].shape == (P,)
    assert pairs['mesh_id'].shape == (P,)
    assert pairs['facet_vertices'].shape == (P, 3, 3)
    assert pairs['mesh_triangles'].shape == (P, 3, 3)


def test_expand_pairs_facet_id_cyclical(pipeline_to_s3):
    batch, mesh = pipeline_to_s3
    pairs = _expand_pairs_gpu(batch, mesh)
    # facet_id cycles 0..11 for each cube-tri group
    facet_ids_first_12 = pairs['facet_id'][:12].cpu().numpy()
    assert list(facet_ids_first_12) == list(range(12))


# ----------------------------------------------------------------------
# Task 7 [P2.2]: _batch_plane_tri_with_clip
# ----------------------------------------------------------------------
from corep_fast.stages.s4_face_point import (
    _batch_plane_tri_with_clip,
    _clip_segment_to_triangle_vectorized,
    _intersect_facet_with_mesh,
)
import numpy as np


def test_batch_clip_single_known_case():
    """Hand-crafted: facet (0,0,0)-(1,0,0)-(0,1,0), mesh tri crossing z=0 plane."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    V = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # facet in z=0
    ], dtype=torch.float32, device=device)
    M = torch.tensor([
        [[0.2, 0.2, -0.5], [0.8, 0.2, 0.5], [0.5, 0.8, 0.0]],  # crosses z=0
    ], dtype=torch.float32, device=device)

    A, B, valid = _batch_plane_tri_with_clip(V, M)
    assert bool(valid[0].item()) is True
    # A and B should lie in z=0 (facet plane) and inside the facet triangle
    assert abs(A[0, 2].item()) < 1e-5
    assert abs(B[0, 2].item()) < 1e-5


def test_batch_clip_parity_vs_numpy():
    """Random seeded (facet, mesh_tri) pairs, compare GPU batch vs numpy reference."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    P = 200
    V = torch.rand((P, 3, 3), dtype=torch.float32, device=device)
    M = torch.rand((P, 3, 3), dtype=torch.float32, device=device) * 0.8 + 0.1

    A, B, valid = _batch_plane_tri_with_clip(V, M)

    # Numpy reference: call _intersect_facet_with_mesh for each pair
    V_cpu = V.cpu().numpy().astype(np.float64)
    M_cpu = M.cpu().numpy().astype(np.float64)
    valid_cpu = valid.cpu().numpy()

    mismatches = 0
    for i in range(P):
        segs = _intersect_facet_with_mesh(
            V_cpu[i, 0], V_cpu[i, 1], V_cpu[i, 2],
            M_cpu[i:i + 1],  # single mesh tri
        )
        gpu_said_valid = bool(valid_cpu[i])
        cpu_said_valid = len(segs) > 0
        if gpu_said_valid != cpu_said_valid:
            mismatches += 1

    # Allow up to 5% mismatches (edge cases with ε=1e-8 vs ε=1e-7 tolerances)
    assert mismatches <= P * 0.05, f"{mismatches}/{P} mismatches"
