"""Unit tests for corep_fast/containers.py MeshTensors (Task 4) and CubeBatch (Task 5)."""
import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors


# ---------------------------------------------------------------------------
# MeshTensors
# ---------------------------------------------------------------------------

def _make_cube_trimesh() -> trimesh.Trimesh:
    """A unit cube centered at origin — 12 triangles, 8 vertices, watertight."""
    return trimesh.creation.box(extents=[1., 1., 1.])


def _make_open_plane_trimesh() -> trimesh.Trimesh:
    """Flat square made of 2 triangles — 4 boundary edges, no non-manifold."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_mesh_tensors_from_trimesh_cube():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert isinstance(mt.vertices, torch.Tensor)
    assert mt.vertices.dtype == torch.float32
    assert mt.vertices.shape == (8, 3)
    assert mt.faces.dtype == torch.int32
    assert mt.faces.shape == (12, 3)
    assert mt.triangles.shape == (12, 3, 3)
    assert mt.face_normals.shape == (12, 3)
    assert mt.face_adj.shape == (12, 3)
    assert mt.resolution == 64
    assert str(mt.device) == 'cpu'


def test_mesh_tensors_cube_has_no_boundaries():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.boundaries.shape[0] == 0


def test_mesh_tensors_open_plane_has_4_boundaries():
    mesh = _make_open_plane_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.boundaries.shape == (4, 2, 3)
    assert mt.boundary_face_ids.shape == (4,)
    assert mt.boundary_face_ids.dtype == torch.int32


def test_mesh_tensors_normalization_puts_mesh_in_unit_cube():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.vertices.min() >= 0.0 - 1e-6
    assert mt.vertices.max() <= 1.0 + 1e-6


def test_mesh_tensors_triangles_match_faces_gather():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    gathered = mt.vertices[mt.faces.long()]
    assert torch.allclose(gathered, mt.triangles)


def test_mesh_tensors_face_adj_cube_is_dense():
    """Every face of a cube shares an edge with exactly 3 neighbors — no -1 entries."""
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert (mt.face_adj >= 0).all()


def test_mesh_tensors_face_adj_open_plane_has_boundary_gaps():
    """Open plane has 2 faces sharing 1 edge — 5 of 6 adjacency slots are -1."""
    mesh = _make_open_plane_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    neg_count = (mt.face_adj == -1).sum().item()
    assert neg_count == 4  # 2 faces × 3 edges - 2 shared = 4 unshared


def test_mesh_tensors_validate_rejects_empty():
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))
    with pytest.raises(ValueError):
        MeshTensors.from_trimesh(empty, resolution=64, device='cpu')


def test_mesh_tensors_to_cuda_device_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cuda')
    assert mt.vertices.is_cuda
    assert mt.faces.is_cuda
    assert mt.triangles.is_cuda


# ---------------------------------------------------------------------------
# CubeBatch
# ---------------------------------------------------------------------------
from corep_fast.containers import CubeBatch, CubeStatus


def _make_empty_cube_batch(N: int = 3, device: str = 'cpu') -> CubeBatch:
    return CubeBatch.empty(num_cubes=N, resolution=64, device=torch.device(device))


def test_cube_batch_empty_shapes():
    cb = _make_empty_cube_batch(N=5)
    assert cb.num_cubes == 5
    assert cb.cube_indices.shape == (5, 3)
    assert cb.cube_indices.dtype == torch.int32
    assert cb.cube_hash.shape == (5,)
    assert cb.cube_hash.dtype == torch.int64
    assert cb.edge_weights.shape == (5, 18)
    assert cb.edge_weights.dtype == torch.int32
    assert cb.face_weights.shape == (5, 12)
    assert cb.status.shape == (5,)
    assert cb.num_components.shape == (5,)


def test_cube_batch_empty_csr_offsets():
    cb = _make_empty_cube_batch(N=5)
    assert cb.tri_offsets.shape == (6,)
    assert cb.tri_offsets.dtype == torch.int64
    assert torch.equal(cb.tri_offsets, torch.zeros(6, dtype=torch.int64))
    assert cb.tri_values.shape == (0,)
    assert cb.bnd_offsets.shape == (6,)
    assert cb.bnd_values.shape == (0,)
    assert cb.point_offsets.shape == (6,)


def test_cube_batch_status_enum():
    assert CubeStatus.OK == 0
    assert CubeStatus.AMBIGUOUS == 1
    assert CubeStatus.UNSOLVABLE == 2
    assert CubeStatus.BUDGET_EXCEEDED == 3


def test_cube_batch_set_tri_csr():
    """Test populating the tri_values/tri_offsets CSR from a list of per-cube lists."""
    cb = _make_empty_cube_batch(N=3)
    per_cube_tris = [
        torch.tensor([5, 7, 9], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    ]
    cb = cb.set_tri_csr(per_cube_tris)
    assert torch.equal(cb.tri_offsets, torch.tensor([0, 3, 3, 4], dtype=torch.int64))
    assert torch.equal(cb.tri_values, torch.tensor([5, 7, 9, 2], dtype=torch.int32))


def test_cube_batch_get_tri_for_cube():
    """Test slicing a single cube's triangles via CSR offsets."""
    cb = _make_empty_cube_batch(N=3)
    per_cube_tris = [
        torch.tensor([5, 7, 9], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    ]
    cb = cb.set_tri_csr(per_cube_tris)
    assert torch.equal(cb.get_tris(0), torch.tensor([5, 7, 9], dtype=torch.int32))
    assert cb.get_tris(1).shape == (0,)
    assert torch.equal(cb.get_tris(2), torch.tensor([2], dtype=torch.int32))


def test_cube_batch_num_loops_is_zero_when_empty():
    cb = _make_empty_cube_batch(N=3)
    assert cb.num_loops == 0
    assert cb.num_loop_edges == 0


def test_cube_batch_cube_hash_roundtrip():
    """Verify cube_hash encodes (ix, iy, iz) reversibly within resolution bounds."""
    cb = _make_empty_cube_batch(N=3)
    indices = torch.tensor([[1, 2, 3], [0, 0, 0], [63, 63, 63]], dtype=torch.int32)
    cb = cb.with_cube_indices(indices)
    expected_hashes = torch.tensor(
        [1 * 64 * 64 + 2 * 64 + 3, 0, 63 * 64 * 64 + 63 * 64 + 63],
        dtype=torch.int64,
    )
    assert torch.equal(cb.cube_hash, expected_hashes)


def test_cube_batch_invariants_check_csr_monotone():
    cb = _make_empty_cube_batch(N=3)
    cb.invariants_check('empty')  # should not raise

    # Corrupt the tri_offsets and verify the check catches it
    bad_offsets = torch.tensor([0, 5, 3, 7], dtype=torch.int64)  # non-monotone
    cb_bad = cb.with_tri_offsets(bad_offsets)
    with pytest.raises(AssertionError):
        cb_bad.invariants_check('test_bad')
