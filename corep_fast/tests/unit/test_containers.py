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
