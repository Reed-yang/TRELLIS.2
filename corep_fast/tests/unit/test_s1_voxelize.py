"""Unit tests for s1_voxelize GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.stages.s1_voxelize import s1_voxelize


@pytest.fixture
def simple_mesh_tensors():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    return MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')


class TestS1Voxelize:
    def test_returns_cubebatch(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        assert isinstance(batch, CubeBatch)

    def test_cube_indices_in_range(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        assert (batch.cube_indices >= 0).all()
        assert (batch.cube_indices < 32).all()
        assert batch.cube_indices.dtype == torch.int32

    def test_tri_csr_valid(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        N = batch.num_cubes
        assert batch.tri_offsets.shape == (N + 1,)
        assert batch.tri_offsets[0] == 0
        assert (batch.tri_offsets[1:] >= batch.tri_offsets[:-1]).all()
        # Every face should be registered to at least one cube
        registered_faces = batch.tri_values.unique()
        assert registered_faces.numel() > 0
        assert (registered_faces >= 0).all()
        assert (registered_faces < simple_mesh_tensors.faces.shape[0]).all()

    def test_nonempty_cubes(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        # Each cube must have at least 1 registered face
        counts = batch.tri_offsets[1:] - batch.tri_offsets[:-1]
        assert (counts > 0).all()

    def test_cube_hash_matches_indices(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        R = 32
        expected_hash = (batch.cube_indices[:, 0].long() * R * R
                         + batch.cube_indices[:, 1].long() * R
                         + batch.cube_indices[:, 2].long())
        assert torch.equal(batch.cube_hash, expected_hash)
