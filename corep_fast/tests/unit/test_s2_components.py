"""Unit tests for s2_components GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components


@pytest.fixture
def batch_after_s1():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    return s1_voxelize(mt, 32, torch.device('cpu')), mt


class TestS2Components:
    def test_num_components_populated(self, batch_after_s1):
        batch, mesh = batch_after_s1
        result = s2_components(batch, mesh)
        assert result.num_components.shape == (result.num_cubes,)
        assert (result.num_components >= 1).all(), "Every occupied cube should have >=1 component"

    def test_num_boundary_populated(self, batch_after_s1):
        batch, mesh = batch_after_s1
        result = s2_components(batch, mesh)
        assert result.num_boundary.shape == (result.num_cubes,)
        # For a closed icosphere, no boundary cubes
        assert (result.num_boundary == 0).all()

    def test_single_face_cube_has_one_component(self, batch_after_s1):
        batch, mesh = batch_after_s1
        result = s2_components(batch, mesh)
        counts = batch.tri_offsets[1:] - batch.tri_offsets[:-1]
        single_face_mask = counts == 1
        if single_face_mask.any():
            assert (result.num_components[single_face_mask] == 1).all()
