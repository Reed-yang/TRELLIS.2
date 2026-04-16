"""Unit tests for s4_face_point GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point


@pytest.fixture
def batch_after_s3():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    return batch, mt


class TestS4FacePoint:
    def test_face_weights_shape(self, batch_after_s3):
        batch, mesh = batch_after_s3
        result = s4_face_point(batch, mesh)
        assert result.face_weights.shape == (result.num_cubes, 12)
        assert result.face_weights.dtype == torch.int32
        assert (result.face_weights >= 0).all()

    def test_component_points_csr(self, batch_after_s3):
        batch, mesh = batch_after_s3
        result = s4_face_point(batch, mesh)
        N = result.num_cubes
        assert result.point_offsets.shape == (N + 1,)
        assert result.point_offsets[0] == 0
        # Number of points per cube should equal num_components
        pts_per_cube = (result.point_offsets[1:] - result.point_offsets[:-1]).to(torch.int32)
        assert torch.equal(pts_per_cube, result.num_components)

    def test_points_in_cube_bounds(self, batch_after_s3):
        batch, mesh = batch_after_s3
        result = s4_face_point(batch, mesh)
        if result.point_values.shape[0] > 0:
            # Points should be within reasonable range
            assert (result.point_values >= -0.01).all()
            assert (result.point_values <= 1.01).all()
