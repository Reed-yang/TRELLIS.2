"""Unit tests for s3_edge_weights GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights


@pytest.fixture
def batch_after_s2():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    return batch, mt


class TestS3EdgeWeights:
    def test_shape_and_dtype(self, batch_after_s2):
        batch, mesh = batch_after_s2
        result = s3_edge_weights(batch, mesh)
        assert result.edge_weights.shape == (result.num_cubes, 18)
        assert result.edge_weights.dtype == torch.int32

    def test_non_negative(self, batch_after_s2):
        batch, mesh = batch_after_s2
        result = s3_edge_weights(batch, mesh)
        assert (result.edge_weights >= 0).all()

    def test_original_edges_present(self, batch_after_s2):
        """For a closed mesh, most cubes should have nonzero weights on edges 0-11."""
        batch, mesh = batch_after_s2
        result = s3_edge_weights(batch, mesh)
        original_edges = result.edge_weights[:, :12]
        # At least some cubes must have nonzero original edge weights
        assert original_edges.sum() > 0
