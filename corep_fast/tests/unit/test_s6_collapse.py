"""Unit tests for s6_collapse GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse


@pytest.fixture
def batch_after_s4():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    return batch


class TestS6Collapse:
    def test_loops_populated(self, batch_after_s4):
        result = s6_collapse(batch_after_s4)
        N = result.num_cubes
        assert result.loop_cube_off.shape == (N + 1,)
        assert result.loop_cube_off[0] == 0
        total_loops = int(result.loop_cube_off[-1].item())
        assert total_loops > 0

    def test_loop_edges_valid(self, batch_after_s4):
        result = s6_collapse(batch_after_s4)
        if result.loop_edge_val.numel() > 0:
            assert (result.loop_edge_val >= 0).all()
            assert (result.loop_edge_val <= 17).all()

    def test_status_populated(self, batch_after_s4):
        result = s6_collapse(batch_after_s4)
        ok_count = (result.status == CubeStatus.OK).sum().item()
        assert ok_count > result.num_cubes * 0.9

    def test_loop_count_matches_components(self, batch_after_s4):
        """For OK cubes, loop count should equal num_components."""
        result = s6_collapse(batch_after_s4)
        ok_mask = result.status == CubeStatus.OK
        loops_per_cube = (result.loop_cube_off[1:] - result.loop_cube_off[:-1]).to(torch.int32)
        ok_loops = loops_per_cube[ok_mask]
        ok_nc = result.num_components[ok_mask]
        assert torch.equal(ok_loops, ok_nc), \
            f"Loop count != num_components for {(ok_loops != ok_nc).sum()} OK cubes"
