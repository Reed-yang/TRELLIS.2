"""Unit tests for s7_rank_assign GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse
from corep_fast.stages.s7_rank_assign import s7_rank_assign


@pytest.fixture
def batch_after_s6():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    batch = s6_collapse(batch)
    return batch


class TestS7RankAssign:
    def test_ranks_populated(self, batch_after_s6):
        result = s7_rank_assign(batch_after_s6)
        assert result.loop_edge_rank.shape == result.loop_edge_val.shape
        if result.loop_edge_rank.numel() > 0:
            assert (result.loop_edge_rank >= -1).all()

    def test_point_match_valid(self, batch_after_s6):
        result = s7_rank_assign(batch_after_s6)
        total_loops = int(result.loop_cube_off[-1].item())
        assert result.loop_point_match.shape == (total_loops,)
        ok_mask = result.status == CubeStatus.OK
        for i in range(result.num_cubes):
            if not ok_mask[i]:
                continue
            l_lo = int(result.loop_cube_off[i])
            l_hi = int(result.loop_cube_off[i + 1])
            n_loops = l_hi - l_lo
            if n_loops == 0:
                continue
            matches = result.loop_point_match[l_lo:l_hi]
            assert (matches >= 0).all() and (matches < n_loops).all()

    def test_ranks_within_edge_weight(self, batch_after_s6):
        """Each rank should be in [0, edge_weight-1] for its edge."""
        result = s7_rank_assign(batch_after_s6)
        ok_mask = result.status == CubeStatus.OK
        for i in range(min(result.num_cubes, 50)):
            if not ok_mask[i]:
                continue
            l_lo = int(result.loop_cube_off[i])
            l_hi = int(result.loop_cube_off[i + 1])
            for li in range(l_lo, l_hi):
                e_lo = int(result.loop_edge_off[li])
                e_hi = int(result.loop_edge_off[li + 1])
                edges = result.loop_edge_val[e_lo:e_hi]
                ranks = result.loop_edge_rank[e_lo:e_hi]
                for e, r in zip(edges.tolist(), ranks.tolist()):
                    w = int(result.edge_weights[i, e].item())
                    assert 0 <= r < w, f"Cube {i}: rank {r} out of range for edge {e} (weight={w})"
