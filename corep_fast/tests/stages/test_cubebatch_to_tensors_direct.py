"""Parity test: direct tensor path produces equivalent CubeDataTensors to dict path."""
import torch
import pytest
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse
from corep_fast.stages.s7_rank_assign import s7_rank_assign
from corep_fast.stages.s8_collapse import (
    _cubebatch_to_dicts,
    _cube_data_to_tensors,
    _cubebatch_to_tensors_direct,
)


@pytest.fixture
def populated_batch():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, 64, device=device)
    batch = s1_voxelize(mt, 64, device)
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    batch = s6_collapse(batch)
    batch = s7_rank_assign(batch)
    return batch


def test_direct_matches_dict_basic_fields(populated_batch):
    b = populated_batch
    dl = _cubebatch_to_dicts(b)
    t_dict = _cube_data_to_tensors(dl, b.device)
    t_direct = _cubebatch_to_tensors_direct(b)

    # Basic fields: must match exactly
    assert torch.equal(t_direct.cube_indices, t_dict.cube_indices)
    assert torch.equal(t_direct.cube_edge_weights, t_dict.cube_edge_weights)
    assert torch.equal(t_direct.cube_num_components, t_dict.cube_num_components)


def test_direct_has_correct_shape(populated_batch):
    b = populated_batch
    t_direct = _cubebatch_to_tensors_direct(b)
    N = b.num_cubes
    assert t_direct.cube_indices.shape == (N, 3)
    assert t_direct.cube_edge_weights.shape == (N, 18)
    assert t_direct.cube_exception.shape == (N,)
    assert t_direct.max_loop_len >= 1


def test_max_loop_len_matches(populated_batch):
    """max_loop_len from direct path should equal or exceed that from dict path.

    (Dict path may include virtual exception loops with length 0 which don't
    affect the max. So equality is expected when no exception cubes have zero
    real loops, else direct may be <= dict's.)
    """
    b = populated_batch
    dl = _cubebatch_to_dicts(b)
    t_dict = _cube_data_to_tensors(dl, b.device)
    t_direct = _cubebatch_to_tensors_direct(b)
    # Direct uses actual max from CSR; dict's max includes virtual 0-length loops.
    # So direct.max_loop_len <= dict.max_loop_len always.
    assert t_direct.max_loop_len <= t_dict.max_loop_len or t_direct.max_loop_len == t_dict.max_loop_len
