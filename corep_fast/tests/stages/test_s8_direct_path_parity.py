"""Direct-tensor s8 path must produce identical mesh to dict path."""
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
from corep_fast.stages.s8_collapse import decode_from_cubebatch


def _pipeline(res):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, res, device=device)
    b = s1_voxelize(mt, res, device)
    b = s2_components(b, mt)
    b = s3_edge_weights(b, mt)
    b = s4_face_point(b, mt)
    b = s6_collapse(b)
    b = s7_rank_assign(b)
    return b


@pytest.mark.parametrize("res", [32, 64])
def test_direct_vs_dict_vf_match(res):
    b = _pipeline(res)
    v_dict, f_dict = decode_from_cubebatch(b, merge_decimals=5, use_direct_tensor=False)
    v_direct, f_direct = decode_from_cubebatch(b, merge_decimals=5, use_direct_tensor=True)

    # V/F counts should be identical (vertex welding is deterministic in torch.unique)
    assert v_dict.shape[0] == v_direct.shape[0], (
        f"V count mismatch at res={res}: dict={v_dict.shape[0]} direct={v_direct.shape[0]}"
    )
    assert f_dict.shape[0] == f_direct.shape[0], (
        f"F count mismatch at res={res}: dict={f_dict.shape[0]} direct={f_direct.shape[0]}"
    )

    # Vertex set should be identical (canonical sort)
    v_dict_sorted = torch.sort(v_dict.flatten()).values
    v_direct_sorted = torch.sort(v_direct.flatten()).values
    assert torch.allclose(v_dict_sorted, v_direct_sorted, atol=1e-5), "Vertex sets differ"
