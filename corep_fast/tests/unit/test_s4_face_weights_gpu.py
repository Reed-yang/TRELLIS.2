"""A/B test: GPU face_weights must match CPU MP face_weights exactly."""
import torch
import pytest
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import (
    _compute_face_weights_mp,
    _compute_face_weights_gpu,
)


@pytest.fixture
def pipeline_to_s3():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, 64, device=device)
    b = s1_voxelize(mt, 64, device)
    b = s2_components(b, mt)
    b = s3_edge_weights(b, mt)
    return b, mt


def test_gpu_fw_matches_cpu_mp(pipeline_to_s3):
    batch, mesh = pipeline_to_s3
    fw_cpu = _compute_face_weights_mp(batch, mesh)
    fw_gpu = _compute_face_weights_gpu(batch, mesh)
    diff = (fw_cpu != fw_gpu).sum().item()
    assert diff == 0, f"{diff} cube-facets differ between GPU and CPU paths"
