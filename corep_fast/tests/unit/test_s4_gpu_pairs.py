"""Tests for s4 GPU pair expansion used in face_weights optimization."""
import torch
import pytest
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import _expand_pairs_gpu


@pytest.fixture
def pipeline_to_s3():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, 64, device=device)
    b = s1_voxelize(mt, 64, device)
    b = s2_components(b, mt)
    b = s3_edge_weights(b, mt)
    return b, mt


def test_expand_pairs_shapes(pipeline_to_s3):
    batch, mesh = pipeline_to_s3
    pairs = _expand_pairs_gpu(batch, mesh)
    P = pairs['cube_id'].shape[0]
    assert P % 12 == 0, "Pair count must be multiple of 12 (facets)"
    assert pairs['facet_id'].shape == (P,)
    assert pairs['mesh_id'].shape == (P,)
    assert pairs['facet_vertices'].shape == (P, 3, 3)
    assert pairs['mesh_triangles'].shape == (P, 3, 3)


def test_expand_pairs_facet_id_cyclical(pipeline_to_s3):
    batch, mesh = pipeline_to_s3
    pairs = _expand_pairs_gpu(batch, mesh)
    # facet_id cycles 0..11 for each cube-tri group
    facet_ids_first_12 = pairs['facet_id'][:12].cpu().numpy()
    assert list(facet_ids_first_12) == list(range(12))
