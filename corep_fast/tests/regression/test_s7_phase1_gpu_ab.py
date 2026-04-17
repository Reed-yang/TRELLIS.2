"""A/B test: ensure s7 GPU batched rank tracing matches CPU MP path.

Compares loop_edge_rank tensor between paths.
"""
import os
import types
import torch
import numpy as np
import pytest
from corep_fast.stages.s7_rank_assign import s7_rank_assign


@pytest.mark.parametrize("res", [32, 64, 128])
def test_s7_phase1_gpu_matches_cpu(res):
    # Run s1-s6 to feed s7
    import trimesh
    from pathlib import Path
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse

    mesh_raw = trimesh.creation.icosphere(subdivisions=2)
    mesh_path = str(Path(__file__).parent.parent / "fixtures" / "icosphere_s2_for_test.ply")
    Path(mesh_path).parent.mkdir(parents=True, exist_ok=True)
    mesh_raw.export(mesh_path)
    device = torch.device("cuda")

    # Build MeshTensors (normalized) then mesh_ns namespace matching pipeline convention
    mt = MeshTensors.from_trimesh(mesh_raw, res, device=device)
    verts_np_f32 = np.asarray(mesh_raw.vertices, dtype=np.float32)
    verts_np_f64 = np.asarray(mesh_raw.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_raw.faces, dtype=np.int32)
    triangles_t = torch.from_numpy(verts_np_f32[faces_np]).to(device=device, dtype=torch.float32)
    verts_t = torch.from_numpy(verts_np_f64).to(device=device, dtype=torch.float64)
    faces_t = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32)
    mesh_ns = types.SimpleNamespace(
        triangles=triangles_t,
        vertices=verts_t,
        faces=faces_t,
        face_adj=mt.face_adj,
    )

    batch = s1_voxelize(mesh_ns, res, device)
    batch = s2_components(batch, mesh_ns)
    batch = s3_edge_weights(batch, mesh_ns)
    batch = s4_face_point(batch, mesh_ns)
    batch = s6_collapse(batch)

    # Path (a): CPU MP
    os.environ["COREP_FAST_S7_PHASE1_GPU"] = "0"
    batch_a_rank = s7_rank_assign(_clone_batch(batch)).loop_edge_rank.cpu()
    batch_a_match = s7_rank_assign(_clone_batch(batch)).loop_point_match.cpu()

    # Path (b): GPU batched (currently same as CPU MP — flag has no effect yet)
    os.environ["COREP_FAST_S7_PHASE1_GPU"] = "1"
    batch_b_rank = s7_rank_assign(_clone_batch(batch)).loop_edge_rank.cpu()
    batch_b_match = s7_rank_assign(_clone_batch(batch)).loop_point_match.cpu()

    assert torch.equal(batch_a_rank, batch_b_rank), f"res={res}: loop_edge_rank differs"
    assert torch.equal(batch_a_match, batch_b_match), f"res={res}: loop_point_match differs"


def _clone_batch(batch):
    """Helper: shallow-copy CubeBatch tensors so s7 doesn't mutate the original."""
    import dataclasses
    fields = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    for k, v in fields.items():
        if hasattr(v, 'clone'):
            fields[k] = v.clone()
    return type(batch)(**fields)
