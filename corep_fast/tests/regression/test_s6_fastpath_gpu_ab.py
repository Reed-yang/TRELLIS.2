"""A/B test: GPU fast-path s6 must match CPU fast-path bit-exactly.

Compares loop_cube_off, loop_edge_off, loop_edge_val, status, and
uturn_assignment tensors between paths (a) CPU MP and (b) GPU fast-path.

Mirrors the W2 (s7) test scaffold: trimesh icosphere subdiv=2, individual
stage calls s1->s4, then s6 with COREP_FAST_S6_FASTPATH_GPU env toggle.
"""
import os
import types
import torch
import numpy as np
import pytest
from corep_fast.stages.s6_collapse import s6_collapse


@pytest.mark.parametrize("res", [32, 64, 128])
def test_s6_fastpath_gpu_matches_cpu(res):
    # Run s1-s4 to feed s6
    import trimesh
    from pathlib import Path
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point

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

    # Path (a): CPU MP fast-path (default)
    os.environ["COREP_FAST_S6_FASTPATH_GPU"] = "0"
    batch_a = s6_collapse(_clone_batch(batch))

    # Path (b): GPU fast-path
    os.environ["COREP_FAST_S6_FASTPATH_GPU"] = "1"
    batch_b = s6_collapse(_clone_batch(batch))

    for field in ["loop_cube_off", "loop_edge_off", "loop_edge_val",
                  "status", "uturn_assignment"]:
        a = getattr(batch_a, field).cpu()
        b = getattr(batch_b, field).cpu()
        assert torch.equal(a, b), (
            f"res={res}: {field} differs.  "
            f"a.shape={tuple(a.shape)} b.shape={tuple(b.shape)}"
        )


def _clone_batch(batch):
    """Helper: shallow-copy CubeBatch tensors so s6 doesn't mutate the original."""
    import dataclasses
    fields = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    for k, v in fields.items():
        if hasattr(v, 'clone'):
            fields[k] = v.clone()
    return type(batch)(**fields)
