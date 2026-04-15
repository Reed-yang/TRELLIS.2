# corep_fast/tests/regression/test_s3_ab.py
"""A/B regression: GPU s3_edge_weights vs custom/ feature_edge."""
import types

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.interop.custom_runner import run_custom_through_stage, get_custom_norm_mesh
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights


class TestS3AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s3')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        # Use custom/ normalization for matching input
        norm_mesh = get_custom_norm_mesh(ab_mesh_path)
        verts_np = np.asarray(norm_mesh.vertices, dtype=np.float32)
        faces_np = np.asarray(norm_mesh.faces, dtype=np.int32)
        triangles_t = torch.from_numpy(verts_np[faces_np]).to(device=gpu_device, dtype=torch.float32)

        # MeshTensors needed for face_adj (s2) and triangles (s3)
        mesh_raw = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh_raw, ab_resolution, device=gpu_device)

        # Run s1 with custom-normalized triangles
        mesh_ns = types.SimpleNamespace(triangles=triangles_t)
        batch = s1_voxelize(mesh_ns, ab_resolution, gpu_device)
        batch = s2_components(batch, mt)

        # IMPORTANT: s3 needs the mesh triangles for intersection tests.
        # We must pass a mesh object whose .triangles are in custom/ normalized space.
        # Create a MeshTensors-like namespace with custom-normalized triangles
        mesh_for_s3 = types.SimpleNamespace(
            triangles=triangles_t,
            face_adj=mt.face_adj,
        )
        return s3_edge_weights(batch, mesh_for_s3)

    def test_edge_weights_exact_match(self, custom_regs, gpu_batch):
        """edge_weights must match exactly — they are integer counts."""
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_ew = torch.tensor(
            [r.get('edge_weights', [0]*18) for r in custom_regs], dtype=torch.int32)
        custom_hash = torch.tensor(
            [r['cube_indices'][0]*R*R + r['cube_indices'][1]*R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)

        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()

        gpu_sorted = gpu_batch.edge_weights.cpu()[gpu_order]
        custom_sorted = custom_ew[custom_order]

        mismatches = (gpu_sorted != custom_sorted).any(dim=1).sum().item()
        assert mismatches == 0, (
            f"{mismatches}/{gpu_sorted.shape[0]} cubes have edge_weight mismatches. "
            f"First mismatch at sorted index "
            f"{(gpu_sorted != custom_sorted).any(dim=1).nonzero()[0].item()}"
        )
