# corep_fast/tests/regression/test_s2_ab.py
"""A/B regression: GPU s2_components vs custom/ feature_volume."""
import types

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.interop.custom_runner import run_custom_through_stage, get_custom_norm_mesh
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components


class TestS2AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s2')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        # Use custom/ normalization for apple-to-apple comparison
        norm_mesh = get_custom_norm_mesh(ab_mesh_path)
        verts_np = np.asarray(norm_mesh.vertices, dtype=np.float32)
        faces_np = np.asarray(norm_mesh.faces, dtype=np.int32)
        triangles_t = torch.from_numpy(verts_np[faces_np]).to(device=gpu_device, dtype=torch.float32)

        # Build MeshTensors for face_adj (needed by s2)
        mesh_raw = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh_raw, ab_resolution, device=gpu_device)

        # Run s1 with custom-normalized triangles
        mesh_ns = types.SimpleNamespace(triangles=triangles_t)
        batch = s1_voxelize(mesh_ns, ab_resolution, gpu_device)
        # Run s2 with MeshTensors (for face_adj)
        return s2_components(batch, mt)

    def test_num_components_exact_match(self, custom_regs, gpu_batch):
        custom_nc = torch.tensor(
            [r.get('num_components', 0) for r in custom_regs], dtype=torch.int32)
        gpu_nc = gpu_batch.num_components.cpu()
        # Sort by cube hash for alignment
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_hash = torch.tensor(
            [r['cube_indices'][0] * R * R + r['cube_indices'][1] * R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)
        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()
        assert torch.equal(gpu_nc[gpu_order], custom_nc[custom_order]), \
            "num_components mismatch"

    def test_num_boundary_exact_match(self, custom_regs, gpu_batch):
        custom_nb = torch.tensor(
            [r.get('num_boundary', 0) for r in custom_regs], dtype=torch.int32)
        gpu_nb = gpu_batch.num_boundary.cpu()
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_hash = torch.tensor(
            [r['cube_indices'][0] * R * R + r['cube_indices'][1] * R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)
        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()
        assert torch.equal(gpu_nb[gpu_order], custom_nb[custom_order]), \
            "num_boundary mismatch"
