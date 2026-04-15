"""A/B regression: GPU s1_voxelize vs custom/ voxelize."""
import types

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.interop.custom_runner import run_custom_through_stage, get_custom_norm_mesh
from corep_fast.stages.s1_voxelize import s1_voxelize


class TestS1AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s1')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        # Use the exact same normalized mesh as custom/ to ensure vertex coords match.
        # s1_voxelize only reads mesh.triangles, so we build a lightweight namespace
        # carrying just that field — no need to invoke MeshTensors.from_trimesh (which
        # would apply a second, slightly different normalization).
        norm_mesh = get_custom_norm_mesh(ab_mesh_path)
        verts_np = np.asarray(norm_mesh.vertices, dtype=np.float32)
        faces_np = np.asarray(norm_mesh.faces, dtype=np.int32)
        triangles_np = verts_np[faces_np]  # (F, 3, 3)
        triangles_t = torch.from_numpy(triangles_np).to(device=gpu_device, dtype=torch.float32)

        mesh_ns = types.SimpleNamespace(triangles=triangles_t)
        return s1_voxelize(mesh_ns, ab_resolution, gpu_device)

    def test_cube_count_matches(self, custom_regs, gpu_batch):
        assert gpu_batch.num_cubes == len(custom_regs), \
            f"Cube count: GPU={gpu_batch.num_cubes} vs custom={len(custom_regs)}"

    def test_cube_indices_match(self, custom_regs, gpu_batch):
        custom_indices = torch.tensor(
            [r['cube_indices'] for r in custom_regs], dtype=torch.int32)
        gpu_indices = gpu_batch.cube_indices.cpu()
        # Sort both by hash for comparison
        R = gpu_batch.resolution
        custom_hash = (custom_indices[:, 0].long() * R * R
                       + custom_indices[:, 1].long() * R
                       + custom_indices[:, 2].long())
        gpu_hash = (gpu_indices[:, 0].long() * R * R
                    + gpu_indices[:, 1].long() * R
                    + gpu_indices[:, 2].long())
        assert torch.equal(custom_hash.sort()[0], gpu_hash.sort()[0])

    def test_face_registration_complete(self, custom_regs, gpu_batch):
        """Every (cube, face) pair in custom/ must also appear in GPU output."""
        R = gpu_batch.resolution
        gpu_indices = gpu_batch.cube_indices.cpu()
        gpu_hash = (gpu_indices[:, 0].long() * R * R
                    + gpu_indices[:, 1].long() * R
                    + gpu_indices[:, 2].long())
        hash_to_idx = {int(h): i for i, h in enumerate(gpu_hash.tolist())}

        mismatches = 0
        for reg in custom_regs:
            ci = reg['cube_indices']
            h = ci[0] * R * R + ci[1] * R + ci[2]
            if h not in hash_to_idx:
                mismatches += 1
                continue
            idx = hash_to_idx[h]
            lo = int(gpu_batch.tri_offsets[idx])
            hi = int(gpu_batch.tri_offsets[idx + 1])
            gpu_faces = set(gpu_batch.tri_values[lo:hi].cpu().tolist())
            custom_faces = set(reg['face_indices'])
            if custom_faces != gpu_faces:
                mismatches += 1
        assert mismatches == 0, f"{mismatches} cubes have mismatched face registrations"
