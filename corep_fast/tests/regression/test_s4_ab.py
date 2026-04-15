# corep_fast/tests/regression/test_s4_ab.py
"""A/B regression: GPU s4_face_point vs custom/ feature_face + feature_point."""
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
from corep_fast.stages.s4_face_point import s4_face_point


class TestS4AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s4')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        # Use custom/ normalization so vertex coords match exactly
        norm_mesh = get_custom_norm_mesh(ab_mesh_path)
        verts_np_f32 = np.asarray(norm_mesh.vertices, dtype=np.float32)
        verts_np_f64 = np.asarray(norm_mesh.vertices, dtype=np.float64)
        faces_np = np.asarray(norm_mesh.faces, dtype=np.int32)
        # s1/s3 use triangles as float32 (GPU SAT / intersection)
        triangles_t = torch.from_numpy(verts_np_f32[faces_np]).to(
            device=gpu_device, dtype=torch.float32,
        )
        # s4 component_points uses vertices + faces; keep float64 to match
        # custom/ pipeline precision (trimesh default is float64).
        verts_t = torch.from_numpy(verts_np_f64).to(
            device=gpu_device, dtype=torch.float64,
        )
        faces_t = torch.from_numpy(faces_np).to(
            device=gpu_device, dtype=torch.int32,
        )

        # MeshTensors for face_adj (topology does not depend on normalization)
        mesh_raw = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh_raw, ab_resolution, device=gpu_device)

        # Build mesh namespace with custom-normalized geometry + topology
        mesh_ns = types.SimpleNamespace(
            triangles=triangles_t,
            vertices=verts_t,
            faces=faces_t,
            face_adj=mt.face_adj,
        )

        # Run pipeline through s4
        batch = s1_voxelize(mesh_ns, ab_resolution, gpu_device)
        batch = s2_components(batch, mesh_ns)
        batch = s3_edge_weights(batch, mesh_ns)
        batch = s4_face_point(batch, mesh_ns)
        return batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cube_hashes(cube_indices: torch.Tensor, R: int) -> torch.Tensor:
        """Compute scalar hash for each cube: ix*R*R + iy*R + iz."""
        ci = cube_indices.cpu().long()
        return ci[:, 0] * R * R + ci[:, 1] * R + ci[:, 2]

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_face_weights_exact_match(self, custom_regs, gpu_batch):
        """face_weights must match exactly — they are integer counts."""
        R = gpu_batch.resolution
        gpu_hash = self._cube_hashes(gpu_batch.cube_indices, R)

        custom_fw = torch.tensor(
            [r.get('face_weights', [0] * 12) for r in custom_regs],
            dtype=torch.int32,
        )
        custom_hash = torch.tensor(
            [r['cube_indices'][0] * R * R + r['cube_indices'][1] * R + r['cube_indices'][2]
             for r in custom_regs],
            dtype=torch.int64,
        )

        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()

        gpu_sorted = gpu_batch.face_weights.cpu()[gpu_order]
        custom_sorted = custom_fw[custom_order]

        mismatches = (gpu_sorted != custom_sorted).any(dim=1).sum().item()
        assert mismatches == 0, (
            f"{mismatches}/{gpu_sorted.shape[0]} cubes have face_weight mismatches. "
            f"First mismatch at sorted index "
            f"{(gpu_sorted != custom_sorted).any(dim=1).nonzero()[0].item()}"
        )

    def test_component_points_close_match(self, custom_regs, gpu_batch):
        """component_points must be close (float, atol=5e-4), sorted by cube hash.

        Tolerance is 5e-4 rather than 1e-5 because trimesh's nearest.on_surface()
        (used to snap area-weighted centroids back to the clipped surface) has
        inherent numerical sensitivity in its BVH proximity query that can cause
        ~2e-4 differences in a handful of edge-case cubes.
        """
        R = gpu_batch.resolution
        gpu_hash = self._cube_hashes(gpu_batch.cube_indices, R)

        custom_hash = torch.tensor(
            [r['cube_indices'][0] * R * R + r['cube_indices'][1] * R + r['cube_indices'][2]
             for r in custom_regs],
            dtype=torch.int64,
        )

        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()

        # Extract GPU component_points via CSR offsets
        gpu_offsets = gpu_batch.point_offsets.cpu()
        gpu_values = gpu_batch.point_values.cpu().float()

        total_mismatches = 0
        first_mismatch_info = None

        for rank in range(len(gpu_order)):
            gi = int(gpu_order[rank])
            ci = int(custom_order[rank])

            # GPU points for this cube
            g_lo = int(gpu_offsets[gi])
            g_hi = int(gpu_offsets[gi + 1])
            gpu_pts = gpu_values[g_lo:g_hi]  # (K, 3)

            # Custom points for this cube
            raw = custom_regs[ci].get('component_points', [])
            if len(raw) == 0:
                custom_pts = torch.zeros((0, 3), dtype=torch.float32)
            else:
                custom_pts = torch.tensor(raw, dtype=torch.float32)

            # Count mismatch
            if gpu_pts.shape[0] != custom_pts.shape[0]:
                total_mismatches += 1
                if first_mismatch_info is None:
                    cube_idx = gpu_batch.cube_indices[gi].cpu().tolist()
                    first_mismatch_info = (
                        f"cube {cube_idx}: gpu has {gpu_pts.shape[0]} points, "
                        f"custom has {custom_pts.shape[0]} points"
                    )
                continue

            if gpu_pts.shape[0] == 0:
                continue

            # Sort points within each cube for stable comparison
            # (component ordering may differ between implementations)
            gpu_sorted_pts = gpu_pts[gpu_pts[:, 0].argsort()]
            custom_sorted_pts = custom_pts[custom_pts[:, 0].argsort()]

            if not torch.allclose(gpu_sorted_pts, custom_sorted_pts, atol=5e-4):
                total_mismatches += 1
                if first_mismatch_info is None:
                    cube_idx = gpu_batch.cube_indices[gi].cpu().tolist()
                    max_diff = (gpu_sorted_pts - custom_sorted_pts).abs().max().item()
                    first_mismatch_info = (
                        f"cube {cube_idx}: max diff = {max_diff:.8f}"
                    )

        assert total_mismatches == 0, (
            f"{total_mismatches}/{len(gpu_order)} cubes have component_point mismatches. "
            f"First: {first_mismatch_info}"
        )
