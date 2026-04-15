# corep_fast/tests/regression/test_s6_ab.py
"""A/B regression: GPU s6_collapse vs custom/ collapse_face."""
import types

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.interop.custom_runner import run_custom_through_stage, get_custom_norm_mesh
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse


def _canonical(loop):
    """Canonicalize a cyclic loop under rotation and reflection.

    A loop [e0, e1, e2, e3] is equivalent to any cyclic rotation and to
    its reverse (plus rotations).  Return the lexicographically smallest
    representative as a tuple.
    """
    if not loop:
        return tuple(loop)
    n = len(loop)
    # Forward rotations
    candidates = [tuple(loop[i:] + loop[:i]) for i in range(n)]
    # Backward (reflected) rotations
    rev = list(reversed(loop))
    candidates += [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(candidates)


class TestS6AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s6')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        # Use custom/ normalization so vertex coords match exactly
        norm_mesh = get_custom_norm_mesh(ab_mesh_path)
        verts_np_f32 = np.asarray(norm_mesh.vertices, dtype=np.float32)
        verts_np_f64 = np.asarray(norm_mesh.vertices, dtype=np.float64)
        faces_np = np.asarray(norm_mesh.faces, dtype=np.int32)
        triangles_t = torch.from_numpy(verts_np_f32[faces_np]).to(
            device=gpu_device, dtype=torch.float32,
        )
        verts_t = torch.from_numpy(verts_np_f64).to(
            device=gpu_device, dtype=torch.float64,
        )
        faces_t = torch.from_numpy(faces_np).to(
            device=gpu_device, dtype=torch.int32,
        )

        mesh_raw = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh_raw, ab_resolution, device=gpu_device)
        mesh_ns = types.SimpleNamespace(
            triangles=triangles_t,
            vertices=verts_t,
            faces=faces_t,
            face_adj=mt.face_adj,
        )

        batch = s1_voxelize(mesh_ns, ab_resolution, gpu_device)
        batch = s2_components(batch, mesh_ns)
        batch = s3_edge_weights(batch, mesh_ns)
        batch = s4_face_point(batch, mesh_ns)
        return s6_collapse(batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cube_hash(ix, iy, iz, R):
        return ix * R * R + iy * R + iz

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_status_distribution_matches(self, custom_regs, gpu_batch):
        """Count of OK vs non-OK cubes must match between custom/ and GPU."""
        # Custom: solved cubes lack 'exception' flag (or it is False)
        custom_ok = sum(
            1 for r in custom_regs if not r.get('exception', False)
        )
        custom_exception = sum(
            1 for r in custom_regs if r.get('exception', False)
        )

        # GPU status counts
        status = gpu_batch.status.cpu()
        gpu_ok = int((status == CubeStatus.OK).sum().item())
        gpu_non_ok = int((status != CubeStatus.OK).sum().item())

        assert custom_ok == gpu_ok, (
            f"OK count mismatch: custom={custom_ok}, gpu={gpu_ok}"
        )
        assert custom_exception == gpu_non_ok, (
            f"Non-OK count mismatch: custom exceptions={custom_exception}, "
            f"gpu non-OK={gpu_non_ok}"
        )

    def test_loop_edge_sequences_match(self, custom_regs, gpu_batch):
        """For every OK cube, the set of loops (as canonical edge-index
        sequences) must match between custom/ and GPU."""
        R = gpu_batch.resolution

        # Build GPU hash -> index lookup
        gpu_ci = gpu_batch.cube_indices.cpu()
        gpu_status = gpu_batch.status.cpu()
        gpu_hash_map = {}
        for gi in range(gpu_ci.shape[0]):
            ix, iy, iz = int(gpu_ci[gi, 0]), int(gpu_ci[gi, 1]), int(gpu_ci[gi, 2])
            h = self._cube_hash(ix, iy, iz, R)
            gpu_hash_map[h] = gi

        # GPU CSR arrays
        loop_cube_off = gpu_batch.loop_cube_off.cpu()
        loop_edge_off = gpu_batch.loop_edge_off.cpu()
        loop_edge_val = gpu_batch.loop_edge_val.cpu()

        mismatches = 0
        first_mismatch_info = None

        for reg in custom_regs:
            # Skip exception cubes
            if reg.get('exception', False):
                continue

            ci = reg['cube_indices']
            h = self._cube_hash(ci[0], ci[1], ci[2], R)
            if h not in gpu_hash_map:
                mismatches += 1
                if first_mismatch_info is None:
                    first_mismatch_info = (
                        f"cube {ci}: present in custom/ but not in GPU batch"
                    )
                continue

            gi = gpu_hash_map[h]

            # Verify GPU also marks this cube as OK
            if int(gpu_status[gi]) != CubeStatus.OK:
                mismatches += 1
                if first_mismatch_info is None:
                    first_mismatch_info = (
                        f"cube {ci}: custom=OK but gpu status={int(gpu_status[gi])}"
                    )
                continue

            # Extract GPU loops for this cube
            l_lo = int(loop_cube_off[gi])
            l_hi = int(loop_cube_off[gi + 1])
            gpu_loops = []
            for li in range(l_lo, l_hi):
                e_lo = int(loop_edge_off[li])
                e_hi = int(loop_edge_off[li + 1])
                gpu_loops.append(loop_edge_val[e_lo:e_hi].tolist())

            # Extract custom loops
            custom_loops = reg.get('loops', [])

            # Canonicalize and compare as sorted sets of tuples
            gpu_canonical = sorted([_canonical(lp) for lp in gpu_loops])
            custom_canonical = sorted([_canonical(lp) for lp in custom_loops])

            if gpu_canonical != custom_canonical:
                mismatches += 1
                if first_mismatch_info is None:
                    first_mismatch_info = (
                        f"cube {ci}: loop mismatch.\n"
                        f"  gpu loops (canonical):    {gpu_canonical}\n"
                        f"  custom loops (canonical): {custom_canonical}"
                    )

        assert mismatches == 0, (
            f"{mismatches} OK cubes have loop mismatches. "
            f"First: {first_mismatch_info}"
        )
