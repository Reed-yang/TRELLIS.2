# corep_fast/tests/regression/test_s7_ab.py
"""A/B regression: GPU s7_rank_assign vs custom/ collapse_point."""
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
from corep_fast.stages.s7_rank_assign import s7_rank_assign


def _canonical_with_ranks(edges, ranks):
    """Canonicalize a cyclic (edge, rank) sequence under rotation and reflection.

    Returns the lexicographically smallest representative as a tuple of
    (edge, rank) pairs.  Used for comparing loops across custom/ and GPU
    where the starting point or traversal direction may differ.
    """
    if not edges:
        return ()
    n = len(edges)
    pairs = list(zip(edges, ranks))

    # Forward rotations
    candidates = [tuple(pairs[i:] + pairs[:i]) for i in range(n)]
    # Backward (reflected) rotations
    rev = list(reversed(pairs))
    candidates += [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(candidates)


def _canonical_edges(loop):
    """Canonicalize a cyclic edge-only loop under rotation and reflection."""
    if not loop:
        return tuple(loop)
    n = len(loop)
    candidates = [tuple(loop[i:] + loop[:i]) for i in range(n)]
    rev = list(reversed(loop))
    candidates += [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(candidates)


class TestS7AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s7')

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
        batch = s6_collapse(batch)
        return s7_rank_assign(batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cube_hash(ix, iy, iz, R):
        return ix * R * R + iy * R + iz

    @staticmethod
    def _has_uturns(reg):
        """Check if a custom reg has non-zero face weights (U-turns)."""
        fw = reg.get('face_weights', [])
        return any(w != 0 for w in fw)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_ranks_match(self, custom_regs, gpu_batch):
        """For OK non-U-turn cubes, rank assignments must match exactly.

        U-turn cubes are excluded because custom/ sort_loops_in_cube does not
        handle U-turns and silently falls back to all-zero ranks, while the
        GPU path correctly traces the arc graph with U-turn assignments.
        """
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
        loop_edge_rank = gpu_batch.loop_edge_rank.cpu()

        mismatches = 0
        checked = 0
        skipped_uturn = 0
        first_mismatch_info = None

        for reg in custom_regs:
            if reg.get('exception', False):
                continue

            # Skip U-turn cubes (custom/ cannot rank them correctly)
            if self._has_uturns(reg):
                skipped_uturn += 1
                continue

            ci = reg['cube_indices']
            h = self._cube_hash(ci[0], ci[1], ci[2], R)
            if h not in gpu_hash_map:
                continue

            gi = gpu_hash_map[h]
            if int(gpu_status[gi]) != CubeStatus.OK:
                continue

            # Extract GPU loops with ranks
            l_lo = int(loop_cube_off[gi])
            l_hi = int(loop_cube_off[gi + 1])
            gpu_loops_with_ranks = []
            for li in range(l_lo, l_hi):
                e_lo = int(loop_edge_off[li])
                e_hi = int(loop_edge_off[li + 1])
                edges = loop_edge_val[e_lo:e_hi].tolist()
                ranks = loop_edge_rank[e_lo:e_hi].tolist()
                gpu_loops_with_ranks.append((edges, ranks))

            # Extract custom loops with ranks from sorted_loops
            custom_sorted = reg.get('sorted_loops', [])

            # Canonicalize and compare as sorted sets
            gpu_canonical = sorted(
                _canonical_with_ranks(e, r) for e, r in gpu_loops_with_ranks
            )
            custom_canonical = sorted(
                _canonical_with_ranks(sl['loop'], sl['rank'])
                for sl in custom_sorted
            )

            checked += 1
            if gpu_canonical != custom_canonical:
                mismatches += 1
                if first_mismatch_info is None:
                    first_mismatch_info = (
                        f"cube {ci}: rank mismatch.\n"
                        f"  gpu (canonical):    {gpu_canonical}\n"
                        f"  custom (canonical): {custom_canonical}"
                    )

        assert checked > 0, "No OK non-U-turn cubes were checked"
        assert mismatches == 0, (
            f"{mismatches}/{checked} OK cubes have rank mismatches "
            f"(skipped {skipped_uturn} U-turn cubes). "
            f"First: {first_mismatch_info}"
        )

    def test_point_assignment_match(self, custom_regs, gpu_batch):
        """For OK non-U-turn cubes, each loop must get the same component_point.

        U-turn cubes are excluded because the custom/ rank fallback
        produces wrong centroids, which leads to different Hungarian
        assignments.  Actual point *values* may differ slightly (f32 vs f64
        upstream in s4_face_point), so we use a tolerance of 1e-3.
        """
        POINT_TOL = 1e-3  # f32 vs f64 precision from upstream s4_face_point

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
        loop_point_match = gpu_batch.loop_point_match.cpu()
        point_offsets = gpu_batch.point_offsets.cpu()
        point_values = gpu_batch.point_values.cpu()

        mismatches = 0
        checked = 0
        skipped_uturn = 0
        first_mismatch_info = None

        for reg in custom_regs:
            if reg.get('exception', False):
                continue

            # Skip U-turn cubes
            if self._has_uturns(reg):
                skipped_uturn += 1
                continue

            ci = reg['cube_indices']
            h = self._cube_hash(ci[0], ci[1], ci[2], R)
            if h not in gpu_hash_map:
                continue

            gi = gpu_hash_map[h]
            if int(gpu_status[gi]) != CubeStatus.OK:
                continue

            custom_sorted = reg.get('sorted_loops', [])
            if not custom_sorted:
                continue

            # Extract GPU loops (edge sequences) and their matched points
            l_lo = int(loop_cube_off[gi])
            l_hi = int(loop_cube_off[gi + 1])
            p_lo = int(point_offsets[gi])
            p_hi = int(point_offsets[gi + 1])

            if p_hi <= p_lo:
                continue

            gpu_points = point_values[p_lo:p_hi].cpu().numpy().astype(np.float64)

            # Build a map: canonical edge loop -> component_point for GPU
            gpu_loop_to_point = {}
            for li in range(l_lo, l_hi):
                e_lo = int(loop_edge_off[li])
                e_hi = int(loop_edge_off[li + 1])
                edges = loop_edge_val[e_lo:e_hi].tolist()
                match_idx = int(loop_point_match[li])
                canon = _canonical_edges(edges)
                if match_idx < gpu_points.shape[0]:
                    gpu_loop_to_point[canon] = gpu_points[match_idx]

            # Build a map: canonical edge loop -> component_point for custom
            custom_loop_to_point = {}
            for sl in custom_sorted:
                canon = _canonical_edges(sl['loop'])
                cp = sl.get('component_point')
                if cp is not None:
                    custom_loop_to_point[canon] = np.array(cp, dtype=np.float64)

            # Compare: for each loop present in both, the assigned point must match
            checked += 1
            for canon, gpu_pt in gpu_loop_to_point.items():
                if canon not in custom_loop_to_point:
                    continue
                custom_pt = custom_loop_to_point[canon]
                dist = np.linalg.norm(gpu_pt - custom_pt)
                if dist > POINT_TOL:
                    mismatches += 1
                    if first_mismatch_info is None:
                        first_mismatch_info = (
                            f"cube {ci}, loop {canon}: "
                            f"gpu_pt={gpu_pt.tolist()}, "
                            f"custom_pt={custom_pt.tolist()}, "
                            f"dist={dist:.8f}"
                        )
                    break  # one mismatch per cube is enough

        assert checked > 0, "No OK non-U-turn cubes with points were checked"
        assert mismatches == 0, (
            f"{mismatches}/{checked} OK cubes have point assignment mismatches "
            f"(skipped {skipped_uturn} U-turn cubes). "
            f"First: {first_mismatch_info}"
        )

    def test_loop_point_match_in_range(self, gpu_batch):
        """Every loop_point_match index must be within [0, n_points) for that cube."""
        loop_cube_off = gpu_batch.loop_cube_off.cpu()
        loop_point_match = gpu_batch.loop_point_match.cpu()
        point_offsets = gpu_batch.point_offsets.cpu()
        status = gpu_batch.status.cpu()

        N = gpu_batch.num_cubes
        violations = 0
        checked = 0
        first_violation = None

        for i in range(N):
            if int(status[i]) != CubeStatus.OK:
                continue
            l_lo = int(loop_cube_off[i])
            l_hi = int(loop_cube_off[i + 1])
            n_loops = l_hi - l_lo
            if n_loops == 0:
                continue

            p_lo = int(point_offsets[i])
            p_hi = int(point_offsets[i + 1])
            n_points = p_hi - p_lo

            checked += 1
            for li in range(l_lo, l_hi):
                m = int(loop_point_match[li])
                if m < 0 or (n_points > 0 and m >= n_points):
                    violations += 1
                    if first_violation is None:
                        first_violation = (
                            f"cube {i}: loop {li} matched to point {m}, "
                            f"but n_points={n_points}"
                        )
                    break

        assert checked > 0, "No OK cubes with loops were checked"
        assert violations == 0, (
            f"{violations}/{checked} cubes have out-of-range loop_point_match. "
            f"First: {first_violation}"
        )
