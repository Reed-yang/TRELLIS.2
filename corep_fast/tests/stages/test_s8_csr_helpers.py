"""Tests for CSR expansion helpers used in direct-tensor s8 path."""
import torch
import pytest
from corep_fast.stages.s8_collapse import (
    _csr_expand_to_items,
    _derive_loop_component_point_gpu,
    _pad_ragged_loops_gpu,
)
from corep_fast.containers import CubeBatch, CubeStatus


class TestCsrExpandToItems:
    def test_simple_case(self):
        """offsets=[0,2,5,6] -> [0,0,1,1,1,2]"""
        offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
        result = _csr_expand_to_items(offsets, total=6)
        expected = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_empty_groups_in_middle(self):
        """offsets=[0,2,2,5] -> group 1 is empty -> [0,0,2,2,2]"""
        offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int64)
        result = _csr_expand_to_items(offsets, total=5)
        expected = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_total_zero(self):
        """total=0 -> empty tensor."""
        offsets = torch.tensor([0, 0, 0], dtype=torch.int64)
        result = _csr_expand_to_items(offsets, total=0)
        assert result.numel() == 0

    def test_gpu(self):
        """Runs on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device='cuda')
        result = _csr_expand_to_items(offsets, total=5)
        expected = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device='cuda')
        assert torch.equal(result, expected)


def _make_synthetic_batch(device='cpu'):
    """Build a minimal CubeBatch with diverse cases.

    Cubes:
      0: OK, 2 loops, 3 points, match idx 0 and 1
      1: OK, 0 loops, 0 points
      2: exception, 1 loop (virtual), 1 point
      3: OK, 1 loop, point match -1 (fallback to first)
      4: OK, 2 loops, 0 points (num_pts=0 fallback)
    """
    dev = torch.device(device)
    N = 5
    cube_indices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0]],
                                dtype=torch.int32, device=dev)
    # cube_hash = ix*R^2 + iy*R + iz (R=256)
    R = 256
    cube_hash = (cube_indices[:, 0].to(torch.int64) * R * R
                 + cube_indices[:, 1].to(torch.int64) * R
                 + cube_indices[:, 2].to(torch.int64))
    edge_weights = torch.zeros((N, 18), dtype=torch.int32, device=dev)
    face_weights = torch.zeros((N, 12), dtype=torch.int32, device=dev)
    status = torch.tensor([CubeStatus.OK, CubeStatus.OK, 1,
                           CubeStatus.OK, CubeStatus.OK],
                          dtype=torch.int32, device=dev)
    num_components = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32, device=dev)
    num_boundary = torch.zeros(N, dtype=torch.int32, device=dev)

    # Loop structure: 5 cubes, 2+0+1+1+2 = 6 loops total
    loop_cube_off = torch.tensor([0, 2, 2, 3, 4, 6], dtype=torch.int64, device=dev)
    # Each loop has 3 edges (placeholder)
    loop_edge_off = torch.tensor([0, 3, 6, 9, 12, 15, 18], dtype=torch.int64, device=dev)
    loop_edge_val = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                  0, 1, 2, 3, 4, 5],
                                 dtype=torch.int32, device=dev)
    loop_edge_rank = torch.zeros(18, dtype=torch.int32, device=dev)

    # Loop->point match: loop 0->0, 1->1, 2->0, 3->-1, 4->0, 5->5 (out of range)
    loop_point_match = torch.tensor([0, 1, 0, -1, 0, 5], dtype=torch.int32, device=dev)

    # Points: 3 for cube 0, 0 for 1, 1 for 2, 1 for 3, 0 for 4
    point_offsets = torch.tensor([0, 3, 3, 4, 5, 5], dtype=torch.int64, device=dev)
    point_values = torch.tensor([
        [1.0, 0.0, 0.0],  # cube 0 pt 0
        [0.0, 1.0, 0.0],  # cube 0 pt 1
        [0.0, 0.0, 1.0],  # cube 0 pt 2
        [2.0, 2.0, 2.0],  # cube 2 pt 0
        [3.0, 3.0, 3.0],  # cube 3 pt 0
    ], dtype=torch.float32, device=dev)

    return CubeBatch(
        cube_indices=cube_indices,
        cube_hash=cube_hash,
        tri_offsets=torch.zeros(N + 1, dtype=torch.int64, device=dev),
        tri_values=torch.zeros(0, dtype=torch.int32, device=dev),
        bnd_offsets=torch.zeros(N + 1, dtype=torch.int64, device=dev),
        bnd_values=torch.zeros(0, dtype=torch.int32, device=dev),
        nm_offsets=torch.zeros(N + 1, dtype=torch.int64, device=dev),
        nm_values=torch.zeros(0, dtype=torch.int32, device=dev),
        num_components=num_components,
        num_boundary=num_boundary,
        edge_weights=edge_weights,
        face_weights=face_weights,
        point_offsets=point_offsets,
        point_values=point_values,
        loop_cube_off=loop_cube_off,
        loop_edge_off=loop_edge_off,
        loop_edge_val=loop_edge_val,
        loop_edge_rank=loop_edge_rank,
        loop_point_match=loop_point_match,
        status=status,
        comp_face_off=torch.zeros(N + 1, dtype=torch.int64, device=dev),
        comp_face_val=torch.zeros(0, dtype=torch.int32, device=dev),
        uturn_assignment=torch.full((N, 12, 3), -1, dtype=torch.int32, device=dev),
        device=dev,
        resolution=R,
    )


class TestDeriveLoopComponentPoint:
    def test_basic(self):
        b = _make_synthetic_batch()
        pts = _derive_loop_component_point_gpu(b)
        # L=6
        assert pts.shape == (6, 3)
        # Loop 0 (cube 0, match 0): point_values[0] = [1,0,0]
        assert torch.allclose(pts[0], torch.tensor([1.0, 0.0, 0.0]))
        # Loop 1 (cube 0, match 1): point_values[1] = [0,1,0]
        assert torch.allclose(pts[1], torch.tensor([0.0, 1.0, 0.0]))
        # Loop 2 (cube 2, match 0): point_values[3] = [2,2,2]
        assert torch.allclose(pts[2], torch.tensor([2.0, 2.0, 2.0]))
        # Loop 3 (cube 3, match -1 -> first of cube 3): point_values[4] = [3,3,3]
        assert torch.allclose(pts[3], torch.tensor([3.0, 3.0, 3.0]))
        # Loop 4 (cube 4, num_pts=0): should be zero
        assert torch.allclose(pts[4], torch.zeros(3))
        # Loop 5 (cube 4, num_pts=0): should be zero
        assert torch.allclose(pts[5], torch.zeros(3))


class TestPadRaggedLoops:
    def test_basic(self):
        # 3 loops, lengths 2, 3, 1, max=3
        loop_edge_off = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
        loop_edge_val = torch.tensor([10, 11, 20, 21, 22, 30], dtype=torch.int32)
        loop_edge_rank = torch.tensor([0, 1, 0, 1, 2, 0], dtype=torch.int32)
        edges_flat, ranks_flat = _pad_ragged_loops_gpu(
            loop_edge_off, loop_edge_val, loop_edge_rank, max_loop_len=3,
        )
        # Shape (L, max_loop_len) flattened -> (9,)
        assert edges_flat.shape == (9,)
        # Loop 0: [10, 11, -1]; Loop 1: [20, 21, 22]; Loop 2: [30, -1, -1]
        expected_edges = torch.tensor([10, 11, -1, 20, 21, 22, 30, -1, -1], dtype=torch.int32)
        expected_ranks = torch.tensor([0, 1, -1, 0, 1, 2, 0, -1, -1], dtype=torch.int32)
        assert torch.equal(edges_flat, expected_edges)
        assert torch.equal(ranks_flat, expected_ranks)

    def test_empty(self):
        loop_edge_off = torch.tensor([0], dtype=torch.int64)
        loop_edge_val = torch.zeros(0, dtype=torch.int32)
        loop_edge_rank = torch.zeros(0, dtype=torch.int32)
        edges_flat, ranks_flat = _pad_ragged_loops_gpu(
            loop_edge_off, loop_edge_val, loop_edge_rank, max_loop_len=3,
        )
        assert edges_flat.numel() == 0
        assert ranks_flat.numel() == 0
