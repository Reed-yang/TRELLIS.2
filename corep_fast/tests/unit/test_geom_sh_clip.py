"""Unit tests for GPU Sutherland-Hodgman polygon clipping."""
import torch
import pytest


class TestSHClipAgainstPlane:
    def test_triangle_fully_inside(self):
        from corep_fast.geom.sh_clip import sh_clip_against_plane
        poly = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                               [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                               [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                               [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                             dtype=torch.float32)  # (1, 12, 3) padded
        v_len = torch.tensor([3], dtype=torch.int32)
        plane_n = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
        plane_d = torch.tensor([-1.0], dtype=torch.float32)
        new_poly, new_len = sh_clip_against_plane(poly, v_len, plane_n, plane_d)
        assert int(new_len[0]) == 3

    def test_triangle_fully_outside(self):
        from corep_fast.geom.sh_clip import sh_clip_against_plane
        poly = torch.zeros(1, 12, 3, dtype=torch.float32)
        poly[0, 0] = torch.tensor([0., 0., -5.])
        poly[0, 1] = torch.tensor([1., 0., -5.])
        poly[0, 2] = torch.tensor([0., 1., -5.])
        v_len = torch.tensor([3], dtype=torch.int32)
        plane_n = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
        plane_d = torch.tensor([0.0], dtype=torch.float32)
        new_poly, new_len = sh_clip_against_plane(poly, v_len, plane_n, plane_d)
        assert int(new_len[0]) == 0

    def test_triangle_clipped_to_quad(self):
        from corep_fast.geom.sh_clip import sh_clip_against_plane
        poly = torch.zeros(1, 12, 3, dtype=torch.float32)
        poly[0, 0] = torch.tensor([0., 0., 0.])
        poly[0, 1] = torch.tensor([2., 0., 0.])
        poly[0, 2] = torch.tensor([0., 2., 0.])
        v_len = torch.tensor([3], dtype=torch.int32)
        plane_n = torch.tensor([[-1., 0., 0.]], dtype=torch.float32)
        plane_d = torch.tensor([-1.0], dtype=torch.float32)
        new_poly, new_len = sh_clip_against_plane(poly, v_len, plane_n, plane_d)
        assert int(new_len[0]) == 4


class TestSHClipAABB:
    def test_triangle_inside_unit_cube(self):
        from corep_fast.geom.sh_clip import sh_clip_aabb
        tri = torch.tensor([[[0.2, 0.2, 0.2], [0.8, 0.2, 0.2], [0.5, 0.8, 0.2]]],
                           dtype=torch.float32)
        aabb_min = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
        aabb_max = torch.tensor([[1., 1., 1.]], dtype=torch.float32)
        poly, v_len = sh_clip_aabb(tri, aabb_min, aabb_max)
        assert int(v_len[0]) == 3

    def test_batch_clipping(self):
        from corep_fast.geom.sh_clip import sh_clip_aabb
        tri = torch.tensor([
            [[0.2, 0.2, 0.2], [0.8, 0.2, 0.2], [0.5, 0.8, 0.2]],
            [[-1., -1., 0.0], [2., -1., 0.0], [0.5, 2.0, 0.0]],
        ], dtype=torch.float32)
        aabb_min = torch.tensor([[0., 0., 0.], [0., 0., -1.]], dtype=torch.float32)
        aabb_max = torch.tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.float32)
        poly, v_len = sh_clip_aabb(tri, aabb_min, aabb_max)
        assert poly.shape[0] == 2
        assert int(v_len[0]) == 3
        assert int(v_len[1]) >= 3
