"""Unit tests for GPU plane-triangle intersection."""
import torch
import pytest


class TestPlaneTriangleIntersection:
    def test_triangle_crossing_z_plane(self):
        from corep_fast.geom.plane_tri_intersect import plane_triangle_intersect
        tris = torch.tensor([[[-1., -1., -1.], [1., -1., 1.], [-1., 1., 1.]]],
                            dtype=torch.float32)
        plane_n = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
        plane_pt = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
        seg_p1, seg_p2, valid = plane_triangle_intersect(tris, plane_n, plane_pt)
        assert valid.shape == (1,)
        assert valid[0].item() == True
        assert abs(seg_p1[0, 2].item()) < 1e-5
        assert abs(seg_p2[0, 2].item()) < 1e-5

    def test_triangle_parallel_to_plane(self):
        from corep_fast.geom.plane_tri_intersect import plane_triangle_intersect
        tris = torch.tensor([[[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]]],
                            dtype=torch.float32)
        plane_n = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
        plane_pt = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
        seg_p1, seg_p2, valid = plane_triangle_intersect(tris, plane_n, plane_pt)
        assert valid[0].item() == False

    def test_batch_intersection(self):
        from corep_fast.geom.plane_tri_intersect import plane_triangle_intersect
        tris = torch.tensor([
            [[-1., 0., -1.], [1., 0., -1.], [0., 0., 1.]],
            [[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]],
        ], dtype=torch.float32)
        plane_n = torch.tensor([[0., 0., 1.], [0., 0., 1.]], dtype=torch.float32)
        plane_pt = torch.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)
        seg_p1, seg_p2, valid = plane_triangle_intersect(tris, plane_n, plane_pt)
        assert valid[0].item() == True
        assert valid[1].item() == False

    def test_triangle_one_vertex_on_plane(self):
        """Triangle with one vertex exactly on the plane, others on one side."""
        from corep_fast.geom.plane_tri_intersect import plane_triangle_intersect
        tris = torch.tensor([[[0., 0., 0.], [1., 0., 1.], [0., 1., 1.]]],
                            dtype=torch.float32)
        plane_n = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
        plane_pt = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
        seg_p1, seg_p2, valid = plane_triangle_intersect(tris, plane_n, plane_pt)
        # One vertex on plane, others above -> no crossing (all on same side or boundary)
        # The function should return valid=False since we need vertices on BOTH sides
        assert valid[0].item() == False

    def test_triangle_two_vertices_on_plane(self):
        """Triangle with two vertices on the plane (coplanar edge)."""
        from corep_fast.geom.plane_tri_intersect import plane_triangle_intersect
        tris = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0.5, 0.5, 1.]]],
                            dtype=torch.float32)
        plane_n = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
        plane_pt = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
        seg_p1, seg_p2, valid = plane_triangle_intersect(tris, plane_n, plane_pt)
        # Two vertices at z=0, one at z=1 -> no strict crossing (no vertex below plane)
        assert valid[0].item() == False
