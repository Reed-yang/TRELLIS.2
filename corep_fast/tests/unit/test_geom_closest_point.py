"""Unit tests for GPU closest-point-on-triangle-mesh."""
import numpy as np
import pytest
import torch


def _make_unit_triangle():
    """Single triangle at origin: (0,0,0), (1,0,0), (0,1,0)."""
    return torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], dtype=torch.float32)


class TestClosestPointOnTriangle:
    def test_point_inside_triangle(self):
        from corep_fast.geom.closest_point import closest_point_on_triangles
        tris = _make_unit_triangle()
        query = torch.tensor([[0.2, 0.2, 0.5]])
        pts, dists = closest_point_on_triangles(query, tris)
        assert pts.shape == (1, 1, 3)
        expected = torch.tensor([0.2, 0.2, 0.0])
        assert torch.allclose(pts[0, 0], expected, atol=1e-5)

    def test_point_nearest_vertex(self):
        from corep_fast.geom.closest_point import closest_point_on_triangles
        tris = _make_unit_triangle()
        query = torch.tensor([[-0.5, -0.5, 0.0]])
        pts, dists = closest_point_on_triangles(query, tris)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(pts[0, 0], expected, atol=1e-5)

    def test_point_nearest_edge(self):
        from corep_fast.geom.closest_point import closest_point_on_triangles
        tris = _make_unit_triangle()
        query = torch.tensor([[0.5, -0.5, 0.0]])
        pts, dists = closest_point_on_triangles(query, tris)
        expected = torch.tensor([0.5, 0.0, 0.0])
        assert torch.allclose(pts[0, 0], expected, atol=1e-5)

    def test_batch_queries(self):
        from corep_fast.geom.closest_point import closest_point_on_triangles
        tris = _make_unit_triangle()
        queries = torch.tensor([
            [0.2, 0.2, 1.0],
            [-1.0, -1.0, 0.0],
        ])
        pts, dists = closest_point_on_triangles(queries, tris)
        assert pts.shape == (2, 1, 3)
        assert dists.shape == (2, 1)


class TestClosestPointOnMesh:
    def test_icosphere_surface_snap(self):
        import trimesh
        from corep_fast.geom.closest_point import closest_point_on_mesh
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        triangles = torch.tensor(
            np.asarray(mesh.vertices, dtype=np.float32)[np.asarray(mesh.faces)],
            dtype=torch.float32,
        )
        queries = torch.tensor([[0.0, 0.0, 0.45], [0.3, 0.3, 0.0]], dtype=torch.float32)
        snapped, face_idx = closest_point_on_mesh(queries, triangles)
        assert snapped.shape == (2, 3)
        assert face_idx.shape == (2,)
        radii = snapped.norm(dim=-1)
        assert torch.allclose(radii, torch.tensor(0.4), atol=0.02)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_matches_cpu(self):
        import trimesh
        from corep_fast.geom.closest_point import closest_point_on_mesh
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
        triangles = torch.tensor(
            np.asarray(mesh.vertices, dtype=np.float32)[np.asarray(mesh.faces)],
            dtype=torch.float32,
        )
        queries = torch.randn(50, 3, dtype=torch.float32) * 0.5
        cpu_pts, cpu_idx = closest_point_on_mesh(queries, triangles)
        gpu_pts, gpu_idx = closest_point_on_mesh(queries.cuda(), triangles.cuda())
        assert torch.allclose(cpu_pts, gpu_pts.cpu(), atol=1e-5)
