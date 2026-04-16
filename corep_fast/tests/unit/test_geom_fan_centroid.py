"""Unit tests for GPU fan-triangulation + area-weighted centroid."""
import torch
import pytest


class TestFanAreaCentroid:
    def test_unit_triangle(self):
        from corep_fast.geom.fan_centroid import fan_area_centroid
        poly = torch.zeros(1, 12, 3, dtype=torch.float32)
        poly[0, 0] = torch.tensor([0., 0., 0.])
        poly[0, 1] = torch.tensor([1., 0., 0.])
        poly[0, 2] = torch.tensor([0., 1., 0.])
        v_len = torch.tensor([3], dtype=torch.int32)
        centroid, area = fan_area_centroid(poly, v_len)
        assert centroid.shape == (1, 3)
        expected = torch.tensor([[1./3, 1./3, 0.]])
        assert torch.allclose(centroid, expected, atol=1e-5)
        assert area[0] > 0

    def test_unit_square(self):
        from corep_fast.geom.fan_centroid import fan_area_centroid
        poly = torch.zeros(1, 12, 3, dtype=torch.float32)
        poly[0, 0] = torch.tensor([0., 0., 0.])
        poly[0, 1] = torch.tensor([1., 0., 0.])
        poly[0, 2] = torch.tensor([1., 1., 0.])
        poly[0, 3] = torch.tensor([0., 1., 0.])
        v_len = torch.tensor([4], dtype=torch.int32)
        centroid, area = fan_area_centroid(poly, v_len)
        expected = torch.tensor([[0.5, 0.5, 0.]])
        assert torch.allclose(centroid, expected, atol=1e-5)

    def test_batch(self):
        from corep_fast.geom.fan_centroid import fan_area_centroid
        poly = torch.zeros(3, 12, 3, dtype=torch.float32)
        poly[0, 0] = torch.tensor([0., 0., 0.])
        poly[0, 1] = torch.tensor([2., 0., 0.])
        poly[0, 2] = torch.tensor([0., 2., 0.])
        # Polygon 1: empty (0 vertices)
        v_len = torch.tensor([3, 0, 3], dtype=torch.int32)
        poly[2, 0] = torch.tensor([1., 1., 1.])
        poly[2, 1] = torch.tensor([2., 1., 1.])
        poly[2, 2] = torch.tensor([1., 2., 1.])
        centroid, area = fan_area_centroid(poly, v_len)
        assert centroid.shape == (3, 3)
        assert area[1] == 0.0
