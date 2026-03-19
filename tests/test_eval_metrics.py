import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from scripts.eval_metrics import chamfer_distance, f_score, normal_consistency


class TestChamferDistance:
    def test_identical_point_clouds_returns_zero(self):
        points = torch.rand(100, 3).cuda()
        cd = chamfer_distance(points, points)
        assert cd < 1e-6, f"CD of identical point clouds should be ~0, got {cd}"

    def test_known_distance(self):
        p1 = torch.tensor([[0.0, 0.0, 0.0]]).cuda()
        p2 = torch.tensor([[1.0, 0.0, 0.0]]).cuda()
        cd = chamfer_distance(p1, p2)
        assert abs(cd - 1.0) < 1e-5, f"CD should be 1.0, got {cd}"

    def test_symmetric(self):
        p1 = torch.rand(50, 3).cuda()
        p2 = torch.rand(80, 3).cuda()
        cd1 = chamfer_distance(p1, p2)
        cd2 = chamfer_distance(p2, p1)
        assert abs(cd1 - cd2) < 1e-5, f"CD should be symmetric: {cd1} vs {cd2}"


class TestFScore:
    def test_identical_points_returns_one(self):
        points = torch.rand(100, 3).cuda()
        fs = f_score(points, points, threshold=0.01)
        assert fs > 0.99, f"F-score of identical points should be ~1.0, got {fs}"

    def test_far_points_returns_zero(self):
        p1 = torch.zeros(100, 3).cuda()
        p2 = torch.ones(100, 3).cuda() * 10
        fs = f_score(p1, p2, threshold=0.01)
        assert fs < 0.01, f"F-score of far points should be ~0, got {fs}"


class TestNormalConsistency:
    def test_identical_normals_returns_one(self):
        points = torch.rand(100, 3).cuda()
        normals = torch.randn(100, 3).cuda()
        normals = normals / normals.norm(dim=1, keepdim=True)
        nc = normal_consistency(points, normals, points, normals)
        assert nc > 0.99, f"NC of identical normals should be ~1.0, got {nc}"

    def test_opposite_normals_returns_one(self):
        points = torch.rand(50, 3).cuda()
        normals = torch.randn(50, 3).cuda()
        normals = normals / normals.norm(dim=1, keepdim=True)
        nc = normal_consistency(points, normals, points, -normals)
        assert nc > 0.99, f"NC of opposite normals should be ~1.0 (abs cosine), got {nc}"
