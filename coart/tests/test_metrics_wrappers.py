"""Unit tests for coart.eval.metrics wrappers using trimesh primitives."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from coart.eval.metrics import (
    chamfer_distance,
    compute_topo_metrics,
    f_score_multi,
    normal_consistency,
    sample_surface,
)


@pytest.fixture
def icosphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


def test_sample_surface_shapes(icosphere):
    pts, nrms = sample_surface(icosphere, num_points=10000)
    assert pts.shape == (10000, 3)
    assert nrms.shape == (10000, 3)
    assert pts.dtype == np.float32
    assert nrms.dtype == np.float32


def test_chamfer_self_is_zero(icosphere):
    pts, _ = sample_surface(icosphere, num_points=5000)
    assert chamfer_distance(pts, pts) < 1e-6


def test_normal_consistency_self_is_one(icosphere):
    pts, nrms = sample_surface(icosphere, num_points=5000)
    assert normal_consistency(pts, nrms, pts, nrms) > 0.99


def test_f_score_self_is_one(icosphere):
    pts, _ = sample_surface(icosphere, num_points=5000)
    fs = f_score_multi(pts, pts, thresholds=[0.001, 0.01, 0.1])
    for thr, v in fs.items():
        assert v > 0.999, f"F@{thr} self-score should ≈1.0, got {v}"


def test_topo_metrics_watertight_sphere(icosphere):
    m = compute_topo_metrics(icosphere)
    assert m["is_watertight"] == 1.0
    assert m["n_components"] == 1.0
    assert abs(m["euler_number"] - 2.0) < 0.5
