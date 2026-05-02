"""Tests for scripts/coart_compare_occupancy.py."""
import importlib.util
import pathlib

import numpy as np
import pytest
import trimesh

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module():
    p = ROOT / "scripts" / "coart_compare_occupancy.py"
    spec = importlib.util.spec_from_file_location("coart_compare_occupancy", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_max_pool_correctness():
    co = _load_module()
    cubes = np.array([[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=np.int32)
    pooled = co.downsample_cubes(cubes, factor=2)
    expected = np.unique(np.array([[0,0,0], [0,0,0], [1,1,1], [1,1,1]], dtype=np.int32), axis=0)
    np.testing.assert_array_equal(np.sort(pooled, axis=0), np.sort(expected, axis=0))


def test_iou_function():
    co = _load_module()
    a = np.array([[0,0,0], [1,1,1], [2,2,2]], dtype=np.int32)
    b = np.array([[0,0,0], [1,1,1], [3,3,3]], dtype=np.int32)
    # intersection={0,1}, union={0,1,2,3} → 2/4 = 0.5
    assert abs(co.iou(a, b) - 0.5) < 1e-6
    assert co.iou(a, a) == 1.0
    assert co.iou(np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)) == 1.0
