"""Unit tests for corep_fast/profiling/topology_equivalence.py."""
import pytest
import torch

from corep_fast.containers import CubeBatch
from corep_fast.profiling.topology_equivalence import (
    EquivalenceReport,
    check_layer1_integer_fields,
    check_topology_equivalence,
)


def _make_cube_batch_with_fields(num_cubes=3, device='cpu') -> CubeBatch:
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device(device))
    cb.num_components[:] = torch.arange(num_cubes, dtype=torch.int32) % 3 + 1
    cb.num_boundary[:] = torch.arange(num_cubes, dtype=torch.int32) % 2
    cb.edge_weights[:] = torch.arange(num_cubes * 18, dtype=torch.int32).reshape(num_cubes, 18) % 4
    cb.face_weights[:] = torch.arange(num_cubes * 12, dtype=torch.int32).reshape(num_cubes, 12) % 3
    cb.status[:] = torch.zeros(num_cubes, dtype=torch.int32)
    return cb


def test_l1_equivalence_passes_on_identical_batches():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is True
    assert report.num_cubes == 3
    assert len(report.mismatches) == 0


def test_l1_equivalence_fails_on_num_components_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.num_components[1] = 3  # diverge
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'num_components' for m in report.mismatches)
    assert report.mismatches[0].cube_idx == 1
    assert report.mismatches[0].value_a == 2
    assert report.mismatches[0].value_b == 3


def test_l1_equivalence_fails_on_edge_weights_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.edge_weights[0, 5] = 99  # diverge at cube 0, edge 5
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'edge_weights' for m in report.mismatches)


def test_l1_equivalence_fails_on_face_weights_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.face_weights[2, 7] = 99
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'face_weights' for m in report.mismatches)


def test_l1_equivalence_fails_on_status_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.status[0] = 2  # UNSOLVABLE
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'status' for m in report.mismatches)


def test_l1_equivalence_fails_on_shape_mismatch():
    cb_a = _make_cube_batch_with_fields(num_cubes=3)
    cb_b = _make_cube_batch_with_fields(num_cubes=4)
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'shape' for m in report.mismatches)


def test_top_level_check_layer1_only_runs_when_shapes_match():
    """The top-level check_topology_equivalence should short-circuit on shape mismatch."""
    cb_a = _make_cube_batch_with_fields(num_cubes=3)
    cb_b = _make_cube_batch_with_fields(num_cubes=4)
    report = check_topology_equivalence(cb_a, cb_b, layers=['l1'])
    assert not report.all_passed()
    assert report.layer1.passed is False
