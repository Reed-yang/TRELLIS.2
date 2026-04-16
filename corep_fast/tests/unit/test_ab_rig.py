"""Unit tests for corep_fast/profiling/ab_rig.py."""
from unittest import mock

import pytest
import torch

from corep_fast.profiling.ab_rig import ABReport, ab_run_from_cube_batches


def _make_identical_pair():
    from corep_fast.containers import CubeBatch
    from corep_fast.profiling.harness import ProfilingCollector
    cb = CubeBatch.empty(num_cubes=2, resolution=64, device=torch.device('cpu'))
    pc = ProfilingCollector()
    return cb, cb, pc, pc


def test_ab_report_all_passed_on_identical():
    cb_a, cb_b, pc_a, pc_b = _make_identical_pair()
    report = ab_run_from_cube_batches(cb_a, cb_b, pc_a, pc_b)
    assert report.equivalence.all_passed()


def test_ab_report_speedup_calculation():
    from corep_fast.profiling.harness import ProfilingCollector, StageRecord
    pc_a = ProfilingCollector()
    pc_a._records['s1_voxelize'] = StageRecord(wall_time_s=10.0)
    pc_b = ProfilingCollector()
    pc_b._records['s1_voxelize'] = StageRecord(wall_time_s=1.0)

    from corep_fast.containers import CubeBatch
    cb = CubeBatch.empty(num_cubes=2, resolution=64, device=torch.device('cpu'))
    report = ab_run_from_cube_batches(cb, cb, pc_a, pc_b)
    assert report.speedups_per_stage['s1_voxelize'] == pytest.approx(10.0, rel=0.01)
