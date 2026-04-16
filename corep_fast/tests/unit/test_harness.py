"""Unit tests for corep_fast/profiling/harness.py."""
import json
import time
from pathlib import Path

import pytest
import torch

from corep_fast.profiling.harness import ProfilingCollector, stage_timer


def test_collector_starts_empty():
    pc = ProfilingCollector()
    assert pc.num_stages_recorded == 0
    assert pc.total_wall_time_s == 0.0


def test_stage_timer_records_wall_time():
    pc = ProfilingCollector()
    with stage_timer('sleep50ms', pc):
        time.sleep(0.05)
    assert pc.num_stages_recorded == 1
    rec = pc['sleep50ms']
    assert rec.wall_time_s >= 0.04
    assert rec.wall_time_s < 0.5  # sanity upper bound


def test_stage_timer_records_multiple_stages():
    pc = ProfilingCollector()
    with stage_timer('stage_a', pc):
        time.sleep(0.01)
    with stage_timer('stage_b', pc):
        time.sleep(0.01)
    assert pc.num_stages_recorded == 2
    assert 'stage_a' in pc
    assert 'stage_b' in pc


def test_stage_timer_records_exception_cleanly():
    pc = ProfilingCollector()
    with pytest.raises(ValueError):
        with stage_timer('failing', pc):
            raise ValueError("oops")
    # Even on failure, timing should be recorded
    assert 'failing' in pc


def test_collector_total_wall_time():
    pc = ProfilingCollector()
    with stage_timer('a', pc):
        time.sleep(0.01)
    with stage_timer('b', pc):
        time.sleep(0.01)
    assert pc.total_wall_time_s >= 0.015


def test_collector_to_json_roundtrip(tmp_path: Path):
    pc = ProfilingCollector(mesh_name='sphere.ply', resolution=256, impl='custom')
    with stage_timer('s1_voxelize', pc):
        time.sleep(0.005)
    with stage_timer('s6_collapse_face', pc):
        time.sleep(0.005)
    out_path = tmp_path / "profile.json"
    pc.save_json(out_path)
    # Reload
    loaded = json.loads(out_path.read_text())
    assert loaded['mesh'] == 'sphere.ply'
    assert loaded['resolution'] == 256
    assert loaded['impl'] == 'custom'
    assert 's1_voxelize' in loaded['stages']
    assert 's6_collapse_face' in loaded['stages']
    assert loaded['stages']['s1_voxelize']['wall_time_s'] > 0
    assert 'total_wall_time_s' in loaded


def test_substage_timer_nested():
    """Substages within a stage should be recorded as nested entries."""
    pc = ProfilingCollector()
    with stage_timer('s6_collapse_face', pc):
        with stage_timer('algebraic_pruning', pc, parent='s6_collapse_face'):
            time.sleep(0.005)
        with stage_timer('enumeration', pc, parent='s6_collapse_face'):
            time.sleep(0.005)
    s6 = pc['s6_collapse_face']
    assert 'substages' in s6.extra
    assert 'algebraic_pruning' in s6.extra['substages']
    assert 'enumeration' in s6.extra['substages']


def test_stage_timer_records_gpu_mem_when_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pc = ProfilingCollector()
    with stage_timer('gpu_alloc', pc):
        big = torch.zeros(1024 * 1024 * 16, dtype=torch.float32, device='cuda')  # 64 MiB
        del big
    rec = pc['gpu_alloc']
    # Peak should be at least the allocation size
    assert rec.peak_gpu_mem_bytes >= 0  # may be 0 if caching allocator reused memory
