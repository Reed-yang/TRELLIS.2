"""Unit tests for corep_fast/profiling/report_builder.py."""
import json
from pathlib import Path

from corep_fast.profiling.report_builder import build_markdown_summary


def test_build_markdown_summary_basic(tmp_path: Path):
    data = [
        {
            'mesh': 'sphere.ply',
            'resolution': 256,
            'status': 'ok',
            'num_cubes': 100,
            'total_pipeline_s': 5.0,
            'stages': {
                's1_voxelize': {'wall_time_s': 2.0, 'peak_gpu_mem_bytes': 1000},
                's6_collapse_face': {'wall_time_s': 2.5, 'peak_gpu_mem_bytes': 2000},
            },
        },
    ]
    json_path = tmp_path / 'profile.json'
    json_path.write_text(json.dumps(data))
    md = build_markdown_summary(str(json_path))
    assert '# CoReP Profiling Summary' in md
    assert 'sphere.ply' in md
    assert 's1_voxelize' in md
    assert '2.00' in md or '2.0' in md
