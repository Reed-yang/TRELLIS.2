"""
Profiling harness: GPU-fenced stage timer + result collector.

Usage
-----
    pc = ProfilingCollector(mesh_name='sphere.ply', resolution=1024, impl='corep_fast')
    with stage_timer('s1_voxelize', pc):
        ...
    with stage_timer('s6_collapse_face', pc):
        with stage_timer('algebraic_pruning', pc, parent='s6_collapse_face'):
            ...
    pc.save_json('profiling/runs/sphere_1024.json')

Reference: spec §7.1.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Record + Collector
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """One timing record for a single stage (or substage)."""
    wall_time_s: float = 0.0
    peak_gpu_mem_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  # holds substages, notes, etc.


@dataclass
class ProfilingCollector:
    """
    Holds all timing records for one pipeline invocation on one mesh.

    Fields `mesh_name`, `resolution`, `impl` are metadata passed through to
    the JSON output unchanged.
    """
    mesh_name: str = ''
    resolution: int = 0
    impl: str = ''          # 'custom' or 'corep_fast' or 'ab'
    _records: dict[str, StageRecord] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        wall_time_s: float,
        peak_gpu_mem_bytes: int = 0,
        parent: Optional[str] = None,
    ) -> None:
        rec = StageRecord(
            wall_time_s=wall_time_s, peak_gpu_mem_bytes=peak_gpu_mem_bytes,
        )
        if parent is None:
            # Preserve existing extra (e.g. substages recorded by nested timers)
            if stage in self._records:
                rec.extra = self._records[stage].extra
            self._records[stage] = rec
        else:
            if parent not in self._records:
                self._records[parent] = StageRecord()
            self._records[parent].extra.setdefault('substages', {})[stage] = \
                {'wall_time_s': wall_time_s, 'peak_gpu_mem_bytes': peak_gpu_mem_bytes}

    @property
    def num_stages_recorded(self) -> int:
        return len(self._records)

    @property
    def total_wall_time_s(self) -> float:
        return sum(r.wall_time_s for r in self._records.values())

    def __contains__(self, stage: str) -> bool:
        return stage in self._records

    def __getitem__(self, stage: str) -> StageRecord:
        return self._records[stage]

    def save_json(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'mesh': self.mesh_name,
            'resolution': self.resolution,
            'impl': self.impl,
            'stages': {
                name: {
                    'wall_time_s': rec.wall_time_s,
                    'peak_gpu_mem_bytes': rec.peak_gpu_mem_bytes,
                    **({'substages': rec.extra['substages']}
                       if 'substages' in rec.extra else {}),
                }
                for name, rec in self._records.items()
            },
            'total_wall_time_s': self.total_wall_time_s,
            'total_peak_gpu_mem_bytes': max(
                (r.peak_gpu_mem_bytes for r in self._records.values()),
                default=0,
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Stage timer context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def stage_timer(
    name: str,
    collector: ProfilingCollector,
    *,
    parent: Optional[str] = None,
):
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
        mem_before = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_before = 0

    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda_available:
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() - mem_before
        else:
            peak_mem = 0
        wall = time.perf_counter() - t0
        collector.record(
            stage=name,
            wall_time_s=wall,
            peak_gpu_mem_bytes=max(peak_mem, 0),
            parent=parent,
        )
