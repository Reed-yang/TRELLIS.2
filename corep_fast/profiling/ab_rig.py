"""
A/B comparison rig: run custom/ and corep_fast/ on the same mesh and
compare topology equivalence + speedup.

Reference: spec §7.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from corep_fast.containers import CubeBatch
from corep_fast.profiling.harness import ProfilingCollector
from corep_fast.profiling.topology_equivalence import (
    EquivalenceReport,
    check_topology_equivalence,
)


@dataclass
class ABReport:
    """Result of a single-mesh A/B comparison."""
    equivalence: EquivalenceReport
    custom_timings: ProfilingCollector
    fast_timings: ProfilingCollector
    speedups_per_stage: dict[str, float] = field(default_factory=dict)
    speedup_total: float = 1.0

    def summary_str(self) -> str:
        lines = [
            "A/B Report",
            f"  Equivalence: {'PASS' if self.equivalence.all_passed() else 'FAIL'}",
            f"  Total speedup: {self.speedup_total:.1f}×",
        ]
        for stage, sp in sorted(self.speedups_per_stage.items()):
            lines.append(f"    {stage}: {sp:.1f}×")
        return "\n".join(lines)


def ab_run_from_cube_batches(
    cb_custom: CubeBatch,
    cb_fast: CubeBatch,
    timings_custom: ProfilingCollector,
    timings_fast: ProfilingCollector,
    layers: list[str] | None = None,
) -> ABReport:
    equiv = check_topology_equivalence(cb_custom, cb_fast, layers=layers)

    speedups: dict[str, float] = {}
    for stage_name in timings_custom._records:
        t_custom = timings_custom._records[stage_name].wall_time_s
        t_fast = timings_fast._records.get(stage_name)
        if t_fast is not None and t_fast.wall_time_s > 0:
            speedups[stage_name] = t_custom / t_fast.wall_time_s
        elif t_custom > 0:
            speedups[stage_name] = float('inf')

    total_custom = timings_custom.total_wall_time_s
    total_fast = timings_fast.total_wall_time_s
    total_speedup = total_custom / total_fast if total_fast > 0 else float('inf')

    return ABReport(
        equivalence=equiv,
        custom_timings=timings_custom,
        fast_timings=timings_fast,
        speedups_per_stage=speedups,
        speedup_total=total_speedup,
    )
