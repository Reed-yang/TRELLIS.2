"""
Build Markdown summary from profiling JSON output.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_markdown_summary(json_path: str) -> str:
    """Read a baseline_runner JSON and produce a Markdown summary string."""
    data = json.loads(Path(json_path).read_text())
    lines = ["# CoReP Profiling Summary", ""]

    # Header table
    lines.append("| Mesh | Resolution | Cubes | Total (s) | Status |")
    lines.append("|------|-----------|------:|----------:|--------|")
    for entry in data:
        mesh = entry.get('mesh', '?')
        res = entry.get('resolution', '?')
        cubes = entry.get('num_cubes', '?')
        total = entry.get('total_pipeline_s', 0)
        status = entry.get('status', '?')
        lines.append(f"| {mesh} | {res} | {cubes} | {total:.2f} | {status} |")

    # Per-stage breakdown
    lines.extend(["", "## Per-Stage Breakdown", ""])
    for entry in data:
        if entry.get('status') != 'ok':
            continue
        mesh = entry.get('mesh', '?')
        res = entry.get('resolution', '?')
        stages = entry.get('stages', {})
        total = entry.get('total_pipeline_s', 1)

        lines.append(f"### {mesh} @ {res}")
        lines.append("")
        lines.append("| Stage | Time (s) | % of Total | Peak GPU (MB) |")
        lines.append("|-------|--------:|-----------:|--------------:|")
        for sname, sdata in sorted(stages.items()):
            wall = sdata.get('wall_time_s', 0)
            pct = (wall / total * 100) if total > 0 else 0
            mem = sdata.get('peak_gpu_mem_bytes', 0) / (1024 * 1024)
            lines.append(f"| {sname} | {wall:.2f} | {pct:.1f}% | {mem:.1f} |")
        lines.append("")

    return "\n".join(lines)
