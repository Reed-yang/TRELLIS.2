"""Aggregate GPU kernels across all stages from torch.profiler trace, emit top-20
with (stage, py-line, heuristic class) attribution.

Uses ijson streaming. Stage ranges extracted from python_function events
matching the 7 stage entry functions (nvtx markers are NOT captured in
torch.profiler Chrome trace — confirmed in Task 8).

Heuristic classes:
- LNB: launch-bound   — mean_us < 10 AND count > 1000
- MMB: memory-bound   — name matches {copy, memset, scatter, gather, index, cat, slice}
- CMB: compute-bound  — name matches {gemm, conv, reduce, sum, matmul} AND mean_us > 100
- CPU: CPU-bound      — stage-level time minus sum(stage GPU kernels) > 30% stage wall
                        AND this kernel's mean_us < 50 (so attributed "incidental GPU work in a CPU-heavy stage")
- UNK: unknown        — couldn't be classified; needs ncu

Usage:
    python tmp/profile_deep/analyze_layer2_kernels.py \\
        tmp/profile_deep/results/layer1_res256_run2_trace.json \\
        tmp/profile_deep/results/top20_kernels_res256.csv
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import ijson


# Reuse the same stage-function mapping as Task 8's analyzer
STAGE_FN_MAP = {
    "s1_voxelize": "s1_voxelize",
    "s2_components": "s2_components",
    "s3_edge_weights": "s3_edge_weights",
    "s4_face_point": "s4_face_point",
    "s6_collapse": "s6_collapse",
    "s7_rank_assign": "s7_rank_assign",
    "s8_decode": "decode_from_cubebatch",  # actual fn name per Task 8 finding
}

MMB_PATTERNS = re.compile(
    r"(copy|memset|scatter|gather|^index|index_elementwise|cat|slice|"
    r"contiguous|view|as_strided|reduce|scan|"
    r"devicereduce|deviceselect|devicescan)",
    re.IGNORECASE,
)
CMB_PATTERNS = re.compile(r"(gemm|conv|reduce|sum|matmul|bmm)", re.IGNORECASE)


def iter_events(path: str):
    """Stream ph='X' events from a torch.profiler Chrome trace via ijson."""
    with open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            yield ev


def extract_stage_ranges(trace_path: str):
    """Pass 1: find stage NVTX-equivalent (python_function) ranges.

    Returns:
        list of (stage_name, ts_start, ts_end, dur_us) sorted by ts_start.
        dur_us returned as int.
    """
    ranges = []
    for ev in iter_events(trace_path):
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "") or ""
        if not isinstance(name, str):
            continue
        # Stage entry python_function events: name contains ": <fn_name>"
        # (the with_stack frame format puts the function name after the colon)
        for stage, fn_name in STAGE_FN_MAP.items():
            if name.endswith(f": {fn_name}") or name == fn_name:
                ts = float(ev.get("ts", 0) or 0)
                dur = float(ev.get("dur", 0) or 0)
                ranges.append((stage, ts, ts + dur, dur))
                break
    ranges.sort(key=lambda r: r[1])
    return ranges


def find_stage(ts, ranges):
    for name, s, e, _ in ranges:
        if s <= ts <= e:
            return name
    return "__unassigned__"


def extract_py_line(ev):
    """Best-effort py-line from ev args. torch.profiler embeds 'External id',
    'Call stack' (string), etc. Prefer a line mentioning corep_fast."""
    args = ev.get("args") or {}
    stack = args.get("Call stack") or args.get("External id") or ""
    if not isinstance(stack, str):
        return ""
    for line in stack.split("\n"):
        s = line.strip()
        if "corep_fast" in s:
            return s[:120]  # truncate to stay compact in CSV
    first = stack.split("\n")[0].strip()
    return first[:120]


def classify(name: str, mean_us: float, count: int,
             stage_wall_us: float, stage_gpu_us: float) -> str:
    name_s = name or ""
    # 1. LNB — launch-bound: many small kernels
    if mean_us < 10 and count > 1000:
        return "LNB"
    # 2. CMB — compute-bound: gemm/conv/matmul AND large enough per call
    if CMB_PATTERNS.search(name_s) and mean_us > 100:
        return "CMB"
    # 3. MMB — memory-bound by kernel name pattern
    if MMB_PATTERNS.search(name_s):
        return "MMB"
    # 4. CPU — fallback for small incidental GPU work in CPU-dominated stages
    if stage_wall_us > 0:
        cpu_frac = (stage_wall_us - stage_gpu_us) / stage_wall_us
        if cpu_frac > 0.30 and mean_us < 50:
            return "CPU"
    return "UNK"


def main():
    trace_path = sys.argv[1]
    out_csv = sys.argv[2]

    print(f"[pass 1] extracting stage ranges from {trace_path}", file=sys.stderr)
    ranges = extract_stage_ranges(trace_path)
    print(f"[pass 1] found {len(ranges)} stage ranges: {[r[0] for r in ranges]}",
          file=sys.stderr)

    stage_wall = {name: dur for name, _, _, dur in ranges}
    stage_gpu = defaultdict(float)
    kern = defaultdict(lambda: {"count": 0, "total_us": 0.0, "py_line": ""})

    print(f"[pass 2] scanning kernel events", file=sys.stderr)
    for ev in iter_events(trace_path):
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        if "kernel" not in cat:
            continue
        ts = float(ev.get("ts", 0) or 0)
        dur = float(ev.get("dur", 0) or 0)
        name = ev.get("name") or "<anon>"
        stage = find_stage(ts, ranges)
        k = (stage, name)
        kern[k]["count"] += 1
        kern[k]["total_us"] += dur
        if not kern[k]["py_line"]:
            kern[k]["py_line"] = extract_py_line(ev)
        stage_gpu[stage] += dur

    rows = []
    for (stage, name), s in kern.items():
        mean = s["total_us"] / max(1, s["count"])
        cls = classify(name, mean, s["count"],
                       stage_wall.get(stage, 0), stage_gpu.get(stage, 0))
        rows.append({
            "stage": stage,
            "kernel": name[:160],
            "count": s["count"],
            "total_ms": s["total_us"] / 1000,
            "mean_us": mean,
            "class": cls,
            "py_line": s["py_line"],
        })
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    top20 = rows[:20]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "kernel", "count",
                                           "total_ms", "mean_us",
                                           "class", "py_line"])
        w.writeheader()
        for r in top20:
            w.writerow({
                **r,
                "total_ms": f"{r['total_ms']:.3f}",
                "mean_us": f"{r['mean_us']:.2f}",
            })
    print(f"[write] {out_csv}", file=sys.stderr)

    unk = sum(1 for r in top20 if r["class"] == "UNK")
    dist = {c: sum(1 for r in top20 if r["class"] == c) for c in ("CMB", "MMB", "LNB", "CPU", "UNK")}
    print(f"[class distribution] {dist}", file=sys.stderr)
    print(f"[UNK rate] {unk}/20 = {100*unk/20:.0f}%", file=sys.stderr)
    if unk > 4:
        print(f"[WARN] UNK rate > 20% (DoD #4 threshold) — classifier may need tuning",
              file=sys.stderr)


if __name__ == "__main__":
    main()
