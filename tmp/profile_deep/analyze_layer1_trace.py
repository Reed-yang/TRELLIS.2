"""Parse torch.profiler Chrome trace, emit per-stage top-N op CSV.

Uses ijson streaming to handle large (>2GB) trace files without loading the
full JSON into memory.

Usage:
    python tmp/profile_deep/analyze_layer1_trace.py \\
        tmp/profile_deep/results/layer1_res256_run2_trace.json \\
        tmp/profile_deep/results/per_stage_ops_res256.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import ijson


# Stage labels (canonical names matching the monkeypatch_nvtx NVTX labels)
# paired with the python_function *suffix* that the torch profiler records
# for the outer stage wrapper. Torch emits python_function events named like
# "corep_fast/stages/s1_voxelize.py(18): s1_voxelize", so we match on the
# ": <fn>" suffix. Note s8's canonical label is "s8_decode" but the actual
# entry function is `decode_from_cubebatch` inside s8_collapse.py.
STAGE_FN_MAP = {
    "s1_voxelize": "s1_voxelize",
    "s2_components": "s2_components",
    "s3_edge_weights": "s3_edge_weights",
    "s4_face_point": "s4_face_point",
    "s6_collapse": "s6_collapse",
    "s7_rank_assign": "s7_rank_assign",
    "s8_decode": "decode_from_cubebatch",
}

TOP_N = 30


def iter_events(path: str):
    """Stream 'X' (Complete) events from a torch.profiler Chrome trace.

    Torch profiler produces `{"schemaVersion":..., "deviceProperties":[...],
    "traceEvents":[...], ...}`. We stream traceEvents array only to avoid
    loading the whole (~3.8GB) JSON into memory.
    """
    with open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            yield ev


def stage_label_for_event(name: str):
    """Return the canonical stage label if `name` matches the outer
    stage-wrapper python_function event, else None.

    We match the ": <fn>" suffix to pick only the outermost wrapper (avoids
    helper functions like `_expand_candidates` in the same file). For s8 the
    wrapper is `decode_from_cubebatch` but we label it as `s8_decode`.
    """
    for lbl, fn in STAGE_FN_MAP.items():
        if name.endswith(": " + fn):
            return lbl
    return None


def find_stage(ts, ranges):
    # ranges sorted by start_ts ascending; linear scan is fine for 7 entries.
    for name, s, e in ranges:
        if s <= ts <= e:
            return name
    return "__unassigned__"


def main():
    trace_path = sys.argv[1]
    out_csv = sys.argv[2]

    # Pass 1: collect stage ranges (the outer python_function wrapper spans).
    ranges = []
    for ev in iter_events(trace_path):
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        lbl = stage_label_for_event(name)
        if lbl is None:
            continue
        # Cast Decimal to float for arithmetic + comparison downstream.
        ts = float(ev.get("ts", 0))
        dur = float(ev.get("dur", 0))
        ranges.append((lbl, ts, ts + dur))
    ranges.sort(key=lambda r: r[1])
    print(
        f"[info] pass 1: {len(ranges)} stage ranges found: "
        f"{[r[0] for r in ranges]}",
        file=sys.stderr,
    )

    # Pass 2: aggregate per-(stage, op) stats.
    agg = defaultdict(lambda: {"count": 0, "cpu_us": 0.0, "device_us": 0.0})
    total_events = 0
    for ev in iter_events(trace_path):
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        # Skip the stage-wrapper events themselves (they're range markers).
        if stage_label_for_event(name) is not None:
            continue
        ts = float(ev.get("ts", 0))
        dur = float(ev.get("dur", 0))
        stage = find_stage(ts, ranges)
        key = (stage, name)
        total_events += 1
        agg[key]["count"] += 1
        cat = (ev.get("cat") or "").lower()
        # Heuristic: torch.profiler marks GPU kernels with cat like "kernel".
        if "kernel" in cat:
            agg[key]["device_us"] += dur
        else:
            agg[key]["cpu_us"] += dur
    print(
        f"[info] pass 2: {total_events} events aggregated into "
        f"{len(agg)} (stage,op) keys",
        file=sys.stderr,
    )

    # Top-N per stage by device_us (fallback cpu_us).
    per_stage = defaultdict(list)
    for (stage, op), stats in agg.items():
        per_stage[stage].append(
            (op, stats["count"], stats["cpu_us"], stats["device_us"])
        )
    for stage, rows in per_stage.items():
        rows.sort(key=lambda r: (r[3], r[2]), reverse=True)
        per_stage[stage] = rows[:TOP_N]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "op_name", "count", "cpu_us", "device_us"])
        for stage in sorted(per_stage.keys()):
            for op, count, cpu_us, dev_us in per_stage[stage]:
                w.writerow([stage, op, count, f"{cpu_us:.1f}", f"{dev_us:.1f}"])
    total_rows = sum(len(r) for r in per_stage.values())
    print(
        f"[write] {out_csv}  ({total_rows} rows across {len(per_stage)} stages)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
