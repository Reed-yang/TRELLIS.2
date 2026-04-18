"""Extract top-20 blocking D2H hotspots from torch.profiler Chrome trace.

Algorithm:
  1. First pass — collect (ts, dur, name, cat, args) for:
     - Events with cat == 'gpu_memcpy' whose name suggests D2H
       (e.g. 'Memcpy DtoH')
     - Events with cat == 'cuda_runtime' whose name in
       {'cudaStreamSynchronize', 'cudaDeviceSynchronize'}
     - All python_function events (for stack attribution)
  2. Second pass — for each memcpy/sync event, find the deepest-stack
     python_function event whose [ts, ts+dur] strictly contains the
     memcpy/sync event's ts. That gives the innermost Python frame.
  3. Aggregate (stage, frame_name) -> (count, total_dur_ms)
  4. Sort by total_dur_ms desc, take top 20.

Output: tmp/sync_spike/top20_d2h.json — list of 20 records:
  {rank, stage, frame_name, count, total_dur_ms, avg_dur_us, sample_ts_us}

Stage is derived from the first python_function ancestor matching
{'s1_voxelize','s2_components','s3_edge_weights','s4_face_point',
 's6_collapse','s7_rank_assign','s8_decode','decode_from_cubebatch'}.

Usage: python -m tmp.sync_spike.extract_top20_d2h
"""
import ijson
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = REPO_ROOT / "tmp/profile_deep/results/layer1_res256_run1_trace.json"

STAGE_NAMES = {
    "s1_voxelize", "s2_components", "s3_edge_weights", "s4_face_point",
    "s6_collapse", "s7_rank_assign", "s8_decode", "decode_from_cubebatch",
}


def first_pass(trace_path: Path):
    """Return (memcpy_sync, py_events).

    memcpy_sync: list of (ts_us, dur_us, kind, name) where kind in {'d2h','sync'}
    py_events:    list of (ts, end_ts, name, dur) for all python_function events
    """
    memcpy_sync = []
    py_events = []
    with open(trace_path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            cat = ev.get("cat", "")
            name = ev.get("name", "")
            ts = ev.get("ts")
            dur = ev.get("dur", 0)
            if ts is None:
                continue
            ts = float(ts)
            dur = float(dur) if dur else 0.0
            if cat == "gpu_memcpy" and "DtoH" in name:
                memcpy_sync.append((ts, dur, "d2h", name))
            elif cat == "cuda_runtime" and name in (
                "cudaStreamSynchronize", "cudaDeviceSynchronize"
            ):
                memcpy_sync.append((ts, dur, "sync", name))
            elif cat == "python_function" and dur > 0:
                py_events.append((ts, ts + dur, name, dur))
    return memcpy_sync, py_events


def attribute(memcpy_sync, py_events):
    """For each event, find the deepest-stack containing python_function frame.

    For speed: sort py_events by ts. For each target, binary-search candidates
    with ts <= target_ts, then among those, retain ones whose end_ts >= target_ts.
    The deepest is the candidate with the smallest interval (end - start) that
    still contains the target.
    """
    import bisect
    py_events.sort(key=lambda e: e[0])
    ts_list = [e[0] for e in py_events]

    attributed = []
    for ev_ts, ev_dur, kind, ev_name in memcpy_sync:
        i = bisect.bisect_right(ts_list, ev_ts)
        cands = []
        # 10k-event backward scan: python_function events in this trace span
        # ~14M entries over ~13s wall-time (~100k events/sec). A 10k-event
        # window covers ~0.1s of trace, far exceeding any realistic single
        # python call-stack lifetime (depth ≤ ~200 frames). If a future trace
        # has a long-running top-level frame that spans more than 10k nested
        # events, increase this bound.
        for j in range(i - 1, max(-1, i - 10_000), -1):
            p_ts, p_end, p_name, p_dur = py_events[j]
            if p_end >= ev_ts:
                cands.append((p_end - p_ts, p_name))
        cands.sort()  # smallest interval first
        deepest = cands[0][1] if cands else "<no frame>"
        stage = "<unknown>"
        # Stage attribution walks outermost-first: STAGE_NAMES are top-level pipeline
        # functions (s1_voxelize, s4_face_point, ...), not inner helpers. The deepest
        # frame is the most specific (for `frame_name`), but the stage must come from
        # the outermost ancestor matching a stage name.
        for _, pname in reversed(cands):
            hit = next((s for s in STAGE_NAMES if s in pname), None)
            if hit is not None:
                stage = hit
                break
        attributed.append({
            "kind": kind,
            "ev_name": ev_name,
            "ts_us": ev_ts,
            "dur_us": ev_dur,
            "frame_name": deepest,
            "stage": stage,
        })
    return attributed


def aggregate_top_n(attributed, n: int = 20):
    """Aggregate by (stage, frame_name) for d2h-only, sort by total_dur desc."""
    from collections import defaultdict
    agg = defaultdict(lambda: {"count": 0, "total_dur_us": 0.0, "sample_ts_us": None,
                                "stage": "", "frame_name": ""})
    for a in attributed:
        if a["kind"] != "d2h":
            continue
        key = (a["stage"], a["frame_name"])
        slot = agg[key]
        slot["count"] += 1
        slot["total_dur_us"] += a["dur_us"]
        slot["stage"] = a["stage"]
        slot["frame_name"] = a["frame_name"]
        if slot["sample_ts_us"] is None:
            slot["sample_ts_us"] = a["ts_us"]

    rows = []
    for (stage, frame), v in agg.items():
        rows.append({
            "stage": stage,
            "frame_name": frame,
            "count": v["count"],
            "total_dur_ms": v["total_dur_us"] / 1000,
            "avg_dur_us": v["total_dur_us"] / max(v["count"], 1),
            "sample_ts_us": v["sample_ts_us"],
        })
    rows.sort(key=lambda r: -r["total_dur_ms"])
    for i, r in enumerate(rows[:n], 1):
        r["rank"] = i
    return rows[:n]


def main():
    if not TRACE_PATH.exists():
        sys.exit(f"Missing {TRACE_PATH}")

    print(f"[scan] reading {TRACE_PATH} (~4GB, streaming)...")
    memcpy_sync, py_events = first_pass(TRACE_PATH)
    print(f"[info] memcpy/sync events: {len(memcpy_sync)}")
    print(f"[info] python_function events: {len(py_events)}")

    d2h_count = sum(1 for e in memcpy_sync if e[2] == "d2h")
    sync_count = sum(1 for e in memcpy_sync if e[2] == "sync")
    print(f"[info] D2H memcpy: {d2h_count}")
    print(f"[info] stream/device sync: {sync_count}")

    if d2h_count < 2000:
        print(f"[warn] D2H count {d2h_count} < 2000; expected ~2011 for this "
              f"trace (see tmp/sync_spike/extract_top20_d2h.log).")

    attributed = attribute(memcpy_sync, py_events)
    top20 = aggregate_top_n(attributed, n=20)

    print(f"\nTop 20 blocking D2H hotspots by aggregate dur_ms:")
    print(f"{'rank':<5}{'stage':<20}{'count':>8}{'total_ms':>12}  {'frame_name'}")
    for r in top20:
        print(f"{r['rank']:<5}{r['stage']:<20}{r['count']:>8}"
              f"{r['total_dur_ms']:>12.3f}  {r['frame_name'][:60]}")

    out_path = REPO_ROOT / "tmp/sync_spike/top20_d2h.json"
    with open(out_path, "w") as f:
        json.dump(top20, f, indent=2)
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
