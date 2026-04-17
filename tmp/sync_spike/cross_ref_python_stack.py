"""Cross-reference the 1.0s cudaDeviceSynchronize timestamp with torch.profiler
python_function events to recover the offending Python frame.

Input:
  - tmp/sync_spike/nsys_top_devicesync.json      (from Task 2)
  - tmp/profile_deep/results/layer1_res256_run1_trace.json  (Chrome trace)

Chrome trace events have keys: name, cat, ph, ts (microseconds), dur (us), args,
pid, tid. For Begin/End (ph=X = complete) events, [ts, ts+dur] is the interval.

IMPORTANT: nsys timestamps are absolute (Unix epoch ns or monotonic ns from
boot depending on nsys version). torch.profiler timestamps are relative to
profiler start. The two timebases differ. This script therefore matches by
RELATIVE POSITION inside the profile window, not by absolute ts:

  1. Find the profiler window [ts_min, ts_max] in the Chrome trace.
  2. Normalize nsys_top_devicesync: fraction = (start_ns - nsys_window_min) /
     (nsys_window_max - nsys_window_min). We approximate this by assuming the
     profiled section starts at nsys_res256_full.sqlite's first runtime event
     and ends at the last (single-run nsys capture; pipeline call is bracketed
     by warmup which should be the dominant pre-measurement phase).
  3. Project the target timestamp into Chrome trace coordinates.
  4. Enumerate python_function events that contain that projected ts.

This is approximate but sufficient to narrow down to 1-3 candidate frames; the
grep step (Task 4) confirms which one.

Usage: python -m tmp.sync_spike.cross_ref_python_stack
"""
import ijson
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NSYS_TOP = REPO_ROOT / "tmp/sync_spike/nsys_top_devicesync.json"
SQLITE_PATH = REPO_ROOT / "tmp/profile_deep/results/nsys_res256_full.sqlite"
TRACE_PATH = REPO_ROOT / "tmp/profile_deep/results/layer1_res256_run1_trace.json"


def nsys_window(sqlite_path: Path, runtime_table: str) -> tuple:
    with sqlite3.connect(str(sqlite_path)) as con:
        cur = con.execute(f"SELECT MIN(start), MAX(end) FROM {runtime_table}")
        lo, hi = cur.fetchone()
    return int(lo), int(hi)


def find_runtime_table(con: sqlite3.Connection) -> str:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name LIKE '%CUPTI_ACTIVITY_KIND_RUNTIME%' OR name LIKE '%CUDA_API%')"
    )
    rows = [r[0] for r in cur.fetchall()]
    if not rows:
        raise RuntimeError("runtime table not found")
    return rows[0]


def trace_window(trace_path: Path) -> tuple:
    """Return (ts_min_us, ts_max_us) of python_function + cpu_op events."""
    ts_min = 10**18
    ts_max = 0
    with open(trace_path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("cat") not in ("python_function", "cpu_op"):
                continue
            ts = ev.get("ts")
            dur = ev.get("dur", 0)
            if ts is None:
                continue
            ts = float(ts)
            dur = float(dur)
            ts_min = min(ts_min, ts)
            ts_max = max(ts_max, ts + dur)
    return float(ts_min), float(ts_max)


def find_containing_python_frames(
    trace_path: Path, target_ts_us: float
) -> list:
    """Find python_function events whose [ts, ts+dur] contains the target.

    We return the top candidates (shortest dur first = deepest stack frame
    most specific to the call).
    """
    hits = []
    with open(trace_path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("cat") != "python_function":
                continue
            ts = ev.get("ts")
            dur = ev.get("dur", 0)
            if ts is None or dur == 0:
                continue
            ts = float(ts)
            dur = float(dur)
            if ts <= target_ts_us <= ts + dur:
                hits.append({
                    "name": ev.get("name", ""),
                    "ts": ts,
                    "dur": dur,
                    "args": ev.get("args", {}),
                })
    hits.sort(key=lambda h: h["dur"])
    return hits


def find_long_python_frames(trace_path: Path, min_dur_us: float) -> list:
    """Fallback: find python_function events with dur >= min_dur_us."""
    hits = []
    with open(trace_path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("cat") != "python_function":
                continue
            ts = ev.get("ts")
            dur = ev.get("dur", 0)
            if ts is None or dur == 0:
                continue
            ts = float(ts)
            dur = float(dur)
            if dur >= min_dur_us:
                hits.append({
                    "name": ev.get("name", ""),
                    "ts": ts,
                    "dur": dur,
                    "args": ev.get("args", {}),
                })
    hits.sort(key=lambda h: h["dur"], reverse=True)
    return hits


def main():
    if not NSYS_TOP.exists():
        sys.exit(f"Missing {NSYS_TOP} — run Task 2 first.")
    top = json.loads(NSYS_TOP.read_text())

    with sqlite3.connect(str(SQLITE_PATH)) as con:
        table = find_runtime_table(con)
    nsys_lo, nsys_hi = nsys_window(SQLITE_PATH, table)

    print("[step1] Computing torch.profiler trace window (pass 1 of 2)...")
    tr_lo_us, tr_hi_us = trace_window(TRACE_PATH)
    print(f"[info] nsys window: {nsys_lo}..{nsys_hi} ns "
          f"(span {(nsys_hi - nsys_lo)/1e9:.2f}s)")
    print(f"[info] torch.profiler window: {tr_lo_us}..{tr_hi_us} us "
          f"(span {(tr_hi_us - tr_lo_us)/1e6:.2f}s)")
    if tr_lo_us >= tr_hi_us:
        sys.exit("[error] trace_window returned empty/invalid range — "
                 "no python_function or cpu_op events found in trace.")

    # Relative position of the 1s sync in nsys coordinates
    nsys_span = nsys_hi - nsys_lo
    frac_start = (top["start_ns"] - nsys_lo) / nsys_span
    frac_end = (top["end_ns"] - nsys_lo) / nsys_span

    tr_span_us = tr_hi_us - tr_lo_us
    projected_ts_us = tr_lo_us + frac_start * tr_span_us
    projected_end_us = tr_lo_us + frac_end * tr_span_us

    print(f"[info] projected target: [{projected_ts_us:.0f}, {projected_end_us:.0f}] us "
          f"(fraction {frac_start:.4f} - {frac_end:.4f})")

    print("[step2] Searching for containing python_function frames (pass 2 of 2)...")
    hits = find_containing_python_frames(
        TRACE_PATH, projected_ts_us
    )

    fallback_applied = None

    if len(hits) == 0:
        print("[warn] Projection missed (N=0). Falling back to long-duration search "
              "(dur >= 500_000 us).")
        fallback_applied = "long_dur_fallback"
        hits = find_long_python_frames(TRACE_PATH, min_dur_us=500_000)
        print(f"[fallback] Found {len(hits)} python_function frames with dur >= 500ms.")
    elif len(hits) > 100:
        max_dur = top["dur_ms"] * 1e3 * 3  # 3x the sync duration in us
        print(f"[warn] Too many hits (N={len(hits)}). Narrowing to dur < {max_dur:.0f} us.")
        fallback_applied = f"narrow_3x_sync_dur (max_dur={max_dur:.0f}us)"
        hits = [h for h in hits if h["dur"] < max_dur]
        print(f"[fallback] After narrowing: {len(hits)} frames.")

    print(f"\nFound {len(hits)} python_function frames containing target. "
          f"Showing shortest-dur top 10 (deepest-stack most specific):")
    print(f"{'dur(us)':>12}  {'name':<60}")
    for h in hits[:10]:
        print(f"{h['dur']:>12}  {h['name'][:60]}")

    if fallback_applied:
        print(f"\n[note] Fallback applied: {fallback_applied}")

    out_path = REPO_ROOT / "tmp/sync_spike/cross_ref_candidates.json"
    with open(out_path, "w") as f:
        json.dump(hits[:10], f, indent=2, default=str)
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
