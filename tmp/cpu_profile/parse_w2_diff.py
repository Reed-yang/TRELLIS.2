"""Diff pre_w2 vs post_w2 main-thread profiles.

Extracts self-time for:
- posix.fork (expected to drop)
- _thread.lock.acquire (expected to drop partially)
- _fastpath_trace_loops_numpy (unchanged - W4's target, not W2)
- _get_local_components_np (unchanged - W5's target)

Also extracts total wall-time if present in profile.
"""
import pstats
from pathlib import Path

PATHS = {
    "pre_w2":  "tmp/cpu_profile/results_main/main_thread_res256_pre_w2.prof",
    "post_w2": "tmp/cpu_profile/results_main/main_thread_res256_post_w2.prof",
}

TARGETS = [
    "posix.fork", "fork",
    "acquire",
    "_fastpath_trace_loops_numpy",
    "_get_local_components_np",
]

def extract(prof_path: str):
    s = pstats.Stats(prof_path)
    rows = []
    total_tt = 0.0
    for (fname, lineno, fn_name), (cc, nc, tt, ct, _callers) in s.stats.items():
        total_tt += tt
        short = Path(fname).name if fname else ""
        for t in TARGETS:
            if t in fn_name:
                rows.append((fn_name, short, lineno, cc, round(tt * 1000, 2), round(ct * 1000, 2)))
    return total_tt, rows

for tag, path in PATHS.items():
    if not Path(path).exists():
        print(f"MISSING: {path}")
        continue
    total, rows = extract(path)
    print(f"\n=== {tag} (total self_tt={total*1000:.1f}ms) ===")
    for r in sorted(rows, key=lambda r: -r[4])[:30]:
        print(f"  cc={r[3]:>6}  self={r[4]:>7.1f}ms  cum={r[5]:>8.1f}ms  {r[1]}:{r[2]}:{r[0]}")
