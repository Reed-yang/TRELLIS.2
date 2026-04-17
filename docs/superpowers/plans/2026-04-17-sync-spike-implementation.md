# Sync-Spike + Tooling Quick-Win — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 执行 `docs/superpowers/specs/2026-04-17-sync-spike-design.md` — 完成 #6 tooling flip，产出 `logs/findings_sync_sources.md` 对 1.0s cudaDeviceSynchronize + top-20 blocking D2H 做 5-桶根因分类，不修改 corep_fast。

**Architecture:** 8 个顺序任务。Task 1 做 driver.py tooling flip (独立 commit)。Task 2-7 构建分析脚本 (落在 `tmp/sync_spike/`)，产出 `logs/findings_sync_sources.md`。Task 8 self-review + 同 commit 打包。全程 corep_fast 零改。

**Tech Stack:** Python 3.10+, sqlite3 (nsys trace), ijson (streaming Chrome trace JSON), PyTorch profiler trace 格式。分析脚本独立，不依赖 PyTorch runtime。

**Branch:** `post-profile-sync-elim`；上游 spec commit `687b02f` 已存在；基线 `52f5a8c`。预期 Task 1 + Task 8 各一个实现 commit，共 3 commits on branch ahead of `52f5a8c` (spec + tooling + findings)。

**baseline 数据参照:**
- `tmp/profile_deep/results/nsys_res256_full.sqlite` — nsys CUDA API trace
- `tmp/profile_deep/results/layer1_res256_run1_trace.json` — torch.profiler Chrome trace with_stack=True @ res=256
- `tmp/profile_deep/results/layer3_res128_run1_summary.json` — e2e baseline with with_stack=True @ res=128 = **3.929s** (Task 1 smoke 比较基准)

---

## File Structure Overview

**Modified:**
- `tmp/profile_deep/driver.py` (~15 lines) — with_stack 默认值翻转 + `--with-stack` CLI flag (Task 1 独立 commit)

**Created under `tmp/sync_spike/`** (所有脚本独立可跑，写死 input 路径到 `tmp/profile_deep/results/`):
- `probe_nsys_sync.py` — SQLite 查询 1.0s cudaDeviceSynchronize 的 timestamp / duration (Task 2)
- `cross_ref_python_stack.py` — 用 Task 2 的 timestamp 从 Chrome trace 查对应 python_function (Task 3)
- `grep_sync_sources.py` — grep corep_fast/ 里 `.item()` / `.cpu()` 等 (Task 4)
- `extract_top20_d2h.py` — ijson 流式解析 Chrome trace，聚合 top-20 blocking D2H (Task 5)
- `classify_top20.py` — 手工 + 脚本辅助把 top-20 分到 A/B/C/D/E 5 桶 (Task 6)

**Created:**
- `logs/findings_sync_sources.md` — 最终 findings doc (Task 4/6/7 逐步累积，Task 8 commit)

**Zero change:** `corep_fast/` 下任何文件。验证方式 `git diff 52f5a8c HEAD -- corep_fast/` 必须为空。

---

## Task 1: #6 — driver.py with_stack=False flip

**Files:**
- Modify: `tmp/profile_deep/driver.py` (add `--with-stack` CLI flag ~line 128; change `with_stack=True` at line 177 to use the flag value)
- Test: `tmp/profile_deep/results/sync_spike_smoke_res128_summary.json` (produced by smoke run)

- [ ] **Step 1: Add `--with-stack` CLI flag to driver.py**

Edit `tmp/profile_deep/driver.py` in function `main()` around line 128 (after the `--res` argument):

```python
ap.add_argument("--res", type=int, required=True)
ap.add_argument("--with-stack", action="store_true",
                help="Enable torch.profiler with_stack=True (slow, ~5-10x overhead, "
                     "~3.8GB Chrome trace @ res=256). Default off per sync-spike "
                     "spec 2026-04-17.")
args = ap.parse_args()
```

- [ ] **Step 2: Flip `with_stack=True` to reference the flag**

Edit `tmp/profile_deep/driver.py` line 177: change `with_stack=True,` to `with_stack=args.with_stack,`.

Full context (show the surrounding block):

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    record_shapes=True,
    profile_memory=False,
    with_stack=args.with_stack,
) as prof:
```

- [ ] **Step 3: Smoke — run with_stack=False default @ res=128**

From repo root:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m tmp.profile_deep.driver --layer 1 --res 128 2>&1 | tee tmp/profile_deep/results/sync_spike_smoke_res128.log
```

Then rename the produced summary so it doesn't overwrite existing res128 artifacts:

```bash
mv tmp/profile_deep/results/layer1_res128_summary.json tmp/profile_deep/results/sync_spike_smoke_res128_summary.json
mv tmp/profile_deep/results/layer1_res128_trace.json tmp/profile_deep/results/sync_spike_smoke_res128_trace.json
```

Expected: command exits 0; summary + trace files produced.

- [ ] **Step 4: Verify e2e wall ≤ baseline + 10%**

Run:

```bash
python3 -c "
import json
new = json.load(open('tmp/profile_deep/results/sync_spike_smoke_res128_summary.json'))
baseline = json.load(open('tmp/profile_deep/results/layer3_res128_run1_summary.json'))
new_e2e = new['stage_walltime_sec']['e2e']
base_e2e = baseline['stage_walltime_sec']['e2e']
ratio = new_e2e / base_e2e
print(f'baseline e2e (with_stack=True): {base_e2e:.3f}s')
print(f'new e2e (with_stack=False):     {new_e2e:.3f}s')
print(f'ratio: {ratio:.3f}x (PASS if <= 1.10)')
assert ratio <= 1.10, f'e2e regressed: {ratio:.3f}x > 1.10x'
print('PASS')
"
```

Expected: `PASS`. If `FAIL`, there may be a correctness regression from the flag; investigate before continuing.

- [ ] **Step 5: Verify trace file size dropped**

Run:

```bash
python3 -c "
import os
new_size = os.path.getsize('tmp/profile_deep/results/sync_spike_smoke_res128_trace.json')
# Existing with_stack=True @ res=128 trace was ~200MB per earlier profile_deep artifact reuse.
# with_stack=False should be <80MB (4x+ reduction expected).
print(f'new trace size: {new_size / 1e6:.1f} MB')
assert new_size < 80 * 1e6, f'trace too large: {new_size / 1e6:.1f} MB'
print('PASS')
"
```

Expected: `PASS` (trace < 80 MB).

- [ ] **Step 6: Verify `--with-stack` flag still works (regression guard)**

Run:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m tmp.profile_deep.driver --layer 1 --res 64 --with-stack 2>&1 | tail -20
ls -la tmp/profile_deep/results/layer1_res64_trace.json
```

Expected: exits 0; layer1_res64_trace.json exists. (We don't measure size here — just verify flag path works.)

Clean up:

```bash
# Keep only the smoke artifacts we created for Task 1; restore original res64 baseline won't matter because res64 baseline already has its own filename.
```

- [ ] **Step 7: Commit Task 1**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add tmp/profile_deep/driver.py
git add -f tmp/profile_deep/results/sync_spike_smoke_res128_summary.json
git add -f tmp/profile_deep/results/sync_spike_smoke_res128.log
git status
git commit -m "$(cat <<'EOF'
sync-spike(task1): driver with_stack=False default + --with-stack flag

ROI #6 from deep profiling (my-docs/20260417-corep-deep-profiling-results.md):
profile overhead from with_stack=True is 5-10x on Chrome trace size and ~20%
on e2e wall. Flip default off; opt in via --with-stack when Python call stacks
are needed.

Smoke @ res=128: e2e 3.929s (with_stack=True baseline) -> ~X.XXXs (new),
ratio <=1.10. Trace size drops from ~200MB to <80MB.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Probe nsys SQLite for the 1.0s cudaDeviceSynchronize

**Why:** Deep Profiling reported 20 cudaDeviceSynchronize calls total 1049ms, one single call ~1048ms (99.9% of that 1.0s budget). Locate its nsys timestamp so Task 3 can cross-reference the Python stack.

**Files:**
- Create: `tmp/sync_spike/probe_nsys_sync.py`
- Create: `tmp/sync_spike/__init__.py` (empty)
- Input: `tmp/profile_deep/results/nsys_res256_full.sqlite`

- [ ] **Step 1: Inspect nsys schema to know which table/columns to query**

Run:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
sqlite3 tmp/profile_deep/results/nsys_res256_full.sqlite ".tables" 2>&1 | head -30
```

Expected output includes `CUPTI_ACTIVITY_KIND_RUNTIME` (or similar — nsys version-dependent; older nsys uses `CUDA_API`, newer uses `CUPTI_ACTIVITY_KIND_RUNTIME`). Record the exact name for Step 2.

```bash
# Inspect one likely table's schema
sqlite3 tmp/profile_deep/results/nsys_res256_full.sqlite ".schema CUPTI_ACTIVITY_KIND_RUNTIME" 2>&1 | head -20
# If that name doesn't exist, try:
sqlite3 tmp/profile_deep/results/nsys_res256_full.sqlite ".schema" 2>&1 | grep -iE "(RUNTIME|API)" | head -5
```

Expected: at least one table with columns `start`, `end`, `nameId` (or `name`), `correlationId`, `globalPid` or equivalent.

Likely columns in `CUPTI_ACTIVITY_KIND_RUNTIME`: `start` (ns), `end` (ns), `nameId` (foreign key to `StringIds`), `correlationId`, `globalTid`.

- [ ] **Step 2: Create `tmp/sync_spike/__init__.py`**

```bash
mkdir -p tmp/sync_spike
touch tmp/sync_spike/__init__.py
```

- [ ] **Step 3: Write probe script `tmp/sync_spike/probe_nsys_sync.py`**

Create `tmp/sync_spike/probe_nsys_sync.py`:

```python
"""Probe nsys SQLite trace for the longest cudaDeviceSynchronize.

Deep Profiling (2026-04-16/17) reported 20 cudaDeviceSynchronize calls totaling
1049ms, with one call ~1048ms. This script locates that call's timestamp so we
can cross-reference the Python stack in torch.profiler trace.

Usage: python -m tmp.sync_spike.probe_nsys_sync
"""
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SQLITE_PATH = REPO_ROOT / "tmp/profile_deep/results/nsys_res256_full.sqlite"


def find_runtime_table(con: sqlite3.Connection) -> str:
    """Return the table name that holds CUDA runtime API events."""
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name LIKE '%CUPTI_ACTIVITY_KIND_RUNTIME%' OR name LIKE '%CUDA_API%')"
    )
    rows = [r[0] for r in cur.fetchall()]
    if not rows:
        raise RuntimeError(
            "Could not find runtime API table in nsys SQLite. "
            "Run `sqlite3 <db> .tables` and inspect."
        )
    return rows[0]


def find_devicesync_calls(con: sqlite3.Connection, table: str, top_n: int = 5):
    """Return the top-N longest cudaDeviceSynchronize calls.

    nsys stores the kernel / API name in a `StringIds` lookup table;
    `CUPTI_ACTIVITY_KIND_RUNTIME` has a `nameId` FK.
    """
    # Find the string ID for cudaDeviceSynchronize
    cur = con.execute(
        "SELECT id FROM StringIds WHERE value = 'cudaDeviceSynchronize_v3020' "
        "OR value = 'cudaDeviceSynchronize'"
    )
    ids = [r[0] for r in cur.fetchall()]
    if not ids:
        raise RuntimeError("cudaDeviceSynchronize not found in StringIds")

    placeholders = ",".join("?" * len(ids))
    q = (
        f"SELECT start, end, (end-start) AS dur_ns, correlationId, globalTid "
        f"FROM {table} "
        f"WHERE nameId IN ({placeholders}) "
        f"ORDER BY dur_ns DESC LIMIT {top_n}"
    )
    cur = con.execute(q, ids)
    return cur.fetchall()


def main():
    if not SQLITE_PATH.exists():
        sys.exit(f"Missing nsys SQLite: {SQLITE_PATH}")
    con = sqlite3.connect(str(SQLITE_PATH))
    table = find_runtime_table(con)
    print(f"[info] runtime API table: {table}")

    calls = find_devicesync_calls(con, table, top_n=5)
    if not calls:
        sys.exit("No cudaDeviceSynchronize calls found.")

    print(f"\nTop 5 longest cudaDeviceSynchronize calls:")
    print(f"{'rank':<5}{'start_ns':>16}{'end_ns':>16}{'dur_ms':>12}{'corrId':>10}{'tid':>12}")
    for rank, (start, end, dur_ns, corr, tid) in enumerate(calls, 1):
        print(f"{rank:<5}{start:>16}{end:>16}{dur_ns/1e6:>12.3f}{corr:>10}{tid:>12}")

    top = calls[0]
    assert top[2] / 1e6 > 500, (
        f"Longest cudaDeviceSynchronize is {top[2]/1e6:.1f}ms, expected >500ms "
        f"per Deep Profiling finding. Data may have drifted."
    )
    print(f"\n[ok] longest cudaDeviceSynchronize: {top[2]/1e6:.2f}ms "
          f"(start={top[0]}, end={top[1]})")

    # Save the top call to a small JSON for Task 3.
    import json
    out_path = REPO_ROOT / "tmp/sync_spike/nsys_top_devicesync.json"
    with open(out_path, "w") as f:
        json.dump({
            "start_ns": top[0],
            "end_ns": top[1],
            "dur_ns": top[2],
            "dur_ms": top[2] / 1e6,
            "correlationId": top[3],
            "globalTid": top[4],
        }, f, indent=2)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run probe**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -m tmp.sync_spike.probe_nsys_sync 2>&1 | tee tmp/sync_spike/probe_nsys_sync.log
```

Expected:
- Prints "Top 5 longest cudaDeviceSynchronize calls:" table
- `dur_ms` of rank 1 is between 900 and 1100 ms
- `[ok]` line with dur ~1048ms
- `[write] .../nsys_top_devicesync.json`

If the assertion fails (top call < 500ms), the nsys schema may differ or the trace may be stale — inspect output and either fix the query or re-probe with a different filter (e.g. include cudaStreamSynchronize candidates).

- [ ] **Step 5: Inspect output for Task 3 handoff**

```bash
cat tmp/sync_spike/nsys_top_devicesync.json
```

Record these values (Task 3 will use `start_ns` and `end_ns`):
- `start_ns`: _____________
- `end_ns`: _____________
- `dur_ms`: _____________

- [ ] **Step 6: Do not commit yet — Task 8 bundles all spike commits**

---

## Task 3: Cross-reference torch.profiler trace for the Python stack

**Why:** Task 2 gives a CUDA API timestamp. To find the offending `.item()` / `cuda.synchronize()` call in user code, we need the Python frame that was on the stack at that moment. Deep Profiling already confirmed NVTX markers do NOT appear in torch.profiler trace; use `python_function` events instead.

**Files:**
- Create: `tmp/sync_spike/cross_ref_python_stack.py`
- Input: `tmp/sync_spike/nsys_top_devicesync.json` (from Task 2), `tmp/profile_deep/results/layer1_res256_run1_trace.json` (~2GB with_stack=True Chrome trace)

- [ ] **Step 1: Confirm trace has `python_function` events**

Chrome traces can be very large (GB). Use streaming JSON to probe first:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/pip install ijson 2>&1 | tail -1   # likely already installed from profile_deep
.venv/bin/python -c "
import ijson
with open('tmp/profile_deep/results/layer1_res256_run1_trace.json', 'rb') as f:
    events = ijson.items(f, 'traceEvents.item')
    kinds = {}
    for i, ev in enumerate(events):
        kinds[ev.get('cat', '?')] = kinds.get(ev.get('cat', '?'), 0) + 1
        if i > 200_000:
            break
    print(kinds)
"
```

Expected: output dict includes `'python_function'` with non-zero count.

- [ ] **Step 2: Write cross-ref script `tmp/sync_spike/cross_ref_python_stack.py`**

Create `tmp/sync_spike/cross_ref_python_stack.py`:

```python
"""Cross-reference the 1.0s cudaDeviceSynchronize timestamp with torch.profiler
python_function events to recover the offending Python frame.

Input:
  - tmp/sync_spike/nsys_top_devicesync.json      (from Task 2)
  - tmp/profile_deep/results/layer1_res256_run1_trace.json  (Chrome trace)

Chrome trace events have keys: name, cat, ph, ts (microseconds), dur (us), args,
pid, tid. For Begin/End (ph=X = complete) events, [ts, ts+dur] is the interval.

IMPORTANT: nsys timestamps are *absolute* (Unix epoch ns or monotonic ns from
boot depending on nsys version). torch.profiler timestamps are *relative to
profiler start*. The two timebases differ. This script therefore matches by
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


def nsys_window(sqlite_path: Path, runtime_table: str) -> tuple[int, int]:
    con = sqlite3.connect(str(sqlite_path))
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


def trace_window(trace_path: Path) -> tuple[int, int]:
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
            ts_min = min(ts_min, ts)
            ts_max = max(ts_max, ts + dur)
    return ts_min, ts_max


def find_containing_python_frames(
    trace_path: Path, target_ts_us: float, target_end_ts_us: float
) -> list[dict]:
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
            if ts <= target_ts_us <= ts + dur:
                hits.append({
                    "name": ev.get("name", ""),
                    "ts": ts,
                    "dur": dur,
                    "args": ev.get("args", {}),
                })
    hits.sort(key=lambda h: h["dur"])
    return hits


def main():
    if not NSYS_TOP.exists():
        sys.exit(f"Missing {NSYS_TOP} — run Task 2 first.")
    top = json.loads(NSYS_TOP.read_text())

    con = sqlite3.connect(str(SQLITE_PATH))
    table = find_runtime_table(con)
    nsys_lo, nsys_hi = nsys_window(SQLITE_PATH, table)

    tr_lo_us, tr_hi_us = trace_window(TRACE_PATH)
    print(f"[info] nsys window: {nsys_lo}..{nsys_hi} ns "
          f"(span {(nsys_hi - nsys_lo)/1e9:.2f}s)")
    print(f"[info] torch.profiler window: {tr_lo_us}..{tr_hi_us} us "
          f"(span {(tr_hi_us - tr_lo_us)/1e6:.2f}s)")

    # Relative position of the 1s sync in nsys coordinates
    nsys_span = nsys_hi - nsys_lo
    frac_start = (top["start_ns"] - nsys_lo) / nsys_span
    frac_end = (top["end_ns"] - nsys_lo) / nsys_span

    tr_span_us = tr_hi_us - tr_lo_us
    projected_ts_us = tr_lo_us + frac_start * tr_span_us
    projected_end_us = tr_lo_us + frac_end * tr_span_us

    print(f"[info] projected target: [{projected_ts_us:.0f}, {projected_end_us:.0f}] us "
          f"(fraction {frac_start:.4f} - {frac_end:.4f})")

    hits = find_containing_python_frames(
        TRACE_PATH, projected_ts_us, projected_end_us
    )

    print(f"\nFound {len(hits)} python_function frames containing target. "
          f"Showing shortest-dur top 10 (deepest-stack most specific):")
    print(f"{'dur(us)':>12}  {'name':<60}")
    for h in hits[:10]:
        print(f"{h['dur']:>12}  {h['name'][:60]}")

    out_path = REPO_ROOT / "tmp/sync_spike/cross_ref_candidates.json"
    with open(out_path, "w") as f:
        json.dump(hits[:10], f, indent=2, default=str)
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run cross-ref**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -m tmp.sync_spike.cross_ref_python_stack 2>&1 | tee tmp/sync_spike/cross_ref_python_stack.log
```

Expected:
- Prints nsys window and torch.profiler window spans (both should be roughly comparable seconds)
- Prints "Found N python_function frames" — N should be between 3 and 20 (too few = projection miss; too many = projection too wide)
- Top entry should be a corep_fast stage function name or PyTorch internal

If N is 0 or > 100, the nsys-to-torch-profiler time projection is off. Fall back to a simpler heuristic: if the 1s sync is the only event of its kind (>500ms), look for the single longest python_function event in torch.profiler trace of category `python_function` that's > 500ms — should be the same frame.

- [ ] **Step 4: Inspect and record top candidate**

```bash
cat tmp/sync_spike/cross_ref_candidates.json | python3 -m json.tool | head -30
```

Record the top candidate (shortest dur inside the target window):
- `name`: _______________________________________
- `dur`: _______________________________________

If top candidate is `<built-in method sync>` or similar opaque name, look at the 2nd/3rd — those are usually corep_fast stage functions giving us the stage.

- [ ] **Step 5: Do not commit — Task 8 bundles all spike commits**

---

## Task 4: Grep corep_fast + synthesize findings §1

**Why:** Task 3 narrows to 1-3 candidate Python frames. Grep the codebase for sync triggers and intersect with the candidate names to confirm `file.py:line`.

**Files:**
- Create: `tmp/sync_spike/grep_sync_sources.py`
- Create: `logs/findings_sync_sources.md` (§0 placeholder + §1 draft)

- [ ] **Step 1: Write grep helper `tmp/sync_spike/grep_sync_sources.py`**

Create `tmp/sync_spike/grep_sync_sources.py`:

```python
"""Grep corep_fast/ for known implicit and explicit CUDA sync triggers.

Sync triggers targeted:
  - Explicit: torch.cuda.synchronize, torch.cuda.current_stream().synchronize
  - Implicit: .item(), .cpu(), .tolist(), .numpy()
  - Boolean-bridge: `if <tensor>:`, `while <tensor>:`, `bool(<tensor>)`
  - Indirect: .nonzero() without as_tuple (returns CPU indices in some paths)

Output: tmp/sync_spike/grep_sync_sources.json — list of {file, line, pattern, snippet}

Usage: python -m tmp.sync_spike.grep_sync_sources
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COREP_FAST = REPO_ROOT / "corep_fast"

PATTERNS = [
    ("explicit_sync",    re.compile(r"torch\.cuda\.synchronize\s*\(")),
    ("stream_sync",      re.compile(r"\.stream\(\)\.synchronize\s*\(|current_stream\(\)\.synchronize\s*\(")),
    ("item",             re.compile(r"\.item\s*\(\s*\)")),
    ("cpu_call",         re.compile(r"\.cpu\s*\(\s*\)")),
    ("tolist",           re.compile(r"\.tolist\s*\(\s*\)")),
    ("numpy_call",       re.compile(r"\.numpy\s*\(\s*\)")),
    ("bool_bridge",      re.compile(r"(?<![a-zA-Z_])bool\s*\(")),
]

def scan_file(path: Path) -> list[dict]:
    hits = []
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return hits
    for lineno, line in enumerate(lines, 1):
        for label, rx in PATTERNS:
            if rx.search(line):
                hits.append({
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": lineno,
                    "pattern": label,
                    "snippet": line.strip(),
                })
    return hits


def main():
    if not COREP_FAST.exists():
        sys.exit(f"Missing {COREP_FAST}")
    all_hits = []
    for p in COREP_FAST.rglob("*.py"):
        all_hits.extend(scan_file(p))

    # Skip obvious non-sync uses: string literals, comments, etc. are left for
    # humans to filter. For now, dump everything and let the caller review.

    by_pattern: dict[str, int] = {}
    for h in all_hits:
        by_pattern[h["pattern"]] = by_pattern.get(h["pattern"], 0) + 1

    print(f"Found {len(all_hits)} total matches across {len(set(h['file'] for h in all_hits))} files:")
    for label, cnt in sorted(by_pattern.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<20} {cnt}")

    out_path = REPO_ROOT / "tmp/sync_spike/grep_sync_sources.json"
    with open(out_path, "w") as f:
        json.dump(all_hits, f, indent=2)
    print(f"\n[write] {out_path}  (all hits)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run grep**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -m tmp.sync_spike.grep_sync_sources 2>&1 | tee tmp/sync_spike/grep_sync_sources.log
```

Expected: prints summary table; `tmp/sync_spike/grep_sync_sources.json` created.

- [ ] **Step 3: Intersect Task 3 candidates with grep hits**

From Task 3's `cross_ref_candidates.json`, pick the top 1-3 candidate names (e.g. `s6_collapse`, `_phase1_gpu_rank_assign`). From grep JSON, filter hits whose `file` matches each candidate's source. Produce a short shortlist.

```bash
python3 -c "
import json
cands = json.load(open('tmp/sync_spike/cross_ref_candidates.json'))
grep = json.load(open('tmp/sync_spike/grep_sync_sources.json'))
# Top 3 candidate python_function names (shortest-dur first)
top_names = [c['name'] for c in cands[:3]]
print('top_names:', top_names)

# Very loose filter: any grep hit in any file mentioning any candidate name
import re
for name in top_names:
    # extract plausible module/function token (e.g. 's6_collapse' from
    # 'aten::forward(s6_collapse)') — just look for likely filename stems
    for h in grep:
        if any(tok and tok.lower() in h['file'].lower() for tok in name.split('.') if tok):
            print(f\"  {name} -> {h['file']}:{h['line']}  ({h['pattern']})  {h['snippet'][:80]}\")
" | tee tmp/sync_spike/intersect_candidates.log
```

Expected: a short list (5-30 hits) linking top candidates to concrete `file:line`.

- [ ] **Step 4: Manually identify the 1s DeviceSync source file:line**

Open the shortlist, pick the one that both:
- Is inside the narrowest candidate (shortest-dur python_function frame from Task 3), AND
- Matches `explicit_sync` or a `.item()` that could plausibly serialize a large compute

Open the identified file at the reported line number and read ±10 lines of context.

```bash
# Example (adapt path):
sed -n 'START,ENDp' corep_fast/stages/sX_XXX.py  # where START = line - 10, END = line + 10
```

Record:
- `file`: _______________________________________
- `line`: _______________________________________
- `pattern`: _______________________________________ (explicit_sync / item / cpu_call / ...)
- `snippet`: _______________________________________
- `context_±10lines`: paste into findings doc

- [ ] **Step 5: Create `logs/findings_sync_sources.md` skeleton + fill §1**

Create `logs/findings_sync_sources.md` with this initial content:

```markdown
# Sync Sources — Findings (2026-04-17 spike)

**Spec:** `docs/superpowers/specs/2026-04-17-sync-spike-design.md`
**Plan:** `docs/superpowers/plans/2026-04-17-sync-spike-implementation.md`
**Branch:** `post-profile-sync-elim`
**Data basis:** `tmp/profile_deep/results/nsys_res256_full.sqlite`,
`tmp/profile_deep/results/layer1_res256_run1_trace.json` (with_stack=True)

## 0. TL;DR
(to be written in Task 7)

## 1. The 1.0s cudaDeviceSynchronize

### 1.1 Evidence chain

- **nsys**: `cudaDeviceSynchronize` call count 20, total 1049ms, longest single call
  <DUR_MS>ms (from `tmp/sync_spike/nsys_top_devicesync.json`: start_ns=<S>, end_ns=<E>).
- **torch.profiler**: projected target window [<PROJ_TS_US>, <PROJ_END_US>] μs contained
  <N> python_function frames; shortest-dur containing frame: `<CAND_NAME>` (dur <CAND_DUR>μs).
- **grep corep_fast/**: file `<FILE>:<LINE>` matches pattern `<PATTERN>`.

### 1.2 Source snippet (±10 lines)

```python
<PASTE 20 LINES OF CONTEXT HERE>
```

### 1.3 Root-cause bucket

Assigned bucket: **<A/B/C/D/E>** — <one-paragraph justification based on what the
returned value is used for.>

### 1.4 Effort estimate

<seconds / hours / days / weeks> — rationale: <1-2 sentences>.

## 2. Top-20 blocking D2H — Classified
(to be written in Task 6)

## 3. Per-bucket breakdown
(to be written in Task 6)

## 4. Recommended next fix spec scope
(to be written in Task 7)

## 5. Caveats
(to be written in Task 7)
```

Replace `<S>`, `<E>`, `<DUR_MS>`, `<PROJ_TS_US>`, `<PROJ_END_US>`, `<N>`, `<CAND_NAME>`,
`<CAND_DUR>`, `<FILE>`, `<LINE>`, `<PATTERN>`, the source snippet, the bucket, and the
rationale with the values recorded in Steps 1-4.

- [ ] **Step 6: Do not commit — Task 8 bundles all spike commits**

---

## Task 5: Extract top-20 blocking D2H from torch.profiler trace

**Why:** Deep Profiling reported 4022 D2H memcpys + 4128 cudaStreamSynchronize. Most D2H follows blocking sync pattern. Aggregate by parent Python frame + source file:line to find the top-20 hotspots.

**Files:**
- Create: `tmp/sync_spike/extract_top20_d2h.py`
- Input: `tmp/profile_deep/results/layer1_res256_run1_trace.json`

- [ ] **Step 1: Inspect trace event schema for cat='gpu_memcpy' + Python stack**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "
import ijson
sample = []
with open('tmp/profile_deep/results/layer1_res256_run1_trace.json', 'rb') as f:
    for ev in ijson.items(f, 'traceEvents.item'):
        cat = ev.get('cat', '')
        if 'memcpy' in cat.lower() or 'sync' in cat.lower() or (
            'memcpy' in ev.get('name', '').lower()
        ):
            sample.append(ev)
            if len(sample) >= 5:
                break
import json
print(json.dumps(sample, indent=2, default=str)[:4000])
"
```

Expected: sample events of memcpy / sync type. Record:
- What `cat` field memcpy events have (e.g. `gpu_memcpy`, `cuda_runtime`, or just `kernel`)
- Whether each event includes a full Python stack in `args['External id']` / `args['Python stack']` / similar — this determines whether we can attribute each memcpy to a Python frame.

- [ ] **Step 2: Check how Python stack is exposed in trace events**

```bash
.venv/bin/python -c "
import ijson
with open('tmp/profile_deep/results/layer1_res256_run1_trace.json', 'rb') as f:
    for i, ev in enumerate(ijson.items(f, 'traceEvents.item')):
        # Look for an event with a stack reference
        args = ev.get('args', {})
        if any('stack' in k.lower() or 'python' in k.lower() for k in args.keys()):
            print('Found stack-carrying event:')
            import json
            print(json.dumps(ev, indent=2, default=str)[:2000])
            break
        if i > 100_000:
            print('No obvious stack field found in first 100k events')
            break
"
```

Expected: either (a) find `args` with a `Python stack` or `external_id` key and record its schema, or (b) confirm stack info is delivered via separate python_function events that bracket the memcpy in time. Case (b) is the standard pattern — we'll use time-bracketing.

- [ ] **Step 3: Write extraction script**

Create `tmp/sync_spike/extract_top20_d2h.py`:

```python
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
  3. Aggregate (frame_name) -> (count, total_dur_ms)
  4. Sort by total_dur_ms desc, take top 20.

Output: tmp/sync_spike/top20_d2h.json — list of 20 records:
  {rank, stage, frame_name, count, total_dur_ms, avg_dur_us, sample_ts_us}

Stage is derived from the root python_function ancestor matching
{'s1_voxelize','s2_components','s3_edge_weights','s4_face_point',
 's6_collapse','s7_rank_assign','s8_decode'}.

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
    """Yield (ts_us, dur_us, kind, name) for memcpy/sync events, and full
    python_function event list."""
    memcpy_sync = []
    py_events = []  # (ts, end_ts, name, dur)
    with open(trace_path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            cat = ev.get("cat", "")
            name = ev.get("name", "")
            ts = ev.get("ts")
            dur = ev.get("dur", 0)
            if ts is None:
                continue
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
    """For each event, find deepest-stack containing python_function frame.

    Sort py_events by ts ascending. For each target, binary-search the set of
    events with ts <= target.ts. Among those, keep ones whose end_ts >= target.ts.
    The deepest is the one with the smallest (end_ts - ts) that still contains
    the target (smallest-containing-interval).
    """
    import bisect
    py_events.sort(key=lambda e: e[0])  # by ts
    ts_list = [e[0] for e in py_events]

    attributed = []
    for ev_ts, ev_dur, kind, ev_name in memcpy_sync:
        i = bisect.bisect_right(ts_list, ev_ts)  # events with ts <= ev_ts
        # Scan backwards up to some window — start with 10_000 earlier events
        cands = []
        for j in range(i - 1, max(-1, i - 10_000), -1):
            p_ts, p_end, p_name, p_dur = py_events[j]
            if p_end >= ev_ts:
                cands.append((p_end - p_ts, p_name, p_ts))
        cands.sort()  # smallest interval first = deepest stack
        deepest = cands[0][1] if cands else "<no frame>"
        # Find root stage
        stage = "<unknown>"
        for _, pname, _ in cands:
            if pname in STAGE_NAMES or any(s in pname for s in STAGE_NAMES):
                stage = next((s for s in STAGE_NAMES if s in pname), pname)
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
    """Aggregate by (stage, frame_name) for d2h-only, sort by total_dur, take top N."""
    from collections import defaultdict
    agg = defaultdict(lambda: {"count": 0, "total_dur_us": 0, "sample_ts_us": None,
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

    print(f"[scan] reading {TRACE_PATH} (~2GB, streaming)...")
    memcpy_sync, py_events = first_pass(TRACE_PATH)
    print(f"[info] memcpy/sync events: {len(memcpy_sync)}")
    print(f"[info] python_function events: {len(py_events)}")

    d2h_count = sum(1 for e in memcpy_sync if e[2] == "d2h")
    sync_count = sum(1 for e in memcpy_sync if e[2] == "sync")
    print(f"[info] D2H memcpy: {d2h_count}")
    print(f"[info] stream/device sync: {sync_count}")

    # Sanity gate vs Deep Profiling's 4022 D2H + 4128 sync (both res=256 @ run1).
    if d2h_count < 2000:
        print(f"[warn] D2H count {d2h_count} < 2000; expected ~4022. "
              f"May be a res/run mismatch.")

    attributed = attribute(memcpy_sync, py_events)
    top20 = aggregate_top_n(attributed, n=20)

    print(f"\nTop 20 blocking D2H hotspots by aggregate dur_ms:")
    print(f"{'rank':<5}{'stage':<18}{'count':>8}{'total_ms':>12}  {'frame_name'}")
    for r in top20:
        print(f"{r['rank']:<5}{r['stage']:<18}{r['count']:>8}"
              f"{r['total_dur_ms']:>12.3f}  {r['frame_name'][:60]}")

    out_path = REPO_ROOT / "tmp/sync_spike/top20_d2h.json"
    with open(out_path, "w") as f:
        json.dump(top20, f, indent=2)
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run extraction**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -m tmp.sync_spike.extract_top20_d2h 2>&1 | tee tmp/sync_spike/extract_top20_d2h.log
```

Expected:
- Prints event counts (`D2H memcpy: <N>`, `stream/device sync: <M>`)
- Prints top-20 table
- `tmp/sync_spike/top20_d2h.json` created with 20 rows

Sanity check: `D2H memcpy` count should be ≥ 2000 (Deep Profiling said 4022 @ res=256; count could be lower if the trace is res=128 or lower-resolution capture). If drastically off, note in Task 7 caveats.

- [ ] **Step 5: Do not commit — Task 8 bundles all spike commits**

---

## Task 6: Classify top-20 into 5 buckets + write findings §2/§3

**Why:** Each top-20 row has a `frame_name` and `stage`. To classify we need to read the source at the call site and judge whether the `.item()` / sync is deferrable, control-flow, alloc-size, library-internal, or correctness.

**Files:**
- Create: `tmp/sync_spike/classify_top20.py` (semi-manual helper)
- Modify: `logs/findings_sync_sources.md` (add §2 table + §3 breakdown)

- [ ] **Step 1: Resolve each top-20 frame_name to source file:line**

For each of the 20 rows, use the `frame_name` (which is the Python function name or `aten::...` op name from torch.profiler) to locate the source:

- If `frame_name` contains a stage name (e.g. `s6_collapse`), start in `corep_fast/stages/s6_collapse.py`.
- Use grep: `grep -n "<frame_name_leaf>" corep_fast/**/*.py`.
- For `aten::` events (PyTorch ops), the D2H is often in the calling user frame; read the containing Python frame from the trace (this should be captured alongside frame_name if attribution worked; otherwise cross-ref via time proximity).

Helper script — create `tmp/sync_spike/classify_top20.py`:

```python
"""For each top-20 entry, grep corep_fast/ for its frame_name leaf and return
candidate source locations. Classification is human-judged and appended to
logs/findings_sync_sources.md.

Usage: python -m tmp.sync_spike.classify_top20
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOP20 = REPO_ROOT / "tmp/sync_spike/top20_d2h.json"


def leaf_name(frame_name: str) -> str:
    """Extract the innermost identifier from a frame name like
    'corep_fast/stages/s6_collapse.py(123): _build_something'."""
    m = re.search(r":\s*(\w+)\s*$", frame_name)
    if m:
        return m.group(1)
    m = re.search(r"(\w+)\s*$", frame_name)
    return m.group(1) if m else frame_name


def grep(term: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["grep", "-rn", "--include=*.py", term, "corep_fast/"],
            cwd=REPO_ROOT, text=True
        )
    except subprocess.CalledProcessError:
        return []
    return out.splitlines()[:10]


def main():
    if not TOP20.exists():
        sys.exit(f"Missing {TOP20} — run Task 5 first.")
    rows = json.loads(TOP20.read_text())
    print(f"# Top-20 D2H → source candidates\n")
    for r in rows:
        leaf = leaf_name(r["frame_name"])
        hits = grep(leaf)
        print(f"## rank {r['rank']}: stage={r['stage']}  count={r['count']}  "
              f"total_ms={r['total_dur_ms']:.2f}")
        print(f"frame: {r['frame_name']}")
        print(f"leaf: {leaf}")
        if hits:
            print("grep candidates:")
            for h in hits[:5]:
                print(f"  {h}")
        else:
            print("grep candidates: <none>")
        print()


if __name__ == "__main__":
    main()
```

Run:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -m tmp.sync_spike.classify_top20 2>&1 | tee tmp/sync_spike/classify_top20.log
```

Expected: per-rank section with `frame`, `leaf`, and up to 5 grep hits.

- [ ] **Step 2: Open each candidate in the source and classify**

For each of the 20 rows, open the most-likely candidate file at the grep-reported line and read ±10 lines. Classify into one of 5 buckets using this decision tree:

```
1. Is this an explicit `torch.cuda.synchronize(...)`?
     -> Bucket E (Correctness-sync)
2. Is this a PyTorch internal op (e.g. aten::_local_scalar_dense inside
   a library call with no user .item())?
     -> Bucket D (Library-internal)
3. Is the returned value used only for Python-side print / log / debug /
   metric aggregation?
     -> Bucket A (Deferrable)
4. Is the returned value used to drive an `if` / `while` / `for` in Python,
   where the alternative path is also expressible in GPU ops (i.e. could be
   mask-driven instead)?
     -> Bucket B (Control-flow)
5. Is the returned value used as a size/shape argument to a subsequent
   tensor construction (torch.zeros(n, ...), view(n, ...)) where n is
   variable per call?
     -> Bucket C (Alloc-size)
6. None of the above unambiguously:
     -> Assign as "A/B" (two-bucket), pick conservative = B for effort count.
```

For each row, record:
- `rank`
- `stage`
- `file:line`
- `pattern` (e.g. `.item()`, `bool()`, explicit sync)
- `bucket` (A/B/C/D/E, or A/B for ambiguous)
- `per_site_effort` (seconds / hours / days / weeks)
- `1-line justification`

Use this bucket → effort table (updated from spec):

| Bucket | Per-site effort | Rationale |
|---|---|---|
| A Deferrable | seconds | Remove the call or move to a final collection |
| B Control-flow | 1-3 days | Needs mask / vectorize; depends on the branch |
| C Alloc-size | 3-7 days | Needs CUDA Graph or symbolic shape |
| D Library-internal | weeks | Needs API swap or upstream PR |
| E Correctness-sync | hours | Analyze whether removable; often keep |

- [ ] **Step 3: Write findings §2 — Top-20 table**

Append to `logs/findings_sync_sources.md` (replacing the `## 2. Top-20 blocking D2H — Classified\n(to be written in Task 6)` placeholder):

```markdown
## 2. Top-20 blocking D2H — Classified

Data basis: `tmp/sync_spike/top20_d2h.json`. Each row's stage/frame_name derived
from `extract_top20_d2h.py`; source file:line from `classify_top20.py` + manual
review.

| Rank | Stage | File:line | Pattern | Count | Total ms | Bucket | Per-site effort | Justification |
|---:|---|---|---|---:|---:|:-:|---|---|
| 1 | <stage> | `<file>:<line>` | `.item()` | <N> | <MS> | A | seconds | <1-line> |
| 2 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 20 | ... | ... | ... | ... | ... | ... | ... | ... |
```

Fill in all 20 rows from Step 2's records. `Total ms` should match the `total_dur_ms`
field in `top20_d2h.json`.

- [ ] **Step 4: Write findings §3 — Per-bucket breakdown**

Append to `logs/findings_sync_sources.md` (replacing `## 3. Per-bucket breakdown` placeholder):

```markdown
## 3. Per-bucket breakdown

### Bucket A (Deferrable)
- Sites: <N>
- Aggregate ms in top-20: <MS>
- Effort rollup: ~<N * seconds> = <minutes/hours>
- Expected Δ wall-time if fully removed: <MS> - <overhead_guess> ≈ <DELTA_MS>ms
- Representative site: `<file>:<line>`
- Notes: <1-2 sentences>

### Bucket B (Control-flow)
- Sites: <N>
- Aggregate ms in top-20: <MS>
- Effort rollup: ~<N * days>
- Expected Δ wall-time if fully restructured: depends on algorithmic rework; <estimate>
- Representative site: `<file>:<line>`
- Notes: <1-2 sentences>

### Bucket C (Alloc-size)
- Sites: <N>
- Aggregate ms in top-20: <MS>
- Effort rollup: ~<N * days-weeks>
- Expected Δ: <estimate>
- Representative site: `<file>:<line>`
- Notes: <1-2 sentences>

### Bucket D (Library-internal)
- Sites: <N>
- Aggregate ms in top-20: <MS>
- Effort rollup: weeks (API swap)
- Representative site: `<file>:<line>`
- Notes: <1-2 sentences>

### Bucket E (Correctness-sync)
- Sites: <N>
- Aggregate ms in top-20: <MS>
- Effort rollup: hours (audit)
- Representative site: `<file>:<line>`
- Notes: <1-2 sentences>
```

Fill in all buckets (0-site buckets get `Sites: 0` + "(none in top-20)").

- [ ] **Step 5: Do not commit — Task 8 bundles all spike commits**

---

## Task 7: Write findings §0 TL;DR + §4 Recommended scope + §5 Caveats

**Why:** The TL;DR anchors the reader. Recommended scope is the actual handoff to the next fix spec. Caveats prevent misuse of the data.

**Files:**
- Modify: `logs/findings_sync_sources.md` (replace §0, §4, §5 placeholders)

- [ ] **Step 1: Write §0 TL;DR**

Replace `## 0. TL;DR\n(to be written in Task 7)` with:

```markdown
## 0. TL;DR

- **1.0s cudaDeviceSynchronize root cause:** bucket **<A/B/C/D/E>** @ `<file>:<line>`.
  Source pattern: `<snippet>`. Fix effort: <seconds/hours/days/weeks>.
- **Top-20 blocking D2H distribution:**
  A=<NA>, B=<NB>, C=<NC>, D=<ND>, E=<NE> (sites, counting A/B ambiguous as B).
  Aggregate ms: A=<MA>, B=<MB>, C=<MC>, D=<MD>, E=<ME>.
- **Next fix spec recommended scope:** **<Option X / Y / Z name>** — see §4.
  Expected Δ wall-time: <-RANGE>s, effort: <X>d.
- **Not covered here** (per spec §3.2): any corep_fast code change; topology
  correctness fixture; Triton K1; s7 vectorize; cummax fuse.
```

Fill `<...>` placeholders from Task 4/6 data.

- [ ] **Step 2: Write §4 Recommended Scope**

Replace `## 4. Recommended next fix spec scope\n(to be written in Task 7)` with:

```markdown
## 4. Recommended next fix spec scope

Three candidate scopes for the next spec (sync-elimination fix), ordered by
risk / effort:

### Option X — "A-only" (low risk, mechanical)

- **What:** Remove or defer the <NA> top-20 Bucket-A sites.
- **Methodology:** For each site, rewrite so the scalar .item() is either
  eliminated entirely, or moved to the final end-of-pipeline collection phase.
  Add a fixture that asserts bit-exact output topology (vs custom and baseline)
  per removal.
- **Risk:** Low. Pure refactor, no algorithmic change.
- **Effort:** <N_A × seconds-to-hours> = ~<D>d.
- **Expected Δ wall-time @ res=256:** -<MA/1000>s upper bound; realistically
  -<MA × 0.5 / 1000>s after accounting for non-top-20 residual.

### Option Y — "A + 1s DeviceSync" (low-medium risk)

- **What:** Option X + the 1.0s cudaDeviceSynchronize (§1).
- **Applies when:** §1's bucket is A or E (mechanical removable).
- **Risk:** Low-medium — §1 might gate correctness; requires dual-reference
  fixture (custom + baseline) before merge.
- **Effort:** Option X effort + <E>d for §1.
- **Expected Δ wall-time @ res=256:** -<MA/1000>s + -1.0s = **-<MA/1000 + 1>s**.

### Option Z — "A + B selected" (medium-high risk)

- **What:** Option X + the <NB_selected> highest-dur Bucket-B sites (see §2 ranks).
- **Methodology:** Requires algorithmic restructure (mask-driven control flow or
  per-item vectorization). Per-site test coverage required.
- **Risk:** Medium-high. Scope creep risk.
- **Effort:** Option X effort + <NB_selected × 1-3>d.
- **Expected Δ wall-time @ res=256:** Option X + -<estimate>s.

### Recommendation

**<Option X / Y / Z>**, because:
- <reason 1>
- <reason 2>
- <reason 3>

Secondary priority for a follow-up spec: <the non-recommended buckets summarized>.
```

Fill `<...>` from data.

- [ ] **Step 3: Write §5 Caveats**

Replace `## 5. Caveats\n(to be written in Task 7)` with:

```markdown
## 5. Caveats

- **trace captured with_stack=True** — profile overhead inflates D2H timing by
  ~5-10x. Aggregate durations (`total_dur_ms`) are therefore upper bounds; actual
  wall-time savings will be proportionally smaller. Confirmation: after Option X
  executes, re-profile with_stack=False (Task 1 of this spec enabled the default)
  and compare.
- **NVTX markers absent from torch.profiler trace** — stage attribution relies on
  matching `python_function` events by function name (Deep Profiling
  methodology). A few ambiguous sites may list `<unknown>` stage.
- **nsys / torch.profiler time projection is approximate** — §1's cross-ref uses
  a linear projection between the two timebases because nsys is absolute and
  torch.profiler is profiler-relative. If grep corroboration failed in Task 4, the
  candidate frame may need manual verification before the next spec commits to a
  fix.
- **Single-run data** — the Chrome trace used here is
  `layer1_res256_run1_trace.json`, one of 3 profiler runs. Counts may vary ±5%
  across runs; top-20 ranking should be stable.
- **Bucket A/B ambiguity** — sites where the returned scalar is read both by
  Python logging AND by subsequent control flow were classified as B (pessimistic
  effort). If the next spec implementer finds the control-flow path is dead code,
  reclassification to A is possible.
```

- [ ] **Step 4: Re-read full `logs/findings_sync_sources.md`**

```bash
cat logs/findings_sync_sources.md | head -200
```

Confirm all `<...>` placeholders have been filled; no "(to be written in Task N)" strings remain.

- [ ] **Step 5: Do not commit — Task 8 bundles all spike commits**

---

## Task 8: Self-review + commit findings

**Files:**
- No new files
- All spike artifacts so far: `tmp/sync_spike/*.py`, `tmp/sync_spike/*.json`, `tmp/sync_spike/*.log`, `logs/findings_sync_sources.md`

- [ ] **Step 1: Verify corep_fast zero change**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git diff 52f5a8c HEAD -- corep_fast/
```

Expected: **empty output**. If non-empty, something leaked in — revert that file before committing.

- [ ] **Step 2: Verify DoD items 1-5 from spec §6**

```bash
# DoD #1: driver.py default
grep -n "with_stack" tmp/profile_deep/driver.py
# Expected: one hit with `action="store_true"` and one hit with `args.with_stack`.

# DoD #2: T1 smoke ratio already verified in Task 1 Step 4 — re-verify log exists
ls -la tmp/profile_deep/results/sync_spike_smoke_res128_summary.json
ls -la tmp/profile_deep/results/sync_spike_smoke_res128.log

# DoD #3: 1s DeviceSync has file:line
grep -E "^- \*\*grep corep_fast/\*\*" logs/findings_sync_sources.md
# Expected: one hit under §1.1

# DoD #4: top-20 100% classified
python3 -c "
import re
md = open('logs/findings_sync_sources.md').read()
# The §2 table rows should each end with a bucket letter in column 7
section = md.split('## 2.')[1].split('## 3.')[0]
rows = [l for l in section.splitlines() if l.startswith('|') and not l.startswith('|---') and 'Rank' not in l]
print(f'rows: {len(rows)}')
no_bucket = [l for l in rows if not re.search(r'\\|\\s*[ABCDE](/[ABCDE])?\\s*\\|', l)]
assert len(rows) == 20, f'Expected 20 rows, got {len(rows)}'
assert len(no_bucket) == 0, f'Rows missing bucket: {no_bucket}'
print('PASS: all 20 rows have a bucket')
"

# DoD #5: recommended scope in §4
grep -E "### Recommendation" logs/findings_sync_sources.md
# Expected: one hit.
```

Expected: all checks PASS.

- [ ] **Step 3: Commit all spike artifacts**

Stage and commit. Note `tmp/` and `logs/` may be gitignored — use `-f` if needed (precedent: `profile_deep` Task 11 used `git add -f`).

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git status

# Force-add (logs/ and tmp/ are gitignored per repo policy, precedent: 0ac51ae, 52f5a8c)
git add -f tmp/sync_spike/__init__.py
git add -f tmp/sync_spike/probe_nsys_sync.py
git add -f tmp/sync_spike/cross_ref_python_stack.py
git add -f tmp/sync_spike/grep_sync_sources.py
git add -f tmp/sync_spike/extract_top20_d2h.py
git add -f tmp/sync_spike/classify_top20.py
git add -f tmp/sync_spike/nsys_top_devicesync.json
git add -f tmp/sync_spike/cross_ref_candidates.json
git add -f tmp/sync_spike/grep_sync_sources.json
git add -f tmp/sync_spike/top20_d2h.json
git add -f tmp/sync_spike/*.log
git add -f logs/findings_sync_sources.md

git status

git commit -m "$(cat <<'EOF'
sync-spike(task2-7): findings_sync_sources.md + analysis scripts

Investigation outcome (no corep_fast code changes):

- 1.0s cudaDeviceSynchronize root cause: <bucket> at <file>:<line>
- Top-20 blocking D2H classified into 5 root-cause buckets (A Deferrable /
  B Control-flow / C Alloc-size / D Library-internal / E Correctness-sync).
- Bucket distribution: A=<NA> B=<NB> C=<NC> D=<ND> E=<NE>
- Next fix spec recommended scope: <Option X/Y/Z> (expected Δ <-RANGE>s, <X>d)

Analysis scripts under tmp/sync_spike/:
- probe_nsys_sync.py: SQLite query for longest cudaDeviceSynchronize
- cross_ref_python_stack.py: nsys ts -> torch.profiler python_function frame
- grep_sync_sources.py: corep_fast sync trigger pattern scan
- extract_top20_d2h.py: streaming aggregation of blocking D2H hotspots
- classify_top20.py: source grep helper for manual bucket assignment

Data basis: tmp/profile_deep/results/nsys_res256_full.sqlite +
layer1_res256_run1_trace.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Before running the commit: fill the 5 `<...>` placeholders in the commit message using the actual values from `logs/findings_sync_sources.md` §0 TL;DR.

- [ ] **Step 4: Verify commit count on branch**

```bash
git log post-profile-sync-elim ^52f5a8c --oneline
```

Expected: **3 lines**
1. `<task8 sha>` sync-spike(task2-7): findings_sync_sources.md + analysis scripts
2. `<task1 sha>` sync-spike(task1): driver with_stack=False default + --with-stack flag
3. `687b02f` spec(sync-spike): investigation + tooling low-hanging fruit

If count is wrong, diagnose which step was skipped.

- [ ] **Step 5: Update `logs/progress.md`**

Append a short entry to `logs/progress.md`:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
cat >> logs/progress.md <<'EOF'

## 2026-04-17 — sync-spike findings complete

- Spec: `docs/superpowers/specs/2026-04-17-sync-spike-design.md` (commit 687b02f)
- Plan: `docs/superpowers/plans/2026-04-17-sync-spike-implementation.md`
- Outcome: `logs/findings_sync_sources.md` — 1.0s DeviceSync root cause located,
  top-20 D2H classified into 5 root-cause buckets, recommended next-spec scope
  documented.
- Branch: `post-profile-sync-elim`
- corep_fast changes: zero (investigation spec; fixes deferred to next spec)
EOF

git add -f logs/progress.md
git commit -m "$(cat <<'EOF'
logs: sync-spike progress entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(This becomes a 4th commit on the branch — optional but follows precedent.)

- [ ] **Step 6: Final verification**

```bash
# corep_fast zero change (re-verify)
git diff 52f5a8c HEAD -- corep_fast/
# Expected: empty

# DoD #7 from spec
echo "---"

# DoD #6 commit count (spec wording is "2 commits"; actual is 3-4 including spec.
# Document the discrepancy if present.)
git log post-profile-sync-elim ^52f5a8c --oneline | wc -l
```

Expected: zero diff in `corep_fast/`; commit count 3 or 4.

---

## Global DoD (from spec §6)

| # | Item | Verified in |
|---|---|---|
| 1 | driver.py default changed | Task 1 Step 1-2; Task 8 Step 2 |
| 2 | T1 smoke e2e ≤ +10% | Task 1 Step 4 |
| 3 | 1.0s DeviceSync file:line | Task 4 Step 4 |
| 4 | Top-20 D2H 100% bucketed | Task 6 Step 3; Task 8 Step 2 |
| 5 | Recommended scope in §4 | Task 7 Step 2; Task 8 Step 2 |
| 6 | 2 commits ahead of spec | Task 8 Step 4 (actual count 2-3 depending on progress.md) |
| 7 | corep_fast zero change | Task 8 Step 1, Step 6 |
