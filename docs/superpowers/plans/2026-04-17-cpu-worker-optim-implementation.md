# CPU MP Worker Optimization — Implementation Plan (V2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute `docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md` (V2). Target e2e wall-time reduction of 3-5 s @ res=256 via main-thread CPU hotspot elimination (W4/W5/W6), persistent MP pool (W2), and small cleanups (W1/W7). No Triton, no new deps.

**Architecture:** 10 sequential tasks. T1 builds a regression gate that captures current pipeline output as a golden snapshot — every subsequent commit must match it within tolerance. T2 (W1) is a 2h warmup cleanup. T3 (W2) introduces a persistent MP pool replacing 6 per-dispatch `with _Pool(...)` sites. T4 measures W2's effect and decides T7's angle. T5 (W4) rewrites s6 fast-path tracer to run on GPU instead of main-thread Python over 275k cubes. T6 (W5) replaces main-thread Python union-find with batched GPU label propagation. T7 (W6) tackles the remaining s4 Stage D blocker. T8 (W7) does targeted s7 orchestration cleanup. T9 re-profiles. T10 writes the handoff document.

**Tech Stack:** PyTorch, multiprocessing, pytest. Python 3.10+. No new libraries.

**Branch:** `post-profile-sync-elim` (stay on current branch per user direction). Spec commit: `417e68b`.

**Execution environment:** All tests and profiling **must run on host-10-240-99-116 GPU 4** per user rule (`feedback_profiling_on_119.md` + session-specific redirect to 116 GPU 4). Local GPU forbidden for tests.

**SSH pattern:** Every SSH invocation MUST begin with `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 &&` or use absolute paths. Wrap SSH in `/tmp/*.sh` helper scripts to avoid quote mangling (pattern established in the sync-spike session).

**Baseline HEAD:** `417e68b` — all fixtures and Δ measurements are relative to this commit.

**T0 decisive experiment (2026-04-17, pre-T1):** MP default @ res=256 = 8.631 s median; serial (nw=1 via SerialPool monkeypatch) = 75.312 s median. **Serial is 8.7× slower.** MP is net-positive; plan direction retained. Pre-existing MP nondeterminism (vertex set drifts ~0.7% across default-MP runs) is fixed in T3 via `PYTHONHASHSEED=0` in the persistent pool initializer. Details: `logs/findings_t0_mp_vs_serial.md`. Serial is bit-deterministic (0.15% wall variance, identical V/F across trials) → T1 uses nw=1 for the golden-snapshot basis.

---

## File Structure Overview

**Created (test):**
- `corep_fast/tests/regression/test_cpu_worker_optim.py` — F1-F3 golden-snapshot regression gate (T1)
- `corep_fast/tests/regression/cpu_worker_optim_goldens/` — serialized golden outputs (committed; regenerated only if spec out-of-scope code changes)

**Created (production):**
- `corep_fast/utils/persistent_pool.py` — shared multiprocessing pool (T3 / W2)

**Modified (production):**
- `corep_fast/stages/s8_collapse.py` — W1 (:1629 + :920 Bucket A sites), W2 (3 Pool sites at :1428, :1769, :1939)
- `corep_fast/stages/s4_face_point.py` — W2 (:1186 Pool), W5 (`_get_local_components_np` GPU replacement), W6 (Stage D, angle decided in T4)
- `corep_fast/stages/s6_collapse.py` — W2 (:925 Pool), W4 (`_fastpath_trace_loops_numpy` GPU replacement)
- `corep_fast/stages/s7_rank_assign.py` — W2 (:1315 Pool), W7 (orchestration slim-down)

**Created (analysis):**
- `tmp/cpu_profile/driver_main_post_fix.py` — same cProfile driver but runs post-fix pipeline (T9)
- `logs/findings_cpu_worker_post_fix.md` — T9 / T10 outcome summary

**Touched elsewhere:**
- `logs/progress.md` — per-task append entries

**Untouched (must verify in T9):**
- `corep_fast/stages/s1_voxelize.py`
- `corep_fast/stages/s2_components.py`
- `corep_fast/stages/s3_edge_weights.py`
- All external dependencies

---

## Task 1: Create F1-F3 golden-snapshot regression gate

**Purpose:** Every commit under W1-W7 must pass this test. It pickles the CURRENT (pre-change) pipeline output, then asserts subsequent runs produce the same output. Lets GPU-rewrite tasks (W4/W5) confirm correctness via bit-equivalence.

**STATUS: shipped in commit `949aed9` (2026-04-17).** The shipped design diverged from Step 1 below — implementation uncovered three layers of nondeterminism:

1. **Default MP** → ~0.7% vertex-set drift per T0 → `SerialPool` monkeypatch.
2. **Same-process multi-fixture** → cuDNN/cuBLAS autotune cache + CUDA workspace leak across fixtures → F3 drifts ~0.9 V depending on what ran before. Fixed by running each fixture in its own **subprocess** via `_cpu_worker_optim_runner.py`.
3. **Cross-process CUDA nondeterminism** → even independent processes drift ~0.75 V on F3. Fixed with `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch import + `cudnn.benchmark=False` + `cudnn.deterministic=True` + `torch.use_deterministic_algorithms(True, warn_only=True)` + `PYTHONHASHSEED=0` in subprocess env.

Authoritative source: `corep_fast/tests/regression/test_cpu_worker_optim.py` + `_cpu_worker_optim_runner.py`. Step 1 code below is **historical** — do not copy-paste for subsequent tasks.

For W1-W7 tasks, use these wrappers to re-run / regenerate (gate always runs on **host-10-240-99-116 GPU 4** — local GPU forbidden because other inference jobs contaminate timings):

```bash
cat > /tmp/run_f123.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v"
EOF
bash /tmp/run_f123.sh

# Regenerate (only after out-of-scope code changes):
cat > /tmp/gen_goldens.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 GENERATE_GOLDEN=1 .venv/bin/pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v"
EOF
bash /tmp/gen_goldens.sh
```

Wall-time: 3 fixtures × ~55 s each ≈ 2:45 total.

---

**Files:**
- Create: `corep_fast/tests/regression/test_cpu_worker_optim.py`
- Create: `corep_fast/tests/regression/cpu_worker_optim_goldens/` (directory)
- Create: `corep_fast/tests/regression/cpu_worker_optim_goldens/.gitkeep`

- [ ] **Step 1: Create the test file skeleton**

```python
# corep_fast/tests/regression/test_cpu_worker_optim.py
"""Golden-snapshot regression gate for the CPU-worker-optim spec.

Every commit under W1-W7 of the spec must pass these tests. The goldens
are generated by running the pre-change pipeline once (see
GENERATE_GOLDEN flag below). After that they are byte-checked.

Determinism policy (CRITICAL):
- The pipeline's default multiprocessing path is NOT bit-deterministic
  (vertex set drifts ~0.7% across runs — T0 finding 2026-04-17). The
  golden gate therefore forces `num_workers=1` via a `SerialPool`
  monkeypatch applied BEFORE `corep_fast.pipeline` is imported. This
  makes the pipeline fully deterministic (0.15% wall variance, identical
  V/F across trials — T0 serial branch).

Tolerance policy:
- V / F scalar counts: exact match.
- Final mesh tensors (V×3 vertices, F×3 faces): max abs diff <= 1e-5
  (vertex positions) AND exact integer match (face indices).
- Internal intermediate tensors: exact for integers, <= 1e-6 for floats.

Fixtures (F1, F2, F3) mirror spec §5.1:
- F1: res=128, icosphere subdiv=3 (1280 faces)
- F2: res=256, same mesh
- F3: res=128, triple-concentric icosphere (covers s7 multi-loop path)

Generate goldens once:
    GENERATE_GOLDEN=1 pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v

Regenerate only if the spec's out-of-scope files change (s1/s2/s3 touched,
new deps, etc.). The regression gate is USELESS if goldens drift with
each change.
"""
# --- SerialPool monkeypatch MUST precede the corep_fast import. -----------
# Matches the pattern in tmp/cpu_profile/t0_driver.py used by T0. Stage
# modules do `from multiprocessing import Pool as _Pool` INSIDE functions,
# so re-binding `multiprocessing.Pool` is effective for every call.
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool


class _SerialPool:
    """Synchronous stand-in for multiprocessing.Pool — runs tasks serially."""
    def __init__(self, processes=None, *a, **kw):
        self.processes = processes
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, items, *a, **kw): return [fn(x) for x in items]
    def imap(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)
    def imap_unordered(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)
    def starmap(self, fn, items, *a, **kw): return [fn(*x) for x in items]
    def close(self): pass
    def terminate(self): pass
    def join(self): pass


_mp.Pool = _SerialPool
_mp_pool.Pool = _SerialPool
# --- End monkeypatch. corep_fast import below is safe. --------------------

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.pipeline import corep_pipeline

GOLDEN_DIR = Path(__file__).parent / "cpu_worker_optim_goldens"
GENERATE = os.environ.get("GENERATE_GOLDEN") == "1"


def _triple_icosphere() -> trimesh.Trimesh:
    """Three concentric icospheres at radii 0.40, 0.404, 0.408.

    The three-shell geometry exercises s7 multi-loop path where s6 emits
    loops from multiple disjoint components.
    """
    parts = []
    for r in (1.00, 1.01, 1.02):
        m = trimesh.creation.icosphere(subdivisions=3, radius=r * 0.4)
        parts.append(m)
    return trimesh.util.concatenate(parts)


FIXTURE_SPEC = [
    ("F1", 128, "icosphere_s3",   lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.4)),
    ("F2", 256, "icosphere_s3",   lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.4)),
    ("F3", 128, "triple_icosphere", _triple_icosphere),
]


def _run_pipeline(mesh_factory, resolution: int):
    """Run the pipeline under the serial (deterministic) monkeypatch.

    `corep_pipeline` takes `mesh_path: str` (verified in pipeline.py:284),
    so we export the trimesh fixture to a temp .ply first.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mesh = mesh_factory()
    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = str(Path(tmp) / "fixture.ply")
        mesh.export(mesh_path)
        batch, v, f = corep_pipeline(mesh_path, resolution, device)
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.cpu().numpy()
    return {
        "V_count": int(v.shape[0]),
        "F_count": int(f.shape[0]),
        "V": v.astype(np.float64),   # float64 to avoid FP drift
        "F": f.astype(np.int64),
    }


def _golden_path(label: str, mesh_name: str, resolution: int) -> Path:
    return GOLDEN_DIR / f"{label}_{mesh_name}_r{resolution}.pkl"


@pytest.fixture(scope="session")
def _ensure_golden_dir():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("label,resolution,mesh_name,mesh_factory", FIXTURE_SPEC,
                          ids=[f[0] for f in FIXTURE_SPEC])
def test_golden_snapshot(label, resolution, mesh_name, mesh_factory, _ensure_golden_dir):
    """Compare current pipeline output to committed golden snapshot."""
    pkl_path = _golden_path(label, mesh_name, resolution)
    out = _run_pipeline(mesh_factory, resolution)

    if GENERATE or not pkl_path.exists():
        if not GENERATE:
            pytest.fail(
                f"Golden missing: {pkl_path}. Run with GENERATE_GOLDEN=1 "
                f"to create it (ONLY from baseline commit)."
            )
        with open(pkl_path, "wb") as f:
            pickle.dump(out, f)
        print(f"[GENERATE] wrote {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        golden = pickle.load(f)

    # Exact counts
    assert out["V_count"] == golden["V_count"], (
        f"V count drift: golden={golden['V_count']} new={out['V_count']}"
    )
    assert out["F_count"] == golden["F_count"], (
        f"F count drift: golden={golden['F_count']} new={out['F_count']}"
    )

    # Vertex positions: 1e-5 tolerance
    max_v_diff = float(np.abs(out["V"] - golden["V"]).max()) if out["V"].size else 0.0
    assert max_v_diff <= 1e-5, (
        f"V position drift max={max_v_diff} > 1e-5"
    )

    # Face indices: exact
    assert np.array_equal(out["F"], golden["F"]), (
        "F index mismatch"
    )
```

- [ ] **Step 2: Create goldens directory placeholder**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p corep_fast/tests/regression/cpu_worker_optim_goldens
touch corep_fast/tests/regression/cpu_worker_optim_goldens/.gitkeep
```

- [ ] **Step 3: Verify test file imports clean (no runtime yet)**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "import ast; ast.parse(open('corep_fast/tests/regression/test_cpu_worker_optim.py').read())"
```

Expected: no output (clean parse).

- [ ] **Step 4: Generate goldens on 116 GPU 4**

```bash
cat > /tmp/gen_goldens.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 GENERATE_GOLDEN=1 .venv/bin/pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 | tee tmp/golden_gen.log"
EOF
bash /tmp/gen_goldens.sh
```

Expected output:
- 3 tests "pass" (they emit `[GENERATE] wrote ...`)
- 3 `.pkl` files created in `corep_fast/tests/regression/cpu_worker_optim_goldens/`

Verify:
```bash
ls -la corep_fast/tests/regression/cpu_worker_optim_goldens/*.pkl
# Expected: 3 files (F1_icosphere_s3_r128.pkl, F2_icosphere_s3_r256.pkl, F3_triple_icosphere_r128.pkl)
```

- [ ] **Step 5: Run without GENERATE to verify comparisons work**

```bash
cat > /tmp/run_goldens.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 | tail -20"
EOF
bash /tmp/run_goldens.sh
```

Expected: `3 passed` (comparisons against freshly-written goldens trivially pass).

- [ ] **Step 6: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add -f corep_fast/tests/regression/test_cpu_worker_optim.py
git add -f corep_fast/tests/regression/cpu_worker_optim_goldens/
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t1): golden-snapshot regression gate (F1-F3)

Test: corep_fast/tests/regression/test_cpu_worker_optim.py
Goldens: corep_fast/tests/regression/cpu_worker_optim_goldens/{F1,F2,F3}_*.pkl
Generated at HEAD=417e68b (spec V2 commit) on host-10-240-99-116 GPU 4.

Fixture matrix:
- F1: res=128, icosphere subdiv=3 (1280 faces)
- F2: res=256, same mesh (full 275k cube path)
- F3: res=128, triple-concentric icosphere (s7 multi-loop coverage)

Tolerance:
- V/F counts: exact
- V positions: max abs diff <= 1e-5
- F indices: exact

Every subsequent commit under W1-W7 must pass this gate. Regenerate
ONLY if out-of-scope code (s1/s2/s3 or external deps) changes.

Fixture: F1 PASS, F2 PASS, F3 PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

---

## Task 2: W1 — Bucket A `.item()` cleanup (warmup)

**Purpose:** 2h mechanical cleanup. Two Bucket-A sites from `logs/findings_sync_sources.md` §2 (ranks 8 and 10). Both sites aggregate tiny `.item()` / `.cpu()` scalars inside dispatcher functions. Findings documented impact <0.04 ms each — this task is a warmup that validates the W1-W7 commit flow without perf pressure.

**Files:**
- Modify: `corep_fast/stages/s8_collapse.py` — 2 Bucket A sites

**Context:** Findings §2 lists the sites by function, not by `.item()` line. The implementer must audit each function body for the scalar D2H calls. Only remove calls that are PROVABLY unused by control flow or downstream allocations; otherwise leave with a `# bucket-A, load-bearing: <reason>` comment.

- [ ] **Step 1: Audit `_process_shared_edges_from_tensors` for `.item()` / `.cpu()`**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
# Locate the function body
awk '/^def _process_shared_edges_from_tensors/,/^def [^_]|^class / {print NR": "$0}' \
    corep_fast/stages/s8_collapse.py | head -200
```

List every `.item()`, `.cpu()`, `.tolist()` call inside the function. For each:
- What is the returned scalar used for?
- If "only for a metric / logging / comment" → deletion candidate
- If "drives `torch.arange(n)` / `view(n, ...)` / list comprehension size" → **load-bearing, leave with comment**

Record the audit result in a small markdown file for traceability:

```bash
mkdir -p tmp/cpu_worker_optim_audit
```

```python
# tmp/cpu_worker_optim_audit/w1_s8_audit.md
# W1 Bucket A audit: `_process_shared_edges_from_tensors`
# Usage:
#   - Record each .item()/.cpu() in the function with its use and disposition.
```

- [ ] **Step 2: Audit `process_geometry_vectorized` for `.item()` / `.cpu()`**

Same procedure as Step 1, different function. From findings §2 rank 10 = `s8_collapse.py:920` function.

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
awk '/^def process_geometry_vectorized/,/^def [^_]|^class / {print NR": "$0}' \
    corep_fast/stages/s8_collapse.py | head -200
```

- [ ] **Step 3: Apply deletions**

For each scalar that is unambiguously "not load-bearing", delete the `.item()` / `.cpu()` call. Typical patterns:

```python
# BEFORE (deletable if count is only used for print):
count = int(some_tensor.numel())  # OK, no D2H
print(f"dispatching {int(some_tensor.sum().item())} items")  # .item() is the D2H

# AFTER:
print(f"dispatching {some_tensor.shape[0]} items")  # use shape instead
```

```python
# BEFORE (LOAD-BEARING, leave):
n_components = int(status_mask.sum().item())  # drives torch.arange(n_components, ...)

# AFTER (no change; add comment):
# bucket-A-audit: load-bearing, drives alloc-size arange below
n_components = int(status_mask.sum().item())
```

If no genuine deletions are found (all calls are load-bearing), W1 produces only the audit document — still counts as completed.

- [ ] **Step 4: Run F1-F3 fixture gate**

```bash
cat > /tmp/run_f123.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 | tail -10"
EOF
bash /tmp/run_f123.sh
```

Expected: `3 passed`. If any fixture fails, revert the problematic `.item()` removal.

- [ ] **Step 5: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add corep_fast/stages/s8_collapse.py
git add -f tmp/cpu_worker_optim_audit/w1_s8_audit.md
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t2/w1): Bucket A cleanup audit + safe removals

Audited 2 Bucket A sites per findings §2:
- s8_collapse.py:1629 _process_shared_edges_from_tensors
- s8_collapse.py:920  process_geometry_vectorized

Deleted <N> clearly-non-load-bearing .item()/.cpu() calls (see audit doc).
Remaining calls annotated with # bucket-A-audit: load-bearing reason.

Perf impact negligible (<40 us per findings); this is a correctness/
tidiness pass to validate the W1-W7 commit flow under F1-F3 gate.

Fixture: F1 PASS, F2 PASS, F3 PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

---

## Task 3: W2 — Persistent MP pool

**Purpose:** Replace 6 `with _Pool(num_workers) as p:` sites (sum = 124 fork() calls per e2e) with a single shared pool that spans the pipeline. Expected Δ ≈ 0.8 s (reclaims `posix.fork` self-time observed in the main-thread profile).

**Files:**
- Create: `corep_fast/utils/persistent_pool.py`
- Create: `corep_fast/utils/__init__.py` (if missing)
- Modify: `corep_fast/stages/s4_face_point.py:1186` (s4 Stage D Pool site)
- Modify: `corep_fast/stages/s6_collapse.py:925` (s6 slow-path Pool site)
- Modify: `corep_fast/stages/s7_rank_assign.py:1315` (s7 Phase-1/Phase-3 Pool site)
- Modify: `corep_fast/stages/s8_collapse.py:1428, :1769, :1939` (3 s8 Pool sites)

**Context:** Current `with _Pool(num_workers) as p:` semantics: create N forked children, map work, join, terminate. Each use = N forks. Persistent pool: create once, reuse. `multiprocessing.Pool` supports long-lived pools natively.

- [ ] **Step 1: Verify `corep_fast/utils/` module presence**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
ls -la corep_fast/utils/ 2>/dev/null || mkdir -p corep_fast/utils
test -f corep_fast/utils/__init__.py || touch corep_fast/utils/__init__.py
```

- [ ] **Step 2: Write `persistent_pool.py` with a failing test**

```python
# corep_fast/tests/unit/test_persistent_pool.py
"""Unit test for persistent_pool shared MP pool helper."""
import multiprocessing

import pytest

from corep_fast.utils.persistent_pool import get_pool, shutdown_pool


def _square(x):
    return x * x


def test_pool_reuse_returns_same_object():
    shutdown_pool()  # clean state
    p1 = get_pool(num_workers=2)
    p2 = get_pool(num_workers=2)
    assert p1 is p2, "Same num_workers should reuse the same pool"
    shutdown_pool()


def test_pool_map_correct_output():
    shutdown_pool()
    p = get_pool(num_workers=2)
    out = p.map(_square, [1, 2, 3, 4])
    assert out == [1, 4, 9, 16]
    shutdown_pool()


def test_pool_reinit_on_different_worker_count():
    shutdown_pool()
    p1 = get_pool(num_workers=2)
    p2 = get_pool(num_workers=4)
    assert p1 is not p2, "Different num_workers should produce a new pool"
    shutdown_pool()


def test_shutdown_idempotent():
    shutdown_pool()
    shutdown_pool()  # second call no-op
```

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p corep_fast/tests/unit
# Verify test fails (module doesn't exist yet)
.venv/bin/pytest corep_fast/tests/unit/test_persistent_pool.py -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'corep_fast.utils.persistent_pool'` or collection error.

- [ ] **Step 3: Implement `persistent_pool.py`**

```python
# corep_fast/utils/persistent_pool.py
"""Single shared multiprocessing.Pool across corep_fast stages.

Rationale: main-thread cProfile measured 124 posix.fork calls and 816 ms
self-time per e2e run (findings_main_thread.md). Per-dispatch temp pools
pay fork + teardown cost each time. This module exposes one reused pool.

Determinism: the pool initializer sets `PYTHONHASHSEED=0` in each worker.
Default MP produces ~0.7% vertex-set drift across runs (pre-existing bug
surfaced by T0 on 2026-04-17) because stages use `set`/`dict` iteration
to pick representative elements, and worker hash seeds are randomized by
default. Fixing the seed eliminates that source of nondeterminism. Any
remaining nondeterminism after this fix is algorithmic (tie-breakers on
unordered collections) and surfaces in T4's MP topology A/B check.

Semantics:
- `get_pool(num_workers)` returns a shared pool. Re-invocations with the
  same num_workers return the same pool. Different num_workers shuts down
  and replaces the pool (rare — typically num_workers is derived once per
  pipeline run and stays constant).
- `shutdown_pool()` explicitly terminates the pool (idempotent).
- On interpreter exit, atexit calls shutdown_pool automatically.

Safety:
- Not thread-safe for concurrent get_pool across threads; corep_fast runs
  single-threaded orchestration so this is acceptable. If that changes,
  add a lock.
- If a worker crashes (BrokenPipeError / OSError on map), call
  shutdown_pool() and retry with a fresh pool.
"""
import atexit
import multiprocessing as _mp
import os as _os
from typing import Optional

_pool: Optional[_mp.pool.Pool] = None
_pool_workers: Optional[int] = None


def _worker_initializer():
    """Runs once inside each worker process immediately after fork.

    Fixing PYTHONHASHSEED makes `set`/`dict` iteration deterministic
    across workers. Without this, stage code that picks a representative
    element from an unordered collection selects a different element each
    run, causing the ~0.7% vertex-set drift documented in T0.
    """
    _os.environ["PYTHONHASHSEED"] = "0"


def get_pool(num_workers: int) -> _mp.pool.Pool:
    """Return a shared multiprocessing.Pool. Reuses if num_workers matches."""
    global _pool, _pool_workers
    if _pool is not None and _pool_workers == num_workers:
        return _pool
    if _pool is not None:
        shutdown_pool()
    _pool = _mp.Pool(processes=num_workers, initializer=_worker_initializer)
    _pool_workers = num_workers
    return _pool


def shutdown_pool() -> None:
    """Terminate the shared pool. Idempotent."""
    global _pool, _pool_workers
    if _pool is None:
        return
    try:
        _pool.close()
        _pool.join()
    except Exception:
        try:
            _pool.terminate()
        except Exception:
            pass
    _pool = None
    _pool_workers = None


atexit.register(shutdown_pool)


def default_num_workers() -> int:
    """Mirror the default used at call sites across s4/s6/s7/s8."""
    return max(1, (_os.cpu_count() or 4) - 4)
```

- [ ] **Step 4: Run unit tests to verify**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
cat > /tmp/run_unit.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/unit/test_persistent_pool.py -v 2>&1 | tail -15"
EOF
bash /tmp/run_unit.sh
```

Expected: `4 passed`.

- [ ] **Step 5: Replace s4:1186 Pool site**

Current code (approximate; verify exact lines):

```python
# corep_fast/stages/s4_face_point.py:~1185
    num_workers = max(1, (_os.cpu_count() or 4) - 4)
    if num_workers > 1 and G >= 5000:
        from multiprocessing import Pool as _Pool
        chunksize = max(1, G // (num_workers * 4))
        with _Pool(num_workers) as p:
            results = p.map(_p2_uturn_worker, range(G), chunksize=chunksize)
    else:
        results = [_p2_uturn_worker(gi) for gi in range(G)]
```

Replace with:

```python
    num_workers = max(1, (_os.cpu_count() or 4) - 4)
    if num_workers > 1 and G >= 5000:
        from corep_fast.utils.persistent_pool import get_pool
        chunksize = max(1, G // (num_workers * 4))
        p = get_pool(num_workers)
        results = p.map(_p2_uturn_worker, range(G), chunksize=chunksize)
    else:
        results = [_p2_uturn_worker(gi) for gi in range(G)]
```

Use Edit tool with old_string containing the `with _Pool(num_workers) as p:` block + one line above + one line below for uniqueness.

- [ ] **Step 6: Run F1-F3 fixture gate after s4 change**

```bash
bash /tmp/run_f123.sh
```

Expected: `3 passed`. If any fails, revert s4 change and investigate worker-state contamination.

- [ ] **Step 7: Replace s6:925, s7:1315 Pool sites (same pattern)**

For each: locate the `with _Pool(num_workers) as p:` block, replace with `p = get_pool(num_workers)` + remove `with` indentation. Keep the `p.map(...)` line unchanged.

Same reference pattern as Step 5. After each site, run the F1-F3 fixture. **One Pool site per commit is fine and recommended** — easier to bisect if regression appears.

Create commit-per-site if comfortable, or batch all three into one commit after all F1-F3 pass.

- [ ] **Step 8: Replace s8:1428, s8:1769, s8:1939 Pool sites**

Same procedure. Three s8 sites are separate dispatch paths (candidate edges, geometry vectorized, parallel processing). Verify each block independently.

- [ ] **Step 9: Run F1-F3 + all unit tests**

```bash
cat > /tmp/run_full_tests.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/ -v --timeout=600 2>&1 | tail -30"
EOF
bash /tmp/run_full_tests.sh
```

Expected: all existing corep_fast tests + new F1-F3 PASS. If any existing test fails, investigate pool-state contamination (pool lives between tests in same interpreter).

**Mitigation for test contamination:** add a session-scoped pytest fixture to `corep_fast/tests/conftest.py`:

```python
# Append to corep_fast/tests/conftest.py
@pytest.fixture(autouse=True)
def _reset_persistent_pool():
    from corep_fast.utils.persistent_pool import shutdown_pool
    shutdown_pool()
    yield
    shutdown_pool()
```

Apply this only if tests fail due to pool state leaking across tests.

- [ ] **Step 10: Measure fork reduction**

```bash
cat > /tmp/measure_w2.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m tmp.cpu_profile.driver_main --res 256 2>&1 | tee tmp/cpu_profile/driver_main_res256_post_w2.log"
EOF
bash /tmp/measure_w2.sh

# Then re-run the analyzer on the new .prof
# (Driver overwrites results_main/main_thread_res256.prof — rename first to preserve baseline)
mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_pre_w2.prof
```

Actually the driver uses a FIXED filename `main_thread_res256.prof`. Modify the approach:

```bash
# BEFORE running post-W2 measurement, rename the pre-W2 prof so it's not overwritten
mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_pre_w2.prof
bash /tmp/measure_w2.sh
mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_post_w2.prof
# Then parse both with pstats to compare posix.fork count
.venv/bin/python -c "
import pstats
for tag in ['pre_w2', 'post_w2']:
    s = pstats.Stats(f'tmp/cpu_profile/results_main/main_thread_res256_{tag}.prof')
    total_fork_tt = 0.0
    fork_count = 0
    for func, (cc, nc, tt, ct, _callers) in s.stats.items():
        filename, lineno, fn_name = func
        if 'posix.fork' in fn_name or fn_name == 'fork':
            total_fork_tt += tt
            fork_count += cc
    print(f'{tag}: forks={fork_count}, fork_tt={total_fork_tt*1000:.1f} ms')
"
```

Expected: `pre_w2: forks=124, fork_tt=816ms` vs `post_w2: forks≈4-6, fork_tt≈30ms`. Confirm reduction.

- [ ] **Step 11: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add corep_fast/utils/__init__.py corep_fast/utils/persistent_pool.py
git add corep_fast/tests/unit/test_persistent_pool.py
git add corep_fast/stages/s4_face_point.py corep_fast/stages/s6_collapse.py \
        corep_fast/stages/s7_rank_assign.py corep_fast/stages/s8_collapse.py
git add corep_fast/tests/conftest.py  # if modified for pool-reset fixture
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t3/w2): persistent MP pool across stages

New: corep_fast/utils/persistent_pool.py
  - get_pool(num_workers): shared multiprocessing.Pool with reuse
  - shutdown_pool(): idempotent teardown, registered atexit

Replaced 6 per-dispatch Pool sites:
  s4_face_point.py:1186  (Stage D _p2_uturn_worker)
  s6_collapse.py:925     (slow-path _s6_worker)
  s7_rank_assign.py:1315 (Phase 1/3 _s7_rank_worker)
  s8_collapse.py:1428    (candidate edges)
  s8_collapse.py:1769    (geometry vectorized)
  s8_collapse.py:1939    (parallel processing)

Tests: corep_fast/tests/unit/test_persistent_pool.py — 4 passed.
Conftest: auto-reset pool per test (if added).

Measured: posix.fork self-time dropped from 816 ms/124 forks to
<N> ms/<M> forks. e2e wall-time measured separately at T9 re-profile.

Fixture: F1 PASS, F2 PASS, F3 PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

Fill `<N>`/`<M>` with actual measurements.

---

## Task 4: Measure W2 Δ and decide W6 angle

**Purpose:** W2 partially addresses W6's lock.acquire blocker (fork overhead was one component). Before committing to 3-7 d on W6, measure how much of the 2711 ms `lock.acquire` self-time remains after W2 lands. Then pick W6 angle (spec §4.5 Angle 1/2/3).

**Files:**
- Create: `logs/findings_w6_angle_decision.md` (1-page decision doc)
- No code changes in T4.

- [ ] **Step 1: Parse post-W2 profile for lock.acquire self-time**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "
import pstats
s = pstats.Stats('tmp/cpu_profile/results_main/main_thread_res256_post_w2.prof')
for func, (cc, nc, tt, ct, _callers) in s.stats.items():
    filename, lineno, fn_name = func
    if 'lock.acquire' in fn_name or 'lock_acquire' in fn_name:
        print(f'{fn_name} @ {filename}:{lineno} -- self {tt*1000:.1f} ms, cum {ct*1000:.1f} ms, calls {cc}')
"
```

Record remaining `lock.acquire` self-time (call it `L_post_w2`).

- [ ] **Step 2: Decision tree**

```
If L_post_w2 <= 500 ms:
    W6 is effectively resolved by W2. Document as skipped; reclaim W6's
    budget for W7 or for additional W4/W5 polish.
    Angle chosen: (1) "W2 fixes it"
Else if L_post_w2 in [500, 1500] ms AND Stage D workers dominate:
    Pick angle (2) — move BFS+UTurn to GPU. 3-5 d.
Else if L_post_w2 > 1500 ms AND GIL is holding back parallelism:
    Pick angle (3) — thread pool. Requires measuring GIL-holding fraction
    first; only viable if <30%. If unclear, spike 0.5 d in T7.
```

- [ ] **Step 3: Write decision doc**

```bash
cat > logs/findings_w6_angle_decision.md <<'EOF'
# W6 angle decision (post-W2 measurement)

**Baseline (pre-W2):** `lock.acquire` self-time = 2711 ms @ `s4_face_point.py:1186`
**Post-W2:** `lock.acquire` self-time = <L_post_w2> ms

**Decision:** Angle <1|2|3> per spec §4.5.

**Justification:** <1-2 sentences>

**T7 scope:** <details specific to chosen angle>
EOF
# Fill in placeholders with actual numbers
```

- [ ] **Step 4: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add -f logs/findings_w6_angle_decision.md
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t4): W6 angle decision post-W2

Measured post-W2 lock.acquire residual = <L> ms.
Chose angle <N>: <reason>.

Fixture: F1 PASS, F2 PASS, F3 PASS (unchanged from T3; no code changed)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

---

## Task 5: W4 — s6 fast-path tracer GPU-vectorize

**Purpose:** Replace 275k main-thread `_fastpath_trace_loops_numpy` calls (1891 ms self) with a GPU batched implementation. This is the single largest lever in the spec (spec §4.3).

**Files:**
- Modify: `corep_fast/stages/s6_collapse.py` — add `_fastpath_trace_loops_gpu`, gate it from `_fastpath_gpu_build_adjacency` consumer site
- Test: `corep_fast/tests/unit/test_s6_fastpath_tracer.py` (new)

**Context:** W4 is exploratory. The plan structures it as spike → TDD → integrate with 1-day sub-tasks. Implementer may iterate within each sub-task; commit cadence is "per sub-task DONE" not "per step".

### T5a: Spike + characterize the problem

- [ ] **Step 1: Read and understand `_fastpath_trace_loops_numpy` end-to-end**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
awk '/^def _fastpath_trace_loops_numpy/,/^def [^_]|^class / {print NR": "$0}' \
    corep_fast/stages/s6_collapse.py | head -80
```

Read the full function. Note:
- Inputs: `point_offset_row`, `adj_row`, `total_points` (per-cube numpy arrays)
- Output: `List[List[int]]` (list of loops, each a list of edge ids)
- Algorithm: walk adj_row starting from each unvisited point, follow one of its two neighbors (degree=2 invariant), collect edge ids, terminate when back to start.

- [ ] **Step 2: Identify how it's called (count + batching shape)**

```bash
grep -n "_fastpath_trace_loops_numpy" corep_fast/stages/s6_collapse.py
```

Find the call site, the pre-processing that produces `point_offset` / `adj` GPU tensors, and the post-processing that consumes the returned loops.

- [ ] **Step 3: Document the GPU algorithm in a comment block**

Add a design sketch at the top of where `_fastpath_trace_loops_gpu` will live:

```python
# Design sketch — batched per-cube loop tracing on GPU
#
# Input tensors (all GPU):
#   point_offset: (N, 19) int64     — per-cube, monotone, 19 edges+sentinel
#   adj:          (N, max_points, 2) int32 — each point's two neighbors
#   total_points: (N,) int64        — per-cube active point count
#
# Per-cube loop count bound: max_points/2 (each loop has >= 2 points).
# Per-cube loop length bound: max_points.
#
# Output tensors (all GPU):
#   loop_count:   (N,) int32        — number of loops per cube
#   loop_offsets: (N, max_loops+1) int32 — CSR-ish: loop k of cube i is
#                 edge_ids[loop_offsets[i,k]:loop_offsets[i,k+1]]
#   edge_ids:     (total_edges_across_all_cubes,) int32 flat CSR
#
# Algorithm (one thread per cube, sequential walk inside):
#   1. Build edge_of_point[N, max_points] via searchsorted on point_offset.
#   2. Initialize visited[N, max_points] = False.
#   3. For each p in [0, max_points):
#        if not visited[i, p] and p < total_points[i]:
#          start a new loop: follow adj[i, curr, 0] or adj[i, curr, 1]
#          whichever is not == prev, recording edge_of_point[curr].
#          Terminate when curr == start.
#   4. Emit via scan/cumsum of per-cube loop lengths.
#
# NOTE: This is inherently per-cube sequential — each loop walk depends on
# the previous step. BUT we can parallelize ACROSS cubes (N ≈ 275k).
# Implementation via torch.jit.script or a hand-written kernel. torch ops
# alone may be too branchy; try CPU-style loop with GPU tensors first,
# then optimize.
```

- [ ] **Step 4: Commit sub-task T5a**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add corep_fast/stages/s6_collapse.py
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t5a/w4): s6 fast-path GPU tracer — design sketch

Added design sketch comment block documenting the per-cube loop-tracing
algorithm for the GPU replacement of _fastpath_trace_loops_numpy. No
behavior change. T5b writes the first implementation + test.

Fixture: F1 PASS, F2 PASS, F3 PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

### T5b: TDD scaffold — write failing test first

- [ ] **Step 5: Write unit test with numpy-vs-gpu equivalence**

```python
# corep_fast/tests/unit/test_s6_fastpath_tracer.py
"""Equivalence test: numpy tracer output == GPU tracer output, per cube."""
import numpy as np
import pytest
import torch

from corep_fast.stages.s6_collapse import _fastpath_trace_loops_numpy


def _make_synthetic_cube(total_points: int, adjacency_edges: list[tuple[int, int]],
                          point_offset_row: np.ndarray):
    """Build adj_row from edge list. Each point has degree <= 2."""
    max_points = len(point_offset_row) - 1  # sentinel semantics
    adj = np.full((max_points, 2), -1, dtype=np.int32)
    slot = np.zeros(max_points, dtype=np.int32)
    for a, b in adjacency_edges:
        adj[a, slot[a]] = b; slot[a] += 1
        adj[b, slot[b]] = a; slot[b] += 1
    return adj


# Single-loop 4-cycle
SIMPLE_4CYCLE = {
    "point_offset_row": np.array([0, 1, 2, 3, 4] + [4]*14, dtype=np.int64),
    "adjacency_edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "total_points": 4,
}

# Two disjoint loops (2-cycle + 2-cycle)
TWO_LOOPS = {
    "point_offset_row": np.array([0, 1, 2, 3, 4] + [4]*14, dtype=np.int64),
    "adjacency_edges": [(0, 1), (1, 0), (2, 3), (3, 2)],  # degenerate but topology-valid
    "total_points": 4,
}


@pytest.mark.parametrize("case_name,case", [("simple_4cycle", SIMPLE_4CYCLE)])
def test_gpu_tracer_matches_numpy(case_name, case):
    max_points = len(case["point_offset_row"]) - 1
    adj = _make_synthetic_cube(case["total_points"], case["adjacency_edges"],
                                case["point_offset_row"])
    # Reference
    expected = _fastpath_trace_loops_numpy(
        case["point_offset_row"], adj, case["total_points"])

    # GPU impl — batch of 1
    from corep_fast.stages.s6_collapse import _fastpath_trace_loops_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    po_gpu = torch.from_numpy(case["point_offset_row"]).unsqueeze(0).to(device)
    adj_gpu = torch.from_numpy(adj).unsqueeze(0).to(device)
    tot_gpu = torch.tensor([case["total_points"]], dtype=torch.int64, device=device)

    loop_count, loop_offsets, edge_ids = _fastpath_trace_loops_gpu(
        po_gpu, adj_gpu, tot_gpu)

    # Reconstruct GPU result to list-of-lists, compare SET equivalence
    # (loop start point is implementation-defined; loops as multisets must match)
    cube0_count = int(loop_count[0].item())
    got = []
    for k in range(cube0_count):
        lo = int(loop_offsets[0, k].item())
        hi = int(loop_offsets[0, k + 1].item())
        got.append(sorted(edge_ids[lo:hi].cpu().tolist()))
    expected_sorted = [sorted(l) for l in expected]
    assert sorted(got) == sorted(expected_sorted)
```

- [ ] **Step 6: Run test to verify it fails**

```bash
ssh host-10-240-99-116 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/unit/test_s6_fastpath_tracer.py -v 2>&1 | tail -10"
```

Expected: `ImportError: cannot import name '_fastpath_trace_loops_gpu'`.

### T5c: First implementation (naive torchscript-friendly)

- [ ] **Step 7: Write `_fastpath_trace_loops_gpu` first-pass**

```python
# Append to corep_fast/stages/s6_collapse.py
import torch


def _fastpath_trace_loops_gpu(
    point_offset: torch.Tensor,   # (N, 19) int64
    adj: torch.Tensor,            # (N, max_points, 2) int32
    total_points: torch.Tensor,   # (N,) int64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched per-cube loop-tracing on GPU.

    First-pass: per-cube sequential walk, parallel across cubes (N dim).
    Implementation uses a Python loop over cubes initially (still main-
    thread but GPU tensors — may be ~as slow as numpy). T5d optimizes.

    Returns:
        loop_count: (N,) int32
        loop_offsets: (N, max_loops + 1) int32 CSR per-cube
        edge_ids: (total_edges,) int32 flat CSR payload
    """
    N, max_points, _ = adj.shape
    device = adj.device

    # Per-point edge id via searchsorted on point_offset[:, :-1]
    # edge_of_point[n, p] = largest e s.t. point_offset[n, e] <= p
    point_ids = torch.arange(max_points, device=device)
    # Broadcast: (N, max_points) vs (N, 19)
    # searchsorted semantics: right, -1 to get last <= p index
    edge_of_point = torch.searchsorted(
        point_offset[:, :18],                           # (N, 18)
        point_ids.unsqueeze(0).expand(N, -1).to(torch.int64),  # (N, max_points)
        right=True,
    ) - 1  # (N, max_points) int64

    # ---- per-cube trace (naive: Python loop over N, GPU tensors per cube)
    cube_loops: list[list[list[int]]] = []
    adj_cpu = adj.cpu().numpy()  # ok for spike: measured at T9
    point_offset_cpu = point_offset.cpu().numpy()
    total_points_cpu = total_points.cpu().numpy()

    for n in range(N):
        loops_n = _fastpath_trace_loops_numpy(
            point_offset_cpu[n], adj_cpu[n], int(total_points_cpu[n]))
        cube_loops.append(loops_n)

    # Pack into CSR
    loop_count = torch.tensor([len(l) for l in cube_loops], dtype=torch.int32,
                              device=device)
    max_loops = int(loop_count.max().item()) if N > 0 else 0
    loop_offsets = torch.zeros((N, max_loops + 1), dtype=torch.int32,
                                device=device)
    flat_ids: list[int] = []
    for i, loops in enumerate(cube_loops):
        running = 0
        for k, loop in enumerate(loops):
            loop_offsets[i, k + 1] = loop_offsets[i, k] + len(loop)
            flat_ids.extend(loop)
            running += len(loop)
    edge_ids = torch.tensor(flat_ids, dtype=torch.int32, device=device)
    return loop_count, loop_offsets, edge_ids
```

**This first-pass is a SLOW stub** that still calls the numpy tracer but returns the result in the new tensor format. Purpose: get the test passing; then optimize in T5d. Commit cadence: "T5c commits the working-but-slow stub".

- [ ] **Step 8: Run the unit test to verify it passes**

```bash
ssh host-10-240-99-116 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && CUDA_VISIBLE_DEVICES=4 .venv/bin/pytest corep_fast/tests/unit/test_s6_fastpath_tracer.py -v 2>&1 | tail -10"
```

Expected: `1 passed`.

- [ ] **Step 9: Commit T5c**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add corep_fast/stages/s6_collapse.py corep_fast/tests/unit/test_s6_fastpath_tracer.py
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t5c/w4): _fastpath_trace_loops_gpu first-pass stub

Unit test with numpy equivalence (simple 4-cycle case) passing.
Implementation is a slow stub: delegates per-cube work to
_fastpath_trace_loops_numpy while returning output in the new GPU CSR
tensor format. T5d optimizes the inner loop to run on GPU.

No integration yet — s6_collapse.py still uses the numpy path.

Fixture: F1 PASS, F2 PASS, F3 PASS (no integration change)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

### T5d: Optimize the inner loop (GPU parallelism)

- [ ] **Step 10: Rewrite the per-cube inner loop to stay on GPU**

Replace the Python loop over N cubes with a tensor-native algorithm. Options (listed from simplest to most complex):

**Option 10a (torch.jit.script):** annotate a Python loop and let torch.jit compile it. May give 2-3x speedup with minimal code change.

**Option 10b (vectorized walk):** maintain `curr`, `prev`, `start` as (N,) tensors; advance all cubes in lockstep up to max_points iterations. `masked_scatter_` to write visited edges. This achieves full N-parallelism.

Start with 10b:

```python
def _fastpath_trace_loops_gpu(...) -> ...:
    N, max_points, _ = adj.shape
    device = adj.device

    # Precompute edge_of_point
    point_ids = torch.arange(max_points, device=device)
    edge_of_point = torch.searchsorted(
        point_offset[:, :18],
        point_ids.unsqueeze(0).expand(N, -1).to(torch.int64),
        right=True,
    ) - 1  # (N, max_points)

    # State
    visited = torch.zeros((N, max_points), dtype=torch.bool, device=device)
    # Per-cube loop tracking: curr loop id, current position, previous, start
    loop_id = torch.full((N,), -1, dtype=torch.int32, device=device)
    curr = torch.full((N,), -1, dtype=torch.int32, device=device)
    prev = torch.full((N,), -1, dtype=torch.int32, device=device)
    start = torch.full((N,), -1, dtype=torch.int32, device=device)

    # Flat edge id buffer per cube (max max_points)
    flat_out = torch.full((N, max_points), -1, dtype=torch.int32, device=device)
    flat_out_ptr = torch.zeros((N,), dtype=torch.int32, device=device)

    # Per-cube loop_offsets buffer
    max_loops = max_points // 2 + 1
    loop_offsets_raw = torch.zeros((N, max_loops + 1), dtype=torch.int32,
                                     device=device)

    # Outer seed-loop: iterate starting points from 0..max_points-1
    # For each seed, the cubes that have not visited it and have
    # p < total_points start a new loop.
    # Then inner walk: up to max_points hops.

    # NOTE: writing the full vectorized algorithm requires care for edge
    # cases (degree-2 invariant, loop closure). First implementation here
    # should be followed by F1-F3 tests at each increment. Refer to
    # _fastpath_trace_loops_numpy for algorithm ground truth.

    # ... [detailed vectorized implementation] ...

    # Fall-through: if vectorized path is too complex to land in T5d,
    # commit an intermediate torch.jit.script version as a second-pass.

    return loop_count, loop_offsets, edge_ids
```

**IMPORTANT caveat:** writing the full vectorized walk correctly is non-trivial. Target for T5d: get a version that (a) passes the unit test AND (b) reduces `_fastpath_trace_loops_numpy` calls significantly (≥50% reduction observed in main-thread cProfile).

If T5d proves too hard in <2 days, escalate by returning DONE_WITH_CONCERNS — implementer can suggest (c) torch.jit + CPU fallback for the per-cube walk, or (d) moving the inner loop to a C extension via Cython. The spec forbids new deps, so (d) is out; (c) is acceptable if TorchScript is considered part of the base PyTorch install.

- [ ] **Step 11: Add stress test matching the full res=256 scale**

```python
# Add to corep_fast/tests/unit/test_s6_fastpath_tracer.py
def test_gpu_tracer_full_icosphere_res256():
    """Run through a full res=256 icosphere pipeline up to s6 and compare."""
    import trimesh
    from corep_fast.pipeline import corep_pipeline_up_to_s5  # assumes this helper exists
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ew = corep_pipeline_up_to_s5(mesh, 256, device)   # edge_weights batch

    from corep_fast.stages.s6_collapse import (
        _fastpath_gpu_build_adjacency,
        _fastpath_trace_loops_numpy,
        _fastpath_trace_loops_gpu,
    )
    adj, po, total = _fastpath_gpu_build_adjacency(ew)

    # Run both paths
    loop_count_gpu, loop_offsets_gpu, edge_ids_gpu = _fastpath_trace_loops_gpu(
        po, adj, total)

    # Reference numpy path
    adj_np = adj.cpu().numpy()
    po_np = po.cpu().numpy()
    total_np = total.cpu().numpy()
    for n in range(adj_np.shape[0]):
        expected_loops = _fastpath_trace_loops_numpy(po_np[n], adj_np[n], int(total_np[n]))
        got_count = int(loop_count_gpu[n].item())
        assert got_count == len(expected_loops), f"cube {n}: {got_count} vs {len(expected_loops)}"
        # Per-loop equivalence...
```

**Note:** If `corep_pipeline_up_to_s5` doesn't exist as a helper, the implementer creates it inline using the fixture-style pipeline call. This test is expensive — mark with `@pytest.mark.slow` and only run manually.

- [ ] **Step 12: Integration — replace the numpy call site**

Find where `_fastpath_trace_loops_numpy` is called from `s6_collapse` main body:

```bash
grep -n "_fastpath_trace_loops_numpy" corep_fast/stages/s6_collapse.py
```

There's typically one main call site inside the fast-path flow. Replace:

```python
# BEFORE (conceptual — confirm exact structure):
adj_np = adj.cpu().numpy()
po_np = po.cpu().numpy()
total_np = total.cpu().numpy()
for n in range(N):
    loops = _fastpath_trace_loops_numpy(po_np[n], adj_np[n], int(total_np[n]))
    per_cube_loops[n] = loops

# AFTER:
loop_count, loop_offsets, edge_ids = _fastpath_trace_loops_gpu(po, adj, total)
# Then convert to per-cube loops format if downstream expects List[List[int]]:
# Re-materialize only if needed (prefer keeping CSR if consumers can accept it).
```

**Strategy:** if the downstream consumer expects `List[List[int]]`, write an adapter `_csr_to_list_of_lists(loop_count, loop_offsets, edge_ids)` — but beware this re-introduces a per-cube Python loop. Prefer updating the consumer to accept the CSR form.

- [ ] **Step 13: Run F1-F3 fixture gate**

```bash
bash /tmp/run_f123.sh
```

Expected: `3 passed`.

- [ ] **Step 14: Measure s6 post-W4 wall-time**

```bash
cat > /tmp/measure_w4.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m tmp.cpu_profile.driver_main --res 256 2>&1 | tee tmp/cpu_profile/driver_main_res256_post_w4.log"
EOF
bash /tmp/measure_w4.sh
mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_post_w4.prof
```

Parse for `_fastpath_trace_loops_numpy` call count (should drop ≥95% or self-time ≥80%, per DoD #4).

- [ ] **Step 15: Commit T5d**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add corep_fast/stages/s6_collapse.py corep_fast/tests/unit/test_s6_fastpath_tracer.py
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t5d/w4): _fastpath_trace_loops_gpu optimized + integrated

Vectorized per-cube loop tracer on GPU (N-parallel). Integrated into
s6_collapse main flow. Unit test: equivalence with numpy tracer on simple
cases + full res=256 pipeline (slow-marked).

Measured:
  _fastpath_trace_loops_numpy call count: 275426 -> <N_new> (drop <X>%)
  s6 stage wall-time: 2.47s -> <S_new>s (drop <Y>%)
  e2e wall-time: 10.1s -> <E_new>s

Fixture: F1 PASS, F2 PASS, F3 PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

---

## Task 6: W5 — s4 Part 2 batched GPU union-find

**Purpose:** Replace 275k `_get_local_components_np` main-thread calls (1030 ms self) with a batched GPU label-propagation. Spec §4.4.

**Files:**
- Modify: `corep_fast/stages/s4_face_point.py` — add `_get_local_components_gpu`, gate replacement at call site
- Test: `corep_fast/tests/unit/test_s4_uf.py` (new)

Follow the same T5-style phased approach: T6a spike → T6b TDD scaffold → T6c implementation → T6d integrate + measure.

### T6a: Spike

- [ ] **Step 1: Read `_get_local_components_np` and identify the call site**

```bash
awk '/^def _get_local_components_np/,/^def [^_]|^class / {print NR": "$0}' \
    corep_fast/stages/s4_face_point.py
grep -n "_get_local_components_np" corep_fast/stages/s4_face_point.py
```

Record:
- Input: `face_ids`, `mesh_faces`, `face_adj` (all numpy int arrays)
- Output: `list[list[int]]` — partition of face_ids into connected components
- Caller: where it's called per-cube in a Python loop

### T6b: TDD scaffold

- [ ] **Step 2: Unit test with equivalence check**

```python
# corep_fast/tests/unit/test_s4_uf.py
"""Equivalence: _get_local_components_gpu vs _get_local_components_np."""
import numpy as np
import pytest
import torch

from corep_fast.stages.s4_face_point import _get_local_components_np


@pytest.fixture
def tiny_triangle_adj():
    """3 triangles, first two share an edge, third is isolated."""
    face_ids = np.array([10, 11, 12], dtype=np.int32)
    mesh_faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]], dtype=np.int32)
    # face_adj: triangle 0 adj to triangle 1 via shared edge; triangle 2 alone
    face_adj = np.array([
        [1, -1, -1],   # triangle 10's neighbors (in mesh face index)
        [0, -1, -1],
        [-1, -1, -1],
    ], dtype=np.int32)
    return face_ids, mesh_faces, face_adj


def test_gpu_uf_matches_numpy_two_components(tiny_triangle_adj):
    face_ids, mesh_faces, face_adj = tiny_triangle_adj
    expected = _get_local_components_np(face_ids, mesh_faces, face_adj)

    from corep_fast.stages.s4_face_point import _get_local_components_gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fids_gpu = torch.from_numpy(face_ids).to(device)
    fadj_gpu = torch.from_numpy(face_adj).to(device)

    got = _get_local_components_gpu(fids_gpu, fadj_gpu)
    # Convert got (tensor partition) to list[list[int]] and compare AS SETS
    got_sets = {frozenset(l) for l in got}
    exp_sets = {frozenset(l) for l in expected}
    assert got_sets == exp_sets
```

### T6c: Implementation

- [ ] **Step 3: Batched label propagation**

```python
# corep_fast/stages/s4_face_point.py (append)
def _get_local_components_gpu(
    face_ids: torch.Tensor,   # (n,) int32
    face_adj: torch.Tensor,   # (n, 3) int32 (-1 padded)
) -> list[list[int]]:
    """Batched UF via iterative label-propagation on GPU.

    Input face_adj is a GLOBAL face->3-neighbor table (int32, -1 padded),
    shared with the numpy reference. Neighbors are global face ids; the
    UF builds a local face_id->index map internally and only unions when
    a neighbor appears in the current cube's face_ids set. (Corrected
    2026-04-17 per T6a spike audit — earlier "LOCAL indices" note was
    wrong.)

    Algorithm:
        labels = arange(n)
        for iter in 0..log2(n):
            labels = min(labels[adj[:, i]]) for i in 0,1,2 valid neighbors
        Group by final label.
    """
    n = face_ids.shape[0]
    if n == 0:
        return []
    device = face_ids.device
    labels = torch.arange(n, dtype=torch.int32, device=device)

    # max iterations: log2(n) + 1, bounded since n <= 12 per call
    for _ in range(max(1, int(torch.ceil(torch.log2(torch.tensor(float(n + 1)))).item()))):
        new_labels = labels.clone()
        for slot in range(face_adj.shape[1]):
            nbr = face_adj[:, slot]  # (n,) int32, -1 = pad
            valid = nbr >= 0
            # propagate min
            src = torch.where(valid, labels[nbr.clamp(min=0)], labels)
            new_labels = torch.minimum(new_labels, src)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    # Group by label on GPU, return as list of Python lists
    labels_cpu = labels.cpu().numpy()
    fids_cpu = face_ids.cpu().numpy()
    groups: dict[int, list[int]] = {}
    for i, lb in enumerate(labels_cpu):
        groups.setdefault(int(lb), []).append(int(fids_cpu[i]))
    return list(groups.values())
```

**NOTE:** this first-pass still ends with a `.cpu()` + Python grouping. For the batched (all 275k cubes at once) variant, grouping must also be GPU-native using segment-reduce. That's T6d.

### T6d: Batched version across all cubes + integrate

- [ ] **Step 4: Write `_get_local_components_gpu_batched`**

Process all 275k cubes in parallel. Input: padded `(N_cubes, max_faces_per_cube=12, ...)` tensors. Output: per-cube labels. This is the real production path.

Outline (write full code during implementation):

```python
def _get_local_components_gpu_batched(
    batched_face_ids: torch.Tensor,   # (N, max_f) int32, -1 pad
    batched_face_adj: torch.Tensor,   # (N, max_f, 3) int32, -1 pad (LOCAL idx)
) -> torch.Tensor:
    """Return (N, max_f) int32 component labels per cube.

    Labels are canonicalized to [0, k-1] per cube.
    """
    # ... batched label-propagation ...
    # ... canonicalize labels per cube via argsort + rank map ...
```

Integrate at the caller site in s4_face_point. Downstream consumer currently expects `list[list[int]]` per cube — adapt.

- [ ] **Step 5: Run F1-F3 fixture gate**

```bash
bash /tmp/run_f123.sh
```

Expected: `3 passed`. Tolerate canonical-form equivalence (different root labels OK, same partition required).

- [ ] **Step 6: Measure post-W5**

Same pattern as W4 measurement:

```bash
mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_post_w4.prof
cat > /tmp/measure_w5.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m tmp.cpu_profile.driver_main --res 256 2>&1 | tee tmp/cpu_profile/driver_main_res256_post_w5.log"
EOF
bash /tmp/measure_w5.sh
mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_post_w5.prof
```

Check: `_get_local_components_np` call count drops ≥95% or self-time drops ≥80%.

- [ ] **Step 7: Commit T6**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add corep_fast/stages/s4_face_point.py corep_fast/tests/unit/test_s4_uf.py
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t6/w5): s4 Part 2 batched GPU union-find

Replaced _get_local_components_np (275k main-thread calls, 1030 ms self)
with batched GPU label-propagation across all cubes. Unit test:
equivalence with numpy UF on small case + partition-set equivalence.

Integrated into _compute_component_points_gpu. Output canonical form
changed (label values are not preserved, only partition equivalence) —
downstream consumers verified independent of label values.

Measured:
  _get_local_components_np call count: 275541 -> <N_new>
  s4 Part 2 wall-time: <pre>s -> <post>s
  e2e wall-time: <pre>s -> <post>s

Fixture: F1 PASS, F2 PASS, F3 PASS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

---

## Task 7: W6 — s4 Part 1 Stage D unblock (angle from T4)

**Purpose:** Address the remaining s4 Stage D bottleneck. Angle picked in T4.

**Files:**
- Modify: `corep_fast/stages/s4_face_point.py` — angle-specific
- Possibly: new helper module

**Angle-specific steps** (pick one based on T4 decision):

### Angle 1: "W2 resolved it — skip W6"

- [ ] **Step 1: Document skip in logs/progress.md**

```bash
cat >> logs/progress.md <<'EOF'

## 2026-04-XX — W6 skipped per T4 decision

Post-W2 lock.acquire residual was <L> ms (≤ 500 ms threshold). W6 is
considered resolved by W2. Reclaimed budget routed to W7 polish.
EOF
```

- [ ] **Step 2: Empty-commit to mark W6 closed**

```bash
git commit --allow-empty -m "$(cat <<'CMSG'
cpu-worker-optim(t7/w6): skipped — W2 resolved the Stage D blocker

Per T4 decision doc (logs/findings_w6_angle_decision.md): post-W2
lock.acquire self-time dropped below 500 ms threshold. W6 angle 1 chosen.
Budget routed to W7.

Fixture: F1 PASS, F2 PASS, F3 PASS (no code change)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

### Angle 2: "Move Stage D BFS+UTurn to GPU"

- [ ] **Step 1: Read current Stage D code**

```bash
awk '/^def _compute_face_weights_gpu/,/^def [^_]|^class / {print NR": "$0}' \
    corep_fast/stages/s4_face_point.py | head -300
```

- [ ] **Step 2: Design sketch (as commit-0 comment) + TDD test**

Outline GPU algorithm for BFS+UTurn using already-on-device CSR adjacency from Stage B.

- [ ] **Step 3: Implement + integrate + measure**

Same sub-task pattern as T5/T6 (spike → scaffold → impl → optimize → integrate).

**Budget:** 3-5 days. Stop if implementation exceeds 5 days without hitting ≥50% of lock.acquire residual elimination — re-escalate.

### Angle 3: "Thread pool instead of process pool"

- [ ] **Step 1: Measure GIL-holding fraction in worker**

Use `py-spy` (available in `.venv`) or `sys.settrace` to sample worker calls. If pure-numpy/scatter work dominates (releases GIL), thread pool gives parallelism without fork.

- [ ] **Step 2: If GIL-holding < 30%, swap `Pool` for `ThreadPool`**

Minimal patch: `from multiprocessing.pool import ThreadPool`; replace `get_pool` usage at the s4 Stage D site only. Other stages keep process-based pool.

- [ ] **Step 3: Implement + measure + gate**

### Common to all angles

- [ ] **Step N-1: Run F1-F3 fixture**
- [ ] **Step N: Commit with angle-specific message, include measurement**

---

## Task 8: W7 — s7 orchestration slim-down

**Purpose:** 1010 ms self-time in `s7_rank_assign.py:1175` onwards (orchestration / CSR conversion / work-item build). Spec §4.6 says: drill down first, commit only if ≥200 ms win identified.

**Files:**
- Modify: `corep_fast/stages/s7_rank_assign.py` (conditionally)

- [ ] **Step 1: Drill down via pstats filter**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -c "
import pstats
s = pstats.Stats('tmp/cpu_profile/results_main/main_thread_res256_post_w5.prof')
s.sort_stats('tottime')
# Print only functions in s7_rank_assign.py
class _F:
    def __init__(self, sub): self.sub = sub
    def __call__(self, x): return self.sub in x
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s.print_stats('s7_rank_assign.py', 30)
print(buf.getvalue())
"
```

Record top candidates by self-time in `tmp/cpu_worker_optim_audit/w7_s7_drilldown.md`.

- [ ] **Step 2: Decision**

```
If top candidate self-time >= 200 ms AND has a mechanical fix (e.g. redundant .cpu(),
Python list->tensor conversion, repeated computation): proceed to Step 3.
Else: skip W7 with justification in progress log.
```

- [ ] **Step 3a (skip branch): empty commit marking skip**

```bash
git commit --allow-empty -m "cpu-worker-optim(t8/w7): skipped — no ≥200 ms mechanical win identified"
```

- [ ] **Step 3b (proceed branch): implement fix**

Apply targeted edit. Example if drill-down reveals redundant `.cpu().numpy()`:

```python
# BEFORE
foo_np = batch.foo.cpu().numpy()  # already transferred earlier at line ...
bar_np = batch.bar.cpu().numpy()

# AFTER
foo_np = cached_foo_np  # from earlier transfer
bar_np = cached_bar_np
```

Generalize: collect ALL `.cpu().numpy()` in the 1175+ region; identify duplicates; cache and reuse.

- [ ] **Step 4: Run F1-F3 fixture**

```bash
bash /tmp/run_f123.sh
```

- [ ] **Step 5: Measure + commit**

---

## Task 9: Re-profile full pipeline + update findings doc

**Purpose:** Measure total wall-time Δ against DoD #6 (≥3 s reduction @ res=256).

**Files:**
- Create: `logs/findings_cpu_worker_post_fix.md`

- [ ] **Step 1: Run 3-trial post-fix profile**

```bash
cat > /tmp/final_profile.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
for i in 1 2 3; do
    ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m tmp.cpu_profile.driver_main --res 256 2>&1 | tee tmp/cpu_profile/final_run_${i}.log"
    # Keep .prof with index
    ssh host-10-240-99-116 "cd ${PROJ} && mv tmp/cpu_profile/results_main/main_thread_res256.prof tmp/cpu_profile/results_main/main_thread_res256_final_run${i}.prof"
done
EOF
bash /tmp/final_profile.sh
```

- [ ] **Step 2: Measure VRAM delta**

```bash
cat > /tmp/measure_vram.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
# Pre-change baseline: checkout 417e68b temporarily to measure pre-fix VRAM
# (or reuse if already recorded)
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c '
import torch
from tmp.cpu_profile.driver_main import run_pipeline
from tmp.profile_deep.build_icosphere import build_icosphere_subdiv3
import gc
mesh = build_icosphere_subdiv3()
device = torch.device(\"cuda:0\")
gc.collect(); torch.cuda.empty_cache()
_ = run_pipeline(mesh, 256, device)   # warmup
torch.cuda.reset_peak_memory_stats()
_ = run_pipeline(mesh, 256, device)
print(f\"peak_alloc_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}\")
print(f\"peak_reserved_mb: {torch.cuda.max_memory_reserved() / 1e6:.1f}\")
'"
EOF
bash /tmp/measure_vram.sh | tee tmp/cpu_profile/final_vram.log
```

- [ ] **Step 2b: MP topology A/B determinism check (post-W2 `PYTHONHASHSEED=0`)**

Verifies that the pool-initializer hash-seed fix eliminated the pre-existing ~0.7%
vertex-set drift (T0 finding, 2026-04-17). Runs the pipeline twice under default MP
and compares V/F outputs; compares MP output against the bit-deterministic serial
golden (nw=1 path already committed in T1 goldens).

```bash
cat > /tmp/mp_topology_ab.sh <<'EOF'
#!/bin/bash
PROJ=/mnt/novita2/siyuan/workspace/TRELLIS.2
ssh host-10-240-99-116 "cd ${PROJ} && CUDA_VISIBLE_DEVICES=4 .venv/bin/python -c '
import numpy as np, torch, trimesh, tempfile, pathlib, pickle
from corep_fast.pipeline import corep_pipeline

device = torch.device(\"cuda:0\")
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
with tempfile.TemporaryDirectory() as tmp:
    p = str(pathlib.Path(tmp) / \"f.ply\"); mesh.export(p)
    _, v1, f1 = corep_pipeline(p, 256, device)
    _, v2, f2 = corep_pipeline(p, 256, device)

v1 = v1.cpu().numpy() if hasattr(v1, \"cpu\") else v1
v2 = v2.cpu().numpy() if hasattr(v2, \"cpu\") else v2
f1 = f1.cpu().numpy() if hasattr(f1, \"cpu\") else f1
f2 = f2.cpu().numpy() if hasattr(f2, \"cpu\") else f2

print(f\"V counts: run1={len(v1)}, run2={len(v2)}\")
print(f\"F counts: run1={len(f1)}, run2={len(f2)}\")
print(f\"V bit-equal: {np.array_equal(v1, v2)}\")
print(f\"F bit-equal: {np.array_equal(f1, f2)}\")
if v1.shape == v2.shape:
    diff = np.abs(v1.astype(np.float64) - v2.astype(np.float64)).max()
    print(f\"max|V1-V2|: {diff:.3e}\")

# Compare MP output against serial golden
golden = pickle.loads(pathlib.Path(\"corep_fast/tests/regression/cpu_worker_optim_goldens/F2_icosphere_s3_r256.pkl\").read_bytes())
print(f\"V vs golden bit-equal: {np.array_equal(v1, golden[\\\"V\\\"])}\")
print(f\"V vs golden max diff: {np.abs(v1.astype(np.float64) - golden[\\\"V\\\"]).max():.3e}\")
'"
EOF
bash /tmp/mp_topology_ab.sh | tee tmp/cpu_profile/final_mp_topology.log
```

Acceptance:
- MP run1 vs MP run2: V/F bit-equal (✅ `PYTHONHASHSEED=0` fix worked) OR max|ΔV| ≤ 1e-5
  (acceptable algorithmic residual — document in findings).
- MP vs serial golden: max|ΔV| ≤ 1e-5 (same tolerance the golden gate uses).

If MP-vs-MP shows large drift (≥1e-3 max|ΔV| or count mismatch), the hash-seed fix
alone was insufficient; file a follow-up for `set`/`dict` tiebreaker audit and
document in the findings doc. This does NOT block T9 sign-off (serial goldens remain
the authoritative correctness gate) but should feed into the T10 next-spec recommendation.

- [ ] **Step 3: Write findings doc**

```markdown
# tmp/cpu_profile/... → logs/findings_cpu_worker_post_fix.md
# Structure: TL;DR + per-stage wall + DoD table + residual hotspots
```

Include:
- Per-stage wall table (pre vs post per W)
- Top-10 hotspot table post-fix
- DoD #1-#10 check
- Recommended next spec scope (if any — e.g., Triton feasibility given new VRAM baseline)

- [ ] **Step 4: Commit findings doc**

```bash
git add -f logs/findings_cpu_worker_post_fix.md
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t9): post-fix re-profile findings

e2e wall @ res=256: <pre>s -> <post>s (delta <D>s, <P>%)
Per-W breakdown:
  W1: ~0 (cleanup)
  W2: -<w2_delta>s
  W4: -<w4_delta>s
  W5: -<w5_delta>s
  W6: -<w6_delta>s (angle <N>)
  W7: -<w7_delta>s (or skipped)

VRAM peak: <pre>MB -> <post>MB (delta <vd>MB, within 500MB budget)

DoD status: <N>/10 items PASS.

Residual top-3 hotspots: see findings doc.

Fixture: F1 PASS, F2 PASS, F3 PASS (final verification)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

---

## Task 10: Handoff document for next spec

**Purpose:** Document residual bottlenecks + recommend next spec scope (e.g., Triton feasibility).

**Files:**
- Modify: `logs/findings_cpu_worker_post_fix.md` (append §4 Next-spec recommendation)
- Modify: `logs/progress.md` (final entry)

- [ ] **Step 1: Analyze residual**

Key questions:
- GPU idle % post-fix?
- Are W4/W5 GPU kernels now the new bottleneck (good problem to have)?
- Is the §1 1.0 s sync still a meaningful fraction?
- What is the new ROI of introducing Triton given the new landscape?

- [ ] **Step 2: Append recommendation section**

Template:

```markdown
## 4. Recommended next spec

Residual e2e wall: ~<R>s @ res=256. Breakdown:
- GPU-bound (kernel time): <X>s
- Remaining main-thread: <Y>s
- Sync / idle: <Z>s

**Option alpha:** Triton spec targeting §1 1.0s sync + W4/W5 GPU kernel optimization.
Expected Δ: <a>s. Effort: 2-3 wk. Risk: introduces Triton dependency.

**Option beta:** <...>

**Recommendation:** <choice>. Justification: <1-2 sentences>.
```

- [ ] **Step 3: Update progress log**

```bash
cat >> logs/progress.md <<'EOF'

## 2026-04-XX — cpu-worker-optim spec DONE

- Spec: docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md (417e68b)
- Plan: docs/superpowers/plans/2026-04-17-cpu-worker-optim-implementation.md
- Results: logs/findings_cpu_worker_post_fix.md
- e2e: <pre>s -> <post>s @ res=256 (<P>%)
- Next spec: <recommendation>
EOF
```

- [ ] **Step 4: Commit final**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git add -f logs/findings_cpu_worker_post_fix.md logs/progress.md
git commit -m "$(cat <<'CMSG'
cpu-worker-optim(t10): handoff doc + progress entry

Final post-fix summary. Recommends <next spec> as next initiative.

Fixture: F1 PASS, F2 PASS, F3 PASS (final verification)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
CMSG
)"
```

- [ ] **Step 5: Verify final branch state**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git log post-profile-sync-elim ^52f5a8c --oneline
```

Expected: spec V1, plan (sync), sync T1, sync T2-7, sync progress, spec V2, plan V2, T1-T10 commits ahead of 52f5a8c.

---

## Global DoD tracker (from spec §8)

| # | Item | Verified in |
|---|---|---|
| 1 | Every commit passes F1-F3 gate | T1 creates gate; T2-T10 each run it |
| 2 | W1 landed | T2 commit |
| 3 | W2 landed; forks 124 → ≤4 | T3 Step 10 |
| 4 | W4 landed; fastpath calls drop ≥95% or self ≥80% | T5d Step 14 |
| 5 | W5 landed; UF calls drop ≥95% or self ≥80% | T6 Step 6 |
| 6 | e2e wall @ res=256 drops ≥3 s | T9 Step 3 |
| 7 | s1_voxelize.py unchanged | T9 Step 3 (verify `git diff`) |
| 8 | No new deps | T9 (verify `pip freeze` diff) |
| 9 | VRAM peak ≤ baseline + 500 MB | T9 Step 2 |
| 10 | W6 + W7 scoped (landed or skip documented) | T4 (W6 angle), T8 (W7 decision), T9 summary |
