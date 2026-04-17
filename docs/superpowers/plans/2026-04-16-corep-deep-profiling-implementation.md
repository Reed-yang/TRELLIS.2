# CoReP Deep Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the 4-layer pre-Triton bottleneck profiling described in the spec, producing annotated nsys timeline, per-stage Chrome trace, e2e top-20 GPU kernel attribution, res=256/128 scaling comparison, and ROI-ranked next-step candidate list.

**Architecture:** All instrumentation via monkey-patch in `tmp/profile_deep/` (no `corep_fast/` source modifications per `feedback_no_modify_repo.md`). All runs on 119 GPU 0 via SSH `.sh` wrappers (per project `CLAUDE.MD` + user SSH rule). Post-processing runs locally. 12 serial tasks, linear execution, no worktree / subagent parallelism needed.

**Tech Stack:** PyTorch 2.x + `torch.profiler` + `nvtx` + `torch.cuda.Event` + Nsight Systems (`nsys`) on H100, CSV / JSON / markdown post-processing.

**Spec:** `docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md`
**Baseline to compare against:** `tmp/e2e_profile_m2.py` + Phase 2 final profile (`my-docs/20260416-pre-triton-final-pass-results.md`)
**Current state:** `gpu-pipeline` @ `731a939` (Phase 2 final), res=256 e2e 9.13s, res=128 e2e 3.40s

---

## Pre-Task Setup

- [ ] **Step 0.1: Verify pre-conditions**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git branch --show-current              # expect: gpu-pipeline
git log -1 --oneline                   # expect: 82240cc docs(spec): ... deep profiling ...
ls docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md  # must exist
ls tmp/e2e_profile_m2.py               # must exist (reference driver)
ls my-docs/20260416-pre-triton-final-pass-results.md  # must exist (baseline timing)
```

Expected: all files exist, branch `gpu-pipeline`, spec already committed.

- [ ] **Step 0.2: Verify 119 GPU 0 is free**

Create probe script:

```bash
cat > tmp/probe_119.sh <<'EOF'
#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
uptime
git rev-parse HEAD
EOF
chmod +x tmp/probe_119.sh
ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/probe_119.sh'
```

Expected: GPU 0 utilization 0%, memory ~0-200 MB, `uptime` load < 2.0, HEAD matches local.

If GPU 0 is busy, STOP and tell the user — do not attempt local GPU (`feedback_profiling_on_119.md`).

---

## Task 1: Scaffolding — directory layout + icosphere builder

**Files:**
- Create: `tmp/__init__.py` (empty — needed so `tmp.profile_deep.*` imports work)
- Create: `tmp/profile_deep/__init__.py` (empty)
- Create: `tmp/profile_deep/build_icosphere.py`
- Create: `tmp/profile_deep/results/.gitkeep`

Shared deterministic icosphere builder used by `driver.py`. Uses trimesh to match Phase 2 benchmark exactly.

- [ ] **Step 1.1: Create directory + helper**

```bash
mkdir -p tmp/profile_deep/results
touch tmp/__init__.py tmp/profile_deep/__init__.py tmp/profile_deep/results/.gitkeep
```

Note: `tmp/__init__.py` is new and empty — other existing `tmp/*.py` scripts run as standalone scripts (not imported), so adding this file is non-invasive.

- [ ] **Step 1.2: Write `build_icosphere.py`**

```python
"""Deterministic icosphere subdiv-3 builder, matches Phase 2 benchmark exactly.

Reference: tmp/e2e_profile_m2.py:127 uses radius=0.4, subdivisions=3.
Result: V=642, F=1280.
"""
import trimesh


def build_icosphere_subdiv3() -> trimesh.Trimesh:
    """Build deterministic icosphere matching Phase 2 benchmark (radius=0.4)."""
    return trimesh.creation.icosphere(subdivisions=3, radius=0.4)


if __name__ == "__main__":
    m = build_icosphere_subdiv3()
    print(f"V={m.vertices.shape[0]} F={m.faces.shape[0]}")
    assert m.vertices.shape[0] == 642, f"expected V=642, got {m.vertices.shape[0]}"
    assert m.faces.shape[0] == 1280, f"expected F=1280, got {m.faces.shape[0]}"
    print("OK")
```

- [ ] **Step 1.3: Verify it builds the right mesh**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
python tmp/profile_deep/build_icosphere.py
```

Expected output:
```
V=642 F=1280
OK
```

- [ ] **Step 1.4: Commit**

```bash
git add -f tmp/profile_deep/
git commit -m "profile-deep(task1): scaffold + icosphere builder"
```

---

## Task 2: `monkeypatch_nvtx.py` — Stage-level NVTX wrapper

**Files:**
- Create: `tmp/profile_deep/monkeypatch_nvtx.py`
- Test: `tmp/profile_deep/test_monkeypatch_import.py`

Wraps each stage entry function with `torch.cuda.nvtx.range`. MUST be imported before any `corep_fast.*` import.

- [ ] **Step 2.1: Write failing import test**

```python
# tmp/profile_deep/test_monkeypatch_import.py
"""Smoke test: monkeypatch imports cleanly and patches all 7 stage entries."""
import sys


def test_patches_all_stages():
    # Clean slate
    for mod in list(sys.modules.keys()):
        if mod.startswith('corep_fast'):
            del sys.modules[mod]
    # Apply patches
    from tmp.profile_deep import monkeypatch_nvtx
    monkeypatch_nvtx.apply_stage_nvtx()
    # Verify each entry function has been wrapped
    import corep_fast.stages.s1_voxelize as s1
    import corep_fast.stages.s2_components as s2
    import corep_fast.stages.s3_edge_weights as s3
    import corep_fast.stages.s4_face_point as s4
    import corep_fast.stages.s6_collapse as s6
    import corep_fast.stages.s7_rank_assign as s7
    import corep_fast.stages.s8_collapse as s8
    for fn, expected_name in [
        (s1.s1_voxelize, "s1_voxelize"),
        (s2.s2_components, "s2_components"),
        (s3.s3_edge_weights, "s3_edge_weights"),
        (s4.s4_face_point, "s4_face_point"),
        (s6.s6_collapse, "s6_collapse"),
        (s7.s7_rank_assign, "s7_rank_assign"),
        (s8.decode_from_cubebatch, "s8_decode"),
    ]:
        assert hasattr(fn, "_nvtx_wrapped"), f"{expected_name} not wrapped"
        assert fn._nvtx_wrapped == expected_name, f"{fn._nvtx_wrapped} != {expected_name}"
    print(f"OK: 7 stages patched.")


if __name__ == "__main__":
    test_patches_all_stages()
```

- [ ] **Step 2.2: Run test to verify it fails (module not found)**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
python tmp/profile_deep/test_monkeypatch_import.py
```

Expected: `ModuleNotFoundError: No module named 'tmp.profile_deep.monkeypatch_nvtx'` OR `ImportError`.

- [ ] **Step 2.3: Write `monkeypatch_nvtx.py` stage-level wrapper**

```python
"""Monkey-patch NVTX ranges around corep_fast stage entry functions.

Must be imported BEFORE any corep_fast.* import. The `apply_*` functions are
idempotent — calling twice re-wraps the original (not the already-wrapped).

Design: wrap each stage entry, record attribute `_nvtx_wrapped = <name>` on the
patched function so the smoke test can detect successful patching.
"""
import functools
import torch
import torch.cuda.nvtx as nvtx


# (module_path, function_name, nvtx_label)
_STAGE_ENTRIES = [
    ("corep_fast.stages.s1_voxelize", "s1_voxelize", "s1_voxelize"),
    ("corep_fast.stages.s2_components", "s2_components", "s2_components"),
    ("corep_fast.stages.s3_edge_weights", "s3_edge_weights", "s3_edge_weights"),
    ("corep_fast.stages.s4_face_point", "s4_face_point", "s4_face_point"),
    ("corep_fast.stages.s6_collapse", "s6_collapse", "s6_collapse"),
    ("corep_fast.stages.s7_rank_assign", "s7_rank_assign", "s7_rank_assign"),
    ("corep_fast.stages.s8_collapse", "decode_from_cubebatch", "s8_decode"),
]


def _make_nvtx_wrapper(fn, label):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nvtx.range_push(label)
        try:
            return fn(*args, **kwargs)
        finally:
            nvtx.range_pop()
    wrapper._nvtx_wrapped = label
    wrapper._original = fn
    return wrapper


def apply_stage_nvtx():
    """Patch each stage entry function. Idempotent: re-patches original."""
    import importlib
    for mod_path, fn_name, label in _STAGE_ENTRIES:
        mod = importlib.import_module(mod_path)
        original = getattr(mod, fn_name)
        # If already wrapped, rewrap the true original to keep one level.
        if hasattr(original, "_original"):
            original = original._original
        setattr(mod, fn_name, _make_nvtx_wrapper(original, label))


# Convenience for "import activates all patches"
if __name__ != "__main__":
    # Defer: caller must explicitly call apply_stage_nvtx() (makes test easier).
    pass
```

- [ ] **Step 2.4: Run test to verify it passes**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
python tmp/profile_deep/test_monkeypatch_import.py
```

Expected: `OK: 7 stages patched.`

- [ ] **Step 2.5: Commit**

```bash
git add -f tmp/profile_deep/monkeypatch_nvtx.py tmp/profile_deep/test_monkeypatch_import.py
git commit -m "profile-deep(task2): stage-level NVTX monkey-patch + smoke test"
```

---

## Task 3: Sub-stage CUDA event injection

**Files:**
- Modify: `tmp/profile_deep/monkeypatch_nvtx.py` (add `apply_substage_events`)
- Modify: `tmp/profile_deep/test_monkeypatch_import.py` (add test)

Add CUDA event instrumentation around sub-functions in s4/s6/s7. Events accumulate into a module-level dict, driver dumps at pipeline end.

**Sub-stage targets (confirmed against `corep_fast/stages/*.py`):**

| Stage | Function | Label |
|---|---|---|
| s4 | `_compute_face_weights_gpu` (s4_face_point.py:1119) | `s4/face_weights_gpu_entry` |
| s4 | `_expand_pairs_gpu` (s4_face_point.py:817) | `s4/expand_pairs` |
| s4 | `_batch_plane_tri_with_clip` (s4_face_point.py:929) | `s4/plane_tri_clip` |
| s4 | `_compute_face_weights_mp` (s4_face_point.py:132) | `s4/face_weights_mp` |
| s4 | `_compute_component_points_gpu` (s4_face_point.py:411) | `s4/component_points` |
| s6 | `_fastpath_gpu_build_adjacency` (s6_collapse.py:190) | `s6/fastpath_adj_build` |
| s6 | `_fastpath_trace_loops_numpy` (s6_collapse.py:379) | `s6/fastpath_trace` |
| s6 | `_collapse_with_uturns` (s6_collapse.py:569) | `s6/slowpath_uturns` |
| s6 | `_collapse_with_uturns_tracked` (s6_collapse.py:652) | `s6/slowpath_tracked` |
| s7 | `_build_adjacency_gpu` (s7_rank_assign.py:634) | `s7/adj_build_gpu` |
| s7 | `_phase1_gpu_rank_assign` (s7_rank_assign.py:903) | `s7/phase1_gpu_bfs` |

- [ ] **Step 3.1: Extend test with sub-stage assertion**

Append to `tmp/profile_deep/test_monkeypatch_import.py`:

```python
def test_substage_events_register():
    # Clean import state
    for mod in list(sys.modules.keys()):
        if mod.startswith('corep_fast'):
            del sys.modules[mod]
    from tmp.profile_deep import monkeypatch_nvtx
    monkeypatch_nvtx.apply_stage_nvtx()
    monkeypatch_nvtx.apply_substage_events()
    assert len(monkeypatch_nvtx._SUBSTAGE_TIMINGS) == 0, "should start empty"
    # Verify a known sub-stage fn is patched
    import corep_fast.stages.s4_face_point as s4
    assert hasattr(s4._compute_face_weights_gpu, "_event_wrapped")
    import corep_fast.stages.s6_collapse as s6
    assert hasattr(s6._fastpath_gpu_build_adjacency, "_event_wrapped")
    import corep_fast.stages.s7_rank_assign as s7
    assert hasattr(s7._phase1_gpu_rank_assign, "_event_wrapped")
    print("OK: sub-stage events registered.")


if __name__ == "__main__":
    test_patches_all_stages()
    test_substage_events_register()
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
python tmp/profile_deep/test_monkeypatch_import.py
```

Expected: first test passes, second fails with `AttributeError: module 'tmp.profile_deep.monkeypatch_nvtx' has no attribute 'apply_substage_events'`.

- [ ] **Step 3.3: Extend `monkeypatch_nvtx.py` with sub-stage events**

Append to `tmp/profile_deep/monkeypatch_nvtx.py`:

```python
# Sub-stage CUDA event targets: (module_path, fn_name, label)
_SUBSTAGE_ENTRIES = [
    ("corep_fast.stages.s4_face_point", "_compute_face_weights_gpu", "s4/face_weights_gpu_entry"),
    ("corep_fast.stages.s4_face_point", "_expand_pairs_gpu",         "s4/expand_pairs"),
    ("corep_fast.stages.s4_face_point", "_batch_plane_tri_with_clip","s4/plane_tri_clip"),
    ("corep_fast.stages.s4_face_point", "_compute_face_weights_mp",  "s4/face_weights_mp"),
    ("corep_fast.stages.s4_face_point", "_compute_component_points_gpu", "s4/component_points"),
    ("corep_fast.stages.s6_collapse",   "_fastpath_gpu_build_adjacency", "s6/fastpath_adj_build"),
    ("corep_fast.stages.s6_collapse",   "_fastpath_trace_loops_numpy",   "s6/fastpath_trace"),
    ("corep_fast.stages.s6_collapse",   "_collapse_with_uturns",         "s6/slowpath_uturns"),
    ("corep_fast.stages.s6_collapse",   "_collapse_with_uturns_tracked", "s6/slowpath_tracked"),
    ("corep_fast.stages.s7_rank_assign", "_build_adjacency_gpu",     "s7/adj_build_gpu"),
    ("corep_fast.stages.s7_rank_assign", "_phase1_gpu_rank_assign",  "s7/phase1_gpu_bfs"),
]


# Accumulator: label -> list of ms measurements (multiple calls per run possible)
_SUBSTAGE_TIMINGS: dict[str, list[float]] = {}


def _make_event_wrapper(fn, label):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            end.record()
            # Defer sync — accumulate events, sync + compute ms happens at dump.
            _SUBSTAGE_TIMINGS.setdefault(label, []).append((start, end))
    wrapper._event_wrapped = label
    wrapper._original = fn
    return wrapper


def apply_substage_events():
    """Patch sub-stage functions with CUDA event timing."""
    import importlib
    for mod_path, fn_name, label in _SUBSTAGE_ENTRIES:
        mod = importlib.import_module(mod_path)
        original = getattr(mod, fn_name)
        if hasattr(original, "_original"):
            original = original._original
        setattr(mod, fn_name, _make_event_wrapper(original, label))


def dump_substage_timings() -> dict[str, dict]:
    """Sync CUDA, compute elapsed ms for each (start, end) pair, return {label: {count, total_ms, per_call_ms: [...]}}."""
    torch.cuda.synchronize()
    out: dict[str, dict] = {}
    for label, pairs in _SUBSTAGE_TIMINGS.items():
        per_call = [s.elapsed_time(e) for s, e in pairs]
        out[label] = {
            "count": len(per_call),
            "total_ms": sum(per_call),
            "per_call_ms": per_call,
        }
    return out
```

- [ ] **Step 3.4: Run test to verify it passes**

```bash
python tmp/profile_deep/test_monkeypatch_import.py
```

Expected: both tests OK.

- [ ] **Step 3.5: Commit**

```bash
git add tmp/profile_deep/monkeypatch_nvtx.py tmp/profile_deep/test_monkeypatch_import.py
git commit -m "profile-deep(task3): add sub-stage CUDA event injection"
```

---

## Task 4: `driver.py` — unified e2e driver

**Files:**
- Create: `tmp/profile_deep/driver.py`

Unified driver that can be invoked in 3 modes (layer 0 = naked run for nsys to wrap; layer 1 / 3 = torch.profiler context). Emits timing JSON + Chrome trace (when applicable).

**Reference pipeline call pattern:** `tmp/e2e_profile_m2.py` — study it to get the exact `corep_fast` pipeline invocation (stages to call, argument order, device placement).

- [ ] **Step 4.1: Study reference driver**

```bash
cat tmp/e2e_profile_m2.py | grep -E "def |import corep_fast|s[12346789]_"
```

Expected: lists imports and stage function calls; note the order + arguments to replicate in driver.py.

- [ ] **Step 4.2: Write `driver.py`**

```python
"""CoReP deep profiling driver — layer 0 (nsys-wrapped), layer 1/3 (torch.profiler).

Usage:
    python tmp/profile_deep/driver.py --layer 1 --res 256
    nsys profile ... python tmp/profile_deep/driver.py --layer 0 --res 256

Pipeline call pattern mirrors tmp/e2e_profile_m2.py (verify before running).
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# MUST patch before any corep_fast import
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tmp.profile_deep import monkeypatch_nvtx
monkeypatch_nvtx.apply_stage_nvtx()
monkeypatch_nvtx.apply_substage_events()

import numpy as np
import torch

from tmp.profile_deep.build_icosphere import build_icosphere_subdiv3


RESULTS_DIR = REPO_ROOT / "tmp/profile_deep/results"


def set_deterministic():
    torch.manual_seed(42)
    np.random.seed(42)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def run_pipeline(mesh, res: int, device: torch.device) -> dict:
    """Run the full corep_fast pipeline end-to-end, return per-stage wall-time.

    Call pattern mirrors tmp/e2e_profile_m2.py verbatim. NVTX ranges (from
    monkeypatch) mark each stage; we still record explicit per-stage walltime
    for the summary JSON.
    """
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign
    from corep_fast.stages.s8_collapse import decode_from_cubebatch

    t = {}

    gc.collect(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    mt = MeshTensors.from_trimesh(mesh, res, device=device)
    torch.cuda.synchronize()
    t["load"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s1_voxelize(mt, res, device)
    torch.cuda.synchronize(); t["s1"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s2_components(batch, mt)
    torch.cuda.synchronize(); t["s2"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s3_edge_weights(batch, mt)
    torch.cuda.synchronize(); t["s3"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s4_face_point(batch, mt)
    torch.cuda.synchronize(); t["s4"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s6_collapse(batch)
    torch.cuda.synchronize(); t["s6"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s7_rank_assign(batch)
    torch.cuda.synchronize(); t["s7"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    v, f = decode_from_cubebatch(batch, merge_decimals=5)
    torch.cuda.synchronize(); t["s8"] = time.perf_counter() - t0

    t["s1_s7"] = sum(t[k] for k in ("s1", "s2", "s3", "s4", "s6", "s7"))
    t["e2e"] = t["load"] + t["s1_s7"] + t["s8"]
    t["_cubes"] = batch.num_cubes
    t["_V"] = int(v.shape[0])
    t["_F"] = int(f.shape[0])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["0", "1", "3"], required=True,
                    help="0 = naked run (nsys wraps externally); 1/3 = torch.profiler")
    ap.add_argument("--res", type=int, required=True)
    args = ap.parse_args()

    set_deterministic()
    mesh = build_icosphere_subdiv3()
    device = torch.device("cuda:0")

    # Warmup
    print(f"[warmup] pipeline at res={args.res}...")
    gc.collect()
    torch.cuda.empty_cache()
    _ = run_pipeline(mesh, args.res, device)
    torch.cuda.synchronize()

    # Reset accumulators (discard warmup events)
    monkeypatch_nvtx._SUBSTAGE_TIMINGS.clear()
    gc.collect()
    torch.cuda.empty_cache()

    out_prefix = RESULTS_DIR / f"layer{args.layer}_res{args.res}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    header = {
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nvidia_smi": nvidia_smi_snapshot(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "layer": args.layer,
        "resolution": args.res,
    }

    print(f"[measure] layer={args.layer} res={args.res}...")
    if args.layer == "0":
        t_stages = run_pipeline(mesh, args.res, device)
    else:
        trace_path = str(out_prefix) + "_trace.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
            on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        ) as prof:
            t_stages = run_pipeline(mesh, args.res, device)
            prof.step()
        print(f"[write] {trace_path}")

    substage = monkeypatch_nvtx.dump_substage_timings()
    summary_path = str(out_prefix) + "_summary.json"
    with open(summary_path, "w") as f:
        # Serialize: drop per_call_ms lists (keep count + total) to stay small.
        serializable_sub = {
            k: {"count": v["count"], "total_ms": v["total_ms"]}
            for k, v in substage.items()
        }
        json.dump({
            "header": header,
            "stage_walltime_sec": t_stages,
            "substage_ms": serializable_sub,
        }, f, indent=2)
    print(f"[write] {summary_path}")
    print(f"[done] e2e = {t_stages.get('e2e', -1):.3f}s")


if __name__ == "__main__":
    main()
```

**Critical caveat**: the `run_pipeline` function is a skeleton. Before committing, the implementer MUST open `tmp/e2e_profile_m2.py` and copy the exact stage invocation sequence into `run_pipeline`. The imports for `corep_fast_mesh_to_ply` and `MeshTensors` above are placeholders — use whatever the reference driver uses.

- [ ] **Step 4.3: Quick smoke-run at res=64 (fast) locally-but-on-119 to verify driver runs**

```bash
cat > tmp/smoke_driver_res64.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
python tmp/profile_deep/driver.py --layer 1 --res 64
EOF
chmod +x tmp/smoke_driver_res64.sh
ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/smoke_driver_res64.sh' 2>&1 | tail -30
```

Expected: completes in < 30s, prints `[done] e2e = X.XXXs`, writes `layer1_res64_trace.json` + `layer1_res64_summary.json` to `tmp/profile_deep/results/`.

- [ ] **Step 4.4: Commit**

```bash
git add -f tmp/profile_deep/driver.py tmp/smoke_driver_res64.sh
git commit -m "profile-deep(task4): e2e driver + res=64 smoke run OK"
```

---

## Task 5: Smoke check — verify monkey-patch overhead < 10%

**Files:**
- Create: `tmp/run_smoke_overhead.sh`
- Create: `tmp/profile_deep/smoke_compare.py`

Compare walltime of patched pipeline vs. unpatched pipeline at res=128 (fast, already has Phase 2 baseline = 3.40s). Decision gate: if delta > 10%, fall back to NVTX-only (skip sub-stage events).

- [ ] **Step 5.1: Write `smoke_compare.py`**

```python
"""Run the pipeline patched and unpatched, compare walltime."""
import gc
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from tmp.profile_deep.build_icosphere import build_icosphere_subdiv3


def _pipeline_once(mesh, res: int, device: torch.device):
    """Single pipeline run, no timing. Caller handles timing + sync."""
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign
    from corep_fast.stages.s8_collapse import decode_from_cubebatch

    mt = MeshTensors.from_trimesh(mesh, res, device=device)
    batch = s1_voxelize(mt, res, device)
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    batch = s6_collapse(batch)
    batch = s7_rank_assign(batch)
    return decode_from_cubebatch(batch, merge_decimals=5)


def _time_reps(mesh, res: int, reps: int, device: torch.device) -> list[float]:
    # Warmup
    _pipeline_once(mesh, res, device)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        _pipeline_once(mesh, res, device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def run_unpatched(mesh, res: int, reps: int = 3) -> list[float]:
    device = torch.device("cuda:0")
    return _time_reps(mesh, res, reps, device)


def run_patched(mesh, res: int, reps: int = 3) -> list[float]:
    from tmp.profile_deep import monkeypatch_nvtx
    monkeypatch_nvtx.apply_stage_nvtx()
    monkeypatch_nvtx.apply_substage_events()
    device = torch.device("cuda:0")
    return _time_reps(mesh, res, reps, device)


def main():
    mesh = build_icosphere_subdiv3()
    res = 128

    # Run in separate subprocess calls so each starts fresh:
    # Call 1: unpatched; Call 2: patched. Driver below does one per call via env var.
    mode = sys.argv[1]  # "patched" or "unpatched"
    times = run_patched(mesh, res) if mode == "patched" else run_unpatched(mesh, res)
    import statistics
    print(f"{mode} median={statistics.median(times):.3f}s  all={times}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Write SSH wrapper that runs both modes**

```bash
cat > tmp/run_smoke_overhead.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
echo "=== UNPATCHED ==="
python tmp/profile_deep/smoke_compare.py unpatched
echo "=== PATCHED ==="
python tmp/profile_deep/smoke_compare.py patched
EOF
chmod +x tmp/run_smoke_overhead.sh
```

- [ ] **Step 5.3: Run the smoke check on 119**

```bash
ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/run_smoke_overhead.sh' 2>&1 | tee tmp/profile_deep/results/smoke_overhead.log
```

Expected output format:
```
=== UNPATCHED ===
unpatched median=3.4XXs  all=[3.4, 3.4, 3.4]
=== PATCHED ===
patched median=3.5XXs  all=[3.5, 3.5, 3.5]
```

- [ ] **Step 5.4: Decision gate**

Compute `(patched - unpatched) / unpatched`:
- If < 10%: proceed normally with both stage + sub-stage patches.
- If ≥ 10%: **document in `tmp/profile_deep/results/smoke_overhead.log`** that fallback is triggered. Modify driver to call only `apply_stage_nvtx()`, not `apply_substage_events()`. All Layer 1/3 runs use NVTX-only; sub-stage timings are forfeited.

Append the decision to the log:

```bash
echo "Decision: delta=X.X%, fallback=[yes|no]" >> tmp/profile_deep/results/smoke_overhead.log
```

- [ ] **Step 5.5: Commit**

```bash
git add -f tmp/profile_deep/smoke_compare.py tmp/run_smoke_overhead.sh tmp/profile_deep/results/smoke_overhead.log
git commit -m "profile-deep(task5): smoke check — overhead X.X% ($([ X% < 10% ] && echo OK || echo FALLBACK))"
```

---

## Task 6: Layer 0 — Nsight Systems run @ res=256

**Files:**
- Create: `tmp/run_layer0_nsys.sh`
- Output: `tmp/profile_deep/results/nsys_res256_full.nsys-rep`
- Output: `tmp/profile_deep/results/nsys_res256_stats.txt`

- [ ] **Step 6.1: Verify nsys is on 119**

```bash
ssh host-10-240-99-119 'which nsys && nsys --version'
```

Expected: `nsys` found at path, version 2023+ preferred. If not found, STOP and ask the user.

- [ ] **Step 6.2: Write nsys wrapper**

```bash
cat > tmp/run_layer0_nsys.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
OUT=/mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/profile_deep/results
mkdir -p "$OUT"

# Run nsys; --nvtx captures our stage ranges. --cuda-memory-usage adds sync points
# (small overhead, worth it for d2h/h2d visibility).
nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --cuda-memory-usage=true \
    --sample=cpu \
    --python-backtrace=cuda \
    --force-overwrite=true \
    --output="$OUT/nsys_res256_full" \
    python tmp/profile_deep/driver.py --layer 0 --res 256

# Post: stats dump (text-friendly)
nsys stats --format csv --output "$OUT/nsys_res256_stats" "$OUT/nsys_res256_full.nsys-rep" \
    > "$OUT/nsys_res256_stats.txt" 2>&1 || \
    nsys stats "$OUT/nsys_res256_full.nsys-rep" > "$OUT/nsys_res256_stats.txt" 2>&1

echo "nsys output:"
ls -lh "$OUT"/nsys_res256*
EOF
chmod +x tmp/run_layer0_nsys.sh
```

- [ ] **Step 6.3: Run nsys on 119**

```bash
ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/run_layer0_nsys.sh' 2>&1 | tee tmp/profile_deep/results/nsys_res256_runlog.txt
```

Expected: completes in 20-30s walltime (nsys slowdown 2-3x over 9.13s baseline); produces `.nsys-rep` file (100MB-1GB) + `stats.txt`.

- [ ] **Step 6.4: Extract key observations from stats**

Read `tmp/profile_deep/results/nsys_res256_stats.txt` — look for:
1. **"CUDA API Summary"** — kernel launch count + total time
2. **"CUDA Kernel Summary"** — top kernels by time
3. **"NVTX Range Summary"** — confirms our 7 stage ranges appear with expected walltime
4. **"CUDA Memory Operations Summary"** — h2d / d2h volume and count

Write a short observations file:

```bash
cat > tmp/profile_deep/results/layer0_observations.md <<'EOF'
# Layer 0 — Nsight Systems observations (res=256)

## Run info
- git sha: $(git rev-parse HEAD)
- Date: $(date -u +%Y-%m-%dT%H:%M:%S)

## NVTX stage ranges (sanity)
| Stage | Wall ms | # launches |
|---|---:|---:|
| s1_voxelize | ? | ? |
| s2_components | ? | ? |
| ... |

## Top 10 CUDA kernels by total time
| Rank | Name | Total ms | Count | Avg µs |
|---|---|---:|---:|---:|
| 1 | ? | ? | ? | ? |

## Memory ops
- h2d: ? GB in ? calls
- d2h: ? GB in ? calls

## 3 concrete observations (DoD #2)
1. GPU idle window: [describe — when, how long, between which stages]
2. Stream serialization: [describe — single-stream? cross-stream sync points?]
3. CPU sync point: [where is an explicit sync that blocks GPU? how long?]
EOF
```

Fill in the `?` cells from `nsys_res256_stats.txt`.

- [ ] **Step 6.5: Commit**

```bash
git add -f tmp/run_layer0_nsys.sh tmp/profile_deep/results/nsys_res256_stats.txt \
    tmp/profile_deep/results/nsys_res256_runlog.txt \
    tmp/profile_deep/results/layer0_observations.md
# Do NOT commit the .nsys-rep binary file (can be 100MB-1GB); add to .gitignore if size > 100MB.
git commit -m "profile-deep(task6): Layer 0 nsys run @ res=256 + observations"
```

Note: `.nsys-rep` is kept locally for later re-analysis if needed, not committed.

---

## Task 7: Layer 1 — torch.profiler run @ res=256

**Files:**
- Create: `tmp/run_layer1_torch.sh`
- Output: `tmp/profile_deep/results/layer1_res256_trace.json`
- Output: `tmp/profile_deep/results/layer1_res256_summary.json`

- [ ] **Step 7.1: Write SSH wrapper**

```bash
cat > tmp/run_layer1_torch.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
python tmp/profile_deep/driver.py --layer 1 --res 256
EOF
chmod +x tmp/run_layer1_torch.sh
```

- [ ] **Step 7.2: Run on 119 (3 runs for median)**

```bash
for i in 1 2 3; do
    echo "=== run $i ===" | tee -a tmp/profile_deep/results/layer1_res256_runlog.txt
    ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/run_layer1_torch.sh' 2>&1 \
        | tee -a tmp/profile_deep/results/layer1_res256_runlog.txt
    # Move this run's output to a unique name
    mv tmp/profile_deep/results/layer1_res256_trace.json \
       tmp/profile_deep/results/layer1_res256_run${i}_trace.json
    mv tmp/profile_deep/results/layer1_res256_summary.json \
       tmp/profile_deep/results/layer1_res256_run${i}_summary.json
done
```

Note: `mv` commands run locally because the output files are on the shared filesystem (119 and local see the same paths per existing project setup). If not shared, scp first.

- [ ] **Step 7.3: Pick median run by e2e**

```bash
python -c "
import json, glob
summaries = []
for p in sorted(glob.glob('tmp/profile_deep/results/layer1_res256_run*_summary.json')):
    d = json.load(open(p))
    summaries.append((p, d['stage_walltime_sec'].get('e2e', 0)))
for p, e in summaries: print(p, e)
summaries.sort(key=lambda x: x[1])
median_path = summaries[len(summaries)//2][0]
print('median:', median_path)
"
```

Record which run is the median. Use that trace for Layer 1 analysis.

- [ ] **Step 7.4: Commit**

```bash
git add -f tmp/run_layer1_torch.sh \
    tmp/profile_deep/results/layer1_res256_run*_summary.json \
    tmp/profile_deep/results/layer1_res256_runlog.txt
# Chrome traces (.json) can be large — add them too; if > 100MB each, gzip first.
git add -f tmp/profile_deep/results/layer1_res256_run*_trace.json
git commit -m "profile-deep(task7): Layer 1 torch.profiler @ res=256, 3 runs"
```

---

## Task 8: `analyze_layer1_trace.py` — per-stage op CSV

**Files:**
- Create: `tmp/profile_deep/analyze_layer1_trace.py`
- Output: `tmp/profile_deep/results/per_stage_ops_res256.csv`

Parse Chrome trace JSON, filter events by NVTX range, emit per-stage top-30 ops.

- [ ] **Step 8.1: Write analyzer**

```python
"""Parse torch.profiler Chrome trace, emit per-stage top-N op CSV.

Usage:
    python tmp/profile_deep/analyze_layer1_trace.py \
        tmp/profile_deep/results/layer1_res256_run2_trace.json \
        tmp/profile_deep/results/per_stage_ops_res256.csv

Filter logic:
- NVTX range events define stage boundaries (events with cat containing 'user_annotation'
  and name matching one of our 7 stage labels).
- Each op/kernel event is attributed to a stage by its ts (timestamp) falling inside
  that stage's NVTX range.
- Aggregate per (stage, op_name): count, total_device_us, total_cpu_us.
- Emit top-30 per stage by total_device_us.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


STAGE_LABELS = {
    "s1_voxelize", "s2_components", "s3_edge_weights", "s4_face_point",
    "s6_collapse", "s7_rank_assign", "s8_decode",
}


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def extract_stage_ranges(events: list[dict]) -> list[tuple[str, int, int]]:
    """Return [(stage_label, ts_start, ts_end)] from NVTX complete events."""
    ranges = []
    for ev in events:
        # torch.profiler NVTX ranges show up as ph='X' with cat='user_annotation'
        if ev.get("ph") == "X" and ev.get("name") in STAGE_LABELS:
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            ranges.append((ev["name"], ts, ts + dur))
    return ranges


def find_stage(ts: int, ranges: list[tuple[str, int, int]]) -> str:
    for name, s, e in ranges:
        if s <= ts <= e:
            return name
    return "__unassigned__"


def main():
    trace_path = sys.argv[1]
    out_csv = sys.argv[2]

    events = load_trace(trace_path)
    ranges = extract_stage_ranges(events)
    print(f"[info] found {len(ranges)} stage ranges: {[r[0] for r in ranges]}")

    # Aggregate per (stage, op_name).
    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "cpu_us": 0.0, "device_us": 0.0}
    )
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if name in STAGE_LABELS:
            continue  # skip the stage NVTX events themselves
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        stage = find_stage(ts, ranges)
        key = (stage, name)
        agg[key]["count"] += 1
        cat = ev.get("cat", "")
        # Heuristic: 'kernel' category = device time; everything else = cpu time.
        if "kernel" in cat.lower() or ev.get("args", {}).get("device") is not None:
            agg[key]["device_us"] += dur
        else:
            agg[key]["cpu_us"] += dur

    # Per-stage top-30 by device_us (fallback to cpu_us if all zero).
    per_stage: dict[str, list] = defaultdict(list)
    for (stage, op), stats in agg.items():
        per_stage[stage].append((op, stats["count"], stats["cpu_us"], stats["device_us"]))
    for stage, rows in per_stage.items():
        rows.sort(key=lambda r: (r[3], r[2]), reverse=True)  # device first
        per_stage[stage] = rows[:30]

    # Write CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "op_name", "count", "cpu_us", "device_us"])
        for stage in sorted(per_stage.keys()):
            for op, count, cpu_us, dev_us in per_stage[stage]:
                w.writerow([stage, op, count, f"{cpu_us:.1f}", f"{dev_us:.1f}"])
    print(f"[write] {out_csv}  ({sum(len(r) for r in per_stage.values())} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Run on the median Layer 1 trace**

```bash
MEDIAN_TRACE=$(ls tmp/profile_deep/results/layer1_res256_run*_trace.json | head -1)  # replace with actual median from Task 7
python tmp/profile_deep/analyze_layer1_trace.py \
    "$MEDIAN_TRACE" \
    tmp/profile_deep/results/per_stage_ops_res256.csv
```

- [ ] **Step 8.3: Sanity-check output**

```bash
# Must have 8 stages (7 NVTX + __unassigned__) OR 7 (if all assigned)
cut -d, -f1 tmp/profile_deep/results/per_stage_ops_res256.csv | sort -u
# Must have ~30 rows per stage
cut -d, -f1 tmp/profile_deep/results/per_stage_ops_res256.csv | sort | uniq -c
# Top row per stage should be GPU-time heavy
head -50 tmp/profile_deep/results/per_stage_ops_res256.csv
```

Expected: 7 stage labels plus possibly `__unassigned__` (whose rows, if any, should be dominated by setup/teardown overhead and sum to < 5% of e2e).

- [ ] **Step 8.4: Commit**

```bash
git add -f tmp/profile_deep/analyze_layer1_trace.py \
    tmp/profile_deep/results/per_stage_ops_res256.csv
git commit -m "profile-deep(task8): analyze_layer1_trace — per-stage top-30 ops CSV"
```

---

## Task 9: `analyze_layer2_kernels.py` — top-20 kernel + heuristic classifier

**Files:**
- Create: `tmp/profile_deep/analyze_layer2_kernels.py`
- Output: `tmp/profile_deep/results/top20_kernels_res256.csv`

- [ ] **Step 9.1: Write analyzer with heuristic classifier**

```python
"""Aggregate GPU kernels across all stages from torch.profiler trace, emit top-20
with (stage, py-line, heuristic class) attribution.

Heuristic classes:
- LNB: launch-bound   — mean_us < 10 AND count > 1000
- MMB: memory-bound   — name matches {copy, memset, scatter, gather, index, cat, slice}
- CMB: compute-bound  — name matches {gemm, conv, reduce, sum, matmul} AND mean_us > 100
- CPU: CPU-bound      — stage-level time minus sum(stage GPU kernels) > 30% stage wall
- UNK: unknown        — couldn't be classified; needs ncu

Usage:
    python tmp/profile_deep/analyze_layer2_kernels.py \
        tmp/profile_deep/results/layer1_res256_run2_trace.json \
        tmp/profile_deep/results/top20_kernels_res256.csv
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


STAGE_LABELS = {
    "s1_voxelize", "s2_components", "s3_edge_weights", "s4_face_point",
    "s6_collapse", "s7_rank_assign", "s8_decode",
}


MMB_PATTERNS = re.compile(
    r"(copy|memset|scatter|gather|^index|cat|slice|contiguous|view|as_strided)",
    re.IGNORECASE,
)
CMB_PATTERNS = re.compile(r"(gemm|conv|reduce|sum|matmul|bmm)", re.IGNORECASE)


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def extract_stage_ranges(events):
    ranges = []
    for ev in events:
        if ev.get("ph") == "X" and ev.get("name") in STAGE_LABELS:
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            ranges.append((ev["name"], ts, ts + dur, dur))
    return ranges


def find_stage(ts, ranges):
    for name, s, e, _ in ranges:
        if s <= ts <= e:
            return name
    return "__unassigned__"


def extract_py_line(ev) -> str:
    """Best-effort: pull py-stack top frame from event args."""
    stack = ev.get("args", {}).get("Call stack", "")
    if not stack:
        return ""
    # Stack is usually newline-delimited; find first line matching corep_fast
    for line in stack.split("\n"):
        if "corep_fast" in line:
            return line.strip()
    return stack.split("\n")[0].strip()


def classify(name: str, mean_us: float, count: int, stage_wall_us: float,
             stage_gpu_us: float) -> str:
    # CPU-bound check is per-stage, caller supplies stage-level numbers.
    if stage_wall_us > 0 and (stage_wall_us - stage_gpu_us) / stage_wall_us > 0.30:
        # CPU-bound classification applied at stage level, not kernel level.
        # But if this specific kernel is small and the stage is CPU-heavy, flag CPU.
        if mean_us < 50:
            return "CPU"
    if MMB_PATTERNS.search(name):
        return "MMB"
    if CMB_PATTERNS.search(name) and mean_us > 100:
        return "CMB"
    if mean_us < 10 and count > 1000:
        return "LNB"
    return "UNK"


def main():
    trace_path = sys.argv[1]
    out_csv = sys.argv[2]

    events = load_trace(trace_path)
    ranges = extract_stage_ranges(events)

    # stage -> total wall time (µs)
    stage_wall = {name: dur for name, _, _, dur in ranges}
    # stage -> total GPU kernel time (µs)
    stage_gpu: dict[str, float] = defaultdict(float)

    # (stage, kernel_name) -> stats
    kern: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "py_line": ""}
    )
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "").lower()
        if "kernel" not in cat:
            continue
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        name = ev.get("name", "<anon>")
        stage = find_stage(ts, ranges)
        kern[(stage, name)]["count"] += 1
        kern[(stage, name)]["total_us"] += dur
        if not kern[(stage, name)]["py_line"]:
            kern[(stage, name)]["py_line"] = extract_py_line(ev)
        stage_gpu[stage] += dur

    # Flatten + sort by total_us, take top 20.
    rows = []
    for (stage, name), s in kern.items():
        mean = s["total_us"] / max(1, s["count"])
        cls = classify(name, mean, s["count"],
                        stage_wall.get(stage, 0), stage_gpu.get(stage, 0))
        rows.append({
            "stage": stage,
            "kernel": name,
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
        w = csv.DictWriter(f, fieldnames=["stage", "kernel", "count", "total_ms",
                                           "mean_us", "class", "py_line"])
        w.writeheader()
        for r in top20:
            w.writerow(r)
    print(f"[write] {out_csv}")

    unk_count = sum(1 for r in top20 if r["class"] == "UNK")
    print(f"UNK rate: {unk_count}/20 = {100*unk_count/20:.0f}%")
    if unk_count > 4:  # 20% threshold per DoD
        print("[WARN] UNK rate > 20% — classifier may need tuning, flag for review.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Run on median trace**

```bash
MEDIAN_TRACE=$(ls tmp/profile_deep/results/layer1_res256_run*_trace.json | head -1)
python tmp/profile_deep/analyze_layer2_kernels.py \
    "$MEDIAN_TRACE" \
    tmp/profile_deep/results/top20_kernels_res256.csv
```

- [ ] **Step 9.3: Review UNK rate (DoD #4: ≤ 20%)**

```bash
awk -F, 'NR>1 {print $6}' tmp/profile_deep/results/top20_kernels_res256.csv | sort | uniq -c
```

Expected: class distribution; `UNK` count ≤ 4 out of 20. If higher, add a note to results doc + consider extending `MMB_PATTERNS`/`CMB_PATTERNS`.

- [ ] **Step 9.4: Commit**

```bash
git add -f tmp/profile_deep/analyze_layer2_kernels.py \
    tmp/profile_deep/results/top20_kernels_res256.csv
git commit -m "profile-deep(task9): analyze_layer2_kernels — top-20 kernel + heuristic class"
```

---

## Task 10: Layer 3 — res=128 run + scaling analysis

**Files:**
- Create: `tmp/run_layer3_torch.sh`
- Create: `tmp/profile_deep/analyze_layer3_scaling.py`
- Output: `tmp/profile_deep/results/per_stage_ops_res128.csv`
- Output: `tmp/profile_deep/results/scaling_table.csv`

- [ ] **Step 10.1: Write SSH wrapper**

```bash
cat > tmp/run_layer3_torch.sh <<'EOF'
#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
python tmp/profile_deep/driver.py --layer 3 --res 128
EOF
chmod +x tmp/run_layer3_torch.sh
```

- [ ] **Step 10.2: Run on 119 (3 runs for median)**

```bash
for i in 1 2 3; do
    echo "=== run $i ===" | tee -a tmp/profile_deep/results/layer3_res128_runlog.txt
    ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/run_layer3_torch.sh' 2>&1 \
        | tee -a tmp/profile_deep/results/layer3_res128_runlog.txt
    mv tmp/profile_deep/results/layer3_res128_trace.json \
       tmp/profile_deep/results/layer3_res128_run${i}_trace.json
    mv tmp/profile_deep/results/layer3_res128_summary.json \
       tmp/profile_deep/results/layer3_res128_run${i}_summary.json
done
```

- [ ] **Step 10.3: Pick median and run Layer-1 analyzer on it (reuse)**

```bash
python -c "
import json, glob
summaries = []
for p in sorted(glob.glob('tmp/profile_deep/results/layer3_res128_run*_summary.json')):
    d = json.load(open(p))
    summaries.append((p, d['stage_walltime_sec'].get('e2e', 0)))
summaries.sort(key=lambda x: x[1])
print('median:', summaries[len(summaries)//2][0])
"
# Replace "runN" with the median below:
python tmp/profile_deep/analyze_layer1_trace.py \
    tmp/profile_deep/results/layer3_res128_run2_trace.json \
    tmp/profile_deep/results/per_stage_ops_res128.csv
```

- [ ] **Step 10.4: Write `analyze_layer3_scaling.py`**

```python
"""Compare per-stage ops between res=256 and res=128, emit scaling table.

Usage:
    python tmp/profile_deep/analyze_layer3_scaling.py \
        tmp/profile_deep/results/per_stage_ops_res256.csv \
        tmp/profile_deep/results/per_stage_ops_res128.csv \
        tmp/profile_deep/results/scaling_table.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_stage_totals(csv_path: str) -> dict[str, float]:
    """Sum device_us per stage."""
    totals: dict[str, float] = defaultdict(float)
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            totals[row["stage"]] += float(row["device_us"])
    return dict(totals)


def main():
    hi_csv, lo_csv, out_csv = sys.argv[1:4]
    hi = load_stage_totals(hi_csv)  # res=256
    lo = load_stage_totals(lo_csv)  # res=128

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "device_us_res256", "device_us_res128", "ratio", "flag"])
        for stage in sorted(set(hi) | set(lo)):
            h = hi.get(stage, 0)
            l = lo.get(stage, 0)
            ratio = h / l if l > 0 else float("inf")
            if l == 0:
                flag = "no_res128_data"
            elif ratio < 2:
                flag = "below_2x_fixed_cost_dominant"
            elif ratio > 10:
                flag = "above_10x_superlinear"
            else:
                flag = "ok_2x_to_10x"
            w.writerow([stage, f"{h:.1f}", f"{l:.1f}", f"{ratio:.2f}", flag])
    print(f"[write] {out_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.5: Run scaling analysis**

```bash
python tmp/profile_deep/analyze_layer3_scaling.py \
    tmp/profile_deep/results/per_stage_ops_res256.csv \
    tmp/profile_deep/results/per_stage_ops_res128.csv \
    tmp/profile_deep/results/scaling_table.csv
cat tmp/profile_deep/results/scaling_table.csv
```

Expected output: every stage row shows ratio + flag. Any stage with flag ≠ `ok_2x_to_10x` needs a written hypothesis in the results doc (DoD #5).

- [ ] **Step 10.6: Commit**

```bash
git add -f tmp/run_layer3_torch.sh \
    tmp/profile_deep/analyze_layer3_scaling.py \
    tmp/profile_deep/results/layer3_res128_run*_summary.json \
    tmp/profile_deep/results/layer3_res128_run*_trace.json \
    tmp/profile_deep/results/layer3_res128_runlog.txt \
    tmp/profile_deep/results/per_stage_ops_res128.csv \
    tmp/profile_deep/results/scaling_table.csv
git commit -m "profile-deep(task10): Layer 3 res=128 run + scaling table"
```

---

## Task 11: Results document + ROI candidate list

**Files:**
- Create: `my-docs/20260416-corep-deep-profiling-results.md`

Aggregate Layer 0/1/2/3 into one document. Produces the ROI-ranked list — **the primary deliverable**.

- [ ] **Step 11.1: Draft results doc skeleton**

```bash
cat > my-docs/20260416-corep-deep-profiling-results.md <<'EOF'
# CoReP Deep Profiling — Results

> Date: 2026-04-16
> Branch: gpu-pipeline @ <git sha>
> Hardware: 119 H100, GPU 0
> Spec: docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md
> Plan: docs/superpowers/plans/2026-04-16-corep-deep-profiling-implementation.md
> Baseline: Phase 2 final, res=256 e2e 9.13s (vs custom 141.05s = 15.46x)

---

## TL;DR

<1-2 paragraphs: top 3 findings + immediate next-step recommendation>

---

## Layer 0 — Nsight Systems timeline observations

<Paste key sections from tmp/profile_deep/results/layer0_observations.md>

Key observations (DoD #2, ≥3 required):
1. …
2. …
3. …

---

## Layer 1 — Per-stage op breakdown @ res=256

<Insert key rows from tmp/profile_deep/results/per_stage_ops_res256.csv>

### s1_voxelize (0.144s, 1.6% of e2e)
Top 5 ops:
| Op | Count | Device ms | CPU ms |
|---|---:|---:|---:|
| … | … | … | … |

### s2_components (0.083s, 0.9%)
<similar>

### s3_edge_weights (0.002s, 0.0%)
<similar>

### s4_face_point (4.850s, 53.1%)
<similar, with sub-stage events if available>

### s6_collapse (2.445s, 26.8%)
<similar, with sub-stage events if available>

### s7_rank_assign (1.417s, 15.5%)
<similar, with sub-stage events if available>

### s8_decode (0.113s, 1.2%)
<similar>

---

## Layer 2 — Top-20 GPU kernel attribution

From tmp/profile_deep/results/top20_kernels_res256.csv:

| Rank | Stage | Kernel | Total ms | Mean µs | Count | Class | Py-line |
|---:|---|---|---:|---:|---:|---|---|
| 1 | … | … | … | … | … | … | … |
| … |

Class distribution: CMB=X, MMB=X, LNB=X, CPU=X, UNK=X.

---

## Layer 3 — Resolution scaling (res=256 vs res=128)

From tmp/profile_deep/results/scaling_table.csv:

| Stage | t@256 | t@128 | Ratio | Flag |
|---|---:|---:|---:|---|
| … | … | … | … | … |

### Flagged stages (ratio ∉ [2, 10])
<for each flagged stage, write a 1-2 sentence hypothesis — DoD #5>

- **stage_X** (ratio N.NN, flag=Y): hypothesis = …

---

## ROI-ranked next-step candidates (≥5 required — DoD #6)

Ranked by predicted Δ on res=256 e2e, higher first.

| # | Target | Hypothesis | Predicted Δ | Effort | Risk | Data source |
|---|---|---|---:|---|---|---|
| 1 | s4 Stage D BFS (Triton K1) | launch-bound MP, per (cube, facet) graph work ≪ MP overhead | -2 to -3s | 1-2 wk | M | Layer 2 + Layer 0 launch-count |
| 2 | … | … | … | … | … | … |
| 3 | … |
| 4 | … |
| 5 | … |

---

## Recommendations

<Based on the data, recommend ONE primary next-step with concrete rationale from the profile.>

---

## Appendix: provenance

- git sha: <fill>
- Phase 2 baseline: my-docs/20260416-pre-triton-final-pass-results.md
- Raw data: tmp/profile_deep/results/
- Trace files: layer{0,1,3}_res{256,128}_run*_trace.json (+ nsys-rep)
EOF
```

- [ ] **Step 11.2: Fill in every section**

Replace every `…` / `<fill>` placeholder with actual data. The top-20 kernel table and scaling table can be copy-pasted directly from the CSV files.

**Critical**: for each "flagged" stage in the scaling table, write a concrete hypothesis (not just "unusual"). Example:
> **s2_components** (ratio 1.3, flag=below_2x_fixed_cost_dominant): hypothesis = stage is launch-bound — 80% of its 0.083s is kernel launch overhead, which is fixed regardless of R. At higher R the relative % drops further.

- [ ] **Step 11.3: Draft ROI candidates (≥ 5)**

For each candidate:
1. Target (specific stage + sub-function)
2. Hypothesis linking profile data to the bottleneck
3. Predicted Δ (range; cite data source)
4. Effort estimate (days/weeks)
5. Risk level

Example candidates to evaluate (fill from actual data):
- **s4 Triton K1** — already in Triton handoff spec
- **d2h/h2d elimination** — if Layer 0 shows > 100MB of d2h traffic
- **stream parallelism** — if Layer 0 shows GPU idle > 100ms
- **s6 fast-path GPU batching** — if Layer 1 shows s6 has launch-bound kernels
- **s7 Phase-2 rank fill vectorization** — if sub-stage events show Phase-2 > 100ms
- **CPU-side Python loop** — if Layer 1 shows any stage has > 30% CPU time

Keep only candidates backed by data from this profile run.

- [ ] **Step 11.4: Smoke-review the doc**

```bash
# Sanity: no TBD/TODO left
grep -iE "TBD|TODO|<fill>|…" my-docs/20260416-corep-deep-profiling-results.md | head
```

Should produce no output (or only literal ellipsis in quoted tables).

- [ ] **Step 11.5: Commit**

```bash
git add my-docs/20260416-corep-deep-profiling-results.md
git commit -m "profile-deep(task11): results doc + ROI candidates"
```

---

## Task 12: Final DoD verification + wrap-up

**Files:**
- Modify: `logs/progress.md` (append entry)
- Create: `tmp/profile_deep/DONE.md` (short checklist)

- [ ] **Step 12.1: Verify each DoD item from spec §9**

Create `tmp/profile_deep/DONE.md`:

```markdown
# DoD verification

- [x] 1. All D1–D7 files present
  - D1 spec: docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md
  - D2–D6 results: my-docs/20260416-corep-deep-profiling-results.md
  - D7 raw: tmp/profile_deep/results/*
- [x] 2. Layer 0 ≥ 3 observations  (lines X-Y of results doc)
- [x] 3. Layer 1 8 × top-30 op tables for res=256 + res=128
- [x] 4. Top-20 kernel table: every row attributed + classified; UNK ≤ 20% (actual: N/20)
- [x] 5. Scaling table flags all ratios ∉ [2, 10] with hypothesis
- [x] 6. ROI candidates ≥ 5, ranked by predicted speedup
- [x] 7. Smoke-check overhead < 10% (or fallback documented)
- [x] 8. Every CSV/JSON has provenance header
```

Fill checkboxes from actual state; flip to `[ ]` if anything missing and go fix.

- [ ] **Step 12.2: Append progress.md entry**

```bash
cat >> logs/progress.md <<'EOF'

---

## CoReP Deep Profiling (2026-04-16)

### Goal
Pre-Triton kernel-level bottleneck investigation to inform next-stage choice among
Triton K1 / mesh-cleanup-port / s1-sat-hardening.

### Output
- Spec: docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md
- Plan: docs/superpowers/plans/2026-04-16-corep-deep-profiling-implementation.md
- Results: my-docs/20260416-corep-deep-profiling-results.md
- Raw: tmp/profile_deep/results/

### Key findings
<1-3 bullets, copy TL;DR from results doc>

### Recommended next step
<copy from results doc>
EOF
```

- [ ] **Step 12.3: Final commit**

```bash
git add logs/progress.md tmp/profile_deep/DONE.md
git commit -m "profile-deep(task12): DoD verification + progress log"
```

- [ ] **Step 12.4: Report to user**

Summarize in Chinese:
- Phase 2 9.13s baseline 确认
- 3 条 Layer 0 核心观察
- Top-5 kernel + 分类
- Scaling 异常的 stage
- ROI top-3 推荐
- 建议下一步方向

---

## Appendix A — If Smoke Check triggers fallback (Task 5 delta > 10%)

Modify `driver.py`'s head section:

```python
# from:
from tmp.profile_deep import monkeypatch_nvtx
monkeypatch_nvtx.apply_stage_nvtx()
monkeypatch_nvtx.apply_substage_events()

# to:
from tmp.profile_deep import monkeypatch_nvtx
monkeypatch_nvtx.apply_stage_nvtx()
# apply_substage_events() skipped — overhead > 10% per smoke check (see results/smoke_overhead.log)
```

Then:
- `sub_stage_timings.json` is empty in subsequent runs — acknowledged.
- Layer 1 analysis still works (relies on NVTX ranges, not sub-stage events).
- Sub-stage visibility in s4/s6/s7 is reduced to what NVTX alone + kernel-level trace can infer.
- Document this in results doc §Layer1 with a caveat.

## Appendix B — Pipeline call pattern source

The `run_pipeline` in Task 4 driver.py is transcribed directly from `tmp/e2e_profile_m2.py:38-108`. If any stage signature changes between the spec date (2026-04-16) and implementation, the implementer should re-sync from the reference file. Signatures at time of plan authoring (confirmed via grep):

- `MeshTensors.from_trimesh(mesh, resolution, device=...)`
- `s1_voxelize(mt, resolution, device)`
- `s2_components(batch, mt)`
- `s3_edge_weights(batch, mt)`
- `s4_face_point(batch, mt)`
- `s6_collapse(batch)`
- `s7_rank_assign(batch)`
- `decode_from_cubebatch(batch, merge_decimals=5) → (v, f)`
