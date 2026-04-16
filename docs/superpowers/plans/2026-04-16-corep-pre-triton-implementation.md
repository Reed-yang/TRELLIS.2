# CoReP Pre-Triton Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all torch / vectorization / algorithm-layer optimizations identified in Phase 1 analyses, restricted to those marked "do" in the bottleneck registry. Push remaining bottlenecks to a state where only Triton can further accelerate them.

**Architecture:** 2 git worktrees on 119 GPU 0/1, two batches of 2 worktrees each. Each worktree owns one stage's optimization, writes A/B test first (TDD), implements + per-stage profile + commits. Phase 3 merges + e2e profile + writes Triton handoff doc.

**Tech Stack:** PyTorch 2.x on H100, multiprocessing fallback removal where ROI justifies, GPU CSR gathers, batched BFS / parallel union-find, scatter/gather over ragged tensors.

**Spec:** `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`
**Phase 1 reports:** `tmp/pretriton_s{4,6,7,8}_analysis.md`
**Baseline:** res=128 e2e 7.41s, res=256 e2e 21.37s (median of 3, from `tmp/pretriton_baseline_res*.json`, 119 GPU 0)
**Soft target:** res=256 e2e ≤ 13s (from spec §5)

---

## Optimization Registry

### Will-do (in priority order)

| ID | Stage | Description | Predicted ΔT @ res=256 | Risk | Files | Worktree |
|---|-------|-------------|------------------------:|------|-------|---------|
| O1 | s8 | Vectorize 4-cube geometry, fix `_EDGE_OFFSET_TABLE` vs `get_local_edge` encoding mismatch, eliminate Python fallback for 99.95% of edges | **-2.8s** | M (encoding alignment) | `s8_collapse.py:830-1596` | W1 (Batch 1, GPU 0) |
| O2 | s7 | Parallel batched BFS for Phase 1 rank tracing (per-cube serial → all-cube parallel, GPU CSR adjacency) | **-2 to -3s** | M-H (graph alg complexity) | `s7_rank_assign.py:70-258, 427-490` | W2 (Batch 1, GPU 1) |
| O3 | s7 | Batched cyclic loop alignment (`_match_loops_to_ranks` tensorize) | **-1 to -2s** | L | `s7_rank_assign.py:259-313` | W2 (after O2, same worktree) |
| O5 | s6 | Fast-path GPU vectorization (parallel union-find + ragged loop tracing, 80% of cubes) | **-0.7 to -0.9s** | M | `s6_collapse.py:72-155, 502-655` | W3 (Batch 2, GPU 0) |
| O7 | s6 | Work-item dispatch tensor cleanup (eliminate 275K Python iter at line 580-590, 635-639) | **-0.02 to -0.05s** | L | `s6_collapse.py:580-590, 635-639` | W3 (after O5, same worktree) |

**Total predicted gain (if all succeed): -6.5 to -7.8s**
**Predicted Phase-2 e2e @ res=256: 13.6-14.9s** (border on soft target ≤13s)

### Will-skip (with rationale)

| ID | Stage | Description | Reason |
|---|-------|-------------|--------|
| O4 | s8 | Narrow candidate fallback predicate (alternative to O1) | Only used as **fallback within W1** if O1 cannot fully fix encoding. Not a separate task. |
| O6 | s4 | Component connectivity GPU union-find | ROI ~0.05s; high implementation cost; per S4 report §F: "ROI 仅 0.05s，不推荐" |

### Triton-only (Phase 2 documents these for handoff, does NOT implement)

| Item | Stage | Current time | Why Triton-only |
|---|-------|-------------:|-----------------|
| Stage D BFS U-turn | s4 | 4.3s | Per (cube, facet) graph too small (K≤30), MP overhead dominant; needs warp-level GPU kernel |
| Slow-path Cartesian product | s6 | 0.3-0.5s | Per-cube product size variance huge (10-100K), GPU pre-allocation difficult; persistent kernel needed |
| Step E 4-cube fallback (residual 0.05% pathological) | s8 | <0.1s after O1 | Rare edge cases that defeat full vectorization; small Triton kernel for the residual subset |

---

## Pre-Task Setup

- [ ] **Step 0.1: Verify pre-conditions**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git status                                    # must be clean
git branch --show-current                     # must be gpu-pipeline
ls -la tmp/pretriton_baseline_res128.json tmp/pretriton_baseline_res256.json  # must exist
ls -la tmp/pretriton_s{4,6,7,8}_analysis.md  # must exist
ssh host-10-240-99-119 'CUDA_VISIBLE_DEVICES=0,1 nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'  # GPU 0+1 should be 0%
```

If git status not clean: STOP and report. If baseline JSONs missing: re-run Task 1 of Phase 1 plan. If GPU busy: wait or notify user.

- [ ] **Step 0.2: Read all 4 Phase 1 analysis reports for full context**

```bash
cat tmp/pretriton_s4_analysis.md
cat tmp/pretriton_s6_analysis.md
cat tmp/pretriton_s7_analysis.md
cat tmp/pretriton_s8_analysis.md
```

These contain detailed algorithm descriptions, file:line citations, and Triton handoff specs. The implementation tasks below cite these reports for technical detail.

- [ ] **Step 0.3: Create worktrees per `superpowers:using-git-worktrees`**

For Batch 1, create 2 worktrees:

```bash
# W1: s8 vectorize (GPU 0)
git worktree add .claude/worktrees/pre-triton-s8 -b pre-triton/s8

# W2: s7 batched BFS + cyclic align (GPU 1)
git worktree add .claude/worktrees/pre-triton-s7 -b pre-triton/s7
```

For Batch 2 (created later, after Batch 1 verified):

```bash
git worktree add .claude/worktrees/pre-triton-s6 -b pre-triton/s6
```

(No worktree for s4 — skipped per registry.)

---

## Batch 1 (parallel): O1 + O2 + O3

Two worktrees run in parallel. Each is owned by its own implementer subagent.

### W1 / Task 1: O1 — s8 vectorize 4-cube geometry (eliminate Python fallback)

**Worktree:** `.claude/worktrees/pre-triton-s8` (branch `pre-triton/s8`)
**GPU:** 0
**Files:**
- Modify: `corep_fast/stages/s8_collapse.py:830-1596`
- Test: `corep_fast/tests/regression/test_s8_4cube_vectorized_ab.py` (new)
- Reference: `tmp/pretriton_s8_analysis.md` §B, §C-row-1, §D
- Reference: `my-docs/20260415-corep-fast-stage2-analysis.md` §3.3 + §4.1 (encoding mismatch root cause)

- [ ] **W1.1: Switch to W1 worktree and read the s8 analysis report**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s8
git status  # must be on pre-triton/s8 branch
cat tmp/pretriton_s8_analysis.md  # already on this branch (committed in Phase 1)
```

Note: The Phase 1 analyses are committed on `gpu-pipeline`, so they exist on this branch too.

- [ ] **W1.2: Pin GPU 0 for this worktree's profile/test runs**

In each subsequent ssh command, set `CUDA_VISIBLE_DEVICES=0` explicitly. Don't rely on inheritance.

- [ ] **W1.3: Write A/B test that captures current 4-cube fallback semantics**

Create `corep_fast/tests/regression/test_s8_4cube_vectorized_ab.py`:

```python
"""A/B regression test: ensure s8 vectorized-only (no Python fallback) matches
custom/ baseline V/F bit-exact for 4-cube edges.

Currently the test compares two corep_fast s8 paths:
  (a) baseline: vectorized partial-cube + Python 4-cube fallback (current default)
  (b) candidate: vectorized partial-cube + vectorized 4-cube (PHASE 2 W1 work)

The candidate must produce V/F that match (a) exactly (same V count, same F count,
same vertex set after canonical sort, same triangle set after canonical sort).
"""
import os
import torch
import numpy as np
import pytest
from corep_fast.stages.s8_collapse import (
    decode_from_cubebatch,
    process_geometry_vectorized,
)


@pytest.mark.parametrize("res", [32, 64, 128])
def test_4cube_vectorized_matches_python_fallback(res):
    # Run pipeline through s7 to get CubeBatch
    # Use icosphere subdiv=2 (built into trimesh)
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh_path = str(Path(__file__).parent.parent / "fixtures" / f"icosphere_s2_for_test.ply")
    Path(mesh_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)
    device = torch.device("cuda")
    from corep_fast.pipeline import corep_encode
    batch = corep_encode(mesh_path, resolution=res, device=device)

    # Path (a): current default with Python fallback
    os.environ["COREP_FAST_S8_4CUBE_VECTORIZED"] = "0"
    v_a, f_a = decode_from_cubebatch(batch)

    # Path (b): pure vectorized
    os.environ["COREP_FAST_S8_4CUBE_VECTORIZED"] = "1"
    v_b, f_b = decode_from_cubebatch(batch)

    # V/F count parity
    assert v_a.shape == v_b.shape, f"res={res}: V count differs ({v_a.shape} vs {v_b.shape})"
    assert f_a.shape == f_b.shape, f"res={res}: F count differs"

    # Canonical sort and compare (vertex order may differ)
    def _canon_vertices(v):
        return torch.unique(v.round(decimals=5), dim=0)

    def _canon_triangles(v, f):
        # Replace face indices with sorted vertex coordinates
        tri_coords = v[f]  # (F, 3, 3)
        # Sort within each triangle, then canonicalize across triangles
        tri_sorted = torch.sort(tri_coords.reshape(-1, 9), dim=0).values
        return tri_sorted

    cv_a, cv_b = _canon_vertices(v_a), _canon_vertices(v_b)
    assert torch.allclose(cv_a, cv_b, atol=1e-4), f"res={res}: canonical vertex set differs"

    ct_a, ct_b = _canon_triangles(v_a, f_a), _canon_triangles(v_b, f_b)
    assert torch.allclose(ct_a, ct_b, atol=1e-4), f"res={res}: canonical triangle set differs"
```

Adapt fixture path to whatever `corep_fast/tests/conftest.py` provides — read it first if unsure.

- [ ] **W1.4: Run test against current code; SHOULD FAIL with "COREP_FAST_S8_4CUBE_VECTORIZED unknown"**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s8
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/regression/test_s8_4cube_vectorized_ab.py -v" 2>&1 | tail -20
```

Expected: PASS for path (a), then identical PASS for path (b) (since flag has no effect yet — old code) — no, the test asserts they're equal, so it should pass with flag off. The point is to lock in baseline parity first. If the test passes on current code (flag has no effect), good. If it fails, debug fixture/imports.

- [ ] **W1.5: Add `COREP_FAST_S8_4CUBE_VECTORIZED` flag to `corep_fast/config.py`**

Add (where other M2 flags live, around `USE_DIRECT_GRIDS_S8`):

```python
S8_4CUBE_VECTORIZED = os.environ.get("COREP_FAST_S8_4CUBE_VECTORIZED", "0") == "1"
```

Default OFF (preserves current safe behavior). Phase 3 may flip to ON after merge.

- [ ] **W1.6: Map the local-edge encoding mismatch**

Read `corep_fast/stages/s8_collapse.py:1986-2005` (`_EDGE_OFFSET_TABLE`) and `custom/collapse.py::get_local_edge` to confirm the mismatch detailed in `my-docs/20260415-corep-fast-stage2-analysis.md` §4.1.

Write a short comment block at the top of `s8_collapse.py` documenting both encodings and the conversion table needed. This is the heart of W1 — without correctness here, the rest fails.

- [ ] **W1.7: Implement the unified `process_geometry_vectorized` for 4-cube edges**

In `corep_fast/stages/s8_collapse.py`:
- Verify (or fix) `process_geometry_vectorized` to handle 4-cube edges identically to `_process_shared_edge_geometry` (custom).
- Add encoding conversion (if not already present) in the `gather` step that consumes `n_loc` (line 867-912 area).
- Add `if S8_4CUBE_VECTORIZED:` branch in `_process_shared_edges_torch` (around line 1206) that processes ALL edges via vectorized path, skipping `_build_grids_from_tensors` + Python fallback.

Key: keep the OLD path working when flag is off. Only the NEW branch runs when ON.

- [ ] **W1.8: Run A/B test with flag ON across res 32, 64, 128**

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 COREP_FAST_S8_4CUBE_VECTORIZED=1 .venv/bin/python -m pytest corep_fast/tests/regression/test_s8_4cube_vectorized_ab.py -v" 2>&1 | tail -30
```

Expected: PASS at all 3 resolutions. If FAIL at res=32 only: tiny mesh may not exercise the divergent edge paths — check stderr to see which assertion failed.

If FAIL at res=64+: encoding fix incomplete. Read the divergent-edge diagnosis in `my-docs/20260415-corep-fast-stage2-analysis.md` §3.3 (9 edges out of 16,980 on icosphere@res=64). Identify which encoding case fails. Iterate. If after 3 iterations still failing: this is the "0.05% pathological case" — IMPLEMENT a hybrid: GPU vectorized for predicate `4-cube AND any_neighbor.num_loops < 2`, Python fallback for the residual ~2%. Document the predicate as `O4` partial completion.

- [ ] **W1.9: Per-stage profile @ res=128 + 256**

```bash
# Profile with flag OFF (baseline)
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python tmp/e2e_profile_m2.py --res 128 --out tmp/w1_off_res128.json"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/w1_off_res256.json"

# Profile with flag ON (candidate)
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 COREP_FAST_S8_4CUBE_VECTORIZED=1 .venv/bin/python tmp/e2e_profile_m2.py --res 128 --out tmp/w1_on_res128.json"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 COREP_FAST_S8_4CUBE_VECTORIZED=1 .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/w1_on_res256.json"

# Compare s8 stage time
.venv/bin/python -c "
import json
for res in [128, 256]:
    o = json.load(open(f'tmp/w1_off_res{res}.json'))['new']
    n = json.load(open(f'tmp/w1_on_res{res}.json'))['new']
    print(f'res={res}: s8 OFF={o[\"s8\"]:.2f}s  ON={n[\"s8\"]:.2f}s  delta={n[\"s8\"]-o[\"s8\"]:+.2f}s')
"
```

Expected: ΔS8 ≥ -2s @ res=256. If ΔS8 < -1s, the optimization didn't fire as expected — debug.

- [ ] **W1.10: Commit W1 work**

```bash
git add corep_fast/stages/s8_collapse.py corep_fast/config.py corep_fast/tests/regression/test_s8_4cube_vectorized_ab.py
git add -f tmp/w1_*.json
git commit -m "perf(s8): vectorize 4-cube geometry, eliminate Python fallback

Phase 2 W1 of pre-Triton work. Resolves local-edge encoding mismatch
between _EDGE_OFFSET_TABLE and custom/get_local_edge.

Result @ res=256 (119 GPU 0):
  s8: <baseline>s -> <candidate>s (saved <X>s)
  e2e: <baseline>s -> <candidate>s

Toggle: COREP_FAST_S8_4CUBE_VECTORIZED=1 (default off until Phase 3 merge)
A/B parity: V/F canonical match @ res 32/64/128 (icosphere subdiv=2)
"
```

Fill in the `<X>s` numbers from W1.9 profile output before running this commit.

---

### W2 / Task 2: O2 + O3 — s7 parallel BFS rank tracing + batched cyclic alignment

**Worktree:** `.claude/worktrees/pre-triton-s7` (branch `pre-triton/s7`)
**GPU:** 1
**Files:**
- Modify: `corep_fast/stages/s7_rank_assign.py:70-313, 427-490`
- Test: `corep_fast/tests/regression/test_s7_phase1_gpu_ab.py` (new)
- Reference: `tmp/pretriton_s7_analysis.md` §B (Phase 1 algorithm), §C, §E (Triton handoff for context)

- [ ] **W2.1: Switch to W2 worktree**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s7
git status  # on pre-triton/s7 branch
```

- [ ] **W2.2: Pin GPU 1**

All ssh commands use `CUDA_VISIBLE_DEVICES=1`.

- [ ] **W2.3: Write A/B test for O2 (Phase 1 GPU rank tracing)**

Create `corep_fast/tests/regression/test_s7_phase1_gpu_ab.py`:

```python
"""A/B test: ensure s7 GPU batched rank tracing matches CPU MP path.

Compares loop_edge_rank tensor between paths.
"""
import os
import torch
import pytest
from corep_fast.stages.s7_rank_assign import s7_rank_assign


@pytest.mark.parametrize("res", [32, 64, 128])
def test_s7_phase1_gpu_matches_cpu(res):
    # Run s1-s6 to feed s7
    import trimesh
    from pathlib import Path
    from corep_fast.containers import MeshTensors
    from corep_fast.stages import (
        s1_voxelize, s2_components, s3_edge_weights,
        s4_face_point, s6_collapse,
    )
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh_path = str(Path(__file__).parent.parent / "fixtures" / "icosphere_s2_for_test.ply")
    Path(mesh_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)
    device = torch.device("cuda")
    mt = MeshTensors.from_trimesh(mesh, device=device)
    batch = s1_voxelize(mt, resolution=res)
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    batch = s6_collapse(batch)

    # Path (a): CPU MP
    os.environ["COREP_FAST_S7_PHASE1_GPU"] = "0"
    batch_a_rank = s7_rank_assign(_clone_batch(batch)).loop_edge_rank.cpu()
    batch_a_match = s7_rank_assign(_clone_batch(batch)).loop_point_match.cpu()

    # Path (b): GPU batched
    os.environ["COREP_FAST_S7_PHASE1_GPU"] = "1"
    batch_b_rank = s7_rank_assign(_clone_batch(batch)).loop_edge_rank.cpu()
    batch_b_match = s7_rank_assign(_clone_batch(batch)).loop_point_match.cpu()

    assert torch.equal(batch_a_rank, batch_b_rank), f"res={res}: loop_edge_rank differs"
    assert torch.equal(batch_a_match, batch_b_match), f"res={res}: loop_point_match differs"


def _clone_batch(batch):
    """Helper: shallow-copy CubeBatch tensors so s7 doesn't mutate the original."""
    import dataclasses
    fields = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    for k, v in fields.items():
        if hasattr(v, 'clone'):
            fields[k] = v.clone()
    return type(batch)(**fields)
```

Note: `batch.clone()` may not exist on `CubeBatch`; if not, copy the tensors needed for s7 input or rerun pipeline.

- [ ] **W2.4: Run test on current code (pass with flag off)**

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest corep_fast/tests/regression/test_s7_phase1_gpu_ab.py -v"
```

Expected: PASS (both paths run the CPU MP code currently, since GPU flag has no effect).

- [ ] **W2.5: Add config flag**

In `corep_fast/config.py`:

```python
S7_PHASE1_GPU = os.environ.get("COREP_FAST_S7_PHASE1_GPU", "0") == "1"
```

- [ ] **W2.6: Implement O2 — GPU batched parallel BFS for Phase 1 rank tracing**

Per `tmp/pretriton_s7_analysis.md` §C-row-1 + §E:
- Input: GPU CSR adjacency built from `edge_weights` and `uturn_assignment` for ALL cubes at once
- Algorithm: parallel layer-by-layer BFS (all 275K cubes in one tensor pass)
- Output: `loop_edge_rank` (E,) int32 GPU tensor

Pseudocode:

```python
def _phase1_rank_tracing_gpu(batch):
    # 1. Build (cube, edge, rank) adjacency via batched gather
    #    For each cube: from edge_weights (N, 18) + uturn_assignment (N, 12, 3)
    #    Construct all (edge_a, rank_a) -> (edge_b, rank_b) pairs in one tensor.
    # 2. Parallel BFS: starting from each unvisited (cube, edge, rank), trace until back to start.
    # 3. Match to s6 loop_edge_val via cyclic comparison.
    # 4. Emit loop_edge_rank (E,) int32.
    ...
```

Where to wire: `s7_rank_assign` line 490 entry, branch on `S7_PHASE1_GPU` flag.

- [ ] **W2.7: Run O2 A/B test**

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=1 COREP_FAST_S7_PHASE1_GPU=1 .venv/bin/python -m pytest corep_fast/tests/regression/test_s7_phase1_gpu_ab.py -v"
```

Expected: PASS at all 3 resolutions. If FAIL: debug graph adjacency construction. The bug is most likely in the rank normalization for edges 2,6,3,7 (per §B).

- [ ] **W2.8: Profile O2**

Same pattern as W1.9, but:

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=1 .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/w2_o2_off_res256.json"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=1 COREP_FAST_S7_PHASE1_GPU=1 .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/w2_o2_on_res256.json"
.venv/bin/python -c "
import json
o = json.load(open('tmp/w2_o2_off_res256.json'))['new']
n = json.load(open('tmp/w2_o2_on_res256.json'))['new']
print(f's7 OFF={o[\"s7\"]:.2f}s  ON={n[\"s7\"]:.2f}s  delta={n[\"s7\"]-o[\"s7\"]:+.2f}s')
"
```

Expected: ΔS7 ≥ -1.5s.

- [ ] **W2.9: Commit O2**

```bash
git add corep_fast/stages/s7_rank_assign.py corep_fast/config.py corep_fast/tests/regression/test_s7_phase1_gpu_ab.py
git add -f tmp/w2_o2_*.json
git commit -m "perf(s7): GPU batched parallel BFS for Phase 1 rank tracing

Replaces per-cube CPU MP serial DFS with all-cubes parallel BFS
on GPU CSR adjacency.

Result @ res=256 (119 GPU 1):
  s7: <baseline>s -> <candidate>s (saved <X>s)

Toggle: COREP_FAST_S7_PHASE1_GPU=1 (default off until Phase 3)
"
```

- [ ] **W2.10: Implement O3 — batched cyclic loop alignment**

Per `tmp/pretriton_s7_analysis.md` §C-row-2:
- Input: s6 loop_edge_val (E,) + GPU traced loops from O2
- Algorithm: tensorize cyclic-shift comparisons across all (s6_loop, traced_loop) pairs
- Output: rank_list per s6 loop

If O2 already fully replaces Phase 1 (including matching), O3 may be subsumed. Check: does the O2 GPU BFS naturally emit the matched ranks, or does it need a separate alignment step? If the latter, implement O3 as additional GPU passes; if the former, mark O3 as DONE-via-O2.

Add `COREP_FAST_S7_CYCLIC_GPU` flag if separate, or document subsumption.

- [ ] **W2.11: A/B test O3 (if separate from O2)**

If separate: write `test_s7_cyclic_align_gpu_ab.py` similar to W2.3 but specifically for `_match_loops_to_ranks` output. If subsumed by O2: skip this step, note in commit.

- [ ] **W2.12: Profile O3 + Commit**

If O3 separate:

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=1 COREP_FAST_S7_PHASE1_GPU=1 COREP_FAST_S7_CYCLIC_GPU=1 .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/w2_o3_on_res256.json"
```

Commit:

```bash
git add corep_fast/stages/s7_rank_assign.py corep_fast/config.py corep_fast/tests/regression/test_s7_cyclic_align_gpu_ab.py
git add -f tmp/w2_o3_*.json
git commit -m "perf(s7): batched cyclic loop alignment on GPU

Result @ res=256 (119 GPU): <baseline_s>s -> <candidate_s>s (saved <delta_s>s)
A/B parity: V/F equivalent at res 32/64/128 (icosphere subdiv=2)
"
```

If subsumed: no extra commit; document in W2.10 commit message.

---

### Batch 1 Synchronization Checkpoint

After both W1 and W2 commit:

- [ ] **B1.1: Verify both worktrees clean**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s8 && git status
cd /mnt/novita2/siyuan/workspace/TRELLIS.2/.claude/worktrees/pre-triton-s7 && git status
```

Both should show "nothing to commit, working tree clean".

- [ ] **B1.2: Verify each branch has its commits**

```bash
git -C .claude/worktrees/pre-triton-s8 log --oneline gpu-pipeline..pre-triton/s8
git -C .claude/worktrees/pre-triton-s7 log --oneline gpu-pipeline..pre-triton/s7
```

Each should show 1-2 perf commits.

- [ ] **B1.3: Run full test suite on each branch**

```bash
cd .claude/worktrees/pre-triton-s8 && ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/ -q" 2>&1 | tail -10
cd .claude/worktrees/pre-triton-s7 && ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest corep_fast/tests/ -q" 2>&1 | tail -10
```

Expected: 215 passed (or whatever the current test count is) on each branch.

If any test fails: revert that worktree's most recent commit, debug, re-attempt.

---

## Batch 2 (parallel): O5 + O7

After Batch 1 sync:

### W3 / Task 3: O5 + O7 — s6 fast-path GPU + work-item dispatch cleanup

**Worktree:** `.claude/worktrees/pre-triton-s6` (branch `pre-triton/s6`)
**GPU:** 0 (W1 work is committed; GPU 0 free)
**Files:**
- Modify: `corep_fast/stages/s6_collapse.py:72-655`
- Test: `corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py` (new)
- Reference: `tmp/pretriton_s6_analysis.md` §B (algorithm), §C (Torch matrix)

- [ ] **W3.1: Switch to W3 worktree**

```bash
git worktree add .claude/worktrees/pre-triton-s6 -b pre-triton/s6
cd .claude/worktrees/pre-triton-s6
```

- [ ] **W3.2: Add config flag for O5**

In `corep_fast/config.py`:

```python
S6_FASTPATH_GPU = os.environ.get("COREP_FAST_S6_FASTPATH_GPU", "0") == "1"
```

- [ ] **W3.3: Write A/B test for O5**

Create `corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py`:

```python
"""A/B test: GPU fast-path s6 must match CPU fast-path bit-exactly."""
import os, torch, pytest
from corep_fast.stages.s6_collapse import s6_collapse


@pytest.mark.parametrize("res", [32, 64, 128])
def test_s6_fastpath_gpu_matches_cpu(res):
    # Run s1-s4 to feed s6
    import trimesh
    from pathlib import Path
    from corep_fast.containers import MeshTensors
    from corep_fast.stages import (
        s1_voxelize, s2_components, s3_edge_weights, s4_face_point,
    )
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh_path = str(Path(__file__).parent.parent / "fixtures" / "icosphere_s2_for_test.ply")
    Path(mesh_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)
    device = torch.device("cuda")
    mt = MeshTensors.from_trimesh(mesh, device=device)
    batch = s1_voxelize(mt, resolution=res)
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)

    os.environ["COREP_FAST_S6_FASTPATH_GPU"] = "0"
    batch_a = s6_collapse(_clone_batch(batch))

    os.environ["COREP_FAST_S6_FASTPATH_GPU"] = "1"
    batch_b = s6_collapse(_clone_batch(batch))

    for field in ["loop_cube_off", "loop_edge_off", "loop_edge_val", "status", "uturn_assignment"]:
        a = getattr(batch_a, field).cpu()
        b = getattr(batch_b, field).cpu()
        assert torch.equal(a, b), f"res={res}: {field} differs"


def _clone_batch(batch):
    """Helper: shallow-copy CubeBatch tensors so s6 doesn't mutate the original."""
    import dataclasses
    fields = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    for k, v in fields.items():
        if hasattr(v, 'clone'):
            fields[k] = v.clone()
    return type(batch)(**fields)
```

- [ ] **W3.4: Run test against current code (PASS w/ flag off, identical output)**

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py -v"
```

- [ ] **W3.5: Implement O5 — GPU fast-path**

Per `tmp/pretriton_s6_analysis.md` §C + §E:
- Input: edge_weights (N, 18) for ALL cubes (filter to fast-path mask first)
- Algorithm: parallel ragged adjacency construction → parallel union-find or per-cube CUDA-thread DFS for 2-regular graph cycle traversal
- Output: ragged loops (CSR layout: `loop_cube_off`, `loop_edge_off`, `loop_edge_val`)

Branch on `S6_FASTPATH_GPU` flag in `s6_collapse` line 502.

- [ ] **W3.6: Run A/B**

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 COREP_FAST_S6_FASTPATH_GPU=1 .venv/bin/python -m pytest corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py -v"
```

PASS required.

- [ ] **W3.7: Profile O5**

Same pattern. Expected ΔS6 ≥ -0.5s @ res=256.

- [ ] **W3.8: Commit O5**

```bash
git add corep_fast/stages/s6_collapse.py corep_fast/config.py corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py
git add -f tmp/w3_o5_*.json
git commit -m "perf(s6): GPU fast-path vectorization (80% of cubes)

Result @ res=256 (119 GPU): <baseline_s>s -> <candidate_s>s (saved <delta_s>s)
A/B parity: V/F equivalent at res 32/64/128 (icosphere subdiv=2)
"
```

- [ ] **W3.9: Implement O7 — work-item dispatch tensor cleanup**

Per `tmp/pretriton_s6_analysis.md` §C-row-3:
- Replace per-cube Python tolist() loop at line 580-590 with torch.nonzero + GPU tensor extraction
- Replace CSR pack loop at line 635-639 with `torch.cumsum` for offsets and tensor concat for values

This is mostly a rewrite of orchestration code; A/B test from W3.3 already covers correctness (output must match).

- [ ] **W3.10: A/B + Profile O7**

```bash
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 COREP_FAST_S6_FASTPATH_GPU=1 .venv/bin/python -m pytest corep_fast/tests/regression/test_s6_fastpath_gpu_ab.py -v"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 COREP_FAST_S6_FASTPATH_GPU=1 .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/w3_o7_on_res256.json"
```

Note: O7 doesn't have its own flag — it's an unconditional refactor. The A/B test confirms parity.

- [ ] **W3.11: Commit O7**

```bash
git add corep_fast/stages/s6_collapse.py
git add -f tmp/w3_o7_*.json
git commit -m "perf(s6): tensor-native work-item dispatch (eliminate 275K Python iter)

Result @ res=256 (119 GPU): <baseline_s>s -> <candidate_s>s (saved <delta_s>s)
A/B parity: V/F equivalent at res 32/64/128 (icosphere subdiv=2)
"
```

---

### Batch 2 Synchronization Checkpoint

- [ ] **B2.1: Verify W3 clean + tests pass**

```bash
cd .claude/worktrees/pre-triton-s6 && git status
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/ -q" 2>&1 | tail -10
```

Expected: clean + 215 passed.

---

## Phase 3: Merge + Final Profile + Triton Handoff Doc

### Task 4: Merge 3 worktree branches into integration branch

- [ ] **3.1: Create integration branch**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2  # main worktree
git checkout -b pre-triton/all
```

- [ ] **3.2: Merge in ROI order (highest first)**

```bash
git merge --no-ff pre-triton/s8 -m "merge: pre-triton/s8 (O1: vectorize 4-cube)"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/ -q" 2>&1 | tail -5  # verify
git merge --no-ff pre-triton/s7 -m "merge: pre-triton/s7 (O2+O3: GPU rank tracing + cyclic align)"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/ -q" 2>&1 | tail -5
git merge --no-ff pre-triton/s6 -m "merge: pre-triton/s6 (O5+O7: fast-path GPU + dispatch cleanup)"
ssh host-10-240-99-119 "cd $(pwd) && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/ -q" 2>&1 | tail -5
```

If any merge has conflicts: resolve manually, prefer the later branch's logic for files it owns. If any test fails after merge: investigate (cross-stage interaction). Likely candidates:
- s7's dependency on s6's `uturn_assignment` (M1 contract)
- s8's dependency on s7's `loop_edge_rank` and `loop_point_match`

- [ ] **3.3: Enable all flags by default in `corep_fast/config.py`**

Flip `S8_4CUBE_VECTORIZED`, `S7_PHASE1_GPU`, `S7_CYCLIC_GPU` (if exists), `S6_FASTPATH_GPU` to default `"1"`.

```bash
git add corep_fast/config.py
git commit -m "feat(pretriton): enable all Phase 2 optimizations by default"
```

### Task 5: Final e2e profile

- [ ] **3.4: Run e2e baseline once more on integration branch (3x median)**

```bash
ssh host-10-240-99-119 "bash $(pwd)/tmp/run_pretriton_baseline_119_gpu0.sh" 2>&1 | tee tmp/run_pretriton_final_119_gpu0.log
mv tmp/pretriton_baseline_res128.json tmp/pretriton_final_res128.json
mv tmp/pretriton_baseline_res256.json tmp/pretriton_final_res256.json
# (the script overwrites the baseline JSONs; rename to preserve)
```

- [ ] **3.5: Compare baseline vs final**

```bash
.venv/bin/python -c "
import json
print(f'{\"\":<10}{\"Baseline\":>10}{\"Final\":>10}{\"Delta\":>10}{\"Speedup\":>10}')
for res in [128, 256]:
    b = json.load(open(f'tmp/pretriton_baseline_res{res}.json.bak'))['new']  # rename above kept .bak?
    f = json.load(open(f'tmp/pretriton_final_res{res}.json'))['new']
    for stage in ['s4', 's6', 's7', 's8', 'e2e']:
        delta = f[stage] - b[stage]
        spd = b[stage] / max(f[stage], 1e-6)
        print(f'res={res} {stage:<6}{b[stage]:>10.2f}{f[stage]:>10.2f}{delta:>+10.2f}{spd:>9.2f}x')
"
```

(Adjust file naming — make sure original baseline survives the script overwrite. Either keep a copy before, or use a different output script.)

- [ ] **3.6: Write final results doc**

Create `my-docs/20260416-pre-triton-final-pass-results.md` with:
- Stage breakdown table (baseline vs final at res=128/256)
- Per-optimization actual ΔT vs predicted
- Soft target met or missed (≤13s @ res=256)
- Total speedup vs custom (target was 11x, started at 6.73x = current baseline / custom 141.05s)

```bash
git add -f my-docs/20260416-pre-triton-final-pass-results.md
git commit -m "docs(pretriton): final Phase 2 e2e profile results"
```

### Task 6: Triton handoff doc

- [ ] **3.7: Write `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md`**

Synthesize the Triton handoff from §E of each Phase 1 stage analysis + final profile data. Sections:

```markdown
# CoReP Triton Kernel Handoff Spec

## 1. Post-Phase-2 stage breakdown (actual)

Reproduce the final-results table from `my-docs/20260416-pre-triton-final-pass-results.md` here, including:
- Per-stage time (s1-s8) at res=128 + res=256 for both `gpu-pipeline` baseline and `pre-triton/all` final
- Total e2e + speedup vs custom 141.05s baseline
- Source: `tmp/pretriton_final_res128.json` and `tmp/pretriton_final_res256.json`

## 2. Triton kernels still needed (in priority order)

### K1. s4 Stage D BFS U-turn (~Y s, X% of e2e)
- Input tensors: ...
- Output: face_weights (N, 12) int32 contributions
- Recommended grid/block/SMEM: ...
- Reference: tmp/pretriton_s4_analysis.md §E.1

### K2. s6 slow-path persistent kernel (~Y s)
- ...

### K3. s8 4-cube residual (~Y s, only if W1 left a residual)
- ...

## 3. Cross-kernel data contract
[any shared layout requirements]

## 4. Implementation priority recommendation
1. K1 (highest ROI, well-understood algorithm)
2. K2 (medium ROI, complex enumeration logic)
3. K3 (only if needed)
```

```bash
git add -f docs/superpowers/specs/2026-04-16-corep-triton-handoff.md
git commit -m "docs(triton-handoff): post Phase-2 spec for remaining kernel work"
```

### Task 7: Wrap-up

- [ ] **3.8: Print summary to user**

Report:
- Phase 2 complete on `pre-triton/all` branch
- Final e2e @ res=128: <X>s (vs baseline <Y>s = saved <Z>s)
- Final e2e @ res=256: <X>s (vs baseline <Y>s = saved <Z>s, vs custom 141.05s = <N>x speedup)
- Soft target ≤13s: met / missed (with reason)
- 215 (or whatever) tests passing
- Triton handoff doc at `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md`

- [ ] **3.9: Update memory**

Save:

```markdown
---
name: pre_triton_phase2_complete
description: Pre-Triton Phase 2 implementation complete; corep_fast now ready for Triton kernel work
type: project
---
Pre-Triton Phase 2 complete on 2026-04-16 on branch pre-triton/all.

Status:
- 5 torch optimizations landed: O1 (s8 4-cube vectorize), O2 (s7 GPU BFS),
  O3 (s7 cyclic align), O5 (s6 fast-path GPU), O7 (s6 dispatch cleanup)
- Final e2e res=256: <X>s (vs baseline 21.37s, vs custom 141.05s = <N>x speedup)
- Soft target ≤13s: <met/missed>

Why: Last torch-layer optimization pass before Triton work, per spec
docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md.

How to apply: Triton work starts from pre-triton/all branch using
docs/superpowers/specs/2026-04-16-corep-triton-handoff.md as input.
```

- [ ] **3.10: Ask user about next step**

> "Phase 2 complete. Ready to merge `pre-triton/all` into `gpu-pipeline`, then start Triton kernel work per the handoff spec? Or hold for review first?"

---

## Acceptance Criteria

For Phase 2 (this plan) to be considered complete:

1. ✅ All "do" optimizations (O1, O2, O3, O5, O7) implemented OR explicitly skipped with documented reason
2. ✅ A/B parity tests pass at res=32/64/128 for each optimization (V/F equivalent or bit-exact per spec §1.2 case-by-case rule)
3. ✅ Per-stage profile shows the predicted ΔT (within ±50%) for each optimization
4. ✅ Final e2e profile @ res=256 shows total ΔT ≥ 4s vs baseline (target ~6-8s)
5. ✅ All existing unit tests still pass (215 or current)
6. ✅ `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md` written
7. ✅ `my-docs/20260416-pre-triton-final-pass-results.md` written

**Soft target** (not blocking): res=256 e2e ≤ 13s. If missed, document the gap (e.g., "O1 only saved 1.5s instead of 2.8s due to encoding fix complexity") and recommend whether to push for the gap with Triton or accept.

---

## Notes for the Executing Engineer

- **TDD discipline**: every optimization gets an A/B test BEFORE implementation. The flag must be off by default during development; tests assert flag-off and flag-on produce equivalent output.
- **Worktree discipline**: never modify files outside the worktree's stage. Cross-stage changes happen only at merge.
- **Profile discipline**: profile both res=128 and res=256 after each optimization. If ΔT is < 50% of predicted, debug before committing.
- **Encoding pitfall** (W1 specifically): the `_EDGE_OFFSET_TABLE` vs `get_local_edge` mismatch is THE source of all 4-cube fallback divergence. Don't try to vectorize 4-cube without first writing out the encoding tables and the conversion. See `my-docs/20260415-corep-fast-stage2-analysis.md` §4.1 for the full diagnosis.
- **Hungarian** (s7 Phase 3): per `tmp/pretriton_s7_analysis.md` §C-row-3, this is already optimal as serial scipy. DO NOT try to GPU-ify it.
- **CPU contention on 119**: per `feedback_profiling_on_119.md`, profile measurements have ~7-17% noise. Take median of 3 runs for any decision-relevant numbers.
- **Run from main worktree, not subagent**: Phase 3 merge tasks (3.1-3.7) must be done from `/mnt/novita2/siyuan/workspace/TRELLIS.2` (the original repo), not a worktree, to use the worktree branches for merge.
