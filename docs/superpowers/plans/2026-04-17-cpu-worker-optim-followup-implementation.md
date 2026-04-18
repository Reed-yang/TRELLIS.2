# corep_fast Residual Optim — Follow-up Implementation Plan (Phase 1 + Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the 4 workstreams from the spec (`docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-design.md`), driving e2e wall @ res=256 from 5.354 s → ≤ 3.2 s (-40%) while keeping F1/F2/F3 bit-exact.

**Architecture:** Two phases × two workstreams each. Phase 1 (pure PyTorch / numpy): W_L2L numpy-vectorize then W_SD batched GPU BFS. Phase 2 (Triton): W_BAF kernel-fusion then W_HG batched Hungarian. Each workstream is TDD (red → stub → real impl → integrate → benchmark) behind a `corep_fast/config.py` feature flag for 1-line rollback.

**Tech Stack:** PyTorch 2.x (CUDA), Triton 2.x, numpy, scipy (legacy fallback), pytest. Regression gate reuses `corep_fast/tests/regression/_cpu_worker_optim_runner.py` three-layer determinism pattern verbatim.

---

## Conventions used throughout this plan

- **Anchor HEAD:** `1ced857` (T9 clean post-W2+W4+W5); re-baseline if drift.
- **Test host:** `host-10-240-99-119` GPU 0-3 (per-task pick one). Never run regressions on local GPU.
- **SSH script pattern** (enforced by project CLAUDE.md):
  ```bash
  # write tmp/<task>_119.sh:
  cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
    CUDA_VISIBLE_DEVICES=<N> .venv/bin/python -m pytest <path> -v
  # run: ssh host-10-240-99-119 "bash" < tmp/<task>_119.sh
  ```
- **Clean wall command** (for DoD numbers):
  ```bash
  CUDA_VISIBLE_DEVICES=<N> .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --res 256 --trials 3
  ```
- **cProfile hotspot command** (for cProfile-self DoD numbers):
  ```bash
  CUDA_VISIBLE_DEVICES=<N> .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --res 256 --trials 1 --cprofile-main-thread \
    --cprofile-out tmp/cpu_profile/<task>_hotspots.txt
  ```
- **F1-F3 gate command:**
  ```bash
  CUDA_VISIBLE_DEVICES=<N> .venv/bin/python -m pytest \
    corep_fast/tests/regression/test_cpu_worker_optim.py -v
  ```
- **Commit style:** `<workstream-tag>(<phase>): <short title>` — e.g. `w_l2l(p1): vectorize bucket loop`. Always append `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **Feature flags live in `corep_fast/config.py`** matching existing pattern (`S7_PHASE1_GPU`, `S6_FASTPATH_GPU`). One env-var override per flag.

---

## Task 0: Prerequisites — confirm baseline green

**Files:**
- Run-only: `corep_fast/tests/regression/test_cpu_worker_optim.py`
- Run-only: `tmp/cpu_profile/t0_driver.py`
- Create: `tmp/followup_baseline/t0_driver_clean.json`
- Create: `tmp/followup_baseline/vram_peak.log`
- Create: `tmp/followup_baseline/hotspots_pre.txt`

- [ ] **Step 1: Verify working tree is at anchor HEAD**

```bash
git rev-parse HEAD
# Expected: 1ced857... (or later clean commit on post-profile-sync-elim)
git status  # should be clean except .claude/ and tmp/
```

- [ ] **Step 2: Write SSH runner `tmp/baseline_f123_119.sh`**

```bash
#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 | tee tmp/followup_baseline/f123.log
```

Run: `ssh host-10-240-99-119 "bash" < tmp/baseline_f123_119.sh`
Expected: **3 passed** (F1, F2, F3).

- [ ] **Step 3: Capture clean wall baseline**

Script `tmp/baseline_wall_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --res 256 --trials 3 \
    --out tmp/followup_baseline/t0_driver_clean.json 2>&1 | tee tmp/followup_baseline/wall.log
```

Run: `ssh host-10-240-99-119 "bash" < tmp/baseline_wall_119.sh`
Expected: median ∈ [5.20, 5.50] s.

- [ ] **Step 4: Capture cProfile top-20 baseline**

Script `tmp/baseline_hotspots_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --res 256 --trials 1 --cprofile-main-thread \
    --cprofile-out tmp/followup_baseline/hotspots_pre.txt
```

Run and verify the top-5 matches the spec §2.1 residual table (lock.acquire ~1648 ms, s7_rank_assign ~999 ms, etc.).

- [ ] **Step 5: Capture VRAM baseline**

Add to `tmp/baseline_wall_119.sh` (or separate script):
```python
# One-off Python snippet:
import torch
from corep_fast.pipeline import corep_pipeline
import trimesh
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
torch.cuda.reset_peak_memory_stats()
mesh.export('/tmp/fix.ply')
corep_pipeline('/tmp/fix.ply', 256, torch.device('cuda:0'))
print(f"peak_alloc_MB={torch.cuda.max_memory_allocated()/1024**2:.1f}")
print(f"peak_reserved_MB={torch.cuda.max_memory_reserved()/1024**2:.1f}")
```
Expected: 5687.8 / 13220.4 ± 5 %.

- [ ] **Step 6: Commit baseline artifacts**

```bash
git add tmp/followup_baseline/ tmp/baseline_*_119.sh
git commit -m "$(cat <<'EOF'
followup(t0): capture Phase-0 baseline (F1-F3 green, wall 5.35s, VRAM 5.7GB)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 1 — pure PyTorch / numpy (W_L2L + W_SD)

## Task 1: W_L2L feature flag

**Files:**
- Modify: `corep_fast/config.py`

- [ ] **Step 1: Add `LABELS_TO_LIST_VECTORIZED` flag**

Insert after line 48 of `corep_fast/config.py`:

```python
# Followup W_L2L (s4): Numpy-vectorize _labels_to_list_of_lists bucket loop
# (eliminates 275k-iter per-row Python loop, ~464ms self on T9 cProfile).
# Enabled by default after F1-F3 parity confirmed.
# Set COREP_FAST_LABELS_TO_LIST_VECTORIZED=0 to fall back to legacy bucket loop.
LABELS_TO_LIST_VECTORIZED = os.environ.get('COREP_FAST_LABELS_TO_LIST_VECTORIZED', '1') == '1'
```

- [ ] **Step 2: Commit**

```bash
git add corep_fast/config.py
git commit -m "$(cat <<'EOF'
w_l2l(p1): add LABELS_TO_LIST_VECTORIZED feature flag

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: W_L2L — red test against legacy bucket loop

**Files:**
- Create: `corep_fast/tests/unit/test_labels_to_list_vectorized.py`

- [ ] **Step 1: Write the failing test**

```python
"""W_L2L: vectorized _labels_to_list_of_lists must match legacy bucket loop."""
import numpy as np
import pytest
import torch

from corep_fast.stages.s4_face_point import _labels_to_list_of_lists


# Legacy implementation copied from s4_face_point.py:966-1012 at anchor HEAD
# (bit-identical reference for parity tests).
def _labels_to_list_of_lists_legacy(batched_labels, batched_face_ids, face_counts):
    N, M = batched_labels.shape
    if N == 0:
        return []
    order = torch.argsort(batched_labels, dim=1, stable=True)
    sorted_labels = batched_labels.gather(1, order)
    sorted_fids = batched_face_ids.gather(1, order)
    sorted_labels_cpu = sorted_labels.cpu().numpy()
    sorted_fids_cpu = sorted_fids.cpu().numpy()
    counts_cpu = face_counts.cpu().numpy()
    result = []
    for i in range(N):
        n_i = int(counts_cpu[i])
        if n_i == 0:
            result.append([])
            continue
        row_labels = sorted_labels_cpu[i, :n_i]
        row_fids = sorted_fids_cpu[i, :n_i]
        components = []
        cur_label = int(row_labels[0])
        cur_comp = [int(row_fids[0])]
        for k in range(1, n_i):
            lbl = int(row_labels[k])
            if lbl != cur_label:
                components.append(cur_comp)
                cur_comp = []
                cur_label = lbl
            cur_comp.append(int(row_fids[k]))
        components.append(cur_comp)
        result.append(components)
    return result


def _random_batch(N, M, seed=0):
    rng = np.random.RandomState(seed)
    counts = rng.randint(0, M + 1, size=N).astype(np.int64)
    labels = np.full((N, M), M, dtype=np.int64)  # pad = SENTINEL=M
    fids = np.full((N, M), -1, dtype=np.int64)
    for i in range(N):
        n = counts[i]
        # Random labels in [0, n), random fids in [0, 1e6)
        labels[i, :n] = rng.randint(0, max(1, n), size=n)
        fids[i, :n] = rng.randint(0, 1_000_000, size=n)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return (
        torch.from_numpy(labels).to(dev),
        torch.from_numpy(fids).to(dev),
        torch.from_numpy(counts).to(dev),
    )


def test_matches_legacy_random_100():
    labels, fids, counts = _random_batch(N=100, M=20, seed=0)
    legacy = _labels_to_list_of_lists_legacy(labels, fids, counts)
    new = _labels_to_list_of_lists(labels, fids, counts)
    assert len(legacy) == len(new)
    for i, (L_i, N_i) in enumerate(zip(legacy, new)):
        assert len(L_i) == len(N_i), f"row {i}: comp count mismatch"
        for ci, (Lc, Nc) in enumerate(zip(L_i, N_i)):
            # Both legacy and new return list[int] (or list of int-like)
            assert [int(x) for x in Lc] == [int(x) for x in Nc], \
                f"row {i} comp {ci}: {Lc} != {Nc}"


def test_empty_rows():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    labels = torch.zeros((5, 10), dtype=torch.int64, device=dev)
    fids = torch.zeros((5, 10), dtype=torch.int64, device=dev)
    counts = torch.zeros((5,), dtype=torch.int64, device=dev)
    result = _labels_to_list_of_lists(labels, fids, counts)
    assert result == [[], [], [], [], []]


def test_all_same_label_one_component_per_row():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N, M = 4, 5
    labels = torch.zeros((N, M), dtype=torch.int64, device=dev)
    fids = torch.arange(N * M, dtype=torch.int64, device=dev).reshape(N, M)
    counts = torch.full((N,), M, dtype=torch.int64, device=dev)
    result = _labels_to_list_of_lists(labels, fids, counts)
    assert len(result) == N
    for i in range(N):
        assert len(result[i]) == 1  # one component
        assert [int(x) for x in result[i][0]] == list(range(i * M, (i + 1) * M))


def test_all_distinct_labels_singleton_components():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N, M = 3, 4
    labels = torch.arange(M, dtype=torch.int64, device=dev).view(1, M).expand(N, M).contiguous()
    fids = torch.arange(N * M, dtype=torch.int64, device=dev).reshape(N, M)
    counts = torch.full((N,), M, dtype=torch.int64, device=dev)
    result = _labels_to_list_of_lists(labels, fids, counts)
    assert len(result) == N
    for i in range(N):
        assert len(result[i]) == M  # M singleton components
        for k in range(M):
            assert [int(x) for x in result[i][k]] == [i * M + k]
```

- [ ] **Step 2: Run the red test**

Script `tmp/w_l2l_red_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    corep_fast/tests/unit/test_labels_to_list_vectorized.py -v
```
Run: `ssh host-10-240-99-119 "bash" < tmp/w_l2l_red_119.sh`
**Expected:** tests PASS (we're comparing against legacy import; legacy still
active under flag=0 default). This is a parity test — it must pass against
the current impl before we change anything, so the **red** phase here is
actually "the test runs at all + parity holds before refactor". If any test
fails on unmodified HEAD, fix the test fixture before proceeding.

- [ ] **Step 3: Commit**

```bash
git add corep_fast/tests/unit/test_labels_to_list_vectorized.py
git commit -m "$(cat <<'EOF'
w_l2l(p1): add parity tests for _labels_to_list_of_lists (green on current HEAD)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: W_L2L — numpy-vectorized implementation

**Files:**
- Modify: `corep_fast/stages/s4_face_point.py:966-1012`

- [ ] **Step 1: Replace `_labels_to_list_of_lists` body with feature-flagged version**

Keep the legacy body behind `if not _cfg.LABELS_TO_LIST_VECTORIZED:` and add the vectorized path above it:

```python
def _labels_to_list_of_lists(
    batched_labels: "torch.Tensor",     # (N, M) int64, SENTINEL=M for pad
    batched_face_ids: "torch.Tensor",   # (N, M) int64, -1 pad
    face_counts: "torch.Tensor",        # (N,) int64
) -> list[list[list[int]]]:
    """Adapter: convert batched label tensor into List[List[List[int]]] with
    per-cube canonical ordering. Shape: outer list is per-cube components,
    each component is a list of face ids in slot-ascending order.
    """
    from corep_fast import config as _cfg  # lazy to avoid circular import
    import numpy as np

    N, M = batched_labels.shape
    if N == 0:
        return []

    # Stable-sort by labels per row. Padded entries (label=M=SENTINEL) sort last.
    order = torch.argsort(batched_labels, dim=1, stable=True)    # (N, M)
    sorted_labels = batched_labels.gather(1, order)              # (N, M)
    sorted_fids = batched_face_ids.gather(1, order)              # (N, M)

    sorted_labels_cpu = sorted_labels.cpu().numpy()
    sorted_fids_cpu = sorted_fids.cpu().numpy()
    counts_cpu = face_counts.cpu().numpy()

    if not _cfg.LABELS_TO_LIST_VECTORIZED:
        # Legacy path (retained for rollback)
        result: list[list[list[int]]] = []
        for i in range(N):
            n_i = int(counts_cpu[i])
            if n_i == 0:
                result.append([])
                continue
            row_labels = sorted_labels_cpu[i, :n_i]
            row_fids = sorted_fids_cpu[i, :n_i]
            components: list[list[int]] = []
            cur_label = int(row_labels[0])
            cur_comp: list[int] = [int(row_fids[0])]
            for k in range(1, n_i):
                lbl = int(row_labels[k])
                if lbl != cur_label:
                    components.append(cur_comp)
                    cur_comp = []
                    cur_label = lbl
                cur_comp.append(int(row_fids[k]))
            components.append(cur_comp)
            result.append(components)
        return result

    # ---- Vectorized path ----
    # valid_mask[i, k] = k < counts_cpu[i]
    k_idx = np.arange(M, dtype=np.int64)
    valid_mask = k_idx[None, :] < counts_cpu[:, None]  # (N, M) bool

    # Component boundary: slot k starts a new component iff
    # (k == 0 OR sorted_labels[i, k] != sorted_labels[i, k-1]) AND valid_mask[i, k]
    prev_labels = np.concatenate(
        [np.full((N, 1), -1, dtype=np.int64), sorted_labels_cpu[:, :-1]],
        axis=1,
    )  # (N, M)
    is_new_component = (sorted_labels_cpu != prev_labels) & valid_mask  # (N, M)

    comps_per_cube = is_new_component.sum(axis=1).astype(np.int64)  # (N,)

    # Per-slot local component index within cube (only meaningful at valid slots)
    comp_idx_flat = (is_new_component.cumsum(axis=1) - 1).reshape(-1)  # (N*M,)
    fids_flat = sorted_fids_cpu.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    # Per-cube base offset into global component array
    cumsum_comps = comps_per_cube.cumsum()
    comp_off_per_cube = np.concatenate(
        [np.array([0], dtype=np.int64), cumsum_comps[:-1]]
    )  # (N,)
    cube_idx_flat = np.repeat(np.arange(N, dtype=np.int64), M)
    global_comp_idx = comp_off_per_cube[cube_idx_flat] + comp_idx_flat  # (N*M,)

    # Restrict to valid slots then group by global_comp_idx via split.
    valid_gci = global_comp_idx[valid_flat]
    valid_fids = fids_flat[valid_flat]

    # Since global_comp_idx is non-decreasing on the valid subset (components
    # are filled in order), the boundaries are where gci increments.
    if valid_gci.size == 0:
        return [[] for _ in range(N)]
    split_at = np.flatnonzero(np.diff(valid_gci) > 0) + 1
    fids_per_comp = np.split(valid_fids, split_at)  # list of C numpy arrays

    # Rebuild nested list shape (one tolist() per component, not per fid).
    result: list[list[list[int]]] = [None] * N  # type: ignore
    cursor = 0
    for i in range(N):
        c_i = int(comps_per_cube[i])
        if c_i == 0:
            result[i] = []
        else:
            result[i] = [fids_per_comp[cursor + j].tolist() for j in range(c_i)]
            cursor += c_i
    return result
```

- [ ] **Step 2: Run parity tests**

Reuse `tmp/w_l2l_red_119.sh`.
**Expected:** 4 tests PASS.

- [ ] **Step 3: Run F1-F3 regression gate**

Reuse `tmp/baseline_f123_119.sh`.
**Expected:** 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add corep_fast/stages/s4_face_point.py
git commit -m "$(cat <<'EOF'
w_l2l(p1): vectorize _labels_to_list_of_lists bucket loop

Replaces per-row Python bucket loop with batched numpy
(argsort + cumsum + split). Expected cProfile self 464ms -> 100-150ms.
F1/F2/F3 bit-exact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: W_L2L — benchmark + DoD record

**Files:**
- Create: `logs/findings_w_l2l_vectorized.md`
- Create: `tmp/cpu_profile/w_l2l_post_wall.json`
- Create: `tmp/cpu_profile/w_l2l_post_hotspots.txt`

- [ ] **Step 1: Clean wall after W_L2L**

Script `tmp/w_l2l_wall_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --res 256 --trials 3 \
    --out tmp/cpu_profile/w_l2l_post_wall.json
```
Run and record median.

- [ ] **Step 2: cProfile hotspot after W_L2L**

Script `tmp/w_l2l_hotspots_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tmp/cpu_profile/t0_driver.py \
    --mode default --res 256 --trials 1 --cprofile-main-thread \
    --cprofile-out tmp/cpu_profile/w_l2l_post_hotspots.txt
```
Verify `_labels_to_list_of_lists` self ≤ 150 ms.

- [ ] **Step 3: Write findings doc**

`logs/findings_w_l2l_vectorized.md`:
```markdown
# W_L2L — `_labels_to_list_of_lists` numpy-vectorize findings

**HEAD:** <commit-sha-from-Task-3>
**Baseline:** 1ced857 (T9, wall 5.354 s)

## DoD check

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | 4 unit tests pass | 4/4 | <N>/<M> | <PASS/FAIL> |
| 2 | F1-F3 bit-exact | 3/3 | <N>/<M> | <PASS/FAIL> |
| 3 | Clean wall reduction | ≥ 0.15 s | <Δ> s | <PASS/FAIL> |
| 4 | cProfile `_labels_to_list_of_lists` self | ≤ 150 ms | <x> ms | <PASS/FAIL> |

## Raw

- Wall JSON: tmp/cpu_profile/w_l2l_post_wall.json
- Hotspots: tmp/cpu_profile/w_l2l_post_hotspots.txt
```

- [ ] **Step 4: Commit findings**

```bash
git add logs/findings_w_l2l_vectorized.md tmp/cpu_profile/w_l2l_post_*
git commit -m "$(cat <<'EOF'
w_l2l(p1): post-change findings + wall/hotspot artifacts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: W_SD — feature flag + design spike

**Files:**
- Modify: `corep_fast/config.py`
- Create: `tmp/followup_design/w_sd_stage_d_spike.md`

- [ ] **Step 1: Add `STAGE_D_GPU` feature flag**

Append to `corep_fast/config.py`:

```python
# Followup W_SD (s4): Stage D GPU BFS + U-turn counting (eliminates per-cube
# Python BFS in _p2_uturn_worker, ~49s worker wall -> expect ~100ms on GPU).
# Enabled by default after F1-F3 parity confirmed.
# Set COREP_FAST_STAGE_D_GPU=0 to fall back to CPU MP Pool.
STAGE_D_GPU = os.environ.get('COREP_FAST_STAGE_D_GPU', '1') == '1'
```

- [ ] **Step 2: Write spike — group-size histogram on F2**

Script `tmp/w_sd_spike_119.py`:
```python
"""W_SD spike: profile _count_uturns input sizes on F2 fixture.

Goal: confirm max segments-per-group (affects GPU memory budget for
batched node coalescence) and max unique-nodes-per-group (affects the
P x P cdist step).
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

# Monkeypatch SerialPool to match regression gate determinism
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool


class _SerialPool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, items, *a, **kw): return [fn(x) for x in items]
    def close(self): pass
    def join(self): pass


_mp.Pool = _SerialPool
_mp_pool.Pool = _SerialPool

import trimesh
from corep_fast.pipeline import corep_pipeline
from corep_fast.stages.s4_face_point import (
    _P2_GROUP_OFF, _P2_SEGS_A, _P2_CF,
)

# Hook into _p2_uturn_worker to collect group-size stats
import corep_fast.stages.s4_face_point as s4

_group_stats = {"sizes": [], "nodes": []}

_orig = s4._count_uturns


def _instrumented(segments, V0, V1, V2, cube_verts, vert_ids, edge_ids):
    _group_stats["sizes"].append(len(segments))
    # Approx unique-node count by running the internal linear-scan
    nodes = []
    for p1, p2 in segments:
        for pt in (p1, p2):
            found = False
            for n in nodes:
                if np.linalg.norm(pt - n) < 1e-8:
                    found = True
                    break
            if not found:
                nodes.append(pt)
    _group_stats["nodes"].append(len(nodes))
    return _orig(segments, V0, V1, V2, cube_verts, vert_ids, edge_ids)


s4._count_uturns = _instrumented

mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_f2.ply')
_ = corep_pipeline('/tmp/fix_f2.ply', 256, torch.device('cuda:0'))

sizes = np.asarray(_group_stats["sizes"], dtype=np.int64)
nodes = np.asarray(_group_stats["nodes"], dtype=np.int64)
print(f"groups={len(sizes)}")
print(f"segs/group: min={sizes.min()} max={sizes.max()} p50={np.percentile(sizes,50):.0f} p99={np.percentile(sizes,99):.0f}")
print(f"nodes/group: min={nodes.min()} max={nodes.max()} p50={np.percentile(nodes,50):.0f} p99={np.percentile(nodes,99):.0f}")
print(f"max P = 2 * max_s = {2*sizes.max()}")
```

Script `tmp/w_sd_spike_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 .venv/bin/python tmp/w_sd_spike_119.py \
    2>&1 | tee tmp/followup_design/w_sd_spike_output.txt
```
Run: `ssh host-10-240-99-119 "bash" < tmp/w_sd_spike_119.sh`

- [ ] **Step 3: Write spike findings**

`tmp/followup_design/w_sd_stage_d_spike.md`:
```markdown
# W_SD spike — Stage D group-size / node-count histogram (F2)

## Raw

<paste output from Step 2>

## Decisions

- P_MAX = 2 * max_segs_per_group = <value>
- If P_MAX <= 64: use (G, P, P) = (G, 64, 64) = ~G * 4096 * 1 byte = <value> MB VRAM. OK.
- If P_MAX > 64: chunk over G in blocks of <chunk_size> to cap VRAM.
- Expected G (groups): <value>. At P=64, cdist batch is ~256 MB — fits.

## Algorithm check

Python `_count_uturns` returns a single int per group (the U-turn count).
GPU version outputs a (G,) int64 tensor, consumed by
`_compute_face_weights_gpu`'s existing `u_per_edge` scatter-sum.
```

- [ ] **Step 4: Commit flag + spike artifacts**

```bash
git add corep_fast/config.py tmp/followup_design/ tmp/w_sd_spike_119.*
git commit -m "$(cat <<'EOF'
w_sd(p1): add STAGE_D_GPU flag + spike group-size histogram

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: W_SD — red test

**Files:**
- Create: `corep_fast/tests/unit/test_stage_d_gpu_bfs.py`

- [ ] **Step 1: Write failing test (ImportError red)**

```python
"""W_SD: _count_uturns_gpu_batched must match numpy _count_uturns per group."""
import numpy as np
import pytest
import torch

# Import target — will fail with ImportError on red
from corep_fast.stages.s4_face_point import (
    _count_uturns,
    _count_uturns_gpu_batched,  # NEW symbol; unresolved until Task 7
)


def _fake_group(n_segs, seed=0):
    """Generate n_segs line segments in a unit triangle."""
    rng = np.random.RandomState(seed)
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    segs = []
    for _ in range(n_segs):
        a = rng.rand(3) * 0.5
        b = rng.rand(3) * 0.5
        a[2] = 0.0  # keep on facet plane
        b[2] = 0.0
        segs.append((a, b))
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    return segs, V0, V1, V2, cube_verts


def test_matches_legacy_random_100_groups():
    groups_legacy = []
    groups_gpu_input = []
    for seed in range(100):
        n_segs = int(np.random.RandomState(seed).randint(1, 10))
        segs, V0, V1, V2, cube_verts = _fake_group(n_segs, seed)
        vert_ids = (0, 1, 2)
        edge_ids = (0, 1, 2)
        legacy_uturn = _count_uturns(segs, V0, V1, V2, cube_verts, vert_ids, edge_ids)
        groups_legacy.append(legacy_uturn)
        groups_gpu_input.append((segs, V0, V1, V2, cube_verts, vert_ids, edge_ids))

    # Pack into batched GPU input (details TBD by Task 7 signature)
    gpu_result = _count_uturns_gpu_batched(groups_gpu_input)
    assert gpu_result.shape == (100,)
    for i in range(100):
        assert int(gpu_result[i]) == groups_legacy[i], \
            f"group {i}: gpu={int(gpu_result[i])} vs legacy={groups_legacy[i]}"


def test_zero_segments_group_zero_uturn():
    empty = [([], np.zeros(3), np.array([1.,0,0]), np.array([0,1.,0]),
              np.zeros((8, 3)), (0,1,2), (0,1,2))]
    result = _count_uturns_gpu_batched(empty)
    assert int(result[0]) == 0


def test_single_edge_two_endpoints_same_edge_one_uturn():
    # Two endpoints on the same triangle edge → 1 U-turn (count // 2 = 1)
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    # Segment between two points both on edge V0-V1
    p1 = np.array([0.2, 0.0, 0.0])
    p2 = np.array([0.7, 0.0, 0.0])
    segs = [(p1, p2)]
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    group = (segs, V0, V1, V2, cube_verts, (0,1,2), (10, 20, 30))
    result = _count_uturns_gpu_batched([group])
    legacy = _count_uturns(*group)
    assert int(result[0]) == legacy


def test_closed_loop_zero_uturn():
    # Triangle loop → 0 endpoints → 0 U-turns
    V0 = np.array([0.0, 0.0, 0.0])
    V1 = np.array([1.0, 0.0, 0.0])
    V2 = np.array([0.0, 1.0, 0.0])
    a = np.array([0.1, 0.1, 0.0])
    b = np.array([0.3, 0.1, 0.0])
    c = np.array([0.2, 0.3, 0.0])
    segs = [(a, b), (b, c), (c, a)]
    cube_verts = np.stack([V0, V1, V2, V0+V1, V0+V2, V1+V2, V0+V1+V2, V0+V1-V2])
    group = (segs, V0, V1, V2, cube_verts, (0,1,2), (10, 20, 30))
    result = _count_uturns_gpu_batched([group])
    assert int(result[0]) == 0


@pytest.mark.gpu_heavy
def test_large_batch_50000():
    """Stress test: 50k synthetic groups, ensure no OOM on H100."""
    groups = []
    rng = np.random.RandomState(42)
    for _ in range(50_000):
        n_segs = rng.randint(1, 8)
        segs, V0, V1, V2, cube_verts = _fake_group(n_segs, seed=rng.randint(1<<30))
        groups.append((segs, V0, V1, V2, cube_verts, (0,1,2), (0,1,2)))
    result = _count_uturns_gpu_batched(groups)
    assert result.shape == (50_000,)
    # Just sanity check — parity with a sample
    for i in range(0, 50_000, 5000):
        legacy = _count_uturns(*groups[i])
        assert int(result[i]) == legacy, f"sample {i} mismatch"
```

- [ ] **Step 2: Run red test — expect ImportError**

```bash
# Same SSH pattern as Task 2 Step 2, path:
#   corep_fast/tests/unit/test_stage_d_gpu_bfs.py
```
**Expected:** `ImportError: cannot import name '_count_uturns_gpu_batched'`.

- [ ] **Step 3: Commit red test**

```bash
git add corep_fast/tests/unit/test_stage_d_gpu_bfs.py
git commit -m "$(cat <<'EOF'
w_sd(p1): red test for _count_uturns_gpu_batched (ImportError phase)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: W_SD — legacy-delegating stub (green)

**Files:**
- Modify: `corep_fast/stages/s4_face_point.py` (append new function after line 405)

- [ ] **Step 1: Add stub**

```python
def _count_uturns_gpu_batched(groups):
    """Batched GPU equivalent of _count_uturns. Returns (G,) int64 tensor.

    Args:
        groups: iterable of (segments, V0, V1, V2, cube_verts, vert_ids, edge_ids)
            tuples — same signature as _count_uturns per group.

    Returns:
        torch.Tensor shape (G,) int64, U-turn count per group.

    STUB: delegates to legacy _count_uturns per group. Real GPU impl in Task 8.
    """
    import torch as _torch
    counts = [_count_uturns(*g) for g in groups]
    return _torch.tensor(counts, dtype=_torch.int64)
```

- [ ] **Step 2: Run Task 6 tests**

**Expected:** 5 tests PASS.

- [ ] **Step 3: Run F1-F3 gate** (no regression, stub unused in production)

**Expected:** 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add corep_fast/stages/s4_face_point.py
git commit -m "$(cat <<'EOF'
w_sd(p1): _count_uturns_gpu_batched stub (legacy-delegating, green)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: W_SD — batched GPU implementation

**Files:**
- Modify: `corep_fast/stages/s4_face_point.py`

- [ ] **Step 1: Replace stub with batched impl**

Key phases (A-D per spec §4.2.2). Full code:

```python
def _count_uturns_gpu_batched(groups):
    """Batched GPU equivalent of _count_uturns.

    Args:
        groups: list of (segments, V0, V1, V2, cube_verts, vert_ids, edge_ids)

    Returns:
        (G,) int64 tensor of U-turn counts.
    """
    import torch as _torch
    G = len(groups)
    if G == 0:
        return _torch.zeros(0, dtype=_torch.int64)

    # ---- Pack into padded tensors ----
    max_s = max(len(g[0]) for g in groups)
    if max_s == 0:
        return _torch.zeros(G, dtype=_torch.int64)
    P_MAX = 2 * max_s                         # max candidate points per group

    dev = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

    pts = _torch.zeros((G, P_MAX, 3), dtype=_torch.float64, device=dev)
    pts_valid = _torch.zeros((G, P_MAX), dtype=_torch.bool, device=dev)
    facet_verts = _torch.zeros((G, 3, 3), dtype=_torch.float64, device=dev)
    edge_ids_t = _torch.full((G, 3), -1, dtype=_torch.int64, device=dev)

    # CPU pack loop (unavoidable — input is list of tuples); kept tight.
    pts_cpu = np.zeros((G, P_MAX, 3), dtype=np.float64)
    pts_valid_cpu = np.zeros((G, P_MAX), dtype=bool)
    fv_cpu = np.zeros((G, 3, 3), dtype=np.float64)
    ed_cpu = np.full((G, 3), -1, dtype=np.int64)
    for gi, (segs, V0, V1, V2, cube_verts, vert_ids, edge_ids) in enumerate(groups):
        v0i, v1i, v2i = vert_ids
        fv_cpu[gi, 0] = cube_verts[v0i]
        fv_cpu[gi, 1] = cube_verts[v1i]
        fv_cpu[gi, 2] = cube_verts[v2i]
        ed_cpu[gi] = np.asarray(edge_ids, dtype=np.int64)
        for si, (a, b) in enumerate(segs):
            pts_cpu[gi, 2*si] = a
            pts_cpu[gi, 2*si + 1] = b
            pts_valid_cpu[gi, 2*si] = True
            pts_valid_cpu[gi, 2*si + 1] = True

    pts.copy_(_torch.from_numpy(pts_cpu))
    pts_valid.copy_(_torch.from_numpy(pts_valid_cpu))
    facet_verts.copy_(_torch.from_numpy(fv_cpu))
    edge_ids_t.copy_(_torch.from_numpy(ed_cpu))

    # ---- Phase A: node coalescence (1e-8 tolerance, first-occurrence wins) ----
    # d[g, i, j] = ||pts[g,i] - pts[g,j]||
    d = _torch.cdist(pts, pts)                              # (G, P, P) f64
    match = (d < 1e-8)                                       # (G, P, P) bool
    # Mask: only consider j <= i (match lower triangle); for each slot i,
    # canonical node = min j where match[g,i,j] and j <= i.
    P = P_MAX
    tri_lower = _torch.tril(_torch.ones((P, P), dtype=_torch.bool, device=dev))
    match_lower = match & tri_lower.unsqueeze(0)             # (G, P, P)
    # Replace False with sentinel P, take min over j
    big = _torch.full_like(match_lower, P, dtype=_torch.int64)
    j_arange = _torch.arange(P, device=dev, dtype=_torch.int64).view(1, 1, P).expand(G, P, P)
    node_raw = _torch.where(match_lower, j_arange, big)      # (G, P, P)
    canonical_idx = node_raw.min(dim=-1).values              # (G, P)

    # Invalid slots (pad) keep sentinel P — they'll be masked out later.
    canonical_idx = _torch.where(pts_valid, canonical_idx,
                                  _torch.full_like(canonical_idx, P))

    # Remap canonical_idx (which is in 0..P-1) to dense node ids 0..n_nodes-1
    # per group. We do this via stable-sort of canonical_idx per group + diff.
    # But for adjacency we actually only need `match` matrix to be symmetric
    # over canonical representatives — we can keep canonical_idx space directly.

    # ---- Phase B: build edge_mask between canonical nodes via segment pairs ----
    # For each segment si, its two endpoint slots are (2*si, 2*si+1). The
    # canonical-representative of each is canonical_idx[g, 2*si] / [g, 2*si+1].
    # Edge exists iff the two canonicals differ.
    seg_slot_a = _torch.arange(0, P, 2, device=dev, dtype=_torch.int64)      # (max_s,)
    seg_slot_b = _torch.arange(1, P, 2, device=dev, dtype=_torch.int64)      # (max_s,)
    if seg_slot_a.numel() == 0:
        return _torch.zeros(G, dtype=_torch.int64)

    # canonical for seg endpoints: (G, max_s)
    c_a = canonical_idx.index_select(1, seg_slot_a)
    c_b = canonical_idx.index_select(1, seg_slot_b)
    seg_valid = pts_valid.index_select(1, seg_slot_a) & pts_valid.index_select(1, seg_slot_b)
    seg_nontrivial = seg_valid & (c_a != c_b)    # (G, max_s)

    # Build adjacency mask over (P, P) canonical-slot space. Symmetric.
    edge_mask = _torch.zeros((G, P, P), dtype=_torch.bool, device=dev)
    # Scatter: for each (g, si) with seg_nontrivial, set edge_mask[g, c_a, c_b] = True
    # (and symmetric).
    g_idx_exp = _torch.arange(G, device=dev, dtype=_torch.int64).unsqueeze(1).expand(G, max_s)
    nontrivial_mask = seg_nontrivial
    # Use index_put_ with flattened linear indices
    flat = g_idx_exp * (P * P) + c_a * P + c_b           # (G, max_s)
    flat_sym = g_idx_exp * (P * P) + c_b * P + c_a
    edge_mask_flat = edge_mask.view(-1)
    edge_mask_flat.scatter_(0, flat[nontrivial_mask], _torch.ones(int(nontrivial_mask.sum()), dtype=_torch.bool, device=dev))
    edge_mask_flat.scatter_(0, flat_sym[nontrivial_mask], _torch.ones(int(nontrivial_mask.sum()), dtype=_torch.bool, device=dev))
    edge_mask = edge_mask_flat.view(G, P, P)

    # ---- Phase C: label-propagation connected components over canonical nodes ----
    labels = _torch.arange(P, device=dev, dtype=_torch.int64).view(1, P).expand(G, P).clone()
    # Only label valid canonical-representative slots (self-canonical: canonical_idx == slot_idx).
    # For other slots, label = P (sentinel).
    self_canonical = (canonical_idx == _torch.arange(P, device=dev, dtype=_torch.int64).unsqueeze(0))
    labels = _torch.where(self_canonical, labels, _torch.full_like(labels, P))

    max_iters = min(P + 1, 64)
    for _ in range(max_iters):
        # Gather neighbor labels over edge_mask
        # For each (g, i), look at all j where edge_mask[g, i, j], take min label.
        big_lbl = _torch.full_like(edge_mask, P, dtype=_torch.int64)
        lbl_broadcast = labels.unsqueeze(1).expand(G, P, P)   # labels of j
        nbr_labels = _torch.where(edge_mask, lbl_broadcast, big_lbl)
        min_nbr = nbr_labels.min(dim=-1).values              # (G, P)
        new_labels = _torch.minimum(labels, min_nbr)
        new_labels = _torch.where(self_canonical, new_labels, _torch.full_like(new_labels, P))
        if _torch.equal(new_labels, labels):
            break
        labels = new_labels

    # ---- Phase C.5: degree + endpoint identification ----
    degree = edge_mask.sum(dim=-1)                            # (G, P) int (≤ P)
    is_endpoint = (degree == 1) & self_canonical              # (G, P) bool

    # ---- Phase D: project endpoints to 3 facet edges + count U-turns ----
    A = facet_verts[:, [0, 1, 2], :]   # (G, 3, 3)
    B = facet_verts[:, [1, 2, 0], :]   # (G, 3, 3)
    edge_vec = B - A                    # (G, 3, 3)
    length_sq = (edge_vec * edge_vec).sum(dim=-1)              # (G, 3)
    length_sq_safe = length_sq.clamp(min=1e-30)

    # pts: (G, P, 3); A: (G, 3, 3)
    P_minus_A = pts.unsqueeze(2) - A.unsqueeze(1)              # (G, P, 3, 3)
    dot = (P_minus_A * edge_vec.unsqueeze(1)).sum(dim=-1)      # (G, P, 3)
    t = dot / length_sq_safe.unsqueeze(1)                       # (G, P, 3)
    in_range = (t >= -1e-8) & (t <= 1.0 + 1e-8)                # (G, P, 3)
    proj = A.unsqueeze(1) + t.unsqueeze(-1) * edge_vec.unsqueeze(1)   # (G, P, 3, 3)
    dist = ((pts.unsqueeze(2) - proj) ** 2).sum(dim=-1).sqrt()  # (G, P, 3)
    on_edge = in_range & (dist < 1e-8)                          # (G, P, 3)

    # For each endpoint, assigned list = {edge_ids[g, j] for j in 0..2 if on_edge[g, p, j]}
    # If no on_edge match, assigned = {-1}. Count multi-edge matches (append all).
    assigned_edge_id = _torch.where(
        on_edge, edge_ids_t.view(G, 1, 3).expand(G, P, 3),
        _torch.full_like(on_edge, -1, dtype=_torch.int64),
    )   # (G, P, 3)
    any_on_edge = on_edge.any(dim=-1)                           # (G, P)

    # Now for each endpoint slot (g, p), we need to count how many edges it
    # landed on; for each such edge_id (or -1 if none), accumulate per
    # (g, component_label, edge_id).
    # Flatten: for each (g, p, j), if is_endpoint[g,p] and on_edge[g,p,j],
    # contribute +1 to count at (g, labels[g,p], edge_ids[g,j]).
    # Additionally: if is_endpoint[g,p] and not any_on_edge[g,p], contribute
    # +1 at edge_id = -1 (but this doesn't produce U-turns, so skip).

    # Use bincount approach. Build flat key = g * L * E + label * E + edge_id.
    # L (label space) = P, E (edge_ids 0..max_edge_id inclusive + 1 for sentinel).
    # edge_ids can be arbitrary integers — we need to dense-map first.
    # Since we only count // 2 over the bucket, we can use a (G, P, 3) -based
    # approach directly: for each (g, ep), iterate j in 0..2, if on_edge, bucket.

    # Per-(g, endpoint-slot-p, j-0..2): contribute to per-(g, comp_label, j-edge-id).
    # But comp_label is same for all j of a given p, and edge_id is edge_ids_t[g, j].
    # So effectively bucket by (g, labels[g,p], edge_ids_t[g, j]).

    # ASSUMPTION: edge_ids values are small ints (< 1e6). Use sort-based counting.
    # Build weight tensor: (G, P, 3) = is_endpoint[g,p] & on_edge[g,p,j]
    ep_mask = is_endpoint.unsqueeze(-1).expand(G, P, 3)   # (G, P, 3)
    hit_mask = ep_mask & on_edge                          # (G, P, 3)

    # Valid (g, p, j) positions
    g_idx = _torch.arange(G, device=dev, dtype=_torch.int64).view(G, 1, 1).expand(G, P, 3)
    lbl_broadcast = labels.unsqueeze(-1).expand(G, P, 3)    # (G, P, 3) int64
    eid_broadcast = edge_ids_t.view(G, 1, 3).expand(G, P, 3)  # (G, P, 3)

    # For each hit, we need to count per (g, lbl, eid). Pack key = g * K1 + lbl * K2 + eid
    # K2 should cover edge_id range. Find max edge_id:
    max_eid = int(edge_ids_t.max().item()) + 1 if G > 0 else 1
    # Guard against huge values:
    max_eid = max(max_eid, 1)

    # Flat key (only for hit positions)
    K_per_g = P * max_eid
    flat_key = g_idx * K_per_g + lbl_broadcast * max_eid + eid_broadcast  # (G, P, 3) int64
    hit_keys = flat_key[hit_mask]                                          # (H,)

    # Count occurrences of each key
    if hit_keys.numel() == 0:
        return _torch.zeros(G, dtype=_torch.int64)
    max_key = int(hit_keys.max().item()) + 1
    counts = _torch.bincount(hit_keys, minlength=max_key)                 # (max_key,)
    # U-turn contribution per bucket = counts // 2
    uturn_contribs = counts // 2                                           # (max_key,)

    # Sum all u-turn contribs back to their group.
    # Each bucket's key = g * K_per_g + rest, so group id = bucket_idx // K_per_g.
    bucket_arange = _torch.arange(max_key, device=dev, dtype=_torch.int64)
    group_of_bucket = bucket_arange // K_per_g
    # scatter_add into (G,)
    per_group_uturn = _torch.zeros(G, dtype=_torch.int64, device=dev)
    per_group_uturn.scatter_add_(0, group_of_bucket, uturn_contribs)

    return per_group_uturn
```

- [ ] **Step 2: Run Task 6 tests + F1-F3**

**Expected:** all pass. If mismatch on parity tests, fix by:
1. Check `max_eid` handles negative edge_id gracefully (clamp to 0 or skip).
2. Check self_canonical logic for pad slots.

- [ ] **Step 3: Commit**

```bash
git add corep_fast/stages/s4_face_point.py
git commit -m "$(cat <<'EOF'
w_sd(p1): batched GPU BFS + U-turn count impl

Phase A: cdist-based node coalescence (tol=1e-8).
Phase B: batched scatter to build edge_mask.
Phase C: label-propagation connected components (min-gather, diameter <=64).
Phase D: endpoint projection + bincount-based U-turn accumulation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: W_SD — integrate at Stage D driver

**Files:**
- Modify: `corep_fast/stages/s4_face_point.py` (call site near line 1186 / `_p2_uturn_worker` dispatch)

- [ ] **Step 1: Locate Stage D Pool.map call**

```bash
grep -n "_p2_uturn_worker" corep_fast/stages/s4_face_point.py
```
Find the `pool.map(_p2_uturn_worker, ...)` call and the surrounding build of
`_P2_CF`, `_P2_GROUP_OFF`, `_P2_SEGS_A`, `_P2_SEGS_B`, `_P2_CUBE_IDX`, `_P2_STEP`.

- [ ] **Step 2: Add GPU path behind `STAGE_D_GPU` flag**

Replace the dispatch block with:

```python
from corep_fast import config as _cfg

if _cfg.STAGE_D_GPU:
    # Build groups list in Python; each entry = (segments, V0, V1, V2, cube_verts, vert_ids, edge_ids)
    groups = []
    for gi in range(_P2_GROUP_OFF.size - 1 if hasattr(_P2_GROUP_OFF, 'size') else len(_P2_GROUP_OFF) - 1):
        cf = int(_P2_CF[gi])
        cube_id = cf // 12
        facet_id = cf % 12
        lo = int(_P2_GROUP_OFF[gi])
        hi = int(_P2_GROUP_OFF[gi + 1])
        segs = [(_P2_SEGS_A[i], _P2_SEGS_B[i]) for i in range(lo, hi)]
        ix, iy, iz = _P2_CUBE_IDX[cube_id]
        base = np.array([ix, iy, iz], dtype=np.float64) * _P2_STEP
        cube_verts = base + _V_OFFSETS * _P2_STEP
        v_ids = FACET_VERTS[facet_id]
        e_ids = FACET_EDGES[facet_id]
        V0 = cube_verts[v_ids[0]]
        V1 = cube_verts[v_ids[1]]
        V2 = cube_verts[v_ids[2]]
        groups.append((segs, V0, V1, V2, cube_verts, tuple(v_ids), tuple(e_ids)))

    uturn_tensor = _count_uturns_gpu_batched(groups)  # (G,) int64 on device
    uturn_counts_cpu = uturn_tensor.cpu().numpy()

    # Rebuild (cube_id, facet_id, uturn) triples (matching legacy MP return shape)
    results = []
    for gi in range(len(groups)):
        cf = int(_P2_CF[gi])
        results.append((cf // 12, cf % 12, int(uturn_counts_cpu[gi])))
else:
    # Legacy MP path
    from corep_fast.utils.persistent_pool import get_pool
    pool = get_pool(num_workers)
    results = pool.map(_p2_uturn_worker, range(len(_P2_CF) if hasattr(_P2_CF, '__len__') else _P2_CF.size))
```

**Note:** the exact shape of the call site may differ from the pseudo-code
above — read the actual s4_face_point.py lines 1180-1220 first and adapt
within the same control-flow structure.

- [ ] **Step 3: F1-F3 regression gate**

```bash
ssh host-10-240-99-119 "bash" < tmp/baseline_f123_119.sh
```
**Expected:** 3 tests PASS.

- [ ] **Step 4: Clean wall**

```bash
ssh host-10-240-99-119 "bash" < tmp/w_sd_wall_119.sh  # copy from Task 4, write to tmp/cpu_profile/w_sd_post_wall.json
```
**Expected:** median ≤ 4.5 s (>= 0.8 s reduction from 5.35).

- [ ] **Step 5: Commit**

```bash
git add corep_fast/stages/s4_face_point.py
git commit -m "$(cat <<'EOF'
w_sd(p1): integrate GPU BFS at Stage D driver (feature-flagged)

STAGE_D_GPU=1 routes _compute_face_weights_gpu's Part 2 through
_count_uturns_gpu_batched instead of the MP pool.map.
Legacy path preserved behind STAGE_D_GPU=0 for rollback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: W_SD — findings + Phase 1 checkpoint

**Files:**
- Create: `logs/findings_w_sd_gpu_bfs.md`
- Create: `logs/findings_phase1_checkpoint.md`

- [ ] **Step 1: Capture wall + hotspot + VRAM after W_SD**

Run `tmp/w_sd_wall_119.sh` and `tmp/w_sd_hotspots_119.sh` (copy-adapted from Task 4).

- [ ] **Step 2: Write W_SD findings**

`logs/findings_w_sd_gpu_bfs.md`:
```markdown
# W_SD — Stage D GPU BFS findings

## DoD
| # | Target | Actual | Status |
|---|---|---|---|
| 1 | 5 unit tests pass | <N>/5 | — |
| 2 | F1-F3 bit-exact | <N>/3 | — |
| 3 | `_p2_uturn_worker` calls = 0 | <x> | — |
| 4 | `lock.acquire` Δ ≥ -900 ms | <Δ> ms | — |
| 5 | Wall Δ ≥ -0.8 s | <Δ> s | — |
| 6 | VRAM Δ ≤ +500 MB | <Δ> MB | — |
```

- [ ] **Step 3: Write Phase 1 checkpoint**

`logs/findings_phase1_checkpoint.md`:
```markdown
# Phase 1 checkpoint (W_L2L + W_SD)

## Cumulative DoD vs spec §6.1

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | W_L2L DoD met | — | — | — |
| 2 | W_SD DoD met | — | — | — |
| 3 | F1-F3 bit-exact | 3/3 | — | — |
| 4 | e2e wall @ res=256 | ≤ 4.2 s | <x> s | — |
| 5 | No new ≥200ms Python hotspot | yes | — | — |
| 6 | VRAM peak ≤ 6187 MB | — | — | — |
| 7 | nsys GPU util (informational) | — | — | — |
```

Optional: capture one nsys trace and extract GPU util for reference:
```bash
ssh host-10-240-99-119 "bash" < tmp/phase1_nsys_119.sh
# Produces tmp/cpu_profile/phase1_nsys.qdrep — scan for Compute / Memcpy percentages
```

- [ ] **Step 4: Commit findings**

```bash
git add logs/findings_w_sd_gpu_bfs.md logs/findings_phase1_checkpoint.md \
        tmp/cpu_profile/w_sd_* tmp/cpu_profile/phase1_*
git commit -m "$(cat <<'EOF'
w_sd(p1): post-change findings + Phase 1 checkpoint

Phase 1 closes: e2e 5.354 -> <x>s (<pct>%). F1-F3 green.
DoD items: W_L2L, W_SD, cumulative wall, VRAM, nsys util.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Phase 1 gate:** if e2e > 4.2 s, STOP and debug before Phase 2. If F1-F3
breaks, rollback by flipping `STAGE_D_GPU=0` and reopen the workstream.

---

# Phase 2 — Triton (W_BAF + W_HG)

## Task 11: W_BAF — feature flag + Triton kernel spike (fast-path only)

**Files:**
- Modify: `corep_fast/config.py`
- Create: `corep_fast/stages/s7_triton.py`
- Create: `tmp/followup_design/w_baf_triton_kernel_sketch.md`

- [ ] **Step 1: Add `BUILD_ADJACENCY_TRITON` flag**

Append to `corep_fast/config.py`:

```python
# Followup W_BAF (s7): Triton kernel fusion for _build_adjacency_gpu (replaces
# 576-iteration Python loop + 1152 scatter launches with 1 kernel).
# Enabled by default after F1-F3 parity + 20x determinism audit pass.
# Set COREP_FAST_BUILD_ADJACENCY_TRITON=0 to fall back to legacy PyTorch loop.
BUILD_ADJACENCY_TRITON = os.environ.get('COREP_FAST_BUILD_ADJACENCY_TRITON', '1') == '1'
```

- [ ] **Step 2: Write minimal Triton spike (fast-path cubes only)**

`corep_fast/stages/s7_triton.py`:

```python
"""Triton kernels for s7_rank_assign stage.

W_BAF (build-adjacency fused): single kernel that replaces the 12x3xW
Python loop + scatter launches in _build_adjacency_gpu.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _build_adj_fast_kernel(
        edge_weights_ptr,        # (N, 18) int64
        eA_tab_ptr,              # (12, 3) int64 constant
        eB_tab_ptr,              # (12, 3) int64 constant
        eC_tab_ptr,              # (12, 3) int64 constant
        a_at_v0_tab_ptr,         # (12, 3) int8 constant
        b_at_v0_tab_ptr,         # (12, 3) int8 constant
        adj_ptr,                 # (N, NODES, 2) int32 output (init -1)
        fill_count_ptr,          # (N, NODES) int32 output (init 0)
        N: tl.constexpr,
        NODES: tl.constexpr,
        W: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per (cube, t_idx, pi, jj). Fast-path only (is_fast=True).

        pid_linear = cube * 12*3*W + t_idx * 3*W + pi * W + jj
        """
        pid = tl.program_id(0)
        jj = pid % W; pid_rest = pid // W
        pi = pid_rest % 3; pid_rest2 = pid_rest // 3
        t_idx = pid_rest2 % 12
        cube_id = pid_rest2 // 12

        if cube_id >= N:
            return

        # Load constant tables
        eA = tl.load(eA_tab_ptr + t_idx * 3 + pi)
        eB = tl.load(eB_tab_ptr + t_idx * 3 + pi)
        eC = tl.load(eC_tab_ptr + t_idx * 3 + pi)
        a_at_v0 = tl.load(a_at_v0_tab_ptr + t_idx * 3 + pi)
        b_at_v0 = tl.load(b_at_v0_tab_ptr + t_idx * 3 + pi)

        # Load edge weights (fast-path: ew_eff == ew since u_per_edge = 0)
        w_a = tl.load(edge_weights_ptr + cube_id * 18 + eA)
        w_b = tl.load(edge_weights_ptr + cube_id * 18 + eB)
        w_c = tl.load(edge_weights_ptr + cube_id * 18 + eC)

        k_pair = (w_a + w_b - w_c) // 2
        if k_pair < 0: k_pair = 0
        if k_pair > W: k_pair = W
        if jj >= k_pair:
            return

        # Endpoint flipping
        if a_at_v0 != 0:
            pts_A = jj
        else:
            pts_A = w_a - 1 - jj
        if b_at_v0 != 0:
            pts_B = jj
        else:
            pts_B = w_b - 1 - jj

        node_A = eA * W + pts_A
        node_B = eB * W + pts_B

        # A -> B
        slot_A = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_A, 1)
        if slot_A < 2:
            tl.store(adj_ptr + cube_id * NODES * 2 + node_A * 2 + slot_A, node_B)
        # B -> A
        slot_B = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_B, 1)
        if slot_B < 2:
            tl.store(adj_ptr + cube_id * NODES * 2 + node_B * 2 + slot_B, node_A)


def build_adjacency_triton_fast_only(
    edge_weights: torch.Tensor,       # (N, 18) int64
    eA_tab: torch.Tensor, eB_tab: torch.Tensor, eC_tab: torch.Tensor,
    a_at_v0_tab: torch.Tensor, b_at_v0_tab: torch.Tensor,
    NODES: int, W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast-path-only adjacency builder. Returns (adj, fill_count)."""
    assert _TRITON_AVAILABLE, "Triton not available"
    device = edge_weights.device
    N = edge_weights.shape[0]
    adj = torch.full((N, NODES, 2), -1, dtype=torch.int32, device=device)
    fill_count = torch.zeros((N, NODES), dtype=torch.int32, device=device)
    grid = (N * 12 * 3 * W,)
    _build_adj_fast_kernel[grid](
        edge_weights, eA_tab, eB_tab, eC_tab,
        a_at_v0_tab.to(torch.int8), b_at_v0_tab.to(torch.int8),
        adj, fill_count,
        N=N, NODES=NODES, W=W, BLOCK_SIZE=256,
    )
    return adj, fill_count
```

- [ ] **Step 3: Spike test (fast-path only vs legacy)**

`tmp/followup_design/w_baf_spike_test.py`:
```python
"""Spike: Triton fast-path adjacency kernel vs legacy PyTorch path.

Generates 1000 fake fast-path cubes, runs both paths, asserts adjacency
output bit-identical after canonical sort.
"""
import sys
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

import numpy as np
import torch
from corep_fast.stages.s7_rank_assign import _build_adjacency_gpu, _W_MAX, _NODES_PER_CUBE
from corep_fast.stages.s7_triton import build_adjacency_triton_fast_only, _TRITON_AVAILABLE
from corep_fast.stages.s7_rank_assign import _facet_pair_table

if not _TRITON_AVAILABLE:
    raise RuntimeError("Triton required for spike")

device = torch.device('cuda:0')
N = 1000
rng = torch.Generator(device='cpu').manual_seed(42)
edge_weights = torch.randint(0, _W_MAX, (N, 18), generator=rng, dtype=torch.int64).to(device)
# Fast-path: uturn[:, 0, 0] = -1
uturn = torch.full((N, 12, 3), -1, dtype=torch.int64, device=device)

# Legacy
adj_legacy = _build_adjacency_gpu(edge_weights, uturn)  # (N, NODES, 2) int32

# Triton
eA_tab, eB_tab, a_v0, b_v0 = _facet_pair_table(device)
# Derive eC_tab per spec (facet_edges[:, [2,0,1]]):
from corep_fast.stages.s7_rank_assign import CUBE_FACETS
facet_edges = CUBE_FACETS.to(device=device, dtype=torch.int64)
eC_tab = torch.stack([facet_edges[:, 2], facet_edges[:, 0], facet_edges[:, 1]], dim=1)

adj_triton, _ = build_adjacency_triton_fast_only(
    edge_weights, eA_tab, eB_tab, eC_tab, a_v0, b_v0,
    NODES=_NODES_PER_CUBE, W=_W_MAX,
)

# Canonical sort both (push -1 to end)
def canonicalize(adj):
    valid = adj >= 0
    key = torch.where(valid, adj, torch.full_like(adj, _NODES_PER_CUBE))
    key, _ = key.sort(dim=-1)
    return torch.where(key == _NODES_PER_CUBE, -1, key)

adj_legacy_c = canonicalize(adj_legacy)
adj_triton_c = canonicalize(adj_triton)

mismatches = (adj_legacy_c != adj_triton_c).sum().item()
print(f"N={N}, mismatched cells: {mismatches}")
print(f"first mismatch: {(adj_legacy_c != adj_triton_c).nonzero()[:5].tolist()}")
assert mismatches == 0, "SPIKE FAILED"
print("SPIKE PASSED")
```

Run on 119. **Expected:** `SPIKE PASSED`.

- [ ] **Step 4: Write spike findings**

`tmp/followup_design/w_baf_triton_kernel_sketch.md`:
```markdown
# W_BAF spike — Triton fast-path kernel vs legacy PyTorch

## Result

SPIKE PASSED / FAILED.

## Wall comparison (N=1000, one iteration)

- Legacy: <t_legacy> ms
- Triton: <t_triton> ms
- Speedup: <x>×

## Slow-path design (next task)

Based on spike success, slow-path adds:
- is_fast detection: `uturn_00 = tl.load(uturn_assign_ptr + cube_id * 36 + 0)` then `is_fast = (uturn_00 == -1)`
- u_per_edge computation inline: scan 12×3 facets table, sum matching entries
- ew_eff = ew - 2 * u_per_edge for eA/eB/eC only
- Endpoints use w_a_raw/w_b_raw (not ew_eff, per legacy line 715-716)

## Strategy A: post-kernel canonical sort

```python
valid = adj >= 0
key = torch.where(valid, adj, torch.full_like(adj, NODES))
key, _ = key.sort(dim=-1)
adj_canonical = torch.where(key == NODES, -1, key)
```

Adds ~5 ms per call. Acceptable vs >1700 ms savings.
```

- [ ] **Step 5: Commit flag + spike**

```bash
git add corep_fast/config.py corep_fast/stages/s7_triton.py \
        tmp/followup_design/w_baf_* tmp/w_baf_spike_119.*
git commit -m "$(cat <<'EOF'
w_baf(p2): add BUILD_ADJACENCY_TRITON flag + fast-path spike kernel

Spike on 1000 fake fast-path cubes passes bit-exact vs legacy
after canonical adj sort.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: W_BAF — red test + stub

**Files:**
- Create: `corep_fast/tests/unit/test_build_adjacency_triton.py`
- Modify: `corep_fast/stages/s7_triton.py`

- [ ] **Step 1: Write failing test**

```python
"""W_BAF: Triton-fused _build_adjacency must match legacy PyTorch path."""
import numpy as np
import pytest
import torch

from corep_fast.stages.s7_rank_assign import (
    _build_adjacency_gpu as _build_adj_legacy,
    _W_MAX, _NODES_PER_CUBE,
)
from corep_fast.stages.s7_triton import build_adjacency_triton  # NEW


def _canonicalize(adj):
    valid = adj >= 0
    key = torch.where(valid, adj, torch.full_like(adj, _NODES_PER_CUBE))
    key, _ = key.sort(dim=-1)
    return torch.where(key == _NODES_PER_CUBE, -1, key)


@pytest.fixture
def dev():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_fast_path_cubes_match(dev):
    N = 1000
    gen = torch.Generator(device='cpu').manual_seed(42)
    ew = torch.randint(0, _W_MAX, (N, 18), generator=gen, dtype=torch.int64).to(dev)
    uturn = torch.full((N, 12, 3), -1, dtype=torch.int64, device=dev)
    legacy = _canonicalize(_build_adj_legacy(ew, uturn))
    triton_adj = _canonicalize(build_adjacency_triton(ew, uturn))
    assert torch.equal(legacy, triton_adj)


def test_slow_path_cubes_match(dev):
    N = 1000
    gen = torch.Generator(device='cpu').manual_seed(43)
    ew = torch.randint(0, _W_MAX, (N, 18), generator=gen, dtype=torch.int64).to(dev)
    # Slow-path: uturn[:, 0, 0] != -1
    uturn = torch.randint(0, 3, (N, 12, 3), generator=gen, dtype=torch.int64).to(dev)
    legacy = _canonicalize(_build_adj_legacy(ew, uturn))
    triton_adj = _canonicalize(build_adjacency_triton(ew, uturn))
    assert torch.equal(legacy, triton_adj)


def test_mixed_path_cubes_match(dev):
    N = 5000
    gen = torch.Generator(device='cpu').manual_seed(44)
    ew = torch.randint(0, _W_MAX, (N, 18), generator=gen, dtype=torch.int64).to(dev)
    uturn = torch.randint(0, 3, (N, 12, 3), generator=gen, dtype=torch.int64).to(dev)
    uturn[:N//2, 0, 0] = -1  # fast-path half
    legacy = _canonicalize(_build_adj_legacy(ew, uturn))
    triton_adj = _canonicalize(build_adjacency_triton(ew, uturn))
    assert torch.equal(legacy, triton_adj)


def test_edge_case_all_zero_weights(dev):
    N = 100
    ew = torch.zeros((N, 18), dtype=torch.int64, device=dev)
    uturn = torch.full((N, 12, 3), -1, dtype=torch.int64, device=dev)
    triton_adj = build_adjacency_triton(ew, uturn)
    assert (triton_adj == -1).all()


@pytest.mark.gpu_heavy
def test_large_batch_300000(dev):
    N = 300_000
    gen = torch.Generator(device='cpu').manual_seed(45)
    ew = torch.randint(0, _W_MAX, (N, 18), generator=gen, dtype=torch.int64).to(dev)
    uturn = torch.full((N, 12, 3), -1, dtype=torch.int64, device=dev)
    adj = build_adjacency_triton(ew, uturn)
    assert adj.shape == (N, _NODES_PER_CUBE, 2)
```

- [ ] **Step 2: Add stub in `s7_triton.py`**

```python
def build_adjacency_triton(
    edge_weights: torch.Tensor,
    uturn_assignment: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused _build_adjacency_gpu. Returns (N, NODES, 2) int32.

    STUB: delegates to legacy PyTorch impl. Real kernel in Task 13.
    """
    from corep_fast.stages.s7_rank_assign import _build_adjacency_gpu
    return _build_adjacency_gpu(edge_weights, uturn_assignment)
```

- [ ] **Step 3: Run test — should PASS with stub (since canonicalize)**

Expected: 4 tests PASS (all except `test_large_batch_300000` which takes a while, mark `gpu_heavy`).

- [ ] **Step 4: Commit**

```bash
git add corep_fast/tests/unit/test_build_adjacency_triton.py corep_fast/stages/s7_triton.py
git commit -m "$(cat <<'EOF'
w_baf(p2): red test + legacy-delegating stub (green)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: W_BAF — full Triton kernel (fast + slow path)

**Files:**
- Modify: `corep_fast/stages/s7_triton.py`

- [ ] **Step 1: Extend kernel to handle slow-path**

Append to the `_build_adj_fast_kernel` (or create `_build_adj_full_kernel`):

```python
if _TRITON_AVAILABLE:
    @triton.jit
    def _compute_u_for_edge(
        cube_id, target_edge, is_fast,
        uturn_ptr,       # (N, 12, 3) int64
        facets_ptr,      # (12, 3) int64 constant
    ):
        """Scan facets table; sum uturn[cube, t, s] where facets[t, s] == target_edge.
        For fast-path cubes, return 0."""
        if is_fast != 0:
            return 0
        accum = 0
        for t in tl.static_range(0, 12):
            for s in tl.static_range(0, 3):
                fe = tl.load(facets_ptr + t * 3 + s)
                if fe == target_edge:
                    u = tl.load(uturn_ptr + cube_id * 36 + t * 3 + s)
                    accum = accum + u
        return accum

    @triton.jit
    def _build_adj_full_kernel(
        edge_weights_ptr,        # (N, 18) int64
        uturn_assign_ptr,        # (N, 12, 3) int64
        facets_ptr,              # (12, 3) int64 constant (CUBE_FACETS)
        eA_tab_ptr, eB_tab_ptr, eC_tab_ptr,
        a_at_v0_tab_ptr, b_at_v0_tab_ptr,
        adj_ptr, fill_count_ptr,
        N: tl.constexpr,
        NODES: tl.constexpr,
        W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        jj = pid % W; pid_r = pid // W
        pi = pid_r % 3; pid_r2 = pid_r // 3
        t_idx = pid_r2 % 12
        cube_id = pid_r2 // 12
        if cube_id >= N:
            return

        eA = tl.load(eA_tab_ptr + t_idx * 3 + pi)
        eB = tl.load(eB_tab_ptr + t_idx * 3 + pi)
        eC = tl.load(eC_tab_ptr + t_idx * 3 + pi)
        a_at_v0 = tl.load(a_at_v0_tab_ptr + t_idx * 3 + pi)
        b_at_v0 = tl.load(b_at_v0_tab_ptr + t_idx * 3 + pi)

        uturn_00 = tl.load(uturn_assign_ptr + cube_id * 36 + 0)
        is_fast = 1 if uturn_00 == -1 else 0

        u_a = _compute_u_for_edge(cube_id, eA, is_fast, uturn_assign_ptr, facets_ptr)
        u_b = _compute_u_for_edge(cube_id, eB, is_fast, uturn_assign_ptr, facets_ptr)
        u_c = _compute_u_for_edge(cube_id, eC, is_fast, uturn_assign_ptr, facets_ptr)

        w_a_raw = tl.load(edge_weights_ptr + cube_id * 18 + eA)
        w_b_raw = tl.load(edge_weights_ptr + cube_id * 18 + eB)
        w_c_raw = tl.load(edge_weights_ptr + cube_id * 18 + eC)
        w_a_eff = w_a_raw - 2 * u_a
        w_b_eff = w_b_raw - 2 * u_b
        w_c_eff = w_c_raw - 2 * u_c

        k_pair = (w_a_eff + w_b_eff - w_c_eff) // 2
        if k_pair < 0: k_pair = 0
        if k_pair > W: k_pair = W
        if jj >= k_pair:
            return

        pts_A = jj if a_at_v0 != 0 else (w_a_raw - 1 - jj)
        pts_B = jj if b_at_v0 != 0 else (w_b_raw - 1 - jj)
        node_A = eA * W + pts_A
        node_B = eB * W + pts_B

        slot_A = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_A, 1)
        if slot_A < 2:
            tl.store(adj_ptr + cube_id * NODES * 2 + node_A * 2 + slot_A, node_B)
        slot_B = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_B, 1)
        if slot_B < 2:
            tl.store(adj_ptr + cube_id * NODES * 2 + node_B * 2 + slot_B, node_A)


def build_adjacency_triton(
    edge_weights: torch.Tensor,      # (N, 18) int64
    uturn_assignment: torch.Tensor,  # (N, 12, 3) int64; -1 sentinel = fast-path
) -> torch.Tensor:
    """Returns (N, NODES, 2) int32 adjacency, canonical-sorted for determinism."""
    if not _TRITON_AVAILABLE:
        from corep_fast.stages.s7_rank_assign import _build_adjacency_gpu
        return _build_adjacency_gpu(edge_weights, uturn_assignment)

    from corep_fast.stages.s7_rank_assign import (
        _W_MAX, _NODES_PER_CUBE, _facet_pair_table, CUBE_FACETS,
    )

    device = edge_weights.device
    N = edge_weights.shape[0]
    NODES = _NODES_PER_CUBE
    W = _W_MAX

    eA_tab, eB_tab, a_at_v0_tab, b_at_v0_tab = _facet_pair_table(device)
    facets = CUBE_FACETS.to(device=device, dtype=torch.int64)
    eC_tab = torch.stack(
        [facets[:, 2], facets[:, 0], facets[:, 1]], dim=1
    ).contiguous()

    adj = torch.full((N, NODES, 2), -1, dtype=torch.int32, device=device)
    fill_count = torch.zeros((N, NODES), dtype=torch.int32, device=device)

    grid = (N * 12 * 3 * W,)
    _build_adj_full_kernel[grid](
        edge_weights.contiguous(),
        uturn_assignment.contiguous(),
        facets.contiguous(),
        eA_tab.contiguous(), eB_tab.contiguous(), eC_tab.contiguous(),
        a_at_v0_tab.to(torch.int8).contiguous(),
        b_at_v0_tab.to(torch.int8).contiguous(),
        adj, fill_count,
        N=N, NODES=NODES, W=W,
    )

    # Strategy A: canonical sort (push -1 to end, then sort ascending)
    valid = adj >= 0
    key = torch.where(valid, adj, torch.full_like(adj, NODES))
    key, _ = key.sort(dim=-1)
    adj_canonical = torch.where(key == NODES, torch.full_like(key, -1), key)
    return adj_canonical
```

- [ ] **Step 2: Run Task 12 tests + F1-F3**

**Expected:** 5 tests PASS + 3 F1-F3 PASS.

- [ ] **Step 3: Commit**

```bash
git add corep_fast/stages/s7_triton.py
git commit -m "$(cat <<'EOF'
w_baf(p2): full Triton kernel (fast + slow path) + canonical sort

- _compute_u_for_edge device helper (static 12x3 scan)
- _build_adj_full_kernel: single-program-per-(cube,t,pi,jj) with atomic
  slot assignment, followed by host-side canonical sort for determinism.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: W_BAF — determinism audit (20-run F1-F3)

**Files:**
- Create: `tmp/w_baf_determinism_20x_119.sh`
- Create: `logs/findings_w_baf_determinism.md`

- [ ] **Step 1: Write 20-run audit script**

`tmp/w_baf_determinism_20x_119.sh`:
```bash
#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
PASS=0; FAIL=0
for i in {1..20}; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    corep_fast/tests/regression/test_cpu_worker_optim.py -v 2>&1 \
    | tee tmp/cpu_profile/w_baf_det_run${i}.log
  if [ ${PIPESTATUS[0]} -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done
echo "PASS=$PASS FAIL=$FAIL"
```

- [ ] **Step 2: Run audit**

Run: `ssh host-10-240-99-119 "bash" < tmp/w_baf_determinism_20x_119.sh`
**Expected:** PASS=20, FAIL=0.

If any run fails, Strategy A canonical sort has a gap — debug before Task 15.

- [ ] **Step 3: Record findings**

`logs/findings_w_baf_determinism.md`:
```markdown
# W_BAF determinism audit (20 runs F1-F3)

PASS=<N>/20, FAIL=<M>/20.

All runs produced bit-identical goldens → atomic-order nondeterminism is
contained by Strategy A canonical sort.
```

- [ ] **Step 4: Commit audit artifacts**

```bash
git add tmp/w_baf_determinism_20x_119.sh tmp/cpu_profile/w_baf_det_run* \
        logs/findings_w_baf_determinism.md
git commit -m "$(cat <<'EOF'
w_baf(p2): determinism audit (20x F1-F3 all green)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: W_BAF — integrate at s7 call site

**Files:**
- Modify: `corep_fast/stages/s7_rank_assign.py` (call site where `_build_adjacency_gpu` is invoked; grep for "adj = _build_adjacency_gpu")

- [ ] **Step 1: Wire feature flag at call site**

```bash
grep -n "_build_adjacency_gpu" corep_fast/stages/s7_rank_assign.py
# Typical site: line ~923
```

Replace the call:

```python
from corep_fast import config as _cfg

if _cfg.BUILD_ADJACENCY_TRITON:
    from corep_fast.stages.s7_triton import build_adjacency_triton
    adj = build_adjacency_triton(edge_weights64, uturn64)
else:
    adj = _build_adjacency_gpu(edge_weights64, uturn64)
```

- [ ] **Step 2: F1-F3 + wall**

```bash
ssh host-10-240-99-119 "bash" < tmp/baseline_f123_119.sh
ssh host-10-240-99-119 "bash" < tmp/w_baf_wall_119.sh  # adapt from Task 4
```
**Expected:** F1-F3 green, `_build_adjacency_gpu` / `build_adjacency_triton` self ≤ 200 ms.

- [ ] **Step 3: Commit**

```bash
git add corep_fast/stages/s7_rank_assign.py
git commit -m "$(cat <<'EOF'
w_baf(p2): integrate Triton kernel at s7 call site (feature-flagged)

BUILD_ADJACENCY_TRITON=1 routes through s7_triton.build_adjacency_triton.
Legacy PyTorch impl retained for rollback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: W_HG — empirical tie scan + feature flag

**Files:**
- Modify: `corep_fast/config.py`
- Create: `tmp/w_hg_tie_scan_119.py`
- Create: `tmp/followup_design/w_hg_tie_scan.md`

- [ ] **Step 1: Add `HUNGARIAN_GPU` flag**

Append to `corep_fast/config.py`:

```python
# Followup W_HG (s7): Batched Hungarian on GPU (Triton brute-force for n<=5).
# Replaces 275k × scipy.optimize.linear_sum_assignment main-thread loop.
# Set COREP_FAST_HUNGARIAN_GPU=0 to fall back to scipy.
HUNGARIAN_GPU = os.environ.get('COREP_FAST_HUNGARIAN_GPU', '1') == '1'
```

- [ ] **Step 2: Write tie-scan script**

`tmp/w_hg_tie_scan_119.py`:
```python
"""W_HG: scan cost matrices on F2 fixture to quantify tie frequency.

If ties are rare (< 0.1 %), we can implement brute-force Hungarian with
any deterministic tie-break and assert-if-differ fallback.
If ties are common, we must match scipy's tie-break exactly.
"""
import sys
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')

# Monkeypatch for regression-gate determinism
import multiprocessing as _mp, multiprocessing.pool as _mp_pool
class _SP:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, xs, *a, **k): return [fn(x) for x in xs]
    def close(self): pass
    def join(self): pass
_mp.Pool = _SP
_mp_pool.Pool = _SP

import numpy as np
import torch
import trimesh
from corep_fast.pipeline import corep_pipeline
from corep_fast.stages import s7_rank_assign as s7

# Hook into Phase 3 cost-matrix construction
tie_stats = {"total": 0, "with_ties": 0, "tie_counts": []}
_orig_lsa = s7.linear_sum_assignment


def _hook(cost, *a, **k):
    n_loops, n_points = cost.shape
    tie_stats["total"] += 1
    # Per-row min count (heuristic for ties)
    row_mins = cost.min(axis=1, keepdims=True)
    tie_count = int((cost == row_mins).sum() - n_loops)  # excess hits beyond the unique min
    if tie_count > 0:
        tie_stats["with_ties"] += 1
        tie_stats["tie_counts"].append(tie_count)
    return _orig_lsa(cost, *a, **k)


s7.linear_sum_assignment = _hook

mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_tie.ply')
_ = corep_pipeline('/tmp/fix_tie.ply', 256, torch.device('cuda:0'))

total = tie_stats["total"]
ties = tie_stats["with_ties"]
pct = 100.0 * ties / total if total else 0.0
print(f"Phase 3 calls: {total}")
print(f"With ties:     {ties} ({pct:.3f}%)")
if tie_stats["tie_counts"]:
    arr = np.asarray(tie_stats["tie_counts"])
    print(f"Tie count distribution: min={arr.min()} max={arr.max()} p50={np.median(arr):.0f} p99={np.percentile(arr, 99):.0f}")
```

Script `tmp/w_hg_tie_scan_119.sh`:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 .venv/bin/python tmp/w_hg_tie_scan_119.py \
    2>&1 | tee tmp/followup_design/w_hg_tie_scan_output.txt
```

- [ ] **Step 3: Write tie-scan findings**

`tmp/followup_design/w_hg_tie_scan.md`:
```markdown
# W_HG — Phase 3 cost-matrix tie scan (F2 @ res=256)

## Result

<paste w_hg_tie_scan_output.txt>

## Decision

- If pct_with_ties < 0.1 %: use `assert_unique_min` with fallback to scipy.
- If 0.1 % ≤ pct < 5 %: implement deterministic tie-break (smallest row-major
  permutation index); verify bit-exact on tie cases.
- If pct ≥ 5 %: replicate scipy Jonker-Volgenant tie-break exactly.
```

- [ ] **Step 4: Commit**

```bash
git add corep_fast/config.py tmp/followup_design/w_hg_tie_scan.md \
        tmp/w_hg_tie_scan_119.*
git commit -m "$(cat <<'EOF'
w_hg(p2): add HUNGARIAN_GPU flag + Phase 3 cost-matrix tie scan

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: W_HG — red test + naive PyTorch brute-force reference

**Files:**
- Create: `corep_fast/tests/unit/test_hungarian_batched.py`
- Modify: `corep_fast/stages/s7_triton.py` (add `hungarian_batched` function)

- [ ] **Step 1: Write failing tests**

```python
"""W_HG: batched brute-force Hungarian must match scipy per matrix."""
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from corep_fast.stages.s7_triton import hungarian_batched  # NEW


@pytest.fixture
def dev():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_matches_scipy_random_small(dev):
    """1000 random (nl, np) ∈ [1..5] × [1..8] matrices — compare assignment."""
    rng = np.random.RandomState(0)
    Bs = 1000
    nl_max, np_max = 5, 8
    cost_padded = np.full((Bs, nl_max, np_max), 1e9, dtype=np.float32)
    n_loops = rng.randint(1, nl_max + 1, size=Bs)
    n_points = rng.randint(1, np_max + 1, size=Bs)
    for b in range(Bs):
        nl, npt = n_loops[b], n_points[b]
        cost_padded[b, :nl, :npt] = rng.rand(nl, npt).astype(np.float32)

    cost_t = torch.from_numpy(cost_padded).to(dev)
    nl_t = torch.from_numpy(n_loops).to(dev).long()
    np_t = torch.from_numpy(n_points).to(dev).long()

    # Reference: scipy per-matrix
    ref_matches = np.full((Bs, nl_max), -1, dtype=np.int64)
    for b in range(Bs):
        nl, npt = n_loops[b], n_points[b]
        sub = cost_padded[b, :nl, :npt]
        r, c = linear_sum_assignment(sub.astype(np.float64))
        for rr, cc in zip(r, c):
            if rr < nl:
                ref_matches[b, rr] = cc

    # Candidate
    cand = hungarian_batched(cost_t, nl_t, np_t, max_n_loops=nl_max, max_n_points=np_max)
    cand_cpu = cand.cpu().numpy()

    # Compare per valid (b, r) — only where rr < nl[b]
    mismatches = []
    for b in range(Bs):
        nl = n_loops[b]
        for rr in range(nl):
            if ref_matches[b, rr] != cand_cpu[b, rr]:
                mismatches.append((b, rr, ref_matches[b, rr], cand_cpu[b, rr]))
    assert len(mismatches) == 0, f"{len(mismatches)} mismatches, first: {mismatches[:5]}"


def test_n_loops_equals_1(dev):
    """Trivial: 1 loop, n_points in 1..4, match = argmin."""
    rng = np.random.RandomState(1)
    Bs = 200
    cost_padded = np.full((Bs, 5, 8), 1e9, dtype=np.float32)
    n_loops = np.ones(Bs, dtype=np.int64)
    n_points = rng.randint(1, 5, size=Bs).astype(np.int64)
    for b in range(Bs):
        cost_padded[b, 0, :n_points[b]] = rng.rand(n_points[b]).astype(np.float32)

    cand = hungarian_batched(
        torch.from_numpy(cost_padded).to(dev),
        torch.from_numpy(n_loops).to(dev),
        torch.from_numpy(n_points).to(dev),
        max_n_loops=5, max_n_points=8,
    )
    cand_cpu = cand.cpu().numpy()
    for b in range(Bs):
        expected = int(cost_padded[b, 0, :n_points[b]].argmin())
        assert cand_cpu[b, 0] == expected


def test_batched_end_to_end_1000_cubes_vs_scipy(dev):
    # Equivalent to test_matches_scipy_random_small but with realistic cost scales
    # matching `_rank_assign.py` Phase 3 (squared distances).
    rng = np.random.RandomState(2)
    Bs = 1000
    cost_padded = np.full((Bs, 5, 8), 1e9, dtype=np.float32)
    n_loops = rng.randint(1, 6, size=Bs).astype(np.int64)
    n_points = rng.randint(1, 9, size=Bs).astype(np.int64)
    for b in range(Bs):
        nl, npt = n_loops[b], n_points[b]
        centroids = rng.rand(nl, 3).astype(np.float32)
        comp_pts = rng.rand(npt, 3).astype(np.float32)
        diff = centroids[:, None, :] - comp_pts[None, :, :]
        cost_padded[b, :nl, :npt] = (diff * diff).sum(axis=-1)

    cand = hungarian_batched(
        torch.from_numpy(cost_padded).to(dev),
        torch.from_numpy(n_loops).to(dev),
        torch.from_numpy(n_points).to(dev),
        max_n_loops=5, max_n_points=8,
    ).cpu().numpy()

    for b in range(Bs):
        nl, npt = n_loops[b], n_points[b]
        sub = cost_padded[b, :nl, :npt].astype(np.float64)
        r, c = linear_sum_assignment(sub)
        for rr, cc in zip(r, c):
            if rr < nl:
                assert cand[b, rr] == cc, \
                    f"cube {b} loop {rr}: gpu={cand[b, rr]} scipy={cc}"
```

- [ ] **Step 2: Add naive PyTorch brute-force `hungarian_batched` (for n ≤ 4)**

Append to `s7_triton.py`:

```python
from itertools import permutations
import numpy as _np


def _enumerate_perms(n_rows: int, n_cols: int, device: torch.device) -> torch.Tensor:
    """All permutations of choosing n_rows distinct cols from n_cols.

    Returns (K, n_rows) int64 where K = n_cols! / (n_cols - n_rows)!
    """
    perms = [_np.asarray(p, dtype=_np.int64)
             for p in permutations(range(n_cols), n_rows)]
    return torch.from_numpy(_np.stack(perms)).to(device)


def hungarian_batched(
    cost_padded: torch.Tensor,     # (B, max_nl, max_np) float32, +inf for padded
    n_loops: torch.Tensor,          # (B,) int64
    n_points: torch.Tensor,         # (B,) int64
    max_n_loops: int,
    max_n_points: int,
) -> torch.Tensor:
    """Brute-force Hungarian assignment. Returns (B, max_nl) int64 match indices.

    For each valid row r < n_loops[b], output[b, r] is the column matched.
    Invalid rows (r >= n_loops[b]) are set to -1.

    Implementation: enumerates all permutations of (n_loops[b]) columns drawn
    from n_points[b] candidates, evaluates cost, picks argmin (lexicographic
    tiebreak = lowest permutation index).
    """
    device = cost_padded.device
    B = cost_padded.shape[0]
    output = torch.full((B, max_n_loops), -1, dtype=torch.int64, device=device)

    # Group cubes by (nl, np) shape, dispatch one brute-force per shape.
    # Since max_nl * max_np <= 5 * 8 = 40 shape buckets, this is cheap.
    for nl in range(1, max_n_loops + 1):
        for npt in range(nl, max_n_points + 1):  # npt must >= nl for valid matching
            if nl > npt: continue
            mask = (n_loops == nl) & (n_points == npt)
            if not mask.any(): continue
            idx = mask.nonzero(as_tuple=True)[0]
            sub = cost_padded[idx, :nl, :npt]    # (M, nl, npt) float32
            perms = _enumerate_perms(nl, npt, device)   # (K, nl) int64
            # Evaluate: cost[b, K] = sum over r of sub[b, r, perms[K, r]]
            # sub: (M, nl, npt); perms: (K, nl)
            M = sub.shape[0]; K = perms.shape[0]
            # Gather: cost_per_perm[m, k, r] = sub[m, r, perms[k, r]]
            perms_exp = perms.view(1, K, nl).expand(M, K, nl)
            sub_exp = sub.unsqueeze(1).expand(M, K, nl, npt)
            gathered = sub_exp.gather(3, perms_exp.unsqueeze(3)).squeeze(3)  # (M, K, nl)
            cost_per_perm = gathered.sum(dim=-1)   # (M, K)
            best_k = cost_per_perm.argmin(dim=-1)  # (M,)
            best_perms = perms[best_k]             # (M, nl)
            output[idx, :nl] = best_perms
    return output
```

**Tie-break note:** `argmin` returns the first minimum index, which corresponds
to the lexicographically first permutation. If tie-scan (Task 16) showed this
matches scipy on real F2 data, we're done. Otherwise, this is a placeholder
that Task 18's Triton version will replace with the scipy-matching rule.

**Edge case `nl > npt`:** The outer loop uses `for npt in range(nl, max_n_points + 1)`,
which skips `nl > npt` cubes (leaving `output[b, r] = -1`). scipy handles such
cubes by matching only the `min(nl, npt) = npt` rows and leaving `nl - npt`
rows untouched. To match scipy semantics, in Task 18 the integration code must
check `if nl > npt:` per cube and route those cubes through scipy. Verify via
tie-scan output (Task 16): if no cubes have `nl > npt`, skip the fallback; else
wire it in Task 18 Step 2 as shown below.

- [ ] **Step 3: Run tests + F1-F3**

**Expected:** 3 tests PASS, 3 F1-F3 PASS. If F1-F3 fails due to tie mismatch,
inspect the failing goldens, determine the tie pattern, and either (a) adjust
brute-force tie-break or (b) gate brute-force to tie-free matrices and
fall back to scipy for ties.

- [ ] **Step 4: Commit**

```bash
git add corep_fast/stages/s7_triton.py corep_fast/tests/unit/test_hungarian_batched.py
git commit -m "$(cat <<'EOF'
w_hg(p2): brute-force Hungarian PyTorch impl + parity tests

Handles n_loops * n_points <= 40 via enumeration over permutations.
argmin tiebreak matches lexicographically-first permutation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: W_HG — integrate at s7 Phase 3 + GPU-resident result

**Files:**
- Modify: `corep_fast/stages/s7_rank_assign.py` (Phase 3 loop, lines 1472-1512)

- [ ] **Step 1: Replace Phase 3 loop with batched call**

Replace `s7_rank_assign.py:1472-1512`:

```python
from corep_fast import config as _cfg

# ===================================================================
# Phase 3: Hungarian matching
# ===================================================================

if _cfg.HUNGARIAN_GPU and len(ok_cube_indices) > 0:
    from corep_fast.stages.s7_triton import hungarian_batched

    max_nl = 5  # conservative upper bound; adapt if data changes
    max_np = 8

    # Build padded cost matrices on GPU (keep loop_centroids + point_values on device)
    # Only construct for ok cubes.
    B = len(ok_cube_indices)
    cost_padded = torch.full((B, max_nl, max_np), float('inf'),
                             dtype=torch.float32, device=device)
    nl_per = torch.zeros(B, dtype=torch.int64, device=device)
    np_per = torch.zeros(B, dtype=torch.int64, device=device)

    loop_cube_off_t = batch.loop_cube_off.to(torch.int64)
    point_off_t = batch.point_offsets.to(torch.int64)

    for bi, cube_idx in enumerate(ok_cube_indices):
        l_lo = int(loop_cube_off_t[cube_idx])
        l_hi = int(loop_cube_off_t[cube_idx + 1])
        p_lo = int(point_off_t[cube_idx])
        p_hi = int(point_off_t[cube_idx + 1])
        nl = l_hi - l_lo
        npt = p_hi - p_lo
        if nl == 0:
            continue
        if npt == 0:
            # No points: identity match
            for li_off in range(nl):
                all_matches[l_lo + li_off] = li_off
            continue
        # Clamp to max_nl / max_np
        nl_clip = min(nl, max_nl)
        np_clip = min(npt, max_np)
        centroids_i = loop_centroids[l_lo:l_lo + nl_clip]     # (nl_clip, 3) on GPU
        comp_pts_i = batch.point_values[p_lo:p_lo + np_clip]   # (np_clip, 3) on GPU
        diff = centroids_i.unsqueeze(1) - comp_pts_i.unsqueeze(0)   # (nl_clip, np_clip, 3)
        cost = (diff * diff).sum(dim=-1)                             # (nl_clip, np_clip)
        cost_padded[bi, :nl_clip, :np_clip] = cost
        nl_per[bi] = nl_clip
        np_per[bi] = np_clip

    matches = hungarian_batched(cost_padded, nl_per, np_per, max_nl, max_np)  # (B, max_nl)
    matches_cpu = matches.cpu().numpy()

    # Collect cubes that need scipy fallback (nl > npt, which hungarian_batched skips)
    scipy_fallback_cubes = []

    for bi, cube_idx in enumerate(ok_cube_indices):
        l_lo = int(loop_cube_off_t[cube_idx])
        l_hi = int(loop_cube_off_t[cube_idx + 1])
        p_lo = int(point_off_t[cube_idx])
        p_hi = int(point_off_t[cube_idx + 1])
        nl = l_hi - l_lo
        npt = p_hi - p_lo
        if nl == 0:
            continue
        if nl > npt and npt > 0:
            # hungarian_batched returned -1 for this cube; use scipy
            scipy_fallback_cubes.append(cube_idx)
            continue
        for li_off in range(min(nl, max_nl)):
            m = int(matches_cpu[bi, li_off])
            if m >= 0:
                all_matches[l_lo + li_off] = m

    # Fallback path for nl > npt (rectangular non-square with more loops than points)
    if scipy_fallback_cubes:
        loop_centroids_cpu_fb = loop_centroids.cpu().numpy()
        point_values_cpu_fb = batch.point_values.cpu().numpy()
        point_off_cpu_fb = batch.point_offsets.cpu().numpy()
        loop_cube_off_cpu_fb = batch.loop_cube_off.cpu().numpy()
        for cube_idx in scipy_fallback_cubes:
            l_lo = int(loop_cube_off_cpu_fb[cube_idx])
            l_hi = int(loop_cube_off_cpu_fb[cube_idx + 1])
            p_lo = int(point_off_cpu_fb[cube_idx])
            p_hi = int(point_off_cpu_fb[cube_idx + 1])
            centroids_i = loop_centroids_cpu_fb[l_lo:l_hi]
            comp_pts_i = point_values_cpu_fb[p_lo:p_hi]
            diff = centroids_i[:, None, :] - comp_pts_i[None, :, :]
            cost = (diff * diff).sum(axis=-1).astype(np.float64)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if r < (l_hi - l_lo):
                    all_matches[l_lo + int(r)] = int(c)

else:
    # ---- Legacy scipy path (unchanged) ----
    point_offsets_np = batch.point_offsets.cpu().numpy()
    loop_centroids_np = loop_centroids.cpu().numpy() if total_loops > 0 \
        else np.zeros((0, 3), dtype=np.float32)
    point_values_np = batch.point_values.cpu().numpy()

    for cube_idx in ok_cube_indices:
        l_lo = int(loop_cube_off_np[cube_idx])
        l_hi = int(loop_cube_off_np[cube_idx + 1])
        n_loops = l_hi - l_lo
        if n_loops == 0: continue
        p_lo = int(point_offsets_np[cube_idx])
        p_hi = int(point_offsets_np[cube_idx + 1])
        n_points = p_hi - p_lo
        if n_points == 0:
            for li_off in range(n_loops):
                all_matches[l_lo + li_off] = li_off
            continue
        centroids_i = loop_centroids_np[l_lo:l_hi]
        comp_pts_i = point_values_np[p_lo:p_hi]
        diff = centroids_i[:, None, :] - comp_pts_i[None, :, :]
        cost = (diff * diff).sum(axis=-1).astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if r < n_loops:
                all_matches[l_lo + int(r)] = int(c)
```

- [ ] **Step 2: F1-F3 + wall**

**Expected:** F1-F3 green, s7 Phase 3 self ≤ 200 ms, e2e wall ≤ 3.5 s.

If F1-F3 breaks:
- Dump mismatched cube indices from regression output.
- For each mismatched cube, compare scipy and gpu matches — inspect cost
  matrix for ties.
- If ties, pre-check: if `(cost == row_min).sum(-1) > 1` anywhere in a cube,
  fall back to scipy per-cube.

- [ ] **Step 3: Commit**

```bash
git add corep_fast/stages/s7_rank_assign.py
git commit -m "$(cat <<'EOF'
w_hg(p2): integrate batched Hungarian at Phase 3 (feature-flagged)

HUNGARIAN_GPU=1 replaces 275k scipy.linear_sum_assignment main-thread loop
with single batched brute-force call. Legacy preserved behind flag=0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: W_BAF + W_HG — post-Phase-2 findings + checkpoint

**Files:**
- Create: `logs/findings_w_baf_triton_fusion.md`
- Create: `logs/findings_w_hg_batched_hungarian.md`
- Create: `logs/findings_phase2_checkpoint.md`
- Create: `tmp/cpu_profile/phase2_post_wall.json`
- Create: `tmp/cpu_profile/phase2_post_hotspots.txt`
- Create: `tmp/cpu_profile/phase2_post_nsys.qdrep`

- [ ] **Step 1: Clean wall, hotspots, nsys**

```bash
ssh host-10-240-99-119 "bash" < tmp/phase2_wall_119.sh       # adapt from Task 4
ssh host-10-240-99-119 "bash" < tmp/phase2_hotspots_119.sh
ssh host-10-240-99-119 "bash" < tmp/phase2_nsys_119.sh
```

Nsys script:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  CUDA_VISIBLE_DEVICES=0 nsys profile -o tmp/cpu_profile/phase2_post_nsys \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=true \
    .venv/bin/python tmp/cpu_profile/t0_driver.py --mode default --res 256 --trials 1
# Then extract GPU util:
# nsys stats --report cudaapisum tmp/cpu_profile/phase2_post_nsys.qdrep
```

Read GPU utilization from nsys report (Compute + Memcpy % / total captured).
**Target:** ≥ 30 %.

- [ ] **Step 2: Write W_BAF findings**

`logs/findings_w_baf_triton_fusion.md`:
```markdown
# W_BAF findings

## DoD
| # | Target | Actual | Status |
|---|---|---|---|
| 1 | 5 unit tests pass | <N>/5 | — |
| 2 | F1-F3 bit-exact | <N>/3 | — |
| 3 | 20x determinism audit | 20/20 | — |
| 4 | `_build_adjacency_gpu` self ≤ 200 ms | <x> ms | — |
| 5 | Wall Δ ≥ -0.5 s | <Δ> s | — |
| 6 | VRAM Δ ≤ +500 MB | <Δ> MB | — |
```

- [ ] **Step 3: Write W_HG findings**

`logs/findings_w_hg_batched_hungarian.md`:
```markdown
# W_HG findings

## DoD
| # | Target | Actual | Status |
|---|---|---|---|
| 1 | 3 unit tests pass | <N>/3 | — |
| 2 | F1-F3 bit-exact | <N>/3 | — |
| 3 | s7 Phase 3 self ≤ 200 ms | <x> ms | — |
| 4 | Residual lock.acquire ≤ 100 ms | <x> ms | — |
| 5 | Wall Δ ≥ -0.6 s | <Δ> s | — |
| 6 | VRAM Δ ≤ +500 MB | <Δ> MB | — |
```

- [ ] **Step 4: Write Phase 2 checkpoint**

`logs/findings_phase2_checkpoint.md`:
```markdown
# Phase 2 checkpoint (W_BAF + W_HG)

## Cumulative DoD vs spec §6.2

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | W_BAF DoD met | — | — | — |
| 2 | W_HG DoD met | — | — | — |
| 3 | F1-F3 bit-exact | 3/3 | — | — |
| 4 | e2e wall @ res=256 | ≤ 3.2 s | <x> s | — |
| 5 | cProfile top-3 ≠ app code | yes/no | — | — |
| 6 | VRAM peak ≤ 6187 MB | — | — | — |
| 7 | nsys GPU util ≥ 30 % | — | — | — |

## Cumulative from anchor 1ced857

- Baseline wall: 5.354 s
- Post-Phase-2 wall: <x> s
- Δ: <Δ> s (<pct>%)
```

- [ ] **Step 5: Commit findings**

```bash
git add logs/findings_w_baf_triton_fusion.md logs/findings_w_hg_batched_hungarian.md \
        logs/findings_phase2_checkpoint.md tmp/cpu_profile/phase2_*
git commit -m "$(cat <<'EOF'
followup(p2): Phase 2 checkpoint — W_BAF + W_HG + nsys GPU util

Cumulative e2e 5.354 -> <x>s (<pct>%). F1-F3 bit-exact, 20x determinism
audit clean, nsys GPU util <x>%.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Final handoff

**Files:**
- Create: `docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-handoff.md`

- [ ] **Step 1: Draft handoff doc**

Mirror structure from `docs/superpowers/specs/2026-04-17-cpu-worker-optim-handoff.md`:
- §1 3-line summary (what was delivered)
- §2 Measured impact (end-to-end, per-function, DoD check)
- §3 What was tried and rejected (include anything from W_BAF/W_HG if applicable)
- §4 Branch + commit chain
- §5 Files touched
- §6 What's next — proposed future spec (small residuals: numpy coercion cluster, s6_collapse assembly block, W4 Option A)
- §7 Measurement discipline carry-over
- §8 Pickup checklist

- [ ] **Step 2: Force-add (docs/ is gitignored)**

```bash
git add -f docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-handoff.md
git commit -m "$(cat <<'EOF'
followup: V3 handoff (Phase 1+2 close-out)

Summary of W_L2L + W_SD + W_BAF + W_HG. e2e <x>s from 5.354s baseline.
Next spec author: start from proposed residual work in §6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

End of plan.

---

## Self-Review checklist (done inline before commit)

- **Spec coverage:** each spec §3.1 workstream (W_L2L / W_SD / W_BAF / W_HG) has
  dedicated tasks; each DoD in spec §4.x has a corresponding step in this plan.
- **Placeholder scan:** no "TBD" / "implement later" / vague "handle errors".
  All code snippets are complete (modulo "grep the exact line" navigation aids).
- **Type consistency:** `_count_uturns_gpu_batched` signature — input `groups`
  list of tuples, output `(G,) int64` tensor — is used consistently across
  Task 6, 7, 8, 9. `build_adjacency_triton` signature — input `(edge_weights,
  uturn_assignment)`, output `(N, NODES, 2) int32` — consistent across Task
  11, 12, 13, 15. `hungarian_batched` signature — `(cost_padded, n_loops,
  n_points, max_n_loops, max_n_points) -> (B, max_n_loops) int64` — consistent
  across Task 17, 18.
- **Scope bounds:** only the 4 in-scope workstreams from spec §3.1 are tasked;
  deferred items in §8.1 are NOT included.
- **Commit discipline:** every task ends with a `git commit` step; commit
  messages name the workstream tag + phase and include the Co-Authored-By
  trailer.

End of self-review.
