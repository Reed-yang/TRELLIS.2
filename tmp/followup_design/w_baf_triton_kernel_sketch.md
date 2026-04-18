# W_BAF spike findings (Task 11)

**Branch:** `post-profile-sync-elim`  **Date:** 2026-04-17
**Test node:** `host-10-240-99-116` GPU 4, CUDA_VISIBLE_DEVICES=4
**Fixture:** `trimesh.creation.icosphere(subdivisions=3, radius=0.4)` @ res=256

## Step 1: Real `_build_adjacency_gpu` wall in F2 pipeline

```
_build_adjacency_gpu calls: 1
Total wall: 38.61 ms
N values: [275541]
Avg ms/call: 38.61
```

**Verdict:** The function is called **exactly once** per pipeline run with
N=275541 cubes, taking **38.61 ms** of wall time (with
`torch.cuda.synchronize()` fences on either side — i.e. full device-visible
wall, not CPU self). The spec §2.1 figure "1913 ms self" was wrong for
this routine: it is NOT a 1.9-second hotspot. This confirms
`logs/findings_t0_baseline_concerns.md`.

## Step 2: Source inspection notes

- `corep_fast/stages/s7_rank_assign.py:634-800` — `_build_adjacency_gpu`.
- Constants: `_W_MAX = 16`, `_NODES_PER_CUBE = 18 * 16 = 288`.
- Current impl runs a `12 * 3 * W_MAX = 576` Python-level for-loop wrapping
  batched scatters. Each iteration: compute `valid_arc` mask, `mask.any()`
  early-exit, `cube_arange[mask]`, then paired adj/fill_count writes for
  both A→B and B→A directions.
- Tables used: `_facet_pair_table(device)` returns `(eA, eB, a_at_v0, b_at_v0)`
  each (12,3); `CUBE_FACETS` gives eC via `stack([c[:,2],c[:,0],c[:,1]])`.
- Fast-path detection: `is_fast = (uturn_assignment[:, 0, 0] == -1)`; slow-path
  adds U-turn correction + additional scatter loop (lines ~816-900).

## Step 3 & 4: Triton fast-path kernel + timing

Spike kernel: `corep_fast/stages/s7_triton.py::_build_adj_fast_kernel`.

- Grid = `N * 12 * 3 * W_MAX`, one scalar work-item per arc candidate.
- `tl.atomic_add` on `fill_count`, `tl.store` into `adj` if slot < 2.
- Fast-path only (uses `ew` directly, no u-correction).

### Synthetic random `ew` benchmark (uturn all -1)

| N       | Legacy ms | Triton ms | Speedup | Saved ms |
|---------|----------:|----------:|--------:|---------:|
| 1,000   | 151.26    | 0.46      | 329.7×  | +150.80  |
| 10,000  | 181.61    | 4.00      |  45.4×  | +177.61  |
| 100,000 | 234.87    | 40.04     |   5.87× | +194.83  |
| 275,541 | 225.53    | 110.26    |   2.05× | +115.27  |

> Random `ew ∈ [0,16)` forces ~all 576 inner iters to be non-empty, which
> makes legacy's `mask.any()` early-exit useless. Parity check fails on
> random data because legacy assumes `k_pair ≤ ew`, which random violates
> (causes pts_A = w_a-1-jj to go negative and wrap to node=-1-like slots).

### Real F2 data benchmark (captured via monkeypatch)

```
Captured 1 calls; N=275541; fast-path fraction = 0.9996 (275428/275541)

Legacy median wall: 38.264 ms
Triton median wall: 95.780 ms
Speedup: 0.40x
Saved per call: -57.516 ms

Fast-only parity: 4 mismatches out of 275428 cubes
  (e.g. fast-cube[40690] node[287]: legacy=[-1,-1], triton=[224,-1])
```

**Critical observations:**
1. On real production tensors Triton is **2.5× slower** than legacy
   (−57.5 ms regression). Real `ew` is sparse: most `k_pair` clamp to 0,
   so legacy's `mask.any()` short-circuits hundreds of the 576 iters for
   free; the Triton kernel must launch N×12×3×W = ~158 M programs regardless.
2. The 4 parity mismatches all pin at `node[287]` (last node of last edge) —
   probably a late-bound endpoint race inherent to the "atomic slot" design
   (two atomic_adds see slot < 2 but the non-atomic `tl.store` lands
   out of program order). A real impl would need `atomic_cas` or
   reduce+recompute, not just atomic_add+store.
3. Fast-path fraction is **99.96%** on the icosphere fixture; slow-path
   coverage would add kernel complexity for essentially zero real benefit.

## Step 5: Decision — ESCALATE_DESCOPE

Real-F2 wall ceiling for **ANY** replacement of `_build_adjacency_gpu` is
**38.26 ms** (that's all there is to save). The spike's Triton kernel actually
regresses by ~57 ms on real data, so even a refined kernel would need to:

1. Fully eliminate the 38 ms, AND
2. Not regress slow-path (0.04 % of cubes), AND
3. Achieve correct atomic 2-slot assignment (spike shows 4 races already).

…for a **best-case 38 ms wall saving**, which is **below the 100 ms escalation
threshold** documented in `logs/findings_t0_baseline_concerns.md` §"Mitigation".

**Recommendation:** Descope Tasks 11-15 (W_BAF) entirely. Reclaim the
engineering budget for other Phase 2 items with verified ROI (e.g. W_HG,
W_CMB) or Phase 3 prep work.

**Reasoning:**
- ROI ceiling 38 ms < 100 ms escalation threshold.
- Spike kernel regresses on real data, so "just ship the spike" is not viable.
- A correctness-complete kernel requires either atomic_cas or a multi-pass
  reduction — non-trivial work for ≤38 ms wall gain.
- Fast-path fraction is 99.96% on icosphere but slow-path ratio is
  workload-dependent; chasing the last 0.04 % adds maintenance surface
  for no budget reason.

## Appendix: Triton fast-kernel file

Left in tree as `corep_fast/stages/s7_triton.py` so future sessions can
rerun the spike via `tmp/w_baf_spike_real_116.sh`.
`BUILD_ADJACENCY_TRITON` flag remains default-off in `corep_fast/config.py`
(no call site added — flag is unreferenced for now).
