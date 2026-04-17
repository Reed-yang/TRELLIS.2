# W7 — s7 orchestration drill-down

**Date:** 2026-04-17
**HEAD:** `ff461a31b1586d5264b5361c5e3921dce8b41c2f` (post-W4+W5)
**Profile source:** `tmp/cpu_profile/results_main/main_thread_res256_post_w5.prof`
**Task:** T8a — spike to determine if there is a ≥200 ms mechanical win in
`s7_rank_assign.py` orchestration / residual work that justifies T8b.

## Top-20 s7 hotspots (post-W5)

```
fn                                                  line     calls     self_ms      cum_ms
_build_adjacency_gpu                                 634         1      1913.7      1925.1
s7_rank_assign                                      1175         1       984.6      3635.2
_get_ordered_points                                   53      1540         9.7         9.9
_phase1_gpu_rank_assign                              903         1         6.9      2043.3
_trace_with_ranks_uturn_assignment                   154        99         2.5        15.6
<genexpr>                                           1143      4752         0.9         0.9
_assign_face_connections                             190      3564         0.9        10.8
<genexpr>                                           1142      1287         0.7         1.6
_get_common_vertex                                    42      4020         0.7         0.7
_trace_loops_from_adj                                106        99         0.6         0.7
_assign_uturns                                       204      3564         0.4         0.5
_match_loops_to_ranks                                260        99         0.3         0.6
<dictcomp>                                           166        99         0.3         0.3
_s7_rank_worker                                      428        99         0.2        16.3
<listcomp>                                           230       872         0.1         0.1
<genexpr>                                            284       971         0.1         0.1
<listcomp>                                            62       757         0.1         0.1
_get_k_for_vertex                                    181       200         0.1         0.2
<listcomp>                                           285        99         0.1         0.1
<listcomp>                                           280        99         0.0         0.0
```

## Direct callees of `s7_rank_assign` (self-time breakdown)

```
fn                                                 file                   calls   self_ms   cum_ms
_phase1_gpu_rank_assign                            s7_rank_assign.py         1      6.89  2043.3
<method 'sum' of 'numpy.ndarray' objects>          <builtin>            275539     40.71   306.3
<built-in method scipy.optimize._lsap.linear_sum_a <builtin>            275539    152.25   152.3
<method 'astype' of 'numpy.ndarray' objects>       <builtin>            275539     77.43    77.4
<method 'cpu' of 'torch._C.TensorBase' objects>    <builtin>                11     29.15    29.1
<built-in method torch.repeat_interleave>          <builtin>                 4     18.58    18.6
<method 'append' of 'list' objects>                <builtin>            275539      9.90     9.9
<built-in method torch.tensor>                     <builtin>                 1      8.37     8.4
<method 'to' of 'torch._C.TensorBase' objects>     <builtin>                11      4.96     5.0
```

Total cum from callees: 2650.6 ms. `s7_rank_assign` self=984.6 ms is pure-Python
overhead, overwhelmingly dominated by the **Phase 3 per-cube loop** (line
1485–1512) iterating 275 539 times (≈ #OK cubes).

## Candidates with ≥200 ms mechanical win

**None found.**

Rationale for each inspected hotspot:

### `_build_adjacency_gpu` (1913.7 ms self, line 634)
Genuine GPU compute (builds per-cube (edge,rank) adjacency via 12·3·W=576
batched scatter iterations + U-turn loop). Already fully vectorized on GPU;
self-time is CUDA kernel-launch overhead and real math. **Not a mechanical
win** — would require Triton fusion (tracked separately under the Triton K1
work item), not a residual cleanup.

### `s7_rank_assign` (984.6 ms self, line 1175)
The 984 ms self-time is Python overhead of the Phase 3 Hungarian-matching
loop at lines 1485–1512: for each of ~275 539 OK cubes, it builds a small
`(n_loops, n_points)` cost matrix and calls `scipy.optimize.linear_sum_assignment`.

Callee breakdown shows 275 539 calls each to `sum`, `astype`, `linear_sum_assignment`,
`append` — the loop itself is the cost driver. Vectorizing Hungarian across
cubes is NOT mechanical (variable-size `n_loops × n_points` matrices require
padded packing + custom Hungarian; scipy is scalar-call-only). Any such rewrite
would be a full algorithmic refactor, not an orchestration cleanup.

Micro-wins inside Phase 3 (non-mechanical-threshold):
- `astype(np.float64)` 77.4 ms — if scipy accepts float32 we could drop it; but
  scipy's `linear_sum_assignment` C signature expects float64, so required.
- `sum(axis=-1)` 40.7 ms + `diff*diff` — already minimal numpy; non-mechanical.

### `.cpu()` redundancy check (lines 1206, 1207, 1215, 1218)
Suspected pattern: four unconditional `batch.X.cpu().numpy()` transfers at the
top of `s7_rank_assign` that are only consumed on the legacy CPU path (not the
default GPU path). Specifically:
- `ew_np` (line 1206) — used in legacy CPU path at L1296, L1421
- `ci_np` (line 1207) — **unused everywhere** (dead)
- `uturn_np` (line 1215) — used in legacy CPU path at L1302
- `fw_np` (line 1218) — **unused everywhere** (dead)

Attribution from s7-level pstats: **`.cpu()` direct from `s7_rank_assign`
totals only 11 calls / 29.15 ms.** Conditionalizing or removing these saves
<30 ms — far below the 200 ms threshold.

(Note: the 243.6 ms cumulative `.cpu()` shown under the "functions called by
s7_rank_assign" view attributes recursive/indirect calls, not direct. Direct
attribution is the 29 ms figure.)

### `.item()` redundancy check
Direct `.item()` from `s7_rank_assign` is 1 call / 0.03 ms (the
`total_loops = int(batch.loop_cube_off[-1].item())` at L1212). The 171 ms
cumulative `.item()` attribution includes deep descendants. No load-bearing
`.item()` cache exists to eliminate.

### Per-cube serial loops at lines 1233-1256 / 1247-1256
Two Python `for i in range(N)` loops over ~275 K cubes for status/offset
inspection. Self-cost is small (well under 50 ms based on N * loop-body-cost);
these are bookkeeping, not mechanical duplication.

## Candidates rejected (<200 ms or not mechanical)

| pattern | cost | reason rejected |
|---------|------|-----------------|
| `_build_adjacency_gpu` | 1913 ms self | real GPU compute; not mechanical, Triton territory |
| `s7_rank_assign` Phase 3 loop 275K iterations | 984 ms self | variable-size scipy Hungarian per cube; vectorization is a major refactor, not mechanical |
| top-of-function `.cpu().numpy()` x4 (L1206/1207/1215/1218) | ~30 ms attributable | below threshold; also includes 2 dead variables (ci_np, fw_np) worth ~1 line cleanup |
| `cost.astype(np.float64)` | 77.4 ms | required by scipy signature; irreducible |
| `diff*diff.sum(axis=-1)` | 40.7 ms | already minimal numpy |
| Status/offset inspection loops (L1233-1256) | <50 ms | bookkeeping, not duplication |

## Decision

**Recommendation: SKIP W7.**

No candidate ≥200 ms mechanical win exists in `s7_rank_assign` orchestration.
The two large self-time entries are:
1. `_build_adjacency_gpu` — genuine GPU compute (Triton territory, separate workstream)
2. `s7_rank_assign`'s Phase 3 loop — algorithmic Hungarian-per-cube, needs refactor not cleanup

Minor hygiene opportunity (not worth a dedicated task): `ci_np` and `fw_np` at
lines 1207/1218 are dead; `ew_np`/`uturn_np` could be conditionalized to the
legacy CPU path — total savings ≈15-25 ms. Could be folded into any future
routine s7 edit but not worth a standalone ticket.

**W7 budget should be reassigned.**
