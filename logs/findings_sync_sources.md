# Sync Sources — Findings (2026-04-17 spike)

**Spec:** `docs/superpowers/specs/2026-04-17-sync-spike-design.md`
**Plan:** `docs/superpowers/plans/2026-04-17-sync-spike-implementation.md`
**Branch:** `post-profile-sync-elim`
**Data basis:** `tmp/profile_deep/results/nsys_res256_full.sqlite`,
`tmp/profile_deep/results/layer1_res256_run1_trace.json` (with_stack=True)

## 0. TL;DR

- **1.0s cudaDeviceSynchronize root cause:** Bucket **C (Alloc-size)** @
  `corep_fast/stages/s1_voxelize.py:56` — `total_pairs = int(offsets[-1].item())`
  feeds `torch.arange(total_pairs, ...)` in `_expand_candidates`; the `.item()`
  blocks until cumsum completes. Fix effort: 2–4 h (standalone) or 1–2 wk if
  folded into a Triton fused kernel (ROI #4).
- **Top-20 blocking D2H distribution:** A=2, B=6, C=7, D=4, E=0 (plus one
  A/B dual-label → total 20). Aggregate ms: **B=198.57** (98%), D=4.32,
  C=0.07, A=0.06, E=0. The picture is NOT "many small .item() leaks"; it is
  **six bulk `.cpu().numpy()` dispatches at the GPU→CPU multiprocessing
  boundary** in `s6_collapse` / `s7_rank_assign` / `s4_face_point` /
  `s8_collapse`.
- **Next fix spec recommended scope:** **Option Y — "A-only + 1.0s
  DeviceSync (§1)"** — mechanical removal of 2 Bucket-A sites + deferred-alloc
  fix for the one 1.0s sync (bucket C via on-device alternative). Expected Δ
  wall-time: **–1.0 to –1.1 s @ res=256** (dominated by the §1 fix); effort
  ~3–5 d. Bucket B (98% of top-20 ms) is **deliberately out of scope** for
  the next incremental spec — it is an architectural rewrite (GPU fused
  kernels to replace MP workers) that belongs to a separate track aligned
  with Triton K1/K2 work (ROI #4). See §4 for the full menu.
- **Not covered here** (per spec §3.2): any corep_fast code change; topology
  correctness fixture (deferred to the actual fix spec); Triton K1 design;
  s7 vectorize; cummax fuse.

## 1. The 1.0s cudaDeviceSynchronize

### 1.1 Evidence chain

- **nsys**: `cudaDeviceSynchronize` call count 20, total 1049ms, longest single call
  1048.37ms (from `tmp/sync_spike/nsys_top_devicesync.json`:
  start_ns=3231746876, end_ns=4280118614).
- **torch.profiler**: projected target window (fraction 0.0094–0.0560 of the nsys recording span) contained 7 python_function frames; shortest-dur frames:
  1. `<built-in method searchsorted of type object>` (7.4 ms)
  2. `corep_fast/stages/s1_voxelize.py(106): _expand_candidates` (47.9 ms)
  3. `corep_fast/stages/s1_voxelize.py(18): s1_voxelize` (57.7 ms)
- **grep corep_fast/**: `corep_fast/stages/s1_voxelize.py:56` matches pattern
  `item` — `total_pairs = int(offsets[-1].item())` reads the last element of a
  GPU tensor to Python int, forcing a D2H scalar transfer and blocking the host
  until all preceding CUDA work (cumsum) completes. This is the only sync-pattern
  hit in s1_voxelize.py.

Note on `searchsorted`: T3 identified `searchsorted` as the innermost frame
(7.4 ms), but it is called *inside* `_expand_candidates` (line 122), which
itself is called *after* `total_pairs` is already materialized on the host at
line 56. The DeviceSync event precedes `_expand_candidates` in wall time; the
`.item()` at line 56 is the proximate trigger. `searchsorted` at line 122 is
a separate, much smaller host-device round-trip that does not account for the
1.0 s spike.

### 1.2 Source snippet (±10 lines)

```python
# corep_fast/stages/s1_voxelize.py lines 46-67
    # Integer AABB: candidate cubes for each triangle
    imin = tri_min.floor().to(torch.int32).clamp(min=0, max=R - 1)  # (F, 3)
    imax = tri_max.floor().to(torch.int32).clamp(min=0, max=R - 1)  # (F, 3)
    # Number of candidate cubes per triangle
    spans = (imax - imin + 1).to(torch.int64)  # (F, 3)
    counts_per_tri = spans[:, 0] * spans[:, 1] * spans[:, 2]  # (F,)

    # Step 2: expand -- enumerate all (triangle, candidate_cube) pairs
    offsets = torch.zeros(counts_per_tri.shape[0] + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts_per_tri, dim=0)
    total_pairs = int(offsets[-1].item())   # <-- SYNC TRIGGER (line 56)

    # For each pair, identify triangle_id and cube (ix, iy, iz)
    pair_tri_ids, pair_cube_coords = _expand_candidates(
        imin, imax, spans, offsets, total_pairs, device)

    # Step 3: SAT test -- 13 axes
    hits = _sat_test_batch(tri_grid, pair_tri_ids, pair_cube_coords, R, device)

    # Step 4: compact hits into CSR
    hit_tri_ids = pair_tri_ids[hits]
    hit_cube_coords = pair_cube_coords[hits]
```

### 1.3 Root-cause bucket

Assigned bucket: **C** (Alloc-size) — The scalar `total_pairs` retrieved via
`.item()` is immediately passed as the `total_pairs: int` argument to
`_expand_candidates`, where it drives `torch.arange(total_pairs, ...)` (line
121). PyTorch's `arange` requires a concrete Python integer for the stop
argument, so the host *must* know this value before the allocation can proceed.
This is a classic alloc-size pattern: a reduction result (cumsum prefix-sum
tail) is transferred to the host solely to size the next allocation, blocking
the GPU pipeline for 1.0 s while waiting for the cumsum kernel to finish.

A syntactic cleanup (dropping the redundant `int(...)` wrapper around `.item()`)
does not eliminate the sync — `.item()` itself is the blocking D2H. Any real fix
must keep `total_pairs` on-device. The canonical path is a fused Triton kernel
that performs the prefix-sum and the candidate expansion entirely on GPU,
avoiding the host round-trip (see ROI #4 in
`my-docs/20260417-corep-deep-profiling-results.md`).

### 1.4 Effort estimate

**Medium (2–4 hours)** — A minimal fix (keep cumsum on GPU, pass `offsets[-1]`
directly as a shape argument via `.item()` deferred into a separate stream, or
pre-compute on CPU using a light triangles-AABB-volume upper bound) can be
coded in under an hour, but the correct fix — a fused Triton kernel that avoids
materializing the prefix-sum prefix on the host — is the Triton K1 candidate
already identified in the deep-profile ROI table and is estimated at 2–4 hours
of kernel authoring + tuning.

## 2. Top-20 blocking D2H — Classified

Data basis: `tmp/sync_spike/top20_d2h.json`. Each row's stage / frame_name
derived from `extract_top20_d2h.py`; source file:line from
`classify_top20.py` + manual review. Pattern column records the concrete
call construct (`.item()`, `.cpu()`, explicit sync, etc.). Total DtoH events
observed in this trace: **2011** (= 1953 Pinned + 58 Pageable; see §5 caveat
for the 2011-vs-4022 reconciliation vs Deep Profiling's nsys-based count).

Methodology note: for rows whose frame is the generic built-in
`<built-in method cpu of Tensor>` (ranks 1, 2, 3, 4, 6) the `.cpu()`
attribution is resolved by grepping the owning stage's source file and
picking the call sites that feed the immediately-following CPU / multiprocessing
worker block. Ranks 1/2/3/4/6 each correspond to *multiple* `.cpu().numpy()`
calls in the same stage; we report the most representative call (the one
fronting the MP dispatch) and treat the row as aggregate.

For rank 5 (`sh_clip.py:108 _emit`): the function itself has no `.cpu()` or
`.item()`; the D2H is the boolean-mask indexing (`out[c_idx[mask], ...]`,
`vertex[mask]`) on lines 139–140, which triggers an implicit D2H-free
`nonzero` kernel but produces a very small host-side round-trip via the
indexing machinery. Per-call dur (2.3 us) confirms this is an internal
micro-sync, not a user `.item()`.

| Rank | Stage | File:line | Pattern | Count | Total ms | Bucket | Per-site effort | Justification |
|---:|---|---|---|---:|---:|:-:|---|---|
| 1 | s6_collapse | `corep_fast/stages/s6_collapse.py:846-850` | `.cpu().numpy()` | 11 | 60.57 | B | 1-3 days | Bulk transfer of edge_weights / face_weights / fast & slow masks ahead of CPU MP worker dispatch — values drive per-cube Python traversal (graph build + loop tracing). Eliminating requires rewriting the CPU worker on GPU (non-trivial algorithmic rework). |
| 2 | s7_rank_assign | `corep_fast/stages/s7_rank_assign.py:1206-1218` | `.cpu().numpy()` | 16 | 51.99 | B | 1-3 days | Bulk transfer of edge_weights / cube_indices / uturn_assignment / face_weights / CSR arrays for Phase 1 CPU MP rank re-tracing + Phase 3 Hungarian matching. Used to drive scipy.linear_sum_assignment per cube. |
| 3 | s4_face_point | `corep_fast/stages/s4_face_point.py:148-152` | `.cpu().numpy()` | 10 | 45.61 | B | 1-3 days | Pre-MP bulk transfer of cube_indices / tri_offsets / tri_values / mesh verts+faces for `_fw_worker_indexed` (per-cube face-weight computation in numpy). Algorithmic rework required to stay on GPU. |
| 4 | `<unknown>` | s6/s7 sibling call sites (e.g. `s6_collapse.py:880,907` fast-path finalization) | `.cpu().numpy()` | 7 | 38.20 | B | 1-3 days | Stage tag lost in `_torch_profiler_top` tree; per-call dur (5.5 ms) matches r1/r3 profile exactly — same class of bulk D2H before CPU worker dispatch. Treat as additional s6/s7 Phase-2 transfers. |
| 5 | `<unknown>` (geom/sh_clip) | `corep_fast/geom/sh_clip.py:139-140` | boolean-mask indexing | 1080 | 2.50 | D | weeks | Leaf frame of a 10-iteration `sh_clip` loop; implicit D2H comes from `out[c_idx[mask], ...]` scatter which drives a `nonzero` host-side sizing internally. Per-call 2.3 us → PyTorch runtime sync, not a user `.item()`. The `_emit` docstring (sh_clip.py:117-119) claims the advanced-index scatter avoids `mask.any()` sync, but the scatter still materializes the internal `nonzero` shape — this confirms Bucket D (library-internal). Fix requires custom Triton kernel for masked scatter. |
| 6 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py:283-292` | `.cpu().numpy()` | 11 | 2.19 | B | 1-3 days | Bulk CSR+status transfer at the start of `_process_shared_edges_*` which then does Python-level edge enumeration and welding; per-call dur (199 us) is an order of magnitude smaller than r1/r2/r3 because s8 tensors are smaller. |
| 7 | s7_rank_assign | `corep_fast/stages/s7_rank_assign.py:634` | built-in (scatter_add_ / internal) | 778 | 1.80 | D | weeks | 778 micro D2Hs inside `_build_adjacency_gpu` at 2.3 us each; no user-visible `.item()` here. Attributed to PyTorch library-internal sync inside scatter_add_ / gather on large (N, 12, 3) tensors. Cannot be eliminated without replacing with custom kernel. |
| 8 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py:1629` | host-side dispatch (`_process_shared_edges_from_tensors`) | 16 | 0.04 | A | seconds | Tiny per-call 2.4 us × 16; these are log/metric aggregations within the direct-tensor path entry — deferrable if we chose to do so, but total impact negligible (<40 us). |
| 9 | `<unknown>` | `<no frame>` (unresolved) | unknown built-in | 9 | 0.02 | A/B | seconds - 1 day | Frame lost; per-call 2.5 us suggests small scalar/metric D2H. Dual-labeled because without source we cannot rule out control-flow; pessimistic effort used. |
| 10 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py:920` | host-side dispatch (`process_geometry_vectorized`) | 9 | 0.02 | A | seconds | Micro host-side round-trips at entry of the vectorized geometry builder; trivially deferrable. Negligible total. |
| 11 | `<unknown>` | `<built-in method repeat_interleave>` | torch internal | 8 | 0.02 | D | weeks | `repeat_interleave` issues an internal D2H when repeats is a tensor (alloc-size pattern inside aten). Not user-facing; swap requires PyTorch-level change. |
| 12 | s6_collapse | `corep_fast/stages/s6_collapse.py:233` + `:244` | `.item()` (max reduction for arange) | 7 | 0.02 | C | 3-7 days | `max_points = int(total_points_per_cube.max().item())` + `max_w = int(edge_weights_fast.max().item())` directly drive subsequent `torch.arange` / shape allocations on GPU. Alloc-size pattern. Fix: fused Triton kernel or on-device bound. |
| 13 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py:182` (`_pad_ragged_loops_gpu`) | alloc-size path | 6 | 0.01 | C | 3-7 days | `max_loop_len` (caller at s8_collapse.py:662 via `int(lengths.max().item())`) drives padding tensor alloc `(L, max_loop_len)`. Alloc-size sync. |
| 14 | s2_components | `corep_fast/stages/s2_components.py:214` + `:239` | `.item()` (n_valid, nbr_face) | 5 | 0.01 | B | 1-3 days | `n_valid = int(cube_mask.sum().item())` gates a Python-side `range(n_valid)` loop over slot_j; `nbrs[slot_j, e].item()` reads per-step scalar for a `!= -1` check. Control-flow pattern (alternative: vectorize via mask). |
| 15 | s4_face_point | `corep_fast/stages/s4_face_point.py:1119` (`_compute_face_weights_gpu`) | `.item()` (reduction → alloc) | 4 | 0.01 | C | 3-7 days | `_compute_face_weights_gpu` end-of-body uses `int(valid.any().item())` (line 1149) as a Python bool guard, but the high-freq leaf sites are more consistent with alloc-size `.max().item()`. Pessimistic bucket C. |
| 16 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py:777` (`build_edge_neighbor_table`) | alloc-size via caller | 4 | 0.01 | C | 3-7 days | Called from s8_collapse.py:891 (`R = int(cube_indices.max().item()) + 2`) — alloc-size for downstream tensors sized by resolution bound. |
| 17 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py` (internal `nonzero`) | `<built-in method nonzero>` | 4 | 0.01 | D | weeks | `nonzero` returns a dynamic-shape tensor, forcing a D2H to materialize the row count. Library-internal; replace with fixed-size masked gather if feasible. |
| 18 | decode_from_cubebatch (s8) | `corep_fast/stages/s8_collapse.py:1045` / `:891` / `:662` | `.item()` | 3 | 0.01 | C | 3-7 days | Various `int(x.max().item())` call sites driving allocations (`max_loops`, `R`, `max_loop_len`). Alloc-size pattern. |
| 19 | s2_components | `corep_fast/stages/s2_components.py:47` | `.item()` (max for padded alloc) | 3 | 0.01 | C | 3-7 days | `max_faces = int(counts.max().item())` sizes `torch.full((N, max_faces), -1, ...)` tensor on line 55. Classic alloc-size. |
| 20 | s4_face_point | `corep_fast/stages/s4_face_point.py:440` | `.item()` (cumsum tail → alloc) | 2 | 0.01 | C | 3-7 days | `total_points = int(point_offsets[-1].item())` gates early-return and indirectly caps the centroid tensor allocation. Alloc-size. |

## 3. Per-bucket breakdown

### Bucket A (Deferrable)
- Sites: 2 (ranks 8, 10; +0.5 dual-label with r9 ⇒ effective 2.5)
- Aggregate ms in top-20: 0.0601 ms (r8+r10); +0.0224 ms r9 (dual) ⇒ ~0.08 ms total
- Effort rollup: <1 hour (remove/defer the log-dispatch D2Hs)
- Expected Δ wall-time if fully removed: 0.06-0.08 ms upper bound; with `with_stack=True` overhead correction: ~0.02-0.05 ms (rounding error).
- Representative site: `corep_fast/stages/s8_collapse.py:1629` (`_process_shared_edges_from_tensors`)
- Notes: Micro-impact; listed mainly because the bucket is non-empty. Not worth standalone fix; auto-resolves when upstream `.cpu()` call sites are removed.

### Bucket B (Control-flow)
- Sites: 6 (ranks 1, 2, 3, 4, 6, 14; +0.5 dual-label with r9 ⇒ effective 6.5)
- Aggregate ms in top-20: 198.5730 ms (r1+r2+r3+r4+r6+r14); +0.0224 ms r9 (dual) ⇒ ~198.60 ms total
- Effort rollup: ~6-18 days (each of 5 primary CPU-MP sites is a worker → GPU algorithmic rework; r14 is a simpler in-stage vectorization).
- Expected Δ: depends on whether MP workers can be replaced by GPU kernels. Upper bound 198.6 ms; with `with_stack=True` overhead deflator (~2-5x): realistic ~40-100 ms of wall-time. The dominant CPU MP workers (s6 fastpath/slowpath trace, s7 Phase-1 rank retrace, s4 face-weight compute) have been on the optimization roadmap for a while; converting them is blocked by algorithmic complexity (union-find / loop tracing with irregular shapes).
- Representative site: `corep_fast/stages/s6_collapse.py:846-850` (largest single bulk-D2H block; 5 tensors × avg ~10 ms each).
- Notes: This bucket *dominates* the top-20 total_ms. All primary sites share the same shape: GPU computes per-cube features, then `.cpu().numpy()` the lot, then CPU MP workers chew on numpy arrays. Eliminating requires rewriting the MP workers on GPU — Triton candidate territory.

### Bucket C (Alloc-size)
- Sites: 7 (ranks 12, 13, 15, 16, 18, 19, 20)
- Aggregate ms in top-20: 0.0736 ms (sum of 7 rows at the ms tail of the distribution)
- Effort rollup: ~3-7 days × 7 = 3-7 weeks if fixed individually; ~1-2 weeks if a single "deferred-alloc / GPU-sized-arange" helper shim is built.
- Expected Δ wall-time: ~0.07 ms upper bound (top-20 only); with `with_stack=True` deflator (2–5×): realistic ~0.02–0.04 ms (noise-level). **However**, the §1 1.0 s `s1_voxelize.py:56` sync is *itself* bucket C and alone dominates — see §1 for the large win.
- Representative site: `corep_fast/stages/s2_components.py:47` (`max_faces = int(counts.max().item())` drives `torch.full((N, max_faces), ...)` allocation; canonical alloc-size pattern.
- Notes: All 7 sites are `int(x.max().item())` or `int(x[-1].item())` calls where x is a reduction/cumsum tail used as a shape. All fixable by the same pattern (Triton prefix-sum + on-device arange or deferred shape via dynamic kernels). **Even though in-top-20 impact is only 0.08 ms, the class is load-bearing because s1_voxelize.py:56 is in the same class (1.0 s).**

### Bucket D (Library-internal)
- Sites: 4 (ranks 5, 7, 11, 17)
- Aggregate ms in top-20: 4.3197 ms (r5 2.50 + r7 1.80 + r11 0.02 + r17 0.01; minor rounding on r11/r17).
- Effort rollup: weeks (API swap: custom Triton kernel or PyTorch upstream patch).
- Expected Δ wall-time: ~4.32 ms upper bound (top-20 only); with `with_stack=True` deflator (2–5×): realistic ~1–2 ms. Not actionable in a sync-elim spike; noted for completeness.
- Representative site: `corep_fast/geom/sh_clip.py:139-140` (boolean-mask scatter → internal `nonzero` sync).
- Notes: These D2Hs are inside PyTorch built-ins (`nonzero`, `repeat_interleave`, boolean-mask indexing, scatter variants) and do not expose user code to fix. They will remain until we write custom Triton/CUDA kernels for those ops. Low priority given impact.

### Bucket E (Correctness-sync)
- Sites: 0
- Aggregate ms in top-20: 0.00 ms
- Effort rollup: hours (audit) — but no sites to audit from top-20.
- Representative site: (none in top-20)
- Notes: No explicit `torch.cuda.synchronize(...)` appears in the top-20 D2H list. A full-repo grep (from Task 4 `tmp/sync_spike/grep_sync_sources.log`) may still surface safety syncs elsewhere, but within the profiled top-20 window there are zero E-bucket sites. This is a good sign for the hot path.

## 4. Recommended next fix spec scope

Three candidate scopes for the next sync-elimination fix spec, ordered by
risk / effort:

### Option X — "A-only" (low risk, mechanical)

- **What:** Remove or defer the **2 Bucket-A sites** from §2 (representative:
  `corep_fast/stages/s8_collapse.py:1629`; second site: see §2 rank 8 /
  secondary A row).
- **Methodology:** For each site, rewrite so the scalar `.item()` is either
  eliminated entirely, or moved to the final end-of-pipeline collection
  phase. Add a per-site fixture asserting bit-exact output topology (vs
  custom and baseline) before/after the change.
- **Risk:** Very low. Pure refactor, no algorithmic change.
- **Effort:** ~2 h total (seconds per site × 2 sites + test boilerplate).
- **Expected Δ wall-time @ res=256:** –0.06 ms upper bound (top-20 only).
  After `with_stack=True` overhead deflator: effectively noise-level — this
  is a cleanliness fix, not a perf fix.
- **Verdict:** **Do NOT standalone-spec this.** Not worth the ceremony of a
  separate spec. Either bundle into Option Y or drop.

### Option Y — "A + 1.0s DeviceSync (§1)" ← RECOMMENDED

- **What:** Option X's 2 A-sites **plus** the §1 site
  `corep_fast/stages/s1_voxelize.py:56` (1.0s cudaDeviceSynchronize, bucket
  C alloc-size).
- **Methodology:**
  1. Wrap the 2 Bucket-A sites per Option X.
  2. For the §1 site, implement a deferred-allocation alternative: either
     (i) keep `offsets[-1]` on-device and pass a GPU-size sentinel into a
     Triton kernel that does arange internally, OR (ii) pre-allocate a
     max-sized buffer once and slice — trading memory for latency; the
     conservative upper bound is `F * R^3` where `F` is the triangle count
     and `R` is the voxel resolution (since `spans` is clamped to `[1, R]`
     per axis via the `imin`/`imax` clamp on lines 47-48, so
     `counts_per_tri = spans_x * spans_y * spans_z <= R^3`; both `F` and
     `R` are known at entry without any sync — see `s1_voxelize.py:40-56`).
  3. Topology-correctness gate: golden-fixture pipeline @ res=128 and
     res=256 must produce cube counts, adjacency, and decode output
     bit-identical to pre-change baseline (custom and fast_M2 dual-reference).
- **Risk:** Low-medium. Option (i) needs a small Triton scan or reuse of
  existing cumsum infra; option (ii) is trivial but wastes memory during
  low-load calls. Neither touches the MP worker path — zero interaction
  with §2 Bucket-B.
- **Effort:** ~3–5 d total (0.5 d for Option X portion; 2–4 d for §1 fix +
  fixture + validation).
- **Expected Δ wall-time @ res=256:** **–1.0 to –1.1 s** (dominated by §1's
  single 1048 ms sync; A-site savings are noise).
- **Verdict:** **Recommended as next fix spec.** Clean risk/reward, testable
  topology gate, unblocks the single largest deterministic bubble.

### Option Z — "A + C-only + selected B" (medium-high risk)

- **What:** Option Y's scope **plus** the 7 Bucket-C sites from §2 (all
  alloc-size `.item()` patterns in s2/s4/s6/s7) **plus** 1–2 highest-dur
  Bucket-B entries (but NOT all six — that would be the MP rewrite track).
- **Methodology:** For Bucket-C sites, same deferred-alloc pattern as §1's
  fix — some are reusable fixes via a small helper. For the 1–2 selected
  Bucket-B entries, requires per-site fast↔slow boundary rewrite (GPU kernel
  to replace the MP worker); each is a 3–5 d refactor with its own topology
  fixture.
- **Risk:** Medium-high. Bucket-B rework can regress correctness subtly
  (MP worker outputs drive s6 topology decisions). Scope creep likely.
- **Effort:** ~2–3 wk total.
- **Expected Δ wall-time @ res=256:** Option Y's –1.0 to –1.1 s + ~0 ms
  from C sites (already tiny) + 40–80 ms from the 1–2 selected Bucket-B
  rewrites (after 2–5× `with_stack=True` deflator).
- **Verdict:** Over-scope for a single incremental spec. Only consider if
  the Triton K1/K2 track is on hold and incremental MP-replacement is
  judged worthwhile.

### Recommendation

**Option Y** is the recommended next spec because:

1. **Single-spec deliverable targets a single decisive bottleneck** — the
   1.0s sync at `s1_voxelize.py:56` is 83% of all currently-measurable
   blocking sync time (1048 ms of ~1250 ms total sync budget per §1+§2).
2. **Clean risk envelope** — the fix touches one stage, one function,
   bounded test surface. Option X's 2 A-sites tag along at negligible
   cost.
3. **Avoids scope-creep into architectural rework** — the 98%-of-top-20
   Bucket-B load belongs with the Triton K1/K2 GPU-replacement track
   (ROI #4 in
   `my-docs/20260417-corep-deep-profiling-results.md`), not with a
   sync-elimination spec.

**Follow-up track (separate spec later):** The Bucket-B rewrite is a
distinct initiative — replace the 4–6 CPU multiprocessing workers in
s6/s7/s4/s8 with GPU kernels (Triton candidates or fused PyTorch). That
spec should be scoped on its own, ideally after Option Y lands and the
post-fix re-profile confirms bucket-B is truly the dominant residual.

## 5. Caveats

- **Trace captured with `with_stack=True`** — torch.profiler's full-stack
  attribution inflates per-event wall-time by ~2–5× for memcpy/sync
  events (measured as part of ROI #6: the `with_stack=False` flip in
  `tmp/profile_deep/driver.py` committed at 99c06b7 showed res=128 e2e
  ratio 0.743×). Aggregate durations in §2/§3 (`total_dur_ms`) are
  therefore **upper bounds**; actual wall-time savings from removing
  sync sites will be proportionally smaller. Confirmation path: after
  Option Y lands, re-profile with `with_stack=False` (now default) and
  measure the residual.
- **NVTX markers do NOT appear in torch.profiler trace** — stage attribution
  in §2 relies on matching `python_function` events by function name, not
  NVTX ranges (Deep Profiling 2026-04-16/17 methodology, per
  `my-docs/20260417-corep-deep-profiling-results.md`). A few ambiguous
  sites (row 4, row 9 in §2) list `<unknown>` stage because no
  STAGE_NAMES ancestor was found in the 10k-event backward-scan window.
- **nsys ↔ torch.profiler time projection is approximate** — §1's
  cross-reference uses a linear projection between the two timebases
  because nsys is absolute ns and torch.profiler is profiler-relative μs.
  Cross-verified via grep + source reading (`s1_voxelize.py:56` has the
  only `.item()` in that stage). If future spikes need to locate a sync
  in a file with multiple `.item()` calls, the projection may need
  tightening.
- **Single-run data** — the Chrome trace used here is
  `layer1_res256_run1_trace.json`, one of three profiler runs in the
  Deep Profiling 3-run dataset. Counts may vary ±5% across runs; top-20
  ranking is stable by construction (the top Bucket-B entries are each
  ~10× larger than the rank-20 site).
- **D2H event counts: 2011 in this trace vs ~4022 in Deep Profiling** —
  the 4022 figure in `my-docs/20260417-corep-deep-profiling-results.md`
  came from a different source (nsys CUDA API trace, which counts both
  blocking and non-blocking memcpys). The 2011 here is torch.profiler's
  own gpu_memcpy DtoH event count — both Pinned and Pageable variants
  combined (T5 spec review verified filter completeness). Both numbers
  are correct for their respective measurement frames.
- **Bucket A/B ambiguity** — rank 9 (`<no frame>` at 0.02 ms) was
  classified dual-label A/B and counted pessimistically as B for effort.
  If the next spec implementer can resolve the frame, it reclassifies to
  A; negligible impact.
- **with_stack=True deflator not applied uniformly** — §3 Bucket A and B
  call out the overhead deflator explicitly; Bucket C and D omit it.
  C's numbers are already noise-level (0.07 ms), D is small (4.32 ms)
  and changes little; the Option Y vs Option Z decision is robust to
  this asymmetry.
