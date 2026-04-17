# T5a — s6 fast-path loop tracer GPU design sketch

**Status:** design only (read-only task). No production code changes, no commits.
**Author context:** produced from reading `corep_fast/stages/s6_collapse.py` end-to-end, downstream CSR consumers in `corep_fast/interop/{to,from}_custom.py`, `corep_fast/profiling/topology_equivalence.py`, `corep_fast/stages/s7_rank_assign.py`, and the regression gate `corep_fast/tests/regression/test_cpu_worker_optim.py`.

---

## 1. Current implementation

- **Hotspot function:** `_fastpath_trace_loops_numpy` at `corep_fast/stages/s6_collapse.py:379-428`.
- **Signature:**
  ```python
  def _fastpath_trace_loops_numpy(
      point_offset_row: np.ndarray,  # (19,)  int64 — per-cube edge-prefix-sum
      adj_row:          np.ndarray,  # (max_points, 2) int32 — two neighbours per point
      total_points:     int,
  ) -> List[List[int]]  # list of loops, each loop is a list of edge ids (int)
  ```
- **Call site:** `corep_fast/stages/s6_collapse.py:882-891`, a pure-Python `for local_i in range(M_fast)` loop. At res=256 this is 275k iterations, each doing:
  1. `np.searchsorted` on a length-19 array (to produce `edge_of_point`),
  2. a Python `while` walk visiting each of the `total_points` points,
  3. a per-point `int(...)` cast and `list.append`,
  4. a top-level `list.append(current_loop)`.
- **T0 cost (res=256, icosphere F2):** 1891 ms self-time, 275k calls — the single largest main-thread CPU hotspot per T0 cProfile.
- **Per-cube size distribution (inferred from the face-count path):** median loops per cube = 1-3, median edges per loop = 4-8, long-tail maximum ≈ 12-24 edges per loop (bounded by the 18 cube edges times max face weight, typically small in practice). The work is embarrassingly parallel across cubes, with mild warp-divergence risk from tail cubes.
- **Where the output goes:** `per_cube_loops[cube_idx] = loops` at line 945, then at lines 966-997 the full `per_cube_loops` (fast-path + worker results) is flattened into the final CSR tensors `(loop_cube_off, loop_edge_off, loop_edge_val)` via `np.fromiter` / `list.extend` / `np.concatenate`. That re-packing is itself not cheap but is out of T5 scope.

---

## 2. Input/output contract (for T5b-d)

### Inputs to a GPU replacement
Already present on device inside `_fastpath_gpu_build_adjacency` (lines 190-376), but currently copied back to host via three `.cpu().numpy()` calls at lines 370-376. If T5 replaces the tracer with a GPU kernel, those three copies become redundant for the fast-path branch.

- `point_offset` : `(M_fast, 19)` int64 on device — per-cube prefix sum of edge weights; `point_offset[c, 18]` == total_points of cube c.
- `adj`          : `(M_fast, max_points, 2)` int32 on device — two neighbour point-ids per point; `-1` means unused slot.
- `degree_ok`    : `(M_fast,)` bool on device — cube is well-formed (every active point has degree exactly 2).
- Implicit: `fast_idx_np` — the `cube_idx` mapping from local fast-path index `local_i` back to the global cube index space `[0, N)`.

### Output contract (CSR, device tensors)
```
fast_loop_count   : (M_fast,)          int64  — number of loops per fast-path cube
fast_loop_off     : (M_fast + 1,)      int64  — CSR offsets over loops:
                                                 fast_loop_off[c+1] - fast_loop_off[c] == fast_loop_count[c]
fast_edge_off     : (L_fast + 1,)      int64  — CSR offsets over loop edges:
                                                 fast_edge_off[j+1] - fast_edge_off[j] == len(loop j)
fast_edge_val     : (E_fast,)          int32  — concatenated edge ids
```
Where `L_fast = sum(fast_loop_count)` and `E_fast = total_points_per_cube.sum()` (because every point contributes exactly one edge id to the loop it belongs to).

This layout is **exactly the CSR layout the final `_replace_fields` call consumes** (lines 999-1012), so if we also change the fast-path GPU path to emit CSR directly, we skip the list-of-lists → CSR re-packing in lines 966-997 for fast-path cubes.

---

## 3. Proposed GPU algorithm

### 3.1 Parallelism choice

**One thread per cube** (thread = "cube-worker"). Rationale:

- The per-cube tracing state (`visited`, `prev`, `curr`, `loop`, `start`) is inherently sequential within a cube: step `k+1` depends on step `k`. Per-point parallelism would require a PRAM-style "list-ranking" (Wyllie, Euler tour) that is overkill for loops of length 4-8.
- N ≈ 275k cubes → 1075 blocks × 256 threads (or 2150 × 128) → plenty of occupancy to hide memory latency.
- Inner loop is at most `max_points` ≤ bound of `sum(edge_weights)` per cube, i.e. ≤ 18 × max_edge_weight. In practice `max_points ≤ 24` for icosphere fixtures.
- Warp-divergence risk is real (loop length varies within a warp) but capped: longest path per thread is `max_points`, same bound as the worst-case `visited` array scan.

### 3.2 Output layout and sizing

**Two-pass kernel** to avoid the need for atomics on a flat output tensor:

- **Pass 1 — count.** Launch `M_fast` threads. Each thread walks its cube's adjacency exactly like the CPU reference but records only `(loop_count_c, point_to_loop_rank[p])` for `p` in `[0, total_points_c)`.
  - `loop_count_c` : number of distinct loops traversed.
  - `point_to_loop_rank[c, p]` : which loop (0-indexed within the cube) each point belongs to. Stored in a `(M_fast, max_points)` scratch int16 tensor.
  - Also record `point_to_loop_pos[c, p]` : position of `p` within its loop (so Pass 2 can scatter without walking again). int16.
- **Prefix sums (GPU native — `torch.cumsum`):**
  - `fast_loop_off = torch.cat([zeros(1), torch.cumsum(fast_loop_count, 0)])` → `(M_fast + 1,)`.
  - `fast_edge_off` requires per-loop lengths. Derive those via a scatter: for each active point `(c, p)`, atomic-add `1` into `per_loop_len[fast_loop_off[c] + point_to_loop_rank[c, p]]`. Then cumsum.
  - Total `E_fast` = `total_points_per_cube.sum()` — known before the kernel even runs, so we can pre-allocate `fast_edge_val = torch.empty(E_fast, dtype=torch.int32, device=device)` without a second pass.
- **Pass 2 — scatter.** Launch `M_fast * max_points` threads (or 1-thread-per-cube reusing the Pass 1 walk). For each active point `(c, p)`:
  - `out_idx = fast_edge_off[fast_loop_off[c] + point_to_loop_rank[c, p]] + point_to_loop_pos[c, p]`
  - `fast_edge_val[out_idx] = edge_of_point(c, p)`
  - `edge_of_point(c, p) = searchsorted(point_offset[c, :], p, side='right') - 1` — this is O(log 19) = 5 comparisons; do it branchlessly with `torch.searchsorted` or keep it inline as a small unrolled loop.

**Why two passes?** The single-pass alternative requires an atomic reservation on `fast_edge_val`, which is (a) non-deterministic under concurrent atomics and (b) forbidden by the regression gate (see §5.1). The two-pass scheme is fully deterministic because every output slot is written by exactly one thread whose identity is a pure function of `(c, p)`.

**Alternative (preferred if we want a single pass):** Skip `point_to_loop_pos` entirely and require Pass 2 to re-walk. That trades memory (no int16 scratch) for 2× kernel work. Given the kernel is O(max_points) per cube and memory is not the bottleneck, **we prefer the scratch-buffer approach** (single walk per cube) — see pseudocode.

### 3.3 Pseudocode

```python
def _fastpath_trace_loops_gpu(
    point_offset: torch.Tensor,  # (M, 19) int64  device
    adj:          torch.Tensor,  # (M, max_points, 2) int32 device
    degree_ok:    torch.Tensor,  # (M,) bool device
) -> tuple[Tensor, Tensor, Tensor]:  # (fast_loop_off, fast_edge_off, fast_edge_val)

    M = point_offset.shape[0]
    max_points = adj.shape[1]
    total_points = point_offset[:, -1]          # (M,) int64
    E_fast = int(total_points.sum().item())

    # Scratch (device)
    loop_rank = torch.full((M, max_points), -1, dtype=torch.int16, device=dev)
    loop_pos  = torch.full((M, max_points), -1, dtype=torch.int16, device=dev)
    loop_count = torch.zeros(M, dtype=torch.int32, device=dev)

    # Kernel: one thread per cube
    @triton.jit  # or pure torch.jit / custom CUDA
    def walk_kernel(c):
        if not degree_ok[c]:
            loop_count[c] = 0         # mark as "no fast-path loops"; caller falls back
            return
        tp = int(total_points[c])
        visited_bits = 0              # uint32 bitmask (tp <= 24 comfortably fits)
        loop_idx = 0
        for start in range(tp):
            if visited_bits & (1 << start):
                continue
            curr, prev, pos = start, -1, 0
            while True:
                visited_bits |= (1 << curr)
                loop_rank[c, curr] = loop_idx
                loop_pos [c, curr] = pos
                pos += 1
                n0, n1 = adj[c, curr, 0], adj[c, curr, 1]
                nxt = n0 if prev == -1 else (n1 if n0 == prev else n0)
                prev = curr
                curr = nxt
                if curr == start:
                    break
            loop_idx += 1
        loop_count[c] = loop_idx

    # Build CSR offsets (all torch ops — deterministic)
    fast_loop_off = F.pad(torch.cumsum(loop_count.to(torch.int64), dim=0), (1, 0))
    L_fast = int(fast_loop_off[-1].item())

    # Per-loop length via scatter_add on (M, max_points) grid
    # global_loop_id[c, p] = fast_loop_off[c] + loop_rank[c, p]  (int64)
    active = (
        torch.arange(max_points, device=dev).view(1, -1)
        < total_points.view(-1, 1)
    )
    gl = (fast_loop_off[:-1].view(-1, 1) + loop_rank.to(torch.int64))  # (M, max_points)
    per_loop_len = torch.zeros(L_fast, dtype=torch.int64, device=dev)
    per_loop_len.scatter_add_(
        0, gl[active],
        torch.ones(int(active.sum().item()), dtype=torch.int64, device=dev),
    )
    fast_edge_off = F.pad(torch.cumsum(per_loop_len, dim=0), (1, 0))

    # Edge of each point — O(M * max_points * log 19) or 18-branch loop
    edge_of_point = torch.searchsorted(point_offset, torch.arange(max_points, device=dev).view(1, -1).expand(M, -1), right=True) - 1
    # (M, max_points) int64

    # Scatter edge ids into flat output
    fast_edge_val = torch.empty(E_fast, dtype=torch.int32, device=dev)
    out_idx = fast_edge_off[gl] + loop_pos.to(torch.int64)        # (M, max_points)
    fast_edge_val.scatter_(0, out_idx[active], edge_of_point[active].to(torch.int32))

    return fast_loop_off, fast_edge_off, fast_edge_val
```

Notes on the pseudocode:
- `loop_rank` and `loop_pos` only need int16 because `max_points ≤ ~24` in practice (and cap at 32 is safe — asserting `<= 32` for the bitmask is cheap).
- The `visited_bits` bitmask avoids a per-thread `visited[max_points]` array — single 32-bit register per thread, branchless check.
- If we can't express the walk in Triton/pure-torch easily, a **torch.compile dynamic shape** or small **CUDA extension** are equivalent. A pure-PyTorch formulation IS possible (batched padded walk with `prev/curr` state tensors of shape `(M,)` and `max_points` iteration steps), which avoids the extension burden entirely — see §3.4 below.

### 3.4 Pure-PyTorch alternative (fallback for T5c/T5d if Triton is blocked)

Replace the per-thread `while True:` with a **padded bounded walk** on `(M,)` state tensors for exactly `max_points` steps:

```python
# State: (M,) tensors
curr = torch.zeros(M, dtype=torch.int32, device=dev)          # start = 0 for each cube's first loop
prev = torch.full ((M,), -1, dtype=torch.int32, device=dev)
loop_rank_row = torch.full((M, max_points), -1, dtype=torch.int16, device=dev)
loop_pos_row  = torch.full((M, max_points), -1, dtype=torch.int16, device=dev)
visited       = torch.zeros((M, max_points), dtype=torch.bool,  device=dev)
loop_idx      = torch.zeros(M, dtype=torch.int16, device=dev)
active_cube   = (total_points > 0) & degree_ok

for step in range(max_points):   # upper bound on combined walk length
    # Gather two neighbours
    n0 = adj[arange_M, curr, 0]
    n1 = adj[arange_M, curr, 1]
    nxt = torch.where(prev == -1, n0, torch.where(n0 == prev, n1, n0))
    # Record
    loop_rank_row[arange_M, curr] = torch.where(active_cube, loop_idx, -1)
    loop_pos_row [arange_M, curr] = torch.where(active_cube, pos_counter, -1)
    visited      [arange_M, curr] = True
    # Step forward; on "wrap to start" move to next unvisited point and bump loop_idx
    wrapped = (nxt == loop_start)
    pos_counter = torch.where(wrapped, torch.zeros_like(pos_counter), pos_counter + 1)
    loop_idx    = torch.where(wrapped, loop_idx + 1, loop_idx)
    # Find next start on wrap: scan visited row for first False
    next_start  = torch.argmin(visited.to(torch.int8), dim=1)  # 0 if any False, else 0 (caller checks all visited)
    any_unvisited = ~visited.all(dim=1)
    loop_start  = torch.where(wrapped & any_unvisited, next_start, loop_start)
    prev        = torch.where(wrapped, torch.full_like(prev, -1), curr)
    curr        = torch.where(wrapped, next_start, nxt)
    active_cube &= any_unvisited | ~wrapped                    # inactive when all visited
```

This runs `max_points` iterations (≈ 24) of vectorized ops over `(M,)` tensors — that's 24 × ~8 elementwise/gather ops on tensors of size 275k, each takes microseconds. Expected total: **< 5 ms** vs current **1891 ms** → ≥ 99.7% self-time reduction, well past the 80% target.

Correctness concerns with the padded-walk formulation:
- `argmin(visited.to(int8), dim=1)` returns 0 when all elements are True — that's fine because we gate it with `any_unvisited` and hold `active_cube` off once done.
- For cubes with `total_points < max_points`, the "padding" points have `visited` set pre-loop to True (or equivalent) so `argmin` never returns an inactive slot.
- The `wrapped & any_unvisited` mask is evaluated deterministically (pure elementwise logic, no atomics).

**Recommendation:** T5d should prefer the pure-PyTorch padded-walk (§3.4) as the first concrete implementation, fall back to a Triton/CUDA kernel only if the padded-walk doesn't hit the 80% target (it almost certainly will — 5 ms << 1891 ms).

---

## 4. Consumer adaptation

Two options:

### Option A (PREFERRED) — emit CSR directly from fast-path GPU, skip the Python list
Change the call-site in `s6_collapse` (lines 861-891) to keep fast-path CSR on device. Then line 943-946 (`populate fast-path GPU results from dict`) is replaced with a tensor-concat step in the final assembly (lines 964-1001):

```python
# Before (lines 944-946):
for cube_idx, (loops, status) in fastpath_gpu_results.items():
    per_cube_loops[cube_idx] = loops
    per_cube_status[cube_idx] = status

# After — keep fast-path CSR separate, merge with worker CSR at the end:
# fast_loop_off, fast_edge_off, fast_edge_val : produced by GPU
# worker_loop_off, worker_edge_off, worker_edge_val : from the existing flattening of per_cube_loops[worker_cube_idx]
# Merge by fast_idx_np-indexed interleave:
final_n_loops_per_cube = zeros(N, int64)
final_n_loops_per_cube[fast_idx_np] = fast_loop_count
final_n_loops_per_cube[worker_idx_np] = worker_loop_count
# Then build final loop_cube_off via cumsum, and gather/place fast_edge_off/val and worker_edge_off/val.
```

Pros: zero host ↔ device round-trips for fast-path tracing; ~275k `list.append` calls eliminated.
Cons: more invasive surgery on the assembly block (lines 935-1001). The interleaving requires two scatter_s and a final `torch.cumsum`.

### Option B — CSR → list-of-lists adapter (keeps assembly unchanged)

```python
def _csr_to_list_of_lists(
    fast_loop_off: Tensor, fast_edge_off: Tensor, fast_edge_val: Tensor
) -> List[List[List[int]]]:
    """Convert GPU CSR output back to the list[list[int]] shape s6 currently expects."""
    M = fast_loop_off.numel() - 1
    loops_per_cube: List[List[List[int]]] = [[] for _ in range(M)]
    lo_np = fast_loop_off.cpu().numpy()
    eo_np = fast_edge_off.cpu().numpy()
    ev_np = fast_edge_val.cpu().numpy()
    for c in range(M):
        for li in range(lo_np[c], lo_np[c + 1]):
            loops_per_cube[c].append(ev_np[eo_np[li]:eo_np[li + 1]].tolist())
    return loops_per_cube
```

**Estimated adapter cost:** `M_fast ≈ 275k` outer-loop iterations, each doing ~2 `list.append` and one `np.ndarray.tolist()`. Back-of-envelope: ~100-300 ms, roughly 10-20% of the original 1891 ms. Still hits the 80% target easily but leaves performance on the table.

### Recommendation
**T5b/T5c** start with **Option B** (adapter) to keep the surgery surface small and land the correctness win quickly. **T5d** migrates to **Option A** for the full win, gated by the F1-F3 regression gate.

---

## 5. Risk analysis

### 5.1 Determinism (HIGH importance — regression gate is bit-exact)

| Op | Used? | Risk | Mitigation |
|---|---|---|---|
| `scatter_` (no duplicates) | Yes (writing edge_val into unique slots) | None — each output slot written exactly once | OK |
| `scatter_add_` (with duplicates) | Yes (building `per_loop_len`) | Commutative integer add → deterministic on CUDA for int64 | Use `int64` only; avoid float |
| Atomic add for concurrent loop-count reservation | **No, by design** (two-pass architecture sidesteps this) | Would be nondeterministic | Avoided |
| `torch.cumsum` | Yes | Deterministic on CUDA | OK |
| `torch.searchsorted` | Yes | Deterministic | OK |
| Custom CUDA/Triton kernel | Optional | Must avoid race conditions on `loop_rank` / `loop_pos` writes | Each (c, p) written by exactly one thread (if one-thread-per-cube) — OK |

### 5.2 Warp divergence and occupancy (MEDIUM)

- One thread per cube → 32 threads in a warp can traverse loops of lengths ∈ {4, 4, 4, 4, …, 24}. Longest thread determines warp retire time. Divergence cost ≈ 6× for the tail warp. But: 1075 blocks × 8 warps = 8600 warps total, so a handful of tail warps is a drop in the bucket.
- The padded-walk alternative (§3.4) runs `max_points` uniform iterations for all M cubes → **zero warp divergence**, but does "wasted" work for cubes with short loops. Given `max_points` is tiny (~24), this is clearly fine.

### 5.3 Memory pressure (LOW)

- Scratch: `loop_rank (M, max_points)` int16 + `loop_pos (M, max_points)` int16 = `2 × 275_000 × 24 × 2` = **26 MB**. Trivial.
- Output: `E_fast ≤ total_points.sum()` bounded by total edges × 2-ish = <10 MB.
- No memory bottleneck expected.

### 5.4 Numerical / semantic traps

- `edge_of_point` via `searchsorted(..., right=True) - 1`: matches the numpy reference (line 399-400). Verify direction/sentinel exactly — reference uses `side='right'`, torch uses `right=True`. Equivalent.
- `adj_row[p, slot] == -1` (unused slot) must never be followed. In the reference, this can't happen because the caller gates on `degree_ok`. Preserve that gate in the GPU path.
- **Tie-breaking in neighbour choice when `prev == -1`**: reference picks `n0`. GPU must also pick `n0` (i.e., `adj[c, curr, 0]`) for the first step. With `point_offset_row` deterministic and `adj_row` deterministically packed by `_fastpath_gpu_build_adjacency` (which uses `stable=True` sort), loops will be traversed in the same canonical direction → identical edge-id sequences → identical CSR bytes → regression gate passes.

### 5.5 Top 3 risks

1. **Regression-gate bit-exactness under reordering.** Any reshuffle of `per_cube_loops` insertion order (e.g., processing cubes in GPU thread-id order vs original `local_i` order) could flip the order of loops within a cube. Mitigation: because `per_cube_loops[cube_idx]` is indexed by cube_idx (not appended), insertion order across cubes is irrelevant. **But** the order of loops within a cube IS observable. The CPU reference walks `start = 0, 1, 2, ...`; the GPU padded-walk does too. This must be preserved.
2. **`max_points` upper bound.** If any mesh ever has `max_points > 32` for a single cube, the bitmask trick breaks. Mitigate by asserting `max_points <= 32` at kernel entry and falling back to the CPU path per-cube otherwise. (In practice `max_points` is bounded by `sum(edge_weights) ≤ 18 * max_edge_weight` and fast-path cubes have small edge weights; still — assert and fall back is the safe play.)
3. **Interleave merge complexity (Option A).** Merging fast-path CSR + worker CSR into a single `(loop_cube_off, loop_edge_off, loop_edge_val)` trio without going through `List[List[List[int]]]` requires three synchronized scatters. Easy to get wrong; add a unit test that exercises a mixed-path fixture (F3 already does).

---

## 6. T5b - T5d phased plan

Each sub-task produces a commit that must pass `pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v` on all three fixtures (F1/F2/F3). `num_workers=1` enforced by the runner.

### T5b — TDD scaffold (unit tests only, no production changes)

- **Deliverables:**
  1. `corep_fast/tests/unit/test_s6_fastpath_trace_loops_gpu.py`.
  2. Fixture: hand-crafted `(point_offset, adj, degree_ok)` for 3-4 tiny cubes covering:
     - Cube with 0 loops (`total_points = 0`).
     - Cube with exactly 1 loop of length 4.
     - Cube with 2 disjoint loops (mixed lengths).
     - Cube with `degree_ok = False` (must produce 0 loops / sentinel).
  3. Test runs both the existing `_fastpath_trace_loops_numpy` (per cube) and the new `_fastpath_trace_loops_gpu` (batched) and asserts CSR-equivalent output.
- **Exit criterion:** tests are written and currently FAIL because `_fastpath_trace_loops_gpu` doesn't exist yet. Commit marked `test(s6-fastpath-gpu): scaffold`.

### T5c — Stub implementation: pure-PyTorch padded-walk, Option B adapter

- **Deliverables:**
  1. New `_fastpath_trace_loops_gpu` in `corep_fast/stages/s6_collapse.py` implementing §3.4 (padded walk on `(M,)` state tensors).
  2. Adapter `_csr_to_list_of_lists` (Option B).
  3. Wire into `s6_collapse` behind a config flag `_cfg.S6_FASTPATH_GPU_TRACE_LOOPS` (default OFF) to allow A/B testing. When OFF, fall back to the current numpy per-cube loop.
  4. Turn ON for the default CI/regression gate run (`_cfg` default = True).
- **Exit criterion:**
  - Unit tests from T5b pass.
  - F1/F2/F3 regression gate bit-identical.
  - cProfile on res=256 shows `_fastpath_trace_loops_numpy` total time drop ≥ 80% (target ~95%).
  - Commit: `perf(s6): GPU batched fast-path loop tracer (Option B adapter)`.

### T5d — Option A integration: CSR stays device-resident

- **Deliverables:**
  1. Refactor `s6_collapse` assembly block (lines 935-1001) to accept a device-resident fast-path CSR and interleave it with worker results using `scatter_` over the `final_n_loops_per_cube` indexing described in §4 Option A.
  2. Remove `_csr_to_list_of_lists` adapter.
  3. Remove three `.cpu().numpy()` copies in `_fastpath_gpu_build_adjacency` for outputs that are now consumed on device (but keep `degree_ok_np` since it's still needed by the flag check).
- **Exit criterion:**
  - Unit + regression gate still pass.
  - Measurable reduction in main-thread CPU time for s6 assembly step (profile before/after).
  - Commit: `perf(s6): keep fast-path CSR on device, drop list-of-lists round-trip`.

---

## 7. Open questions

1. **Triton vs pure-PyTorch padded-walk?** Based on the padded-walk analysis in §3.4, pure PyTorch should already beat the 80% target by a wide margin. Question for the controller: is there a reason (e.g., T5d target) to push for Triton anyway? **Recommendation:** stay pure PyTorch for T5c; re-evaluate after measuring.
2. **`max_points` hard cap.** Should T5c assert `max_points <= 32` and raise (forcing T5d to revisit), OR fall back to the CPU path when violated (silently correct)? **Recommendation:** assert in T5c to catch regressions; add graceful fallback in T5d if we ever see it in production.
3. **Interaction with Option A merge and slow-path worker results.** Option A needs the slow-path (CPU worker) results also in CSR. Currently worker results arrive as `List[Tuple[cube_idx, loops, status, assignment]]`. Converting those to CSR on the main thread is cheap but is itself a small optimization that could be folded into T5d or split out.

---

## 8. Surprises from reading the source

- `_fastpath_gpu_build_adjacency` already does its heavy lifting on GPU (lines 190-368), then does `.cpu().numpy()` on three outputs (lines 371-376) just so a Python `for` loop on the main thread can call the trivial `_fastpath_trace_loops_numpy`. The whole point of T5 is to close this last stretch of host-round-trip that the W3 phase-2 effort left on the table.
- The final CSR `(loop_cube_off, loop_edge_off, loop_edge_val)` already matches exactly the layout this design proposes emitting from the tracer. The "CSR pass 2" at lines 966-997 is already spending meaningful time on `np.fromiter` + `np.concatenate`; skipping that for fast-path cubes (Option A) compounds the win.
- `per_cube_loops` is a `List[List[List[int]]]` of size `N`, but fast-path dict (`fastpath_gpu_results`) keys by global cube_idx — so the interleave works naturally via index assignment. Option A's surgery surface is smaller than it first appeared.
- The regression gate (`test_cpu_worker_optim.py`) uses `num_workers=1` + per-fixture subprocess isolation + `PYTHONHASHSEED=0`. Bit-exactness is enforced; any determinism drift in the GPU kernel will be caught immediately. This is good news for T5 — we have a tight oracle.
