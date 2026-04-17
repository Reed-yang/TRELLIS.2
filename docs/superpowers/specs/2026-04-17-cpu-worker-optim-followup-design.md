# corep_fast Residual Optim — Follow-up Design Spec (Phase 1 + Phase 2 combined)

**Date:** 2026-04-17
**Branch:** `post-profile-sync-elim` (continuation, post-handoff)
**Anchor HEAD:** `c0a2956` (T7a close-out of prior spec) or `1ced857` (T9 clean wall)
**Prior spec (closed):** `docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md` (V2)
**Handoff basis:** `docs/superpowers/specs/2026-04-17-cpu-worker-optim-handoff.md`
**Summary basis:** `my-docs/20260417-cpu-worker-optim-summary.md` (505 lines, §6 ROI table)
**Measurement config:** `host-10-240-99-116` GPU 4 (H100 80 GB) or `host-10-240-99-119` GPU 0-3; fixture `icosphere subdiv=3 radius=0.4`, res=256

---

## 0. Why this spec exists

The prior spec (`…-design.md` V2) closed with DoD 6/6 green at e2e **5.354 s** (-38 %
from 8.631 s). That work eliminated the top-3 main-thread hotspots
(`posix.fork`, `_fastpath_trace_loops_numpy`, `_get_local_components_np`) and
was intentionally **Triton-free** to stay low-risk.

Post-change top-20 cProfile shows four residual hotspots (see §2 data). All four
are now either (a) non-mechanical refactors or (b) require device-side control
flow the previous PyTorch-only path could not express. This spec plans the next
ROI-ordered cut.

**Net goal of this spec:** 5.354 s → **2.5–3.2 s** (-40 % to -53 %) by
attacking those four hotspots in two phases:

| Phase | Workstreams | Dep | Effort | Expected Δ |
|---|---|---|---:|---|
| **1** (pure PyTorch) | W_L2L + W_SD | none | 1 week | -1.2 ~ 1.9 s |
| **2** (requires Triton) | W_BAF + W_HG | Phase 1 ships, F1-F3 green | 2-3 weeks | -1.3 ~ 2.5 s |

Phase 1 alone is a 22 %-36 % reduction and carries zero Triton debt. Phase 2
is the heavier investment but uses Phase 1 as a stable baseline.

---

## 1. Goal

Keep DoD from prior spec (F1-F3 bit-exact, VRAM ≤ +500 MB), and add:
**cumulative wall-time reduction ≥ 1.5 s at Phase 1 completion and ≥ 3 s at
Phase 2 completion**; in addition **GPU utilization reported** (nsys single
capture, target ≥ 30 % after Phase 2 vs current estimated ~10-15 %).

Non-goal: Triton for W4/W5/W2 residuals (their ROI is too low; kept on the
deferred list in §7).

---

## 2. Background — what the data says

### 2.1 Post-W2+W4+W5 top-20 main-thread cProfile (from `tmp/cpu_profile/t9_final_hotspots.txt`)

| Rank | Self (ms) | Calls | Function | File:line | Covered by this spec? |
|---:|---:|---:|---|---|---|
| 1 | 1648.5 | 4 | `_thread.lock.acquire` | main thread waits on `pool.map` (Stage D + s7 Phase 3) | **W_SD** (Stage D part) |
| 2 | 999.0 | 1 | `s7_rank_assign` | s7_rank_assign.py:1175 (Phase 3 Hungarian loop) | **W_HG** |
| 3 | 513.7 | 1 | `s6_collapse` | s6_collapse.py top-level (assembly + CSR repack) | deferred (see §7) |
| 4 | 464.1 | 1 | `_labels_to_list_of_lists` | s4_face_point.py:966 | **W_L2L** |
| 5-10 | ~100-300 ea | ~275k ea | numpy coercions (`asarray`/`tolist`/`astype`) | s4 / s7 | deferred (see §7) |
| — | 1913.7 | 1 | `_build_adjacency_gpu` | s7_rank_assign.py:634 | **W_BAF** (not main-thread Python but GPU compute + host loop control) |

Total addressed by this spec: **≈5.0 s** of the 5.354 s residual. After realistic
recovery (not every ms is reclaimable), expected net: **-2.5 to -3.0 s**.

### 2.2 Why GPU is idle 85-90 % of wall

From `my-docs/20260417-cpu-worker-optim-summary.md` §5.2:

| Residual | idle reason | category |
|---|---|---|
| `lock.acquire 1648 ms` | worker runs **pure-Python BFS/graph-walk** (`_count_uturns`, 49.7 s worker wall / 275 k cubes); GPU sits idle while main thread blocks on `pool.map` | (A) pure CPU algorithm |
| `s7_rank_assign 999 ms` | 275 539 × `scipy.optimize.linear_sum_assignment` on main thread | (A) pure CPU algorithm |
| `_labels_to_list_of_lists 464 ms` | 1 × `.cpu()` then Python bucket loop | (B) CPU↔GPU boundary |
| `_build_adjacency_gpu 1913 ms` | GPU **is** busy but with 576 tiny scatter launches + host-side loop controlling them | (D) device-side control flow needed → Triton |

**Phase 1 attacks (A) + (B)** with batched PyTorch. **Phase 2 attacks (A-small)
+ (D)** with Triton.

### 2.3 Determinism constraint (carried from prior spec)

Every workstream must pass F1/F2/F3 regression gate. Gate enforces bit-exact V/F
count and V position max-abs-diff ≤ 1e-5 vs frozen nw=1 goldens. The gate's
three-layer determinism (SerialPool monkeypatch + subprocess-per-fixture +
CUDA deterministic flags) is reused verbatim from
`corep_fast/tests/regression/_cpu_worker_optim_runner.py`. **Do not modify the
runner** unless adding a new fixture.

---

## 3. Scope

### 3.1 In-scope (ROI-ordered)

| ID | Workstream | Phase | Primary file:line | Expected Δ | Effort | Tool |
|---|---|:---:|---|---:|---:|---|
| **W_L2L** | `_labels_to_list_of_lists` numpy-vectorize | 1 | `s4_face_point.py:966` | 0.2 – 0.4 s | 0.5 – 1 d | numpy |
| **W_SD** | Stage D GPU BFS (W6 Angle 2 from prior spec) | 1 | `s4_face_point.py:305,398,1329` | 1.0 – 1.5 s | 3 – 5 d | PyTorch |
| **W_BAF** | `_build_adjacency_gpu` Triton fusion | 2 | `s7_rank_assign.py:634` | 0.5 – 1.0 s | 4 – 7 d | **Triton** |
| **W_HG** | s7 Phase 3 batched Hungarian on GPU | 2 | `s7_rank_assign.py:1485-1512` | 0.8 – 1.5 s | 4 – 7 d | **Triton** (or PyTorch fallback) |

Phase 1 total: 1.2-1.9 s in 3.5-6 d. Phase 2 total: 1.3-2.5 s in 8-14 d.
Combined: 2.5-4.4 s in 12-20 d (≈3-4 weeks).

### 3.2 Out of scope / deferred (see §7 for the rationale)

- W4 Option A (CSR-native downstream of `_fastpath_trace_loops_gpu`) — ROI too low
- `s6_collapse` assembly block (513 ms) — mostly numpy repack, no mechanical win
- numpy coercion cluster (~759 ms) — will be absorbed incidentally by any future
  "keep intermediates on GPU through s4→s7" pass
- s1/s2/s3 optimizations (combined < 0.3 s)
- Any new dependency beyond Triton (no Cython / Numba / Rust)

### 3.3 Rejected (from prior spec, do not re-open without new data)

- "Delete MP" — T0 decisive experiment showed serial is 8.7× slower
- W6 Angle 3 (ThreadPool) — GIL-holding 57.9 %, dead per T7a
- W1 Bucket A deletions — both sites load-bearing per T2 audit
- W7 s7 orchestration cleanup — zero ≥200 ms mechanical candidate per T8a

### 3.4 VRAM discipline (unchanged from prior spec)

Each workstream records peak allocated + reserved VRAM pre / post. Hard limit:
any single W that adds > 500 MB @ res=256 must have explicit user sign-off.
Baseline @ HEAD `1ced857`: 5687.8 MB allocated / 13220.4 MB reserved.

---

## 4. Per-workstream design

### 4.1 W_L2L — `_labels_to_list_of_lists` numpy-vectorize

#### 4.1.1 Current state (`corep_fast/stages/s4_face_point.py:966-1012`)

```python
def _labels_to_list_of_lists(
    batched_labels: torch.Tensor,     # (N, M) int64, SENTINEL=M for pad
    batched_face_ids: torch.Tensor,   # (N, M) int64, -1 pad
    face_counts: torch.Tensor,        # (N,) int64
) -> list[list[list[int]]]:
    order = torch.argsort(batched_labels, dim=1, stable=True)    # (N, M)
    sorted_labels = batched_labels.gather(1, order)              # (N, M)
    sorted_fids = batched_face_ids.gather(1, order)              # (N, M)

    # 1 × cpu() + 1 × cpu() + 1 × cpu() — single D2H staging
    sorted_labels_cpu = sorted_labels.cpu().numpy()
    sorted_fids_cpu = sorted_fids.cpu().numpy()
    counts_cpu = face_counts.cpu().numpy()

    # ← pure Python per-row bucket loop (N ≈ 275k)
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
```

**Measurement:** 464 ms self @ T9 cProfile. 275 k+ `int()` casts + 275 k+ list
appends + 275 k+ inner Python loop over up to M≈20 per row.

**Consumer:** `_compute_component_points_gpu` in s4_face_point.py calls this to
feed its downstream SH-clip + fan-centroid loop, which reads
`list[list[list[int]]]` shape `[cube][component][face_id]`.

#### 4.1.2 Design — batch numpy pack

Replace the per-row Python loop with batched numpy operations using two boundary
flags — `is_new_component` and `is_last_face` — derived vectorized.

```python
# All on CPU after single .cpu() each:
# sorted_labels_cpu (N, M) int64
# sorted_fids_cpu   (N, M) int64
# counts_cpu        (N,)    int64

# Mask: valid slot = k < counts_cpu[i]
k_idx = np.arange(M, dtype=np.int64)          # (M,)
valid_mask = k_idx[None, :] < counts_cpu[:, None]   # (N, M) bool

# Component boundary: slot k starts a new component if
#   k == 0 OR sorted_labels[i, k] != sorted_labels[i, k-1]
# (restricted to valid slots)
prev_labels = np.concatenate(
    [np.full((N, 1), -1, dtype=np.int64), sorted_labels_cpu[:, :-1]],
    axis=1,
)   # (N, M)
is_new_component = (sorted_labels_cpu != prev_labels) & valid_mask   # (N, M)

# For each cube, count the number of components = sum of is_new_component
comps_per_cube = is_new_component.sum(axis=1)     # (N,)
# Total components across all cubes
C = int(comps_per_cube.sum())

# Cumulative component index per valid slot:
#   slot i,k contributes to comp_idx[i,k] = cumsum(is_new_component) - 1
comp_idx_flat = (is_new_component.cumsum(axis=1) - 1).reshape(-1)   # (N*M,)
fids_flat = sorted_fids_cpu.reshape(-1)                              # (N*M,)
valid_flat = valid_mask.reshape(-1)                                  # (N*M,)

# Per-cube base component offset
comp_off_per_cube = np.concatenate(
    [np.array([0], dtype=np.int64), comps_per_cube.cumsum()[:-1]]
)   # (N,)

# Global component index = per-cube base + per-slot local comp idx
cube_idx_flat = np.repeat(np.arange(N, dtype=np.int64), M)           # (N*M,)
global_comp_idx = comp_off_per_cube[cube_idx_flat] + comp_idx_flat   # (N*M,)

# Group fids by global component via np.argsort + np.split or bincount:
# fastest is to sort by global_comp_idx and split on boundaries.
valid_gci = global_comp_idx[valid_flat]          # (V,)  V = valid.sum()
valid_fids = fids_flat[valid_flat]                # (V,)

# Global split points (where gci increments)
split_at = np.flatnonzero(np.diff(valid_gci) > 0) + 1   # (C-1,)
fids_per_comp = np.split(valid_fids, split_at)           # list of C numpy arrays

# Finally rebuild nested list shape using comps_per_cube:
result = [None] * N
cursor = 0
for i in range(N):
    c_i = int(comps_per_cube[i])
    # Each inner list is a numpy array; consumer already handles np.ndarray
    result[i] = [fids_per_comp[cursor + j].tolist() for j in range(c_i)]
    cursor += c_i
return result
```

**Expected speedup:** inner Python loop over `N*M ≈ 5.5M` slot-casts → batched
numpy of the same work. The outer `for i in range(N)` remains because the
return type is `list[list[list[int]]]` — but it now only does **C** `tolist()`
calls (C ≈ total components ≈ 350k) instead of 5.5M `int()` casts. Net ≈ 3-5×.

**Alternative (if downstream tolerates `list[np.ndarray]` instead of
`list[list[int]]`):** skip the final `.tolist()` on each component — save
another ~100 ms. Check `_compute_component_points_gpu` consumer to see if it
iterates with `for fid in comp` or does `np.asarray(comp)`; the latter means
`np.ndarray` is a drop-in.

#### 4.1.3 Determinism

- `np.argsort` is stable (`kind='stable'` guaranteed since numpy 1.15).
- `cumsum` + `flatnonzero` + `split` are all deterministic.
- No atomics, no hash-based sets.

**F1-F3 gate:** must pass bit-exact. If any ordering differs, regression test
catches it.

#### 4.1.4 TDD hooks

```
tests/unit/test_labels_to_list_vectorized.py
    test_matches_legacy_python_bucket_loop()
        # Random (N, M) labels + counts; assert new == old
    test_empty_rows_preserved()
        # counts_cpu[i] = 0 → result[i] = []
    test_all_same_label_single_component()
        # 1 component per cube
    test_all_distinct_labels_singleton_components()
        # n_i components per cube
```

Plus F1-F3 regression gate on commit.

#### 4.1.5 Risks

- **Downstream consumer type:** if `_compute_component_points_gpu` does
  `len(comp)` or `for fid in comp`, both work for `list[int]` and
  `np.ndarray`. Not `.append()` though. **Audit pre-commit.**
- **Memory:** `(N*M)` intermediate flats ≈ 5.5M × 8 bytes = 44 MB; negligible.

#### 4.1.6 DoD

- [ ] All 4 unit tests pass
- [ ] F1-F3 regression gate passes
- [ ] Clean wall `tmp/cpu_profile/t0_driver.py --mode default --res 256` ≥ 0.15 s below pre-change baseline
- [ ] cProfile `_labels_to_list_of_lists` self ≤ 150 ms (from 464 ms, -68 %)

---

### 4.2 W_SD — Stage D GPU BFS (W6 Angle 2 from prior spec)

#### 4.2.1 Current state

**Driver (`s4_face_point.py:1329 _p2_uturn_worker`):** runs on MP workers,
one worker call per group `gi` (groups are packed facets on segments).

```python
def _p2_uturn_worker(gi: int) -> tuple[int, int, int]:
    cf = int(_P2_CF[gi])
    cube_id = cf // 12
    facet_id = cf % 12
    lo = int(_P2_GROUP_OFF[gi])
    hi = int(_P2_GROUP_OFF[gi + 1])
    segs = [(_P2_SEGS_A[i], _P2_SEGS_B[i]) for i in range(lo, hi)]
    # ... reconstruct facet verts from fork-inherited globals ...
    return (cube_id, facet_id, _count_uturns(segs, V0, V1, V2, cube_verts, v_ids, e_ids))
```

**Core (`s4_face_point.py:305 _count_uturns`):**

```python
def _count_uturns(segments, V0, V1, V2, cube_verts, vert_ids, edge_ids) -> int:
    # Phase 1: Build topological graph (node coalescence via linear scan)
    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    for p1, p2 in segments:
        idx1 = _find_or_add_node(nodes, p1)    # O(n) scan per call
        idx2 = _find_or_add_node(nodes, p2)    # O(n) scan per call
        if idx1 != idx2:
            edges.append((min(idx1, idx2), max(idx1, idx2)))

    unique_edges = set(edges)
    adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for u, v in unique_edges:
        adj[u].append(v); adj[v].append(u)

    # Phase 2: BFS connected components
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(len(nodes)):
        if i not in visited:
            comp = []
            q = [i]; visited.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor); q.append(neighbor)
            components.append(comp)

    # Phase 3: Endpoint edge-projection and U-turn pair counting
    v0i, v1i, v2i = vert_ids
    facet_verts_for_edges = [cube_verts[v0i], cube_verts[v1i], cube_verts[v2i]]
    total_uturns = 0
    for comp in components:
        endpoints = [n for n in comp if len(adj[n]) == 1]
        if not endpoints:
            continue  # closed loops don't contribute
        edge_counts = {-1: 0}
        for e_idx in edge_ids:
            edge_counts[e_idx] = 0
        for ep in endpoints:
            P = nodes[ep]
            assigned = []
            for j in range(3):
                A = facet_verts_for_edges[j]
                B = facet_verts_for_edges[(j + 1) % 3]
                edge_vec = B - A
                length = np.linalg.norm(edge_vec)
                if length < 1e-12: continue
                t = np.dot(P - A, edge_vec) / (length ** 2)
                if -1e-8 <= t <= 1.0 + 1e-8:
                    proj = A + t * edge_vec
                    if np.linalg.norm(P - proj) < 1e-8:
                        assigned.append(edge_ids[j])
            if not assigned:
                assigned.append(-1)
            for e_id in assigned:
                edge_counts[e_id] += 1
        for e_id, count in edge_counts.items():
            if e_id != -1:
                total_uturns += count // 2
    return total_uturns
```

**Measurement (T7a `logs/findings_w6_gil_spike.md`):**
- 125 273 ms worker wall
- `_count_uturns` self = 49 679 ms (40 % of total self_tt, 1 881 777 calls)
- `_find_or_add_node` self = 3 963 ms (3 974 410 calls, O(n) scan)
- Main-thread `lock.acquire` = 1648 ms (waiting on `pool.map`)

#### 4.2.2 Design — batched GPU BFS + endpoint projection

**Data layout (pre-call, build once per `_compute_face_weights_gpu`):**

| Tensor | Shape | Meaning |
|---|---|---|
| `group_seg_off` | (G+1,) int64 | CSR offset into segments, per group |
| `group_segs_A` | (S, 3) float64 | segment start points |
| `group_segs_B` | (S, 3) float64 | segment end points |
| `group_cube_id` | (G,) int64 | cube index per group |
| `group_facet_id` | (G,) int64 | facet index (0..11) per group |
| `cube_verts_all` | (N, 8, 3) float64 | precomputed per-cube vertex coords |

All derived from existing data already on GPU via Stage B of
`_compute_face_weights_gpu`. No new host-side work.

**Phase A — batched node coalescence (per-group KxK symmetric clustering):**

For each group, we have up to `max_s` segments (observed max ≈ 20). Each
segment has 2 endpoints → up to `2 * max_s` candidate points. We need a
canonical "unique nodes" labeling compatible with the Python
`_find_or_add_node` (insertion-order, 1e-8 tolerance).

Since `max_s` is tiny, we can do a **fully batched O(max_s²) intra-group**
comparison:

```python
# pts: (G, 2*max_s, 3) — packed seg A then B, masked by valid
# pts_mask: (G, 2*max_s) bool

# Pairwise dist: d[g, i, j] = ||pts[g,i] - pts[g,j]||
d = torch.cdist(pts, pts)                          # (G, P, P) float64
match = d < 1e-8                                    # (G, P, P) bool

# For each slot i, its canonical index is min j where match[g,i,j] and j <= i
# (first-occurrence wins → matches Python insertion-order semantics).
# Mask upper triangle: only j <= i counts
tri = torch.tril(torch.ones_like(match[0]))        # (P, P) lower-tri
match_lower = match & tri.unsqueeze(0).bool()
# For each (g, i), min j over valid
node_idx = (~match_lower * P).min(dim=-1).values    # (G, P) — sentinel P if none, but diag match_lower[i,i]=True so always ≥ valid
# Actually simpler: since match_lower[i,i]==True always, min is well-defined.

# Compact: per-group unique node ids = unique values of node_idx
# We then rename to 0..n_nodes-1 per group via inverse map
```

This is O(G * P²) with P ≤ 40, so ≈ 160k G × 1600 = 256M ops per call —
≈ 1-2 ms on H100.

**Phase B — build adjacency and find connected components (batched
label-propagation, same technique as W5):**

```python
# edge_list[g]: pairs (ua, ub) where ua, ub are canonical node idx within group g
# Batched: edge_mask[g, i, j] bool indicates "edge between nodes i and j in g"
edge_mask = match  # reuse pts coalescence, zero out diagonals

# Label propagation: labels[g, i] = i initially
# labels[g, i] = min(labels[g, i], min_{j: edge_mask[g,i,j]} labels[g, j])
labels = torch.arange(P, device=device).view(1, P).expand(G, P).clone()
for _ in range(max_iters):   # diameter ≤ P, typically < 8
    # Gather min over neighbors
    neighbor_labels = torch.where(
        edge_mask, labels.unsqueeze(1).expand(G, P, P), torch.full_like(edge_mask, P, dtype=labels.dtype)
    )
    min_nbr = neighbor_labels.min(dim=-1).values
    new_labels = torch.minimum(labels, min_nbr)
    if torch.equal(new_labels, labels): break
    labels = new_labels
```

**Phase C — endpoint classification (per-node degree from edge_mask):**

```python
degree = edge_mask.sum(dim=-1)                # (G, P) int
is_endpoint = (degree == 1) & pts_mask       # (G, P) bool
```

**Phase D — edge-projection + U-turn counting:**

For each endpoint, project onto the 3 triangle edges and record which (or
sentinel -1). Then per component × per edge_id, count endpoints → `count // 2`.

```python
# facet_verts: (G, 3, 3) — facet triangle verts per group (from cube_verts_all[cube_id, v_ids])
A = facet_verts[:, torch.tensor([0,1,2])]              # (G, 3, 3)
B = facet_verts[:, torch.tensor([1,2,0])]              # (G, 3, 3)
edge_vec = B - A                                        # (G, 3, 3)
length_sq = (edge_vec * edge_vec).sum(dim=-1)           # (G, 3)
# for each node p in each group g, project onto 3 edges
P_minus_A = pts.unsqueeze(2) - A.unsqueeze(1)           # (G, P, 3, 3)
t = (P_minus_A * edge_vec.unsqueeze(1)).sum(dim=-1) / (length_sq.unsqueeze(1) + 1e-30)  # (G, P, 3)
in_range = (t >= -1e-8) & (t <= 1.0 + 1e-8)            # (G, P, 3)
proj = A.unsqueeze(1) + t.unsqueeze(-1) * edge_vec.unsqueeze(1)   # (G, P, 3, 3)
dist_to_proj = ((pts.unsqueeze(2) - proj) ** 2).sum(dim=-1).sqrt()  # (G, P, 3)
on_edge = in_range & (dist_to_proj < 1e-8)              # (G, P, 3)

# assigned_edges[g, p, j] = edge_ids[g, j] if on_edge[g, p, j] else -1
# For endpoints that project to no edge, assigned = -1 sentinel
edge_ids_b = edge_ids.view(G, 1, 3).expand(G, P, 3)     # (G, P, 3)
assigned = torch.where(on_edge, edge_ids_b, torch.full_like(edge_ids_b, -1))

# U-turn count per group:
#   For each (group, component, edge_id), count how many endpoints are on it,
#   then sum count // 2 over edge_ids ≠ -1.

# labels are valid 0..P-1 component ids per group
# Use torch.bincount with flat (group, label, edge_id) key.

# Flat key = ((g * max_nodes) + label[g,p]) * max_edge_ids + (edge_id + 1)
# For each endpoint's each assigned edge (up to 3), +1
# Then sum count // 2 only for edge_id != -1.
```

Tiebreaker to match Python: endpoint iteration order inside a component must
match `_count_uturns`'s `for ep in endpoints` (which is the BFS order of
visiting nodes). Since `count_uturns` only does `count // 2`, **the order
doesn't affect the output** — only the count does. So the final integer is
invariant to visit order. Good.

**Total return:** `(G,) int64` tensor of U-turn counts per group. Back in
`_compute_face_weights_gpu` the caller fuses these into
`edge_weights - 2 * u_per_edge` (already GPU-side).

#### 4.2.3 Determinism

- `cdist` is deterministic (batched GEMM).
- `torch.argsort(stable=True)` used.
- `torch.equal` for loop termination.
- `bincount` is deterministic when input is int64 on CUDA for counts
  (uses atomics but **only for counts, not for ordering**; counts are
  commutative).
- No ordering-dependent operation. Result is an integer count.

**F1-F3 gate:** must pass bit-exact. Since `_count_uturns` returns an integer
and downstream consumption is `edge_weights - 2 * u_per_edge`, any divergence
manifests as different edge weights → different loop_edge_rank → different V/F.

#### 4.2.4 TDD hooks

```
tests/unit/test_stage_d_gpu_bfs.py
    test_matches_legacy_numpy(num_groups=100, seed=0)
        # Random synthetic groups; pass identical input to both paths.
        # Assert per-group U-turn count matches.
    test_zero_segments_group_zero_uturns()
    test_closed_loop_no_endpoints_zero_contribution()
    test_single_edge_two_endpoints_one_uturn_if_same_edge_id()
    test_batched_larger_than_gpu_mem_chunked()
        # G = 500k, ensure we don't OOM (chunk over G if needed)
```

Plus F1-F3 regression gate.

#### 4.2.5 Risks

1. **Numerical tolerance drift:** nodes coalesce at 1e-8 Euclidean distance.
   numpy version uses linear scan with same tolerance. GPU `cdist` may have
   ±1 ULP differences that flip the `<1e-8` decision. **Mitigation:** tighten
   comparison to `1e-8 + eps_float64 * 1e3` as safety margin, or run GPU
   coalescence at f64 exclusively.

2. **P² memory at large groups:** if some group has P=200 candidates, (G, P, P)
   is 160k × 40k × 1 byte = 6.4 GB. Observed max P is ≈ 40 but must
   **bound-check at runtime** and fall back to chunked or legacy for outliers.

3. **Sign of face_weight:** U-turn count only affects `edge_weights` through
   `ew_eff = ew - 2 * u_per_edge`. If `u_per_edge` is off by 1 anywhere,
   downstream s7 rank-assign diverges. **Belt-and-braces:** add a unit test
   that compares per-group U-turn count for 5 representative cubes from the F2
   fixture.

4. **Worker pool retirement:** if W_SD replaces all of Stage D worker calls,
   the persistent pool in s4 is unused after this change. **Keep pool alive**
   (other stages still use it); just stop dispatching Stage D through it.

#### 4.2.6 DoD

- [ ] 5 unit tests pass
- [ ] F1-F3 bit-exact
- [ ] `_p2_uturn_worker` call count drops to 0 (MP dispatch removed)
- [ ] `lock.acquire` self drops by ≥ 900 ms (Stage D component; s7 Phase 3 component remains until W_HG lands — target its residual in W_HG DoD, not here)
- [ ] Clean wall reduction ≥ 0.8 s vs pre-change baseline
- [ ] VRAM peak ≤ baseline + 500 MB

#### 4.2.7 Execution subtasks (TDD)

1. **Spike** (~0.5 d): run numpy `_count_uturns` on one F2 group in isolation,
   record node count histogram + group-size histogram. Confirm P ≤ some bound
   (or decide chunking strategy).
2. **Red test** (~0.5 d): write `test_stage_d_gpu_bfs.py::test_matches_legacy_numpy`
   — should fail with ImportError since `_count_uturns_gpu_batched` doesn't exist.
3. **Stub** (~0.5 d): `_count_uturns_gpu_batched(groups) → u_per_edge_tensor`
   calls legacy numpy path; green test.
4. **Real impl** (~2-3 d): write Phase A/B/C/D as above.
5. **Integration** (~0.5 d): replace `pool.map(_p2_uturn_worker, ...)` call at
   s4:1186 with one GPU batch call. Keep legacy path behind a `_cfg.STAGE_D_GPU`
   flag for rollback.
6. **Full regression** (~0.5 d): F1/F2/F3 + clean wall + VRAM.

---

### 4.3 W_BAF — `_build_adjacency_gpu` Triton fusion

#### 4.3.1 Current state (`s7_rank_assign.py:634-800`)

```python
def _build_adjacency_gpu(edge_weights, uturn_assignment) -> adj:
    # ... setup: facets, is_fast, uturn_clean, u_per_edge, ew_eff, k_pair ...
    # k_pair: (N, 12, 3) int64 = (w_a + w_b - w_c) // 2, clamped [0, W]

    # ... pts_A, pts_B, node_A, node_B: all (N, 12, 3, W) int64 ...

    # The killer: fixed-size 576-iteration Python loop
    for t_idx in range(12):
        for pi in range(3):
            for jj in range(W):
                mask = valid_arc[:, t_idx, pi, jj]   # (N,) bool
                if not mask.any():
                    continue
                idx = cube_arange[mask]              # (M,) cube indices
                src_A = node_A[idx, t_idx, pi, jj]
                src_B = node_B[idx, t_idx, pi, jj]
                # Place into adj: for each (cube, src), find first empty slot
                # using fill_count, write the dst.
                slot_A = fill_count[idx, src_A]
                fill_count[idx, src_A] = slot_A + 1
                adj[idx, src_A, slot_A] = src_B.to(torch.int32)
                # ...and symmetric B→A ...
    return adj
```

**Measurement:** 1913 ms self. 576 kernel launches × 2 (bidirectional) = 1152
small scatter ops. GPU utilization during this function ≈ 20-30 % (most time is
launch overhead + host loop control).

#### 4.3.2 Design — single Triton kernel

**Key insight:** the 576 iterations are **embarrassingly parallel** at the
`(cube, t_idx, pi, jj)` level; the race condition is only on the per-`(cube,
src_node)` slot assignment (0 or 1). A Triton kernel with atomic-increment on
`fill_count` + conditional write to `adj[slot]` fuses all of this.

**Kernel signature:**

```
@triton.jit
def _build_adjacency_fused_kernel(
    edge_weights_ptr,       # (N, 18) int64
    uturn_assign_ptr,       # (N, 12, 3) int64
    facets_ptr,             # (12, 3) int64  (constant)
    eA_tab_ptr, eB_tab_ptr, # (12, 3) int64  (constant)
    a_at_v0_tab_ptr, b_at_v0_tab_ptr,  # (12, 3) bool (constant)
    eC_tab_ptr,             # (12, 3) int64  (constant)
    adj_ptr,                # (N, NODES, 2) int32 — output, init -1
    fill_count_ptr,         # (N, NODES) int32 — output, init 0
    N: tl.constexpr,
    NODES: tl.constexpr,  # = 18 * W_MAX
    W: tl.constexpr,      # = W_MAX
    BLOCK_SIZE: tl.constexpr,
):
    # Each program: handles one cube × one facet × one pair × one j
    # pid decomposition: pid = cube_id * (12 * 3 * W) + t_idx * (3 * W) + pi * W + jj
    pid = tl.program_id(0)
    jj = pid % W; pid //= W
    pi = pid % 3; pid //= 3
    t_idx = pid % 12; pid //= 12
    cube_id = pid

    if cube_id >= N: return

    # Load k_pair for this (cube, t_idx, pi)
    # k_pair = (w_a + w_b - w_c) // 2 with ew_eff
    # w_a, w_b, w_c: from ew_eff (N, 18) via eA_tab/eB_tab/eC_tab (12, 3)
    eA = tl.load(eA_tab_ptr + t_idx * 3 + pi)
    eB = tl.load(eB_tab_ptr + t_idx * 3 + pi)
    eC = tl.load(eC_tab_ptr + t_idx * 3 + pi)

    # is_fast detection from uturn_assign
    uturn_00 = tl.load(uturn_assign_ptr + cube_id * (12 * 3) + 0)
    is_fast = (uturn_00 == -1)

    # Compute ew_eff for eA, eB, eC (with u-subtraction if slow-path)
    # ... (inline scatter-sum of uturn over facets_ptr to get u_per_edge[eA/eB/eC]) ...
    # This is the trickiest part: reconstructing u_per_edge for just these 3
    # edges without materializing the full (N, 18) buffer.
    u_a = _compute_u_for_edge(cube_id, eA, is_fast, uturn_assign_ptr, facets_ptr)
    u_b = _compute_u_for_edge(cube_id, eB, is_fast, uturn_assign_ptr, facets_ptr)
    u_c = _compute_u_for_edge(cube_id, eC, is_fast, uturn_assign_ptr, facets_ptr)

    w_a_raw = tl.load(edge_weights_ptr + cube_id * 18 + eA)
    w_b_raw = tl.load(edge_weights_ptr + cube_id * 18 + eB)
    w_c_raw = tl.load(edge_weights_ptr + cube_id * 18 + eC)
    w_a_eff = w_a_raw - 2 * u_a
    w_b_eff = w_b_raw - 2 * u_b
    w_c_eff = w_c_raw - 2 * u_c

    k_pair = max(0, min(W, (w_a_eff + w_b_eff - w_c_eff) // 2))

    if jj >= k_pair: return   # no arc for this j

    # Compute src/dst node ids
    a_at_v0 = tl.load(a_at_v0_tab_ptr + t_idx * 3 + pi)
    b_at_v0 = tl.load(b_at_v0_tab_ptr + t_idx * 3 + pi)
    pts_A = jj if a_at_v0 else (w_a_raw - 1 - jj)
    pts_B = jj if b_at_v0 else (w_b_raw - 1 - jj)
    node_A = eA * W + pts_A
    node_B = eB * W + pts_B

    # Both directions: A→B then B→A
    # Atomic increment on fill_count[cube, node_X]; result is the slot idx.
    slot_A = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_A, 1)
    if slot_A < 2:
        tl.store(adj_ptr + cube_id * NODES * 2 + node_A * 2 + slot_A, node_B)

    slot_B = tl.atomic_add(fill_count_ptr + cube_id * NODES + node_B, 1)
    if slot_B < 2:
        tl.store(adj_ptr + cube_id * NODES * 2 + node_B * 2 + slot_B, node_A)
```

**Helper `_compute_u_for_edge(cube_id, target_edge, is_fast, uturn, facets)`**:
scan the 12×3 facet table; for each slot where `facets[t_idx, slot] ==
target_edge`, add `uturn[cube_id, t_idx, slot]` (if not is_fast else 0). This
is 36 loads per call; cheap.

**Launch geometry:** `grid = (N * 12 * 3 * W,)` with 1 program per (cube, t, pi,
jj) tuple. For N=275k, W=6 → 59.4M programs. Block size 256 → 232k blocks.
Modern H100 handles this in < 50 ms based on analog from simpler scatter kernels.

**Expected speedup:** 1913 ms → 100-200 ms (**-1.7 to -1.8 s**, 10x-20x).

#### 4.3.3 Determinism

**Non-trivial.** The `atomic_add` on `fill_count` returns the *current* value
before add, so the slot assignment depends on the order atomics resolve. On
CUDA, this order is nondeterministic across programs.

**Impact assessment:** downstream consumer `_phase1_gpu_rank_assign` reads `adj`
and does BFS from a canonical start. If adj[cube, src, 0] vs adj[cube, src, 1]
swap, the BFS visit order changes → loop ordering changes → V/F counts match
(topologically identical) but V positions may differ by permutation-of-equals.

**Two mitigation strategies:**

**Strategy A — canonicalize after fill:**
```python
# After kernel: sort adj[cube, src, :] ascending so slot 0 < slot 1 always
adj_sorted, _ = adj.sort(dim=-1)
# But -1 sentinels (from `if slot_A < 2` early-out) need to stay at end:
valid_mask = adj >= 0
# Push -1s to the back:
adj_canonical = torch.where(valid_mask, adj, torch.full_like(adj, NODES))
adj_canonical, _ = adj_canonical.sort(dim=-1)
adj_canonical = torch.where(adj_canonical == NODES, -1, adj_canonical)
```

This gives bit-identical output regardless of atomic order. **This is the
recommended approach.** Sorts `(N, NODES, 2)` → one kernel launch, ~5 ms.

**Strategy B — eliminate atomics:**
Since `fill_count` always ends up at 0, 1, or 2 per node, pre-compute the slot
using a different approach: sort arcs by `(cube, src_node, dst_node)` and then
use a segmented `cumsum` for slot assignment. More complex but fully
deterministic and faster (no atomics).

**Decision:** start with Strategy A (simpler, proven). If DoD met, stop. If
Strategy A bottlenecks at the sort, upgrade to Strategy B.

**F1-F3 gate with Strategy A:** will pass.

#### 4.3.4 TDD hooks

```
tests/unit/test_build_adjacency_gpu_fused.py
    test_fast_path_cube_matches_legacy(num_cubes=1000)
        # All cubes is_fast; compare adj output with legacy non-Triton path
    test_slow_path_cube_matches_legacy(num_cubes=1000)
        # All cubes slow-path with random uturn_assignment
    test_mixed_path(num_cubes=5000)
        # 50/50 fast/slow; compare canonical-sorted adj bit-identical
    test_edge_cases()
        # k_pair = 0 for all → adj all -1
        # W = max, k_pair at boundary
    test_large_batch(num_cubes=300000)
        # No OOM, wall ≤ 200 ms on H100
```

Plus F1-F3 regression gate (critical: this is the highest-risk determinism
item in the spec).

#### 4.3.5 Risks

1. **Determinism (primary risk):** covered by Strategy A sort above. Verify
   on all 3 fixtures before declaring W_BAF green.

2. **Triton kernel complexity:** requires device-side branching, table loads,
   and atomic slot assignment. **Spike first** (~1 d) — a "minimum viable"
   kernel that only handles fast-path cubes, compare with legacy `k_pair`
   output before adding U-turn logic. If the spike fails to converge in 1 d,
   fall back to **Strategy C:** vectorize the Python loop in PyTorch (single
   scatter over flattened (N, 12, 3, W) tensor). Expected speedup 3-5x vs
   Triton's 10-20x but risk-free.

3. **Team Triton ramp-up:** if team has no Triton experience, add 2 d for
   environment setup + reading `_phase1_gpu_rank_assign` (already uses Triton
   elsewhere — confirm patterns).

4. **Kernel autotune cache:** Triton's JIT caches per-shape. Across fixtures
   the shape `(N, 12, 3, W)` differs in N only — Triton handles this fine via
   the `tl.constexpr` signature.

#### 4.3.6 DoD

- [ ] 5 unit tests pass
- [ ] F1-F3 bit-exact (after Strategy A canonicalization)
- [ ] `_build_adjacency_gpu` self ≤ 200 ms (from 1913 ms, -90 %)
- [ ] Clean wall reduction ≥ 0.5 s vs Phase-1-end baseline
- [ ] VRAM peak ≤ baseline + 500 MB

#### 4.3.7 Execution subtasks

1. **Spike** (~1 d): minimal fast-path-only Triton kernel; verify output vs
   legacy on 100 fake fast-path cubes. If fails → fall back to Strategy C.
2. **Red test** (~0.5 d): 5 unit tests failing.
3. **Stub** (~0.5 d): Triton kernel that delegates to legacy path internally
   (just to thread the dispatch wiring).
4. **Real impl** (~2 d): full kernel with fast + slow paths + Strategy A sort.
5. **Integration** (~0.5 d): swap call at s7:634. Keep legacy behind
   `_cfg.BUILD_ADJACENCY_TRITON` flag.
6. **Determinism audit** (~1 d): run F1-F3 20× with `torch.manual_seed`
   randomized; all 20 must produce bit-identical output (tests stochastic
   stability of atomic order across runs).
7. **Full regression** (~0.5 d).

---

### 4.4 W_HG — s7 Phase 3 batched Hungarian on GPU

#### 4.4.1 Current state (`s7_rank_assign.py:1485-1512`)

```python
for cube_idx in ok_cube_indices:                  # ~275k iterations
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

    centroids_i = loop_centroids_np[l_lo:l_hi]         # (n_loops, 3)
    comp_pts_i = point_values_np[p_lo:p_hi]             # (n_points, 3)
    diff = centroids_i[:, None, :] - comp_pts_i[None, :, :]
    cost = (diff * diff).sum(axis=-1).astype(np.float64)   # (n_loops, n_points)

    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if r < n_loops:
            all_matches[l_lo + int(r)] = int(c)
```

**Measurement:** 999 ms self (includes all iteration overhead + scipy calls).
**Typical shape:** `n_loops` × `n_points` is small — 1×1 to at most 5×8 per
cube. scipy `linear_sum_assignment` on such tiny inputs has high Python
overhead.

#### 4.4.2 Design options

##### Option 1 (Triton) — custom batched Hungarian kernel

Since inputs are tiny (max 5×8), a **single warp per cube** can solve one
Hungarian via brute-force enumeration for n ≤ 4 and a Jonker-Volgenant for
n = 5. For n ≤ 4 (estimated 80 %+ of cubes), **brute-force over all n!
permutations** is cheaper than any matrix algorithm.

```python
# Batched: pad all cost matrices to (max_loops, max_points) = (5, 8)
# with +∞ for invalid entries
cost_padded: (N_ok, 5, 8) float32
n_loops_per_cube: (N_ok,) int32
n_points_per_cube: (N_ok,) int32

# Triton kernel: one warp per cube
@triton.jit
def hungarian_brute_kernel(cost_ptr, n_loops_ptr, n_points_ptr, match_ptr, ...):
    pid = tl.program_id(0)
    nl = tl.load(n_loops_ptr + pid)
    np_ = tl.load(n_points_ptr + pid)
    if nl == 0: return
    # ... load cost block (5, 8) ...
    # Branch on nl: 1 → trivial argmin; 2 → nested-loop; 3,4,5 → permutation
```

**Complexity:** for n=5 brute-force is 5! = 120 perms × 5 loads = 600 ops.
Across 275k cubes = 165M ops. One H100 warp handles ~10 TFLOPS int ops → sub
millisecond.

##### Option 2 (PyTorch fallback) — batched greedy / auction

Not exact Hungarian but can be bit-compatible with scipy on small inputs if we
use auction algorithm with small ε and deterministic tie-breaking.

**Risk:** auction output != scipy output when there are cost ties. Would break
F1-F3.

##### Option 3 (CPU parallel) — scipy via ProcessPool

Already tried in pre-W2 form. cProfile shows per-cube scipy overhead is
~3.6 µs; 275k × 3.6 = 1 s. Pickling overhead dominates. **Rejected.**

##### Recommended: Option 1 (Triton brute-force)

Risk mitigated by:
- The algorithm is deterministic (exhaustive enumeration over all permutations,
  pick minimum with deterministic tie-breaker = lowest row-major permutation
  index).
- Validation: spike a 1000-cube sample, compare `match_ij` output bit-identical
  to scipy for 100 % of samples.

**Expected speedup:** 999 ms → 50-150 ms (**-0.8 to -0.95 s**).

#### 4.4.3 Determinism

**Hungarian tie-handling:** scipy `linear_sum_assignment` uses Jonker-Volgenant,
which has its own tie-break rule. For F1-F3 bit-exact, we need to **either**:
(a) match scipy's tie-break exactly (doc-opaque, risk), or
(b) ensure our inputs have no ties (assert tie-free then match).

**Empirical check:** scan F1-F3 cost matrices; count how many have cost-ties.
If < 0.1 %, a simple "reject if ties found, fall back to scipy" protects DoD.

If ties are common (> 1 %), Option 1 needs to explicitly replicate scipy's
tie-break. **Spike this first (0.5 d):** dump cost_padded on F2 fixture,
histogram the "min count in each row" metric.

#### 4.4.4 TDD hooks

```
tests/unit/test_hungarian_batched.py
    test_matches_scipy_on_random_small(n=1000)
        # Random (nl, np) ∈ [1..5] × [1..8]; compare assignments bit-identical
    test_brute_force_n_equals_1(n=500)
    test_brute_force_n_equals_2_2(n=500)
    test_brute_force_n_equals_3_5(n=500)
    test_brute_force_n_equals_5_8(n=500)
    test_tie_break_matches_scipy(...)
        # CRITICAL: dump real F2 cost matrices with ties, verify match
    test_batched_1000_cubes_end_to_end_vs_scipy()
```

Plus F1-F3 regression gate.

#### 4.4.5 Risks

1. **Tie-break equivalence (HIGHEST risk):** address by §4.4.3 empirical check
   + explicit tie-break matching scipy.
2. **Padding overhead:** padding 275k cost matrices to (5, 8) wastes compute
   but is still net-win vs Python dispatch.
3. **GPU↔CPU for results:** output `all_matches` is consumed by CPU for final
   `torch.tensor(all_matches, ...)` pack at s7:1525. Keep result on GPU and
   skip the `.cpu()` round-trip entirely — modify the pack site.
4. **Integration with W_BAF:** if W_BAF ships first, `adj` tensor is already
   canonical on GPU; Phase 3 already reads from GPU so no new coupling.

#### 4.4.6 DoD

- [ ] 7 unit tests pass
- [ ] F1-F3 bit-exact
- [ ] s7 Phase 3 self ≤ 200 ms (from 999 ms, -80 %)
- [ ] Residual `lock.acquire` self ≤ 100 ms (was held up by Phase 3 wait; with Stage D gone in W_SD and Phase 3 GPU in W_HG, lock.acquire should bottom out)
- [ ] Clean wall reduction ≥ 0.6 s vs Phase-1-end baseline
- [ ] VRAM peak ≤ baseline + 500 MB

#### 4.4.7 Execution subtasks

1. **Empirical tie scan** (~0.5 d): dump F2 cost matrices, histogram tie-count.
2. **Spike** (~1 d): naive brute-force n ≤ 4 PyTorch version (no Triton). Run
   on 1000-cube F2 sample, compare with scipy.
3. **Red tests** (~0.5 d).
4. **Triton kernel** (~2-3 d): brute-force n ≤ 4 + Jonker-Volgenant for n = 5
   (or brute-force all if ≤ 5).
5. **Integration** (~0.5 d): swap s7:1485 loop with kernel call; update pack
   site to accept GPU tensor.
6. **Full regression** (~0.5 d).

---

## 5. Dependencies & execution order

```
Phase 1 (sequential; W_L2L first because it's shortest)
  W_L2L  (0.5-1 d)  —— independent, no dependencies
  W_SD   (3-5 d)    —— depends on Phase-1 regression suite (same F1-F3 runner)

Phase 2 (sequential; W_BAF before W_HG because W_BAF output feeds Phase 1 of s7)
  W_BAF  (4-7 d)    —— must land before W_HG to avoid retesting Phase 1 twice
  W_HG   (4-7 d)    —— can overlap with W_BAF's integration if two engineers
```

**Inter-phase gate:** after Phase 1 lands, freeze baseline wall + run F1-F3
+ capture post-Phase-1 cProfile top-20. This becomes Phase 2's baseline.

**Parallelization opportunity:** W_L2L and W_SD touch the same file
(`s4_face_point.py`). W_BAF and W_HG touch the same file
(`s7_rank_assign.py`). Serialize within-phase. Cross-phase can parallelize
if Phase 1 W_SD and Phase 2 W_BAF are done by different engineers (files
don't overlap).

**Rollback discipline:** every workstream lands behind a `_cfg.<FLAG>`
feature toggle so reverts are one line. Follow the pattern of
`_cfg.S7_PHASE1_GPU` already in s7_rank_assign.py.

---

## 6. Definition of Done (cumulative)

### 6.1 Phase 1 complete

- [ ] W_L2L DoD met
- [ ] W_SD DoD met
- [ ] F1-F3 bit-exact on HEAD
- [ ] Clean e2e wall @ res=256 ≤ **4.2 s** (from 5.354 s, -22 %)
- [ ] cProfile no new ≥ 200 ms Python hotspot introduced
- [ ] VRAM peak ≤ 6187 MB allocated (baseline 5687 + 500 cap)
- [ ] nsys-captured GPU util recorded (informational, not DoD)

### 6.2 Phase 2 complete

- [ ] W_BAF DoD met
- [ ] W_HG DoD met
- [ ] F1-F3 bit-exact on HEAD
- [ ] Clean e2e wall @ res=256 ≤ **3.2 s** (from 5.354 s, -40 %)
- [ ] cProfile top-3 residual dominated by torch internals, not application code
- [ ] VRAM peak ≤ 6187 MB (+500 cap)
- [ ] nsys GPU util ≥ **30 %** (DoD; estimated current ~10-15 %)

---

## 7. Risks & mitigations (cross-cutting)

| # | Risk | Likelihood | Impact | Mitigation |
|---:|---|:---:|:---:|---|
| R1 | F1-F3 breaks on W_SD tolerance drift | M | H | §4.2.5 risk 1 — use f64 + widened eps |
| R2 | W_BAF Triton atomic nondeterminism | H | H | §4.3.3 Strategy A canonical sort — non-negotiable |
| R3 | W_HG tie-break != scipy | M | H | §4.4.3 empirical tie scan + explicit tie-break |
| R4 | Phase 2 Triton team ramp-up slower than estimated | M | M | Add 2 d buffer; fall back to Strategy C in W_BAF |
| R5 | Downstream consumer of `_labels_to_list_of_lists` expects `list[int]` not `np.ndarray` | L | L | Audit pre-commit (§4.1.5); cheap fix if needed |
| R6 | Persistent pool holds stale state after Stage D removed | L | M | §4.2.5 risk 4 — keep pool, stop Stage D dispatch only |
| R7 | VRAM regression on large fixtures | L | M | §4.1.5 / §4.2.5 / §4.3.5 — all < 50 MB per workstream by design |
| R8 | cProfile instrumentation inflation masks true wall | L | L | §7.1 methodology — always cross-verify with clean wall |

### 7.1 Measurement discipline (inherited from prior spec)

- **Clean wall-time is the only DoD-graded metric.** cProfile inflates 15-25 %
  in this pipeline. Use `tmp/cpu_profile/t0_driver.py --mode default --res
  256 --trials 3` for all DoD numbers.
- **cProfile self-time is Python-only; doesn't count C-ext time.** Use only
  for hotspot *ordering*, never for *magnitude*.
- **Every commit runs F1-F3 on 119 (GPU 0-3 idle) or 116 GPU 4.** Never run
  regressions on local GPU.
- **Three-layer determinism required for all test commits:**
  `PYTHONHASHSEED=0` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
  `torch.backends.cudnn.deterministic=True` + SerialPool monkeypatch +
  subprocess-per-fixture. Pattern lives in
  `corep_fast/tests/regression/_cpu_worker_optim_runner.py` — reuse verbatim.

### 7.2 Rollback protocol

Each workstream ships behind a `_cfg.<FLAG>` boolean. Post-merge, if a
bug surfaces in production use:

```python
# corep_fast/config.py
LABELS_TO_LIST_VECTORIZED = True     # W_L2L
STAGE_D_GPU = True                   # W_SD
BUILD_ADJACENCY_TRITON = True        # W_BAF
HUNGARIAN_GPU = True                 # W_HG
```

Flip → revert path uses legacy. Only remove flags after one release cycle
of stable usage.

---

## 8. Out of scope (do not bundle)

### 8.1 Deferred (possible future spec)

- **`_s6_collapse` assembly block** (513 ms, s6:top-level): mostly numpy
  repack glue code around `_fastpath_trace_loops_gpu` output. Refactoring
  it requires redesigning the `loop_count / loop_offsets / edge_ids` →
  `List[List[int]]` adapter in W4, which is its own micro-spec. Estimated
  ROI 0.1-0.2 s/d, not this spec.

- **numpy coercion cluster** (~759 ms across 276k+ calls): distributed across
  `asarray`, `tolist`, `astype` at CPU↔GPU boundaries. No single point-fix
  buys ≥ 200 ms. Would be reclaimed incidentally by any future "keep s4→s7
  intermediates on GPU" refactor.

- **W4 Option A** (CSR-native downstream for `_fastpath_trace_loops_gpu`):
  100-300 ms. Bundle only if `s6_collapse` assembly block is being
  refactored for another reason.

### 8.2 Rejected (from prior spec)

- "Delete MP entirely" — T0 decisive experiment (serial 8.7× slower)
- W6 Angle 3 (ThreadPool) — T7a GIL-holding 57.9 %
- W1 Bucket A deletions — T2 audit load-bearing
- W7 s7 orchestration cleanup — T8a no mechanical ≥200 ms candidate

---

## 9. Artifacts this spec will produce

### 9.1 Test files (new)

- `corep_fast/tests/unit/test_labels_to_list_vectorized.py` (W_L2L)
- `corep_fast/tests/unit/test_stage_d_gpu_bfs.py` (W_SD)
- `corep_fast/tests/unit/test_build_adjacency_gpu_fused.py` (W_BAF)
- `corep_fast/tests/unit/test_hungarian_batched.py` (W_HG)

### 9.2 Design sketches (new, as preconditions to each W)

- `tmp/corep_followup_design/w_sd_stage_d_spike.md` (W_SD Phase A-D math walkthrough)
- `tmp/corep_followup_design/w_baf_triton_kernel_sketch.md` (W_BAF kernel + determinism plan)
- `tmp/corep_followup_design/w_hg_brute_force_sketch.md` (W_HG brute-force Hungarian layout)
- `tmp/corep_followup_design/w_hg_tie_scan.md` (empirical tie-count on F1-F3)

### 9.3 Findings (new, one per completed W)

- `logs/findings_w_l2l_vectorized.md`
- `logs/findings_w_sd_gpu_bfs.md`
- `logs/findings_w_baf_triton_fusion.md`
- `logs/findings_w_hg_batched_hungarian.md`

### 9.4 Modified source files

- `corep_fast/stages/s4_face_point.py` (W_L2L + W_SD)
- `corep_fast/stages/s7_rank_assign.py` (W_BAF + W_HG)
- `corep_fast/config.py` or equivalent (4 new feature flags)
- Triton kernels: likely inline in `s7_rank_assign.py` or a new
  `corep_fast/stages/s7_triton.py` for W_BAF and W_HG.

### 9.5 Spec + plan + handoff (this flow)

- **Spec:** this file
  (`docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-design.md`)
- **Plan** (next step, via `superpowers:writing-plans` skill):
  `docs/superpowers/plans/2026-04-17-cpu-worker-optim-followup-implementation.md`
- **Handoff** (at completion): will be placed alongside this file with
  `-handoff.md` suffix mirroring the prior spec's convention.

---

## 10. Pickup checklist for the plan author

When transitioning to `superpowers:writing-plans`:

1. Read `my-docs/20260417-cpu-worker-optim-summary.md` §5 (GPU idle analysis)
   and §6 (ROI-sorted strategies).
2. Read this spec end-to-end.
3. Run F1/F2/F3 on anchor HEAD `1ced857` to confirm baseline is still green
   before any planning:
   ```
   ssh host-10-240-99-119 "bash" < tmp/run_f123_on_119.sh
   ```
   (Write the script per SSH protocol; include
   `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/regression/test_cpu_worker_optim.py -v`)
4. Run clean wall baseline:
   ```
   python tmp/cpu_profile/t0_driver.py --mode default --res 256 --trials 3
   ```
   Expected: median 5.35 ± 0.15 s.
5. Plan granularity (per `superpowers:writing-plans` skill):
   - Each task = 1-2 commits (TDD red / stub / real impl / integrate)
   - Bite-sized step: 2-5 min each
   - Full code in every step (no "implement the kernel" placeholders — show
     the Triton body)
   - Expected task count: **~20-28** (4-6 for W_L2L, 6-8 for W_SD, 6-8 for W_BAF,
     6-8 for W_HG; with spike + TDD split each)

6. Spec self-review before leaving this file (already completed inline).

End of spec.
