# T6a — s4 Part 2 local UF GPU design sketch

Author: T6a spike (read-only design task)
Date: 2026-04-17
Target: replace `_get_local_components_np` in `corep_fast/stages/s4_face_point.py`
to cut main-thread self-time (T0: 1030 ms at res=256, 275k calls) by ≥80 %
and call count by ≥95 %.

---

## Current implementation

- **Location:** `corep_fast/stages/s4_face_point.py:734`
- **Signature:**
  ```python
  def _get_local_components_np(
      face_ids: np.ndarray,   # (n_i,) int32, face ids belonging to one cube
      mesh_faces: np.ndarray, # (F, 3) int32 — UNUSED (dead arg, kept for back-compat)
      face_adj: np.ndarray,   # (F, 3) int32 — global face->3-neighbor table
  ) -> list[list[int]]:
  ```
- **Call site:** `s4_face_point.py:501` inside a Python `for ci in range(N)` loop
  over *every* cube (after copying `comp_face_val`, `mesh_faces`, `face_adj` to
  CPU numpy). `N ≈ 275k` at res=256, hence the 275k calls.
- **Per-cube cost (numpy):** 3-4 µs for typical n_i ∈ [2, 20]; measured aggregate
  1030 ms self-time. The `mesh_faces` argument is never read (design lint).
- **Algorithm:** path-compressed iterative union-find with no rank. For every
  face id `fid`, inspect its 3 global neighbors in `face_adj[fid]`; if a neighbor
  also lives in this cube (membership via `face_set`), union the two indices.
  Finally bucket indices by `find(idx)` root.

### Representative tiebreaker (IMPORTANT)

The numpy code uses two implicit tiebreakers that the GPU replacement MUST
reproduce bit-exactly (T0 flagged F1/F2/F3 golden byte-diffs will trip on any
deviation — component order drives the identity of every `point_values` row
downstream, which then feeds s6 collapse and the final mesh).

1. **Root choice inside a component.** `union(a, b)` always does
   `parent[ra] = rb`, i.e. the *second* root absorbs the first. Because of path
   compression the final root is a deterministic function of the union order,
   NOT the `min(idx)`. However, because **s2 already delivers `face_ids` sorted
   by component label** (stable sort, labels = slot index, so the first slot of
   each component is its minimum), the union pattern inside any one component
   is "same-label range merging" and the final root ends up at the *last*
   visited `idx` of that component — equivalently the `max(idx)` within the
   component. This is **not** exposed externally because the root id is
   discarded after grouping; what leaks out is only the *group membership*.
2. **Group enumeration order in the returned `list[list[int]]`.**
   `groups.setdefault(root, []).append(...)` — the k-th returned component is
   the one whose root was *first hit* by the `for idx, fid in enumerate(face_ids)`
   walk. Combined with s2's pre-sorted layout, this means the k-th returned
   component is the one whose minimum-slot face appears earliest in `face_ids`.
   Since s2 sorted by `(component label, slot within cube)`, **the GPU
   replacement must emit components in the same order they already appear in
   `face_ids`**, i.e. the first-occurrence order of their s2 labels.
3. **Face order inside a component.** Each component's `list[int]` is populated
   by iterating `enumerate(face_ids)` in input order. Output face order ==
   `face_ids` slice order restricted to that component. Equivalent to "stable
   filter by component id", easy to reproduce on GPU.

**Concrete consequence for GPU rewrite:** "root = min(idx)" and "component
order = first-occurrence of root" satisfy tiebreaker (2) and (3) exactly because
s2 already layouts face_ids so that min-idx within a component coincides with
the component's earliest element. (1) is invisible to callers, so the GPU
version is free to choose *any* deterministic representative — **min idx
recommended** for monotonic simplicity.

---

## Input/output contract

### Inputs at the s4 boundary (before the per-cube loop)

From `batch` (all on device except `num_components_np`):

- `comp_face_val : (T_total,) int32` — global face ids, already grouped by
  (cube, component) contiguously. Produced by `s2_components._label_propagation`
  → stable sort by `(cube, label)`.
- `comp_face_off : (N+1,) int64` — CSR offsets: cube `ci` owns
  `comp_face_val[comp_face_off[ci] : comp_face_off[ci+1]]`.
- `num_components : (N,) int32` — expected component count per cube (from s2).
- `mesh.face_adj : (F, 3) int32` — global face neighbor table (-1 pad).

**Critical observation:** `comp_face_val` is already component-grouped by s2.
The s4 UF is therefore *redundant in the happy path* — it re-discovers the
boundaries s2 already computed. The only thing missing is a per-cube
`component_offsets` CSR.

Two fixes available:

- **Option A (cheapest, aligned with constraint "only add new files"):** don't
  re-run UF at all; instead extend s2 to emit component-boundary CSR and wire
  it into s4. *Rejected for T6 scope* because it requires modifying s2, and the
  user rule is "don't modify existing repo components, only add new files".
- **Option B (chosen):** keep the semantic contract identical — a standalone
  `_get_local_components_gpu(...)` that reconstructs components from scratch
  using `face_adj` and the pre-sorted `face_ids`. This is what we design below.

### Function-level contract (batched GPU replacement)

```
_get_local_components_gpu(
    comp_face_val   : (T_total,) int32  on device,
    comp_face_off   : (N+1,) int64      on device,
    num_components  : (N,) int32        on device,
    face_adj        : (F, 3) int32      on device,
) -> BatchedComponents {
    # CSR over (cube, component), flat over global (cube, component) pairs
    comp_off   : (P+1,) int64           # P = total_points = sum(num_components)
    comp_cube  : (P,) int64             # cube index each component belongs to
    comp_faces : (T_total,) int32       # face ids, component-grouped
    #   face_ids for component k = comp_faces[comp_off[k]:comp_off[k+1]]
}
```

Semantics:

- Component order inside cube `ci` matches the **first-occurrence order of
  s2-labels within `face_ids[comp_face_off[ci]:comp_face_off[ci+1]]`** —
  reproduced automatically by any algorithm that (a) labels each face with its
  component's min-input-idx and (b) stable-sorts by that label.
- Face order inside a component matches input order (stable filter).
- If the GPU version discovers fewer components than `num_components[ci]`,
  we pad with empty components at the end (same as current numpy loop, see
  `s4_face_point.py:508`). If it discovers more, we truncate to
  `num_components[ci]` — **but in practice it will always equal** because s2
  is the source of truth.

### Per-cube size distribution (from T0 profile + field knowledge)

- `N ≈ 275k cubes at res=256`, `275k` UF calls. Because `num_components ≥ 1` for
  every active cube, every cube has at least 1 entry.
- Typical `n_i = |face_ids in cube|`: **2–20** (thin mesh shell intersecting a
  voxel yields mostly a couple of triangles). Icosphere_s3 bench: empirical
  median 4, p95 ≈ 12, p99 ≈ 25. Triple-icosphere (F3 fixture): max can spike
  to ~60 where three shells meet a single cube.
- Sum of `n_i` across all cubes = `T_total` ≈ **1.5–3 M** at res=256.
- Component count per cube: mostly 1, occasionally 2–3 (thin shells, self-intersecting
  models). Worst case `num_components[ci]` ≤ `n_i`.

Memory footprint for padded (N, max_faces) layouts at res=256:

| max_faces | bytes per (N,M) int32 tensor | total for 4 aux tensors |
|-----------|------------------------------|-------------------------|
| 16        | 17.6 MB                      | ~70 MB                  |
| 32        | 35.2 MB                      | ~140 MB                 |
| 64        | 70.4 MB                      | ~280 MB                 |

→ Pad-to-max is viable up to max_faces ≈ 64 with a few hundred MB headroom.
Fallback needed above that.

---

## Proposed GPU algorithm

### Primary: batched label propagation (re-use s2's approach)

s2 already implements exactly this pattern in `s2_components._label_propagation`
(lines 112-144): build a per-cube adjacency matrix `(N, M, M)` via broadcasted
equality between `neighbors_of[i,j,e]` and `padded_faces[i,k]`, then iterate
`label[i,j] = min(label[i,j], min_k adj[i,j,k] * label[i,k])` until fixed point.

For s4 we reuse the same template but operate on the `(cube, component)`
sub-scope: each of the **P ≈ 275k–550k** "virtual cubes" is a single component
cluster from s2, typically size 1–20.

```python
def _get_local_components_gpu(
    comp_face_val: torch.Tensor,   # (T_total,) int32, sorted by (cube, s2_label)
    comp_face_off: torch.Tensor,   # (N+1,) int64
    num_components: torch.Tensor,  # (N,) int32
    face_adj: torch.Tensor,        # (F, 3) int32
) -> BatchedComponents:

    N = comp_face_off.numel() - 1
    device = comp_face_val.device

    # --- Step 1: build (N, M) padded view of face_ids per cube --------------
    counts = (comp_face_off[1:] - comp_face_off[:-1]).to(torch.int64)   # (N,)
    max_faces = int(counts.max())
    if max_faces == 0:
        return _empty_components(N, device)

    # padded_faces[i, j] = comp_face_val[comp_face_off[i] + j]  if j < counts[i] else -1
    col_idx = torch.arange(max_faces, device=device).unsqueeze(0)
    mask = col_idx < counts.unsqueeze(1)                                # (N, M) bool
    flat_idx = comp_face_off[:-1].unsqueeze(1) + col_idx                # (N, M)
    safe_flat = flat_idx.clamp(max=comp_face_val.numel() - 1)
    padded_faces = torch.where(mask, comp_face_val[safe_flat], -1)      # (N, M)

    # --- Step 2: look up each face's 3 global neighbors ---------------------
    valid_faces = padded_faces.clamp(min=0).long()                      # safe index
    neighbors_of = face_adj[valid_faces]                                # (N, M, 3)

    # --- Step 3: build per-cube (M, M) adjacency via broadcast eq ----------
    # adj[i, j, k] = True iff faces at slots j and k of cube i share a mesh edge
    if max_faces <= 64:
        # direct O(N * M^2) comparison — same path s2 uses
        nbr_exp   = neighbors_of.unsqueeze(-1)                          # (N, M, 3, 1)
        faces_exp = padded_faces.unsqueeze(1).unsqueeze(2)              # (N, 1, 1, M)
        match     = (nbr_exp == faces_exp) & mask.unsqueeze(1).unsqueeze(2)
        adj       = match.any(dim=2)                                    # (N, M, M)
        adj       = adj | adj.transpose(1, 2)
        adj       = adj & mask.unsqueeze(2) & mask.unsqueeze(1)

        # --- Step 4: iterate min-label propagation ------------------------
        # Seed: label[i, j] = j  (== slot index == idx into face_ids slice)
        labels = col_idx.expand(N, max_faces).clone()                   # (N, M)
        labels[~mask] = max_faces                                       # sentinel

        for _ in range(min(max_faces, 32)):
            old = labels
            # min neighbor label via where+min
            lbl_bcast = labels.unsqueeze(1).expand(N, max_faces, max_faces)
            lbl_bcast = torch.where(adj, lbl_bcast,
                                    torch.full_like(lbl_bcast, max_faces))
            min_nbr = lbl_bcast.min(dim=2).values                       # (N, M)
            labels = torch.minimum(labels, min_nbr)
            labels[~mask] = max_faces
            if torch.equal(labels, old):
                break
    else:
        # fallback: per-cube sequential UF (reuse s2's _label_propagation_sequential
        # shape — but only for the spill-over cubes, likely a tiny fraction).
        labels = _label_propagation_sequential_s4(
            padded_faces, neighbors_of, mask, N, max_faces, device,
        )

    # --- Step 5: stable-sort by label, emit CSR -----------------------------
    # Because we seeded labels with slot index (j), and union-by-min produces
    # labels equal to min slot in the component, the sort order gives us:
    #   group order = ascending min-slot = s2's first-occurrence order ✓
    #   face order within group = ascending slot = input order ✓
    labels_for_sort = labels.clone()
    labels_for_sort[~mask] = max_faces  # padding last
    _, order = labels_for_sort.sort(dim=1, stable=True)                 # (N, M)
    sorted_faces = padded_faces.gather(1, order)                        # (N, M)
    sorted_lbl   = labels_for_sort.gather(1, order)                     # (N, M)

    # Component boundaries inside each cube: where sorted_lbl[:, j] != [:, j-1]
    diff = torch.zeros_like(sorted_lbl, dtype=torch.bool)
    diff[:, 0] = mask.gather(1, order)[:, 0]
    if max_faces > 1:
        rhs = sorted_lbl[:, 1:] != sorted_lbl[:, :-1]
        rhs_mask = mask.gather(1, order)[:, 1:]
        rhs_lbl_valid = sorted_lbl[:, 1:] < max_faces
        diff[:, 1:] = rhs & rhs_mask & rhs_lbl_valid

    per_cube_ncomp = diff.sum(dim=1).to(torch.int64)                    # (N,)

    # Flatten into global CSR over P = sum(per_cube_ncomp)
    #   comp_off[k]   = offset of k-th (cube, component) into comp_faces
    #   comp_cube[k]  = which cube owns k
    #   comp_faces    = concatenated face ids (all valid slots of sorted_faces)
    valid_sorted_mask = mask.gather(1, order)                           # (N, M)
    comp_faces = sorted_faces[valid_sorted_mask].to(torch.int32)        # (T_total,)

    # Convert diff + mask into per-component face counts via run-length
    # encoding (scatter_add over cumulative diff prefix).
    # ... (standard RLE pattern: cumsum(diff) - 1 gives local comp id;
    #      scatter to (cube_id * max_components_global + local_comp_id); etc.)
    comp_off, comp_cube = _rle_to_csr(diff, mask, per_cube_ncomp)

    # Truncate to num_components[ci] if the GPU found more (pad if fewer).
    comp_off, comp_cube, comp_faces = _reconcile_with_num_components(
        comp_off, comp_cube, comp_faces, num_components,
    )
    return BatchedComponents(comp_off, comp_cube, comp_faces)
```

**Convergence analysis.** Label propagation converges in *diameter* iterations
of the cube-local adjacency graph. For n_i ≤ 20 and components being small
connected subgraphs of face_adj, diameter ≤ n_i ≤ 20. We cap at
`min(max_faces, 32)` which is conservative. s2 already ships this loop in
production, so no new risk.

**Expected speedup.** s2 at res=256 runs in ~30-60 ms on GPU (from T0 log).
s4's UF Part is **the same problem** on **slightly fewer** inputs (only active
component cubes, not all occupied cubes). Projection: 30-80 ms on GPU vs.
1030 ms on CPU → **>10× speedup**, ≥97 % self-time reduction (blows past 80 %
target). Call count goes from 275k → **1** single GPU function → 100 %
reduction (blows past 95 % target).

### Alternative: block-parallel UF (one thread-block per cube)

Custom CUDA / Triton kernel: one block = one cube, shared-memory UF over
`n_i ≤ 64` slots. Pros: asymptotically fastest. Cons:

- Requires writing CUDA / Triton (rules out: "only add new files" plus
  project-wide "no Triton in the corep_fast fast-path" rule from the V2 spec).
- Hard to match numpy tiebreakers bit-exactly across warps.
- s2's existing pure-PyTorch label-prop is already empirically fast enough for
  this workload (order of tens of ms on 275k cubes).

**Decision: reject alternative.** Primary path is sufficient.

### Output layout

**Chosen: CSR over (cube, component).**

- `comp_off : (P+1,) int64` — contiguous face-id ranges.
- `comp_cube : (P,) int64` — cube index per component.
- `comp_faces : (T_total,) int32` — flat face ids, component-grouped.

This matches what s4's downstream code (`flat_face_ids`, `flat_comp_ids`,
`comp_tri_offsets` at lines 523-532) already builds by hand from
`list[np.ndarray]`. The GPU return **replaces those three numpy arrays
directly**, avoiding another CPU round-trip.

Dense `(N, max_components, max_faces)` is rejected: wastes memory and requires
CPU `.tolist()` downstream (which is what we want to kill).

---

## Tiebreaker audit (critical for determinism)

| Tiebreaker                          | numpy version                              | GPU label-prop version                                 | Match? |
|-------------------------------------|--------------------------------------------|--------------------------------------------------------|--------|
| Representative / root id            | Last-visited idx in union chain (arbitrary)| min slot (== min input idx)                             | Different internally, but root id is never exposed externally ✓ |
| Inter-component order in output     | First occurrence of root during iter      | ascending min-slot (= s2's first-occurrence order)     | **Equivalent** because s2 pre-sorted by min-slot ✓     |
| Intra-component face order          | `enumerate(face_ids)` input order         | `stable_sort` preserves slot order (= input order)     | **Identical** ✓                                         |
| Pad behavior (nc mismatch)          | `components[:nc]`, pad empty at end       | `_reconcile_with_num_components` same policy           | **Identical** by construction ✓                         |
| Cube iteration order                | `for ci in range(N)`                      | tensor row order                                        | **Identical** ✓                                         |

**Determinism verification plan (T6b):**

- Parametric test: generate 1000 random small graphs (n ∈ [1, 20]), compute
  components via numpy and via the GPU impl, assert `output == output`
  element-wise including order.
- Inject the F1/F2/F3 fixtures and compare `point_values` tensor bit-exactly
  against baseline pickle.

---

## Risk analysis

### Determinism risks

1. **Nondeterministic GPU scatter.** None in the design; only `sort(stable=True)`,
   pure elementwise ops, and `min()`. All deterministic on CUDA.
2. **`torch.equal` early-stop break nondeterminism.** If we early-break after
   convergence, the number of iterations depends on data but output is the
   fixed point regardless — deterministic.
3. **`face_adj[valid_faces]` with clamp-to-0 on padding slots.** The dummy
   lookup result is masked out by `mask` before entering adj. Safe — but code
   review must confirm the mask is applied *before* any `.any()` that could
   otherwise pick up a stale match. Same pattern s2 uses; known-good.

### Convergence risk

- Worst-case iteration count = longest component diameter in any cube.
- Empirical bound for typical meshes: < 15.
- Cap at `min(max_faces, 32)` is 1.5-2× safety.
- Pathological meshes (tight serpentine intersections inside a single cube):
  could exceed 32. Fallback = reuse `s2_components._label_propagation_sequential`
  semantics on a per-cube basis (see `max_faces > 64` branch above).

### Memory

- At res=256, `N ≤ 275k`, `max_faces ≈ 20-25` typically. 64-cap gives worst-case
  ~280 MB of aux tensors — tolerable on A100/H100 class GPUs that already hold
  the full mesh (GBs of tri-values).
- Spike risk if a single cube has n_i = 200+ (highly degenerate): `max_faces`
  tensors blow up. Mitigation: clamp `max_faces = min(N.max(), 64)` and route
  outliers through sequential fallback.

### Integration risk

- s4 currently consumes `components : list[list[int]]` and feeds them into
  `_snap_centroids_to_components` (line 618). That function also takes a list.
  **T6d must either** (a) rewrite `_snap_centroids_to_components` to take CSR
  (preferred — kills another CPU round-trip), or (b) `.tolist()` the CSR back
  (negates most of the win).
- The spec rule "only add new files" is tight here: `_get_local_components_np`
  and `_snap_centroids_to_components` already exist. Proposal: place the GPU
  replacement alongside as `corep_fast/stages/s4_face_point_gpu.py` exporting
  `_get_local_components_gpu` and `_snap_centroids_to_components_csr`, and
  **gate** the rewrite behind a flag (`USE_GPU_LOCAL_UF = True/False`) so the
  regression gate can A/B it.

---

## T6b–T6d plan

### T6b (TDD scaffold) — new file, no logic

Files to add:

1. `corep_fast/stages/s4_face_point_gpu.py` — stub exporting
   `_get_local_components_gpu(...)` raising `NotImplementedError`, plus a
   dataclass `BatchedComponents`.
2. `corep_fast/tests/unit/test_s4_local_uf_gpu.py` — new file with:
   - Randomised property test (1000 cubes × n ∈ [1, 20]) comparing
     `_get_local_components_gpu` against `_get_local_components_np` for
     bit-exact equality (group order + face order).
   - Hand-crafted edge cases: empty cube, single-face cube, fully-connected
     component, 3 disjoint components in one cube, cube with degenerate
     face_adj (-1 neighbors).
   - Device marker: run on CUDA if available, fall back to skip.

Exit criterion: tests red, stub raising.

### T6c (numpy-delegating stub) — make tests pass

Fill `_get_local_components_gpu` with a shim that:

- Loops over cubes on CPU calling `_get_local_components_np` exactly as today.
- Re-packs results into `BatchedComponents` CSR format.
- **Purpose:** locks in the output contract and proves the CSR packer is
  tiebreaker-correct. No speedup yet.

Exit criterion: all T6b tests green, regression gate (F1/F2/F3) green after
integrating the CSR return into `build_face_points_gpu`.

### T6d (GPU implementation + integration)

1. Replace the CPU loop body in `_get_local_components_gpu` with the batched
   label-prop algorithm described above. Keep the fallback branch for
   `max_faces > 64` routing to sequential.
2. Replace `_snap_centroids_to_components` with CSR-native
   `_snap_centroids_to_components_csr` that consumes `BatchedComponents`
   directly, avoiding `.tolist()` / `.cpu().numpy()` round-trips.
3. Flip the gate in `s4_face_point.build_face_points_gpu` to call the GPU
   path unconditionally (no feature flag in shipping code).
4. Profile main-thread cProfile again; check self-time for s4 Part 2 drops to
   <50 ms (T6 target = 80 % of 1030 ms ⇒ ≤ 206 ms; we expect ≤ 50 ms).
5. Regression gate F1/F2/F3 must pass byte-exact.

Exit criterion:
- cProfile: `_get_local_components_*` absent from top-20 hotspots.
- Regression gate: green.
- End-to-end wall-time: measurable improvement at res=256 (expected >500 ms
  off main thread).

---

## Open questions

1. **`mesh_faces` arg is dead.** The current numpy function takes `mesh_faces`
   but never reads it. Is this an early-design leftover, or does downstream
   code rely on the signature? Grep says no — can the GPU version drop it?
   *Recommendation: drop in T6b, adjust call site in T6c.*
2. **Should T6 also eliminate the `.cpu().numpy()` triple at s4_face_point.py
   lines 480-482** (`comp_face_val`, `mesh_faces`, `face_adj` → CPU)? These
   copies are justified only by the CPU UF and will become dead weight as
   soon as T6d lands. *Recommendation: remove in T6d alongside the UF
   replacement, not in T6b/c.*
3. **Fallback threshold.** Is `max_faces = 64` the right cutover to the
   sequential path? Needs a quick profile run on F3 (triple icosphere) to
   confirm (a) how often a cube exceeds it and (b) whether the `(N, M, M)`
   adjacency tensor at M = 64 actually fits (64² × 275k = 1.1 GB in int8,
   8.8 GB in bool stored as byte → may OOM on 40 GB cards with the rest of
   the pipeline live). *Recommendation: profile in T6d with
   `max_faces ∈ {32, 48, 64}` and pick the smallest that keeps the fast path
   hot on F1/F2/F3.*
4. **Can we skip UF entirely?** As noted under "Input/output contract", s2
   already knows the component boundaries. The cleanest fix is to have s2
   emit a `component_offsets` CSR that s4 reads directly, making s4's UF a
   no-op. This violates "don't modify existing repo components" but deletes
   the hotspot instead of accelerating it. *Recommendation: park as V3
   follow-up; T6 stays within V2 scope and rebuilds on GPU.*
5. **Integration with a parallel T5a (s6 GPU tracer).** Both tasks operate
   on different stages and different files. No file overlap expected but
   both touch `CubeBatch`-style APIs. Coordinate merge order: ship T6 first
   (smaller blast radius), then T5a.
