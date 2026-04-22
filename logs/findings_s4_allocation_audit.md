# s4_face_point.py allocation audit — 2026-04-21

Audit of tensor allocations in `_count_uturns_from_packed` to understand OOM coverage of the Phase A chunked dispatcher (under construction, throughput-168k).

## Phase-by-phase O(P²) map

All numbers below assume one group (G=1, worst case); scale by G for multi-group.

### Phase A (node coalescence, lines 732-750)

| Line | Tensor | Shape | Dtype | Bytes @ G=1 | Bytes @ G=1, P=2.5M |
|---|---|---|---|---|---|
| 735 | `d = cdist(pts, pts)` | `(G, P, P)` | f64 | **8·G·P²** | **50.0 TiB** ← the peak |
| 736 | `valid_pair` | `(G, P, P)` | bool | G·P² | 6.25 TiB |
| 738 | `d = where(valid_pair, d, 1.0)` | `(G, P, P)` | f64 | 8·G·P² | 50.0 TiB *(inplace-like)* |
| 739 | `match` | `(G, P, P)` | bool | G·P² | 6.25 TiB |
| 741 | `tri_lower` | `(P, P)` | bool | P² | 6.25 TiB |
| 742 | `match_lower` | `(G, P, P)` | bool | G·P² | 6.25 TiB |
| 744 | `j_ar.expand(G, P, P)` | *view* | — | 0 (view) | 0 |
| 745 | `big_P = full_like(j_ar, P)` | `(G, P, P)` | i64 | 8·G·P² | 50.0 TiB |
| 746 | `node_raw` | `(G, P, P)` | i64 | 8·G·P² | 50.0 TiB |

Output: `canonical_idx (G, P) i64` — small (20 MB for G=1, P=2.5M).

**Phase A peak bytes ≈ `5·(8·G·P²)` + bool ≈ ~45·G·P² bytes total temporarily allocated** before the min reduction. With QW1 freeing (now subsumed by chunked dispatcher), could drop to ~16·G·P².

### Phase B (edge_mask construction, lines 753-765)

| Line | Tensor | Shape | Dtype | Bytes @ G=1, P=2.5M |
|---|---|---|---|---|
| 753 | `edge_mask` | `(G, P, P)` | bool | **G·P² = 6.25 TiB** |
| 754 | `g_idx_exp` | `(G, max_s)` | i64 | 8·G·(P/2) = 10 MB |
| 758-759 | `flat`, `flat_sym` | `(G·max_s,)` | i64 | 8·G·(P/2) |

**Phase B peak: `G·P²` bytes (bool edge_mask).** 1/8 of Phase A's `d`. For G=1 P=2.5M = 6.25 TiB.

### Phase C (label propagation, lines 768-786) — **ALSO O(P²) per iteration**

Critical finding: Phase C allocates TWO `(G, P, P) int64` tensors PER ITERATION.

```python
max_iters = P                                                # line 774
for _it in range(max_iters):
    lbl_broadcast = labels.unsqueeze(1).expand(G, P, P)      # VIEW (0 bytes)
    big_lbl = _torch.full_like(lbl_broadcast, P)             # NEW (G, P, P) i64
    nbr_labels = _torch.where(edge_mask, lbl_broadcast, big_lbl)  # NEW (G, P, P) i64
    min_nbr = nbr_labels.min(dim=-1).values                  # (G, P)
    ...
```

| Line | Tensor | Shape | Dtype | Bytes @ G=1, P=2.5M |
|---|---|---|---|---|
| 777 | `lbl_broadcast` *(view)* | `(G, P, P)` | i64 | 0 |
| 778 | `big_lbl = full_like(...)` | `(G, P, P)` | i64 | **8·G·P² = 50 TiB** |
| 779 | `nbr_labels = where(...)` | `(G, P, P)` | i64 | **8·G·P² = 50 TiB** |

**Phase C peak: `2·8·G·P²` bytes per iteration** — LARGER than Phase A. For G=1 P=2.5M = 100 TiB per iteration (though edge_mask must also be live = 6.25 TiB).

Convergence: nominally `P` iterations, practically converges in ~`diameter` iterations (often <20 for well-connected components). But peak memory is per-iteration, not cumulative.

### Phase D (endpoint projection, lines 800-845)

| Line | Tensor | Shape | Dtype | Bytes @ G=1, P=2.5M |
|---|---|---|---|---|
| 806 | `P_minus_A` | `(G, P, 3, 3)` | f64 | 72·G·P = 180 MB |
| 807 | `dot_` | `(G, P, 3)` | f64 | 24·G·P = 60 MB |
| 810 | `proj` | `(G, P, 3, 3)` | f64 | 72·G·P = 180 MB |
| 811 | `dist` | `(G, P, 3)` | f64 | 24·G·P = 60 MB |

**Phase D is O(G·P), trivially safe even for extreme meshes.**

## Implication for the chunked Phase A dispatcher

### Coverage analysis by OOM size

Historical OOM log characteristics (2193 entries, throughput-168k branch, 2026-04-21):

| Band | Count | % | Fix by Phase A chunked alone? |
|---|---|---|---|
| ≤ 80 GiB | 1014 | 46.2 % | Partial — Phase C will OOM if it was the reported failure |
| 80-200 GiB | 364 | 16.6 % | Partial — Phase B = 10-25 GiB fits; Phase C = 80-200 GiB still OOMs |
| 200-500 GiB | 308 | 14.1 % | **No** — Phase C at same size OOMs after A chunked |
| 500-1000 GiB | 220 | 10.0 % | **No** — Phase B also OOMs (60-120 GiB) |
| 1000-5000 GiB | 277 | 12.6 % | **No** — All phases O(P²) OOM |
| > 5000 GiB | 10 | 0.5 % | **No** — need full P² memory rewrite |

**Best case (optimistic) coverage of Phase A chunked alone**: the OOMs that happened specifically at Phase A's `d` allocation, where Phase C would succeed. The PyTorch error message reports whichever allocation tripped first; without instrumentation we don't know the split.

**Worst case: zero net coverage** if Phase C always fails at the same P after Phase A is fixed.

### What's really needed for full coverage

To handle OOMs beyond trivial: also chunk Phase B and Phase C, OR rewrite them to use sparse edge representations:

1. **Phase B**: edge_mask has at most max_s = P/2 True entries. Store as sparse edge list `(src, dst)` of length max_s (bytes: 16·P). 1/(P/16) compression ratio.
2. **Phase C**: operate on sparse edges with `scatter_reduce_('amin')` propagation over a CSR edge list — same pattern as the Task 8 sparse module (corrected for symmetric edges).

Both would require substantial refactoring. Rough scope: +300-500 lines, 2-3 days of work.

## Recommendation for current OOM-fallback track

1. **Ship Phase A chunked dispatcher** (S1 subagent in progress). It covers the subset of OOM cases where Phase A was the specific first-failed allocation.
2. **Measure empirically** via the 4-GPU validation runner (S2, committed). For each target mesh, record exactly which phase fails.
3. **If coverage is inadequate** (e.g., median-90GiB and medium-400GiB meshes still fail with Phase C OOM), add Phase B+C chunking as a separate follow-up.
4. **Extreme 51 TiB mesh** is outside scope — 119k vertices legitimate, not data corruption (checked: 23.5 % coincident vertex clusters suggest 75-part glTF Scene with 36 copies stacked per anchor, but no duplicate faces or degenerate edges). Need full sparse rewrite to handle.

## Action items
- [x] 2026-04-21 T3 run on 117 (killed after 2 meshes completed): both meshes
  OOMed in **s2_components, not s4**. The historical "Tried to allocate X GiB"
  sizes reported in the 168k log are predominantly s2 first-failures, not
  s4 Phase A. s4 Phase A chunked dispatcher works (s4 completed in 13-27 s
  on both meshes) but covers a smaller failure class than assumed.
- [x] 2026-04-21 follow-up: s2 chunked dense dispatch spec kicked off
  (subagent S3, commit pending) — chunks N into batches of N_chunk so the
  dense `(N, M, M) int64 adj_labels` clone inside the label-prop loop
  doesn't blow the 80 GiB HBM. Tier-0/1/2 pattern mirrored from s4.
- [ ] If Phase C OOM dominates, draft a Phase-C sparse-edge follow-up spec.

## T3 2026-04-21 empirical evidence (2-mesh partial run, rest killed)

Stage-wall evidence from per-rank JSON (`scripts/oom_validation/results/*.json`):

| Mesh | s1 | s2 | s3 | **s4** | Status | Requested |
|---|---:|---:|---:|---:|---|---|
| easy-40GiB    | 0.17 s | **164.9 s** | 0.006 s | 12.8 s ✓ | OOM | **40.15 GiB** |
| median-90GiB  | 0.08 s | **300.7 s** | 0.014 s | 27.1 s ✓ | OOM | **90.00 GiB** |

**s4 ran to completion on both meshes.** The OOM occurred inside s2_components
during the 164-300 s window — time spent is consistent with the label-prop
iteration loop being the site: the loop allocates `(N, M, M) int64` adj_labels
on every step until OOM trips. That matches the plan-doc analysis of
`adj_labels = labels.unsqueeze(1).expand(N, M, M).clone()` at s2_components.py
line 141.

The historical 168k log OOM sizes (40 GiB, 90 GiB, 400 GiB, 1800 GiB …) are
NOT s4 Phase A d-tensor sizes — they are s2 adj_labels allocations: for M=64
and N_cubes in the millions,  `8·N·M² = 32768·N bytes`:
- 40 GiB → N ≈ 1.3 M cubes.
- 90 GiB → N ≈ 2.9 M cubes.
- 400 GiB → N ≈ 13 M cubes.
- 1800 GiB → N ≈ 58 M cubes.

A res=512 volume has 134 M cells total, so boundary-cube counts of 1-60 M
are plausible on dense meshes. The 8·N·M² scaling explains the entire
historical OOM band — no exotic Phase-A path needed.

## Revised coverage estimate

With **s4 Phase A chunked dispatcher** only:
- Covers OOMs that specifically trip on s4's `cdist(pts, pts)` allocation.
- These are RARE: the two meshes we sampled both tripped on s2 first.
- Empirical rate: 0 / 2 targeted meshes were s4-limited in the T3 sample.

With **s2 chunked dispatcher** (pending):
- Covers OOMs that trip on s2's `(N, M, M)` adj_labels allocation.
- Expected to cover the vast majority of the 2193 historical OOMs (median
  97 GiB, p99 3511 GiB sizes match N ∈ [3 M, 110 M] cube bands).

With **both together**:
- Expected near-full coverage of the 168k OOM class, modulo the truly
  extreme (>80 GiB) cases where both s2 AND s4 would need to go chunked
  on multiple paths simultaneously.

## Implementation note — `torch.cdist` non-determinism near zero

While implementing the chunked dispatcher the S1 subagent (commit `ea338d0`)
discovered that `torch.cdist(pts, pts)` uses a matmul-based expansion
(`sqrt(a·a + b·b − 2a·b)`) that is numerically non-deterministic at the
1e-8 tolerance boundary: GPU and CPU backends produced different
`canonical_idx` on parity tests, and even on GPU alone `cdist(pts, pts)[i, i]`
occasionally rounded to ~1e-8 instead of 0 — breaking the self-match
identity. All three tiers (dense GPU / chunked GPU / chunked CPU) were
therefore switched to the manual form `sqrt(sum((a − b) ** 2, dim=-1))`,
which yields exactly 0 for self-distance on both backends. Side effect:
Tier 0 now also differs slightly from the historical in-place cdist output
on near-tolerance pairs, but all F1/F2/F3 + stage-AB regression goldens
still pass, confirming no real downstream impact. Trade-off is documented
in the module docstring.

This is worth remembering for any future CUDA port that uses `cdist` in
a coalescence-threshold context: matmul cdist is fine for "sort by
distance" but NOT for "is this less than epsilon" predicates at small
epsilon.

## 51 TiB mesh geometry check (2026-04-21)

Inspected the worst-case OOM mesh
(`raw/hf-objaverse-v1/glbs/000-134/b74a3e5b9e6c4547b2692299fdd740c7.glb`,
sha256 prefix `0119449726…`, historical OOM 51316 GiB):

- Loaded as a `trimesh.Scene` with **75 sub-meshes**; flattening via
  `force='mesh'` yields V = 119 290, F = 175 152.
- **Face-index dedup**: 175 150 unique tuples, 2 exact dups — geometry
  is NOT duplicated at the triangle level.
- **Vertex-position dedup @ 1e-6**: 91 198 unique positions, 28 092
  coincident (23.5 %). Top-5 coincident positions each have exactly
  36 vertices stacked — strongly suggesting a rigged/anim'd character
  with 36 "clones" anchored at shared transform points.
- **Degenerate faces**: 0 (min-edge < 1e-8). No zero-area faces.
- Face area p50 = 1e-3, p99 = 26, max = 167 (scene units) — wide range
  but all non-trivial.

Conclusion: the 51 TiB mesh is LEGITIMATE geometry, not a data-corruption
or glTF-instancing artifact. Its pathological P_MAX comes from many
sub-meshes intersecting the same voxel face after flattening. A
vertex-dedup pre-filter might shrink the problem but cannot eliminate it
— the 36-copies-per-anchor pattern means most voxel-face intersections
would still have ~36× the segment count of a normal mesh. Handling this
asset well ultimately requires the sparse-edge Phase B/C rewrite, not
just dedup.

## Cross-refs
- Phase A chunked impl: `corep_fast/stages/s4_phase_a_chunked.py` (commits `ea338d0` + `28d2103`).
- 4-GPU runner: `scripts/oom_validation/` (commit `860960a`).
- OOM log: `datasets/ObjaverseXL_sketchfab/logs/precompute_feat18_r512_g8/rank*.log`.
- Previous related findings: `logs/findings_qw5_regression.md`.
