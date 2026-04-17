# CoReP Triton Kernel Handoff Spec

> Date: 2026-04-16
> Source branch: `pre-triton/all` (Phase 2 final state)
> Phase 1 analyses: `tmp/pretriton_s{4,6,7,8}_analysis.md`
> Final results: `my-docs/20260416-pre-triton-final-pass-results.md`

---

## 1. Post-Phase-2 stage breakdown (actual, res=256)

| Stage | Time (s) | % of e2e |
|---|---:|---:|
| s1 voxelize | 0.144 | 1.6% |
| s2 components | 0.083 | 0.9% |
| s3 edge_weights | 0.002 | 0.0% |
| **s4 face_point** | **4.850** | **53.1%** ← biggest remaining bottleneck |
| s6 collapse | 2.445 | 26.8% |
| s7 rank_assign | 1.417 | 15.5% |
| s8 decode | 0.113 | 1.2% |
| **e2e** | **9.126** | 100% |
| (load) | (small) | — |

vs custom 141.05s baseline: **15.46x**.

---

## 2. Triton kernels still needed (priority order)

### K1. s4 Stage D BFS U-turn (highest priority)

**Current cost:** 4.3-4.85s @ res=256 (~89% of s4 stage time)
**Source code:** `corep_fast/stages/s4_face_point.py:1119` (`_compute_face_weights_gpu`), Stage D MP at line 1189-1191
**Phase 1 analysis:** `tmp/pretriton_s4_analysis.md` §B.1 (algorithm) + §E.1 (kernel design)

**Algorithm summary**

Per (cube, facet) group with K segments (typical K≈2-3, max ~30):
1. Build segment graph: deduplicated endpoints + edges
2. BFS connected components
3. Per-component endpoint U-turn count (which facet edges hit by graph leaves)

Currently runs in CPU MP, 100K-150K groups @ res=256. MP overhead (~1-2ms per group) dominates the actual graph work.

**Recommended kernel design (from Phase 1 analysis)**

- **Grid:** `(G,)` where G = number of (cube, facet) groups (~150K)
- **Block:** 32 threads (one warp per group)
- **Shared mem:** ~2.1 KB per block:
  - node_pool[64]: 64 nodes × 12B = 768B
  - adj_list: 256 edges × 4B = 1024B
  - visited bitmap: 64 bits = 256B
  - endpoint_counts[18 edges]: 72B
- **Per-group compute:** ~15K cycles (K≤30, depth=5 BFS)
- **Memory bandwidth:** ~50 MB total reads at res=256
- **Arithmetic intensity:** ~30 FLOPs/byte (low, memory-bound)

**Input tensors (already prepared by GPU Stages A+B in `_compute_face_weights_gpu`)**

```
segments_A:     (S, 3) float32  — line segment endpoints (start)
segments_B:     (S, 3) float32  — line segment endpoints (end)
group_offsets:  (G+1,) int64    — CSR offsets into segments arrays
cube_indices:   (N, 3) int32    — cube voxel coords for endpoint projection
facet_verts:    (12, 3, 3) float32 — constant facet vertex offsets (in cube)
facet_edges:    (12, 3) int32   — constant facet edge indices
resolution:     scalar int
```

**Output tensor**

```
fw_contribution: (G,) int32   — U-turn count per (cube, facet) group
```

Then scatter to `(N, 12) face_weights` via `cube_id = group_offsets_inverse[g] // 12`, `facet_id = ... % 12`.

**Predicted speedup:** 4.3s → 2-3s (Phase 1 estimate, ~1.5-2x) — modest because algorithm is fundamentally bandwidth-bound on tiny graphs. Triton's win is eliminating 100K Python MP launches, not algorithmic.

**Risk / consideration**

- Block occupancy on H100 will be low because each (cube, facet) graph fits in 1 warp; can fix by batching multiple groups per block (32 groups per block → 4-warp scheduling)
- Need fallback for the rare K>16 case (extend shared mem dynamically or split-kernel)

---

### K2. s6 slow-path persistent kernel (lower priority)

**Current cost:** ~0.3-0.5s @ res=256 (10-20% of s6 stage time, all in slow-path cubes with face_weights > 0)
**Source code:** `corep_fast/stages/s6_collapse.py:_collapse_with_uturns` (line 296-377), `_collapse_with_uturns_tracked` (line 379-469)
**Phase 1 analysis:** `tmp/pretriton_s6_analysis.md` §B.2 + §C row 2 + §E

**Algorithm summary**

Per slow-path cube:
1. Enumerate (u1, u2, u3) per facet bounded by face_weights[f] (typical 1-50 valid triples per facet)
2. Cartesian product over 12 facets (budget 100K combinations per cube)
3. For each assignment: trace loops + canonicalize + check U-turn validity
4. Classify outcome (OK / UNSOLVABLE / AMBIGUOUS / BUDGET_EXCEEDED)

**Recommended kernel design**

- **Grid:** `(N_slow_chunks,)` where each chunk = 8 cubes (warp-per-cube)
- **Block:** 256 threads (8 warps × 32)
- **Shared mem:** ~16 KB per block:
  - face_valid_triples: (12, max_triples=50) = 2400B
  - product_buffer: (max_product=1000, 3B/triple) = 3000B
  - loop_stack: 128 entries = 512B
  - atomics for loop_count / best assignment
- **Workflow:** persistent kernel per warp; cycle-iterate cartesian product candidates with early-exit on first valid

**Predicted speedup:** 0.5s → 0.1-0.2s (3-5x). Lower priority because absolute saving is small.

**Risk / consideration**

- Per-cube product_size variance is huge (10 - 100K). Pre-allocation needs per-cube max size that may waste memory.
- Persistent kernel pattern with warp-level work-stealing avoids the worst case.

---

### K3. s8 4-cube residual (NOT NEEDED — already handled)

The Phase 1 spec proposed a Triton kernel for the 0.05% pathological 4-cube edges. But W1's hybrid implementation routes those 0.05% to the existing Python fallback, which is now < 0.1s total. The Triton kernel is not justified.

Keep the existing hybrid as-is; Triton effort is better spent on K1.

---

## 3. Cross-kernel data contracts

K1 and K2 are independent — operate on different stage's data. No shared layout requirements.

Both kernels should accept GPU tensor inputs (already on device after Phase 2's GPU stages) and return GPU tensor outputs (consumed by remaining Python orchestration).

---

## 4. Implementation priority recommendation

| Priority | Kernel | Predicted gain | Effort estimate |
|---|---|---:|---|
| 1 | K1 (s4 BFS) | -2 to -3s @ res=256 | 1-2 weeks |
| 2 | K2 (s6 slow-path persistent) | -0.3s | 1 week |

After K1: predicted res=256 e2e ~6-7s = **20-23x vs custom 141s**.
After K1 + K2: predicted res=256 e2e ~6s = **~23x vs custom**.

**Note:** Phase 2 already exceeded the spec's stretch target (e2e ≤ 10s). Triton work brings the ceiling to ~23x speedup; whether it's worth pursuing depends on downstream demand.

---

## 5. References

- Phase 1 stage analyses: `tmp/pretriton_s{4,6,7,8}_analysis.md`
- Phase 2 spec: `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`
- Phase 2 master plan: `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md`
- Final results: `my-docs/20260416-pre-triton-final-pass-results.md`
- M1 spec (original full-vectorization design): `docs/superpowers/specs/2026-04-16-corep-full-vectorization-design.md`
- Existing geom kernels (input to K1): `corep_fast/geom/{plane_tri_intersect, sh_clip, closest_point}.py`
