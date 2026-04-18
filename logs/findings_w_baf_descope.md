# W_BAF descope decision

**HEAD at decision:** `045996b` (Task 11 spike)
**Tasks descoped:** Tasks 12, 13, 14, 15 of cpu-worker-optim-followup-implementation plan

## Why

Spike (`tmp/followup_design/w_baf_triton_kernel_sketch.md`, `tmp/followup_design/w_baf_spike_*.txt`)
falsified the spec §2.1 ROI assumption.

| Metric | Spec §2.1 | Spike actual |
|---|---|---|
| `_build_adjacency_gpu` self_ms | 1913 | **38.26** (50× overestimate) |
| Triton vs legacy on real F2 | 10–20× faster | **0.40× (2.5× SLOWER)** |
| Triton parity (after canonical sort) | bit-exact | 4 mismatches (atomic 2-slot race) |

**Root cause of speedup illusion:** the synthetic random `edge_weights` test inflated
legacy cost by filling every `(t_idx, pi, jj)` slot, forcing `mask.any()` to fire on
every iteration. Real F2 data is sparse — the 99.96% fast-path cubes have many slots
where `valid_arc=False`, letting legacy short-circuit. Triton runs all 576 programs
unconditionally, so on real data it loses to legacy by ~57 ms.

**Race condition:** the spike kernel uses `tl.atomic_add(fill_count, 1)` then
`tl.store(adj[..slot..])`. The order in which programs increment fill_count vs
store into adj is not guaranteed atomic w.r.t. each other across programs, leading
to the `[224, -1]` vs `[-1, -1]` mismatch we observed. Fixing requires `atomic_cas`
loops or a multi-pass reduction — significant additional work for a workstream
whose upper-bound saving is < 38 ms anyway.

## Consequence

- Phase 2 wall budget shifts entirely to W_HG.
- Spec §6.2 cumulative target (3.2s e2e) achievable only at upper end of W_HG
  estimate (-1.5s).
- Realistic post-Phase-2 wall: 3.0–3.7s.

## Artifacts retained

- `corep_fast/stages/s7_triton.py` — spike kernel (not wired to any call site;
  flag `BUILD_ADJACENCY_TRITON` defaults to `'0'` and is unused)
- `tmp/w_baf_spike_*.{py,sh}` — measurement scripts (rerunnable)
- `tmp/followup_design/w_baf_*` — full findings

## Future re-open conditions

W_BAF is worth revisiting only if either of these changes:

1. `_build_adjacency_gpu` self_ms grows beyond ~300ms (e.g., higher resolution
   or denser fixtures change the cost-balance).
2. A different algorithmic refactor of s7 makes the function the only Phase-2
   bottleneck left after W_HG ships.

Until then, leave the spike in place and move on.
