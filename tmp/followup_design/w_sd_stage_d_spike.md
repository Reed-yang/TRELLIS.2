# W_SD spike — Stage D group-size / node-count histogram (F2)

## Raw output

```
groups = 1881777
segs/group: min=1 max=5 mean=1.06 p50=1 p95=2 p99=2
nodes/group: min=2 max=7 mean=2.06 p50=2 p95=3 p99=3
P_MAX (= 2 * max_segs) = 10
(G, P, P) float64 bytes = 1,505,421,600 bytes = 1.40 GB
```

Fixture: F2 (icosphere subdivisions=3, radius=0.4, res=256) via
`tmp/w_sd_spike_116.py` with SerialPool monkeypatch and
`_count_uturns` instrumentation.

## Decision criteria for Task 8 (batched GPU impl)

| Metric | Value | Implication |
|---|---:|---|
| G (total groups) | 1,881,777 | pass count — one single batched pass feasible |
| max segs/group | 5 | P_MAX = 2*5 = 10 |
| p99 segs | 2 | vast majority (>99%) have ≤ 2 segments |
| mean segs | 1.06 | extremely sparse — most groups have 1 segment |
| max nodes | 7 | comfortably ≤ P_MAX=10 |
| mean nodes | 2.06 | consistent with mean segs*2 ~ 2.12 (tiny dedup) |
| (G, P, P) f64 memory | 1.40 GB | fits easily in 80GB H100 |

**Decision:** P_MAX = 10 ≤ 64 **and** (G, P, P) = 1.40 GB ≤ 4 GB → use
**single batched cdist** path, no chunking required.

Memory note: (G, P, P) with f32 would halve to 0.70 GB, and with dtype=bool
adjacency it would be ~188 MB. Plenty of headroom even accounting for
intermediate tensors + segment coordinates (G, P_MAX, 3) float32 ~ 226 MB.

## Task 8 path: single batched cdist, no chunking

Algorithm sketch:
1. Pack endpoints `(G, P_MAX, 3)` float32, with mask `(G, P_MAX)` bool.
2. `cdist` → `(G, P_MAX, P_MAX)` float32; threshold at 1e-8 → adjacency.
3. BFS via iterated boolean matmul (converge in ≤ log2(P_MAX) ≈ 4 steps).
4. U-turn count per group = (num_connected_components expressed via
   legacy's edge-based formula — re-derive from s4._count_uturns spec).

Because p99 segs = 2, most groups finish in 1 BFS iteration, so a
handful of matmul rounds covers the rare max=5 case.

Safety: if a future fixture produces max_segs > 64, the guard in Task 8
should fall back to legacy CPU path and emit a warning.
