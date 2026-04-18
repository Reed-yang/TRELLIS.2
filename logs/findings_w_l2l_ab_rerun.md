# W_L2L A/B interleaved re-test — original "regression" was environmental noise

**Date:** 2026-04-18
**Node:** host-10-240-99-116 GPU 5 (quiet; load_before=14.59, climbed to 41 mid-run)
**Method:** 3 rounds × (A=flag0, B=flag1) × 3 trials each = 9 trials per mode, fully interleaved
**Artifacts:** `tmp/followup_baseline/ab_test/round{1,2,3}_{A,B}.{json,log}`, `tmp/followup_baseline/ab_test_output.log`

## Raw trial walls (seconds)

| Round | A (flag=0 legacy) | B (flag=1 vectorized) |
|---|---|---|
| 1 | 5.570, 5.345, 5.455 | 5.491, 5.541, 5.258 |
| 2 | 5.515, 5.476, 5.371 | 5.385, 5.483, 5.384 |
| 3 | 5.659, 5.889, 5.482 | 5.631, 5.448, 5.507 |

## Aggregate

| Mode | N | Mean | Median | Stdev | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| A legacy | 9 | 5.529 | 5.482 | 0.165 | 5.345 | 5.889 |
| B vectorized | 9 | 5.459 | 5.483 | 0.107 | 5.258 | 5.631 |

**Δ(B - A):**
- Mean: **-0.070 s** (B slightly faster)
- Median: **+0.0003 s** (essentially tied)
- Welch t-statistic: **-1.075**, |t|<2 → 95% CI overlap → no statistically significant difference

## Resolution of the earlier "+0.362 s regression" finding

The initial non-interleaved baseline (5.337 s median on `d0f6bf5`) was captured at
a different moment from the post-W_L2L vectorized run (5.699 s median on
`9ffb76f`). Between those two captures, CPU contention on 116 rose (yushen + wekanode
workloads began). The ~0.36 s differential reflected the **environment shift**, not
an algorithmic difference. Interleaved A/B here cancels environmental drift and
shows the paths are equivalent on wall.

## Decision: keep flag=0 (legacy)

Rationale:
- Walls are indistinguishable (-0.07 s mean, +0.0003 s median, t=−1.075).
- cProfile observability is cleaner with legacy: `_labels_to_list_of_lists`
  owns its 484 ms self — one hotspot to reason about. Under vectorized, the
  same work scatters across `np.split`, `tolist`, `<listcomp>` in the top-20,
  obscuring what to optimize next.
- Vectorized code stays in the repo behind the flag for future retry (e.g. if
  a downstream consumer switches to CSR-native and the tolist overhead
  disappears, flag=1 might tilt positive).

No flag change required (already at `'0'` from commit `c621580`).
