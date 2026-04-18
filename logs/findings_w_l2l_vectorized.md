# W_L2L — _labels_to_list_of_lists vectorize findings

**HEAD:** cb69001 (w_l2l(p1): vectorize _labels_to_list_of_lists bucket loop)
**Baseline HEAD:** a837eda (pre-W_L2L)
**Test host:** host-10-240-99-116 GPU 4

## DoD

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| 1 | 4 unit tests pass | 4/4 | 4/4 | PASS |
| 2 | F1-F3 bit-exact | 3/3 | 3/3 | PASS |
| 3 | Clean wall Δ | ≥ -0.15 s | +0.362 s (regression) | FAIL |
| 4 | `_labels_to_list_of_lists` self | ≤ 150 ms | 121.8 ms | PASS |

## Raw

- Wall JSON: `tmp/cpu_profile/w_l2l_post_wall.json`
- Hotspots: `tmp/cpu_profile/w_l2l_post_hotspots.txt`
- Wall trials: [5.699, 5.608, 5.700]; median = 5.699 s
- Wall delta vs baseline 5.337s: **+0.362 s** (regression)
- `_labels_to_list_of_lists` self_ms: 484.0 → 121.8 ms (**-362 ms self, -76.2%**)

## Analysis: why self dropped but wall regressed

The vectorized path reduced `_labels_to_list_of_lists` self by 362 ms as expected, BUT
profile-induced overhead from the new numpy operations pushed other helpers into the
top-20 (they were masked before by the dominant 484-ms loop):

Post-change top hotspots adjacent to our change (from
`tmp/cpu_profile/w_l2l_post_hotspots.txt`):
- rank 5: `<listcomp>` @ s4_face_point.py:1059 — 311.2 ms self / 342.0 cum
  (this is the `[fids_per_comp[cursor + j].tolist() for j in range(c_i)]` loop)
- rank 6: numpy `tolist()` — 310.7 ms self / 551,870 calls
  (the per-component tolist() calls inside that listcomp)
- rank 12: `array_split` — 148.1 ms self / 340.9 cum
  (the `np.split(valid_fids, split_at)` call)
- rank 15: `_labels_to_list_of_lists` — 121.8 ms self / 826.3 cum

Cum of `_labels_to_list_of_lists` is now 826.3 ms. Previous cum was ≈484+overhead ≈520 ms.
Net impact to s4 cum: +300 ms regression despite target self savings.

## Wall timing

Baseline wall median: 5.337 s (from logs/findings_t0_baseline_concerns.md region)
Post-W_L2L wall median: 5.699 s
Delta: **+0.362 s** (regression)

Note: wall variance between trials is low (5.608, 5.699, 5.700).

## Per-plan guidance

Per plan:
> "If DoD item 4 fails (e.g., new self is 200-300 ms not ≤150), report
>  DONE_WITH_CONCERNS with actual numbers; do NOT try to 'fix' by making further
>  algorithmic changes — the plan was to try the vectorize-then-measure."

DoD item 4 PASSES (121.8 ≤ 150). But DoD item 3 (wall Δ) FAILS with +0.362s regression.
Reporting as **DONE_WITH_CONCERNS** so downstream plan can decide:

1. **Rollback option**: set `COREP_FAST_LABELS_TO_LIST_VECTORIZED=0` (env-level
   rollback available; no code change needed).
2. **Iterate option**: the 311+310 ms from listcomp+tolist and 148 ms from
   `np.split` dominate. Possible next-step: eliminate `np.split` by using
   single-pass bincount-based boundary slicing, and call `tolist()` once on
   the full flat array instead of per-component.
3. **Drop option**: keep legacy; revert commit cb69001. Parity tests + feature
   flag stay (cheap to keep for future attempt).
