# W_HG — Phase 3 cost-matrix tie scan (F2 @ res=256, F3 @ res=128)

## Raw output

```
=== F2 (icosphere s3 @ res=256) ===
Phase 3 calls: 275539
Brute-force enumerated: 100000 (cap=100000)
[truncated_at_100000]
Skipped (out of brute-force scope nl>5 or npts>8): 0
Rect-reverse (npts < nl, must scipy-fallback): 13
With ties (multiple permutations achieve min cost): 0
Tie pct (over brute-enumerated): 0.0000%
Matrix shape distribution:
  nl: min=1 max=2 p50=1 p99=1 p999=1
  npts: min=1 max=1 p50=1 p99=1 p999=1
  cubes with nl > 5 (out of brute-force scope): 0
  cubes with npts > 8: 0
  shape histogram (nl<=5, npts<=8):
    nl=1 npts=1: 275526  (99.9953%)
    nl=2 npts=1: 13  (0.0047%)

=== F3 (triple icosphere @ res=128, multi-loop path) ===
Phase 3 calls: 113607
Brute-force enumerated: 100000 (cap=100000)
[truncated_at_100000]
Skipped (out of brute-force scope nl>5 or npts>8): 0
Rect-reverse (npts < nl, must scipy-fallback): 13
With ties (multiple permutations achieve min cost): 0
Tie pct (over brute-enumerated): 0.0000%
Matrix shape distribution:
  nl: min=1 max=2 p50=1 p99=1 p999=1
  npts: min=1 max=2 p50=1 p99=1 p999=1
  cubes with nl > 5 (out of brute-force scope): 0
  cubes with npts > 8: 0
  shape histogram (nl<=5, npts<=8):
    nl=1 npts=1: 113592  (99.9868%)
    nl=2 npts=1: 13  (0.0114%)
    nl=2 npts=2: 2  (0.0018%)
```

Note: brute-force enumeration was capped at 100 000 calls per fixture; the
cap never fires inside an interesting shape — F2 runs the cap on 100 k out
of 275 539 cubes, F3 on 100 k out of 113 607 cubes, but because 99.99 %+
of shapes are 1×1 the sampled population is representative: a 1×1 matrix
cannot have ties (only one permutation exists), so the tie pct is an upper
bound on ties in the unseen 175 k / 13 k tail only for the rare non-1×1
shapes (which constitute ≤ 0.015 % of all calls).

## Decision

| Tie pct | Action for Task 17 brute-force tie-break |
|---|---|
| < 0.1 % | Use lex-smallest permutation index (argmin on flattened costs); for the rare ties, fall back to scipy per-cube. |
| 0.1 – 5 % | Implement deterministic tie-break that matches scipy's Jonker-Volgenant. Spike a small Python harness comparing JV output vs lex-min on tied matrices. |
| > 5 % | Replicate scipy tie-break exactly; cannot use lex-min fallback (too many escapes). |

**Decision:** SAFE-FALLBACK (lex-smallest + scipy fallback on detected ties)

**Reasoning:**

- Empirical tie pct = **0 / 100 000 = 0.0000 %** on both F2 and F3 scans —
  far below the 0.1 % threshold. Even padded by the un-enumerated 175 k
  F2 / 13 k F3 tail, the worst-case tie pct is bounded by the fraction of
  non-1×1 shapes, which is ≤ 0.015 %.
- Matrix shape distribution is pathologically trivial: 99.99 %+ of cubes
  are `(nl=1, npts=1)` which has a single permutation and cannot tie.
  The rare non-trivial shapes (`2×1`, `2×2`) contribute ≤ 15 / 275 539
  = 0.0054 % (F2) and ≤ 15 / 113 607 = 0.0132 % (F3).
- All observed shapes fit within the brute-force enumeration envelope
  `nl ≤ 5, npts ≤ 8` — zero cubes exceed either bound on F2/F3.
- `2×1` rect-reverse (npts < nl, 13 occurrences each fixture) always
  must scipy-fallback regardless (Hungarian requires npts ≥ nl for a
  valid assignment covering all rows).

**Recommendation for Task 17:**

1. Batched brute-force kernel enumerates all permutations for the common
   small-shape bucket (nl ≤ 5, npts ≤ 8 — covers 100 % of F2/F3 cubes)
   and picks lex-smallest permutation index on cost equality.
2. On the rare path where two permutations tie within `1e-12`, fall back
   to scipy per-cube (cost is negligible given tie pct ≤ 0.015 %).
3. For shapes outside the `(5, 8)` envelope (not observed on F2/F3 but
   may appear on complex meshes): scipy fallback per-cube.
4. For `npts < nl` (rect-reverse, 13 cubes per fixture): scipy fallback
   per-cube; padding to square would corrupt scipy's cost semantics.

Assert-then-fallback keeps bit-exactness against scipy: the GPU kernel
detects ties during reduction (compare-and-swap on `(cost, perm_idx)`
tuples yields lex-smallest by permutation index; a second pass checks
whether another permutation's cost is within epsilon of the min, and if
so routes the cube to scipy).

## Matrix shape distribution

**F2 (icosphere s3 @ res=256) — 275 539 cubes total:**

| Shape | Count | Pct |
|---|---|---|
| 1×1 | 275 526 | 99.9953 % |
| 2×1 (rect-reverse → scipy) | 13 | 0.0047 % |

**F3 (triple icosphere @ res=128, multi-loop) — 113 607 cubes total:**

| Shape | Count | Pct |
|---|---|---|
| 1×1 | 113 592 | 99.9868 % |
| 2×1 (rect-reverse → scipy) | 13 | 0.0114 % |
| 2×2 | 2 | 0.0018 % |

## Edge cases

- Calls with `nl > 5` or `npts > 8` (out of brute-force scope): **0 on
  both F2 and F3**. Task 17 can treat these as cold-path scipy fallback
  without meaningful ROI loss.
- Calls with `npts < nl` (rect-reverse, per spec must scipy-fallback):
  **13 on F2, 13 on F3** (all 2×1). These will always dispatch to scipy
  regardless of the GPU kernel; count is negligible.
- 1×1 is a no-op (single permutation, no tie possible). Task 17's GPU
  kernel should short-circuit this shape to `argmin(cost[0])` with zero
  enumeration cost; 99.99 %+ of wall time lives here.

## Task 16 deliverable — summary for Task 17 implementer

| Question | Answer |
|---|---|
| Tie-break strategy | **Lex-smallest permutation index + scipy fallback on detected ties** |
| Complexity reduction vs full scipy-replication | High — no need to replicate Jonker-Volgenant internals |
| Hot-path shape | 1×1 (99.99 %+) — trivial argmin, no enumeration |
| Cold-path shape envelope for brute-force | nl ≤ 5, npts ≤ 8 (covers 100 % F2/F3 non-trivial) |
| Scipy-fallback triggers | (a) tie within 1e-12; (b) npts < nl; (c) shape outside (5,8) envelope |
| Expected scipy-fallback pct | ≤ 0.015 % on F2/F3 (≈ 15–50 cubes per fixture) |
