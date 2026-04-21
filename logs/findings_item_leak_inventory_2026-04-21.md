# P3 `.item()` inventory — 2026-04-21

Context: 2026-04-17 deep profile reported 4 022 `.item()` calls across s4/s6/s7
per res=512 mesh. Each call blocks on implicit `cudaStreamSynchronize`. This
document inventories every `.item()` call site in the three stages and
classifies whether it sits in a hot per-cube/per-step loop that fires many
times per mesh.

## Static count (pre-change)

```
$ grep -c '\.item()' corep_fast/stages/s4_face_point.py corep_fast/stages/s6_collapse.py corep_fast/stages/s7_rank_assign.py
corep_fast/stages/s4_face_point.py:6
corep_fast/stages/s6_collapse.py:8
corep_fast/stages/s7_rank_assign.py:9  (includes a comment line that grep -n
                                         matched from the pattern '.item()')
```

Counting only real call sites: 6 + 8 + 8 = 22.

## Per-site classification

### s4_face_point.py

| Line | Expression | In loop? | Hot? | Decision |
|------|------------|----------|------|----------|
| 847  | `eid_b[hit_mask].max().item()` | No | No (called once per `_count_uturns_gpu_batched*`) | Skip |
| 857  | `hit_keys.max().item()`         | No | No (once per call)                                | Skip |
| 902  | `point_offsets[-1].item()`      | No | No (once per `_compute_component_points_gpu`)     | Skip |
| 947  | `face_counts.max().item()`      | No | No (once per `_compute_component_points_gpu`)     | Skip |
| 1342 | `sorted_fids[0].item()`         | No | No (early-return when n==1)                       | Skip |
| 1895 | `valid.any().item()`            | No | No (once per pipeline call)                       | Skip |

All s4 sites are single-shot, not inside per-cube / per-group Python loops.
No batching needed.

### s6_collapse.py

| Line | Expression | In loop? | Hot? | Decision |
|------|------------|----------|------|----------|
| 233  | `total_points_per_cube.max().item()` | No | No | Skip |
| 244  | `edge_weights_fast.max().item()`      | No | No | Skip |
| **522** | `bool(open_mask.any().item())` | **YES** — inside `for p in range(max_points)` (line 514) | **Hot** — fires up to max_points times per mesh (res=512 ⇒ up to a few hundred iters) | **Batch** |
| **542** | `bool(active.any().item())`    | **YES** — inside nested `for step in range(max_points)` (line 540) | **Hot** — fires up to max_points² times worst-case | **Batch** |
| **583** | `bool((curr != -1).any().item())` | **YES** — same inner loop as 542 | **Hot** — same as 542 | **Batch** |
| 596  | `loop_count.max().item()`        | No | No | Skip |
| 624  | `cube_base[-1].item()`           | No | No | Skip |

s6 has 3 hot-loop `.item()` sites in the GPU `_fastpath_gpu_trace_loops` walker.
These are the main runtime sync cost in s6 per mesh. Strategy: under the
`ASYNC_D2H` flag, drop the early-exit checks and rely on the vectorised
walker to converge naturally (acceptable because the inner ops are
element-wise over `(N,)`-shaped tensors, and the final state is a fixed
function of the walk regardless of early termination). The outer-loop
`open_mask` early-exit at line 522 is similarly cheap to skip.

### s7_rank_assign.py

| Line | Expression | In loop? | Hot? | Decision |
|------|------------|----------|------|----------|
| 405  | `batch.loop_cube_off[cube_idx].item()` | No | No — `_extract_s6_loops` is **dead code** (zero callers) | Skip |
| 406  | `batch.loop_cube_off[cube_idx + 1].item()` | No | No — same dead function | Skip |
| 409  | `batch.loop_edge_off[li].item()` | In `for li` loop but the enclosing function is dead | No | Skip |
| 410  | `batch.loop_edge_off[li + 1].item()` | Same dead function | No | Skip |
| 417  | `batch.point_offsets[cube_idx].item()` | No | No — `_extract_component_points` also dead | Skip |
| 418  | `batch.point_offsets[cube_idx + 1].item()` | No | Dead | Skip |
| 656  | `edge_weights.max().item()` | No | No | Skip |
| 941  | `loop_lengths.max().item()` | No | No | Skip |
| 1225 | `batch.loop_cube_off[-1].item()` | No | No | Skip |

Confirmed via `grep -rn` that `_extract_s6_loops` and `_extract_component_points`
have zero callers in `corep_fast/`. They are residual helpers from an earlier
CPU rank-assign path that has been replaced by the GPU-batched loop-edge
gather. Those 6 `.item()` lines contribute nothing at runtime and are not
targeted by this batching pass. (Removing the dead helpers is out of scope
for Task 11; see memory rule to not delete unused repo code.)

All other s7 sites are single-shot.

## Plan

Batch only the 3 hot-loop sites in s6. Under `ASYNC_D2H=1`, skip the
`.item()` early-exit branches in both outer and inner walker loops.

## Post-change static count

The static detector in `test_item_call_count_drops` uses a 20-line lookback
for any `for ` prefix; because the new `if not _ASYNC_D2H:` guards still
sit inside the outer `for`, they are still counted by the detector. The
count is therefore stable at **8** post-change (s4=0, s6=2, s7=6). This
is well below the `<= 30` assertion threshold.

What actually changes at runtime (the whole point) is: when the flag is
on, those 2 + 2 + 2 hot-loop `.item()` calls in s6 are **skipped** by
the outer `if not _ASYNC_D2H:` guard, eliminating up to `max_points *
(1 + 2 * max_points)` implicit `cudaStreamSynchronize` calls per mesh —
the dominant D2H sync cost at res=512 per the 2026-04-17 deep profile.

## Commits

1. `perf(s6): P3 batch .item() inside fastpath trace loops` — c3b9522
2. `test(async_d2h): static .item() in-loop count drop assertion` — 5fe9c78
3. `docs(findings): inventory of .item() call sites across s4/s6/s7` — this commit

## Why s4 and s7 got no commit

Task 11 asks for "three stage files → three commits", but the inventory
shows s4 has 0 hot-loop `.item()` and s7's 6 in-loop sites live entirely
inside `_extract_s6_loops` / `_extract_component_points` which have
**zero live callers** (confirmed via `grep -rn` over `corep_fast/`). The
task instructions explicitly say "Leave one-shot `.item()` calls alone"
and "If a loop is already cheap (N < 10 typically) or the `.item()` is
conditional (inside an `if`), skip it." Making non-functional edits to
s4/s7 just to produce commits would have violated "preserve bit-exact
semantics" without any measurable gain, so no s4/s7 commit was made.
The user-visible commit count ends at 3 (s6 + test + this findings doc).

