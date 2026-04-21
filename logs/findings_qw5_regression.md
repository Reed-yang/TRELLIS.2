# QW5 Regression Investigation — 2026-04-21

## TL;DR

QW5 (lowering `s2_components` dense-adjacency threshold from **64 → 32** under
`COREP_FAST_VRAM_RESCUE=1`) degraded s2 wall time by **2 orders of magnitude**
on specific mesh topologies. After investigation, the optimisation is **disabled
in-place** (threshold reverts to 64 even under VRAM_RESCUE=1); the flag and
rescue track remain for QW1/QW3/QW6 + OOM retry. This doc records why the
decision was reached.

## Empirical evidence

10-minute 500-mesh A/B on host-10-240-99-119 (8 × H100, res=512):

| Config | Flags | n_ok / wall | mean | median | p95 | p99 | s2 p99 | throughput |
|---|---|---|---|---|---|---|---|---|
| BASELINE | all OFF | 261 / 300s | 8.38s | 8.22s | 14.5s | 17.3s | **2.7s** | 0.870 mesh/s |
| PROD | VRAM_RESCUE=1 only | 169 / 600s | 24.02s | 6.36s | 140.9s | 171.3s | **162.2s** | 0.282 mesh/s |
| ALL | VRAM+SPARSE+ASYNC=1 | 147 / 600s | 26.18s | 5.95s | 160.4s | 203.2s | 195.4s | 0.245 mesh/s |

Same-mesh comparison (147 meshes processed in all three runs):

| sha256 | bucket | ref_enc_s | s2 BASE | s2 PROD | s2 ALL |
|---|---|---|---|---|---|
| 0898cb8a62eb… | very_fast | 4.5 | **0.62s** | **73.06s** | **102.75s** |
| 02fca5ab5689… | very_fast | 3.9 | 0.13s | 56.49s | 70.61s |
| 0e8053881e00… | very_fast | 2.4 | 0.23s | 34.06s | 40.84s |
| 04b552bae061… | fast | 8.9 | 0.06s | 0.08s | 0.07s |
| 0d73de770f75… | fast | 8.6 | 0.06s | 0.05s | 0.05s |

Regression is **not uniform** — most meshes are unaffected. Affected meshes
see a 100×–400× slowdown concentrated entirely in s2_components.

## Root-cause analysis

### Two s2 paths

`corep_fast/stages/s2_components.py` dispatches connected-components counting
based on `max_faces = padded_faces.shape[1]`:

1. **Dense path** (line 120-147, `if max_faces <= _dense_threshold:`) — one big
   batched GPU op: `match = (neighbors_exp == faces_exp)` producing a `(N, M, 3, M)`
   tensor, then iterative label propagation. Fully vectorised, few CUDA syncs.
   For N=600k cubes with M=64: ~4.6 GB temporary, handled in a few seconds.
2. **Sequential path** (`_label_propagation_sequential`, line 220-270) — a
   **Python `for cube_idx in range(N)` loop** that does per-cube Union-Find
   using Python `list` + `dict`, with repeated `.item()` / `.tolist()` D2H
   reads.

### Cost of the sequential path

Per cube with `n_valid` faces, the inner block:

```python
for slot_j in range(n_valid):
    for e in range(3):
        nbr_face = int(nbrs[slot_j, e].item())   # <- D2H sync per call
        if nbr_face in face_to_slot:
            union(slot_j, face_to_slot[nbr_face])
```

performs **`3 × n_valid` `.item()` calls per cube**, each carrying an implicit
`cudaStreamSynchronize`. Plus `int(cube_mask.sum().item())`, `face_ids.tolist()`,
`torch.where(cube_mask)[0]`, and per-slot scalar writes `labels[cube_idx,
valid_indices[s]] = find(s)` (another ~n_valid syncs).

For a cube with n_valid=50: **~450 D2H syncs per cube**. At ~10-20 µs per sync
(kernel-launch + D2H RTT on H100), that's **~5-10 ms per cube** — before
any actual work.

### What QW5 actually does

Threshold 64 → 32 under VRAM_RESCUE:

- Cubes with `max_faces ≤ 32`: unchanged (dense path both before and after).
- Cubes with `max_faces ∈ [33, 64]`: **moved from dense → sequential**.
- Cubes with `max_faces > 64`: unchanged (sequential path both before and after).

For a mesh where `max_faces` ends up in the 33-64 band (common when a mesh has
moderately dense face clusters per voxel cube), QW5 redirects **the entire
batched GPU op to a Python loop with N iterations and hundreds of D2H syncs
per iteration**.

Observed on mesh `0898cb8a62eb` (very_fast bucket, ref 4.5s):
- BASELINE: dense path, s2 = 0.6 s.
- QW5 ON: sequential path, s2 = 73 s.  Ratio: **122×** slowdown.

### Why the original QW5 rationale was wrong

Plan §5.1 cited: *"At res=512, max_faces occasionally reaches 40-100, pushing
s2 VRAM to 62 GB p99."* The claim: lower threshold to 32 → avoid the (N, M, M)
spike.

But the 62 GB spike is caused by **max_faces > 64**, not 33-64:

- `(N, 40, 3, 40)` at N=1M = 4.8 GB — fits easily.
- `(N, 100, 3, 100)` at N=2M = 60 GB — this is the spike.

Cubes with `max_faces > 64` already go through `_label_propagation_sequential`
**before** QW5 — they never touched the dense allocation. QW5 doesn't reduce
the spike; it just moves more of the fast-dense-handled `[33, 64]` band into
the slow-sequential path.

**QW5 misdiagnosed the problem.** The actual 62 GB VRAM peak comes from
`max_faces > 64` cubes, which sequential already handles. QW5 provides zero
VRAM benefit on those and imposes 100-400× wall regression on cubes where the
dense path was fine.

## Decision

**Disable QW5 in-place** (patch `s2_components.py` to keep `_dense_threshold = 64`
unconditionally). Keep `VRAM_RESCUE` flag + all other quick wins active:

- QW1 (s4 free cdist intermediates) — pure memory free, no wall cost.
- QW3 (s6 trim loop_start_slot) — pure memory free, no wall cost.
- QW6 (s4 single-group CPU fallback) — only fires on OOM; zero cost when not
  triggered, eliminates 80+ GB allocations when it does.
- OOM retry chain — only fires on OOM.

Result: `VRAM_RESCUE=1` is now a pure win (100% success vs 75.7% baseline)
with no s2 regression.

**Not reverting the commit** — the QW5 commit `120c5a7` stays in history for
traceability; the disabling patch references this file so future readers can
find the reasoning. If someone later wants to revisit the M > 64 case (62 GB
VRAM spike), they can rewrite `_label_propagation_sequential` in vectorised
form first, then decide whether a threshold change still makes sense.

## Followup (future spec)

- Vectorise `_label_propagation_sequential`. The Union-Find logic can be done
  with `torch.scatter_reduce_(reduce='amin')` over a CSR edge list — similar
  to the Task-8 sparse module, but without that module's own perf issues.
  Once vectorised, sequential becomes competitive with dense for all M and
  the threshold debate becomes moot.
- Real 62 GB s2 VRAM spike (max_faces > 64) is not addressed here — production
  workload currently survives it because 80 GB HBM has enough headroom on the
  2026-04-21 test sample. If the 168k run hits OOM in s2 (not s4), revisit.

## Cross-refs

- Profile data: `my-docs/20260421-throughput-profile-results.md`.
- Plan / spec: `docs/superpowers/plans/2026-04-21-168k-throughput-implementation.md`
  Task 4; `docs/superpowers/specs/2026-04-21-168k-throughput-design.md` §5.1 QW5.
- Raw numbers: `tmp/profile_throughput/results/throughput_10min/`,
  `throughput_10min_prod/`, `throughput_5min_baseline/`.
