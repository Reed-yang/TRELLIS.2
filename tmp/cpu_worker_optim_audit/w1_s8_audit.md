# W1 Bucket A audit: s8_collapse.py
Date: 2026-04-17
HEAD at start: a3daaf5b503a70a014e15fb22a5311f4e6fe8ceb

## Method

Target functions (line ranges from `^def` grep):
- `_process_shared_edges_from_tensors`: lines 1629-1800
- `process_geometry_vectorized`: lines 920-1247

Scan pattern inside each function body (`awk 'NR>=start && NR<=end'`):
`\.item\(\)|\.cpu\(\)|\.tolist\(\)`.

## `_process_shared_edges_from_tensors` (1629-1800)

Found scalar / bulk D2H calls:

- line 1788: `tri_verts_torch_np = tri_verts_torch.cpu().numpy()`
  - Usage: bulk transfer of the vectorized-Torch triangle buffer so it can be
    `np.concatenate`-ed with Python MP worker outputs in Step F, then welded
    via `_weld_and_dedup`.
  - Bucket classification per findings §2: this is **Bucket B** (bulk
    transfer fronting CPU work), not a scalar `.item()`-style Bucket A site.
    The top-20 entries for `s8_collapse.py:1629` / `:920` are the
    host-side dispatch micro-sync attributed to the function *entry*, not
    this line.
  - Disposition: **load-bearing** — feeds `np.concatenate` merge below; a
    fully-GPU rewrite would eliminate it but that is Bucket B (1-3 day
    effort, architectural).
  - Annotated with `# bucket-A-audit: load-bearing bulk transfer ...` comment.

No `.item()` / `.tolist()` calls inside this function body.

## `process_geometry_vectorized` (920-1247)

Found scalar D2H calls:

- line 1045: `max_loops = int(num_loops.max().item()) if E > 0 else 0`
  - Usage: immediately drives `torch.arange(max_loops, device=device).view(1, 1, max_loops)`
    at line 1053 and sizes multiple `(E, 4, max_loops, K)` tensors (Steps 4-8).
  - Bucket classification: **Bucket C (alloc-size)** per findings §2 rank 18.
    Not Bucket A. `torch.arange` requires a Python int for the stop argument,
    so the host must materialize this scalar before allocation.
  - Disposition: **load-bearing** — cannot be deleted without a deferred-alloc
    / fused-kernel rewrite (Triton K-class work).
  - Annotated with `# bucket-A-audit: load-bearing (alloc-size ...)` comment.

No `.cpu()` / `.tolist()` calls inside this function body.

## Summary

- Deletions: **0**
- Load-bearing (annotated): **2**
  - `process_geometry_vectorized:1045` — Bucket C alloc-size
  - `_process_shared_edges_from_tensors:1788` — Bucket B bulk transfer
- Expected perf Δ per findings §2: both sites fall outside Bucket A scope
  (Bucket A aggregate in top-20 is ~0.08 ms across ranks 8/10/+0.5·9, and
  findings notes the rank-8 / rank-10 entries are nsys function-entry
  attributions, not source-level `.item()` calls). No measurable perf Δ
  from this commit — purpose is to validate the W1-W7 commit flow.

## Notes

- The wider `.cpu().numpy()` / `.tolist()` set in the file (lines 283-306,
  1394-1395, 1490-1595, 2278-2279) live outside the two target functions
  (`decode_from_cubebatch`, `_process_shared_edges_torch` legacy path,
  `_build_grids_from_cube_map`, `_build_grids_from_tensors`,
  `_build_single_cube_dict`, `s8_collapse_to_ply`) and are explicitly
  **out of scope** for Task W1 (which targets the two functions listed
  above only).
- All such out-of-scope sites are themselves Bucket B bulk transfers
  feeding the Python MP worker path; eliminating them is the separate
  Bucket B track (architectural rewrite).
