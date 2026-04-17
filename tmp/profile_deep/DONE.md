# CoReP Deep Profiling — Definition of Done

Spec: `docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md` §9
Plan: `docs/superpowers/plans/2026-04-16-corep-deep-profiling-implementation.md`
Final results: `my-docs/20260417-corep-deep-profiling-results.md`

---

| # | DoD item | Status | Evidence |
|---|---|---|---|
| 1 | All D1-D7 files present | ✅ PASS | Spec `docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md` (commit 82240cc); Results doc `my-docs/20260417-corep-deep-profiling-results.md` (commit 0ac51ae); per-stage CSVs `tmp/profile_deep/results/per_stage_ops_res{128,256}.csv`; top-20 kernel CSV `tmp/profile_deep/results/top20_kernels_res256.csv`; scaling table `tmp/profile_deep/results/scaling_table.csv`; raw traces `tmp/profile_deep/results/layer{1,3}_res{256,128}_run{1,2,3}_trace.json` (local-only, 3.8GB × 3 + 1.1GB × 3, not git-tracked per Task 7/10 size constraints). |
| 2 | Layer 0 ≥ 3 observations | ✅ PASS | See `tmp/profile_deep/results/layer0_observations.md` §"Three concrete observations" (commit 8915b7f + fix e876acc) — (1) GPU idle ~96%, (2) 4022 D2H ≈ 4128 stream sync = `.item()` pattern, (3) single stream + one 1048 ms `cudaDeviceSynchronize`. |
| 3 | Layer 1 top-30 op per stage at both res | ✅ PASS | `per_stage_ops_res256.csv` (commit a418dab) and `per_stage_ops_res128.csv` (commit 53dfca5). Verified: `awk -F, 'NR>1 {print $1}' ... \| sort \| uniq -c` shows exactly 30 rows for each of s1_voxelize, s2_components, s3_edge_weights, s4_face_point, s6_collapse, s7_rank_assign, s8_decode, and __unassigned__ at both resolutions. |
| 4 | Layer 2 top-20 + attribution + class + UNK ≤ 20% | ✅ PASS | `top20_kernels_res256.csv` (commit 8fc107f after task9-fix2). Verified via `csv.DictReader`: Counter({'MMB': 17, 'LNB': 3}). UNK rate 0/20 = 0% — satisfies ≤20% threshold. Caveat: `py_line` column is empty across all rows (torch.profiler Chrome trace stores `External id` ints not `Call stack` strings; resolving would require a second-pass join — deferred to ncu follow-up). |
| 5 | Scaling table flags + hypotheses | ✅ PASS | `scaling_table.csv` (commit 53dfca5) + results doc §Layer 3 (commit 0ac51ae). Flagged stages: s4 ratio=1.68x (below expected 4x), s7 ratio=1.11x (below expected 4x), `__unassigned__` ratio=1.01x (near-constant). Each flagged stage has a written hypothesis in the results doc §"Flagged stages": s4 and s7 are host-bound (Python-loop overhead dominates, GPU work is minimal); `__unassigned__` is PyTorch framework overhead unrelated to CoReP computation. |
| 6 | ROI list ≥ 5 ranked candidates | ✅ PASS | Results doc §"ROI-Ranked Next-Step Candidates" (commit 0ac51ae) has 6 candidates. Verified: `awk '/## ROI-Ranked/,/## Recommended/' ... \| grep -c "^| [0-9]"` returns 6. Each row has target stage / hypothesis + evidence / estimated Δ@res=256 / effort / risk. |
| 7 | Smoke check < 10% OR documented fallback | ✅ PASS | `tmp/profile_deep/results/smoke_overhead.log` (commit 3af0d99) shows measured delta = 41.3% (above 10% threshold). Fallback documented and APPLIED: `driver.py`'s `apply_substage_events()` disabled in commit 36a1e9d. Spec explicitly allows documented fallback to NVTX-only as an alternative to the <10% requirement. |
| 8 | Provenance headers in CSVs/JSONs | ⚠️ PARTIAL | Summary JSONs carry `header` block (git_sha, timestamp, nvidia_smi, torch_version, cuda_version, layer, resolution, seeds) as set by driver.py commit f9f8368. CSV outputs (per_stage_ops_*.csv, top20_kernels_*.csv, scaling_table.csv) do NOT carry an embedded header — the analyzers write rows only. Traceability is covered by: (a) each CSV is a discrete git artifact with its own commit SHA, (b) `layer1_res256_runlog.txt` and `layer3_res128_runlog.txt` record driver + trace provenance, (c) `smoke_overhead.log` documents the fallback decision. Strict reading of spec §9.8 ("every CSV/JSON") is not fully satisfied; a re-run with prepended comment lines would be needed for literal compliance. |

**Note on item 8:** CSV provenance is traceable via git history and run-logs but not embedded inline. To fully satisfy §9.8 literally, both `analyze_layer1_trace.py` and `analyze_layer3_scaling.py` would need to prepend a comment row; deferred unless user requests a re-run.

---

## Summary

7 of 8 DoD items PASS; 1 PARTIAL (item 8 — CSV files lack inline provenance headers, though full traceability is available via git commits and run-logs). The investigation successfully profiled CoReP at 4 layers, identified the pipeline as 96% GPU-idle with zero compute-bound kernels in the top-20, and ranked 6 concrete next-step ROI candidates.

**Primary deliverable:** `my-docs/20260417-corep-deep-profiling-results.md` (commit 0ac51ae). Top recommendation: shelve Triton K1 (now #4 in ROI); prioritize sync-pattern elimination (`.item()` loop elimination + `cudaDeviceSynchronize` removal in s4/s6/s7) as ROI #1-2 with estimated Δ of 0.5–1.5 s at res=256.
