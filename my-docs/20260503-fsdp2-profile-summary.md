# W3: FSDP2 zero2 profile + decision tree (2026-05-03)

> Investigation of W2.2's +21% step regression via chrome trace + bucket diff.
> Conclusion: NCCL 12.6× explosion + comm_hidden_ratio 96% → 29% drop.

---

## 1. Bucket comparison (DDP baseline vs FSDP2 zero2)

Source traces:
- DDP: `results/profile_dit_runs/chrome_trace_fused_mod_231103/profile/...` (5 active steps, post-fused_mod)
- FSDP2 zero2: `results/profile_dit_runs/w3_trace_zero2_082519/profile/...` (5 active steps, W2.2 commit 6494420)

| Bucket | DDP (ms/step) | FSDP2 zero2 | Δ ms | Δ % |
|---|---|---|---|---|
| **nccl** | **55.0** | **693.6** | **+638.6** | **+1161%** |
| eltwise | 460.5 | 425.8 | -34.7 | -7.5% |
| gemm | 218.9 | 275.0 | +56.1 | +25.6% |
| flash_attn | 210.9 | 97.7 | -113.2 | -53.7% |
| other | 132.8 | 109.5 | -23.3 | -17.5% |
| reduce | 78.7 | 79.8 | +1.1 | +1.4% |
| layernorm | 27.5 | 27.9 | +0.4 | +1.5% |
| memcpy | 7.4 | 3.2 | -4.2 | -56.8% |
| optimizer | 0 | 1.8 | +1.8 | new |
| **TOTAL** | **1191.7** | **1714.2** | **+522.5** | **+43.8%** |

**comm_hidden_ratio: 0.961 (DDP) → 0.286 (FSDP2 zero2)** — overlap collapsed from near-perfect to less than 1/3 hidden.

**compute_stream_busy: 1136.7 (DDP) → 1020.7 (FSDP2)** — compute itself is slightly faster on FSDP2 (eltwise + flash_attn dropped, suggesting bf16 param storage helps DRAM bandwidth).

**comm_stream_busy: 55.0 (DDP) → 693.6 (FSDP2)** — communication time exploded.

## 2. Suspect verdict

| Suspect (from W22_VERDICT) | Verdict |
|---|---|
| Per-step EMA inner.reshard() pollution | ❌ NOT the culprit. eltwise actually decreased |
| Elastic controller under-using mem | ❌ Not primary. compute_stream is normal |
| `no_sync` shim broken with batch_split=2 | ⚠ POSSIBLY contributing (would 2× comm) |
| **NCCL fp32 reduce_dtype** (NEW suspect) | ✅ **STRONG**. spec §6.3 D2 was "defer pending discussion" |
| **Missing FSDP2 prefetch hints** (NEW suspect) | ✅ **STRONG**. Per-block reduce_scatter not overlapping with adjacent block compute |

## 3. Root cause hypothesis (ranked)

### Primary: missing FSDP2 prefetch hints
Per-block reduce_scatter is sequential w.r.t. each block's bwd. PyTorch FSDP2 needs explicit `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` calls to schedule ahead. Without them, comm + compute serialize → comm_hidden_ratio collapses.

Expected fix: add prefetch hints in `coart/dit/parallel/fsdp2.py` after the wrap loop. Should restore comm_hidden_ratio toward 0.8-0.9 range.

### Secondary: reduce_dtype=fp32 doubles bytes vs bf16
Each per-block reduce_scatter is sending fp32 grads (vs DDP's autocast bf16). For ~43M params/block × 4B = 172 MB per reduce_scatter × 30 blocks = 5.16 GB collective traffic per step (vs ~2.6 GB if bf16). Halving via bf16 reduce_dtype would cut nccl roughly in half (~350 ms instead of 693 ms).

Spec §6.3 marked D2 as "defer pending user discussion" because gradient noise needs ablation. **But empirical evidence now shows this is the second-largest lever.** Recommendation: get user approval to flip to bf16 in a follow-up wave.

### Tertiary: `no_sync` shim with batch_split=2
Even if prefetch is fixed, batch_split=2 may double comm if the shim isn't actually skipping reduce_scatter on the accumulation iteration. Need to verify by running with batch_split=1 and see if NCCL drops by ~1/2.

## 4. Spec §6.5 decision tree application

| Tree branch | Condition | Observed | Action |
|---|---|---|---|
| `comm_hidden_ratio < 70%` | yes (0.286 < 0.70) | YES | **TRIGGER** wrap粒度 / prefetch tuning |
| `eltwise_oncritical_ms ≥ 400 ms` | true | yes (425.8) | C7 fused RMSNorm priority **unchanged** for W5 |
| `eltwise_oncritical_ms < 300 ms` | false | n/a | — |
| `optim_step_isolated_ms < 30 ms` | true (1.8 ms) | yes | **C6 ROI ≈ 0** → DROP C6 from W4 |
| `zero2 step time > DDP × 1.10` | yes (×1.21) | YES | **TRIGGER** reduce_dtype=bf16 ablation discussion |
| `bs12 mem peak > 75 GB` | unknown — needs W4 sweep | n/a | check during W4 |

## 5. Revised W4/W5 priorities

| Wave | Original plan | Revised |
|---|---|---|
| W4 C5 (bs↑) | bs=8 → 12 sweep | unchanged — mem savings (-13.7 GB) means bs=12 should fit |
| **W4 C6 (apply_optim_in_backward)** | -30~50 ms | **REMOVED** — optim is 1.8 ms, no ROI |
| W4 NEW: **FSDP2 prefetch hints** | not in original plan | **ADD** — primary fix for +21% regression |
| W4 NEW (optional, needs user ack): **reduce_dtype=bf16** | spec §6.3 D2 deferred | **PROMOTE** to W4 candidate, gated by user approval |
| W5 C7 (fused RMSNorm) | -110~140 ms | unchanged. eltwise still dominates |

## 6. Concrete W4 follow-up patch (FSDP2 prefetch)

Add to `coart/dit/parallel/fsdp2.py` after the wrap loop:

```python
# Schedule fwd-prefetch: each block prefetches the next during its forward
for i in range(len(inner.blocks) - 1):
    inner.blocks[i].set_modules_to_forward_prefetch([inner.blocks[i + 1]])
# Schedule bwd-prefetch: each block prefetches the previous during its backward
for i in range(1, len(inner.blocks)):
    inner.blocks[i].set_modules_to_backward_prefetch([inner.blocks[i - 1]])
```

Expected effect: comm_hidden_ratio 0.286 → ≥ 0.7, reducing critical-path NCCL from 693 ms to ~200 ms. Would close roughly half the +21% regression.

Combined with reduce_dtype=bf16 (D2): NCCL further halves to ~100 ms. Total step time projection: 1714 ms → ~1200 ms ≈ DDP baseline.

## 7. Files

- `scripts/profiling/analyze_overlap.py` — staged
- `logs/wave3/ddp_baseline/{bucket_table.md, overlap_summary.json}` — DDP reference
- `logs/wave3/zero2/{bucket_table.md, overlap_summary.json}` — FSDP2 actual

## 8. Conclusion

W2.2 +21% step regression diagnosed. Primary fix is **FSDP2 prefetch hints** (no numerical impact, ~5 LOC). Secondary fix is **reduce_dtype=bf16** (numerical, needs user approval). C6 (apply_optim_in_backward) is dead — drop from W4. C7 (W5) unchanged.

Recommend W4 ordering:
1. Add FSDP2 prefetch hints (no-risk; should recover most of regression)
2. Run bs↑ sweep (bs=10/12/14)
3. Discuss reduce_dtype=bf16 with user
4. Skip C6 entirely
