# CoReP Deep Profiling — Results

> Date: 2026-04-17
> Branch: gpu-pipeline @ `53dfca539a2f7d31a14059f223e9ad43cd9ece75`
> Hardware: 116 H100 (GPU 0, physical 0; `CUDA_VISIBLE_DEVICES=0`)
> Spec: `docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md`
> Plan: `docs/superpowers/plans/2026-04-16-corep-deep-profiling-implementation.md`
> Baseline (post-s7-fix, commit `83d230d`): res=256 e2e ~10.2s (vs custom 141s = 13.8x)

---

## TL;DR

**Finding 1 — Pipeline is ~96% GPU-idle at res=256.** Kernel+memcpy occupy only ~347 ms per run against 10.056 s e2e (naked NVTX-only nsys run, Task 6). GPU utilisation ≈ 3.5%; the other 9.7 s is CPU-side Python / sync overhead. (Source: Task 6 — `tmp/profile_deep/results/layer0_observations.md`.)

**Finding 2 — No compute-bound kernels in the top-20.** Classifier output: 17 MMB, 3 LNB, 0 CMB, 0 UNK, 0 CPU (Task 9 — `tmp/profile_deep/results/top20_kernels_res256.csv`). The single biggest GPU call is an 11.07 ms `tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>` in s6 (introduced by the s7 bugfix's `cummax` in commit `83d230d`). Pipeline is memory-bandwidth + launch-overhead bound, not compute-bound.

**Finding 3 — s4 and s7 are fixed-cost / host-bound.** Res doubling (128→256) produces 1.68x (s4) and 1.11x (s7) device-time growth respectively — far below the O(R²)=4x band (Task 10 — `tmp/profile_deep/results/scaling_table.csv`). Their wall-clock bottlenecks are Python/CPU orchestration, not GPU work.

**Primary recommendation:** The originally-planned next step (Triton K1 for s4 BFS, estimated −2 to −3 s) addresses <5% of e2e wall-time — s4's total GPU work is only ~37 ms of 6856 ms wall. The real bottleneck is CPU-side sync patterns (4022 D2H `.item()`-style leaks + one 1.048 s `cudaDeviceSynchronize`). **Shelve Triton K1; prioritize sync-pattern elimination in s4/s6/s7 first.**

---

## Layer 0 — Macro timeline (Nsight Systems, res=256)

### Setup
Nsys full-trace (`osrt,cuda,nvtx` backend) on 116 GPU 0 via `tmp/run_layer0_nsys.sh`. Layer-0 mode uses the NVTX-only monkey-patch (sub-stage CUDA events disabled per Task 5 fallback, commit `3af0d99`). Driver runs 1 warmup + 1 measure; both are captured in the same `.nsys-rep`.

### Pipeline outcome
- Naked e2e (driver's printed value): **10.056 s**
- `.nsys-rep` size: 2.6 MB (well below the 100 MB threshold — CPU sampling was auto-disabled on 116's kernel, see runlog warnings)
- Sum-of-NVTX-ranges (measure only): 9.794 s → ~262 ms Python glue outside the 7 stage ranges

### Per-stage NVTX breakdown (measure run)

| Stage | Measure ms | Warmup ms | % of NVTX time |
|---|---:|---:|---:|
| s1_voxelize | 18.71 | 317.95 | 0.2 |
| s2_components | 10.56 | 109.70 | 0.1 |
| s3_edge_weights | 1.27 | 1.70 | 0.0 |
| s4_face_point | **5815.24** | 6059.60 | **58.6** |
| s6_collapse | **2442.29** | 2462.80 | **24.2** |
| s7_rank_assign | **1409.98** | 1423.62 | **14.0** |
| s8_decode | 96.31 | 100.46 | 1.0 |

CoReP has no s5 (historical numbering). All 7 real stages captured. Note Task 7/8/10 layer-1/3 runs use torch.profiler and therefore inflate stage times by ~22% vs. the naked nsys baseline above.

### Top CUDA kernels (both runs combined, from `nsys_res256_stats_cuda_gpu_kern_sum.csv`)

| Rank | Kernel (shortened) | Total ms | Count | Avg µs |
|---:|---|---:|---:|---:|
| 1 | `tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>` (cummax) | 23.58 | 4 | 5894 |
| 2 | `index_elementwise_kernel<OpaqueType<4>>` | 19.54 | 1004 | 19.5 |
| 3 | `reduce_kernel<bool, or_kernel_cuda>` | 15.33 | 1200 | 12.8 |
| 4 | `DeviceSelectSweepKernel<long>` (CUB nonzero scan) | 11.51 | 2522 | 4.6 |
| 5 | `index_elementwise_kernel<OpaqueType<8>>` | 7.94 | 742 | 10.7 |
| 6 | `tensor_kernel_scan_innermost_dim<long, plus>` (cumsum long) | 7.05 | 4 | 1762 |
| 7 | `vectorized_elementwise_kernel<CUDAFunctor_add<long>>` | 6.89 | 58 | 118.8 |
| 8 | `DeviceReduceKernel<unsigned long long, plus>` | 6.73 | 2498 | 2.7 |
| 9 | `elementwise_kernel<CUDAFunctor_add<float>>` | 6.45 | 494 | 13.1 |
| 10 | `elementwise_kernel<where_kernel_impl>` | 6.45 | 32 | 201.7 |

Total kernel time across all 23,836 launches = **259.7 ms** (both runs; ~130 ms/run).

### Memory ops (both runs)

| Op | Total MB | Total ms | Count | Avg/call |
|---|---:|---:|---:|---:|
| **D2H** | 1212.6 | **404.97** (93.1% of memcpy) | **4022** | 0.30 MB |
| D2D | 546.1 | 0.50 | 50 | 10.9 MB |
| H2D | 360.8 | 28.34 | 90 | 4.0 MB |
| memset | 46.4 | 1.15 | 1336 | 0.04 MB |

4022 tiny D2H transfers averaging 301 KB each dominate all memcpy time — each a `.item()` / `.cpu()` / `.tolist()` round trip with implicit CPU↔GPU sync.

### CUDA API summary

| API | Calls | Total ms | % |
|---|---:|---:|---:|
| `cudaDeviceSynchronize` | **20** | **1049.2** | **45.3** |
| `cudaMemcpyAsync` | 4162 | 469.9 | 20.3 |
| `cudaLaunchKernel` | **23,836** | 457.1 | 19.7 |
| `cudaMalloc` | 147 | 176.6 | 7.6 |
| `cudaStreamSynchronize` | **4128** | 144.7 | 6.2 |

**One single `cudaDeviceSynchronize` = 1048 ms** (the Max column); the other 19 calls sum to ~1 ms.

### Stream usage
All 29,334 GPU ops live on **a single default stream (stream id 7)**. Zero stream-level parallelism.

### Three concrete observations (DoD #2)

1. **GPU idle ~96% of wall time.** Sum of kernel (259.7 ms) + memcpy/memset (435.0 ms) across both runs = 694.7 ms → ~347 ms/run. Against 10,056 ms e2e → ~3.5% GPU utilisation.

2. **4022 D2H transfers per pair-of-runs = one sync every ~2.5 ms.** Ratio 4128 `cudaStreamSynchronize` ≈ 4022 D2H is too clean to be coincidence — these are blocking scalar reads (`t.item()`, Python `if scalar_tensor:` comparisons, `.tolist()`), not bulk transfers. Every one forces a round-trip.

3. **Single default stream + one 1.048 s `cudaDeviceSynchronize`.** 19 other DeviceSync calls sum to ~1 ms. One specific host-side barrier is draining a fully-queued GPU in a single flush (likely end-of-s4 or start-of-s6).

---

## Layer 1 — Per-stage op breakdown (torch.profiler, res=256 median run)

### Setup
Task 7 ran `driver.py --layer 1 --res 256` three times. Median e2e was **run 2 at 12.230 s** (torch.profiler overhead ~22% vs. Layer 0 naked 10.056 s). Task 8 parsed the 3.8 GB Chrome trace via `ijson` streaming (see `tmp/profile_deep/analyze_layer1_per_stage.py`) and emitted `tmp/profile_deep/results/per_stage_ops_res256.csv`.

NVTX stage attribution relies on matching `python_function` events by stage-entry function name (torch.profiler does NOT capture `nvtx.range_push/pop`); correctness is verified by the `__unassigned__` bucket holding only 22.6 µs of device time (0.02% of total).

### Per-stage stage wall-time and top-30 device-time aggregate

| Stage | Wall (s, run2) | Device top-30 (ms) | Device / Wall |
|---|---:|---:|---:|
| s1_voxelize | 0.033 | 7.4 | 22.4% |
| s2_components | 0.006 | 3.5 | 58.1% |
| s3_edge_weights | 0.003 | 1.4 | 45.8% |
| s4_face_point | **6.856** | 36.9 | **0.54%** |
| s6_collapse | 3.403 | 17.5 | 0.52% |
| s7_rank_assign | **1.810** | 36.6 | **2.02%** |
| s8_decode | 0.099 | 15.1 | 15.3% |
| e2e | 12.230 | 118.3 (sum) | 0.97% |

The three dominant stages (s4, s6, s7) run at 0.5–2% device-time-over-wall-time — the entire wall-clock is spent on host-side Python.

### Top-5 ops per stage (from `per_stage_ops_res256.csv`)

```
--- s1_voxelize ---
     2.18 ms      40 calls  reduce_kernel<float, sum_functor>
     1.26 ms      30 calls  vectorized_elementwise_kernel<BinaryFunctor>
     0.38 ms       3 calls  CatArrayBatchedCopy<float>
     0.33 ms      10 calls  cross_kernel<float>
     0.30 ms      10 calls  vectorized_elementwise_kernel<AbsFunctor>

--- s2_components ---
     1.49 ms       1 calls  radixSortKVInPlace<2,-1,16,2, long, long>
     0.44 ms       5 calls  reduce_kernel<long>
     0.36 ms      13 calls  vectorized_elementwise_kernel<anonymous>
     0.23 ms       1 calls  bitonicSortKVInPlace<2,-1,16,16, long>
     0.20 ms       8 calls  elementwise_kernel<direct_copy>

--- s3_edge_weights ---
     0.36 ms       8 calls  reduce_kernel<float>
     0.20 ms       6 calls  elementwise_kernel<anonymous>
     0.13 ms       8 calls  vectorized_elementwise_kernel<BinaryFunctor>
     0.13 ms       7 calls  elementwise_kernel<anonymous>
     0.12 ms       4 calls  index_elementwise_kernel<OpaqueType<4>>

--- s4_face_point ---
     4.33 ms    1086 calls  DeviceSelectSweepKernel<long, bool, int>     [LNB]
     3.80 ms     339 calls  index_elementwise_kernel<OpaqueType<4>>
     3.09 ms     238 calls  elementwise_kernel<CUDAFunctor_add<float>>
     2.77 ms    1086 calls  DeviceReduceKernel<int, unsigned long long>  [LNB]
     2.59 ms      17 calls  elementwise_kernel<CUDAFunctor_add<double>>

--- s6_collapse ---
    11.07 ms       1 calls  tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>  (cummax)
     1.76 ms       1 calls  tensor_kernel_scan_innermost_dim<long, plus>  (cumsum)
     0.67 ms      12 calls  index_elementwise_kernel<OpaqueType<4>>
     0.66 ms       3 calls  CatArrayBatchedCopy_aligned16
     0.52 ms       8 calls  DeviceRadixSortOnesweepKernel

--- s7_rank_assign ---
     6.83 ms     587 calls  reduce_kernel<bool, or_kernel_cuda>
     3.18 ms     106 calls  index_elementwise_kernel<OpaqueType<4>>
     2.68 ms       8 calls  vectorized_elementwise_kernel<CUDAFunctor_add<long>>
     2.50 ms      12 calls  elementwise_kernel<direct_copy>
     1.96 ms       5 calls  elementwise_kernel<where_kernel_impl>

--- s8_decode ---
     2.19 ms      33 calls  index_elementwise_kernel<OpaqueType<4>>
     1.42 ms       4 calls  elementwise_kernel<anonymous>
     1.00 ms      30 calls  index_elementwise_kernel (8-byte)
     0.94 ms      16 calls  DeviceRadixSortOnesweepKernel
     0.83 ms      11 calls  DeviceMergeSortMergeKernel
```

Signal: s4's top-5 is CUB-heavy (DeviceSelect+DeviceReduce × 1086 calls each → 2172 CUB launches). s6 is dominated by two large scan kernels (cummax + cumsum). s7 is dominated by a 587-call bool reduction plus index ops.

---

## Layer 2 — Top-20 GPU kernel attribution (res=256)

### Setup
Task 9 heuristic classifier (`tmp/profile_deep/analyze_layer2_top20_kernels.py`) tags each kernel CMB / MMB / LNB / CPU / UNK based on name patterns + (calls, mean-µs) stats. After 2 iterations of rule tuning, UNK = 0/20. Ground-truth verification would require Nsight Compute.

### Class distribution

| Class | Count | Definition |
|---|---:|---|
| MMB | 17 | Memory-bandwidth-bound (reduce/scan/elementwise/index/copy/CUB) |
| LNB | 3 | Launch-bound (mean < 10 µs, count > 1000) |
| CPU | 0 | CPU-bound fallback |
| CMB | 0 | Compute-bound (none found) |
| UNK | 0 | Unclassified |

### Top-20 kernels (from `top20_kernels_res256.csv`)

| # | Stage | Kernel (shortened) | Count | Total ms | Mean µs | Class |
|---:|---|---|---:|---:|---:|---|
| 1 | s6 | `tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>` (cummax) | 1 | 11.074 | 11074.35 | MMB |
| 2 | s7 | `reduce_kernel<bool, or_kernel_cuda>` | 587 | 6.834 | 11.64 | MMB |
| 3 | s4 | `DeviceSelectSweepKernel<long, bool, int>` | 1086 | 4.334 | 3.99 | **LNB** |
| 4 | s4 | `index_elementwise_kernel<OpaqueType<4>>` | 339 | 3.800 | 11.21 | MMB |
| 5 | s7 | `index_elementwise_kernel<OpaqueType<4>>` | 106 | 3.180 | 30.00 | MMB |
| 6 | s4 | `elementwise_kernel<CUDAFunctor_add<float>>` | 238 | 3.090 | 12.99 | MMB |
| 7 | s4 | `DeviceReduceKernel<int, unsigned long long, plus>` | 1086 | 2.766 | 2.55 | **LNB** |
| 8 | s7 | `vectorized_elementwise_kernel<CUDAFunctor_add<long>>` | 8 | 2.676 | 334.56 | MMB |
| 9 | s4 | `elementwise_kernel<CUDAFunctor_add<double>>` | 17 | 2.586 | 152.14 | MMB |
| 10 | s7 | `elementwise_kernel<direct_copy_kernel_cuda>` | 12 | 2.499 | 208.25 | MMB |
| 11 | s8 | `index_elementwise_kernel<OpaqueType<4>>` | 33 | 2.194 | 66.50 | MMB |
| 12 | s1 | `reduce_kernel<float, sum_functor>` | 40 | 2.178 | 54.46 | MMB |
| 13 | s7 | `elementwise_kernel<where_kernel_impl>` | 5 | 1.960 | 391.97 | MMB |
| 14 | s4 | `DeviceReduceSingleTileKernel<int, unsigned long long, plus>` | 1086 | 1.844 | 1.70 | **LNB** |
| 15 | s7 | `elementwise_kernel<where_kernel_impl>` | 12 | 1.801 | 150.06 | MMB |
| 16 | s7 | `tensor_kernel_scan_innermost_dim<long, plus>` (cumsum) | 1 | 1.762 | 1762.49 | MMB |
| 17 | s6 | `tensor_kernel_scan_innermost_dim<long, plus>` (cumsum) | 1 | 1.761 | 1760.70 | MMB |
| 18 | s4 | `elementwise_kernel<where_kernel_impl>` | 14 | 1.691 | 120.75 | MMB |
| 19 | s7 | `vectorized_elementwise_kernel<where_kernel_impl>` | 23 | 1.668 | 72.51 | MMB |
| 20 | s4 | `elementwise_kernel_with_index<arange_cuda_out>` | 362 | 1.586 | 4.38 | MMB |

Sum of top-20 device time ≈ 59.3 ms. Per-stage: s4 = 21.9 ms, s7 = 22.4 ms, s6 = 12.8 ms (dominated by rank 1), s8 = 2.2 ms, s1 = 2.2 ms.

### Standout kernels

- **Biggest single-call:** rank 1 — s6 `tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>` at **11.074 ms for 1 call**. Introduced by Task 3 s7-fix via `cummax` used in the group kth-True selector. Accounts for ~19% of total top-20 device time by itself.

- **3 LNB kernels in s4** (ranks 3, 7, 14): `DeviceSelectSweepKernel`, `DeviceReduceKernel`, `DeviceReduceSingleTileKernel` — each at **1086 calls** with mean 1.7–4.0 µs. Combined total = 8.944 ms. These are the CUB filter+reduce used in s4's GPU fast-path. They'd be an excellent fusion target *if* launch overhead is the real cost — but Layer 3 shows s4 wall-time doesn't track with device work, so eliminating these would only reclaim ~9 ms of 6856 ms wall.

---

## Layer 3 — Resolution scaling (res=256 vs res=128)

### Setup
Task 10 re-ran `driver.py --layer 3 --res 128` three times on 116 GPU 0. Median e2e was **run 1 at 3.929 s**. Re-parsed the Chrome trace via the Task 8 analyzer to produce `per_stage_ops_res128.csv`. Compared per-stage top-30 device_us vs. res=256 to compute per-stage scaling ratios.

### Scaling table (from `scaling_table.csv`)

| Stage | device_us res=256 | device_us res=128 | Ratio | Flag |
|---|---:|---:|---:|---|
| __unassigned__ | 22.6 | 22.4 | 1.01 | below_2x_fixed_cost_dominant |
| s1_voxelize | 7389.5 | 1456.8 | **5.07** | ok_2x_to_10x |
| s2_components | 3458.5 | 1050.8 | **3.29** | ok_2x_to_10x |
| s3_edge_weights | 1365.7 | 423.5 | **3.22** | ok_2x_to_10x |
| s4_face_point | 36906.0 | 21997.1 | **1.68** | below_2x_fixed_cost_dominant |
| s6_collapse | 17531.1 | 4325.2 | **4.05** | ok_2x_to_10x |
| s7_rank_assign | 36585.9 | 33082.8 | **1.11** | below_2x_fixed_cost_dominant |
| s8_decode | 15071.5 | 4271.9 | **3.53** | ok_2x_to_10x |

### Flagged stages — hypotheses

For R doubling, expected O(R²) ≈ 4x or O(R³) ≈ 8x. Ratios outside [2, 10] indicate departure from parallel-compute scaling.

- **s4_face_point (1.68x):** device time grows only 15 ms (22→37 ms) despite cubes more than quadrupling (68,858 → 275,541 = 4.00x cubes). Hypothesis: the 1086 CUB-kernel launches are a **per-iteration fixed overhead** (likely one launch per BFS level or per per-cube branch, not per-cube). Host-side Python orchestration (the 6856 ms wall) doesn't show up in device-time ratios at all. Triton-fusing these kernels would claw back at most 30 ms device time.

- **s7_rank_assign (1.11x):** device time is effectively flat (33 → 37 ms, +11%) across a 4x cube-count change. Hypothesis: per-cube Python control flow dominates; the 587-call `reduce_kernel<bool>` and 106-call `index_elementwise_kernel` are being issued in a loop whose GPU work per iteration is tiny. Converting that loop to a single vectorized tensor op is the correct fix — not fusing the individual kernels.

- **__unassigned__ (1.01x):** trivial, expected (22 µs tail of profiler events outside any stage range).

### Healthy stages
s1 (5.07x), s6 (4.05x), s8 (3.53x), s2 (3.29x), s3 (3.22x) all scale between the O(R²) and O(R³) bands — they scale properly with cube count and benefit from classic GPU optimization.

---

## Caveats

1. **torch.profiler NVTX markers don't capture `nvtx.range_push/pop`.** The monkey-patch markers from Tasks 2-3 work only for nsys. Layers 1-3 fall back to matching `python_function` events by stage-entry function name (verified correct by the `__unassigned__` device-time share of 0.02%).

2. **Sub-stage CUDA events disabled per Task 5 fallback** (41.3% overhead at res=128). Sub-stage visibility within s4/s6/s7 is absent from Layers 1-3; we rely on NVTX (Layer 0) + top-30 op aggregation (Layer 1) to locate hot spots.

3. **Layer 2 class labels are heuristic.** `elementwise_kernel` is classified MMB under the assumption that CoReP has no compute-heavy elementwise ops — likely correct for index/where/add, but Nsight Compute is the ground truth. CMB = 0 is a heuristic claim, not a measurement.

4. **Layer 0 nsys CPU sampling was auto-disabled** on 116's kernel config (warnings: `CPU IP/backtrace sampling not supported, disabling`, `CPU context switch tracing not supported, disabling`). We still captured NVTX + CUDA API + OSRT; absolute wall-time at naked layer-0 (10.056 s) matches post-s7-fix baseline within 0.14 s. Consequence: no OS-level Python backtraces — Layer 2 is required to identify `.item()` call sites.

5. **torch.profiler overhead ~22% at res=256.** Layer 1 absolute times are inflated; relative percentages and per-op ordering are trusted. Stage wall-times from run2 are ~22% higher than Layer 0 nsys baseline (12.230 s vs. 10.056 s).

6. **Single resolution pair (128, 256).** Extrapolation to res=512+ is not validated; the scaling flags are based on 2 data points per stage.

---

## ROI-Ranked Next-Step Candidates

Ranked by **predicted Δ on res=256 e2e wall-time**, largest gain first. Each entry cites the evidence source.

| # | Target | Hypothesis + evidence | Predicted Δ @ res=256 | Effort | Risk |
|---|---|---|---:|---|---|
| 1 | **Eliminate the one 1048 ms `cudaDeviceSynchronize`** | Task 6 Obs 3 + CUDA API summary: 20 `cudaDeviceSynchronize` calls, total 1049.2 ms, of which 1048 ms is a single call. Other 19 calls sum to ~1 ms. One giant sync is draining a fully-queued GPU in a single flush — likely at s4→s6 or s6→s7 transition. Locate and either push work past it, split the batch, or replace with `cudaStreamSynchronize` on the relevant stream. | **−0.5 to −1.0 s** | 1-2 d | M (requires careful tracing of the s4/s6 transition) |
| 2 | **Remove `.item()` / blocking D2H leaks in s4/s6/s7 Python loops** | Task 6 Obs 2: 4022 D2H + 4128 `cudaStreamSynchronize` at ~1:1 ratio = scalar read pattern. Memcpy time alone 404.97 ms; add per-call sync latency (hundreds of µs) and the pattern costs ~0.5-1.5 s. Grep all three stages for `.item()`, `.cpu()`, `int(tensor)`, `bool(tensor)`, `if tensor_scalar:`, `.tolist()`. | **−0.5 to −1.5 s** | 2-4 d (grep-and-verify across three stages) | M |
| 3 | **Vectorize s7's per-cube rank-assignment loop** | Task 10: s7 scales 1.11x for 2x res (4x cubes). Device time effectively flat (33→37 ms) → work is per-cube Python orchestration, not GPU compute. Task 8: s7 device time is only 36.6 ms of 1810 ms wall (2.02%). Converting the per-cube loop to a vectorized tensor op (group-gather + single reduce) should collapse the bulk of the 1.8 s wall. | **−0.3 to −0.8 s** (s7 wall 1.8→~1.0 s) | 3-5 d (s7 has branching per-cube logic) | M |
| 4 | **Fuse s4's 1086-call CUB DeviceSelect+DeviceReduce into a single Triton kernel** | Task 9: 3 LNB kernels in s4 at 1086 launches each (3258 launches combined), total 8.944 ms device time. But Task 10: s4 device time is only 36.9 ms of 6856 ms wall (0.54%) — upper bound on gain is ~30 ms device + any launch overhead saved (~10-30 ms). Original Triton K1 plan (−2 to −3 s) was based on the pre-profile "s4 is GPU-bound" hypothesis, which the data refutes. | **−0.05 to −0.2 s** | 1-2 wk (was the original K1 plan) | M-H |
| 5 | **Fuse the 11 ms s6 cummax with adjacent elementwise ops** | Task 9 rank 1 kernel: 11.074 ms single call (`tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>`). Task 3 s7-fix introduced this `cummax` for the group kth-True selector. Combined with the twin 1.76 ms cumsum (rank 17, same stage s6) and the paired 1.76 ms cumsum in s7 (rank 16), these three scans account for ~14.6 ms of device time. A Triton-fused scan+gather could bring the cummax down to ~4 ms. | **−0.007 s** (GPU time only) | 1 wk | L-M |
| 6 | **(tooling)** Switch future `with_stack=True` runs to `with_stack=False` | Task 7: torch.profiler overhead = 22% at res=256, producing 3.8 GB per trace × 3 runs = 11.4 GB untrackable in git. `with_stack=False` reduces overhead 5-10x and shrinks traces ~10x. For future CI or re-profiling. | N/A (tooling) | 0.5 d | L |

### Why Triton K1 falls to #4 (not #1)

Task 6 + Task 10 decisively reframe the problem. The plan's Triton K1 targeted s4 BFS, estimated −2 to −3 s. In light of the profile:

- s4's total GPU time is only **~37 ms** at res=256 (Task 8 top-30 aggregate).
- Even if Triton eliminated 100% of s4 GPU work, the wall-time drop is bounded by 37 ms (absolute ceiling) plus any incidental reduction in launch/sync overhead.
- The other **6.8 s** of s4 wall is Python/host orchestration — not touched by a Triton kernel.
- Triton K1 was the right idea under the pre-profile mental model ("s4 is GPU-bound"). The data shows s4 is **CPU/host-bound**.

---

## Recommended next-stage plan

**Primary:** Start with **Candidate #2** (`.item()` leak elimination). Highest probability / lowest risk; directly addresses the Layer 0 signature (4022 D2H + 4128 stream-sync). Once removed, Candidates #1 and #3 scope clarifies (some of the 1.048 s sync may itself be one of these leaks).

**Investigation budget before committing to any refactor: 2 days.**

- **Day 1 (Candidate #1):** Identify the single 1048 ms `cudaDeviceSync` source. Re-run a small torch.profiler pass with `with_stack=True` (or a targeted `py-spy` sample), grep stack frames for `cuda.synchronize()` / implicit sync sites. Scope may shrink Candidate #2's upper bound if the sync turns out to be a single `.cpu()` on a large tensor.
- **Day 2 (Candidate #2):** `grep -nE '\.(item|cpu|tolist)\(\)|bool\([^)]*\)\.?' corep_fast/` in s4/s6/s7 — build a register of 30-60 suspected call sites. Prioritize by call count from `per_stage_ops_res256.csv` (Layer 1 gives you per-op call counts already).

If Days 1-2 don't uncover concrete call sites, escalate to Nsight Compute for roofline analysis — this shifts Layer 2 from "heuristic MMB" to ground-truth classification and may reveal CMB kernels we've missed.

**Only after Candidates #1-3 are resolved should Triton K1 (Candidate #4) be reconsidered.**

---

## Appendix — provenance

- git HEAD: `53dfca539a2f7d31a14059f223e9ad43cd9ece75` (branch `gpu-pipeline`)
- Date: 2026-04-17
- Raw data: `tmp/profile_deep/results/`
- Chrome trace files (~3.8 GB × 3 at res=256, ~1.1 GB × 3 at res=128): local-only, not git-tracked (total ~15 GB in `tmp/`)
- Analysis scripts: `tmp/profile_deep/analyze_layer1_per_stage.py`, `analyze_layer2_top20_kernels.py`, `analyze_layer3_scaling.py`
- Run wrappers: `tmp/run_layer0_nsys.sh`, `tmp/run_layer1_torch.sh`, `tmp/run_layer3_torch.sh`, `tmp/run_smoke_overhead.sh`, `tmp/smoke_driver_res64.sh`
- Spec: `docs/superpowers/specs/2026-04-16-corep-deep-profiling-design.md`
- Plan: `docs/superpowers/plans/2026-04-16-corep-deep-profiling-implementation.md`
- Phase 2 baseline: `my-docs/20260416-pre-triton-final-pass-results.md`
- s7 fix: `my-docs/20260417-s7-phase1-gpu-bijective-bug-fix.md` (commit `83d230d`)
- Layer 0 observations: `tmp/profile_deep/results/layer0_observations.md`
- Per-stage ops: `tmp/profile_deep/results/per_stage_ops_res256.csv`, `per_stage_ops_res128.csv`
- Top-20 kernels: `tmp/profile_deep/results/top20_kernels_res256.csv`
- Scaling table: `tmp/profile_deep/results/scaling_table.csv`
