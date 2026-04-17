# Layer 0 — Nsight Systems observations (res=256)

## Run info
- git sha: `36a1e9d57e2cb4cc3f7bf8b4ad72e1e2280860c6`
- Date: `2026-04-17T02:02:49Z`
- Hardware: 116 node, GPU 0 (H100 80GB HBM3)
- Command: `tmp/run_layer0_nsys.sh`
- nsys version: `NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0`

## Pipeline outcome (naked run, layer 0)
- Driver's printed `[done] e2e = 10.056s` (this is the measure-run walltime captured by the driver itself).
- nsys overhead proved **much smaller than expected** here: the driver's printed e2e (10.056 s) sits inside the nsys trace, so absolute times are usable. The NVTX "measure" pass sums to ~9.79 s against the 10.056 s printed e2e — i.e. ~260 ms of Python-level glue outside the 7 NVTX ranges.
- `nsys_res256_full.nsys-rep` = 2.6 MB (surprisingly small, clearly well below the 100 MB threshold from the plan).

## NVTX stage ranges (from NVTX Range Summary)

nsys captured 2 instances per range (warmup + measure). The min column = measure (stable) run, the max = warmup.

| Stage | Measure ms | Warmup ms | Avg ms | # inst | Notes |
|---|---:|---:|---:|---:|---|
| s1_voxelize | 18.71 | 317.95 | 168.33 | 2 | warmup pays for CUDA context/first-use |
| s2_components | 10.56 | 109.70 | 60.13 | 2 | warmup likely triggers cuGraph/triton init |
| s3_edge_weights | 1.27 | 1.70 | 1.49 | 2 | tiny — essentially free |
| s4_face_point | 5815.24 | 6059.60 | 5937.42 | 2 | **dominant — 58.6 % of NVTX time** |
| s6_collapse | 2442.29 | 2462.80 | 2452.55 | 2 | **24.2 % of NVTX time** |
| s7_rank_assign | 1409.98 | 1423.62 | 1416.80 | 2 | **14.0 % of NVTX time** |
| s8_decode | 96.31 | 100.46 | 98.38 | 2 | 1 % |

Sum of measure-run stages = **9 794 ms** (vs. 10 056 ms e2e → ~262 ms glue outside NVTX ranges, consistent with the Python driver's pre/post-stage book-keeping).

Note: CoReP pipeline has no s5 stage — numbering goes s1-s4, s6-s8 by project convention (s5 historical; absent from corep_fast). All 7 stages captured.

## Top 10 CUDA kernels by total time (from CUDA Kernel Summary, both runs)

| Rank | Name (shortened) | Total ms | Count | Avg µs | % |
|---:|---|---:|---:|---:|---:|
| 1 | `tensor_kernel_scan_innermost_dim_with_indices<long, greater_equal>` | 23.58 | 4 | 5 894 | 9.1 |
| 2 | `index_elementwise_kernel` (index_kernel<OpaqueType<4>>) | 19.54 | 1 004 | 19.5 | 7.5 |
| 3 | `reduce_kernel<bool, or_kernel_cuda>` | 15.33 | 1 200 | 12.8 | 5.9 |
| 4 | `DeviceSelectSweepKernel<long>` (cub nonzero scan) | 11.51 | 2 522 | 4.6 | 4.4 |
| 5 | `index_elementwise_kernel<OpaqueType<8>>` (8-byte index) | 7.94 | 742 | 10.7 | 3.1 |
| 6 | `tensor_kernel_scan_innermost_dim<long, plus>` (cumsum long) | 7.05 | 4 | 1 762 | 2.7 |
| 7 | `vectorized_elementwise_kernel<CUDAFunctor_add<long>>` | 6.89 | 58 | 118.8 | 2.7 |
| 8 | `DeviceReduceKernel<unsigned long long, plus>` (cub nonzero count) | 6.73 | 2 498 | 2.7 | 2.6 |
| 9 | `elementwise_kernel<CUDAFunctor_add<float>>` | 6.45 | 494 | 13.1 | 2.5 |
| 10 | `elementwise_kernel<where_kernel_impl>` | 6.45 | 32 | 201.7 | 2.5 |

Total kernel time across all 23 836 launches = **259.7 ms** (both runs combined → ~130 ms per run).

## Memory ops

| Op | Total | Count | Avg per call |
|---|---:|---:|---:|
| D2H | **1 212.6 MB** (**404.97 ms / 93.1 % of memcpy time**) | **4 022** | 0.30 MB |
| D2D | 546.1 MB (0.50 ms) | 50 | 10.9 MB |
| H2D | 360.8 MB (28.34 ms) | 90 | 4.0 MB |
| memset | 46.4 MB (1.15 ms) | 1 336 | 0.04 MB |

Standout: **4 022 tiny D2H transfers averaging 301 KB each** dominate all memcpy time. Each implies a `.cpu()` / `.item()` / `.tolist()` round trip = implicit CPU↔GPU sync.

## CUDA API summary highlights

| API | Calls | Total ms | % |
|---|---:|---:|---:|
| `cudaDeviceSynchronize` | **20** | **1 049.2** | **45.3** |
| `cudaMemcpyAsync` | 4 162 | 469.9 | 20.3 |
| `cudaLaunchKernel` | **23 836** | 457.1 | 19.7 |
| `cudaMalloc` | 147 | 176.6 | 7.6 |
| `cudaStreamSynchronize` | **4 128** | 144.7 | 6.2 |
| `cudaFree` | 73 | 8.94 | 0.4 |
| `cudaHostAlloc` | 3 | 4.11 | 0.2 |
| `cudaMemsetAsync` | 1 336 | 2.67 | 0.1 |

Surprise: one single `cudaDeviceSynchronize` consumes **1 048 ms** (the `Max` column) — almost the entire 1.05 s total. The other 19 `cudaDeviceSynchronize` calls sum to ~1 ms. That one call alone is a **full-second CPU stall**.

## Stream usage

From `cuda_gpu_trace`: every one of the 29 334 GPU ops (kernels + memcpy + memset) is on a **single stream (stream ID = 7, the default stream)**. **Zero stream-level parallelism** in the entire pipeline.

## Three concrete observations (DoD #2)

**1. GPU is idle ~96 % of wall time — kernels do only ~3 % of the work.**
Sum of CUDA kernel execution time for both warmup+measure = 259.7 ms; sum of all memcpy/memset = 435.0 ms (D2H 404.97 + H2D 28.34 + D2D 0.50 + memset 1.15) → total GPU-occupied time ≈ 694.7 ms across two runs, i.e. **~347.3 ms per run** against a 10 056 ms e2e → **GPU utilisation ≈ 3.5 %**. Driver CPU logic, CUB index gymnastics, and D2H syncs dominate; the H100 spends the other 9.7 s per run doing nothing. Layer 1/2 (torch.profiler) must pinpoint where the CPU spins — strongest suspects are s4_face_point (~5.8 s) and s6_collapse (~2.4 s).

**2. 4 022 D2H transfers per pair-of-runs = one sync every ~2.5 ms — likely `.item()` / `.cpu()` leaks.**
The CUDA Memory Ops Summary shows **4 022 D2H memcpies averaging 301 KB each** (1.21 GB total, 93.1 % of memcpy time) plus **4 128 `cudaStreamSynchronize` calls**. That 4128 ≈ 4022 ratio is too clean to be coincidence: almost every D2H has a matching stream-sync before/after it, i.e. these are **blocking scalar reads** (`t.item()`, Python `if scalar_tensor:` comparisons, `.tolist()`), not bulk transfers. Each forces a round-trip CPU-GPU sync. A Phase-3 optimisation pass should hunt these in s4/s6/s7 — candidates include bitmap counts, rank min/max, and per-component loop guards.

**3. Entire pipeline runs on a single default stream (stream 7) with one ~1.0 s `cudaDeviceSynchronize`.**
`cuda_gpu_trace` confirms all 29 334 GPU ops live on stream 7 — **zero cross-stream concurrency**. (Data source: `nsys_res256_full.sqlite` CUPTI activity tables — per-stream breakdown not exposed in the per-report CSVs.) On top of that, one `cudaDeviceSynchronize` call costs **1 048 ms** by itself (the other 19 calls sum to ~1 ms), meaning one specific host-side barrier is draining a fully-queued GPU in a single flush. This is a smoking gun for a batched-then-blocked pattern (likely end of s4_face_point or start of s6_collapse). Two concrete wins are now visible: (a) break the default-stream dependency for independent work (e.g. edge-weight construction vs. face-point picking), and (b) locate the one giant sync point and either push work past it or pipeline it.

## Caveats
- nsys overhead proved far lower than the 2-3× in the plan (driver's `[done] e2e = 10.056 s` vs. expected ~20-30 s with overhead). The `.nsys-rep` is only 2.6 MB because of `osrt,cuda,nvtx` trace + no CPU sampling (warning messages in runlog confirm `CPU IP/backtrace sampling not supported, disabling` and `CPU context switch tracing not supported, disabling`). This means this trace has **no OS-level backtrace data** — we can't blame specific Python callsites from this alone, Layer 2 (torch.profiler) is still required.
- NVTX "Instances = 2" because the driver runs warmup + measure; "Min" ≈ measure-run, "Max" ≈ warmup-run.
- Sub-stage CUDA events inside s4/s6/s7 are NOT captured (disabled per Task 5 fallback, commit `3af0d99`). Layer 1 (torch.profiler, res=64) and Layer 2 (torch.profiler sub-stage, res=256) will provide finer detail there.
