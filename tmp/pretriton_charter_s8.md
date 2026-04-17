# Pre-Triton Analysis (READ-ONLY) — Shared Context

## Your Task

You are an `Explore` subagent assigned to deeply analyze ONE CoReP-fast pipeline stage and produce a markdown report. **You will NOT modify any code.** You only read, profile-data analyze, and write a single markdown report file.

## Working Directory

`/mnt/novita2/siyuan/workspace/TRELLIS.2`

## Project Context

CoReP-fast is the GPU-accelerated rewrite of TRELLIS.2's voxelization pipeline. After M1 + M2 milestones, on H100 res=256 icosphere subdiv=3:
- e2e: 17.92s vs custom baseline 141.05s = **7.87x speedup**
- Remaining bottlenecks: s4 (24%), s6 (14%), s7 (29%), s8 (31%)

Goal: identify all torch / vectorization / algorithm-layer optimizations remaining BEFORE writing Triton kernels. Push every CPU/Python residual section either to GPU torch or document why it is Triton-only.

## Reference Documents (READ THESE)

- Spec for this evaluation: `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`
- M2 spec: `docs/superpowers/specs/2026-04-16-corep-fast-m2-full-efficiency.md`
- M1 spec: `docs/superpowers/specs/2026-04-16-corep-full-vectorization-design.md`
- s8 historical analysis: `my-docs/20260415-corep-fast-stage2-analysis.md`
- e2e profiling history: `my-docs/20260415-e2e-profiling-acceleration-analysis.md` (esp. appendix D/E/F)
- Deep-review: `my-docs/20260415-deep-review-torch-vectorization.md`
- Repo overview: `corep_fast/README.md`
- Stage 2 v2 vectorization spec (s8 specifically): `docs/superpowers/specs/2026-04-15-corep-fast-stage2-v2-torch-vectorization.md`

## Baseline Profile Data

Read both:
- `tmp/pretriton_baseline_res128.json`
- `tmp/pretriton_baseline_res256.json`

These are the freshly re-measured baselines. Use them as the "current time" for your stage.

## Output Requirements

Write a SINGLE markdown file: `tmp/pretriton_s8_analysis.md`

The file MUST follow this exact structure (sections A through E). Total length < 300 lines.

```
# S8 Pre-Triton Analysis

## A. 当前实现快照（针对 corep_fast/stages/s8_collapse.py 当前 HEAD）
- 关键函数 + 行号
- 已 GPU 化部分（标注 M1/M2 提交）vs 仍 CPU 部分
- profile 数据来源（commit + 数据文件路径）

## B. 残留 CPU/Python 段的算法本质
对每个 CPU 段：
- 算法描述（伪代码或文字 < 30 行）
- 为什么 M1/M2 把它留在 CPU
- 数据规模 res=128 / 256（item 数量、平均/最大单元工作量）
- 单元算法复杂度（O 表示 + 常数估算）

## C. Torch 化可行性矩阵
| 段 | 提议的 torch 方案 | 风险（精确性 / 内存峰值 / 实现复杂度） | 预期收益 (s, res=256) | 推荐 |
|---|---|---|---|---|
推荐取值: do / skip / triton-only

## D. 数据 contract 检查
- 输入张量: shape / dtype / 来源 stage
- 输出张量: shape / dtype / 下游 consumer
- 与其它 stage 的 contract 是否需要变动
- 与 custom/ baseline 在 V/F 等价上的依赖路径

## E. Triton handoff
若该 stage 的某段不做 torch，剩余给 Triton 的 kernel 应该长什么样：
- 提议的 kernel 输入张量列表（shape / dtype）
- 提议的 grid / block / shared mem 布局
- 估算的 GPU 工作量（FLOPs + bytes）
- 与 Phase 2 torch 优化的接口契合度
```

## Hard Rules

1. **Do NOT modify any source files** — you are read-only
2. **Do NOT run any code** — analysis only (you may read existing profile JSONs but not invoke profile scripts)
3. **Do NOT speculate** — every claim about current code must cite file:line
4. **Do NOT exceed 300 lines** in your output report
5. **Use Chinese-simplified** for the analysis (matches user preference)

---

## Your Stage: S8 (collapse / mesh decode)

### File

`corep_fast/stages/s8_collapse.py` (2200 lines)

### Functions to focus on

- `decode_from_cubebatch` (line 141) — public entry
- `_cubebatch_to_dicts` (line 181) — legacy dict conversion (M2 P1 may have made this dead-code in hot path)
- `_cubebatch_to_tensors_direct` (line 534) — M2 P1 direct path
- `process_geometry_vectorized` (line 830) — main vectorized geometry
- `_process_shared_edges_torch` (line 1143) — orchestrates vectorized path + Step E candidate fallback
- `_build_grids_from_cube_map` (line 1316) — legacy grid construction (may be dead in default mode)
- `_build_grids_from_tensors` (line 1420) — M2 P3 grid construction (default)
- `_process_shared_edges_from_tensors` (line 1480)
- Step E candidate fallback (line 1222 onward) with MP `_Pool` (line 1279) — the main remaining Python loop

### Specific Investigation Questions

1. The 4-cube candidate fallback (~26% of edges, ~80% of triangles): the historical Stage 2 v2 analysis (`my-docs/20260415-corep-fast-stage2-analysis.md` §3.3) found only **0.05%** of 4-cube edges actually diverge from the GPU path. Why are we still routing all 4-cube edges to fallback? Can a tighter predicate (e.g., `4-cube AND any neighbor has num_loops ≥ 2`) restrict fallback to <2% while keeping correctness? Quote exact line numbers in s8_collapse.py for the predicate.
2. The encoding mismatch (`_EDGE_OFFSET_TABLE` vs `custom/collapse.py::get_local_edge`) called out in the Stage 2 v2 analysis §4.1: is this still present in current code? If so, can a one-time encoding alignment + a hybrid GPU-99.95%/Python-0.05% approach close it? Cite where the mismatch lives.
3. `_process_shared_edges_from_tensors` (line 1480): does it bypass `_build_grids_from_tensors` for the GPU path, or is the M2 P3 work still re-built per call?
4. `_cubebatch_to_dicts` (line 181): is it still called when `USE_DIRECT_TENSOR_S8=1`? If so where? If not, is the dead code worth removing?
5. The `_weld_and_dedup` vertex welding (Stage 1 milestone, ~3.5s historical): is it still on the hot path? Does it use `torch.unique` end-to-end?
6. Per-element `.item()` syncs anywhere in the s8 hot path?

### Output File

`tmp/pretriton_s8_analysis.md`

### Stage-specific reading

- The Stage 2 v2 analysis is the most important ref doc here
- M2 spec section 1 (s8 P1 direct path)
- M2 spec section 3 (s8 P3 candidate)
- `corep_fast/config.py` for the feature flags

### Thoroughness

Be **very thorough**: trace the data flow from `decode_from_cubebatch` entry through the direct-tensor path to Step E fallback. For the 0.05% pathological edge claim, sketch the exact predicate that would distinguish them. Cite line numbers for every claim.
