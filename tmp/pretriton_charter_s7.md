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

Write a SINGLE markdown file: `tmp/pretriton_s7_analysis.md`

The file MUST follow this exact structure (sections A through E). Total length < 300 lines.

```
# S7 Pre-Triton Analysis

## A. 当前实现快照（针对 corep_fast/stages/s7_rank_assign.py 当前 HEAD）
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

## Your Stage: S7 (rank_assign)

### File

`corep_fast/stages/s7_rank_assign.py` (752 lines)

### Functions to focus on

- `s7_rank_assign` (line 490) — entry point
- `_s7_rank_worker` (line 427) — Phase 1 rank tracing MP worker
- `_match_loops_to_ranks` (line 259) — cyclic alignment
- `_trace_with_ranks_fast` (line 70) — fast path
- `_trace_with_ranks_uturn_assignment` (line 153) — slow path consuming s6's uturn_assignment
- `_compute_centroids` (line 314) — GPU centroid interpolation
- `_hungarian_match` (line 364) — Hungarian wrapper
- `_hungarian_worker` (line 462) — MP worker (currently serialized per commit `24ad133`)
- `_extract_s6_loops` (line 402) and `_extract_component_points` (line 414) — per-cube CSR extraction

### Specific Investigation Questions

1. Phase 1 (rank tracing) MP loop: this is the largest s7 bottleneck. The trace per-cube is sequential graph walk on a small (≤12 edges) graph. Is there a batched GPU formulation (parallel BFS layer-by-layer over all cubes) that avoids per-cube Python?
2. `_extract_s6_loops` and `_extract_component_points` per cube: are they per-cube Python loops? If so, can they be replaced by vectorized CSR gathers?
3. `_match_loops_to_ranks`: cyclic alignment per loop is O(K²) per loop. Can this be batched over L loops as a (L, K, K) tensor matmul/comparison?
4. `_compute_centroids`: confirmed GPU. Are there any .item() or .cpu() syncs?
5. Hungarian: commit `24ad133` made it serial scipy. M1 spec said "inherently sequential" but is there a batched GPU Sinkhorn / auction substitute? Quote real ROI estimate (cost matrix is small per cube).
6. The Phase 1 rank tracing for fast-path cubes (uturn_assignment == -1) vs slow-path: are they done in different code branches? If so, is the fast-path branch GPU-feasible while slow-path remains MP?

### Output File

`tmp/pretriton_s7_analysis.md`

### Stage-specific reading

- M1 spec section 4.5 (s7 redesign)
- M1 spec on the claim "graph traversal is inherently sequential" — challenge it
- Recent commits 24ad133, b1476ca, 4e2d8d9

### Thoroughness

Be **very thorough**: read each of Phase 1 / Phase 2 / Phase 3 of s7 in full. Cite line numbers. For each "could be batched" claim, sketch the actual GPU tensor layout that would enable it.
