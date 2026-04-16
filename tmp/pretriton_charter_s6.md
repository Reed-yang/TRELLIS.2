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

Write a SINGLE markdown file: `tmp/pretriton_s6_analysis.md`

The file MUST follow this exact structure (sections A through E). Total length < 300 lines.

```
# S6 Pre-Triton Analysis

## A. 当前实现快照（针对 corep_fast/stages/s6_collapse.py 当前 HEAD）
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

## Your Stage: S6 (collapse / loop extraction)

### File

`corep_fast/stages/s6_collapse.py` (655 lines)

### Functions to focus on

- `s6_collapse` (line 502) — entry point
- `_collapse_fast` (line 72) — fast-path (face_weights all zero)
- `_collapse_with_uturns` (line 296) — slow-path (with U-turn enumeration)
- `_collapse_with_uturns_tracked` (line 379) — slow-path with assignment tracking
- `_s6_worker` (line 476) — MP worker
- per-cube loop at line 580 + MP launch at line 595
- `_get_canonical_loop` (line 162) and `_get_canonical_solution` (line 181)

### Specific Investigation Questions

1. The fast-path `_collapse_fast`: how often does it hit (% of cubes)? Is it pure GPU or Python? If Python, is there a vectorizable batched version of the closed-form loop trace?
2. The slow-path `_collapse_with_uturns`: it does (u1, u2, u3) cartesian product enumeration per cube with budget 100K. What's the actual distribution of product sizes? p50/p99/max at res=256? If most are tiny (≤32), can we batch all slow-path cubes into a (N_slow, MAX_PRODUCT, 3) tensor and do one GPU-side trace?
3. The graph-trace within slow-path uses Hierholzer-style. Can this be expressed as parallel union-find + canonicalization, where canonicalization is the only sequential step?
4. The MP `_s6_worker`: is the per-task payload large (i.e., serialization overhead is real)? Look at `chunking.py` if used.
5. `uturn_assignment` output (M1 contract): is the (N, 12, 3) tensor materialized on GPU or rebuilt from MP worker outputs?
6. Are there per-element `.item()` syncs in the orchestration code (lines 502-655)?

### Output File

`tmp/pretriton_s6_analysis.md`

### Stage-specific reading

- M1 spec section 4.4 (s6 redesign)
- `corep_fast/parallel/` (worker pool harness)

### Thoroughness

Be **very thorough**: trace the data flow from `s6_collapse` entry to CSR output. Cite line numbers for every claim. For Python loops, count how many iterations they do at res=128 / res=256.
