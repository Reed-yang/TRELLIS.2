# CoReP Pre-Triton Final-Pass Design Spec

> 日期: 2026-04-16
> 分支: `gpu-pipeline`
> 硬件目标: 119 节点 H100 80GB HBM3, GPU 0 / GPU 1 闲置
> 目标 mesh: icosphere subdiv=3, res=128 + res=256
> 前置: M2 已完成（res=256 e2e 17.92s = 7.87x vs custom 141.05s）
> 终点: 进入 `corep_triton/` 实施前的最后一次 torch / 算法层评估 + 实施

---

## 0. TL;DR

M2 完成后 `corep_fast/` 在 res=256 已达 7.87x speedup，但仍有 4 个 stage 残留 CPU/Python 段:

| 残留 | 当前 (s) | 占 e2e | 算法本质 |
|---|---:|---:|---|
| **s8** candidate fallback | 5.61 | 31% | 4-cube edges (~26%) 走 Python 几何处理 (~80% triangles) |
| **s7** Phase 1 rank tracing | 5.13 | 29% | per-cube edge-rank 序列重 trace + cyclic align (CPU MP) |
| **s4** BFS U-turn count | 4.27 | 24% | per-(cube, facet) 段图 BFS (Stage C, CPU MP) |
| **s6** slow-path | 2.52 | 14% | (u1,u2,u3) cartesian product + 图 trace (CPU MP) |

本 spec 描述在进入 Triton 之前对这 4 段做一次彻底的 torch / 算法层评估和实施。
策略：**case-by-case 权衡** "exact V/F match" 与 "拓扑等价 + 大幅加速"；不强求拓扑等价，但只在 ROI 明显且风险可控时才放弃精确匹配。

执行结构：3 个 phase
1. **Phase 1（~4h）** — 4 个 explore subagent 并行分析 4 stages，产出 4 份 markdown
2. **Phase 2（~1-2 day）** — 2 路并行实施（GPU 0/1 各 1 个 worktree），分两批
3. **Phase 3（~半天）** — 合并 + e2e profile + Triton handoff 文档

预期收益（保守）：e2e 17.92s → 11-13s = **11-13x vs custom**；为后续 Triton 留出**纯 GPU compute / 内存带宽**层面的 bottleneck。

---

## 1. Motivation & Goal

### 1.1 当前状态

`my-docs/20260415-e2e-profiling-acceleration-analysis.md` 附录 F 记录的 M2 终点 (res=256, 119 H100):

```
custom 141.05s  ─→  M1 30.13s (4.68x)  ─→  M2 17.92s (7.87x)  ─→  ?
                                              │
                                         本 spec 入口
```

M2 的 P1/P2/P3 已经把"data layout 双向转换""GPU pair expansion""cube_map 重建"3 个最容易的大块吃掉。剩下的 4 段都涉及**图遍历 / 序列追踪 / 组合枚举**，M1 的设计哲学曾把它们划归 "graph/combinatorial residency: CPU multiprocessing"。

但用户给出明确指示：在进入 Triton 之前，再做一次彻底评估和实施，确保所有可在 torch / 算法层做的优化都做了，**把 bottleneck 留在必须 Triton 才能解决的层面**。

### 1.2 目标

进入 `corep_triton/` 实施时，残留 bottleneck 应满足：
- **无 per-element Python 循环**：不再有 `for ci in range(N)` over cubes/edges/loops（直接 for 或 MP `Pool` 都不允许；常数循环 ≤ ~12 可保留）
- **数据 contract 100% GPU tensor**：stage 间 hand-off 全 GPU tensor；no dict / no list / no numpy intermediate（除了 host→device 边界与必要 scalar sync）
- **残留瓶颈**只剩三类，全是 Triton 拿手领域：
  - (a) 单 stage 内多个细粒度 kernel launch（fusion 候选）
  - (b) HBM 带宽瓶颈（中间张量过大，应在寄存器/SMEM 内处理）
  - (c) 算术密度不足（warp idle，需要 register tiling）

### 1.3 非目标

- 不动 `custom/`（仍是 A/B 真值）
- 不动 s1 / s2 / s3 (合计 0.27s @ res=256，已极优)
- 不写 Triton kernel（spec 末尾输出的是 Triton handoff，不是实现）
- 不在本 spec 内调整 mesh 输入格式或 pipeline 接口

---

## 2. Phase 1: 4-subagent 并行分析

### 2.1 共享 Charter

每个 explore subagent 接到自包含的 prompt，**只读、不写**，产出 < 300 行 markdown 到 `tmp/pretriton_<stage>_analysis.md`。报告强制结构：

```
# <Stage> Pre-Triton Analysis

## A. 当前实现快照
针对 corep_fast/stages/<file>.py 当前 HEAD：
- 关键函数 + 行号
- 已 GPU 化部分（M1/M2 提交）vs 仍 CPU 部分
- profile 数据来源（标 commit + 数据文件路径）

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
- 提议的 kernel 输入张量列表
- 提议的 grid / block / shared mem 布局
- 估算的 GPU 工作量（FLOPs + bytes）
- 与 Phase 2 torch 优化的接口契合度
```

### 2.2 4 个 subagent 的具体任务

**所有 subagent 共享的环境/上下文** (写在 prompt 顶部):
- 工作目录: `/mnt/novita2/siyuan/workspace/TRELLIS.2`
- 当前分支: `gpu-pipeline`
- 不要修改任何文件
- 当前 e2e profile @ res=256: 17.92s (附录 F)
- 关键参考文档:
  - `docs/superpowers/specs/2026-04-15-corep-fast-stage2-v2-torch-vectorization.md`（s8 vectorization 详细，含 0.05% pathological edge 分析）
  - `docs/superpowers/specs/2026-04-16-corep-fast-m2-full-efficiency.md` (M2 spec)
  - `my-docs/20260415-corep-fast-stage2-analysis.md`（s8 历史分析）
  - `my-docs/20260415-e2e-profiling-acceleration-analysis.md` 附录 D/E/F (profile 数据)
  - `my-docs/20260415-deep-review-torch-vectorization.md`

**Subagent S4** (`Explore` agent, very thorough):
- 文件: `corep_fast/stages/s4_face_point.py` (1197 行)
- 重点函数: `_compute_face_weights_gpu` (line 1119), `_compute_face_weights_mp` + `_fw_worker_indexed` (line 171), 以及 M2 spec 提议的 `_uturn_worker`（实际命名以代码为准）
- 关注: Stage C 的 BFS U-turn 算法是否能 GPU 化（已知挑战：每 (cube, facet) 段数变长 + BFS 串行）
- 也分析: closest_point_on_mesh 调用是否还有 GPU launch overhead 优化空间
- 把 `_compute_face_weights_gpu` 内部各子段（pair expansion / plane-tri / clip / compact / BFS / scatter）逐段标 GPU/CPU 占比

**Subagent S6** (`Explore` agent, very thorough):
- 文件: `corep_fast/stages/s6_collapse.py` (655 行)
- 重点函数: `_collapse_with_uturns`, `_collapse_with_uturns_tracked`, `_s6_worker` (line 476), per-cube loop @ line 580
- 关注: slow-path 的 (u1,u2,u3) cartesian product 是否能 GPU 化（M1 设 budget 100K，要看实际单 cube max product 多大）
- fast-path 是否还有可向量化的部分

**Subagent S7** (`Explore` agent, very thorough):
- 文件: `corep_fast/stages/s7_rank_assign.py` (752 行)
- 重点函数: `_s7_rank_worker` (line 427), `_match_loops_to_ranks` (line 259), `_trace_with_ranks_fast` (line 70)
- 关注:
  - Phase 1 rank tracing 的图遍历能否 batch GPU 化（核心是 Eulerian-loop trace per cube）
  - Hungarian Phase 3 已经串行 scipy（24ad133），是否还有 GPU 化路径（如 batched Sinkhorn）
- M1 spec 称 "inherently sequential" 要被挑战

**Subagent S8** (`Explore` agent, very thorough):
- 文件: `corep_fast/stages/s8_collapse.py` (2200 行)
- 重点函数: `_process_shared_edges_torch` (line 1143), `process_geometry_vectorized` (line 830), `_build_grids_from_tensors` (line 1420), Step E (line 1222) candidate fallback
- 关注:
  - 4-cube candidate fallback 路径（占 ~80% triangles）能否完全 vectorize
  - 0.05% pathological edges 的处理能否用 hybrid approach（98% GPU 路径 + 2% CPU 路径）
  - encoding mismatch (`_EDGE_OFFSET_TABLE` vs `get_local_edge`) 的根因和解法

### 2.3 Phase 1 时长 & 资源

- 4 个 subagent 完全独立、纯只读，可同时启动
- 不占 GPU；subagent 执行预计 60-120 分钟
- 完成后 4 份 markdown 落到 `tmp/pretriton_<s4|s6|s7|s8>_analysis.md`
- 主 Claude 阅读汇总 + 写 master plan 另需 ~1-2h，所以 Phase 1 端到端 ~3-4h

### 2.4 Phase 1 master plan 输出

主 Claude（不是 subagent）阅读 4 份 markdown，写一份 Phase 2 master plan: `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md`，包含：
1. **现状基线表**（res=128 + res=256，重新跑 119 GPU 0 确认）
2. **bottleneck 矩阵汇总**（4 stage × 残留段，标 do / skip / triton）
3. **要做的 torch 优化清单**（按 ROI 排序，每项标 stage / 预期收益 / 风险 / 文件 / 依赖）
4. **要给 Triton 的 handoff 清单**
5. **Phase 2 worktree 分配 + 两批次执行顺序**

---

## 3. Phase 2: 2-worktree 并行实施

### 3.1 Worktree 策略

使用 `superpowers:using-git-worktrees` skill：
- 每个 worktree 一个 stage 的 torch 优化工作
- 每个 worktree 在 `gpu-pipeline` 分支基础上拉新分支 `pre-triton/<stage>`
- 主 Claude 协调启动 / 验证 / 合并；可委派 subagent 在 worktree 内执行实施细节

### 3.2 GPU 资源分配

119 节点 GPU 0 + GPU 1 闲置：
- 每 worktree 进程通过 `CUDA_VISIBLE_DEVICES` 锁定一张卡
- A/B test 和 stage profile 在锁定的卡上跑
- e2e profile（Phase 3）在 GPU 0 单独跑

### 3.3 两批次执行顺序

由 Phase 1 ROI 排序决定，**预估**（最终以 master plan 为准）：

**第 1 批**（GPU 0: stage X | GPU 1: stage Y）— 最高 ROI 2 个
- 候选：s8 candidate fallback（最大块）+ s4 BFS（次大块）

**第 2 批**（GPU 0: stage Z | GPU 1: stage W）— 剩余 2 个
- 候选：s7 rank tracing + s6 slow-path

如果某 stage 的 torch 化在 Phase 1 被判定为 "triton-only"（无 torch 收益），跳过该批次空位，提前进入 Phase 3。

### 3.4 每个 worktree 内部流程

强制顺序（每个 worktree 都遵守）:
1. `git status`（确认 worktree 干净）
2. **A/B test 第一**：写一个能 reproduce 当前 baseline V/F 的 test，作为 commit guard
3. **小步实施**：每次只动 1 个函数，跑 unit test + A/B test
4. 中途 stage profile（仅该 stage 时间，跑 GPU 隔离）
5. 完成后 commit 到 `pre-triton/<stage>` 分支
6. 主 Claude 检查 commit + 验证 + 移到下一 stage

### 3.5 失败回退

每个 stage 完成后单独 profile 验证：
- 如果**单 stage 收益 < 预测的 50%** 或**A/B test 失败**：revert 该 stage 改动，记录到 master plan 的"实测发现"段
- Phase 2 不强制 100% 完成 4 个 stage；只要把"理论可做"的都试过

---

## 4. Phase 3: 合并 + 收尾

### 4.1 合并

- 4 个 `pre-triton/<stage>` 分支按完成顺序 merge 到 `pre-triton/all` 分支
- 合并冲突大概率出现在: pipeline.py orchestration; containers.py 的 CubeBatch 字段
- 每次 merge 后跑 215 个 test 确认无回归

### 4.2 e2e profile（GPU 0）

- 复用 `tmp/e2e_profile_m2.py`，加 `pre-triton` 列
- 跑 res=128 + res=256，每个 5 次取稳态
- 对比表落到 `my-docs/20260416-pre-triton-final-pass-results.md`

### 4.3 Triton handoff 文档

落 `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md`，给后续 Triton 实施工作输入：

```
# CoReP Triton Kernel Handoff Spec

## 1. Torch-stage 后的最终 stage breakdown
（实测数字 + 占比）

## 2. 还需要 Triton 的 bottleneck 列表
对每项：
- 名称、所在 stage、当前时间
- 输入张量 spec（shape / dtype / device / 来源）
- 输出张量 spec
- 算法描述（数学层面）
- 推荐 Triton kernel 设计（grid / block / shared mem / occupancy 估算）
- 预期 Triton 后时间

## 3. 后续 Triton 实施的优先级建议
按 ROI 排序，标 risk
```

---

## 5. 验收标准

**Hard requirement**：
1. Phase 1 产出 4 份分析 + 1 份 master plan
2. Phase 2 中 master plan 标记 "do" 的优化，每个有 commit 落地或在 master plan 里有"放弃理由"
3. Phase 3 e2e profile 实测 + Triton handoff 文档
4. 215 unit test 全部通过
5. A/B vs custom baseline 在 res=128/256 V/F 完全等价（拓扑层面）

**Soft target**:
- e2e @ res=256 ≤ 13s（vs M2 17.92s = 1.4x，vs custom = 11x）
- 所有 stage 不再有 per-element Python 循环（含 MP）
- master plan 明确每个剩余 bottleneck 是否 Triton-only

**Stretch target**:
- e2e @ res=256 ≤ 10s（14x vs custom）
- 完全清掉 MP（仅保留 GPU async 数据流）

---

## 6. 风险

| # | 风险 | 缓解 |
|---|---|---|
| 1 | Subagent 分析不一致或信息冗余 | 4 个 prompt 共享同一份 charter 模板；汇总时主 Claude 人工去重 |
| 2 | 2 个并行 worktree 改动到同一文件（如 s8 candidate 同时动 s4 也改的 helpers） | Phase 1 必须辨明 cross-stage contract；stage 分配按文件分离；如有交集放第 2 批串行 |
| 3 | 收益估算错误 | 每个 stage 完成后单独 profile，不达标可 revert |
| 4 | 119 GPU 0/1 在我们做完前被占用 | Phase 1 不依赖 GPU；Phase 2 1 GPU 即可串行跑；Phase 3 可降级到 res=128 |
| 5 | 4-cube fallback 0.05% pathological edges 在 GPU 化后暴露（A/B 失败） | 保留 hybrid: GPU 处理 99.95% + Python 处理 0.05%；用 master plan 的"已知 known-bad mesh predicate" 标记 |
| 6 | s7 Hungarian 已经串行了 scipy，没有 torch 替代 | Master plan 应明确 "torch 不动 Hungarian"；只看 Phase 1 rank tracing |
| 7 | 完成后总收益不及目标 | 仍是有效输出：把"已无 torch 化空间"的事实文档化，进入 Triton 时不浪费时间 |

---

## 7. 执行依赖

```
[Phase 1]
  └─ 4 explore subagent 并行 (~2h)
       └─ 4 份 tmp/pretriton_*_analysis.md
            └─ 主 Claude 汇总 → master plan
                 │
                 ▼
[Phase 2 batch 1]                [Phase 2 batch 2]
  GPU 0: stage X (worktree A)      GPU 0: stage Z (worktree C)
  GPU 1: stage Y (worktree B)      GPU 1: stage W (worktree D)
       │                                │
       ▼                                ▼
  per-stage A/B + profile           per-stage A/B + profile
       │                                │
       └────────────┬───────────────────┘
                    ▼
              [Phase 3]
                merge → e2e profile → handoff doc
                    │
                    ▼
              进入 corep_triton/
```

---

## 8. 文件输出清单

| 路径 | 类型 | 内容 |
|---|---|---|
| `tmp/pretriton_s4_analysis.md` | Phase 1 | s4 分析 |
| `tmp/pretriton_s6_analysis.md` | Phase 1 | s6 分析 |
| `tmp/pretriton_s7_analysis.md` | Phase 1 | s7 分析 |
| `tmp/pretriton_s8_analysis.md` | Phase 1 | s8 分析 |
| `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md` | Phase 1 | Master plan |
| `tmp/run_pretriton_baseline_119_gpu0.sh` | Phase 1 | baseline 重测脚本 |
| `tmp/pretriton_baseline_res128.json`, `_res256.json` | Phase 1 | baseline 数据 |
| `corep_fast/stages/s*.py` (修改) | Phase 2 | torch 优化代码 |
| `corep_fast/tests/...` (新增) | Phase 2 | A/B test |
| `tmp/pretriton_final_res128.json`, `_res256.json` | Phase 3 | 最终 profile |
| `my-docs/20260416-pre-triton-final-pass-results.md` | Phase 3 | 实测结果 |
| `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md` | Phase 3 | Triton handoff spec |

---

*Spec v1.0. Ready for review.*
