# Sync-Spike + Tooling Quick-Win — Design Spec

**Date:** 2026-04-17
**Branch:** `post-profile-sync-elim` (from `gpu-pipeline@52f5a8c`)
**Precondition:** `my-docs/20260417-corep-deep-profiling-results.md` (Deep Profiling 结论)
**Next step:** 本 spec 产出 findings doc → 由 findings 驱动下一个 fix spec (sync elimination)

---

## 1. Goal

在 **不修改任何 `corep_fast` 语义代码** 的前提下，完成两件事：

1. **#6 Tooling 回报 (稳赚)** — 把 `tmp/profile_deep/driver.py` 的 `with_stack=True` 默认翻为 `False`，降低未来 profile overhead 5-10x。保留 `--with-stack` CLI flag 以便需要 Python 栈时开启。
2. **Investigation Spike (证据收集)** — 把 Deep Profiling 报告定位出来但未归因的 **1.0s cudaDeviceSynchronize** 和 **4128 次 cudaStreamSynchronize / 4022 次 D2H** 按根因分类，产出 `logs/findings_sync_sources.md`，为下一个 **sync elimination fix spec** 提供 evidence-based scope 估算。

本 spec 的输出是一份 findings doc，不是代码 fix。`corep_fast/` 下零代码修改。

## 2. Background

Deep Profiling (2026-04-16/17) 三条头条:

- GPU idle ~96% @ res=256 — pipeline 不是 GPU-bound
- Top-20 kernel 零 CMB — 不是算子算力瓶颈
- s4 / s7 device time 不随 R 翻倍而 R²-scale — host-bound

ROI 表头两项的 effort 都依赖一个未完成的 spike:

- **#1 消 1.0s cudaDeviceSynchronize:** 若根因是 `.item()` → 几行修改；若是 alloc-size sync → 需重构，1-5d
- **#2 消 4022 次 D2H (.item() 泄漏):** 若大多可延迟 → 机械批量；若多数驱动 Python 控制流 → 1-2 周

这次 spike 就是为了把两个 "若" 变成事实。

## 3. Scope

### 3.1 In-scope

| Item | 描述 |
|---|---|
| T1 | `driver.py` 的 `with_stack` 默认值翻转 (#6) |
| T2 | 定位 1.0s cudaDeviceSynchronize 的具体 Python call site |
| T3 | 抽样 top-20 blocking D2H (按 device time 排序)，按 5 桶根因分类 |
| T4 | 产出 `logs/findings_sync_sources.md` |
| T5 | self-review + commit |

### 3.2 Out-of-scope (明确排除)

- 修改 `corep_fast/` 下任何代码 (包括 `.item()` 移除、`sync` 消除、s7 vectorize)
- 新增 / 修改 stage semantics
- 跑 full regression suite (corep_fast 零改 → 无需验证)
- 跑新的 full profiling run (基于已有 `tmp/profile_deep/results/`；若有数据 gap 再补局部)
- Triton K1 工作 (#4)
- cummax fuse (#5)

### 3.3 Why no topology verification in THIS spec

用户总目标要求 "每一步都要确保拓扑正确性(相比 custom 和 baseline)"。由于本 spec 不改 corep_fast，**零语义变化 → 无拓扑回归可能**。拓扑正确性验证的流程 (golden fixtures、custom vs baseline dual-reference、per-task regression gate) 将在**下一个 sync-elimination fix spec 里正式制度化**。本 spec 的 DoD (§6) 不含拓扑检查是有意的。

## 4. Task Breakdown

### T1 — #6 `with_stack=False` tooling flip

**Why:** Deep Profiling 中 `with_stack=True @ res=256` 产生 3.8GB Chrome trace (超 GitHub 2GB 限制)，profile overhead 5-10x。之后只在需要深度 Python 栈归因时临时启用。

**How:**
- 修改 `tmp/profile_deep/driver.py`:
  - 默认 `with_stack=False`
  - 新增 `--with-stack` CLI flag (action='store_true')，覆盖默认值
- 其余 driver 行为 (NVTX 注入、layer switch、summary JSON 写出) 不变

**Validation:**
- 运行 `python -m tmp.profile_deep.driver --layer 1 --res 128` (smoke)，e2e wall 与基线 ± 10%
- 确认 Chrome trace size 明显下降 (with_stack=True 时 res=128 3-run ~200MB；关后预期 < 40MB)

**Deliverable:** 1 commit，仅涉及 `tmp/profile_deep/driver.py`

### T2 — 定位 1.0s `cudaDeviceSynchronize` call site

**Evidence:** `tmp/profile_deep/results/layer0_res256_summary.json` 或 nsys cuApiTrace 显示 `cudaDeviceSynchronize` call count = 20，total = 1049ms，single call max = 1048ms (即 20 次里有 1 次占 99.9%)。

**Method (3 步):**

1. **nsys 侧**: 用 `tmp/profile_deep/results/nsys_res256_full.sqlite` 通过 SQLite 查询 (nsys 生成的 trace 可以 SQL 查)，按时间戳排序找到那次 1048ms 的 cudaDeviceSynchronize 的 start / end；
2. **torch.profiler 侧**: 从 `layer1_res256_run1_trace.json` (Chrome trace JSON) 里按时间戳反查同一时刻的 `python_function` 事件 (Deep Profiling 已确认 NVTX 不进该 trace，所以用 python_function 匹配)；
3. **grep 侧**: 在 `corep_fast/` 下全局 grep `torch.cuda.synchronize(` 和 PyTorch 隐式 sync 触发器: `.item()`、`.cpu()`、`.tolist()`、`.numpy()`、`bool(tensor)`、`if tensor` pattern。

**Cross-ref**: 三条路径应收敛到同一 `file.py:line`。如不收敛，优先 torch.profiler stack (因为它直接有 Python 帧)。

**Deliverable:** `logs/findings_sync_sources.md` §1 段落：
- Stage (s1-s8)
- `file.py:line`
- 源码片段 (±5 行)
- 根因桶 (见 T3 的 5 桶 schema)
- 根据根因的 effort 估算

### T3 — Top-20 blocking D2H 分类

**Evidence:** Deep Profiling 报 `cudaStreamSynchronize` call count = 4128，D2H memcpy = 4022。假设几乎每次 D2H 后跟一次 stream sync (blocking semantics)。这 4022 次分布在哪？要分类。

**Method:**

1. 从 `layer1_res256_run1_trace.json` (with_stack=True 版) 筛出 `cudaMemcpyAsync` with kind=DtoH 且紧跟 `cudaStreamSynchronize` 的 pattern；
2. 按 **(file.py:line, Python function name)** 聚合，取 call count × average_dur 乘积 top-20；
3. 对每条，读源码 ±10 行，按 5 桶分类：

| 桶 | 名称 | 定义 | Effort (per site) |
|---|---|---|---|
| **A** | Deferrable | `.item()`/`.cpu()` 的返回值只用于 Python 侧 print/log/debug/profile — 可删或可延迟到最后 | seconds |
| **B** | Control-flow | `.item()` 驱动 `if/while/for` Python 分支 — 需算法重构 (e.g. mask-driven GPU control，或提前 batch 所有分支) | days |
| **C** | Alloc-size | `.item()` 取出的整数用于后续 `torch.zeros(n, ...)` / `view(n, ...)` / `nonzero()` size — 需 CUDA Graph 或 symbolic size | days-weeks |
| **D** | Library-internal | 来自 PyTorch/CUDA 库内部 (e.g. `.sort()` 的 index，某些 cuSPARSE 调用) — 不可直接改，只能换 API | weeks (换 API) |
| **E** | Correctness-sync | 显式 `torch.cuda.synchronize()` for 跨 stream / timing correctness | case-by-case |

**Deliverable:** `logs/findings_sync_sources.md` §2 段落：top-20 table，列 `(rank, stage, file:line, sample_source_snippet, call_count, aggregate_dur_ms, bucket, per_site_effort)`。

### T4 — 产出 `logs/findings_sync_sources.md`

**Structure:**

```
# Sync Sources — Findings (2026-04-17 spike)

## 0. TL;DR
- 1s DeviceSync 根因: [A/B/C/D/E] @ file.py:line
- Top-20 D2H 分桶: A=__ / B=__ / C=__ / D=__ / E=__
- 下一个 fix spec 推荐 scope: [详见 §4]

## 1. The 1.0s cudaDeviceSynchronize
- nsys 定位时间戳: ...
- torch.profiler python stack: ...
- grep confirm: ...
- Source snippet
- 根因桶与解释
- Effort 估算

## 2. Top-20 Blocking D2H — Classified
- Table (rank × 10 列)
- 每桶 1-2 行 representative 样例

## 3. Per-Bucket Breakdown
- A 桶 (count, 合计 effort, 预期 Δ wall-time)
- B 桶 同
- C 桶 同
- D 桶 同
- E 桶 同

## 4. Recommended Next Fix Spec Scope
- Option X — "A-only": 纯批量删除 / defer，risk=low，预期 Δ -0.3~-0.8s，effort ~1-2d
- Option Y — "A + 1s DeviceSync": 加上 T2 找到的那条 (若根因属 A/E) — risk=low-medium，Δ -0.5~-1.5s，effort ~2-3d
- Option Z — "A + B (selected)": 含算法重构，risk=medium-high，Δ -1~-2s，effort ~1 周
- 推荐: [based on findings]

## 5. Caveats
- trace 来自 res=256 单次 run (nsys 3-run median 已另记录)，with_stack=True 可能拉高 D2H overhead 本身，阅读时打折
- NVTX 不进 torch.profiler trace，stage attribution 依 python_function fn name 匹配 (同 Deep Profiling 的方法)
- D2H 大多数是异步 launch 后的 blocking sync，实际 wall-time 归因见 aggregate_dur_ms 列
```

**Deliverable:** 1 commit 含 `logs/findings_sync_sources.md` (+ 可选 `tmp/sync_spike/` 下分析脚本 if 写了)

### T5 — Self-review + commit

- 检查 placeholder (TBD / TODO / "TBD in §X")
- 检查 5 桶分类的一致性
- 检查每一条是否都有 `file.py:line` (不是 `file.py:???`)
- 检查 §4 推荐 scope 的 effort 估算与 §3 桶内 breakdown 一致
- `git commit -m "..."`

## 5. Methodology Principles

| Principle | Rationale |
|---|---|
| **不重跑 profile** | `tmp/profile_deep/results/` 数据够用；若发现 gap (比如 nsys sqlite 缺列) 再补局部 |
| **分析脚本落 `tmp/sync_spike/`** | 遵循 profile_deep 的 convention；不污染 repo root |
| **`corep_fast/` 零修改** | 保持可 revert；本 spec 无回归风险 |
| **所有分类须有 concrete evidence** | 每个桶判定必须 cite 一个 source file:line (不能 "大概属于 B 桶") |
| **Effort 估算分档写** | seconds / hours / days / weeks — 不要给小数天精度 (给了也是假的) |

## 6. Definition of Done

| # | 验收项 | 检查方式 |
|---|---|---|
| 1 | T1 driver.py 默认值已改 | `grep "with_stack" tmp/profile_deep/driver.py` |
| 2 | T1 smoke e2e wall ≤ 原 +10% | `python -m tmp.profile_deep.driver --layer 1 --res 128` 对比 baseline_res128_runlog.txt |
| 3 | 1.0s DeviceSync 有明确 `file.py:line` | findings doc §1 有源码片段 |
| 4 | Top-20 blocking D2H 100% 分到 5 桶 | findings doc §2 table 每行 bucket 列非空 |
| 5 | Findings doc §4 有下一 spec 推荐 scope | 含 Option X/Y/Z 三档 + 推荐 |
| 6 | 2 commits: (a) driver tooling flip, (b) findings doc + 任何分析脚本 | `git log post-profile-sync-elim ^52f5a8c --oneline` 恰好 2 行 |
| 7 | `corep_fast/` 无改动 | `git diff 52f5a8c HEAD -- corep_fast/` 为空 |

## 7. Out-of-Scope (再次强调)

- 任何 `.item()` 实际移除
- 任何 sync 实际消除
- s7 per-cube loop vectorize (ROI #3)
- Triton K1 fuse (ROI #4)
- cummax scan fuse (ROI #5)
- 新的 regression fixture

这些在下一个 fix spec 里。

## 8. Risks & Caveats

| Risk | Mitigation |
|---|---|
| with_stack=True trace 本身扰动了 D2H 统计 | 已有 nsys cuApiTrace 可交叉验证；findings doc §5 标明 |
| 5 桶分类中的边界 case | 允许标 "A/B" (两桶) 并加注释；统计时按保守桶计 effort |
| nsys sqlite 没足够列定位 1s sync | 则退化为只用 torch.profiler trace + grep；若仍定位不到，在 T2 末尾允许补一次 with_stack=True 局部 re-profile |
| top-20 D2H 主要集中在一两个 pattern (而非 20 种) | 是 finding 本身，不是 risk；按聚合后的 pattern 而非 site 给 scope |

## 9. Files Changed (前瞻)

- Modified: `tmp/profile_deep/driver.py` (~10 lines, with_stack default + CLI flag)
- Created: `logs/findings_sync_sources.md`
- Optional: `tmp/sync_spike/analyze_d2h.py` 等分析工具 (if 写了)

## 10. Handoff to Next Spec

本 spec commit 后，下一个 spec (`2026-04-XX-sync-elimination-implementation-design.md`) 会基于 `logs/findings_sync_sources.md §4 Recommended Scope` 展开。届时正式引入:
- Custom vs baseline dual-reference 拓扑验证
- Per-task regression gate
- Fixture matrix (单球 / 三球 / 随机 mesh)
