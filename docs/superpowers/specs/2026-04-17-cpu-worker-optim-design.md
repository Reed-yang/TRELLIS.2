# CPU MP Worker Optimization — Design Spec (V1)

**Date:** 2026-04-17
**Branch:** `post-profile-sync-elim` (continuation; stays on current branch per user direction)
**Precondition:** sync-spike findings `logs/findings_sync_sources.md` (commit `c64b7cc`)
**Status:** DRAFT — scope framework fixed; §4 specific-hotspot targets pending CPU worker profile data (subagent running in parallel on 116 GPU 4)

---

## 1. Goal

把 corep_fast e2e wall-time 从 ~10 s @ res=256 下拉，**不引入 Triton**，通过减轻 CPU multiprocessing worker 负担和精简 GPU→CPU 数据传递来压缩 GPU idle 时间。

## 2. Background — 为什么不先做 §1 1.0s sync

sync-spike findings 原本推荐 Option Y (`.item()` 消除 + Triton fused kernel)。重新审视后：

- **§1 sync = 1 s**，占 e2e 的 10 %。
- **GPU idle = ~9.6 s**（96 %），是 **§1 sync 的 10 倍**。
- GPU idle 的源头是 **Bucket B 的 6 处 bulk `.cpu().numpy()`** 喂 CPU MP worker，GPU 在等 Python 侧 worker 跑完。
- `.item()` 消除只有 Triton 或 VRAM 不可行的 pre-alloc 两条路。写 Triton 的 **维护代价** 高（引入新依赖 + build 流程复杂化），在这个阶段回报/代价比不划算。

结论：**先压 GPU idle 这条大鱼**。§1 + Triton 推迟到单独 spec，等本轮优化完成后再评估剩余 headroom。

## 3. Scope

### 3.1 In-scope

| # | 工作项 | 预期 Δ | 预期 effort |
|---|---|---|---|
| W1 | **Bucket A 2 处 `.item()` 清理**（findings §2 rank 8 + secondary A 行） | ~0 s (cleanup) | 2 h |
| W2 | **MP Pool 持久化**（如 profile 显示每次 `Pool(...)` fork/IPC 启动代价显著） | 估 0.3-1 s | 1-2 d |
| W3 | **Bucket B 6 处 `.cpu().numpy()` payload 精简**（削减传给 MP worker 的列） | 估 0.2-0.8 s | 1-2 d |
| W4 | **CPU worker 算法热点修复**（profile 数据驱动；具体目标在 profile 返回后填入 §4） | 估 1-3 s | 2-5 d |

**Topology correctness 门：** W1-W4 每一步产出 commit 前必须通过双参照回归（vs custom + vs `fast_M2` baseline，res=128 + res=256 双分辨率，cube_indices / V / F bit-exact）。见 §5。

### 3.2 Out-of-scope

- §1 1.0s sync (`s1_voxelize.py:56 .item()`) 及其 Triton 重写 — 留给**未来独立 Triton-introduction spec**
- Bucket C / D 剩余站点（findings §2 中非 A/B 的条目）
- s7 per-cube loop vectorize（ROI #3）
- cummax scan fuse（ROI #5）
- 任何新依赖（Triton / Cython / Numba / Rust）
- corep_fast 单元测试结构重组（只添加，不改）

### 3.3 VRAM 约束

本 spec 所有变更对 VRAM 影响 ≈0：
- W1/W3：纯删除 `.item()` / 缩小 transfer payload，释放内存
- W2：Pool 持久化只影响 CPU 侧内存
- W4：算法优化视 profile 结果而定；若某个 W4 子项会显著增加 VRAM，必须在子任务中标红并评估

## 4. Specific targets (pending CPU profile data)

> **STATUS:** CPU worker profile subagent runs in parallel on 116 GPU 4.
> Specific function-level targets for W4 will be filled in here once
> `tmp/cpu_profile/findings_cpu_worker.md` is produced. The scope framework
> above (W1-W4) is fixed; only W4's concrete hotspot list depends on data.

### 4.1 W1 targets (已确定)

- `corep_fast/stages/s8_collapse.py:1629` — Bucket A representative (findings §2 rank 8)
- Secondary A row from findings §2 (identify during plan-writing)

### 4.2 W2 scope (半确定)

Survey 所有 stage 的 `multiprocessing.Pool` 调用点：
- `s4_face_point.py` — pool dispatch
- `s6_collapse.py` — pool dispatch
- `s7_rank_assign.py` — pool dispatch
- `s8_collapse.py` — pool dispatch

对每处量化 pool 启动开销（profile 产出的 `Pool.__init__` / `fork` / `pickle.dumps` 时间）。若总 pool startup > 100 ms，落地持久化 pool。若不显著则**放弃 W2**。

### 4.3 W3 targets (从 findings §2 已知)

6 处 Bucket B 站点（findings §2 rank 1-4 + rank 6 + 1 additional，详见 §2 表）：
- `s6_collapse.py:846-850` — 5 连 `.cpu().numpy()`
- `s7_rank_assign.py:1206-1218` — 4 连 bulk transfer
- `s4_face_point.py:148-152` — 5 连 `.cpu().numpy()`
- `s8_collapse.py:283-292` — 10 连 bulk `.cpu().numpy()`
- （其余两站根据 findings §2 确定）

对每处：
1. 追踪下游 MP worker 实际读取哪些列 / 字段
2. 若 worker 不读某列 → 从 transfer 中删除
3. 若 worker 只读标量聚合 → 改为 GPU 侧先 reduce，再传
4. 量化 payload 减小百分比 + 实测 `.cpu()` wall 下降

### 4.4 W4 targets (PENDING PROFILE DATA)

等 `tmp/cpu_profile/findings_cpu_worker.md` 出来后，按以下标准挑前 1-3 个 worker hotspot 进入 W4：
- 该 function 的累计 self-time > 200 ms per call (or per batch)
- 该 function 存在 clear Python-level 低效（不必要 object allocation / 冗余 dict lookup / 未用 numpy 向量化 / 调 list.append in hot loop 等）
- 修复可在 2-5 d 内完成，不需改 stage 间接口

## 5. Topology correctness methodology

本 spec 是**第一个**真正修改 `corep_fast/` 语义代码的 post-profile spec，必须引入 fixture 框架并在**每个 commit** 执行。

### 5.1 Fixture 矩阵

| Fixture | Resolution | Mesh | Baseline 参照 |
|---|---|---|---|
| F1 | res=128 | icosphere subdiv=3 (1280 faces) | `custom/` 参考实现 + pre-change `corep_fast/` |
| F2 | res=256 | 同上 | 同上 |
| F3 | res=128 | triple-concentric sphere (r=1.00/1.01/1.02) | 同上（覆盖 s7 multi-loop path） |

### 5.2 Bit-exact 比对项

对每次 pipeline 运行，比对：
- `cube_indices` tensor（shape, dtype, values bit-equal）
- `V` (cube vertices count) 标量
- `F` (cube faces count) 标量
- `decode_from_cubebatch` 最终输出（V×3, F×3）

允许的 float 数值误差：**0**（bit-exact）除非修改明确引入 scatter_mean 等 non-deterministic op；若有，记录原因并改用 atol/rtol = 1e-6。

### 5.3 每 commit 门禁

plan 中每个 commit 步骤末尾必须跑：
```
pytest tests/corep_fast/test_sync_optim_regression.py::test_F1_F2_F3 -v
```

新建 test 文件（W1 第一个 commit 里创建）：`tests/corep_fast/test_sync_optim_regression.py`。

## 6. Architecture / Files changed

**Modified (production code):**
- `corep_fast/stages/s4_face_point.py` — W3 payload trim; W4 possibly
- `corep_fast/stages/s6_collapse.py` — W1 A-site removal; W3 payload trim; W4 possibly
- `corep_fast/stages/s7_rank_assign.py` — W3 payload trim; W4 possibly
- `corep_fast/stages/s8_collapse.py` — W1 A-site removal; W3 payload trim; W4 possibly
- （新增 MP Pool 工具模块，若 W2 落地：`corep_fast/utils/persistent_pool.py`）

**Created (tests):**
- `tests/corep_fast/test_sync_optim_regression.py` — F1-F3 fixture gate

**Not touched:**
- `corep_fast/stages/s1_voxelize.py` — deferred to future Triton spec
- `corep_fast/stages/s2_components.py` — no W3 target (per findings §2 low ms)
- `corep_fast/stages/s3_edge_weights.py` — no target
- Other corep_fast submodules unless W4 profile surprises require

## 7. Task Breakdown (high-level; plan will refine)

| # | Task | Deps | Effort |
|---|---|---|---|
| T1 | Create fixture regression test (F1-F3) against current baseline | — | 1 d |
| T2 | W1 — Bucket A cleanup | T1 | 2 h |
| T3 | W3 — Bucket B payload trim (per-site loop: s6 / s7 / s4 / s8) | T1 | 1-2 d |
| T4 | W2 — MP Pool persistence (only if profile confirms >100 ms startup) | T1 | 1-2 d |
| T5 | W4 — Top hotspot algorithmic fix #1 (name TBD after profile) | T1, profile done | 2-5 d |
| T6 | W4 — Top hotspot algorithmic fix #2 (if budget remains) | T5 | 2-3 d |
| T7 | Re-profile post-fix; report final Δ wall-time | all above | 0.5 d |
| T8 | Update findings doc with post-fix results; recommend next spec | T7 | 0.5 d |

**Total estimated effort:** 7-14 d（不含 W4 #2 如果 #1 已达目标）。

## 8. Definition of Done

| # | 验收项 | 检查方式 |
|---|---|---|
| 1 | 每个 commit 通过 F1-F3 fixture 门禁 | CI / local pytest; 提交 message 含 fixture PASS 行 |
| 2 | W1 落地（2 处 Bucket A `.item()` 清理） | `git diff` 前后差异 + findings §2 更新 |
| 3 | W3 落地（6 处 payload 精简） | 每处 `.cpu()` 前后 dtype/shape 记录差异 |
| 4 | W2 执行或显式 skip with justification | profile data 驱动决策；decision doc 记录在 `logs/` |
| 5 | W4 至少 1 个 hotspot 被修复 | 该 function self-time 前后对比 |
| 6 | 实测 e2e wall-time 下降 ≥ 1 s @ res=256 | T7 re-profile 产出 |
| 7 | `corep_fast/stages/s1_voxelize.py` 未被修改 | `git diff <base> HEAD -- corep_fast/stages/s1_voxelize.py` 为空 |
| 8 | 无新的外部依赖（`pip freeze` diff 为空 over Python 依赖段） | 对比 `.venv` pip list 前后 |

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| CPU profile data 显示 worker 算法没有 clear 热点（W4 空手而归） | Medium | W4 未发现就 skip；W1-W3 仍有 0.5-1 s 收益；T7 re-profile 会确认 |
| W3 payload trim 改变 worker 输入语义（regression） | Medium | F1-F3 fixture per-commit 拦截；dry-run 对比 payload 内容 |
| W2 持久化 pool 引入 worker state 泄漏（影响下次调用） | Low-Medium | 每次 submit 前显式 reset worker；fixture 覆盖多次连续调用 |
| W4 某个算法改动触发 non-determinism（scatter_mean, random ordering） | Medium | §5.2 允许 atol/rtol = 1e-6 并记录原因；若 bit-exact 不可保留则二维回归（shape + 拓扑等价）|
| 实测 e2e 下降 < 1 s（DoD #6 不达） | Medium | 中期评估点在 T5 末尾；若不达则根据 T7 数据决定是否补做 W4 #2，或终止 spec 并交付部分收益 |
| VRAM 增加（任何子任务意外增大 GPU 内存占用） | Low | 每个 commit 跑 `nvidia-smi` 对比；spec 明确 §3.3 约束 |
| §1 sync 最终证明也必须在这个 spec 内一并修复 | Low | 若 T7 发现 GPU idle 仍 > 2 s 且已排除 CPU 原因，则升级一个子任务 fallback 到 Triton（但此时就不是低风险 spec 了，考虑拆分）|

## 10. Handoff — what feeds the NEXT spec after this

- `logs/findings_cpu_worker_post_fix.md` — post-fix re-profile 结论
- 剩余 headroom estimate（如 GPU idle 仍 > 2 s，建议 Triton 方向）
- W4 未修的 hotspot 列表（后续 spec 候选）
- VRAM baseline 对比（未来 Triton spec 评估 pre-alloc 可行性时用）

## 11. Living document note

§4.4 (W4 具体目标) 将在 CPU profile subagent 完成后（预计 30-60 min 内）填入 this spec，然后 commit 一个 "spec: update W4 targets from profile" patch commit。其他章节稳定。
