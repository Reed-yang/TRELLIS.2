# CoReP Deep Profiling 设计文档（pre-Triton 瓶颈深度调查）

> 日期: 2026-04-16
> 分支: `gpu-pipeline` @ `731a939` (Phase 2 final)
> 硬件目标: 119 节点 H100 80GB HBM3, GPU 0
> Benchmark: icosphere subdiv=3 (V=642, F=1280), res=256 主 + res=128 辅
> 前置: Phase 2 已完成（res=256 e2e **9.13s** = 15.46x vs custom 141.05s）
> 终点: 选定下一阶段优化方向前（Triton K1 / mesh-cleanup-port / s1-sat-hardening）的深度数据支撑
> Owner: 未分配

---

## 0. TL;DR

对 `corep_fast` 做 **4 层纵深 profiling**（res=256 主战场 + res=128 scaling 交叉验证），在 119 GPU 0 上执行，得到：nsys 宏观 timeline、per-stage Chrome trace、e2e top-20 GPU kernel 归因表、resolution scaling 对照、以及按 ROI 排序的 next-step 候选清单。

**显式非目标**：不采 Triton K1 设计所需数据（K 分布 / group 直方图）、不跑 Nsight Compute per-kernel roofline、不用真实 mesh、不测 res=512、不改任何 `corep_fast/` 源码。纯 instrumentation + 分析。

---

## 1. 动机

Phase 2 终点 res=256 e2e 9.13s（vs custom 141.05s = 15.46x）。在决定下一阶段（Triton K1，~1-2 周 effort）之前需要回答：

1. **每个 hot stage 内部时间具体花在哪里** —— 当前数据是 stage 级（s4=4.85s），没有 kernel / sub-function 级。
2. **每个 hot stage 是 compute-bound / memory-bound / launch-bound** —— 决定 Triton 是否是对的工具（Triton 擅长 launch-bound 小图、不擅长 memory-bound 大 matmul）。
3. **是否存在 Phase 2 未处理的隐藏 CPU↔GPU sync / d2h 开销** —— 可能是比 Triton 更便宜的优化机会。
4. **各 bottleneck 如何随 R scale** —— O(R³) 的 stage 在高 res 下会主导，fixed-cost 的会相对变小。

没有这些数据，"next stage" 的决策都是 under-informed。

---

## 2. Scope

### 2.1 In scope
- **Stages**: 全部 8 个（s1, s2, s3, s4, s6, s7, s8）。s1+s2+s3+s8 加起来只占 3.7%，但验证其 saturate 状态、并观察高 R 下是否反转为新瓶颈，本身是产出。
- **Resolutions**: res=256 主 + res=128 scaling 交叉。
- **Input**: icosphere subdiv-3 (V=642, F=1280)，延用 Phase 2 benchmark，保证可比性。
- **Tools**: torch.profiler + Nsight Systems (nsys)，CUDA events 作为轻量 sub-stage 辅助。
- **Hardware**: 119 GPU 0，符合 `feedback_profiling_on_119.md`。

### 2.2 Out of scope
- Nsight Compute (ncu) per-kernel roofline（延后；Layer 2 分类为启发式）
- 真实 .glb/.obj mesh（保持与 Phase 2 数据的可比性）
- res=512（OOM 风险 + 当前问题不需要）
- Triton K1 baseline 数据（K-distribution / group 直方图）—— K1 实施时再采
- 任何 `corep_fast/` 源码修改 —— instrumentation 通过 driver 脚本 monkey-patch 注入
- 多卡 / batched throughput —— 本次只做单 mesh 单卡 kernel 纵深

---

## 3. 架构：4-layer 投资结构

```
┌────────────────────────────────────────────────────────────────┐
│ Layer 0 — 宏观 timeline (nsys)                                  │
│   Input:  1× e2e run @ res=256                                  │
│   Output: .nsys-rep + nsys stats 文本 dump                       │
│   回答:   GPU idle 窗口？kernel launch bubble？stream           │
│           serialization？显式 CPU↔GPU sync 点？                  │
├────────────────────────────────────────────────────────────────┤
│ Layer 1 — per-stage op trace (torch.profiler)                   │
│   Input:  1× e2e run @ res=256 (NVTX 标段)                      │
│   Output: Chrome trace JSON + per-stage top-30 op CSV (×8)      │
│   回答:   每 stage 内 Python → aten → CUDA kernel 时间分层？     │
│           显式 .cpu()/.cuda() 传输？sync op 占比？               │
├────────────────────────────────────────────────────────────────┤
│ Layer 2 — 热点 kernel 归因                                       │
│   Input:  后处理 Layer 1 trace                                   │
│   Output: e2e top-20 GPU kernel 表                              │
│           (kernel / stage / py-line / time / 启发类别)           │
│   回答:   每个 hotspot → compute-bound / memory-bound /         │
│           launch-bound / CPU-bound（启发分类）                    │
├────────────────────────────────────────────────────────────────┤
│ Layer 3 — res=128 scaling 交叉                                   │
│   Input:  重跑 Layer 1 @ res=128                                │
│   Output: per-stage (t_256 / t_128) 表 + top-kernel 对照        │
│   回答:   哪些 bottleneck 是 O(R³) 增长？哪些是 fixed-cost？      │
│           res=128 是否暴露 res=256 看不到的新瓶颈？               │
└────────────────────────────────────────────────────────────────┘
```

**Layer 独立性**：每层独立产出。Layer 0 若直接给出决定性结论，Layers 1–3 仍可完成以求完整，但不阻塞下一阶段决策。任何一层发现数据可疑都可单独重跑，不波及其他层。

---

## 4. 组件

### 4.1 目录布局

```
tmp/profile_deep/
├── driver.py                       # 通用 e2e driver, --res / --layer / --stage-filter
├── monkeypatch_nvtx.py              # NVTX + CUDA event 注入（不改 repo）
├── run_layer0_nsys.sh               # SSH → 119 → nsys profile
├── run_layer1_torch.sh              # SSH → 119 → torch.profiler @ res=256
├── run_layer3_torch.sh              # 同上 @ res=128
├── analyze_layer1_trace.py          # 聚合 per-stage top-N ops
├── analyze_layer2_kernels.py        # e2e top-20 kernel + 启发分类
├── analyze_layer3_scaling.py        # res=256/128 对照
└── results/
    ├── nsys_res256_full.nsys-rep
    ├── nsys_res256_stats.txt
    ├── torch_profile_res256_e2e.json
    ├── torch_profile_res128_e2e.json
    ├── per_stage_ops_res256.csv
    ├── per_stage_ops_res128.csv
    ├── top20_kernels_res256.csv
    └── scaling_table.csv
```

### 4.2 注入策略（monkey-patch，不改 repo）

`monkeypatch_nvtx.py` 在 driver 启动时先 import，在 `corep_fast.*` import 之前 patch：

**Stage-level NVTX ranges**（给 nsys + torch.profiler 做段切分），入口函数已对照 `corep_fast/stages/*.py` 确认：

```python
corep_fast.stages.s1_voxelize.s1_voxelize             (line 18)   → "s1_voxelize"
corep_fast.stages.s2_components.s2_components         (line 18)   → "s2_components"
corep_fast.stages.s3_edge_weights.s3_edge_weights     (line 27)   → "s3_edge_weights"
corep_fast.stages.s4_face_point.s4_face_point         (line 66)   → "s4_face_point"
corep_fast.stages.s6_collapse.s6_collapse             (line 775)  → "s6_collapse"
corep_fast.stages.s7_rank_assign.s7_rank_assign       (line 1082) → "s7_rank_assign"
corep_fast.stages.s8_collapse.decode_from_cubebatch   (line 231)  → "s8_decode"
```

**Sub-stage CUDA events**（细粒度 timing，不膨胀 trace 文件）：

```
# s4: Stage A (segment build) / B (graph build) / C (BFS) / D (U-turn count)
# s6: fast-path GPU 分支 / slow-path 分支 / dispatch
# s7: Phase-1 GPU BFS / Phase-2 rank fill
```

每个 wrapped sub-function 用 `torch.cuda.Event` 打 `start.record()` / `end.record()`，pipeline 结束时 driver `torch.cuda.synchronize()` 后把 `{stage}/{sub}: ms` dump 到 `results/sub_stage_timings_{res}.json`。

**Smoke check**（关键保险）：driver 首次启动跑一次带 patch 的完整 pipeline，总 walltime 与 Phase 2 `tmp/e2e_profile_m2.py` 基准对比。若 delta > 10%，说明注入太侵入，回退到仅 NVTX（不打 CUDA event）。

### 4.3 Driver (`driver.py`)

```python
# Skeleton:
import monkeypatch_nvtx  # 必须第一条 — corep_fast import 之前 patch
import argparse, torch, gc, time
# ...
parser.add_argument('--res', type=int, required=True)
parser.add_argument('--layer', choices=['0','1','3'], required=True)
args = parser.parse_args()

# 1. 构造 deterministic icosphere (V=642, F=1280, subdiv=3)
# 2. Warmup: 1× pipeline run，timing 丢弃（稳定 CUDA context / JIT）
# 3. gc.collect(); torch.cuda.empty_cache()
# 4. 按 --layer 启用 profiler:
#    layer=0 → 无操作（nsys 外部包整个进程）
#    layer=1 or 3 → torch.profiler(schedule(wait=0, warmup=0, active=1))
# 5. 跑 pipeline（NVTX 段 + CUDA event 已生效）
# 6. End profiler; dump trace + sub_stage_timings JSON 到 results/
```

### 4.4 后处理脚本

- `analyze_layer1_trace.py` —— 读 Chrome trace JSON，按 NVTX range 过滤出 stage，聚合 per-stage top-30 ops（按 `self_device_time_total` + `self_cpu_time_total`），写 `per_stage_ops_res{256,128}.csv`。
- `analyze_layer2_kernels.py` —— 跨 stage 聚合 GPU kernel event，按 device time 出 top-20；归因：stage 从 parent NVTX range、py-line 从 `with_stack=True` metadata、启发类别按 duration + kernel 名 + aten parent：
  - **LNB** (launch-bound): 平均 duration < 10µs 且 count > 1000
  - **MMB** (memory-bound): kernel 名匹配 `copy|memset|scatter|gather|index|cat|slice` 或 aten op 为纯数据搬运
  - **CMB** (compute-bound): kernel 名匹配 `gemm|conv|reduce|sum|matmul` **且** 平均 duration > 100µs
  - **CPU** (CPU-bound): stage wall-time − sum(stage 内 GPU kernel 时间) > 30% stage wall-time
  - 其余: **UNK**（启发无法分类 —— 标记为"需 ncu 才能确认"）
- `analyze_layer3_scaling.py` —— join `per_stage_ops_res256.csv` + `per_stage_ops_res128.csv`，计算 `t_256 / t_128` ratio，flag 任何 ratio ∉ [2, 10] 的 stage。阈值 rationale：R 从 128 倍增到 256 时，典型期望 O(R²) ≈ 4x, O(R³) ≈ 8x；ratio < 2 暗示 fixed-cost 主导（launch overhead 或常量开销），> 10 暗示超线性增长（如指数复杂度或 cache cliff）。两端都是异常信号。

---

## 5. 数据流

```
[driver + monkeypatch]
      │
      ├── Layer 0: nsys 外部包进程 → nsys_res256_full.nsys-rep
      │                              + nsys stats → stats.txt
      │
      ├── Layer 1: torch.profiler context → torch_profile_res256_e2e.json
      │                                     + sub_stage_timings_res256.json
      │           └─ [post] analyze_layer1_trace.py → per_stage_ops_res256.csv
      │
      ├── Layer 3: 同 Layer 1 但 --res 128 → torch_profile_res128_e2e.json
      │                                     + sub_stage_timings_res128.json
      │           └─ [post] analyze_layer1_trace.py（复用）→ per_stage_ops_res128.csv
      │
      └── Layer 2: [post-only, 无新 run] analyze_layer2_kernels.py
                   读 torch_profile_res256_e2e.json
                   → top20_kernels_res256.csv
```

分析脚本纯后处理，无需 GPU，采完数据后可在任何节点跑。

---

## 6. 可复现性

- 固定 seed: `torch.manual_seed(42)`, `numpy.random.seed(42)`, icosphere deterministic 生成
- 每次 run 前: `torch.cuda.empty_cache()`, `gc.collect()`, 检查 CPU load (`uptime` < 2.0 否则 warn)
- 每份 trace 包含 provenance header: git SHA (`git rev-parse HEAD`), `nvidia-smi` snapshot, `torch.__version__`, CUDA version —— 写在每份 CSV/JSON 的头部注释
- 所有路径基于 repo root；绝对路径不超出 `/mnt/novita2/siyuan/workspace/TRELLIS.2/`

---

## 7. 风险 & 缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| nsys 全开使 pipeline 慢 2-3x | 绝对 walltime 失真 | Layer 0 只看**相对占比**；absolute baseline 用 Phase 2 的 9.13s |
| torch.profiler 自身 5-15% overhead | timing 漂移 | 同上，交叉核对绝对值 vs Phase 2 baseline，相对 % 是可信产出 |
| Monkey-patch 改变执行路径（如 CUDA event 造成隐式 sync） | 有偏测量 | §4.2 smoke check；delta > 10% 回退到 NVTX-only |
| 119 CPU 被他人污染 | timing 噪声 | run 前 `uptime` + `nvidia-smi` 检查；关键 walltime 取 3-run median；噪声显式记录 |
| res=128 暴露 res=256 看不到的 bottleneck（或反之）| 需要重新解读 | Feature, 不是 bug —— Layer 3 存在的意义 |
| Layer 2 启发分类误判 | ROI 清单排序偏 | 报告置信度；UNK/边界 case 标出；明确 ncu 是后续升级路径 |
| nsys 输出 > 2GB，远程难打开 | 分析阻塞 | 用 `nsys stats` CLI 抽文本表作主要分析入口；.nsys-rep 作备份 |

---

## 8. 工期估算

| 步骤 | Walltime on 119 | 本地分析 |
|---|---:|---:|
| 脚手架 (driver + monkeypatch + 3 sh) | — | 1.5h |
| Smoke check (patched vs 裸跑基准) | 0.5h | 0.5h |
| Layer 0 run + nsys stats + 标注 | 0.5h | 1.5h |
| Layer 1 run (res=256) + 分析 | 0.5h | 2.5h |
| Layer 2 聚合 + 启发分类 | — | 1.5h |
| Layer 3 run (res=128) + 对比 | 0.5h | 1.0h |
| results 文档 + ROI 清单 | — | 1.5h |
| **合计** | **~2h** GPU | **~10h** 分析 |

≈ 1.5 developer-day。GPU 壁钟时间极少，绝大部分是脚手架和解读。

---

## 9. Definition of Done

1. D1–D7 全产出（spec / results md / per-stage CSV / top-20 kernel CSV / scaling table / raw traces）。
2. Layer 0 报告至少标注 **3 条具体观察**（GPU idle 窗 / stream bubble / CPU sync 点）。
3. Layer 1 为 8 个 stage 在两种分辨率下均产出 top-30 op 表。
4. Layer 2 top-20 kernel 表每条都有 `{stage, py-line}` 归因和 `{CMB, MMB, LNB, CPU, UNK}` 分类；UNK 比例 ≤ 20%。
5. Layer 3 scaling 表 flag 任何 ratio ∉ [2, 10] 的 stage 并给出书面假设。
6. ROI 候选清单列出**至少 5 个** next-step 选项，按 predicted speedup 排序，每条含：target stage / 假设 / 预计 Δ@res=256 / effort / 风险等级。
7. Smoke check 证明 monkey-patch overhead < 10%，**或者**已触发并记录 NVTX-only fallback。
8. 每份 CSV/JSON 包含 provenance header（git SHA / 时间戳 / 硬件 / seed）。

---

## 10. 这次 profiling 使能什么（以及不使能什么）

### 使能
对三个 pending next-stage 选项做出 informed 决策：

| Option | 当前估计 | 本轮 profile 如何 sharpen |
|---|---|---|
| Triton K1 (s4 BFS) | -2 to -3s, 1-2 weeks | 确认 s4 主要开销是 launch-bound MP 而非 kernel-internal 计算；验证 Triton 是对的工具 |
| mesh-cleanup-port | icosphere 上影响小 | Layer 3 res=128 trace 若暴露 regression，会记录为 follow-up（本轮不深入） |
| s1-sat-hardening | 性能中性 | Layer 0 nsys 确认 s1 仍仅占 1.6% e2e，severity 保持低 |
| 更便宜的发现（如 d2h sync 消除） | 未知 | Layer 0 + 1 可能暴露这类 ROI candidate，是 net-new 增量产出 |

### 不使能
- Triton K1 kernel 最终设计（需要 K-distribution 数据，延后）
- 真实多 mesh 吞吐量（需要独立 benchmark，不在 scope）
- Compute-bound kernel 调优建议（需要 ncu，不在 scope）

---

## 11. 引用

- Phase 2 终点: `my-docs/20260416-pre-triton-final-pass-results.md`
- Triton handoff spec: `docs/superpowers/specs/2026-04-16-corep-triton-handoff.md`
- Phase 2 baseline profile 脚本: `tmp/e2e_profile_m2.py`（driver 调用模式参考）
- 用户规则: `~/.claude/CLAUDE.md`（SSH 每条 cmd 前 cd-prefix；写 .sh 包装器在 tmp/）
- 项目规则: `/mnt/novita2/siyuan/workspace/TRELLIS.2/CLAUDE.MD`（SSH 经 tmp .sh 脚本）
- Memory: `feedback_profiling_on_119.md`（在 119 profile，禁本地 GPU）、`feedback_no_modify_repo.md`（不改 repo，只加）
