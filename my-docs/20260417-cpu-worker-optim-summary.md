# CPU Worker 优化 — 分支工作总览

> 日期：2026-04-17
> 分支：`post-profile-sync-elim`（起点 `0b1ef14`，收尾 `65eee1c`）
> Spec：`docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md` (V2)
> Plan：`docs/superpowers/plans/2026-04-17-cpu-worker-optim-implementation.md`
> Handoff：`docs/superpowers/specs/2026-04-17-cpu-worker-optim-handoff.md`
> 测量机：`host-10-240-99-116` GPU 4（H100 80GB）；fixture：icosphere `subdiv=3, radius=0.4`，res=256

---

## 1. 起点：之前做到了什么

### 1.1 历史脉络（此分支之前）

```
gpu-pipeline 主分支
  │
  ├── deep-profiling (2026-04-16 ~ 04-17, commit 53dfca5)
  │     发现：res=256 pipeline 96% GPU-idle；10.056s 里 kernel+memcpy 仅 347ms
  │     Top-20 kernel 0 CMB、17 MMB、3 LNB — 完全非 compute-bound
  │     s4/s7 随 R 翻倍仅增长 1.68×/1.11×（vs 期望 4×）— host-bound
  │     推荐：搁置 Triton K1，优先消除同步模式
  │     findings: my-docs/20260417-corep-deep-profiling-results.md
  │
  ├── sync-spike (commit 687b02f → 0b1ef14)
  │     目标：低挂果同步点消除，工具面低风险先行
  │     输出：findings_sync_sources.md；5 个 Bucket 分类（A/B/C/D/E）
  │     发现：Bucket B（6 个 bulk .cpu().numpy() 在 s6/s7/s4/s8 GPU→CPU-MP 边界）
  │            占 top-20 阻塞 D2H 的 98%
  │     产出：with_stack=False 默认 + --with-stack 开关；profiling 开销降
  │     产出：识别出 CMB/W 候选（_fastpath_trace_loops_numpy, _get_local_components_np）
  │     实际代码改动：零（investigation-only spec）
  │
  └── post-profile-sync-elim（本分支当前）
        目标：把 sync-spike 指出的 CPU 瓶颈真正做掉
```

### 1.2 本分支开工时的基线（T0 experiment, 2026-04-17）

起点 HEAD `0b1ef14`：

| 指标 | 值 | 来源 |
|---|---:|---|
| e2e wall @ res=256（干净，非 cProfile instrumented） | **8.631 s** | T0 默认 MP 3-trial 中位数 |
| vs custom baseline | 8.7× | 从 deep-profiling 延续 |
| pipeline V/F @ res=256 | 551,079 / 1,102,152 | 所有 trial 一致 |
| 跨 trial wall variance | 4.3% | default MP 有非确定性 |
| pre-V2 main-thread cProfile 前三热点 | `lock.acquire` 2711ms / `_fastpath_trace_loops_numpy` 1822ms / `_get_local_components_np` 1014ms | findings_sync_sources.md |

**核心观察（支配此分支所有决策）：** 主线程 cProfile 显示 `lock.acquire 2.7s` 等 MP worker；worker cProfile 只显示 ~60ms Python self。盲读数据的直觉是 **"MP 是纯开销，串行能省 2.7s"**。此判断差点让 plan 走偏，T0 决定性实验纠正了它（见 §2.1）。

---

## 2. 三个关键决策节点

### 2.1 T0 决定性实验：MP 必须保留（commit `b6bb12c`）

**问题：** V2 plan 的 W2/W6 都围绕"优化 MP"。若 MP 是净负收益，整个方向错了。

**实验：** 两个 subagent 并行在 116 的 GPU 3 / GPU 4 跑 3 trial e2e，分别用默认 MP 和 serial (monkeypatch `multiprocessing.Pool → SerialPool`)：

| Mode | median (s) | V/F count | wall variance |
|---|---:|---|---:|
| MP default (GPU 3) | **8.631** | 551,079 / 1,102,152 | 4.3% |
| Serial nw=1 (GPU 4) | **75.312** | 同上 | 0.15% (bit-deterministic) |

**结果：serial 慢 8.7×（+66.7s）。MP 绝不能删。**

**误读根因：** cProfile 的 `self`-time 只统计 Python 代码，不含 torch / numpy C-extension 调用。worker 里 60ms Python self 背后可能是秒级的 C 扩展计算（真正在并行），这部分时间在主线程以 `lock.acquire` 形式出现。

**副产品（后续持续受用）：** serial bit-deterministic（0.15% variance），确认可作为 **W4/W5 GPU 重写的 golden oracle**。

**成本：** ~15 min subagent wall time；节省：如果按"删 MP"方向做，可能浪费几天 W2/W6 工作。

Artifact：`logs/findings_t0_mp_vs_serial.md`、`tmp/cpu_profile/t0_driver.py`、`tmp/cpu_profile/t0_{default,serial}.{md,json,log}`

### 2.2 W6 Angle 3 (ThreadPool) 死局：GIL-holding 57.9%（T7a, commit `c0a2956`）

**问题：** W2 把 Stage D `lock.acquire` 从 2711ms 降到 1666ms 后，剩下 1.67s 是 worker 真实 steady-state wall time。Spec 预设三条路：Angle 1（W2 已解决，skip），Angle 2（GPU BFS+UTurn，3-5d 投入），Angle 3（ThreadPool 替 ProcessPool，前提 GIL-holding < 30%）。

**测量（cProfile 分类法）：** 跑 `_p2_uturn_worker` 一次，cProfile 统计 Python vs C-extension self-time：

| 组件 | self time | 分类 |
|---|---:|---|
| worker 总 wall | 125,273 ms | 全部 |
| Python self (GIL-holding) | 71,657 ms | **57.9%** |
| C-ext self (GIL-releasing) | 52,080 ms | 42.1% |

**单函数 Top3（纯 Python BFS / 数据结构）：**
- `_count_uturns` @ s4:305：49,679 ms（1,881,777 次调用）
- `_p2_uturn_worker` @ s4:1329：9,266 ms
- `_find_or_add_node` @ s4:398：3,963 ms（3,974,410 次调用）

**结论：** Angle 3 死局（GIL 占近 2× 阈值）。Angle 2 投入 3-5d 换 ~1-1.5s，DoD 已被 W4+W5 单独满足，**跳过 W6 是 ROI 最高选项**。

Artifact：`logs/findings_w6_gil_spike.md`、`tmp/cpu_profile/t7a_gil_spike.py`

### 2.3 W7 s7 orchestration skip：零 ≥200ms mechanical 候选（T8a, commit `ee7a9c1`）

对 post-W5 的 main-thread cProfile 过滤 `s7_rank_assign.py` 顶级热点：

| 函数 | self (ms) | 分类 |
|---|---:|---|
| `_build_adjacency_gpu` @ s7:634 | 1913.7 | 真 GPU compute（Triton 领域，非 mechanical） |
| `s7_rank_assign` @ s7:1175 | 984.6 | Phase 3 Hungarian × 275,539 次（scipy `linear_sum_assignment`），需向量化 + 自定 Hungarian |
| `_get_ordered_points` @ s7:53 | 9.7 | 已可忽略 |
| 疑似冗余 `.cpu().numpy()` @ s7:1206/1207/1215/1218 | 29 合计 | 远低阈值，2 个死代码、2 个仅遗留路径 |
| `cost.astype(np.float64)` | 77 | scipy 签名要求，不可减 |

**结论：W7 跳过。** 余下的都是 "重大重构" 或 "不可约减"。小清理 ≈15-25 ms，未来 s7 修改时顺手做即可。

Artifact：`tmp/cpu_worker_optim_audit/w7_s7_drilldown.md`、`tmp/cpu_profile/t8a_s7_drill.txt`

---

## 3. 三项落地的核心改造

### 3.1 W2 — 持久 MP Pool（commit `c923e85`）

**问题：** 6 个 stage 里每次 dispatch 都 `with _Pool(N) as p:` 建池 → `posix.fork` 124 次 / 816 ms per e2e run。

**改动：**
- 新建 `corep_fast/utils/persistent_pool.py`：`get_pool(N)` 懒加载全局单例 + `shutdown_pool()` + `atexit` 清理
- 替换 **7** 处 `with _Pool(N) as p` → `p = get_pool(N)`（plan 说 6，实际多一处 `_fw_worker_indexed` at s4:160）
- pool 初始化器设 `PYTHONHASHSEED=0`，根除 T0 发现的 0.7% 顶点集合跨 run 漂移（set/dict tiebreaker 随 worker hash seed 乱序）

**7 个替换点：**
```
s4_face_point.py:160    face-weights _fw_worker_indexed
s4_face_point.py:1211   Stage D _p2_uturn_worker
s6_collapse.py:987      slow-path _s6_worker
s7_rank_assign.py:1314  Phase 1/3 _s7_rank_worker
s8_collapse.py:1428     candidate edges
s8_collapse.py:1769     geometry vectorized
s8_collapse.py:1939     parallel processing
```

**单元测试（5/5 pass）** 验证持久池复用 + 不同 worker 数重建 + worker 内 `PYTHONHASHSEED == "0"`。

**测量（post-W2 vs pre-W2 main-thread cProfile）：**

| 指标 | pre-W2 | post-W2 | Δ |
|---|---:|---:|---|
| `posix.fork` self | 816.3 ms | **0 ms** | **-816 ms (-100%)** |
| `posix.fork` calls | 124 | 0 | -124 |
| `_thread.lock.acquire` self | 2710.8 ms | 1666.2 ms | -1044 ms (-38.5%) |
| `_thread.lock.acquire` calls | 19 | 4 | -15 |

Post-W2 的 lock.acquire 100% 来自单次 `_compute_face_weights_gpu` 的 `pool.map()` 等待——即 Stage D workers 的 steady-state wall time，不再是池生命周期开销。

### 3.2 W4 — s6 Fast-path Tracer GPU 向量化（commit `ff461a3`）

**问题：** `_fastpath_trace_loops_numpy` @ s6:379 是 #1 main-thread 热点 —— 275,426 次 / 1822 ms self。逐 cube 纯 Python 遍历度数=2 的邻接表收集 loop edges。

**设计（T5a spike, `tmp/cpu_worker_optim_design/t5a_s6_fastpath_gpu_sketch.md`）：**
- 纯 PyTorch padded-walk（**不用 Triton**），N 个 cube 同步推进 max_points 步
- 状态张量 `(curr, prev, start, visited)` shape `(N,)` / `(N, max_points)` 全部 device-resident
- 输出 GLOBAL CSR `(loop_count, loop_offsets, edge_ids)` 替代 `List[List[int]]`
- 关键 tiebreaker：外层 start 迭代顺序 `p=0,1,...,max_points-1` 与 numpy 对齐；inner 首步 `prev == -1` 时取 `adj[start, 0]`（即 numpy 的 `n0` 分支）

**实现流程（TDD）：**
- T5b (`244ad96`)：写 `test_s6_fastpath_tracer.py`（ImportError red）
- T5c (`2cd8167`)：`_fastpath_trace_loops_gpu` numpy-delegating stub（TDD green，3/3 pass）
- T5d (`ff461a3`)：padded-walk 真实现 + 集成到 call site 1081-1115

**测量（post-W4 vs post-W2 main-thread cProfile）：**

| 指标 | post-W2 | post-W4 | Δ |
|---|---:|---:|---|
| `_fastpath_trace_loops_numpy` calls | 275,426 | **0** | -100% |
| `_fastpath_trace_loops_numpy` self | 1822.5 ms | 0 ms | **-1822 ms** |
| `_fastpath_trace_loops_gpu` self (NEW) | — | 7.3 ms | — |
| s6 stage wall | 3.432 s | **1.297 s** | **-2.135 s (-62%)** |

净降 1815 ms，远超 DoD #4 的 80% 目标（实际 99.6%）。F1/F2/F3 bit-exact 全过。

**遗留可捡的 10-20% (Option A)：** 当前集成走 CSR→list-of-lists adapter（~100-300ms 被 numpy 重打包吃掉）。若下游 consumer 改吃 CSR，可回收这部分。ROI 低，deferred。

### 3.3 W5 — s4 Part 2 Batched GPU UF（commit `d16c73c`）

**问题：** `_get_local_components_np` @ s4:734 是 #2 main-thread 热点 —— 275,541 次 / 1014 ms self。逐 cube Python union-find 对 `face_adj` 全局 face→neighbor 邻接表求连通分量。

**设计（T6a spike, `tmp/cpu_worker_optim_design/t6a_s4_uf_gpu_sketch.md`）：**
- 批量 label-propagation：`labels[j] = j`，迭代 `min(labels[i], labels[nbr])` 直到收敛
- 关键：numpy UF 的 root id 是 union 顺序决定的，但 **从不外泄**。外部可见的只是 (a) 组件顺序、(b) 组件内 face 顺序
- s2 已经对 `comp_face_val` 做了 `stable_sort by (cube, label)` → GPU 版用 `stable_sort` 自动复现 "first-occurrence 组件顺序 + slot 升序" 的 canonical 输出
- 所有所需 CUDA op（sort / min / gather / elementwise）**全部 deterministic，无 atomics** → bit-identical 可保证

**实现流程（TDD）：**
- T6b (`6d3dada`)：`test_s4_uf.py` red；**纠正 plan 里的 `face_adj` 约定错误**（原说 LOCAL indices，实际是 GLOBAL table）
- T6c (`276b715`)：numpy-delegating stub（4/4 green）
- T6d (`d16c73c`)：批量 GPU label-prop + `_labels_to_list_of_lists` adapter + 集成到 `_compute_component_points_gpu`

**测量（post-W5 vs post-W2 main-thread cProfile）：**

| 指标 | post-W2 | post-W5 | Δ |
|---|---:|---:|---|
| `_get_local_components_np` calls | 275,541 | **0** | -100% |
| `_get_local_components_np` self | 1013.5 ms | 0 ms | **-1014 ms** |
| `_get_local_components_gpu_batched` self | — | 0.1 ms | — |
| s4 stage wall | 7.215 s | **5.891 s** | **-1.324 s (-18.3%)** |
| s4 peak VRAM (N=275k, max_f~20) | — | 3.48 GB | 无 OOM |

**Adapter 开销：** `_labels_to_list_of_lists` 一次 GPU→CPU 搬运 + 纯 Python 分桶 ≈ 452 ms。下游 consumer 未改（保持 `List[np.ndarray]` 语义）。升级到 CSR-native consumer 可再降 400ms，未来如需可做。

---

## 4. 最终性能数据（T9 clean e2e, commit `1ced857`）

### 4.1 端到端 wall-time（非 cProfile instrumented, 3-trial median）

| HEAD | 描述 | wall (s) | Δ vs baseline |
|---|---|---:|---|
| `b6bb12c`（pre-V2, T0） | default MP baseline | **8.631** | — |
| (T0 experiment) | serial nw=1（对比用） | 75.312 | +66.68s（证实 MP 8.7× 必要） |
| `1ced857` (post-W2+W4+W5) | 本分支收尾 | **5.354** | **-3.277 s (-37.97%)** |

3 trial 原始值：`[5.224, 5.354, 5.421] s`（spread 0.2s，稳定）。DoD #5（≥3s reduction）**以 9.2% margin 达成**。

### 4.2 Top-20 Residual 热点（post-W4+W5, `tmp/cpu_profile/t9_final_hotspots.txt`）

| rank | self (ms) | calls | 函数 | 属于 |
|---:|---:|---:|---|---|
| 1 | 1648.5 | 4 | `_thread.lock.acquire` | 主线程等 MP worker（Stage D / s7 Phase 3） |
| 2 | 999.0 | 1 | `s7_rank_assign` | s7 Phase 3 Hungarian × 275k |
| 3 | 513.7 | 1 | `s6_collapse` | s6 assembly + CSR→list repack |
| 4 | 464.1 | 1 | `_labels_to_list_of_lists` | W5 adapter（可消除 ≈400ms） |
| 5-10 | 各 ~100-300 | 各 ~275k | numpy coercions (`asarray`/`tolist`/`astype`) | 跨 s4→s7 中间张量反复 CPU↔GPU |

### 4.3 VRAM

| 指标 | 值 |
|---|---:|
| peak allocated | 5,687.8 MB |
| peak reserved | 13,220.4 MB |
| Spec 约束（+500 MB ceiling） | 未违反 |

### 4.4 DoD 全部满足

| # | Item | Status |
|---|---|---|
| 1 | 不用 Triton、不新增依赖 | PASS（纯 PyTorch + stdlib MP） |
| 2 | W2 fork 降 ≥90% | PASS — 124 → 0（100%） |
| 3 | W4 trace_numpy 消除 ≥80% | PASS — 275k → 0（100%） |
| 4 | W5 UF_np 消除 ≥80% | PASS — 275k → 0（100%） |
| 5 | e2e wall 降 ≥3 s | PASS — -3.28s（+9.2% margin） |
| 6 | F1-F3 bit-exact vs nw=1 goldens | PASS — 3/3 全 commit 通过 |

---

## 5. GPU Idle 分析 — 为什么还能降、要怎么降

### 5.1 GPU idle 变化（估算，非直接测量）

本分支 T9 只跑了 cProfile，没跑 nsys，所以 post-W4+W5 的 GPU util 是推算值：

| 阶段 | GPU kernel+memcpy | wall | GPU util | GPU idle |
|---|---:|---:|---:|---:|
| deep-profiling baseline (10.056s, gpu-pipeline `53dfca5`) | ~347 ms | 10.056 s | **3.5%** | **96.5%** |
| post-W4+W5（本分支收尾 `1ced857`） | ~500-800 ms（W4/W5 把 CPU 工作搬到 GPU，净增 GPU 调用） | 5.354 s | **~10-15%** | **~85-90%** |

**GPU idle 有降（96.5% → ~85-90%），但仍绝对主导。** 关键是 **wall 砍半**：W4/W5 省的不是 GPU 时间，是 **CPU 主线程卡在 Python 循环无法喂 GPU 的时间**。

> 精确 GPU util 数字需要在 HEAD 再跑一次 nsys（0.5d 任务）。下一 spec 的 T9 等价任务建议带上 `GPU util` 作为 DoD 指标（目标：从 ~10-15% 提到 ≥50%）。

### 5.2 5.354 s 里 GPU idle 的真实来源（T9 top-20 residual 分解）

| Residual 热点 | self (ms) | GPU 在做什么 | idle 原因 |
|---|---:|---|---|
| `_thread.lock.acquire` | 1648 | 闲置 | 主线程等 MP worker（Stage D `_count_uturns` 49.7s cube-worker **纯 Python BFS**）— worker 不触 GPU，主线程也没下一步工作可 dispatch |
| `s7_rank_assign` Phase 3 | 999 | 闲置 | 主线程跑 275k 次 `scipy.optimize.linear_sum_assignment`（**纯 CPU Hungarian**），GPU 完全闲 |
| `s6_collapse` assembly | 513 | 闲置 | CSR→list 重打包等 Python 整理 |
| `_labels_to_list_of_lists`（W5 adapter） | 464 | 做完，D2H + CPU 分桶 | 一次 `.cpu()` 强同步 + Python bucket loop |
| numpy coercion 合计（276k+ 次 asarray/tolist/astype） | ~759 | 闲置或小量 memcpy | 跨 stage 中间张量反复 CPU↔GPU 搬运 |

**三类根因：**
1. **真 CPU 算法**（Hungarian、BFS、数据结构构建）— 算法本身没搬上 GPU
2. **CPU↔GPU boundary 强制同步**（`.cpu()` / `.tolist()` / `.item()`）— 数据往返时 GPU 必须等
3. **MP worker 内部纯 Python**（Stage D `_count_uturns`）— worker 不触 GPU，主线程只能 block

### 5.3 Triton vs Python/MP 调度优化 — 互补，非替代

常见误解："Triton 只加速 GPU kernel；要降 GPU idle 应该优化 Python 调度 / MP，不该用 Triton"。**前半对，后半不完整。**

Triton 有两种功能：

**(a) 加速已在 GPU 的 kernel** — 降 kernel 执行时间。GPU 本来 busy 时才有杠杆。
> 在 GPU idle 85%+ 的当前 pipeline，这条路 ROI 极低。deep-profiling 已结论：s4 GPU 工作仅 37ms，Triton 化再快也省 <40ms。

**(b) Kernel fusion + 把 host-side control 搬到 device-side** — 这是本 pipeline 的真机会：
- 典型反例：`_build_adjacency_gpu` (1913ms self) 有 12×3×W 个 scatter launch，每个 launch 前后都有 Python 侧 loop 控制
- Python 侧的工作：dispatch 要时间、launch 有 overhead、中间 buffer 要分配、scheduler 空转
- Triton kernel 可把整个循环编译成单个 device 程序，Python 侧只发一次 launch → **GPU idle 降**（fewer launches + less host orchestration）
- 本质是把 Python 调度工作搬到 GPU，而非单纯"加速"

**降 GPU idle 的路径选择：**

| 瓶颈类型 | 工具 | 例子（本 pipeline） |
|---|---|---|
| 纯 CPU 算法需搬上 GPU | batched PyTorch 足够 | W4/W5 已做；Stage D BFS 可做 |
| 小算法 GPU 版（n<100） | PyTorch 可能低效，Triton 更好 | s7 Phase 3 Hungarian |
| 大量小 kernel launch + host loop 控制 | **必须** Triton | `_build_adjacency_gpu` fusion |
| CPU↔GPU 数据往返 | 消除 `.cpu()` / 保持 GPU 驻留（任何工具） | W5 adapter → CSR-native consumer |
| MP worker 纯 Python 开销 | 搬上 GPU（PyTorch 或 Triton） | Stage D W6 Angle 2 |

**所以："Python 调度 / MP 优化" vs "Triton" 不是非此即彼，都是降 idle 的手段。选哪种取决于具体瓶颈形态。**

> W4/W5 的路线就是典型的 "PyTorch 够用 → 先用 PyTorch"：padded-walk + batched label-prop 都没用 Triton，仍拿到 -3.28s。下一步到了 `_build_adjacency_gpu` fusion 这种有 device-side control flow 需求的瓶颈，PyTorch 表达不出，才必须 Triton。

---

## 6. 后续方向（按 ROI 排序）

ROI 估算标准：**`预期 e2e 降幅 / effort (天)`**。越高越优先。注意 wall 估算都是 cProfile self-time 的直接转化，实际降幅可能因 cProfile instrumentation 膨胀而略有偏差；下一 spec 做完要用干净 wall 验证。

### 6.1 🥇 首选：**Stage D GPU BFS（W6 Angle 2）** — 纯 PyTorch 可达，最高 ROI

**ROI：~0.3-0.5 s/d**

- **现状：** `_count_uturns` @ s4:305 共 49.7s worker wall（跨 275k cube，纯 Python BFS + list/dict/set）→ 主线程 `lock.acquire` 1648ms
- **方案：** 把 BFS 和 U-turn 计数搬到 GPU，复用 Stage B 已有的 CSR segments。可以纯 PyTorch（batched BFS + scan），**不必 Triton**
- **期望回报：** -1.0~1.5 s e2e
- **effort：** 3-5 d
- **风险：** BFS 的 early-termination 语义在 batched GPU 下需要小心（与 `_fastpath_trace_loops_gpu` padded-walk 类似的 state tensor 模式）；Stage D `_count_uturns` 的 tiebreaker 需要 audit
- **解锁：** 一旦 Stage D 释 GIL，W6 Angle 3 (ThreadPool) 重新进入 viable，但多半届时 lock.acquire 已降到 <500ms 不值得再做

### 6.2 🥈 快速清理：**`_labels_to_list_of_lists` 向量化** — 1 天快项

**ROI：~0.2-0.4 s/d**

- **现状：** W5 adapter 一次 GPU→CPU 搬运 + Python bucket loop，464ms self @ s4:966
- **方案：** 用 `np.argsort` + `np.split`（单次 numpy vectorized），或更进一步 sparse-COO 保 GPU 驻留
- **期望回报：** -0.2~0.4 s
- **effort：** 0.5-1 d
- **适配性：** 独立 micro-spec，无依赖，任何时候可插入

### 6.3 🥉 s7 Phase 3 batched Hungarian on GPU — Triton 首选（单独 spec）

**ROI：~0.15-0.3 s/d**

- **现状：** `s7_rank_assign.py:1175` Phase 3 逐 cube × 275,539 次 `scipy.optimize.linear_sum_assignment`，self 984ms
- **方案：** pad per-cube cost matrix 到 max `(n_loops, n_points)`；自定 Triton Hungarian kernel（对 small n 可 warp-parallel assignment）
- **期望回报：** -0.8~1.5 s
- **effort：** 4-7 d（Triton 教学成本 + 正确性验证）
- **注意：** 纯 PyTorch batched Hungarian 也可（e.g. `torch.nonzero` + greedy matching for small n），效率可能低 3-10×。若 6.1 + 6.2 做完 e2e 已到 ~4s，此项 bundle 与否看团队 Triton 预算

### 6.4 s7 `_build_adjacency_gpu` fusion — 必须 Triton，可与 6.3 bundle

**ROI：~0.1-0.2 s/d**

- **现状：** s7:634，1913ms self，内含 12×3×W scatter + U-turn 循环各自 launch
- **方案：** 单 Triton kernel 把 Python loop + 所有 scatter + U-turn 分支 fuse 一起
- **期望回报：** -0.5~1.0 s（主要省 launch overhead + 中间 buffer + host orchestration）
- **effort：** 4-7 d（Triton 首选；纯 PyTorch 表达不出 device-side control flow）
- **打包建议：** 与 6.3 同 spec（都在 `s7_rank_assign.py`，都需要 Triton），effort 合计 7-12 d，合计回报 -1.0~2.0 s

### 6.5 W4 Option A（CSR-native 下游）— 低 ROI，条件触发

**ROI：~0.05-0.15 s/d**

- **现状：** T5d 走 CSR→list-of-lists adapter，~100-300 ms 被 numpy 重打包吃掉
- **方案：** 把 s6 assembly block (lines 935-1001) 改吃 CSR 直接消费
- **期望回报：** -0.1~0.3 s
- **effort：** 1-2 d
- **触发条件：** 仅当 s6 assembly 因别的原因需要重构时顺手做；不建议独立立项

### 6.6 numpy coercion cluster — 不独立立项，随其他 refactor 消

- **现状：** ~759 ms 分摊在 276k+ 次 `asarray` / `tolist` / `astype`
- **方案：** 无独立解法。应在未来"s4→s7 全程 GPU 驻留"refactor 里顺带消除
- **effort：** 不可估算（取决于 umbrella refactor 形态）

### 6.7 已关闭路径（除非有新数据，不要重开）

- **删 MP**：T0 已证 serial 8.7× 慢（75.3 s vs 8.6 s）。
- **W1 s8 Bucket A 清理**：T2 审计 2 处都 load-bearing（drive arange / MP boundary transfer）。
- **W6 Angle 3 (ThreadPool)**：T7a 实测 GIL-holding 57.9%，死局。**重开条件：** 先做 6.1 把 Stage D 重写成释 GIL 的实现（那时 lock.acquire 本身已降到无需 Angle 3）。
- **W7 s7 orchestration cleanup**：T8a 查无 ≥200 ms mechanical 候选。

### 6.8 推荐行动顺序（把 e2e 从 5.354 压到 ~3s）

```
Phase 1（1 week, 纯 PyTorch / numpy，零 Triton 债）：
  6.2 _labels_to_list_of_lists 向量化  (0.5-1d, -0.2~0.4s)
  6.1 Stage D GPU BFS (W6 Angle 2)     (3-5d,   -1.0~1.5s)
  => 预期 e2e：5.354 → 4.0~4.2 s

Phase 2（需 Triton 预算，1.5-2.5 week）：
  6.3 + 6.4 s7 Phase 3 + _build_adjacency_gpu fusion  (7-12d, -1.0~2.0s)
  => 预期 e2e：4.0~4.2 → 2.5~3.2 s

条件触发：
  6.5 W4 Option A（仅 s6 assembly refactor 时）
  6.6 numpy coercion cluster（仅 s4→s7 全 GPU 驻留 refactor 时）
```

**Gate DoD（下一 spec 必带）：**
- F1/F2/F3 bit-exact（沿用本分支 runner + 三层确定性）
- 干净 e2e wall 达标（Phase 1 ≥1.0s，Phase 2 ≥1.5s）
- **nsys 测的 GPU util 显式记录**（从当前 ~10-15% 提升到目标值）
- cProfile top-20 不得重新引入 ≥200 ms 纯 Python 热点

---

## 7. 方法论遗产（必须保留给下个 spec）

### 6.1 F1-F3 回归门 —— 三层确定性，缺一不可

发现于 T1 实施过程：同一 pipeline 在"默认 MP + 无 hash seed"下 ~0.7% 顶点集合漂移，跨 pytest process（即 GENERATE vs RE-RUN）还会再漂移 ~0.9（fixture 顺序依赖的 cuDNN/cuBLAS autotune cache），跨进程即使固定前两者仍有 ~0.75 的 V 漂移（CUDA 本身原子 op 非确定）。

**三层保险（一处缺失就炸）：**
1. **SerialPool monkeypatch (`num_workers=1`)**：stage 里 `from multiprocessing import Pool as _Pool` 是函数作用域，所以 `mp.Pool = SerialPool` 一次生效。
2. **Subprocess-per-fixture**：每个 fixture 独立 Python 进程，避免 CUDA state 跨 fixture 累积。实现于 `corep_fast/tests/regression/_cpu_worker_optim_runner.py`。
3. **确定性 CUDA flags**（subprocess env 级）：
   ```
   PYTHONHASHSEED=0
   CUBLAS_WORKSPACE_CONFIG=:4096:8
   torch.backends.cudnn.benchmark=False
   torch.backends.cudnn.deterministic=True
   torch.use_deterministic_algorithms(True, warn_only=True)
   ```

Gate 执行约 2:45 / 3 fixture。下一 spec 重用本 runner 即可；破一层必挂。

### 6.2 测量纪律

- **clean wall-time 是唯一权威指标。** cProfile instrumentation 在本 pipeline 膨胀 15-25%（T4 post-W2 profiled run 14.9s vs 干净 baseline ~8.6s）。未来用 `tmp/cpu_profile/t0_driver.py --mode default` 测 wall，cProfile 只用于热点排序。
- **cProfile `self` 不含 C-extension。** T0 差点被这个蒙蔽（"worker 60ms Python self → MP 是纯开销" 的误判）。跨验证用 wall-clock，不要只信 cProfile self。
- **所有 GPU 测试在 116 空闲卡**（本分支用 GPU 4；本地 GPU 有其他推理负载污染时序，禁用）。memory: `feedback_profiling_on_119.md`。

### 6.3 并行 subagent 模式（大幅缩短 wall time）

本分支多处用 "多 subagent 并发 + 文件/GPU 解耦" 模式：
- **T0 实验**：2 agent 同时跑 (MP, serial) 在 GPU 3/GPU 4，15 min 得出决定性数据
- **T5a + T6a spike**：2 agent 并发读源码写设计 sketch（零文件冲突）
- **Wave 1**：T2 (s8 audit) + T5b (s6 test) + T6b (s4 test) 三并发，零文件冲突
- **Wave 2**：T5c + T6c 并发（不同 stage 的 stub）
- **Wave 4**：T5d + T6d 并发（不同 stage 的真 GPU 实现）
- **Wave 5a**：T7a (GIL spike, GPU 3) + T8a (s7 drill, 无 GPU) + T9 (clean re-profile, GPU 4) 三并发

关键解耦规则：**不同 stage 文件 + 不同 GPU + 只读 / 各自 commit**。串行只用于 W2（6-7 个 stage 文件都要改），以及测量任务（需先看 W2 committed state）。

---

## 8. Artifacts 索引

### 7.1 Commit chain（`post-profile-sync-elim`, 自 `0b1ef14` 后）

```
65eee1c  T10  handoff + next-spec
c0a2956  T7a  GIL spike (W6 Angle 3 dead — 57.9% GIL)
ee7a9c1  T8a  s7 drill-down (W7 skip — 0 mechanical candidate)
1ced857  T9   clean post-W2+W4+W5 re-profile (-3.28s, -38%)
ff461a3  T5d  W4 s6 padded-walk tracer + integrate
d16c73c  T6d  W5 s4 batched label-prop UF + integrate
2b72a98  T4   W6 angle decision (L_post_w2 = 1666 ms → Angle 3 selected)
c923e85  T3   W2 persistent MP pool (7 Pool sites replaced)
276b715  T6c  s4 UF numpy-delegating stub (green)
2cd8167  T5c  s6 tracer numpy-delegating stub (green)
09f003e  T2   W1 audit (0 deletions, 2 load-bearing annotated)
6d3dada  T6b  s4 UF test red
243b349  plan T6c face_adj fix (LOCAL→GLOBAL)
244ad96  T5b  s6 tracer test red
9061c7f  T5a, T6a spike design sketches
a3daaf5  plan T1 shipped-diverged note
949aed9  T1   golden gate F1/F2/F3 with 3-layer determinism
b6bb12c  T0   decisive MP vs serial experiment (MP kept)
```

总改动：52 文件，+3,801 行。

### 7.2 findings 文档

- `logs/findings_t0_mp_vs_serial.md` — T0 决定性实验
- `logs/findings_w6_angle_decision.md` — post-W2 L_acquire 测量 + angle 选择
- `logs/findings_w6_gil_spike.md` — T7a GIL-holding 测量
- `logs/findings_cpu_worker_post_fix.md` — T9 最终 summary + §4 next-spec

### 7.3 设计 / 审计文档

- `tmp/cpu_worker_optim_design/t5a_s6_fastpath_gpu_sketch.md` — s6 tracer GPU design
- `tmp/cpu_worker_optim_design/t6a_s4_uf_gpu_sketch.md` — s4 UF GPU design
- `tmp/cpu_worker_optim_audit/w1_s8_audit.md` — W1 Bucket A audit
- `tmp/cpu_worker_optim_audit/w7_s7_drilldown.md` — W7 s7 drill-down

### 7.4 profile raw

- `tmp/cpu_profile/t0_{default,serial}.{json,log,md}` — T0 实验
- `tmp/cpu_profile/results_main/main_thread_res256_{pre_w2,post_w2,final}.prof` — 三个节点 cProfile
- `tmp/cpu_profile/t9_clean_wall.{json,log}` — T9 干净 wall-time
- `tmp/cpu_profile/t9_vram.log` — peak VRAM
- `tmp/cpu_profile/t9_final_hotspots.txt` — post-W4+W5 top-20 热点

### 7.5 Spec / Plan / Handoff

- Spec：`docs/superpowers/specs/2026-04-17-cpu-worker-optim-design.md`（V2 data-driven）
- Plan：`docs/superpowers/plans/2026-04-17-cpu-worker-optim-implementation.md`（10 task）
- Handoff：`docs/superpowers/specs/2026-04-17-cpu-worker-optim-handoff.md`（V2 close-out）

---

## 9. 一句话总结

**e2e @ res=256: 8.631 s → 5.354 s（-3.28 s, -38%）**，通过 (1) 持久 MP 池消除 124 次 fork + 39% lock.acquire、(2) s6 纯 PyTorch padded-walk tracer 消 275k 次 numpy 调用、(3) s4 batched GPU label-prop UF 消另外 275k 次 numpy 调用达成。**不用 Triton，不引入新依赖，F1/F2/F3 bit-exact 全 commit 过，DoD 6/6 满足。** 下一 spec 目标：Stage D + s7 Phase 3 Triton port，再降 1.5-2.5 s。
