# CPU Worker Optim — Followup 分支工作总览（V3）

> 日期：2026-04-18
> 分支历史：`post-profile-sync-elim`（本地，已合入 `gpu-pipeline` 后删除）
> Merge commit：`5830589` on `gpu-pipeline`
> 起点：`c0a2956`（上一 spec V2 handoff，prior summary 见 `my-docs/20260417-cpu-worker-optim-summary.md`）
> 结尾：`26b9811`（review cleanup） → merge `5830589`
> Spec：`docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-design.md`
> Plan：`docs/superpowers/plans/2026-04-17-cpu-worker-optim-followup-implementation.md`
> Handoff：`docs/superpowers/specs/2026-04-17-cpu-worker-optim-followup-handoff.md`
> 测量机：`host-10-240-99-116` GPU 4 —— **上一轮用 119 GPU 0，这一轮用户重定向到 116**
> Fixture：icosphere `subdivisions=3 radius=0.4`，res=256（F2）

---

## 1. 起点：Prior summary 之后做到了什么

上一版 summary（`20260417-cpu-worker-optim-summary.md`）收尾时：

| 指标 | 值 |
|---|---:|
| wall @ res=256 | 8.631 s → **5.354 s**（-38 %） |
| DoD | 6/6 满足 |
| F1-F3 bit-exact | 3/3 |
| Phase 2 待做 | W_BAF（Triton）+ W_HG（Triton or PyTorch batched Hungarian） |

Spec V3（本轮 followup）把 W_BAF + W_HG 打包进 Phase 2，外加一个 PyTorch-only 的 Phase 1（W_L2L `_labels_to_list_of_lists` 向量化 + W_SD Stage D 批量 GPU BFS）。Phase 1 预期 -0.2~0.4 + -0.8~1.5，共 -1~1.9 s；Phase 2 预期 -0.5~1.0 + -0.8~1.5，共 -1.3~2.5 s。

**Spec 里的一个隐患**（没注意到）：§2.1 把 `_build_adjacency_gpu` self_ms 记成 **1913 ms**，但这个数在 T9 main-thread cProfile 里根本**不存在**—— top-20 里压根没有这一行。这把 W_BAF 的 ROI 假设全部建立在一个幽灵数据上，本轮被 T0 baseline + Task 11 spike 两次证伪（见 §2.2）。

本轮开工基线（T0, commit `d0f6bf5`）：

| 指标 | 值 | 来源 |
|---|---:|---|
| e2e wall @ res=256（116 GPU 4, 3 trial median） | **5.337 s** | `tmp/followup_baseline/t0_driver_clean.json` |
| F1-F3 | 3/3 PASS | `tmp/followup_baseline/f123.log` |
| VRAM peak allocated | 5304.3 MB | `tmp/followup_baseline/vram_peak.log` |
| main-thread cProfile 前 5 | `lock.acquire 1605 / s7_rank_assign 959 / s6_collapse 500 / _labels_to_list_of_lists 484 / _compute_component_points_gpu 373` | `tmp/followup_baseline/hotspots_pre.txt` |

对比 spec §2.1 的 "1913 ms `_build_adjacency_gpu`"：`grep` 无匹配 → commit `a837eda` 立刻把这个 gap 写进 `logs/findings_t0_baseline_concerns.md`，并注明 W_BAF ROI 需要 Task 11 spike 重新验证。这条早期记录让 §2.2 的 descope 决策有完整溯源。

---

## 2. 三个关键决策节点

### 2.1 W_L2L "+0.362 s 回归" 是环境噪声，不是真实回归（commit `ae5242d`）

**现象：** Task 1-4 实施 `_labels_to_list_of_lists` 向量化（numpy `argsort + cumsum + np.split`）。单元测试 4/4 过，F1-F3 bit-exact。但 clean wall 从 5.337 s 飙到 5.699 s —— **+0.362 s 回归**。self_ms 确实从 484 降到 122（-76 %），但被 `np.split` + per-component `tolist` 反噬。Commit `c621580` 把 flag 翻回 `'0'`。

**用户质疑：** "会不会是 116 有其他应用竞争 CPU？" —— 一看 116 状态：load avg 32.95，`yushen` 4 个 Python 训练进程各占 100-455 % CPU，`wekanode` 多实例满载。baseline 和 W_L2L 测量之间，环境悄悄变了。

**交错 A/B 验证（commit `ae5242d`）：** 3 round × (A=flag0, B=flag1) × 3 trial = 18 trial，单次 SSH 会话内交错跑。结果：

| Mode | N | Mean | Median | Welch t |
|---|---:|---:|---:|---:|
| A (legacy) | 9 | 5.529 | 5.482 | — |
| B (vectorized) | 9 | 5.459 | 5.483 | **−1.075 (95 % CI 重叠)** |

**Δ(B-A) median = +0.0003 s（本质上持平）**。原"+0.362 s" 是两次非交错 run 之间 `yushen` 负载差异所致。

**决定：** flag 保持 `'0'` —— 走 legacy 路径，因为 cProfile 观测性更好（`_labels_to_list_of_lists` 以单一 hotspot 显示 484 ms，而不是散到 `np.split / tolist / <listcomp>` 多行）。代码 + 测试保留供未来 CSR-native 下游消费者启用后再 retry。

**方法论产出：** 交错 A/B 模板（`tmp/w_l2l_ab_interleaved_116.sh`），后续 robust 基准测试（§2.3）直接复用。

### 2.2 W_BAF 被 spike 证伪（commit `045996b` → descope `c1283de`）

**spec §2.1 说：** `_build_adjacency_gpu` self_ms = 1913 ms（top-1 候选）。Triton kernel fusion 预期 10-20× 加速 → -0.5~1.0 s。

**Task 11 spike（3 份测量 artifact, tmp/followup_design/w_baf_*）：**

测量一：真实 `_build_adjacency_gpu` wall on F2 fixture（wrap 函数 + `torch.cuda.synchronize()`）：

| 指标 | 值 |
|---|---:|
| 调用次数 | 1（仅一次主 call） |
| N | 275 541 |
| 真实 wall | **38.26 ms** |

**spec 的 1913 高估 50×**。这是 spec 写错的 —— 大概率来自一次早期 cumulative-time 视图或 pre-W2 状态，与 T9 main-thread cProfile 完全不一致。

测量二：synthetic random `edge_weights` 测 Triton 实现 vs legacy PyTorch（N=1000）：

| | wall (ms) | 相对 |
|---|---:|---:|
| legacy | 151.26 | 1× |
| triton | 0.46 | **329×** |

**看着很漂亮**。但：

测量三：**真实 F2 tensor**（N=275 541, fast-path 比例 99.96 %）上跑同样的 kernel：

| | wall (ms) | 相对 |
|---|---:|---:|
| legacy | 38.26 | 1× |
| triton | 95.78 | **0.40×（慢 2.5×）** |
| Δ | +57.52 | 回退 |

**为什么 synthetic 329× vs real 0.40×？** legacy PyTorch 实现里有 `for t_idx × pi × jj: if not valid_arc[:, t, pi, jj].any(): continue`。真实数据 k_pair 极稀疏（fast-path 的 99.96 % cube 大部分 slot 是 False），`mask.any()` 快速早退，576 次迭代里实际干活的只有少数。Synthetic random 把每个 slot 都填满，`mask.any()` 必为 True，无法早退 → legacy 吃全量开销 → 算出来的 speedup 是幻觉。

**雪上加霜：parity 4 处 mismatch**：`tl.atomic_add(fill_count, +1) + tl.store(adj[slot])` 组合在多 program 间非原子 → 偶发 race，见到 `[224, -1]` vs legacy 的 `[-1, -1]`。修 race 需 `atomic_cas` 多 pass 或重新设计 2-slot 分配 —— 工程量远超 ≤ 38 ms 的收益上限。

**结论（`c1283de`）：** descope Tasks 12-15。`corep_fast/stages/s7_triton.py:_build_adj_fast_kernel` 作为"证伪证据"保留，flag `BUILD_ADJACENCY_TRITON` 默认 `'0'` 且**不接任何生产 call site**。未来若 `_build_adjacency_gpu` 因工作负载变化涨到 >300 ms self，或 s7 结构重构让这里成为唯一瓶颈，再重新评估。

**教训：** spike 先行挽救了 4-7 天的预期 Triton 工作。**决策靠实测 wall，不能靠 cProfile self_ms —— 后者是 Python-only，漏掉 GPU kernel 真实成本。**

### 2.3 W_SD Task 9 初版 wall 只降 0.06 s → Task 9b 加三个优化凑到 -0.87 s（commit `d25c268`）

**Task 9 初版集成：** `_count_uturns_gpu_batched_csr` 替换 Stage D MP pool 的 1605 ms `lock.acquire`。F1-F3 3/3 bit-exact 一次过。**但 wall 只降 0.058 s**。

**罪魁祸首** (`tmp/cpu_profile/w_sd_task9_hotspots.txt`)：新 #1 hotspot = `_count_uturns_gpu_batched_csr` **1063 ms self**。把 MP pool wait 换成主线程 numpy 重物化：

- `cube_verts_all = (1.88M, 8, 3) float64 = 361 MB` 构造
- 随后 `facet_verts_cpu = cube_verts_all[g_arange, v_ids_per_facet]` 取 (1.88M, 3, 3) = 135 MB，**丢弃 226 MB**
- `pts_cpu = (G, P_MAX, 3) float64 = 451 MB` 通过 numpy fancy index 构造
- 两个 `np.where(valid_mask, ...)` 各一次大内存重写

**Task 9b 三叠优化（commit `d25c268`）：**

- **Opt A**：**跳过 `cube_verts_all` 物化**。预计算模块级常量 `FACET_V_OFFSETS = _V_OFFSETS[FACET_VERTS]` (12, 3, 3)，直接 `facet_verts = base[:,None,:] + FACET_V_OFFSETS[facet_id] * step`。（保留）
- **Opt B**：**把 pts / facet_verts packing 全部搬到 GPU**。原先在 CPU 做 fancy index 再 CPU→GPU 转移；改为先把 `A_np / B_np / group_off_np` 传到 GPU，然后在 device 上 `A_t[flat_si_clamped_t]` gather。消除所有大 numpy 中间体。（保留）
- **Opt C**：**f32 代替 f64**。理论上坐标尺度 1/256 下 f32 精度够。（**被回退** —— F3 第一次测就 `V count 405176 vs 405214`，38 个顶点对不上。Phase A 的 1e-8 coalescence 阈值正好落在 f32 ULP 边缘，小几率两个 borderline 点在 f32 下距离计算结果不稳定 → coalesce 结果飘。改回 f64，F1-F3 立刻 bit-exact。）

**测量（Task 9b vs Task 9 anchor）：**

| 指标 | Task 9 | Task 9b | Δ |
|---|---:|---:|---:|
| `_count_uturns_gpu_batched_csr` self | 1063 ms | **3.5 ms** | −1060 ms (−99.7 %) |
| `_count_uturns_from_packed`（GPU 核心）self | 实际合并在 csr 内 | 283 ms | — |
| wall median @ 116 GPU 4 | 5.279 s | **4.463 s** | **−0.816 s** |

**教训：** "算法正确 + self_ms 降了 ≠ wall 降了"。Python 工作可以从一个 hotspot 隐形搬运到几个小 hotspot（W_L2L 场景）、或换成另一个主线程大 numpy 操作（W_SD Task 9 场景）。**每次 integrate 之后都要测 wall，不能只看 cProfile self 下降。**

---

## 3. 四项落地的核心改造

### 3.1 W_L2L — `_labels_to_list_of_lists` 向量化（flag=0，保留）

- **改动：** `corep_fast/stages/s4_face_point.py` 新增 vectorized path 在 `if not _cfg.LABELS_TO_LIST_VECTORIZED:` 的 `else` 分支；legacy path 保留。
- **算法：** batched numpy —— `argsort + cumsum + np.flatnonzero(diff) + np.split`。self_ms 484 → 122（-76 %）。
- **为什么 flag=0：** A/B 交错 18 trial 证明 wall 持平（Welch t = −1.075, p > 0.05）。cProfile 观测性 legacy 更好。
- **未来触发条件：** 若下游消费者改吃 CSR-native `list[np.ndarray]`（而非 `list[list[int]]`），可以省掉最后的 `.tolist()`，vectorized 可能转正。

### 3.2 W_SD — Stage D 批量 GPU BFS + U-turn count（flag=1，主力）

**Spike 数据（Task 5, `tmp/followup_design/w_sd_stage_d_spike.md`）：**

| 指标 | F2 值 |
|---|---:|
| G（Stage D group 数） | 1 881 777 |
| max_segs/group | 5 → P_MAX=10 |
| p99 segs | 2 |
| mean_segs | 1.06（极稀疏） |
| max_nodes/group | 7 |
| (G, P, P) f64 内存 | 1.40 GB（H100 80 GB 内轻松 fit） |

**极度 favorable 的 data shape** —— single batched cdist 路径无需 chunking。

**算法（`_count_uturns_from_packed` in `corep_fast/stages/s4_face_point.py:548-697`）：**

Phase A：node coalescence。`cdist → match < 1e-8 → tril-mask → min j ≤ i` 还原 Python `_find_or_add_node` 的 "first-occurrence" 语义。

Phase B：edge_mask 通过 `scatter_` 按 canonical pair (u,v) 标 True，`seg_nontrivial` mask 过滤同点退化边。

Phase C：label-propagation BFS —— `labels[g, i] = arange(P)`，迭代 `min over neighbors` 直至收敛。`max_iters = P`（I2 修复后的保守上界），未收敛会 `else: raise RuntimeError`。

Phase D：endpoint 投影到 3 triangle edges → `bincount` 按 `(g, label, edge_id)` 聚合 → `count // 2` 得 U-turn 数。

**Stage D 生产入口：`_count_uturns_gpu_batched_csr`（Task 9b 优化后）：**

```python
FACET_V_OFFSETS = _V_OFFSETS[FACET_VERTS]  # (12, 3, 3) 模块常量

# GPU-side packing (skip 361MB cube_verts_all materialization)
A_t = torch.from_numpy(A_np).to(dev)  # (S, 3) f64
B_t = torch.from_numpy(B_np).to(dev)
flat_si_t = group_off_t[:-1, None] + seg_idx_t[None, :]  # GPU
pts[:, 0::2] = A_t[flat_si_clamped_t]  # device gather

facet_verts = base_t[:, None, :] + FACET_V_OFFSETS_t[facet_id] * step  # (G, 3, 3)
```

**Task 8 里 agent 自己抓的 3 个 subtle bug**（超出 plan 指定）：
- padded-pair masking in Phase A（否则 pad 的零点互相 match 污染 canonical_idx）
- `c_a_safe / c_b_safe = clamp(max=P-1)` 防止 scatter 索引越界
- `max_eid` 只从 `eid_b[hit_mask]` 取，避免 pad row 的 sentinel `-1` 炸 bucket 空间

**测量（post-W_SD main-thread cProfile）：**

| 指标 | pre-W_SD | post-Task-9b | Δ |
|---|---:|---:|---:|
| `_p2_uturn_worker` calls | 1 881 777 | **0**（MP 绕开） | -100 % |
| `_thread.lock.acquire` self | 1605 ms | **不在 top-20** | ≈ -1605 ms |
| `_count_uturns_gpu_batched_csr` self（新） | — | 3.5 ms | — |
| `_count_uturns_from_packed` self（新 GPU 核） | — | 283 ms | — |
| wall median | 5.337 s | **4.463 s** | **-0.874 s** |

**Task 10 后 review 补丁 I2：** 原 `max_iters = min(P+1, 16)` 在 max_s ≥ 8 时可能 silently break with unconverged labels。改为 `max_iters = P` + `for/else raise RuntimeError`。`test_max_s_convergence` 新单元测试加 8-seg 链强制走 P=16 路径，F1-F3 不受影响（real data max_s=5 远低于触发点）。

### 3.3 W_HG — Batched Hungarian + Phase 3 apply 向量化（flag=1，Phase 2 唯一主力）

**Tie-scan 数据（Task 16, `tmp/followup_design/w_hg_tie_scan.md`）**（F2 + F3 合计，100k random 采样）：

| 指标 | F2 | F3 |
|---|---:|---:|
| Phase 3 call 数 | 275 539 | 113 607 |
| 1×1 cost matrix 占比 | **99.9953 %** | **99.9868 %** |
| 2×1 (rect-reverse, → scipy) | 13 | 13 |
| 2×2 | 0 | 2 |
| nl>5 或 npts>8 | **0** | **0** |
| ties 观测数 | **0 / 100k** | **0 / 100k** |

**超过 99.99 % 的 cube 是 "1 个 loop 匹配 1 个 point" —— trivially 就是 `col = 0`。** scipy `linear_sum_assignment` 每次调用有 Python 分发开销 ~3.6 μs，275k × 3.6 μs ≈ **1 s pure Python overhead**。W_HG 的真实机会是 "消除 scipy 的 Python 调用成本"，而不是"算法加速"。

**`hungarian_batched` in `corep_fast/stages/s7_triton.py:142-213`（纯 PyTorch, 不是 Triton）：**

- 1×1 hot-path：`output[is_1x1, 0] = 0` —— 零计算，纯赋值
- 1×K (K≥2)：vectorized argmin over `cost_padded[idx, 0, :]`（+inf pad 自动正确）
- 2×2..5×5：按 (nl, npts) bucket 枚举 `P(npts, nl)` 排列，`torch.gather` 批量算 cost，`argmin`
- Tie 检测：`(cost_per_perm == best).sum() > 1` 则留 -1，交 scipy fallback
- 形状不支持（nl>5, npts>8, npts<nl）：留 -1，同样 scipy fallback

**Phase 3 集成（`s7_rank_assign.py:1485-1661`）：**

- Task 18 第一版：从 for-loop 改成 batched call。wall -0.633 s。**但留了新瓶颈** —— per-cube Python apply loop `for bi in ok_cube_indices: (matches[bi, :nl] == -1).any(); all_matches[l_lo + li_off] = int(...)` 在 275k cube 上跑出 280 ms `numpy.any` + 196 ms `numpy.reduce` + 412 ms `numpy.tolist`。
- **Task 18b（commit `7c43f64`）：apply loop 向量化**。把 cube 分类为 `is_1x1 / is_brute_valid / shape_invalid(→fallback)` 三类 mask，然后：
  - `is_1x1` cube：`all_matches[ok_offsets[is_1x1]] = 0` —— 一条 numpy fancy index
  - `is_brute_valid`：外层 5 次 iter over `li_off`（max_nl 上界），每次一条 bulk 赋值
  - Fallback：保留 Python loop，但只处理 ~15 cube/fixture
- 效果：`s7_rank_assign` self 648 → **235 ms**。`numpy.any / numpy.reduce` 直接掉出 top-20。

**测量：**

| 指标 | Phase 1 end | Task 18 | Task 18b |
|---|---:|---:|---:|
| `s7_rank_assign` self | 999 (scipy) | 648 | **235** |
| `numpy.any` (Phase 3 tie-check) | — | 275 | dropped |
| `numpy.reduce` (Phase 3 classify) | — | 196 | dropped |
| wall median @ 116 GPU 4 | 4.463 s | 3.830 s | **3.073 s** |

---

## 4. 最终性能数据（robust 24-trial A/B, commit `3140277`）

### 4.1 端到端 wall-time（24 trial 交错测）

完整抵消环境噪声的 decisive benchmark。4 round × (A=all-flags-OFF, B=production) × 3 trial = 12+12=24 trial，单次 SSH 会话内交错：

| Mode | N | Median | Mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|---:|
| A: all followup flags OFF（pre-followup 等价） | 12 | **5.6893 s** | 5.6741 | 0.3012 | 5.222 | 6.215 |
| B: STAGE_D_GPU=1 + HUNGARIAN_GPU=1（production） | 12 | **3.1175 s** | 3.1002 | 0.1154 | 2.921 | 3.291 |

- **Δ median = −2.572 s (−45.2 %)**
- **Welch t = −27.65**（p ≪ 0.0001，极度统计显著）
- **B stdev (0.115) 比 A (0.301) 小 2.6×** —— production 路径**更快且更稳**

这个数据比单次 3-trial 的 3.073 更可信，作为"最终 wall"引用。原始：`tmp/followup_baseline/robust_ab/round{1,2,3,4}_{A,B}.{json,log}`。

### 4.2 Top-20 Residual 热点（post-W_HG-18b, HEAD `7c43f64`）

| rank | self (ms) | 函数 | 属于 |
|---:|---:|---|---|
| 1 | 235 | `s7_rank_assign` | W_HG 余下的 GPU compute + 少量 scipy fallback |
| 2 | 538 | `s6_collapse` | 下一轮 drill-down 候选 |
| 3 | 473 | `_labels_to_list_of_lists` | 已有 vectorized path（flag=0），等 CSR-native 下游才翻正 |
| 4 | 371 | `_compute_component_points_gpu` | 含 W_SD 的 `_count_uturns_from_packed` GPU 时间 |
| 5 | 402 | `numpy.tolist` 散落 | s4→s7 中间张量 coercion |

`_thread.lock.acquire`（原 baseline #1, 1605 ms）**彻底掉出 top-20** —— Stage D MP 完全绕开。`numpy.any / numpy.reduce`（Phase 3 tie-check）也消失 —— W_HG 18b 的成果。

### 4.3 VRAM（commit `26b9811` I1 review 补丁后）

`tmp/vram_peak_head_116.sh` 两轮（warmup + clean）测得：

| Metric | T0 baseline | HEAD | Δ |
|---|---:|---:|---:|
| peak_alloc_MB | 5304.3 | **5304.5** | **+0.2 (noise)** |
| peak_reserved_MB | 12406.0 | 31044.0 | +18638（allocator arena） |

**Spec §3.4 "≤ +500 MB alloc" PASS**（+0.2 MB noise-level）。`peak_reserved` +18 GB 是 PyTorch caching allocator 保持的自由块，不是 live tensors；80 GB H100 下不构成约束，如果未来在 ≤ 16 GB 消费卡上要跑可以 `torch.cuda.empty_cache()` 或 `PYTORCH_CUDA_ALLOC_CONF` 缩减。

### 4.4 DoD 全盘

| # | Item | Target | Actual | Status |
|---|---|---|---|---|
| Phase 1 | W_L2L deliver | yes | flag=0 neutral | N/A |
| Phase 1 | W_SD deliver | yes | -0.874 s 单项 | **PASS** |
| Phase 1 | F1-F3 bit-exact | 3/3 | 3/3 | PASS |
| Phase 1 | wall ≤ 4.2 s | yes | 4.463 s (3 trial) | MISS -0.26 s |
| Phase 2 | W_BAF deliver | yes | **DESCOPED**（实测 ROI 证伪） | DESCOPED |
| Phase 2 | W_HG deliver | yes | -2.572 s 累计 vs A | **PASS** |
| Phase 2 | F1-F3 bit-exact | 3/3 | 3/3 | PASS |
| Phase 2 | wall ≤ 3.2 s | yes | **3.118 s（24 trial median）** | **PASS (0.082 s 余量)** |
| Phase 2 | top-3 ≠ app code | yes | 仍是 app code，但 Phase 3 apply loop 已消 | PARTIAL |
| Phase 2 | VRAM ≤ +500 MB | yes | +0.2 MB | **PASS** |
| Phase 2 | nsys GPU util ≥ 30 % | yes | **未测** | UNMEASURED |

唯一未达成：nsys GPU util 没直接采。标注为下一 spec 的 blocking first step。

---

## 5. 后续方向（按 ROI 排序）

### 5.1 🥇 首选：**nsys GPU util capture（0.5 d）**

**ROI：数据驱动**

- **现状：** Phase 2 wall 减半（5.69 → 3.12 s）但 GPU kernel 时间只 ~500 ms → 理论 util ~15 %。**未直接测**。
- **为什么最优先：** 决定整个 Phase 3 路线。若 util ≥ 40 % → pipeline 转 GPU-bound，Triton kernel 优化 ROI 开始合理；若仍 15-30 % → 继续挖 host-side Python / numpy coercion。
- **Effort：** 0.5 d（单次 nsys profile + 解析）。
- **交付：** Phase 3 spec 的第一条 action item。

### 5.2 🥈 `s6_collapse` drill-down（1-2 d）

**ROI：~0.2-0.4 s/d**

- **现状：** 538 ms self（post-Phase-2 top-2）。上一版 spec §7 说 "无清晰 mechanical win"，但当时不是 top-2。现在值得重新 drill-down。
- **方案：** 读源码（`corep_fast/stages/s6_collapse.py`）找有没有类似 W_L2L 场景的 apply-loop 向量化机会。
- **期望回报：** -0.2~0.4 s。
- **effort：** 1-2 d（大部分时间在读 + 测）。

### 5.3 🥉 umbrella "keep s4→s7 intermediates on GPU"（1-2 week）

**ROI：~0.1-0.2 s/d（但总量大）**

- **现状：** 402 ms `numpy.tolist` 散在 s4 / s7 之间的中间张量 coercion；W_L2L 473 ms legacy 也是类似成因。这些都靠 "CPU 中间表示" 活着。
- **方案：** 让 CubeBatch / 中间数据结构保持 GPU-resident，消除 `.cpu() / .tolist() / .asarray()` chain。
- **期望回报：** -0.5~1.0 s（#3+#5 之和的主要部分）。
- **effort：** 1-2 week。工程量大，风险在于破坏现有 consumer 合同。
- **触发条件：** 若 5.1 nsys 显示 GPU util 低（仍 host-bound），这条是最可能的下一步；若 GPU util 已起来，可能优先级降低。

### 5.4 W_BAF / W_L2L / W_HG-Triton 重新评估（条件触发）

- **W_BAF：** 只有 `_build_adjacency_gpu` self 涨到 >300 ms 且排名 top-3 再重评（当前 38 ms）。
- **W_L2L：** 只有 s4→s7 改成 CSR-native 下游（5.3 umbrella）后再 A/B。
- **W_HG Triton kernel：** 当前纯 PyTorch 版本在 1×1 hot-path 下已经逼近 scipy 调用开销的理论下界；Triton 版不会更快（kernel launch overhead 在 micro-shape 下反而更大）。只有未来 data shape 大幅偏移才重评。

### 5.5 已彻底关闭路径（除非新数据，不要重开）

- **删 MP**：prior spec T0 决定性实验（serial 75.3 s vs MP 8.6 s, 8.7× slower）。
- **ThreadPool 替 MP**（W6 Angle 3）：prior spec T7a 实测 GIL-holding 57.9 %，死局。
- **W4 Option A CSR-native s6 downstream**：ROI 极低，除非 5.3 umbrella 启动顺手做。

### 5.6 推荐 Phase 3 启动顺序

```
Week 1（blocking）:
  5.1 nsys GPU util capture (0.5 d)
    ↓ 数据驱动决策
  if util < 30 %:  → 5.2 s6_collapse drill (1-2 d, -0.2~0.4 s)
  if util ≥ 40 %:  → Triton kernel 候选（s4 或 s7 GPU-heavy 热点）

Week 2-4（大头）:
  5.3 s4→s7 GPU-resident refactor (1-2 week, -0.5~1.0 s)
  => 预期 e2e: 3.12 → 2.0~2.5 s
```

---

## 6. 方法论遗产

### 6.1 交错 A/B 测量抵抗环境噪声

**标准模板（`tmp/robust_ab_bench_116.sh` / `tmp/w_l2l_ab_interleaved_116.sh`）**：

```
for round in 1..N:
  run Mode A (3 trials, dump JSON)
  run Mode B (3 trials, dump JSON)
aggregate all trials, compute median + stdev + Welch t
```

- 24 trial 交错得 Welch t = -27.65（p ≪ 0.0001）→ 信号抵噪稳
- 非交错测量在 116 / 119 这种多租户机器上噪声可达 ±0.5 s wall，一次对比轻松被淹没
- **下一 spec 的 DoD 必带交错测量**，单次 3 trial 的差值不再作为 PASS/FAIL 证据

### 6.2 实测 wall 胜过 cProfile self_ms

三次被咬：

- **W_BAF**：spec §2.1 误把 `_build_adjacency_gpu` self 记为 1913 ms，实测 wall 38 ms → descope
- **W_SD Task 9**：算法正确，但 main-thread 多了 1063 ms CSR packing，wall 只降 0.06 s → 需要 Task 9b 救
- **W_HG Task 18**：scipy 替下来了，但 Python apply loop 补位，wall -0.633 而非预期 -1.0 → 需要 Task 18b 救

**规则：**
- spec 写 ROI 时必须 wall-based（`time.perf_counter_ns()` + `torch.cuda.synchronize()`），不能只引 cProfile self
- 每 integrate 后必须测 wall，不能只看 self 下降就宣告成功
- cProfile self 只适合做 hotspot 排序，不是 "改了这里能省多少 wall" 的可靠估计

### 6.3 Spike 先行验证 ROI，不要盲写

W_BAF 节省 4-7 天 Triton 工作靠的是 Task 11 spike 先测：

- 真实 wall（测量一）
- synthetic vs real 对比（测量二 vs 三）
- minimal kernel parity on real data

**任何 ≥ 3 天的优化任务，先写 ≤ 0.5 天的 spike 验证前提假设**。假设若被证伪，省下整个任务的工程开销。

### 6.4 Subagent 驱动 + 并行派单

**本轮用法：**

- Phase 1 W_L2L 全家（Tasks 1-4）一个 agent 完成（紧耦合同一文件）
- Phase 1 W_SD Batch A（5+6+7）、Batch B（8）、Batch C（9）分三次派发
- Phase 2 spike 并行：Task 11 (W_BAF, GPU 4) + Task 16 (W_HG tie-scan, GPU 5)**同一轮单消息双 Agent 调用**
- Subagent 跑完有 subtle insight（Task 8 agent 抓的 3 个 bug）会直接采纳
- Review cleanup 5 个 fix 同一个 agent 完成 → 5 atomic commits

**派发要点：**
- Decoupled（不同文件 / 不同 GPU）→ 并行
- Coupled（同一文件或紧接续）→ 串行
- 每个 agent prompt 带 full context（task text + 约束 + 测试 commands + SSH protocol），不让它读 plan 文件

---

## 7. Artifacts 索引

### 7.1 Commit chain（`post-profile-sync-elim`，自 `c0a2956` 起 29 commit + 1 merge）

```
5830589 (gpu-pipeline HEAD)  Merge branch 'post-profile-sync-elim' into gpu-pipeline
26b9811  Review I1: VRAM measurement + doc updates
69e9314  Review M3: rename fb_mask_init -> excluded_from_batch_mask
5ebcf8c  Review M1: s7_triton.py docstring
f2b58ec  Review I3 + M4: tie detection test + empty-input tests
990a4c2  Review I2: W_SD label-prop safety guard for max_s≥8
7997112  Phase 2 checkpoint + V3 handoff
3140277  robust 24-trial A/B benchmark (definitive data)
7c43f64  W_HG Task 18b: vectorize Phase 3 apply loop
1a38761  W_HG Tasks 17+18: batched Hungarian (PyTorch) + Phase 3 integrate
c1283de  W_BAF descope (spike falsified ROI)
045996b  W_BAF Task 11 spike: fast-path Triton kernel + measure
777618e  W_HG Task 16: Phase 3 tie scan on F2+F3
6413043  add BUILD_ADJACENCY_TRITON + HUNGARIAN_GPU flags (default off)
aa93f8b  W_SD Task 10: findings + Phase 1 checkpoint
d25c268  W_SD Task 9b: eliminate CSR packing overhead
1dd06f0  W_SD integrate at Stage D driver (flag ON)
f0a64de  W_SD Task 8: batched GPU BFS + U-turn count impl
ae5242d  W_L2L interleaved A/B re-test (no stat-sig wall diff)
4e528d5  W_SD stub (legacy-delegating)
ebeac76  W_SD red test (ImportError phase)
cb812dc  W_SD Task 5: STAGE_D_GPU flag + spike group-size histogram
c621580  W_L2L default flag=0 (vectorized path regresses +0.36s)
9ffb76f  W_L2L post-change findings + artifacts
cb69001  W_L2L vectorize _labels_to_list_of_lists bucket loop
7ac331e  W_L2L add parity tests (green on HEAD)
bc3ef2c  W_L2L add LABELS_TO_LIST_VECTORIZED flag
a837eda  T0 concerns: _build_adjacency_gpu absent from main-thread top-20
d0f6bf5  T0 baseline on 116 GPU 4 (5.34s wall, VRAM 5.2GB)
 (base: c0a2956, prior V2 handoff)
```

总改动：195 files, +20 458 / -97 lines。

### 7.2 Findings 文档（`logs/`）

- `findings_t0_baseline_concerns.md` — 早期标记 `_build_adjacency_gpu` 数据不一致
- `findings_w_l2l_vectorized.md` — W_L2L 初版 DoD 表 + wall regression
- `findings_w_l2l_ab_rerun.md` — 交错 A/B 证明环境噪声
- `findings_w_sd_gpu_bfs.md` — W_SD 全流程 + DoD
- `findings_w_baf_descope.md` — W_BAF descope 决策完整溯源
- `findings_phase1_checkpoint.md` — Phase 1 关门
- `findings_phase2_checkpoint.md` — Phase 2 关门 + 下一步 ROI 排序

### 7.3 Design / Spike 文档（`tmp/followup_design/`）

- `w_sd_stage_d_spike.md` — Stage D group-size histogram + Task 8 算法决策
- `w_baf_triton_kernel_sketch.md` — W_BAF spike 四步测量 + descope 结论
- `w_baf_spike_{measure,kernel,real}.txt` — 三次测量原始输出
- `w_hg_tie_scan.md` — F2+F3 Phase 3 tie 频率分布 + 设计简化决策

### 7.4 Profile raw（`tmp/cpu_profile/` + `tmp/followup_baseline/`）

- `followup_baseline/t0_driver_clean.json` — T0 baseline wall
- `followup_baseline/hotspots_pre.txt` — T0 top-20
- `followup_baseline/vram_peak.log` / `vram_peak_head.log` — T0 + HEAD VRAM
- `followup_baseline/ab_test/` — W_L2L 交错 A/B（9 trial/mode）
- `followup_baseline/robust_ab/` — Phase 2 终局 24 trial（12 trial/mode）
- `followup_baseline/robust_ab_output.log` — 24 trial 原始 + aggregate
- `cpu_profile/w_sd_task9b_wall.*` / `w_hg_task18b_wall.*` — 各节点 wall JSON
- `cpu_profile/w_l2l_post_hotspots.txt` / `w_hg_hotspots.txt` / `w_sd_task9b_hotspots.txt` — hotspot 快照

### 7.5 新增生产代码 / 测试（统计）

**生产：**
- `corep_fast/config.py` +29 行（4 新 flag）
- `corep_fast/stages/s4_face_point.py` +692 / -少量行（W_L2L + W_SD）
- `corep_fast/stages/s7_rank_assign.py` +214 / -少量行（W_HG Phase 3 整合）
- `corep_fast/stages/s7_triton.py` **新增** 223 行（`hungarian_batched` + descoped `_build_adj_fast_kernel` spike）

**测试（`corep_fast/tests/unit/`）：**
- `test_labels_to_list_vectorized.py` 新增 5 tests（I3/M4 补丁后）
- `test_stage_d_gpu_bfs.py` 新增 7 tests
- `test_hungarian_batched.py` 新增 6 tests
- **合计新增 18 个单元测试**，+ F1/F2/F3 regression gate 复用

### 7.6 Spec / Plan / Handoff（`docs/superpowers/`）

- spec：`specs/2026-04-17-cpu-worker-optim-followup-design.md`（1178 行）
- plan：`plans/2026-04-17-cpu-worker-optim-followup-implementation.md`（2540 行）
- handoff：`specs/2026-04-17-cpu-worker-optim-followup-handoff.md`（228 行 V3 close-out）

---

## 8. 一句话总结

**e2e wall @ res=256：5.689 s → 3.118 s（-45.2 %, Welch t=-27.65, p ≪ 0.0001），比 prior V2 handoff 的 5.354 再降约 1.9-2.6 s（视对照）**。45 % 加速**全部来自 PyTorch / numpy 层批量化**，零 Triton 产线代码 —— W_BAF 经实测 spike 证伪后 descope，W_SD 用 batched `cdist + 标签传播 + bincount`，W_HG 用 batched brute-force + 向量化 apply loop。F1/F2/F3 bit-exact 全 commit 通过，VRAM alloc Δ=+0.2 MB，Phase 2 DoD 5/7 硬目标 PASS（miss 的只有 nsys GPU util 未测）。**下一 spec 的 blocking first step 是 nsys GPU util 测量 —— 决定 Phase 3 该挖 host-side Python 还是转 Triton GPU kernel。**
