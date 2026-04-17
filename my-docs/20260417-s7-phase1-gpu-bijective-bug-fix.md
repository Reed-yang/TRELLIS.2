# S7 Phase1 GPU Rank-Assign Bug: 发现、调试、修复

> 日期: 2026-04-17
> 分支: gpu-pipeline
> 影响 commits: a9407e1 / d802e0c (pre-triton W2 GPU parallel BFS)
> 修复范围: 新增 `(cube, edge_sequence)` 分组 + kth-True 选择器 + intra-loop U-turn CPU fallback

---

## TL;DR

`COREP_FAST_S7_PHASE1_GPU=1` (pre-triton W2) 在多 loop cube 上破坏几何拓扑：
三层嵌套 icosphere @ res=32 丢失 45% 顶点与面，但所有既有 regression 测试通过。

根因：GPU 实现 `_phase1_gpu_rank_assign` 用 `alive.argmax` 独立为每条 s6 loop 选 rank，
缺失 CPU `_match_loops_to_ranks` 的 `used_traced[i]=True` 双射消费约束。
多条 s6 loop 共享 edge sequence 时全部塌到最小 rank → centroid 重合 →
s8 `merge_vertices` 焊掉 → 几何大面积丢失。

修复策略（对齐 `custom/collapse_point.py:147-183` baseline 双射匹配算法）：
1. 用 `torch.unique((cube_id, edges_padded), return_inverse=True)` 按 `(cube, 完整 edge 序列)` 分组
2. stable sort + cummax segment reset 得到组内索引
3. 通过 `cumsum(alive) == within_idx + 1` 取第 k 个 True candidate
4. 对罕见的 intra-loop 连续重复 edge（U-turn 模式，如 `[..., 14, 14, ...]`）做 CPU
   单 cube fallback (`_s7_rank_worker`)

**最终效果**:
- 三层 sphere res=32 几何从 14026 V / 23740 F 恢复到 **25361 V / 50720 F**（对齐 d780be8 0.01% 内）
- s7 Phase1 rank 张量与 CPU **位级相等**（0 / 50497 mismatch）
- 30 个 regression test **全通过**，新增 triple-sphere A/B 测试
- 速度未丢：同节点公平对比下 W2=1 vs W2=0 在 res=128 加速 **1.45x** (s7 单独 **4.85x**)，res=256 加速 **1.37x** (s7 单独 **3.76x**)

---

## 1. 发现路径

### 1.1 用户报症：`corep_pipeline` 输出异常
三层嵌套 icosphere (subdivisions=3, 半径 r = 1.00 / 1.01 / 1.02) 在 HEAD (`gpu-pipeline`) 跑出：
```
V=14026  F=23740  (输出 .ply 文件大小 777K)
```
相比 commit d780be8 的 `V=25358 F=50708`（1.6M）掉 ~45%，用户要求深入定位。

### 1.2 先复建 baseline
- 创建 worktree: `git worktree add TRELLIS.2-d780be8 d780be8`
- 符号链接 `.venv`，复制测试脚本，提交到节点 116 GPU 0 跑
- 结果: `V=25358 F=50708` ✓（即 pre-W2 baseline）

### 1.3 又对照 custom/ 作 golden baseline
- `PipelineConfig(s8_impl='custom')` 用 `run_hybrid_pipeline`
- 结果: `V=25288 F=50436`（纯 custom s1-s8 参考）

三者对比：
| 来源 | V | F | 文件 |
|---|---|---|---|
| custom (s1-s8 全) | 25288 | 50436 | `tmp/test_fast/output_custom.ply` |
| corep_fast @ d780be8 | 25358 | 50708 | `tmp/test_fast/output_d780be8.ply` |
| corep_fast @ HEAD | **14026** | **23740** | `tmp/test_fast/output.ply` |

custom ↔ d780be8 差 <0.5% 视为数值对齐；HEAD 丢 45% 确认回归。

---

## 2. 按 commit 逐层二分定位

### 2.1 d780be8..HEAD 之间的 14 个 commit 分类
全部是 pre-triton Phase 2 GPU 优化及其周边：
- **W1** (S8 4-cube vectorized): `79c83fd`, `89cdc23` test
- **W2** (S7 Phase1 GPU batched BFS): `a9407e1`, `d802e0c`
- **W3** (S6 GPU fast-path): `4506ab1`, `bec5c48`
- 合并 + 默认开启: `d5c29ae`, `3e855f1`, `df09f7d`, `f244cb7`, `e81831c`
- docs + tests: 其它 4 个

`config.py` 里对应三个环境变量默认 ON：
```python
S8_4CUBE_VECTORIZED = env.get('COREP_FAST_S8_4CUBE_VECTORIZED', '1') == '1'  # W1
S7_PHASE1_GPU       = env.get('COREP_FAST_S7_PHASE1_GPU',       '1') == '1'  # W2
S6_FASTPATH_GPU     = env.get('COREP_FAST_S6_FASTPATH_GPU',     '1') == '1'  # W3
```

### 2.2 2^3 Flag 矩阵（8 GPU 并行）
8 个组合各用一卡，固定三层球 res=32：

| tag | W1 | W2 | W3 | V | F |
|---|---|---|---|---|---|
| w10_w20_w30 | 0 | 0 | 0 | 25358 | 50708 ✓ |
| w10_w20_w31 | 0 | 0 | 1 | 25358 | 50708 ✓ |
| w10_w21_w30 | 0 | **1** | 0 | **14026** | **23740** ✗ |
| w10_w21_w31 | 0 | **1** | 1 | **14026** | **23740** ✗ |
| w11_w20_w30 | 1 | 0 | 0 | 25358 | 50708 ✓ |
| w11_w20_w31 | 1 | 0 | 1 | 25358 | 50708 ✓ |
| w11_w21_w30 | 1 | **1** | 0 | **14026** | **23740** ✗ |
| w11_w21_w31 | 1 | **1** | 1 | **14026** | **23740** ✗ |

**W2 (S7_PHASE1_GPU) 单独决定输出**；W1、W3 完全无效。

---

## 3. 深入 s7 Phase1 GPU 代码找机制

### 3.1 定位到 `_phase1_gpu_rank_assign`
`corep_fast/stages/s7_rank_assign.py:903-1078` — 走 16 步批量候选遍历，
按每条 s6 loop 独立跑：
```python
# 每条 loop 枚举 2 * W_MAX = 32 个 (r0, nbr) 候选
# step 1 用 cand_nbr ∈ {0,1} 选 nbr0/nbr1
# step 2+ 用 "not prev_node" 继续沿 2-regular 图走
# 最后 alive.argmax(dim=1) 取第一个活着的 candidate
first_idx = alive.to(torch.int32).argmax(dim=1)
```

对比 CPU 路径 `_match_loops_to_ranks` (L260-309)：
```python
used_traced = [False] * len(traced_loops)
for in_loop in s6_loops:
    for i, t_loop in enumerate(traced_loops):
        if used_traced[i] or len(t_loop) != k:
            continue
        # ... match by forward OR backward cyclic shift ...
        used_traced[i] = True   # ← 关键：消费后标记
        break
    else:
        results.append([0] * k)  # 耗尽 → fallback 全 0
```

**差异**：CPU 对每条 s6 loop 从 pre-enumerated `traced_loops` 池里取一条 **未用过** 的
做前后向 cyclic match；GPU 无此 "已消费" 状态。

### 3.2 `custom/` 基线印证
`custom/collapse_point.py:147-183` 完全同样的 `used_traced[i] = True` 逻辑——
这是 CoReP 算法层面的 canonical behaviour，不是 corep_fast 加的私有 trick。

### 3.3 直接 A/B s7 Phase1 CPU vs GPU 张量
跑 s1-s6 一次，分别用 CPU / GPU 跑 s7 并 diff rank + match 张量：
```
post-s6: N_cubes=3388  total_loops=7009  total_edges=50497

[rank]  CPU sum=22136  GPU sum=19387     (GPU 偏低 12%)
[match] CPU sum=5082   GPU sum=5082
rank  mismatches: 4962 / 50497 (9.8%)
match mismatches: 266 / 7009
```

### 3.4 按 cube 分类 diff
```
684 diff cube 全为 status==OK
  .. 683 / 684 是多 loop cube
  .. 680 / 683 有重复 edge 序列 (dup-edge-seq)
```

冒烟案例 cube 28 (`n_loops=2`)：
```
loop[0] edges = [5, 13, 7, 17, 11, 16, 10, 15]
loop[1] edges = [5, 13, 7, 17, 11, 16, 10, 15]   # 完全相同

edge_weights: 5,7,11,13,15,16,17 均 = 2

CPU ranks:
  loop[0] = [0, 0, 1, 0, 0, 0, 0, 0]
  loop[1] = [1, 1, 0, 1, 1, 1, 1, 1]   # 互补分配
GPU ranks:
  loop[0] = [0, 0, 1, 0, 0, 0, 0, 0]
  loop[1] = [0, 0, 1, 0, 0, 0, 0, 0]   # 塌成同一 rank ✗
```

机制闭环：  
两条 s6 loop 共享 edge seq → 候选 alive mask 完全相同 → 都被 argmax 拉到第 0 个 True →  
插值公式 `t = (r+1)/(W+1)` 得到同一 3D 位置 → s8 `merge_vertices` 焊掉 → 几何塌陷。

### 3.5 为什么现有测试没发现
`corep_fast/tests/regression/test_s7_phase1_gpu_ab.py::test_s7_phase1_gpu_matches_cpu`
使用 `trimesh.creation.icosphere(subdivisions=2)` — 单层球。单层永远不会触发
同一 voxel 边上 W>1 + 重复 s6 loop 的拓扑。嵌套 / 近平行面 / 低分辨率才
是盲区。224/224 通过是**伪阳性信心**。

---

## 4. 修复迭代：三轮收敛

### 轮 1：按 `alive_bits` 分组（错）
尝试：对每条 loop 的 `alive[L, 2W]` 做 bit-pack → 32-bit 模式作为分组键。

```python
alive_bits = (alive.to(torch.int64) << bit_idx.view(1, 32)).sum(dim=1)  # (L,) int64
grp_key = (cube_id << 32) | alive_bits
# ... within-group index ...
```

效果：rank_diff 4962 → **1824**（减 63%）。所有 680 dup-edge-seq cube 修复了。

**残留 1824 差异**：`s7_rank_root_cause.py` 显示剩下 489 diff 非 dup-edge-seq 
cube。Spot cube 31 三条不同序列的 loop 恰好 `alive_bits` 值相同（都只 cand 0 alive，
但对应 3 个不同的 cycle），被错误分到同一组 → loop[1]/loop[2] 被 
`within_group_idx=1/2` 超出 alive count → fallback 到 0。

**教训**：alive bit pattern 只是 side effect，不是 loop 身份。真正的分组键必须是
**(cube_id, 完整 edge 序列)**。

### 轮 2：按 `(cube, edges)` 分组（正确）
```python
row_key = torch.cat([
    cube_per_loop.to(torch.int32).unsqueeze(1),   # (L, 1)
    loop_edges_pad.to(torch.int32),               # (L, K_max)
], dim=1)
_, group_ids = torch.unique(row_key, return_inverse=True, dim=0)
```

效果：rank_diff 1824 → **22**（减 99%）。

**残留 22 条 edges / 4 cubes** 都是 **intra-loop 连续重复 edge** 的 U-turn case：
- cube 966 单 loop: `[0, 12, 3, 17, 8, 14, 14]`（边 14 自交叉）
- cube 954 三 loop 其中一条: `[1, 12, 2, 16, 16, 11, 17, 7, 13, 5, 15]`（边 16 连续）
- 另 2 例类似

手动 Python walker（复现向量化逻辑）能从 `(5, 0)` nbr=0 正确走完 cube 31 loop[1]
得 `[0,0,0,2,1,1,1,2]`，证明 **邻接图正确** 且 walker 逻辑本身正确；但 GPU 向量
化 walker 在 U-turn 连续跳 `(edge=14,r=A) → (edge=14,r=B)` 时 `edge_match` 判断出
问题（具体子 bug 未完全拆解，留待 follow-up）。

### 轮 3：对 walker 失败的 cube 做 CPU 单 cube fallback
```python
# 检测 any_match=False 且 status==OK 的 loop → 找 owning cube
# 对每个 failed cube 调 _s7_rank_worker 重算，chosen_ranks 写回
if not any_match_cpu.all():
    ...
    for cube_i in failed_cubes:
        ...
        _, rank_lists = _s7_rank_worker((cube_i, ew_i, s6_loops, uturn_assign))
        for li_off, ranks in enumerate(rank_lists):
            for step_i, r in enumerate(ranks):
                chosen_ranks_cpu[li, step_i] = r
    chosen_ranks = torch.from_numpy(chosen_ranks_cpu).to(device)
```

关键细节（轮 3 修改过一次）：必须加 `status==_CubeStatus.OK` 过滤——非 OK cube 的
loop 本来就被 `ok_loop_mask` 屏蔽没有 alive，`any_match=False` 是预期；直接调
`_s7_rank_worker` 会踩到 `_trace_with_ranks_fast` 的 `KeyError (edge, -1)`。

**最终**: rank_diff = 0 / 50497, match_diff = 0 / 7009 ✓

---

## 5. 验证矩阵

### 5.1 新单测：三层球 A/B
新增 `corep_fast/tests/regression/test_s7_phase1_gpu_ab.py::test_s7_phase1_gpu_matches_cpu_triple_sphere`。

这个 test 在修复前就能复现 bug：
```
AssertionError: loop_edge_rank differs at 4962 / 50497 positions
(CPU sum=22136, GPU sum=19387)
```

修复后 PASS。

### 5.2 完整 regression 30/30 全过
```
test_e2e_gpu_ab.py .....                     [ 16%]
test_s1_ab.py ...                            [ 26%]
test_s2_ab.py ...                            [ 36%]
test_s3_ab.py .                              [ 40%]
test_s4_ab.py ..                             [ 46%]
test_s6_ab.py ...                            [ 56%]
test_s6_fastpath_gpu_ab.py ...               [ 66%]
test_s7_ab.py ...                            [ 76%]
test_s7_phase1_gpu_ab.py ....                [ 90%]  (3 旧 + 1 新)
test_s8_4cube_vectorized_ab.py ...           [100%]

======================== 30 passed in 172.50s =========================
```

### 5.3 Pipeline 输出对照
| 版本 | V | F |
|---|---|---|
| custom (纯 custom s1-s8) | 25288 | 50436 |
| corep_fast @ d780be8 | 25358 | 50708 |
| **corep_fast @ HEAD + fix** | **25361** | **50720** |

+3V / +12F 相对 d780be8 (0.01%) — 浮点 `scatter_mean` 累加顺序噪声，
rank / match 张量本身位级相等。

### 5.4 性能（同节点公平对比）
串行跑（避免 6 进程并行带来的 MP 池 744 worker 争抢 128 核）。

**res=128** (三次中位数):

| stage | W2=0 (CPU MP) | W2=1 (GPU+fix) | Δ | 加速 |
|---|---:|---:|---:|---:|
| s1 | 0.153 | 0.204 | +0.051 | 0.75x |
| s2 | 0.068 | 0.087 | +0.018 | 0.79x |
| s3 | 0.545 | 0.019 | -0.525 | 28.11x |
| s4 | 2.115 | 2.460 | +0.345 | 0.86x |
| s6 | 0.591 | 0.550 | -0.041 | 1.07x |
| **s7** | **1.958** | **0.403** | **-1.555** | **4.85x** |
| s8 | 0.048 | 0.033 | -0.015 | 1.46x |
| **e2e** | **5.560** | **3.837** | **-1.723** | **1.45x** |

V/F: W2=0 137713/275420 vs W2=1 137714/275424（+1V/+4F）

**res=256**:

| stage | W2=0 | W2=1+fix | Δ | 加速 |
|---|---:|---:|---:|---:|
| s1 | 0.673 | 0.813 | +0.140 | 0.83x |
| s2 | 0.089 | 0.092 | +0.003 | 0.97x |
| s3 | 0.002 | 0.002 | 0 | 0.97x |
| s4 | 4.850 | 4.984 | +0.134 | 0.97x |
| s6 | 2.497 | 2.636 | +0.139 | 0.95x |
| **s7** | **5.695** | **1.515** | **-4.180** | **3.76x** |
| s8 | 0.108 | 0.098 | -0.010 | 1.10x |
| **e2e** | **13.996** | **10.220** | **-3.776** | **1.37x** |

V/F: W2=0 551079/1102152 vs W2=1 551079/1102152（**位级相等**）

**W2=1 gave e2e 1.37-1.45x speedup** even after the fix's overhead 
(`torch.unique` + cumsum)；s7 本身 3.76-4.85x 加速。修复没有吃掉 pre-triton 的加速红利。

---

## 6. 改动清单

### 代码
- `corep_fast/stages/s7_rank_assign.py` (+75 行)
  - 替换 `first_idx = alive.argmax(dim=1)` 为按 `(cube, edges)` 分组 + kth-True 选择器
  - 末尾加 `any_match=False` AND `status==OK` 的 CPU fallback 循环

### 测试
- `corep_fast/tests/regression/test_s7_phase1_gpu_ab.py` (+80 行)
  - `_build_triple_sphere_mesh()` helper
  - `_run_s1_to_s6()` helper  
  - `_s7_with_flag()` 绕过 `os.environ` 的模块级读取限制，直接改 `_cfg.S7_PHASE1_GPU`
  - 新 `test_s7_phase1_gpu_matches_cpu_triple_sphere` 断言 rank_diff==0 & match_diff==0

### 文档 & memory
- `logs/findings_s7_phase1_gpu_regression.md` — 根因报告
- `logs/fix_s7_phase1_gpu_summary.md` — 修复总结
- `.claude/memory/project_s7_phase1_gpu_bug.md` — 记忆条目
- `my-docs/20260417-s7-phase1-gpu-bijective-bug-fix.md` — 本文

### 证据（tmp/）
- `flag_matrix_reconstruct.py` + log — 2^3 flag 矩阵
- `s7_phase1_ab_triple.py` + log — s7 张量 A/B
- `s7_rank_root_cause.py` + log — 按 cube 分类根因
- `s7_walker_debug.py`, `s7_gpu_walker_trace.py` — 邻接图 + 向量化 walker step-by-step 跟踪
- `test_fast_reconstruct.py` + triple-sphere `.ply` 三份 (HEAD_fixed, d780be8, custom)
- `flag_matrix_after_fix.log`, `profile_w2_compare_116.log` — 修复后矩阵 + 性能对照
- `profile_parallel_116.log` — 教训：profiling 不能并行 (MP 池抢核)

---

## 7. 已知 follow-ups

1. **GPU walker 在 U-turn 连续重复 edge 上的向量化 bug** 未根治，当前靠 CPU 
   fallback 兜底。实测 4 / 3388 cube 命中 fallback，额外 overhead 可忽略；但
   原则上 GPU walker 应该自己能走。拆解方向：检查 step 1 的 `cand_nbr` 是否
   对 `prev_node=-1` 初值敏感，以及 U-turn 节点的邻接 slot 顺序是否打断了
   "not prev" 逻辑。
2. **s7 Phase2 GPU `scatter_mean` 的浮点非确定性**：不同 run 可能多出 ±3V / 
   ±12F（0.01%）。不影响正确性，但复现一致性可加 deterministic 模式。
3. **`test_s7_phase1_gpu_matches_cpu` 既有测试的 env-var 操纵方式是 no-op**
   （config 模块早已 import）—— 虽然 "侥幸" 通过，但实际没做 A/B。建议未来
   将其迁移到像新 test 那样直接 monkey-patch `_cfg.S7_PHASE1_GPU`。
