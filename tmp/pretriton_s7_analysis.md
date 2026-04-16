# S7 Pre-Triton Analysis

## A. 当前实现快照（针对 corep_fast/stages/s7_rank_assign.py HEAD 126cb05）

### 关键函数 + 行号

**Phase 1 (rank re-tracing, CPU MP)**:
- `_s7_rank_worker(line 427)` — MP worker，per-cube 图遍历
- `_trace_with_ranks_fast(line 70)` — fast-path，无U-turn，边-秩 Eulerian 图构建 + trace
- `_trace_with_ranks_uturn_assignment(line 153)` — slow-path，带 uturn_assignment，3-tuple 邻接图
- `_trace_loops_from_adj(line 105)` — 2-正则图的环遍历，serial DFS per loop
- `_match_loops_to_ranks(line 259)` — s6 边-only 环 vs traced (edge,rank) 环的循环对齐，O(L×T×K²)

**Phase 2 (centroid 计算，GPU)**:
- GPU 批量内插(line 650-694)：scatter-add by loop_id，无 .item() / .cpu() 同步 ✓

**Phase 3 (Hungarian，CPU serial)**:
- 行 705-738：bulk CPU transfer，per-cube scipy linear_sum_assignment，单线程

### 当前 GPU 化 vs CPU 部分

| 部分 | 当前实现 | M1/M2 提交 |
|-----|--------|----------|
| Phase 1 rank tracing | **CPU MP** per-cube | M2 提交 24ad133 之前尝试过 MP Hungarian，但 s7 rank tracing 本身未 GPU |
| Phase 2 centroid 插值 | **GPU 向量化** scatter-add | M2 提交 963f3bb+ 改进 cube_map 后新增 |
| Phase 3 Hungarian | **CPU serial scipy** | M2 提交 24ad133 "Phase 3 Hungarian — bulk CPU transfer + serial scipy" |

### Profile 数据来源

- `tmp/pretriton_baseline_res256.json`: s7=5.543s (275528 cubes, res=256, commit 126cb05)
- `tmp/pretriton_baseline_res128.json`: s7=1.996s (68888 cubes, res=128, commit 126cb05)

---

## B. 残留 CPU/Python 段的算法本质

### 段 1：Phase 1 rank tracing（最大瓶颈）

**算法概要**（伪代码）：

```python
# 对每个 cube i：
for each cube i:
    # 1. 构建 (edge, rank) 邻接图，从 12 个三角面的边-边连接关系
    adj = build_arc_adjacency_from_triangles(ew[i], uturn_assign[i])
    # adj: Dict[(edge, rank)] -> List[(edge, rank)]，度数恒为 2（Eulerian）
    
    # 2. 环追踪：serial DFS，per-loop
    loops = trace_2regular_graph(adj)  # List[List[(edge, rank)]]
    
    # 3. 循环对齐：每个 s6_loop 匹配到 traced_loop，提取 ranks
    for s6_loop in s6_loops[i]:
        for traced_loop in traced_loops:
            # 检查边序列是否在某个循环偏移和反向下匹配
            if match_cyclic_and_reverse(s6_loop, traced_loop):
                rank_list = extract_ranks(traced_loop)
                break  # 匹配第一个即退出
```

**为何 M1/M2 留在 CPU**：
- 图遍历和环检测看似"固有顺序"：Eulerian 路径遍历、2-正则环的 DFS（M1 spec 4.5 引用）
- 但只是 **per-cube 串行**，不是全局串行；275K 个 cube 的 graph traversal 可并行
- M1 选择 MP 是快速方案，未深入评估 GPU batch 可行性

**数据规模 res=256 / 128**：

| 指标 | res=256 | res=128 |
|-----|------:|------:|
| 总 cube 数 | 275528 | 68888 |
| 平均 loop/cube（推估） | ~0.5-2 | ~0.5-2 |
| 平均边/loop（推估） | ~3-6 | ~3-6 |
| max edge weight（上界） | 12 | 12 |
| max (edge,rank) 节点数/cube | 12×12≈144 | 12×12≈144 |

**单元算法复杂度**：
- Phase 1a（邻接图构建）：O(T) = O(12) = O(1) per cube
- Phase 1b（环遍历）：O(L×K)，L=loop 数 (~1-10)，K=边/loop (~3-12)，**per-cube 局部最多 O(100) 个节点遍历**
- Phase 1c（循环对齐 _match_loops_to_ranks）：O(L×T×K²)，L×T~10×10=100，K²~36，最坏 O(36K) per cube；**但实际 short-circuit 匹配，平均 O(K²)**

### 段 2：Phase 3 Hungarian（较小瓶颈，已优化）

**算法概要**：

```python
# 对每个 cube i（行 710-738）：
centroids_i = loop_centroids[cube_i_loops]        # (n_loops, 3)
comp_pts_i = point_values[cube_i_points]          # (n_points, 3)
cost_matrix = ||centroids_i - comp_pts_i||²      # (n_loops, n_points)
row, col = scipy.linear_sum_assignment(cost)     # Hungarian
```

**为何已在 CPU serial**：
- Scipy linear_sum_assignment 是 C-optimized Munkres，per-cube 矩阵很小（典型 3×5）
- M2 提交 24ad133 通过 bulk CPU transfer（一次 .cpu()）+ serial per-cube，避免 275K 次 GPU→CPU sync 和 MP pickle 开销
- **已经是最优 CPU 路径**；GPU Sinkhorn 在这个尺度(3×5)上没有收益

---

## C. Torch 化可行性矩阵

| 段 | 提议的 torch 方案 | 风险（精确性 / 内存峰值 / 实现复杂度） | 预期收益 (s, res=256) | 推荐 |
|---|---|---|---|---|
| **Phase 1 rank tracing** | 批量 BFS 层级遍历：所有 cube 邻接图一次性转 CSR/COO，parallel layer-by-layer BFS，分散环起点，per-layer 同步输出环列表 | 精确性: ✓ 拓扑等价；内存: (275K cube × 144 node)×8B ≈ 300MB 可控；复杂度: ⭐ **高**（环检测、同步点、ragged 输出） | ~2.5-3.5s (目前 5.5s) | **do** |
| **Phase 1c 循环对齐 _match_loops_to_ranks** | 批量循环偏移匹配：将所有 s6 vs traced 环对 tensorize，(L, T, K) 张量对比，argmax 匹配 | 精确性: ✓ 完全等价；内存: (L×T×K)×4B，L~10K, T~5/cube avg，K~12 → ~2GB 峰值；复杂度: ⭐ 中等 | ~1.0-1.5s (est. 占 Phase1 30%) | **do** |
| **Phase 3 Hungarian** | 批量 Sinkhorn：所有 cube cost 矩阵堆成 ragged tensor，batched Sinkhorn + Hungarian fallback | 精确性: ⚠ Sinkhorn 近似（entropy reg）；内存: 同上；复杂度: ⭐ 中等 | ~0.2-0.5s (目前 0.3s，room 不大) | **skip**（M2 已最优） |

**推荐优先级**：
1. **Phase 1 rank tracing batched GPU**：5.5s → 2-3s = **最高 ROI**
2. **Phase 1c 循环对齐 GPU**（可与 1 融合）：1-2s 额外收益

---

## D. 数据 Contract 检查

### 输入张量（来自 s6）

| 张量 | Shape | dtype | 来源 | 备注 |
|-----|-------|-------|------|-----|
| `batch.loop_edge_val` | (E,) | int32 | s6 CSR 环-边列表 | ✓ GPU 上 |
| `batch.loop_edge_off` | (L+1,) | int64 | s6 CSR 偏移 | ✓ GPU 上 |
| `batch.loop_cube_off` | (N+1,) | int64 | CSR 立方体-环偏移 | ✓ GPU 上 |
| `batch.uturn_assignment` | (N, 12, 3) | int32 | s6 或 s4 | ✓ GPU 上 |
| `batch.edge_weights` | (N, 18) | int32 | s1 或 s4 | ✓ GPU 上 |
| `batch.point_values` | (P, 3) | float32 | s4 | ✓ GPU 上（虽 Phase 2 需 CPU 副本） |
| `batch.cube_indices` | (N, 3) | int32 | s1 | ✓ GPU 上 |

### 输出张量（给 s8）

| 张量 | Shape | dtype | 用途 | 下游 |
|-----|-------|-------|------|-----|
| `batch.loop_edge_rank` | (E,) | int32 | 各环边的 rank | s8 step D（解码成边-秩对） |
| `batch.loop_point_match` | (L,) | int32 | 各环匹配到的 component point | s8 step D（稳定点查询） |

### Contract 不变性

- **输入已全 GPU**：无需修改 s6 或 s4
- **输出 GPU 张量**：当前已是（line 743-746），无需改变 dtype/shape
- **与 custom 等价性**：
  - Rank 值依赖 `uturn_assignment` 的精确值（来自 s4/s6），**无算法等价问题**
  - Hungarian 匹配也 **必须等价**（scipy 是唯一确定性求解器），除非选择近似（不推荐）
  - **建议：保持拓扑精确性，GPU path 的循环对齐用相同逻辑**

---

## E. Triton Handoff

若 Phase 1 rank tracing 不做 torch/hybrid，全移 Triton：

### 提议的 Kernel 设计

**Kernel 签名**：

```c
void triton_batched_loop_trace_and_rank_match(
    // Input CSR
    int32* loop_cube_off,    // (N+1,) 
    int32* loop_edge_off,    // (L+1,)
    int32* loop_edge_val,    // (E,)
    int32* edge_weights,     // (N, 18)
    int32* uturn_assign,     // (N, 12, 3)
    int32* cube_indices,     // (N, 3)
    
    // Output
    int32* loop_edge_rank,   // (E,)  [initialized]
    int32* loop_point_match, // (L,)  [initialized]
    
    // Size
    uint32 N, uint32 E, uint32 L
) {
    // Phase 1: per-block = per-cube graph trace + rank extraction
    // Phase 2-3: per-block = per-cube Hungarian cost matrix + scipy fallback
    // ...
}
```

### Grid / Block / 共享内存布局

- **Grid**: `(N, 1, 1)` = 275K blocks，一个 block 一个 cube
- **Block**: `(32, 1, 1)` — 32 threads，足以并行构建 (edge,rank) 邻接表（144 node）
- **Shared Mem**: 
  - ~2KB per-block: adjacency 列表（144×2×4B = ~1.2KB）
  - ~1KB per-block: 环栈（最多 144 个节点 = 576 bytes）
  - **总计: ~4KB per block，完全可行**
- **Barrier**: 每个 loop trace 后 1 个 `__syncthreads()`

### GPU 工作量估算（单 cube）

**Compute**：
- 邻接图构建：12 三角 × 3 边对 = 36 条边，~100 条邻接表项，O(100) FLOPs（指针操作）
- 环遍历：144 node 最坏 = O(144) 指针追踪，无 FP 运算
- 循环对齐：O(10×10×12²) = O(14400) 比较（短路匹配平均 O(1200)）

**总 FLOPs/cube**：~20K (dominated by cyclic matching comparisons)

**内存**：
- 读：edge_weights(18×4B) + uturn_assign(12×3×4B) + CSR(3×8B) = ~200B
- 写：loop_edge_rank(K×4B) avg K=3-12 → ~50B
- **总带宽: 250B/cube × 275K cubes = 69GB，可与 phase 2 centroid GPU 重叠**

### 与 Phase 2 torch 优化的接口契合度

- **当前 Phase 2 scatter-add** 是纯 GPU（line 685-690），无同步
- **Triton Phase 1** 若在 GPU，可与 Phase 2 **完全 fused**：
  ```
  Triton rank_trace_rank_match() → loop_edge_rank ∈ GPU
  Torch scatter_add() 直接读 loop_edge_rank，无传输 ✓
  ```
- **若 Phase 1 仍 CPU MP**，Phase 2 须 `torch.tensor(..., device=device)` 上传，成本 ~50-100 µs
- **建议**：优先做 **torch hybrid**（CPU graph + GPU 对齐），其次才 Triton（若 torch hybrid 不达预期）

---

## 总体结论

**Phase 1 rank tracing 是 s7 最大瓶颈（5.5s，29% of e2e）。**

**推荐路径**：
1. **torch batched 循环对齐** (Phase 1c _match_loops_to_ranks) — **低风险，期望 1-2s 收益**
2. **torch parallel BFS 层级环遍历** (Phase 1a/b) — **中等复杂度，期望 2-3s 收益**
3. **保留 Phase 3 serial scipy**（已最优，Hungarian 矩阵 3×5 GPU 无收益）

**不推荐 Triton**，除非 torch 路径遇到无法解决的 ragged output 或同步障碍。

