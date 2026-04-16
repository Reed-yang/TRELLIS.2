# S4 Pre-Triton 分析

## A. 当前实现快照

### 关键函数
| 函数 | 行号 | 状态 | 说明 |
|------|------|------|------|
| `s4_face_point` | 66 | 入口点 | GPU/MP 路径选择 |
| `_compute_face_weights_gpu` | 1119 | **M2 P2 GPU** | 4 阶段 GPU→CPU MP pipeline |
| `_compute_face_weights_mp` | 132 | 遗留路径 | 纯 CPU 版本（按需切换） |
| `_compute_component_points_gpu` | 411 | **M2 P3 GPU** | SH clip + 最近点 snap |
| `_p2_uturn_worker` | 1094 | **Stage D CPU** | 分组 BFS + U-Turn 计数 |
| `_expand_pairs_gpu` | 817 | **Stage A GPU** | (cube, facet, tri) 对展开 |
| `_batch_plane_tri_with_clip` | 929 | **Stage A/B GPU** | 平面-三角交 + clip |

### GPU 化部分 (M2 P2)
**已完成** (commit 范围内):
- Stage A (GPU): pair expansion + plane-triangle intersection + Sutherland-Hodgman clip (line 817-1078)
- Stage B (GPU): compact + sort + group by (cube, facet) (line 1148-1166)
- Stage C (GPU→CPU): bulk transfer via `.cpu()` + `.numpy()` (line 1168-1173)

**仍在 CPU** (Stage D + E):
- Stage D (CPU MP): per-(cube, facet) 分组 BFS 图连通分量 + U-Turn 计数 (line 1189-1191, _p2_uturn_worker)
- Stage E (CPU): 散射结果到 (N, 12) fw 数组 (line 1193-1196)

### Profile 数据来源
- **Baseline**: `tmp/pretriton_baseline_res256.json` = 21.37s e2e, s4=4.82s (23%)
- **Baseline**: `tmp/pretriton_baseline_res128.json` = 7.41s e2e, s4=1.94s (26%)
- 当前分支: `gpu-pipeline` (M2 已合并，S4_GPU_FW=1 已启用)

---

## B. 残留 CPU/Python 段的算法本质

### 段 1: Stage D BFS U-Turn 计数

**算法描述** (伪代码, lines 305-395):
```
For each (cube, facet) group gi with segments segs[lo:hi]:
  1. Build segment graph:
     - nodes[] ← unique segment endpoints (tolerance ≤1e-8)
     - edges[] ← (u,v) for each segment, dedup
  2. Connected components via BFS:
     - visited ← union-find style, add neighbors to queue
     - components[] ← list of node sets
  3. Per-component U-Turn count:
     - endpoints ← nodes with degree 1 (graph leaves)
     - For each endpoint P: project onto facet edges (0..2)
       and count which edge(s) it lands on
     - U-Turn = pairs of endpoints on same edge: count//2
  Return: (cube_id, facet_id, uturn_count)
```

**为什么 M1/M2 留在 CPU**:
- Per-(cube, facet) 的 segment 数 K ≈ 1-30（小图，平均 2-3）
- BFS 连通分量找法本身是 O(K+K) = O(K)，非常快 (~0.1ms per group)
- **核心瓶颈** 不在 BFS 算法，而在 **Python loop + list/dict 开销**（line 1189: MP pool.map over G groups）
- GPU 批化收益有限：每个小图独立，难以 coalesce

**数据规模** (res=128/256):
- res=128: 68K cubes × 12 facets = 816K (cube, facet) 对
  - 有效段的 (c,f) 组 ≈ 20-30K （大多数 facet 无有效段）
  - 平均段数 K ≈ 2，max K ≈ 30
  
- res=256: 275K cubes × 12 facets = 3.3M 对
  - 有效段的 (c,f) 组 ≈ 100-150K
  - 平均段数 K ≈ 2-3，max K ≈ 40

**单元复杂度**:
- Per-group 时间: O(K) + O(K^2) 在最坏的完全图（K≈16时 ~256 ops）
- 但平均 K=2 → ~10 ops + MP fork/join overhead ~1-2ms
- **瓶颈**: G × 1-2ms MP overhead = 100-200K groups × 1-2ms = 100-200s **理论下界**
- 实际: 多进程摊薄后 ~4.8s @ res=256（line 1184-1191 控制并发数）

### 段 2: Component Points - CPU 枚举阶段

**算法描述** (lines 484-510):
```
For each cube ci with nc components:
  1. Get face_ids[lo:hi] from comp_face_val (CSR)
  2. Build component graph using face_adj connectivity
  3. Union-Find to extract connected components
  4. Return nc component face-id lists (pad with empty if <nc found)
```

**为什么留在 CPU**:
- face_adj connectivity 查询是 GPU 不友好的稀疏图操作
- Union-Find 实现 (lines 751-777) 需要 per-item parent[] 指针跟踪
- face_ids 数量变长，难以 vectorize（需要 CSR 重新划分）

**数据规模**:
- res=128: ~68K cubes, 平均 ~2 components/cube → ~136K (cube, comp)
- res=256: ~275K cubes, 平均 ~3 components/cube → ~825K (cube, comp)
- Per-cube 面数: 平均 ~2-5，max ~30

**单元复杂度**: O(|face_ids| × log|face_ids|) per cube
- 典型: 5 faces × log(5) ≈ 11 ops
- 但有 Path compression 和 Union by rank，实际接近 O(|face_ids| × α(n))
- GPU 化困难因子: **CSR/graph 导向的稀疏操作，难以批化为 dense tensor 运算**

---

## C. Torch 化可行性矩阵

| 段 | 提议的 torch 方案 | 风险（精确性 / 内存 / 实现复杂度） | 预期收益 (s, res=256) | 推荐 |
|---|---|---|---|---|
| **Stage D BFS U-Turn** | Batch 图 BFS via CSR 或 Triton kernel; 改良：每个 (c,f) 小图构建到 GPU buffer，用 warp-level BFS | 中度精确性风险（floating point tolerance edges），内存低（小图），实现复杂度高 | 0.3-0.5s（~90% 瓶颈优化） | **triton-only** |
| **Component 连通分量** | Torch sparse graph union-find 或 mask-based 连通分量 (batched) | 低精确性风险（确定性图算法），中等内存（CSR 存储），实现复杂度中等 | 0.1-0.2s | **do** |
| **Snap centroid to mesh** | 已 batched GPU (lines 625-731)，逐 component 当前用循环 (line 618)；改为整体矩阵操作 | 无风险（已全 GPU），内存中等（(P, max_k, 3) 张量），实现低 | 0.05-0.1s | **do** |

### C1. 为什么 BFS 推荐 Triton-only

- **Torch 路径瓶颈**: batch BFS 需要 CSR 稀疏矩阵 + 迭代更新
  - torch.sparse 不够高效（设备内存复制多）
  - Python for loop over BFS 迭代仍然需要多次 GPU 同步
  - 每个分组 K 太小（K≤30），无法 vectorize 利用 GPU SIMD
  
- **Triton 优势**:
  - 单个 kernel per-(batch of groups)，avoiding launch overhead
  - 在寄存器/SMEM 内存储小图邻接表（≤16 node × 8 bytes = 128B）
  - warp-level BFS: 一个 warp 处理一个 (cube, facet) group，同步开销 << Python
  - 预期: 10-50μs per group × 100K groups = 1-5s → vs 当前 3-4s，仅 20-50% 改善

- **结论**: Torch 无法显著优化（受限于小图规模 + 稀疏性），直接进 Triton

---

## D. 数据 Contract 检查

### 输入张量 (来自 s3)
| 字段 | Shape | Dtype | 来源 |
|------|-------|-------|------|
| `batch.cube_indices` | (N, 3) | int32 | s3 voxel grid 位置 |
| `batch.tri_offsets` | (N+1,) | int64 | CSR 三角形索引偏移 |
| `batch.tri_values` | (T,) | int32 | mesh 三角形 id |
| `batch.num_components` | (N,) | int32 | 每 cube 连通分量数 |
| `batch.comp_face_off` | (N+1,) | int64 | CSR 面索引偏移 |
| `batch.comp_face_val` | (C,) | int32 | component face ids |
| `mesh.triangles` | (F, 3, 3) | float32 | mesh 三角形顶点 |
| `mesh.faces` | (F, 3) | int32 | mesh 面顶点索引 |
| `mesh.face_adj` | (F, 3) | int32 | mesh 面邻接 |

### 输出张量 (给 s5+)
| 字段 | Shape | Dtype | 下游 Consumer |
|------|-------|-------|--------|
| `face_weights` | (N, 12) | int32 | s5 (edge flip logic) |
| `point_offsets` | (N+1,) | int64 | s8 (component centroid gather) |
| `point_values` | (P, 3) | float32 | s8 (surface snap 起点) |

### Contract 更新需求
**当前无需更改**：所有输入/输出 dtype 和 layout 已在 M2 P2/P3 中固定。

**Torch 化约束**:
- Component 连通分量 (Stage 2): 输出仍为 numpy array 列表 (line 501, `_get_local_components_np`)
  - 若改为 Torch，需要返回 ragged tensor 或 padded tensor
  - 建议: 保留 CPU numpy，但 Torch 化 snap 步骤（已全 GPU）
  
- BFS U-Turn: 输出仍为 Python list[(ci, fi, u)] (line 1194)
  - Torch 化需要 batch 输出张量 → 需要 scatter 逻辑改为 Torch

### 与 custom/ baseline 依赖路径
**V/F 等价依赖**:
- face_weights → edge flip → loop ranking → V/F 拓扑正确性
- 任何 face_weight 错误会导致 V 重复或缺失
- **当前**: _compute_face_weights_gpu 与 _compute_face_weights_mp bit-bit 等价 (已验证 M2 spec 2.10)
- **Torch 化**: 必须保证 BFS 图构建的数值稳定性（endpoint projection 精度 ≤1e-8 容差）

---

## E. Triton Handoff

### E1. Stage D (BFS U-Turn) Triton Kernel

**输入张量列表**:
```
segments_A:     (S, 3) float32  — 线段起点
segments_B:     (S, 3) float32  — 线段终点
group_offsets:  (G+1,) int64    — CSR 分组偏移
cube_indices:   (N, 3) int32    — cube voxel 坐标
facet_verts:    (12, 3, 3) float32 — 常数 facet 顶点相对位置
facet_edges:    (12, 3) int32   — facet 边索引
resolution:     scalar int      — voxel resolution
```

**Grid/Block/Shared Mem 布局**:
```
Grid:    (G, 1, 1)  — 每个 group 一个 block
Block:   (32, 1, 1) — 一个 warp per (cube, facet) group
Shared:  4KB
  - node_pool[512]:   float32[3] × 64 nodes (~768B)
  - adj_list[512]:    int32 × 256 edges (~1KB)
  - visited[64]:      uint32 × 2 (64 bits, ~256B)
  - endpoint_counts[18]: int32 × 18 edges (~72B)
Total:   ~2.1KB ✓ (H100 SMEM 128KB/block, 99% slack)
```

**FLOPs + 带宽估算**:
- Per-group BFS: O(K + K) = O(K) 其中 K ≤ 30
  - 节点编码/比较: 3×3 float ops
  - 邻接表遍历: 1 int op
  - Endpoint 检测: 2 int ops
  - 总: ~15K cycles per group (K=30, depth=5)
- 带宽: 3×S×4 (segments) + 2×G×8 (offsets) + N×12 (cube_idx) = ~50MB @ res=256
- **Arithmetic intensity**: ~15K cycles × 100K groups / 50MB = ~30 FLOPs/byte （低，受限于小图规模）
- **预期时间**: 100K groups × 100μs per group = **10s** (vs 4.8s 当前)
  - **结论**: Triton 在这里不会加速！瓶颈是 MPpool launch overhead 本身，不是算法

### E2. Stage C (Component Connectivity) Torch 优化

**若做 Torch 化** (而非 Triton):

**输入张量**:
```
comp_face_lists:    list[np.array]  — ragged face ids per component (CPU numpy)
mesh_face_adj:      (F, 3) int32    — GPU tensor
```

**改良方案**:
```python
# 1. Ragged → Padded (已在 s4 line 652-656 做过)
max_faces = max(len(f) for f in comp_face_lists)
face_ids_padded = torch.zeros((P, max_faces), dtype=torch.int64, device=device)
# ... pad with -1, mask 有效项

# 2. Batched Union-Find (GPU kernel or Triton)
# Input: (P, max_faces) padded face ids + (F, 3) mesh_face_adj
# Output: (P, max_faces) component_id (per face within component)
# → 然后 argmax per-component 找边界（复杂，skip for now）
```

**预期收益**: 0.05-0.1s （CPU numpy Union-Find 本身很快）

### E3. Snap Centroid (已全 GPU)

**当前状态** (line 625-731):
- 已全 batched GPU，无 CPU Python loop
- (P, max_k, 3) 向量化最近点计算 (Ericson §5.1.5)
- **无需 Triton 优化**（GPU tensor ops 已足够）

---

## F. 总结 & 优化建议

### 当前 s4 分段时间 (res=256, 来自 baseline JSON)
```
s4_total = 4.82s
├─ Part A+B (GPU pair/clip/compact)      ~0.3s  (6%)
├─ Stage C (GPU→CPU transfer)            ~0.05s (1%)
├─ Stage D (CPU MP BFS)                  ~4.3s  (89%) ← 瓶颈
├─ Stage E (CPU scatter)                 ~0.02s (0.4%)
├─ Part B (component_points_gpu)         ~0.15s (3%)
└─ snap_centroids_to_components (GPU)    ~0.00s (0%)
```

### 推荐执行顺序

**短期 (Pre-Triton, Torch化)**:
1. ✅ **Skip** Stage D BFS Torch 化 → 改为 **Triton-only**（Torch 无收益）
2. ✅ **Do** Component 连通分量 Torch化（但 ROI 仅 0.05s，低优先）
3. ✅ **Skip** Centroid snap（已全 GPU）

**后续 (Triton 阶段)**:
- Stage D BFS → warp-level Triton kernel
  - 风险: 小图尺度下 GPU 利用率低（可能比当前 MP 慢）
  - 缓解: 批量 100+ groups per block，用 block-level 同步
  - 预期: 4.3s → 2-3s（如果优化得好）

### 最高 ROI 候选

1. **[需要探索]** BFS worker 序列化开销优化：
   - 当前: MP pool 开销 ~1-2ms per 小 group
   - 改良: 批量 chunks 到单个 worker（减少 pickle 次数）
   - 预期: 4.3s → 3.5-4.0s（5-20% 改善，实施简单）

2. **[可跳过]** Component 连通分量 GPU 化：
   - 当前: ~0.05-0.1s
   - 改良: face_adj 稀疏图 → GPU sparse BFS
   - ROI: 仅 0.05s，复杂度高，不推荐

3. **[推荐 Triton]** 整体 Stage D→Triton kernel fusion：
   - 合并 segment 聚合 + BFS + endpoint 计数
   - 单个 kernel launch per-(batch of 100+ groups)
   - 预期: 4.3s → 2-3s（若 GPU 利用率达到 60%+）
