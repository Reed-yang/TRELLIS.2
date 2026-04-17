# S8 Pre-Triton Analysis

## A. 当前实现快照（针对 corep_fast/stages/s8_collapse.py 当前 HEAD）

### 关键函数与行号
- `decode_from_cubebatch` (line 141): 公共入口，选择 direct-tensor 或 legacy dict 路径
- `_cubebatch_to_tensors_direct` (line 534): M2 P1 direct path，GPU 全向量化，无 Python 循环
- `_cubebatch_to_dicts` (line 181): Legacy dict 转换，仅在 `USE_DIRECT_TENSOR_S8=0` 时调用
- `_process_shared_edges_from_tensors` (line 1480): M2 P3 入口，直接从 CubeBatch CSR 处理
- `_process_shared_edges_torch` (line 1143): M2 Phase 2 实验性 Torch 向量化路径（未默认启用）
- `process_geometry_vectorized` (line 830): Stage 2 v2 向量化几何，共享 4-cube 与 partial-cube 路径
- `_build_grids_from_tensors` (line 1420): M2 P3，构造 candidate 四叉网格，仅涉及参与的 ~100K cubes
- `_build_grids_from_cube_map` (line 1316): Legacy fallback（M2 已推荐弃用）
- `_weld_and_dedup` (line 1980): GPU torch.unique 顶点焊接与去重，97x 加速（0.11s @ res=256）

### 已 GPU 化部分 vs 仍 CPU 部分

**GPU 化（M1 stage 1 + M2 P1/P2/P3）**：
- 边枚举 (vectorized): `compute_global_edge_keys`, `enumerate_unique_edges`, `build_edge_neighbor_table` (line 301-832)
- 几何处理 (vectorized): `process_geometry_vectorized` (line 830-1141) 处理 74% edges (partial-neighbor)
- 顶点焊接: `_weld_and_dedup` 用 torch.unique (line 1980-2073)
- CubeBatch → 张量: `_cubebatch_to_tensors_direct` (line 534-596)

**仍 CPU 部分**：
- Step E fallback (line 1222-1596): 4-cube edges (~26% of edges, **~80% triangles**) 路由 Python `_process_shared_edge_geometry` + MP Pool
  - Grid 构造: `_build_grids_from_tensors` (line 1420-1477) 或 `_build_grids_from_cube_map` (line 1316-1365)
  - Python 几何: `_process_shared_edge_geometry` (line 1772-1970, 原 custom/)
  - Multiprocessing: line 1584-1596，condition `M >= 100_000` 时启用 Pool

**隐藏的 scalar sync**：
- line 572: `max(int(lengths.max().item()), 1)` — max_loop_len 单个 GPU→CPU sync

### Profile 数据来源
- 新数据: `tmp/pretriton_baseline_res256.json` (res=256, H100, icosphere subdiv=3, M2 最终态)
- 新数据: `tmp/pretriton_baseline_res128.json` (res=128, 同硬件)
- 历史分析: `my-docs/20260415-corep-fast-stage2-analysis.md` (Stage 1 终点与 v2 Torch 实验)

**当前时间** (res=256):
- s8: 7.899s (config 开启 `S8_DIRECT_TENSOR=1`, `S8_DIRECT_GRIDS=1`)
- res=128: s8 1.769s
- e2e 17.92s vs custom 141.05s = **7.87x**

---

## B. 残留 CPU/Python 段的算法本质

### 1. Step E: 4-Cube Candidate Fallback

**位置**: line 1222-1596（_process_shared_edges_torch） 或 1567-1596（_process_shared_edges_from_tensors）

**算法描述** (伪代码，源自 custom/collapse.py):
```
candidate_mask ← (neighbor_counts == 4)  # 26% 边
for each edge in candidate_mask:
  grid ← [4 neighbor cubes' data]
  tri_verts ← _process_shared_edge_geometry(grid)
    # Per-grid geometry logic:
    # 1. 从 4 cube 的 loop 里提取穿过该 edge 的 rank 序列
    # 2. 标准化 rank (flip for edges 2/6/3/7)
    # 3. 按 rank 分组点到 points_by_rank[rank][uv] 
    # 4. 检测 conditional-promotion: 全 4-cube 有 loop 但都断联
    # 5. 对每个完整 rank 组 (4 uv slot) 发射 4 个 fan 三角
  返回 (T*3, 3) 三角顶点数组
```

**为什么 M1/M2 留在 CPU**:
- 阶段诊断（my-docs/20260415-corep-fast-stage2-analysis.md §3.3）：**只有 0.05% 的 4-cube 边真正与向量化路径分歧**（9/16,980 on icosphere@res=64）
- 分歧原因（§4.1）：两套 **local-edge 编码约定** 间的不对齐
  - `_EDGE_OFFSET_TABLE` (line 1986-2005): neighbor position (0..3) 索引的 (axis, offset)
  - `custom/collapse.py::get_local_edge(dx,dy,dz)`: UV 坐标输入的逆映射；各 axis 编号不同
- `process_geometry_vectorized` (line 867-912) 尝试在 kernel 内重算 local_edge，但对所有 4-cube edges 未完全匹配
- 无法用纯张量谓词精确区分 divergent vs non-divergent (§4.2)：检测本身就是 `process_geometry_vectorized` 的工作，复杂度相当

**数据规模** (res=256):
- 4-cube edges: ~69K / 264K 边 (26%)
- 贡献三角: ~880K / 1,102M (80%)
- 平均每边工作：~12 loops/per cube × 4 cubes → ~48 loop-local-edge 组合提取 + rank 标准化

**算法复杂度**:
- Per-edge: O(4 cubes × max_loops × loop_len) ≈ O(4 × 8 × 12) = O(384) per-edge tensor ops
  - 标准化 rank: O(1)
  - 点分组 scatter: O(4 × MAX_RANK) = O(128) per-edge
  - 三角发射 fan: O(MAX_RANK × 4) = O(512) per-edge
- 总体: O(E_4cube × 1024) ≈ O(69K × 1024) ≈ 70M ops (低密度，多分支)
- 当前 Python + MP: ~3.3s @ res=256（整个 fallback 路径）

---

## C. Torch 化可行性矩阵

| 段 | 提议的 torch 方案 | 风险（精确性/内存/复杂度） | 预期收益 (s, res=256) | 推荐 |
|---|---|---|---|---|
| **Step E fallback 4-cube geometry** | 完整向量化 `process_geometry_vectorized` 对 4-cube + 移除 Python fallback；修复 local-edge 编码不对齐 | 精确性：中（需完整 encoding 对齐验证）/ 内存：低（4-cube edges 张量化后仍小）/ 复杂度：中 | -2.8s (3.3s → 0.5s) | **do** |
| Step E candidate 谓词窄化 | `neighbor_counts==4 AND any_neighbor.num_loops >= 2` (line 1206) 替换为 tighter 谓词，将 26% edges 缩至 <2% | 精确性：中-高（需 per-edge 精确诊断，见 stage2-analysis.md §3.3）/ 内存：低 / 复杂度：低-中 | -1.5s (混合 GPU 99.95% + Python 0.05%) | **do** |
| Grid 构造 (cube_map rebuild) | 已在 M2 P3：`_build_grids_from_tensors` (line 1420-1477) 仅涉及 candidate 的 ~100K cubes，而非全 275K；`USE_DIRECT_GRIDS_S8=1` 时启用 | 精确性：高 / 内存：低 / 复杂度：低 | -0.6s (已部分实现) | **done** |
| Multiprocessing 本身 | 保留 MP（line 1584-1596），但只对 M >= 100K 启用；cache CPU 数计算（line 1275） | 精确性：高 / 内存：低 / 复杂度：低 | 已 done (M2) | **keep** |

**推荐策略**：
1. **优先 do**: 完整向量化 4-cube 几何（修复 encoding）+ 窄化 candidate 谓词
   - 单独工作项：确认 `process_geometry_vectorized` 在所有 4-cube edges 上的 local-edge 编码完整性
   - 期望 ROI：-2.8s (最高)
2. **已 done**: Grid 构造 (M2 P3) + MP 自适应条件
3. **跳过**: 其他 Python 微优化（收益已大部分提取）

---

## D. 数据 contract 检查

### 输入张量 (来自 s7)
- **CubeBatch** (GPU tensors)：
  - `cube_indices` (N, 3) int32: cube 全局坐标
  - `loop_cube_off` (N+1,) int64: CSR offsets for loops per cube
  - `loop_edge_off` (L+1,) int64: CSR offsets for edges per loop
  - `loop_edge_val` (E,) int32: 局部 edge id (0..17)
  - `loop_edge_rank` (E,) int32: rank (0..7)
  - `edge_weights` (N, 18) int32: per-cube edge weights
  - `point_values` (P, 3) float32: loop 的 component_point
  - `point_offsets` (N+1,) int64: CSR offsets for points
  - `loop_point_match` (L,) int32: index into point_values per loop
  - `status` (N,) int32: exception marker (0=OK, !=0 exception)
  - `resolution` (int): grid 分辨率
- **CubeDataTensors** (M2 P1 中间，GPU)：通过 `_cubebatch_to_tensors_direct` 自动生成
  - `cube_indices`, `cube_edge_weights`, `cube_exception` (N,) bool, `loop_cube_offsets`
  - `loop_component_point` (L, 3) float32: 向量化导出（line 50-89）
  - `loop_edges_flat` (L × max_loop_len,) int32: 补齐的 loop edges
  - `loop_ranks_flat` 同上

### 输出张量 (送入下游: PLY writer)
- **vertices** (V, 3) float32: 焊接后的顶点（CPU）
- **faces** (F, 3) int32: 三角形面（CPU）
- 与 custom/ baseline 的 V/F 等价性要求：**bit-exact** 对 partial-cube edges，**topologically-equiv** 对 4-cube edges（当前通过完全 fallback 保证 exact）

### 与其他 stage 的 contract 需要变动吗？
- **输入 contract**: 无变动。CubeBatch 来自 s7 rank_assign，格式不变
- **输出 contract**: 无变动。输出仍是 (V, F)，feed 到 PLY writer
- **内部 M2 flag**: `USE_DIRECT_TENSOR_S8`, `USE_DIRECT_GRIDS_S8` 都已在 config.py，env var 可控制

### 与 custom/ baseline 在 V/F 等价性上的依赖路径
- **Partial-cube edges** (74%): `process_geometry_vectorized` 代码路径必须与 Python 完全一致（当前已达成 <0.002% 浮点误差）
- **4-cube edges** (26%): 当前通过完全 Python fallback `_process_shared_edge_geometry` 保证 exact parity
  - 若完整向量化，必须修复 local-edge encoding 不对齐，重新A/B测试
- **顶点焊接**: torch.unique 已经与 custom/ dict 去重在浮点精度范围内等价（float64 precision at merge_decimals=5）

---

## E. Triton handoff

若 4-cube fallback 不完全向量化或编码修复困难，遗留给 Triton 的 kernel 架构如下：

### 提议的 Kernel 输入张量列表
- **静态**（per-pipeline）：
  - `edge_axis` (E,) int32: X/Y/Z axis per edge (line 895)
  - `neighbor_cube_ids` (E, 4) int32: 4 neighbor cubes per edge，-1 表示缺失
  - `neighbor_local_edges` (E, 4) int32: local edge id per slot
  - `neighbor_positions` (E, 4) int32: UV position (0..3) per slot
- **动态**（per-batch）：
  - `cube_indices` (N, 3) int32: cube 坐标
  - `loop_edges_flat` (L × K,) int32: 补齐的 loop edges per cube
  - `loop_ranks_flat` (L × K,) int32: 补齐的 rank per cube
  - `loop_component_point` (L, 3) float64: component point per loop（注：float64 维持精度）
  - `loop_cube_offsets` (N+1,) int64: CSR offsets
  - `cube_exception` (N,) bool: exception marker
  - `cube_num_components` (N,) int32: 用于异常处理的 component 计数

### 提议的 grid / block / shared mem 布局
- **Grid**: `(E_4cube,)` blocks，E_4cube ≈ 69K @ res=256
  - 1 program = 1 edge
- **Block**: 32-256 threads/block（典型 64 threads）
  - 每 thread 处理 1 或多个 rank 组（max 8）
  - Rank 标准化 + 点分组在 warp-level 协作
  - 可选：per-thread 寄存器阵列维护 `points_by_rank[rank][uv]` (8×4×3 = 96 float64)
- **Shared memory**: ~4KB per block
  - Thread-local loop data 预加载（edge 的 4 个 neighbor cubes 的 loop info）
  - Warp 同步做 conditional promotion 连通性检查（per-edge BFS/union-find）
- **Output**: 
  - Sparse triangle emission：per-edge prefix-sum 收集 `(tri_count[E], tri_verts[T×3×3])`
  - 或：dense output `(E, MAX_TRIS_PER_EDGE=16, 3, 3)`，后续 compaction

### 估算的 GPU 工作量（FLOPs + bytes）
- **Per-edge compute**:
  - Rank 标准化（flip check + conditional）：~4 comparisons × 4 neighbors = 16 ops
  - Point gathering（CSR lookup + gather）：4 neighbors × 8 loops × 3 coords = 96 loads
  - Rank grouping scatter：4 neighbors × 8 ranks × 4 UV = 128 scatter ops
  - Fan triangle generation：MAX_RANK × 4 tri/rank = 32 tri = 96 verts = 288 coords
  - Total per-edge: ~500 arithmetic ops + 200 memory ops
- **总计** (E=69K):
  - Arithmetic: 69K × 500 = 34.5M ops (低密度，可从 shared mem + L1)
  - Memory: 69K × (loop data fetch + tri output) ≈ 69K × 8KB ≈ 550MB
  - HBM bandwidth: 550MB / 0.5s (预期核心时间) = 1.1TB/s (H100 HBM 3.3TB/s 的 33%，不饱和)
  - **Triton 收益来源**：(a) kernel fusion（loop data prefetch + geometry 一体）；(b) warp-local 分支（不需全局 torch.where）；(c) 寄存器维护 points_by_rank（避免 HBM 往返）
- **预期加速**: 3.3s (current Python) → 0.4-0.8s (Triton，3-8x on this kernel)

### 与 Phase 2 torch 优化的接口契合度
- 若 torch 路径完整向量化 4-cube（移除 fallback），Triton kernel 不需要写
- 若仅窄化 fallback predicate 至 <2% edges（~2K edges），Triton kernel 仍可写但收益有限（0.3K edges × kernel overhead 可能不值）
- **建议**: 
  1. 优先完整向量化 4-cube geometry（修复 encoding）→ 无需 Triton for step E
  2. 若发现无法完整向量化特定 edge 子类，再针对那个子类写 Triton kernel（小 kernel 多個）
  3. Triton 真正的长期投资应该在 step D partial-cube 的 kernel fusion（目前 10+ 个 kernel launch per edge）

---

## 总结与 ROI 排序

| 优化项 | 难度 | 预期收益 (s, res=256) | 建议优先级 |
|---|---|---|---|
| 1. 完整向量化 4-cube 几何（修复 local-edge encoding） | 中 | -2.8s (最高 ROI) | ⭐⭐⭐ |
| 2. 窄化 candidate 谓词（from 26% → <2% edges） | 低-中 | -1.5s (混合 99.95% GPU) | ⭐⭐ |
| 3. 已完成的优化（M2 P3 grid 构造、MP 自适应） | 低 | -0.6s (部分) | done |
| 4. Triton kernel 对 4-cube（如果 torch 无法完整） | 高 | -0.5s (仅在无法向量化时) | ⭐（conditional） |

**关键阻碍**（对应 charter 问题 1）：
- 当前在 line 1206 (`candidate_mask = kept_table.neighbor_counts == 4`) 处路由**所有** 4-cube edges 到 fallback
- 只有 0.05% 的边（9/16K on icosphere@res=64）真正分歧，其余 99.95% 可在 GPU 处理
- 要么：(a) 修复 `process_geometry_vectorized` 的 local-edge encoding 完整性，移除 fallback；(b) 或写 tighter predicate 缩小 fallback 范围（但需精确诊断哪 0.05% divergent）

