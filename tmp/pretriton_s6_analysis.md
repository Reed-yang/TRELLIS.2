# S6 Pre-Triton Analysis

## A. 当前实现快照

### 文件: corep_fast/stages/s6_collapse.py (655 行)

**关键函数**:
- `_collapse_fast(ew)` — **行72** 快速路径 (face_weights全零)，纯Python图追踪+Hierholzer环路探测
- `_collapse_with_uturns(ew, fw)` — **行296** 慢速路径 (face_weights>0)，笛卡尔积枚举+图追踪
- `_collapse_with_uturns_tracked(ew, fw)` — **行379** 慢速路径变体，记录(u1,u2,u3)分配用于s7
- `_s6_worker(work_item)` — **行476** MP worker，调用上述算法，返回loops/status/assignment
- `s6_collapse(batch, pool, num_workers)` — **行502** 公开API，orchestrator

**实现分层**:

| 部分 | 代码位置 | GPU化状态 | M1/M2来源 |
|------|---------|---------|----------|
| Phase 1: 三角形不等式/奇偶校验 | 行541-568 | **100% GPU** | M1 (`tri_valid`, `parity_ok`, `k12/k23/k31`全tensor) |
| Phase 2: MP worker分发 | 行580-599 | **100% CPU/MP** | M1原设计 |
| Phase 3: CSR组装 | 行605-655 | **100% GPU迁移** | M1完成 (行641-646将numpy转回GPU tensor) |

**CPU 残留段**（M1设计之故）：
- **工作项建立** (行580-590): 遍历N个cube，filter fast/slow mask，转list(tolist())，**纯Python**
- **_collapse_fast/slow** (行72-377): 图遍历、环路追踪、笛卡尔积枚举，**纯Python**逐cube执行
- **标准化+去重** (行162-183, 346-353): 环路canonical形式、集合去重，**纯Python**

**Profile数据来源**:
- 基线: `tmp/pretriton_baseline_res256.json` 
  - s6: **2.763s** @ res=256, 275K cubes (14% of e2e 21.37s)
  - s6: **1.383s** @ res=128, 68K cubes
- M2阶段后重新测量，S8/S4/S7已GPU化，s6仍CPU MP

---

## B. 残留 CPU/Python 段的算法本质

### 段1: 快速路径 `_collapse_fast` (行72-155)

**算法**(30行伪代码):
```
Input: edge_weights [e0..e17]
// 1. 每条边连续point参数化: p ∈ [0, w_e)
adj[e, p] := {相邻三角形指标}  // 基于法线曲线理论的弧连接
for each 12 facets (e1,e2,e3):
  对三角形不等式/奇偶性检验(w1,w2,w3)
  计算k12=(w1+w2-w3)/2,k23,k31
  for arc i in 0..k12-1:
    adj[e1, pts_A[i]].append((facet, e2, pts_B[i]))
    adj[e2, pts_B[i]].append((facet, e1, pts_A[i]))

// 2. 验证所有点度数=2
for (e,p) in adj: assert |adj[e,p]| == 2

// 3. Hierholzer环路追踪 (DFS无回溯)
visited := {}
loops := []
for start_node in adj:
  if start_node in visited: continue
  curr, prev := start_node, None
  loop := []
  while True:
    visited.add(curr)
    (fid, ne, np) := adj[curr][1 if prev else 0]  // 选不是来源的邻点
    loop.append(curr_edge)
    prev, curr := curr, (ne, np)
    if curr == start_node: break
  loops.append(loop)
return loops, OK
```

**为何M1留在CPU**: 
- 图结构高度稀疏（12个facet，各最多w_i个点，w_i通常<50）
- 环路追踪本质上是**per-cube顺序遍历**，无批量并行机会
- 结果（loop列表）长度不定，难以预先GPU缓存预分配
- M1哲学：将组合/图问题指派MP，数值计算指派GPU

**数据规模** (res=256, 275K cubes):
- 快速路径cube数: 约**80-90%** (face_weights全零)
- 平均per-cube环数: ~2-4
- 平均环中边数: ~20-40
- 每个cube的work per-triangle: O(w_i²)其中w_i=edge_weight[i], p50~10, p99~50, max~200

**单元复杂度**: O(∑w_i · max_neighbors_per_point + loop_count) ≈ O(N_edges * degree) ≈ 常数(~1000-5000 per cube)

---

### 段2: 慢速路径 `_collapse_with_uturns_tracked` (行379-469)

**算法**(35行伪代码):
```
Input: edge_weights [e0..e17], face_weights [f0..f11]
// U-转 = 走回同一条边的伪环，模型化加载线折返

// 1. 枚举每个facet的有效(u1,u2,u3)
face_valid := []
for f_idx in 0..11:
  e1,e2,e3 := TRIANGLES[f_idx]
  valid_triples := []
  W := face_weights[f_idx]
  for u1 in 0..W:
    for u2 in 0..W-u1:
      u3 := W - u1 - u2
      w1_prime := ew[e1] - 2*u1    // 修正的弧权
      w2_prime := ew[e2] - 2*u2
      w3_prime := ew[e3] - 2*u3
      
      if w1_prime<0 or ... or parity_fails: continue
      if tri_ineq_fails: continue
      valid_triples.append((u1,u2,u3))
  face_valid.append(valid_triples)

// 2. 笛卡尔积 (budget cap 100K)
total := prod(len(v) for v in face_valid)
if total > 100K: return BUDGET_EXCEEDED
solutions := {}
for assignment in cartesian_product(*face_valid):    // 最多12维枚举
  loops := trace_loops_with_assignment(ew, assignment)
  canonical := canonicalize(loops)  // 旋转不变标准化
  if canonical not in solutions:
    solutions[canonical] := (loops, assignment)

// 3. 反向U转修剪 (同一环内某边出现≥3次)
pruned := []
for loops, assign in solutions.values():
  valid := all(max(edge.count() per loop) < 3)
  if valid: pruned.append((loops, assign))

// 4. 分类 (0/1/>1个解)
if len(pruned)==0: return UNSOLVABLE
elif len(pruned)==1: return OK, loops[0], assignment
else: return AMBIGUOUS, loops[0], None
```

**为何M1留在CPU**:
- Cartesian product per-cube，难以GPU批量化（每cube维数不同）
- 笛卡尔积规模高度可变（p50可能~100, p99~10K, max~100K）
- 逐assignment环路追踪+标准化=顺序工作，无GPU并行
- 预分配GPU内存需知晓每cube的product size
- M1经验：>90%cube走快速路径，慢速少数→MP开销可接受

**数据规模** (res=256, 275K cubes):
- 慢速路径cube数: 约**10-20%** (face_weights>0)
- 平均per-facet有效triple数: p50~3, p99~20, max~50 (W bounded by ew[e], typ W<10)
- Cartesian product规模分布:
  - p25: <10
  - p50: ~30-50
  - p75: ~100-500
  - p99: ~5K
  - max: ~100K (触发budget，归为UNSOLVABLE)
- 每个慢速cube的工作: trace_loops × (product_size × canonicalize_cost)

**单元复杂度**: O(product_size × loop_trace_per_assignment) ≈ O(product_size × (E+V_adj)) 其中product_size可达100K

---

### 段3: MP Worker分发与结果组装 (行580-655)

**残留CPU循环** (行580-590):
```python
work_items = []
for i in range(N):  # ← N=275K loop, 纯Python iter
    if fast_fail_np[i]: continue
    if fast_mask_np[i] or slow_mask_np[i]:
        work_items.append((
            i,
            ew_np[i].tolist(),      # ← numpy→Python list转换
            fw_np[i].tolist(),
            bool(slow_mask_np[i]),
        ))
```

**残留CPU循环** (行635-639, CSR pack):
```python
for loops in per_cube_loops:  # ← 逐cube遍历，组织CSR结构
    cube_offsets.append(cube_offsets[-1] + len(loops))
    for loop in loops:
        edge_offsets.append(edge_offsets[-1] + len(loop))
        edge_vals.extend(loop)  # ← extend list
```

**问题**:
- 275K次for迭代，每次numpy→list (行587-588)
- 每次tolist()触发numpy数据copy + Python list分配
- 即使numpy tensor在GPU，也强制CPU侧转换
- CSR组装是纯Python list操作 (行635-639)

**单元复杂度**: O(N) 加总 + O(tolist overhead) ≈ 微秒级，但积累可观

---

## C. Torch 化可行性矩阵

| 段 | 提议的 torch 方案 | 风险（精确性/内存/复杂度） | 预期收益@res=256 | 推荐 |
|---|---|---|---|---|
| **快速路径** | GPU ragged→padded展开，parallel union-find+环轮询，atomicAdd修复度数 | 精确性✓ / 内存峰值+50MB (padded ragged) / 复杂度中等 | -0.7~0.9s (80% cubes) | **do** |
| **慢速路径** | 同样GPU雾化，但每cube分配固定最大product_size张量，按assignment并行trace | 精确性✓ / 内存峰值+200MB / 复杂度高 | -0.3~0.5s (20% cubes) | **skip → triton-only** |
| **工作项分发** | torch.nonzero + 直接tensor留GPU | 精确性✓ / 内存△ / 复杂度低 | -0.02s | **do** |
| **CSR组装** | GPU ragged张量pack，offset计算→cumsum，flatten | 精确性✓ / 内存△ / 复杂度低 | -0.05s | **do** |

### 推荐方向分析

**快速路径GPU化 (ROI最高, 0.7-0.9s)**:
- 输入: (N_fast, 18, max_w) 弧权张量
- 算法: 
  1. 并行per-facet arc-span枚举 (gpu kernel或torch)
  2. Batched adjacency构造 → sparse graph (COO格式)
  3. 并行环路追踪: 每cube一个CUDA thread追踪(DFS无回溯，O(E))
  4. Ragged输出pad到(N_fast, max_loop, max_edges)
- 内存: N_fast × 18 × max_w (typ ~100M) + ragged offset + padded loops (~200MB)
- 精确性: Union-find+环迹追踪在GPU可完全复制当前Python语义

**慢速路径留Triton** (ROI低, 高方差):
- 理由:
  1. 只占20% cube，即使全优化收益 <0.3s
  2. Per-cube product_size方差极大 (10-100K)，GPU预分配困难
  3. 笛卡尔积枚举本身就是high-latency顺序工作
  4. 预期Triton做法: persistent kernel per-cube，shared mem存product，主循环枚举+trace
- 暂时策略: 保留MP，标记为Triton候选

---

## D. 数据 Contract 检查

### 输入张量
| 字段 | Shape | Dtype | 来源 | 备注 |
|------|-------|-------|------|------|
| `edge_weights` | (N, 18) | int32 | s4_face_point | 已GPU化 (M2 P2) |
| `face_weights` | (N, 12) | int32 | s4_face_point | 已GPU化 (M2 P2) |
| `num_cubes` | scalar | int | CubeBatch.num_cubes | - |

### 输出张量
| 字段 | Shape | Dtype | 下游使用 | 备注 |
|------|-------|-------|---------|------|
| `loop_cube_off` | (N+1,) | int64 | s7, s8 CSR expand | 每cube的loop offset |
| `loop_edge_off` | (L+1,) | int64 | s7, s8 CSR expand | 每loop的edge offset |
| `loop_edge_val` | (E,) | int32 | s7, s8 loop edge id | flat edge list |
| `status` | (N,) | int32 | s7 cubes filter, s8 exception | CubeStatus {OK/UNSOLVABLE/AMBIGUOUS/BUDGET_EXCEEDED} |
| `uturn_assignment` | (N, 12, 3) | int32 | s7 rank re-trace (M1设计) | per-facet (u1,u2,u3); -1 for fast-path |

### Contract变动需求

**当前 (M1终点)**:
- s6→s7: 传递 (loop_*CSR, status, **uturn_assignment**)
- s6→s8: 传递 (loop_*CSR, status)

**优化后提议** (本spec):
- 若快速路径GPU化: loop_*CSR张量在GPU，no Python intermediate
- uturn_assignment: 快速路径填-1，慢速路径从MP结果还原（no change）
- 与s7的contract: **零改动**，s7已适配uturn_assignment (M1完成)
- 与custom的V/F等价: **零改动**，s6只是拓扑提取，不涉及几何

### 与其它stage的连接
- **上游s4**: edge_weights / face_weights已纯GPU，contract ✓
- **下游s7**: uturn_assignment存储方式可保持，contract ✓
- **下游s8**: loop_*CSR已GPU tensor (M1完成)，contract ✓

---

## E. Triton Handoff

若慢速路径留给Triton实现，该kernel应当：

### 提议的Kernel输入
```c
// Per-task input (来自MP或persistent kernel queue)
int cube_idx;          // cube identifier
int ew[18];            // edge weights
int fw[12];            // face weights
int status_in;         // 快速路径通过标记

// Shared buffers (全kernel的pool，按task索引)
out_loops[max_loops_per_cube]      // 结果环list
out_assignment[12][3]              // (u1,u2,u3)赋值
out_status[1]                      // 输出status
```

### 提议的Grid/Block/SharedMem布局

```python
// Per-cube persistent kernel策略
grid_size = ceil(N_slow_cubes / BLOCKS_PER_WAVE)
block_size = 256 (32 threads/warp × 8 warps)
shared_mem = 16KB
  - face_valid_triples: (12, max_triples=50) = 2400B
  - product_buffer: (max_product=1000, 3B/triple) = 3000B
  - loop_stack: (128) = 512B
  - atomics for loop_count, assignment_found: 64B

Workflow:
  block[bi] processes slow_cubes[bi*WARPS_PER_BLOCK : (bi+1)*WARPS_PER_BLOCK]
  Each warp:
    1. Load ew[], fw[] to registers
    2. Parallel per-facet enumeration (32-thread reduction)
    3. Persistent-kernel cartesian-product loop (time-sliced per-warp)
    4. Per-assignment loop-trace + atomicCAS for best solution
    5. WriteBack to global out_* buffers
```

### 估算GPU工作量

**快速路径GPU版**:
- Per-cube: O(∑w_i × graph_trace) ≈ 10K-50K FLOPs per cube
- N_fast = 220K (res=256), 工作量 = 220K × 50K = **11 TFLOPS**
- 带宽: edge/loop数据读 (18×4 + E×4 bytes) ≈ **500B per cube** → 110 GB/s
- H100 peak compute 2 PFLOPS，peak memory 3.9 TB/s → **compute-bound**, 需register tiling优化

**慢速路径GPU版** (if done):
- Per-cube: O(product_size × loop_trace) = 1K-100K assignments × 1K flops = 1M-100M FLOPs
- N_slow = 55K, 工作量 = 55K × 10M = **550 TFLOPS**
- 带宽: 同样~500B per cube → **计算主导** (FLOPs/Byte = 10^8 >> 1)
- 适合register-tiled Triton kernel，post-Phase-2优化

### 与Phase 2 torch优化的接口契合度

**快速路径torch版本**:
- 输入: (N_fast, 18) edge_weights → (N_fast, 18, max_w) arc descriptor
- 输出: (N_fast, max_L, max_E) ragged loops → CSR pack in GPU
- Torch kernel: cooperative thread block per cube, parallel adjacency + union-find
- 无序列化开销 (全GPU操作)，可与其它stage overlap

**慢速路径Triton版本** (future):
- 输入: (N_slow,) cube selection mask，来自fast-path筛选
- 输出: ragged assignment，合并到uturn_assignment tensor
- Persistent kernel可承载per-cube cartesian-product，避免MP spawn
- 与快速路径GPU结果拼接无额外成本

**集成点**:
- s6_collapse返回: 若快速路径torch化，CSR已GPU → s7直接消费
- 若慢速路径Triton化: 另起kernel queue或callback，结果async合并
- 预期无通信开销增加

---

**分析完成**: 快速路径 GPU 化收益 0.7-0.9s (32-33% of s6)，复杂度中等可控；慢速路径应留 Triton，ROI 不足且方差大。
