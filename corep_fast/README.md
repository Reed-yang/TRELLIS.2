# corep_fast/ 包文档

> **corep_fast** 是 TRELLIS.2 中 CoReP（Combinatorial Representation）管线的高性能 GPU+多进程重写版本，替代原有 `custom/` 目录下的纯 Python 实现。在 H100 上 icosphere@res=256 下达到 **17.9s e2e**，相比 custom/ baseline（141s）加速 **7.87x**。

---

## 1. 文件组织架构

```
corep_fast/
├── __init__.py                  # 包入口
├── pipeline.py          (306L)  # 管线编排：corep_encode / corep_decode / corep_pipeline
├── containers.py        (442L)  # 核心数据容器：MeshTensors, CubeBatch
├── config.py            (127L)  # 运行时配置：Mode / StageConfig / BackendConfig / feature flags
├── constants.py         (114L)  # 立方体拓扑常量（8顶点 / 18棱 / 12三角小面）
│
├── stages/                      # 8个处理阶段的实现
│   ├── __init__.py
│   ├── s1_voxelize.py   (350L)  # S1: SAT体素化（mesh→cube 注册）
│   ├── s2_components.py (280L)  # S2: GPU标签传播（连通分量）
│   ├── s3_edge_weights.py(250L) # S3: Moller-Trumbore 射线求交（棱权重）
│   ├── s4_face_point.py(1197L)  # S4: U-Turn 检测 + 分量质心计算
│   ├── s6_collapse.py   (655L)  # S6: 法线曲线理论 loop 提取（合并 S5+S6）
│   ├── s7_rank_assign.py(752L)  # S7: Rank 分配 + 匈牙利匹配
│   └── s8_collapse.py  (2200L)  # S8: 网格重建（edge enum + geometry + welding）
│
├── geom/                        # GPU 几何核心算子
│   ├── __init__.py
│   ├── sh_clip.py       (180L)  # Sutherland-Hodgman 三角形→AABB 裁剪
│   ├── fan_centroid.py  (120L)  # 扇形三角化 + 面积加权质心
│   ├── closest_point.py (200L)  # Ericson §5.1.5 最近点查询
│   └── plane_tri_intersect.py   # 平面-三角形求交
│
├── interop/                     # custom/ 格式互转
│   ├── __init__.py
│   ├── from_custom.py   (120L)  # list[dict] → CubeBatch
│   └── to_custom.py     (120L)  # CubeBatch → list[dict]
│
├── parallel/                    # 并行工具
│   └── worker_pool.py    (82L)  # PersistentWorkerPool（fork-COW 友好）
│
├── profiling/                   # 性能测试与 A/B 对比
│   ├── __init__.py
│   ├── harness.py               # ProfilingCollector + stage_timer
│   ├── ab_rig.py                # 阶段级 A/B 比较器
│   ├── baseline_runner.py       # custom/ baseline 测量
│   ├── report_builder.py        # JSON→Markdown 报告生成
│   └── topology_equivalence.py  # 5 层拓扑等价检查
│
└── tests/                       # 215 个测试用例
    ├── conftest.py              # 共享 fixture（cube_mesh, icosphere_mesh 等）
    ├── unit/           (21文件)  # 单元测试（每个模块 / 每个阶段）
    ├── regression/      (7文件)  # A/B 回归测试（vs custom/ baseline）
    └── stages/          (3文件)  # 阶段级集成测试
```

---

## 2. 管线流程概览

CoReP 管线将一个三角网格编码为体素化的组合表示（CubeBatch），然后解码回三角网格。整个流程分为 8 个阶段：

```
输入 .ply / .obj 三角网格
    │
    ├─ MeshTensors.from_trimesh()    归一化到 [0,1]³，转为 GPU 张量
    │
    ├─ [S1] s1_voxelize              SAT 体素化（三角形-AABB 交叉测试）
    │       输出: cube_indices, tri_offsets/tri_values (CSR 面注册)
    │
    ├─ [S2] s2_components            GPU 标签传播（连通分量分析）
    │       输出: num_components, comp_face_off/val
    │
    ├─ [S3] s3_edge_weights          Moller-Trumbore 射线-三角形求交
    │       输出: edge_weights (N, 18) — 每条棱的交点计数
    │
    ├─ [S4] s4_face_point            两部分:
    │       Part A: face_weights — U-Turn 计数（GPU+MP 或 纯 MP）
    │       Part B: component_points — SH 裁剪 + 扇形质心 + 最近点吸附
    │       输出: face_weights (N,12), point_offsets/values (CSR)
    │
    ├─ [S6] s6_collapse              法线曲线理论 loop 提取（合并了 S5）
    │       GPU 验证 + CPU 图遍历 BFS
    │       输出: loop_cube_off, loop_edge_off/val, uturn_assignment, status
    │
    ├─ [S7] s7_rank_assign           rank 重跟踪 + GPU 质心插值 + 匈牙利匹配
    │       输出: loop_edge_rank, loop_point_match
    │
    └─ [S8] s8_collapse / decode     网格重建:
            共享棱枚举 → 向量化几何 → Python fallback → 顶点焊接
            输出: (vertices, faces) 或 .ply 文件
```

**注意**: S5 不作为独立阶段存在，其逻辑已合并到 S6。

---

## 3. 核心数据容器

### 3.1 MeshTensors — 输入网格

```python
from corep_fast.containers import MeshTensors

mt = MeshTensors.from_trimesh(mesh_obj, resolution=256, device='cuda:0')
```

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `vertices` | (V, 3) | float32 | 归一化到 [0,1]³ 的顶点坐标 |
| `faces` | (F, 3) | int32 | 三角面片的顶点索引 |
| `triangles` | (F, 3, 3) | float32 | 预聚合的三角形顶点 `vertices[faces]` |
| `face_normals` | (F, 3) | float32 | 单位法向量 |
| `face_adj` | (F, 3) | int32 | 每个面的 3 个邻接面 ID（-1 表示边界） |
| `boundaries` | (B, 2, 3) | float32 | 开边界线段端点 |
| `nm_edges` | (M, 2, 3) | float32 | 非流形棱（>2 个共享面） |
| `nm_vertices` | (P, 3) | float32 | 蝴蝶结顶点 |
| `center` | (3,) | float32 | 归一化前的质心 |
| `scale` | float | — | 归一化缩放系数（含 0.947 安全边距） |
| `resolution` | int | — | 体素网格分辨率 |
| `device` | torch.device | — | 设备句柄 |

### 3.2 CubeBatch — 管线内部表示

CubeBatch 是所有阶段共享的核心数据结构。每个阶段读取前序输出字段并追加自己的输出字段。

```python
from corep_fast.containers import CubeBatch, CubeStatus
```

**按阶段填充的字段**：

| 阶段 | 字段 | 形状 | 说明 |
|------|------|------|------|
| S1 | `cube_indices` | (N, 3) int32 | 网格坐标 (ix, iy, iz) |
| S1 | `cube_hash` | (N,) int64 | 唯一性哈希 |
| S1 | `tri_offsets/tri_values` | CSR | 每个 cube 注册的三角面 ID |
| S1 | `bnd_offsets/bnd_values` | CSR | 边界棱 |
| S1 | `nm_offsets/nm_values` | CSR | 非流形棱 |
| S2 | `num_components` | (N,) int32 | 连通分量数 |
| S2 | `num_boundary` | (N,) int32 | 边界棱计数 |
| S2 | `comp_face_off/comp_face_val` | CSR | 按分量分组的面 ID |
| S3 | `edge_weights` | (N, 18) int32 | 18 条棱的射线交点计数 |
| S4 | `face_weights` | (N, 12) int32 | 12 个小面的 U-Turn 计数 |
| S4 | `point_offsets/point_values` | CSR | 面积加权分量质心 |
| S6 | `loop_cube_off` | (N+1,) int64 | 每 cube 的 loop 数（CSR 第 1 级） |
| S6 | `loop_edge_off/val` | CSR | 每 loop 的 edge 序列（CSR 第 2 级） |
| S6 | `uturn_assignment` | (N,12,3) int32 | 每小面的 U-Turn 索引 |
| S6 | `status` | (N,) int32 | CubeStatus 枚举 |
| S7 | `loop_edge_rank` | (E,) int32 | 每条 edge crossing 的 rank |
| S7 | `loop_point_match` | (L,) int32 | loop → 分量点匹配 (-1=未匹配) |

**CSR (Compressed Sparse Row) 格式说明**：

```python
# offsets[i]   = values 中第 i 组的起始索引
# offsets[i+1] = 第 i 组的结束索引（不含）
# offsets[0] = 0, offsets[-1] = len(values)

# 示例：3 个 cube，分别有 3/2/4 个注册三角面
tri_offsets = [0, 3, 5, 9]
tri_values  = [f0, f1, f2, f3, f4, f5, f6, f7, f8]
# cube 0 → tri_values[0:3] = [f0, f1, f2]
# cube 1 → tri_values[3:5] = [f3, f4]
# cube 2 → tri_values[5:9] = [f5, f6, f7, f8]
```

**两级 CSR（loops）**：

```python
# 第 1 级：cube → loop
loop_cube_off[i]    # cube i 的第一个 loop 索引
loop_cube_off[i+1]  # cube i 的最后一个 loop（不含）

# 第 2 级：loop → edges
loop_edge_off[l]    # loop l 的第一个 edge 索引
loop_edge_off[l+1]  # loop l 的最后一个 edge（不含）

# 取 cube i 的第 j 个 loop 的所有 edges:
loop_idx = loop_cube_off[i] + j
edges = loop_edge_val[loop_edge_off[loop_idx] : loop_edge_off[loop_idx+1]]
```

### 3.3 CubeStatus 枚举

```python
class CubeStatus:
    OK             = 0  # 正常 cube
    AMBIGUOUS      = 1  # 多个可行解（S6 combinatorial 阶段）
    UNSOLVABLE     = 2  # 无可行解
    BUDGET_EXCEEDED = 3  # 组合搜索超出预算
```

状态非 OK 的 cube 在 S8 重建时走 exception fallback 路径（仅输出单个分量质心作为顶点）。

---

## 4. API 使用方式

### 4.1 端到端管线（最常用）

```python
import torch
from corep_fast.pipeline import corep_pipeline

# 输入任意 .ply / .obj 三角网格，输出重建后的 .ply
batch, vertices, faces = corep_pipeline(
    mesh_path="input.ply",
    resolution=256,               # 体素分辨率（越高越精细，越慢）
    device=torch.device("cuda:0"),
    output_path="output.ply",     # 可选，写 PLY 文件
    merge_decimals=5,             # 顶点焊接精度（小数位）
    num_workers=None,             # MP 工人数，None=自动
)

# 返回值:
#   batch:    CubeBatch   — 完整的编码状态（所有阶段输出）
#   vertices: (V, 3) float32 — 重建网格顶点（归一化坐标 [0,1]³）
#   faces:    (F, 3) int32   — 重建网格三角面
```

### 4.2 分步：编码 + 解码

```python
import torch
from corep_fast.pipeline import corep_encode, corep_decode

device = torch.device("cuda:0")

# 编码：mesh → CubeBatch（S1-S7，包含所有中间表示）
batch = corep_encode(
    mesh_path="input.ply",
    resolution=256,
    device=device,
    num_workers=16,  # 指定 MP 工人数
)

print(f"编码完成: {batch.num_cubes} cubes, {batch.num_loops} loops")

# 解码：CubeBatch → (vertices, faces)（S8）
vertices, faces = corep_decode(batch, merge_decimals=5)
print(f"重建完成: V={vertices.shape[0]}, F={faces.shape[0]}")
```

### 4.3 单步阶段调用（灵活组合）

```python
import torch
import trimesh
from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse
from corep_fast.stages.s7_rank_assign import s7_rank_assign
from corep_fast.stages.s8_collapse import decode_from_cubebatch

device = torch.device("cuda:0")
mesh = trimesh.load("input.ply")

# Step 0: 构造 MeshTensors（归一化 + GPU 转换）
mt = MeshTensors.from_trimesh(mesh, resolution=256, device=device)

# Step 1-7: 逐阶段处理
batch = s1_voxelize(mt, resolution=256, device=device)
batch = s2_components(batch, mt)
batch = s3_edge_weights(batch, mt)
batch = s4_face_point(batch, mt, use_gpu_fw=True)   # use_gpu_fw=True 启用 GPU 加速
batch = s6_collapse(batch, num_workers=16)
batch = s7_rank_assign(batch, num_workers=16)

# Step 8: 解码
vertices, faces = decode_from_cubebatch(batch, merge_decimals=5)
```

### 4.4 混合管线（custom S1-S7 + corep_fast S8）

```python
from corep_fast.pipeline import run_hybrid_pipeline, PipelineConfig

# 使用 custom/ 的 S1-S7（Python baseline）+ corep_fast 的 S8（Torch 向量化重建）
cfg = PipelineConfig(
    s1_to_s7_impl='custom',
    s8_impl='corep_fast',
    merge_decimals=5,
)
output_path = run_hybrid_pipeline(
    "input.ply",
    resolution=256,
    output_path="output.ply",
    config=cfg,
)
```

### 4.5 仅输出 PLY 文件（从 list[dict] 格式）

```python
from corep_fast.stages.s8_collapse import s8_collapse_to_ply

# 如果已有 custom/ 格式的 cube_data_list（list of dict）
ply_path = s8_collapse_to_ply(
    resolution=256,
    cube_data_list=all_regs,        # list[dict]，每个 dict 代表一个 cube
    output_filepath="output.ply",
    merge_decimals=5,
)
```

---

## 5. 几何核心算子 (geom/)

### 5.1 Sutherland-Hodgman 三角形裁剪

```python
from corep_fast.geom.sh_clip import sh_clip_aabb

# 输入：C 个三角形和对应 AABB
triangles = torch.rand(1000, 3, 3, device='cuda')     # (C, 3, 3)
aabb_min  = torch.zeros(1000, 3, device='cuda')        # (C, 3)
aabb_max  = torch.ones(1000, 3, device='cuda') * 0.01  # (C, 3)

# 输出：裁剪后的多边形（最多 12 顶点，-1 padding）
poly, v_len = sh_clip_aabb(triangles, aabb_min, aabb_max)
# poly:  (C, 12, 3) float32 — 多边形顶点（padding 为 0）
# v_len: (C,) int32          — 每个多边形的有效顶点数
```

### 5.2 扇形三角化质心

```python
from corep_fast.geom.fan_centroid import fan_area_centroid

# 接受 sh_clip_aabb 的输出
centroid, area = fan_area_centroid(poly, v_len)
# centroid: (C, 3) float32 — 面积加权质心
# area:     (C,) float32   — 总面积
```

### 5.3 最近点查询

```python
from corep_fast.geom.closest_point import closest_point_on_mesh

queries   = torch.rand(500, 3, device='cuda')            # (P, 3)
triangles = torch.rand(2000, 3, 3, device='cuda')        # (F, 3, 3)

# 找每个 query 在网格上的最近点
closest_pts, face_idx = closest_point_on_mesh(queries, triangles)
# closest_pts: (P, 3) float32 — 最近点坐标
# face_idx:    (P,) int64     — 对应三角面索引
```

### 5.4 平面-三角形求交

```python
from corep_fast.geom.plane_tri_intersect import plane_triangle_intersect

triangles    = torch.rand(1000, 3, 3, device='cuda')  # (P, 3, 3)
plane_normals = torch.rand(1000, 3, device='cuda')     # (P, 3) 已归一化
plane_points  = torch.rand(1000, 3, device='cuda')     # (P, 3)

A, B, valid = plane_triangle_intersect(triangles, plane_normals, plane_points)
# A:     (P, 3) float32 — 交线段起点
# B:     (P, 3) float32 — 交线段终点
# valid: (P,) bool      — 是否存在有效交线段
```

---

## 6. 互操作 (interop/)

### 6.1 custom/ dict → CubeBatch

```python
from corep_fast.interop.from_custom import cube_batch_from_custom
from corep_fast.containers import MeshTensors

# face_registers: custom/ 管线 S1-S7 输出的 list[dict]
batch = cube_batch_from_custom(
    face_registers,
    mesh=mesh_tensors,    # MeshTensors 实例
    include=None,         # None = 包含全部字段; 或指定 {'cube_indices', 'edge_weights', ...}
    device='cuda:0',
)
```

### 6.2 CubeBatch → custom/ dict

```python
from corep_fast.interop.to_custom import custom_from_cube_batch

# 用于 A/B 回归测试或导出到外部分析
face_registers = custom_from_cube_batch(batch, mesh_tensors)
# 返回: list[dict]，每个 dict 包含:
#   'cube_indices': (ix, iy, iz),
#   'edge_weights': [w0, ..., w17],
#   'face_weights': [w0, ..., w11],
#   'sorted_loops': [{'loop': [...], 'rank': [...], 'component_point': [x,y,z]}, ...],
#   'exception': bool,
#   ...
```

---

## 7. 性能分析与测试

### 7.1 性能分析 (Profiling)

```python
from corep_fast.profiling.harness import ProfilingCollector, stage_timer
from corep_fast.pipeline import corep_pipeline
import torch

# 方式一：自动收集
pc = ProfilingCollector(mesh_name="sphere.ply", resolution=256, impl="corep_fast")
batch, v, f = corep_pipeline(
    "sphere.ply", 256, torch.device("cuda:0"),
    collector=pc, num_workers=16,
)
pc.save_json("profiling_result.json")
print(f"总用时: {pc.total_wall_time_s:.2f}s")

# 方式二：手动 stage timer
pc = ProfilingCollector()
with stage_timer('my_custom_stage', pc):
    # 你的代码
    pass
```

### 7.2 运行测试

```bash
# 进入项目目录
cd /mnt/novita2/siyuan/workspace/TRELLIS.2

# 全部测试 (215 个)
.venv/bin/python -m pytest corep_fast/tests/ -v

# 仅单元测试
.venv/bin/python -m pytest corep_fast/tests/unit/ -v

# 仅 A/B 回归测试（对比 custom/ baseline）
.venv/bin/python -m pytest corep_fast/tests/regression/ -v

# 仅阶段级集成测试
.venv/bin/python -m pytest corep_fast/tests/stages/ -v

# 单个阶段
.venv/bin/python -m pytest corep_fast/tests/unit/test_s4_face_point.py -v

# 单个测试
.venv/bin/python -m pytest corep_fast/tests/unit/test_s1_voxelize.py::test_s1_voxelize_cube -v

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest corep_fast/tests/ -v

# 禁用 GPU face_weights 路径（回退到 CPU MP）
COREP_FAST_S4_GPU_FW=0 .venv/bin/python -m pytest corep_fast/tests/unit/test_s4_face_point.py -v
```

### 7.3 拓扑等价检查

```python
from corep_fast.profiling.topology_equivalence import compare_batches_l1

# 对比两个 CubeBatch 的拓扑等价性
report = compare_batches_l1(batch_a, batch_b)
print(report.pretty_print())
# 输出:
#   L1 (integer fields):    PASS  (cube_indices, edge_weights 等逐元素匹配)
#   L2 (loop sets):         PASS
#   ...
```

### 7.4 A/B 阶段对比

```python
from corep_fast.profiling.ab_rig import stage_ab_run

result = stage_ab_run(
    stage_name='s4',
    mesh_path='icosphere.ply',
    resolution=64,
    output_dir='tmp/ab_results/',
)
# result: dict 包含 timing + topology_equivalence report
```

---

## 8. 配置与调优

### 8.1 运行模式

```python
from corep_fast.config import Mode, set_mode

set_mode(Mode.PRODUCTION)  # 默认：跳过所有 invariant 检查，最快
set_mode(Mode.DEBUG)       # 开启 invariant 检查 + dump pickle
set_mode(Mode.STRICT)      # DEBUG + 每阶段 A/B vs custom/（非常慢）
```

### 8.2 Feature Flags（环境变量）

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `COREP_FAST_S8_DIRECT_TENSOR` | `1` | S8 跳过 dict 中间层（M2 P1 优化） |
| `COREP_FAST_S4_GPU_FW` | `1` | S4 使用 GPU 加速的 face_weights（M2 P2 优化） |
| `COREP_FAST_S8_DIRECT_GRIDS` | `1` | S8 从 CubeBatch 直接构建 fallback grid（M2 P3 优化） |

```bash
# 禁用 GPU face_weights（回退到 CPU MP baseline）
export COREP_FAST_S4_GPU_FW=0

# 禁用 direct tensor s8 路径（回退到 dict 中间层）
export COREP_FAST_S8_DIRECT_TENSOR=0
```

### 8.3 数值调优

```python
from corep_fast.config import StageConfig

cfg = StageConfig(
    chunk_size_bytes=512 * 1024 * 1024,   # S1 体素化的峰值内存预算
    prod_k_threshold=100_000,              # S6 组合搜索上限（超出标记 BUDGET_EXCEEDED）
    max_poly_verts=12,                     # SH 裁剪后最大多边形顶点数
)
```

### 8.4 后端选择

```python
from corep_fast.config import BackendConfig

backends = BackendConfig.all_torch()              # 全部使用 PyTorch（当前默认）
backends = backends.with_override(s4_face='triton')  # 某个阶段切换到 Triton
```

---

## 9. 性能数据（H100 80GB, icosphere subdiv=3）

### vs 纯 custom/ baseline (`e2e_custom`)

| 阶段 | custom res=256 | corep_fast M2 | 加速比 |
|------|---------------:|------:|------:|
| S1 (体素化) | 2.23s | 0.19s | **11.9x** |
| S2 (连通分量) | 3.72s | 0.08s | **46.4x** |
| S3 (棱权重) | 18.84s | 0.002s | **9584x** |
| S4 (U-Turn+质心) | 26.67s | 4.27s | **6.3x** |
| S6 (loop 提取) | 13.62s | 2.52s | **5.4x** |
| S7 (rank+匹配) | 12.07s | 5.13s | **2.4x** |
| S8 (网格重建) | 63.89s | 5.61s | **11.4x** |
| **E2E** | **141.05s** | **17.92s** | **7.87x** |

### 各版本进化历程

| 里程碑 | res=256 e2e | vs custom |
|--------|------------:|----------:|
| M1 初始（含 bug） | 203.7s | 0.40x (慢) |
| M1 修复后 | 30.13s | 4.68x |
| **M2 最终** | **17.92s** | **7.87x** |

---

## 10. 完整使用示例

### 10.1 最简使用

```python
import torch
from corep_fast.pipeline import corep_pipeline

batch, verts, faces = corep_pipeline(
    "my_model.ply", 256, torch.device("cuda:0"),
    output_path="reconstructed.ply",
)
print(f"输入 → 输出: V={verts.shape[0]}, F={faces.shape[0]}")
```

### 10.2 批量处理多个 mesh

```python
import torch
from pathlib import Path
from corep_fast.pipeline import corep_encode, corep_decode

device = torch.device("cuda:0")
input_dir = Path("dataset/meshes/")
output_dir = Path("dataset/reconstructed/")
output_dir.mkdir(exist_ok=True)

for ply_path in sorted(input_dir.glob("*.ply")):
    batch = corep_encode(str(ply_path), resolution=256, device=device)
    verts, faces = corep_decode(batch)

    # 保存为 PLY
    from corep_fast.stages.s8_collapse import _write_ply_ascii
    _write_ply_ascii(verts, faces, str(output_dir / ply_path.name))

    print(f"{ply_path.name}: {batch.num_cubes} cubes → V={verts.shape[0]} F={faces.shape[0]}")
```

### 10.3 中间结果检查

```python
import torch
from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
import trimesh

device = torch.device("cuda:0")
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mt = MeshTensors.from_trimesh(mesh, 128, device=device)

# S1
batch = s1_voxelize(mt, 128, device)
print(f"S1 完成: {batch.num_cubes} cubes")
print(f"  每 cube 平均注册 {batch.tri_values.shape[0] / batch.num_cubes:.1f} 个三角面")

# S2
batch = s2_components(batch, mt)
print(f"S2 完成:")
print(f"  单分量 cube: {(batch.num_components == 1).sum().item()}")
print(f"  多分量 cube: {(batch.num_components > 1).sum().item()}")

# 检查特定 cube 的数据
cube_id = 100
tri_lo = batch.tri_offsets[cube_id].item()
tri_hi = batch.tri_offsets[cube_id + 1].item()
print(f"\nCube #{cube_id}:")
print(f"  坐标: {batch.cube_indices[cube_id].tolist()}")
print(f"  注册三角面: {tri_hi - tri_lo} 个 (IDs: {batch.tri_values[tri_lo:tri_hi].tolist()})")
print(f"  分量数: {batch.num_components[cube_id].item()}")
```

### 10.4 Profiling + A/B 对比

```python
import torch
from corep_fast.pipeline import corep_pipeline
from corep_fast.profiling.harness import ProfilingCollector

# 带计时的完整管线
pc = ProfilingCollector(mesh_name="sphere", resolution=256, impl="corep_fast")
batch, v, f = corep_pipeline(
    "sphere.ply", 256, torch.device("cuda:0"),
    collector=pc,
)

# 打印各阶段用时
for stage_name, timing in pc.stages.items():
    print(f"  {stage_name}: {timing['wall_time_s']:.3f}s")

# 保存 JSON
pc.save_json("profiling_result.json")
```

### 10.5 禁用优化进行对比

```bash
# 方式一：环境变量（影响整个进程）
COREP_FAST_S4_GPU_FW=0 COREP_FAST_S8_DIRECT_TENSOR=0 \
    python my_benchmark.py

# 方式二：代码内 flag
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s8_collapse import decode_from_cubebatch

# 旧路径
batch = s4_face_point(batch, mt, use_gpu_fw=False)           # CPU MP
v, f = decode_from_cubebatch(batch, use_direct_tensor=False)  # dict 中间层

# 新路径
batch = s4_face_point(batch, mt, use_gpu_fw=True)            # GPU batch
v, f = decode_from_cubebatch(batch, use_direct_tensor=True)   # direct tensor
```

---

## 11. 立方体拓扑常量

```python
from corep_fast.constants import (
    NUM_VERTICES,       # 8  — 单位立方体顶点数
    NUM_EDGES,          # 18 — 12 轴对齐棱 + 6 面对角线
    NUM_FACETS,         # 12 — 每面 2 个三角形 × 6 面
    CUBE_VERTICES,      # (8, 3) float32 — [0,1]³ 顶点坐标
    CUBE_EDGES,         # (18, 2) int32   — 棱的顶点对
    CUBE_FACETS,        # (12, 3) int32   — 小面的棱三元组
    EDGE_SHARE_FACTORS, # (18,) int32     — 棱共享因子 ([4]*12 + [2]*6)
)
```

**棱编号约定**：
- 0-11: 轴对齐棱（每条被 4 个相邻 cube 共享）
- 12-17: 面对角线（每条被 2 个相邻 cube 共享）

**小面编号约定** (每面 2 个三角形)：
- 0-1: 底面 (z=0)
- 2-3: 顶面 (z=1)
- 4-5: 前面 (y=0)
- 6-7: 右面 (x=1)
- 8-9: 后面 (y=1)
- 10-11: 左面 (x=0)

---

## 12. 常见问题

### Q: 分辨率应该设多少？

- **res=64**: 快速测试，~2K cubes
- **res=128**: 开发调试，~68K cubes
- **res=256**: 标准生产，~275K cubes
- **res=512+**: 高精度场景（GPU 内存 ~8GB @ res=256，线性增长）

### Q: 支持哪些输入格式？

通过 `trimesh.load()` 加载，支持 `.ply`, `.obj`, `.stl`, `.glb`, `.gltf` 等。mesh 必须是三角化的（非三角面会被自动三角化）。

### Q: 如何从重建坐标还原到原始坐标？

```python
# 重建的 vertices 在归一化 [0,1]³ 空间
# 用 MeshTensors 的 center 和 scale 还原:
original_coords = (vertices.cpu().numpy() - 0.5) * mt.scale / 0.947 + mt.center.cpu().numpy()
```

### Q: 某个 stage 报错如何回退到 CPU 路径？

```bash
# 禁用 S4 GPU 加速
export COREP_FAST_S4_GPU_FW=0

# 禁用 S8 direct tensor 路径
export COREP_FAST_S8_DIRECT_TENSOR=0

# 禁用 S8 direct grids 路径
export COREP_FAST_S8_DIRECT_GRIDS=0
```

### Q: 如何查看某个 cube 的详细数据？

```python
def inspect_cube(batch, cube_id):
    """打印 cube 的完整数据。"""
    i = cube_id
    print(f"Cube #{i}")
    print(f"  坐标: {batch.cube_indices[i].tolist()}")
    print(f"  状态: {batch.status[i].item()} (0=OK)")
    print(f"  分量: {batch.num_components[i].item()}")

    # 注册三角面
    t_lo, t_hi = batch.tri_offsets[i].item(), batch.tri_offsets[i+1].item()
    print(f"  注册面: {t_hi - t_lo}")

    # edge_weights
    print(f"  edge_weights: {batch.edge_weights[i].tolist()}")

    # face_weights
    if hasattr(batch, 'face_weights') and batch.face_weights is not None:
        print(f"  face_weights: {batch.face_weights[i].tolist()}")

    # loops
    l_lo = batch.loop_cube_off[i].item()
    l_hi = batch.loop_cube_off[i+1].item()
    print(f"  loops: {l_hi - l_lo}")
    for li in range(l_lo, l_hi):
        e_lo = batch.loop_edge_off[li].item()
        e_hi = batch.loop_edge_off[li+1].item()
        edges = batch.loop_edge_val[e_lo:e_hi].tolist()
        ranks = batch.loop_edge_rank[e_lo:e_hi].tolist()
        match = batch.loop_point_match[li].item()
        print(f"    loop {li-l_lo}: edges={edges}, ranks={ranks}, match={match}")

    # component points
    p_lo = batch.point_offsets[i].item()
    p_hi = batch.point_offsets[i+1].item()
    for pi in range(p_lo, p_hi):
        print(f"    point {pi-p_lo}: {batch.point_values[pi].tolist()}")
```
