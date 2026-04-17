# s7 Phase1 GPU 回归根因定位

节点: 116, GPU 0; 日期: 2026-04-17

## 症状
三层 icosphere (r = 1.00 / 1.01 / 1.02, subdivisions=3) @ res=32：
- custom (s1-s8 全):           V=25288  F=50436  ← 参考真值
- corep_fast @ d780be8:        V=25358  F=50708  ← 数值对齐 custom
- corep_fast @ HEAD:           **V=14026  F=23740**  ← 掉 ~45%

## 定位流程
1. 2^3 flag 矩阵（W1=S8_4CUBE_VECTORIZED, W2=S7_PHASE1_GPU, W3=S6_FASTPATH_GPU）
   - **W2=0 → 25358/50708** (correct)
   - **W2=1 → 14026/23740** (regression)
   - W1, W3 无关
2. 直接 A/B s7 Phase1 GPU vs CPU on 三层球 s6 输出：
   - rank mismatches: **4962 / 50497** (9.8%)
   - match mismatches: 266 / 7009
   - GPU rank sum = 19387, CPU = 22136 → **GPU 系统性偏低 12%**
3. 按 cube 分类：
   - 684 diff cube 全部是 `CubeStatus.OK`
   - **683 / 684 是多 loop cube** (n_loops ≥ 2)
   - **680 / 683 多 loop diff cube 有重复 edge 序列**

## 根因
`corep_fast/stages/s7_rank_assign.py::_phase1_gpu_rank_assign` (L903-1078)
**缺失双射匹配约束**。

CPU 路径 `_match_loops_to_ranks` (L260-309) 维护 `used_traced[i]=True`
标记位 —— 每条 pre-enumerated traced loop 最多被一条 s6 loop 消费。
当 cube 内多条 s6 loop 的 edge 序列相同（几何上是同一 facet 上的
多 rank 交叉，W > 1），CPU 把不同 rank 分给不同 s6 loop。

GPU 路径对每条 s6 loop **独立** 走 candidate walk + `alive.argmax(dim=1)`
取 first-True —— 每条 s6 loop 都拿到最小可行 rank0，两条重合 loop → 相同 ranks。

## 实例 (cube 28)
```
n_loops=2 status=OK
loop[0] edges = [5, 13, 7, 17, 11, 16, 10, 15]
loop[1] edges = [5, 13, 7, 17, 11, 16, 10, 15]   # identical
edge_weights:   edges 5,7,11,13,15,16,17 均 = 2

CPU ranks:
  loop[0] = [0, 0, 1, 0, 0, 0, 0, 0]
  loop[1] = [1, 1, 0, 1, 1, 1, 1, 1]   # complementary
GPU ranks:
  loop[0] = [0, 0, 1, 0, 0, 0, 0, 0]
  loop[1] = [0, 0, 1, 0, 0, 0, 0, 0]   # collision
```

## 几何后果
rank 决定 edge 交叉的 3D 位置 `t = (r+1)/(W+1)`：
- W=2, rank 0 → t=1/3；rank 1 → t=2/3（两点区分）
- GPU 把两条 loop 都塞到 t=1/3 → 两个本应不同的 centroid 变成同一点
- s8 `merge_vertices` 焊掉重复顶点 → ~45% 几何消失

## 为什么现有 A/B 测试没抓到
`corep_fast/tests/regression/test_s7_phase1_gpu_ab.py` 用 `trimesh.creation.icosphere(subdivisions=2)` —— **单层**球面。
单层不会在同一 voxel 产生 W>1 + 重复 s6 loop。
多层 / 嵌套 / 低分辨率平行面才会触发。

## 次要缺陷
Cube 966 (1 / 684) 是单 loop 但 edge 14 重复 (W=2 自交叉):
```
loop edges = [0, 12, 3, 17, 8, 14, 14]
CPU ranks  = [0, 0, 0, 0, 0, 1, 0]
GPU ranks  = [0, 0, 0, 0, 0, 0, 0]
```
intra-loop 重复边 (W≥2 自交叉) GPU 也不能区分两个 crossing
的 rank —— argmax 取最早可行，把第二个 rank 塌到 0。

## 修复方向 (未实施)
1. **快速回滚**：默认 `COREP_FAST_S7_PHASE1_GPU=0`，保留 GPU 路径作可选加速
2. **GPU 侧补双射匹配**：按 cube 分组、对同 cube 内 s6 loop 做
   edge-sequence 聚类，每一簇内按 rank 依次分配（可用 scatter +
   per-cube cumsum）
3. **单元测试**：在 `test_s7_phase1_gpu_ab.py` 加三层 icosphere fixture
   (subdivisions=3, radii 1.00/1.01/1.02) 覆盖 W>1 + 重复 loop 场景

## 证据文件
- `tmp/flag_matrix_reconstruct.py` / `tmp/flag_matrix_116.log`
- `tmp/s7_phase1_ab_triple.py` / `tmp/s7_ab_116.log`
- `tmp/s7_rank_root_cause.py` / `tmp/s7_root_cause_116.log`
- `tmp/test_fast/output.ply` (HEAD), `output_d780be8.ply`, `output_custom.ply`
