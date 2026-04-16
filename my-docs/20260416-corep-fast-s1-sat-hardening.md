# corep_fast S1 SAT + S3/S4 Epsilon 加固（Follow-up）

**Status**: 已诊断，未实施。  
**Owner**: 待分配  
**Estimated effort**: 1.5 day（含 GPU kernel 改动 + 测试）  
**Related**: commits `d780be8`、`4a355c3`  
**Companion follow-up**: `20260416-corep-fast-mesh-cleanup-port.md`  
**Severity**: 低（治本但 scope 大；已有 centering trick 在大多数 case 下 mask 此问题）

---

## 1. 动机

### 1.1 当前状态

`d780be8` 通过把对称 `+0.5` 居中改为非对称 `+(0.489, 0.506, 0.513)` 修复了 sphere 三层球 res=32 的 7 层伪峰 bug。这个 fix **是规避策略**：把对称网格的顶点推离 voxel 整数边界，避免触发下游退化分支。

**fix 没有解决退化分支本身**。它们仍然存在于 S1 SAT、S3 Möller-Trumbore、S4 facet 几何核中。下面三类输入仍会踩雷：

1. **用户自己已经把网格放在 [0,1]³ 且对称居中** —— from_trimesh 的 offset 不会"再次"推开。例：`mesh.vertices = mesh.vertices * 0.5 + 0.25` 后传入。
2. **轴对齐 CAD 网格** —— 立方体 / 平面 / Manhattan 几何即使非对称居中，仍可能有顶点沿单一轴恰好 = i/R。
3. **特定分辨率下的偶然对齐** —— 高分辨率（res=512+）下，bbox 极值点 ÷ R 很可能逼近浮点表示的整数。

### 1.2 4 个具体退化点

#### A. S1 SAT 严格不等号无 epsilon

`corep_fast/stages/s1_voxelize.py:194`（`_test_axis`）：
```python
alive &= ~((tri_min > r) | (tri_max < -r))
```
分离测试用裸 `>` / `<`。当三角形顶点投影 = ±r（即压在 cube 面上），结果取决于 IEEE-754 舍入方向 —— 可能让三角形被 0、1、2 个相邻 cube 同时注册（应是 1 或 2，从不 0）。

**观察证据**：custom 用 numpy 双精度同样裸 `>`/`<`，但顺路加了 `if np.allclose(axis, 0): continue`（voxelize.py:299，见 §C），间接绕开了主要退化。fast 用 float32 + 严格不等，受影响更大。

#### B. S1 SAT 缺 zero-axis guard

同一 `_test_axis`（s1_voxelize.py:201-204）：edge × unit_axis 在 edge 平行于该轴时叉积为零向量，`r = ax.abs().sum(...) * half` 变成 0，`alive &= ~((0 > 0) | (0 < 0))` = no-op，整条 axis 测试退化为"接受所有"。当 9 条 edge×axis test 中有多条同时退化时，假阳性叠加：三角形被本不该相交的 cube 误注册。

**custom 实现** `voxelize.py:299` 显式跳过：
```python
for axis in [...]:
    if np.allclose(axis, 0):
        continue   # zero axis → degenerate; skip this test
```
fast 没有这个 guard。

#### C. S3 Möller-Trumbore epsilon 临界

`corep_fast/stages/s3_edge_weights.py:20`：
```python
_EPSILON = 1e-8
```
单位 [0,1] 网格、float32 精度 ~7 位有效数字 → 相对误差量级 ~1e-7。`1e-8` 比浮点 ULP 还小，根本起不到 tie-break 作用。

具体 trigger：
- L196 `(t >= -_EPSILON) & (t <= 1.0 + _EPSILON)`：射线沿 cube edge 长度归一后，t 在 1.0 附近的舍入易让交点被相邻两条 edge 同时计。
- L194-195 `(u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u+v <= 1.0)`：射线穿过三角形顶点（u≈0/v≈0/u+v≈1）时，相邻三角形重复计数。

#### D. S4 facet-coplanar 分支 trigger 过宽

`corep_fast/stages/s4_face_point.py:230-263`（`_intersect_facet_with_mesh`）：
```python
eps = 1e-8
d_zero = abs(d) <= eps
# ... if d_zero on >= 1 vertex: enter coplanar_edge branch
```
当 mesh 顶点恰好落在 cube facet 平面上（z = i/R），平面距离 ≈ 0 触发 `coplanar_edge` 分支。该分支把整条三角形边作为交线插入节点图，并依赖 L398 `_find_or_add_node(tol=1e-8)` 去重 —— 同样的 epsilon 临界问题。结果：**虚假节点累积、U-Turn 计数错乱**。

### 1.3 为什么 centering fix 之后还要做这个

centering offset `(0.489, 0.506, 0.513)` 把"对称网格的对称顶点"从 0.5 推离 ≥0.006，远大于 1e-8 epsilon —— 因此 d780be8 后这些退化分支**在常规输入下不会触发**。

但本 follow-up 治"凡顶点贴近任意 i/R 的输入"，offset 治不到（offset 只针对 0.5）。例如：
- res=32 时 i/R ∈ {0/32, 1/32, …, 32/32}，共 33 个边界值。
- offset 仅推开了 i=16 一个。
- 输入若 bbox 不对称（如薄板 mesh），归一化后 z 极值可能落在 i=2 或 i=30 附近。

修复后的回报：
- corep_fast 对**任意几何**鲁棒，不再依赖 normalization 把网格"运气好地"放在 voxel 内部。
- 删除 centering offset 这个 magic constant 成为可能（治本后可恢复对称 +0.5，与对称 voxel 网格语义更自然）。
- 与 custom 的对齐从"行为相同（巧合）"升级为"算法相同（保证）"。

---

## 2. 实施方案

### 2.1 改动 A：S1 SAT 加 epsilon tie-break

**File**: `corep_fast/stages/s1_voxelize.py`  
**Function**: `_test_axis`  
**Line**: ~194

```python
# Before:
alive &= ~((tri_min > r) | (tri_max < -r))

# After:
# Tie-break epsilon: in normalized [0,1]³ space at res R, voxel size is 1/R.
# Use a fraction of voxel size as tolerance — far larger than float32 ULP
# (~1e-7 in [0,1]) but still well below voxel scale, so tolerated overlap
# is geometrically negligible.
SAT_EPS = 1e-6   # ≈ 3% of (1/R) at R=32, 0.05% at R=512
alive &= ~((tri_min > r + SAT_EPS) | (tri_max < -r - SAT_EPS))
```

副作用：极少的 false positive（三角形被多注册 1 个 cube），但 cube 的 SAT 内部本就不需要严格判定 —— 多注册一个不会改变最终输出（s2-s8 只看实际几何）。

### 2.2 改动 B：S1 SAT 加 zero-axis guard

**File**: `corep_fast/stages/s1_voxelize.py`  
**Function**: `_test_axis` 调用循环（搜 9 个 edge×axis）  
**Reference**: `custom/voxelize.py:299` 实现

```python
# Sketch (具体位置依 fast 的循环展开方式而定):
# Before each axis test, check axis magnitude
axis_norm = torch.linalg.norm(axis, dim=-1)
zero_axis_mask = axis_norm < 1e-10
# For zero axes (edge parallel to coord axis), skip test → keep alive unchanged.
test_result = ~((tri_min > r + SAT_EPS) | (tri_max < -r - SAT_EPS))
test_result = torch.where(zero_axis_mask.unsqueeze(...), True, test_result)
alive &= test_result
```

GPU vectorization 注意：必须 broadcast zero_axis_mask 到 alive 的形状。在 fast 当前的 `_test_axis` 实现里 axis 是预计算的常量 9-tuple，可在 import 时计算 zero-mask 并 hard-code。

### 2.3 改动 C：S3 Möller-Trumbore epsilon 调到 ULP 之上

**File**: `corep_fast/stages/s3_edge_weights.py`  
**Line**: 20

```python
# Before:
_EPSILON = 1e-8

# After:
# float32 in [0,1] has ULP ~6e-8; t-bounds checks need eps > ULP to be useful.
# 1e-6 is ~16 ULPs at unity, well below any voxel-scale geometric feature.
_EPSILON = 1e-6
```

### 2.4 改动 D：S4 facet-coplanar 分支 trigger 用相对 epsilon

**File**: `corep_fast/stages/s4_face_point.py:230-263`

```python
# Before:
eps = 1e-8
d_zero = abs(d) <= eps

# After:
# Use voxel-scale epsilon: a vertex within 1e-6 of facet plane in normalized
# coords is < 0.001% of voxel side at R=32; safer to treat as on-plane than
# to invoke (numerically fragile) coplanar branch on float noise.
eps = 1e-6
d_zero = abs(d) <= eps
```

同时 L398 `_find_or_add_node(tol=...)` 一并改 1e-8 → 1e-6 保持一致。

### 2.5 验证：先证伪 d780be8 的 centering fix 仍是必需的

逐步移除 / 复位 centering offset 跑回归 ——**预期** Step 1 仍需要 centering 才能过 sphere case；改 A+B+C+D 全做完后，Step 2 可以**移除** centering offset、回到对称 +0.5，sphere case 仍能正确出 3 层。

| Step | centering | A | B | C | D | sphere res=32 layers |
|------|-----------|---|---|---|---|---------------------|
| 0 (现在) | offset | ✗ | ✗ | ✗ | ✗ | 3 ✓ |
| 1 | offset | ✓ | ✓ | ✓ | ✓ | 3 ✓（改进无影响） |
| 2 | symmetric +0.5 | ✓ | ✓ | ✓ | ✓ | **应仍 3 ✓** ← 治本验证 |
| 3 (退化) | symmetric +0.5 | ✗ | ✗ | ✗ | ✗ | 7（已知 bug） |

Step 2 通过 → 证明 4 项改动确实治本。可考虑：
- 把 centering offset 留作"对称网格的额外保险"且加注释说明已是冗余防护。
- 或彻底删除 offset，依赖治本后的算法稳健性。

### 2.6 不在范围内的事项

- 不动 s2/s6/s7/s8（与 SAT/求交 epsilon 无关）。
- 不动 mesh cleanup（见 companion doc）。
- 不试图统一 fast 与 custom 的浮点序 —— 不必要且影响性能。

---

## 3. 风险与回滚

### 3.1 性能

- A、C、D：仅 epsilon 数值常量改动，性能无影响。
- B：每条 edge×axis 多一次 abs+compare，估计增 ~3–5% on S1（S1 占 e2e 1%，影响 < 0.05%）。

### 3.2 数值差异

epsilon 放宽后：
- 三角形被多注册到相邻 cube 的次数轻微增加（< 0.1% on icosphere），不影响输出。
- `loops_per_cube`、`num_components` 在某些边缘 cube 上可能变化，**所有现有 unit test 需要重跑**。
- e2e_gpu_ab 的 0.1% 容忍可能需要短暂放宽到 0.5%（fast 端先调，custom 跟随调或保留 magic 常量）。

### 3.3 回滚策略

每个改动独立 commit：
- `feat(s1): add SAT epsilon tie-break`
- `feat(s1): guard zero-axis SAT degeneracy`
- `feat(s3): widen Moller-Trumbore epsilon to 1e-6`
- `feat(s4): widen coplanar threshold to 1e-6`

任何一个引入回归可独立 revert。Step 2 验证（移除 centering）单独 commit，问题最小。

---

## 4. Definition of Done

1. 4 个 commit 合并后跑 `pytest corep_fast/tests/ -v` 全绿。
2. icosphere/cube/plane/triple-sphere 在 res=32/64/128/256 reconstruction 与 d780be8 状态相同（layer 数、CD ±5%）。
3. 用一组对称输入（顶点贴 i/R）跑 sphere reproduce，验证不再产生伪峰。
4. 在 119 跑 e2e_gpu_ab，0.1% 容忍仍过；如不过，临时放宽到 0.5% 并在 docstring 写明原因。
5. （可选）执行 §2.5 Step 2 验证：移除 centering offset 跑 sphere res=32，仍出 3 层 → 治本确认。

---

## 5. 参考

- `custom/voxelize.py:299` — zero-axis guard 参考实现
- `corep_fast/stages/s1_voxelize.py:163-220` — `_test_axis` 全文
- `corep_fast/stages/s3_edge_weights.py:1-220` — M-T 实现
- `corep_fast/stages/s4_face_point.py:200-450` — facet 几何核
- `tmp/test_fast/diag_triple_sphere.py` — sphere case 对比脚本
- d780be8 commit message — 已记录这是后续 follow-up
