# corep_fast Mesh Cleanup 移植（Follow-up）

**Status**: 已诊断，未实施。  
**Owner**: 待分配  
**Estimated effort**: 0.5 day（含测试）  
**Related**: commits `d780be8`（centering fix）、`4a355c3`（s6 测试放宽）  
**Companion follow-up**: `20260416-corep-fast-s1-sat-hardening.md`

---

## 1. 动机

`corep_fast/containers.py::MeshTensors.from_trimesh` 在归一化输入网格时**漏移植了 custom/voxelize.normalize_mesh 的 4 步网格清理**。完整对照：

| 步骤 | custom/voxelize.py:27-31 | corep_fast/containers.py:69-90 |
|------|--------------------------|--------------------------------|
| `mesh.copy()` | ✓ L27 | ✓（隐式：直接读 vertices） |
| `merge_vertices(merge_tex=True, merge_norm=True)` | ✓ L28 | **❌ 缺失** |
| `remove_unreferenced_vertices()` | ✓ L29 | **❌ 缺失** |
| `unique_faces()` 过滤重复面 | ✓ L30 | **❌ 缺失** |
| `nondegenerate_faces()` 过滤退化面 | ✓ L30 | **❌ 缺失** |
| 归一化（已修） | ✓ L36-56 | ✓ L74-90 |

### 为什么 sphere 测试 case 不需要这些清理

`d780be8` 已经修复了 sphere 类对称网格在 res=32 三层球 case 的 7 层伪峰问题。那个 fix 只动了 centering offset，没动 cleanup。validate 后 fast↔custom 的 cube 注册数差异已 100% 归零、CD 降 55%。

**因此 sphere/icosphere 类**程序化生成**的干净网格不需要 cleanup**。

### 为什么真实数据集仍需要

但下面这些"脏"输入会触发 cleanup 漏失的影响：

1. **GLB / OBJ 中导出的网格**：
   - 同一空间位置常因 UV / normal seam 拆出多个 vertex —— `merge_vertices` 才能合成一个拓扑顶点。
   - 不合并的话 mesh.face_adjacency 会少算邻接，s2 的 num_components 偏多、s6 的 loop 提取受影响。

2. **数据集预处理产物**：
   - 历史脚本输出可能含**未引用的顶点**（被 face 删除但 vertex 没清理）。这些不影响输出但浪费 `MeshTensors.vertices` 张量内存与 GPU 转换带宽。

3. **CAD / 雕塑工具导出**：
   - 重复面（同 3 顶点出现多次）和退化面（3 顶点共线 / 重合）会让 Möller-Trumbore 求交、SAT 测试得到 NaN 或除零。
   - custom 的 `unique_faces() & nondegenerate_faces()` 在归一化前就把这些剔除了，fast 会带病进入 s1。

4. **观察迹象**：
   - profiling 在 `assets/sample.glb` 上跑 fast 比 custom 慢 ~3% 且 V/F count 差异 0.05% —— 与 cleanup 缺失一致。
   - bowl_512、helmet_512 等"真实模型"的 e2e 测试有零星 cube 数差（< 0.1%），sphere 和 cube 完全 0 差。

### Risk 与不修的代价

短期可控（regression suite 全过、icosphere 完美）。**但**：
- 任何用户用 trimesh 默认加载 .glb / .obj 的输入都会**静默地**得到与 custom 不一致的结果（差异极小但持续存在），未来若有 bug 报告，溯源困难。
- 退化面在极端 case 下可能直接让 fast 输出无效顶点（NaN 坐标 → 后续 voxel hash 计算崩溃）。

---

## 2. 实施方案

### 2.1 主要改动

**File**: `corep_fast/containers.py`  
**Function**: `MeshTensors.from_trimesh`  
**Where**: 在 L70 `verts_np = np.asarray(mesh.vertices, ...)` **之前**插入清理。

```python
# corep_fast/containers.py — proposed edit
@classmethod
def from_trimesh(
    cls,
    mesh: trimesh.Trimesh,
    resolution: int,
    device: str | torch.device = 'cpu',
) -> 'MeshTensors':
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(...)

    # NEW: mesh cleanup matching custom/voxelize.normalize_mesh L27-31.
    # Operate on a copy so caller's mesh is not mutated. These trimesh ops
    # all run on CPU-side numpy/cython and are O(V+F); on a 100K-face mesh
    # cleanup adds <50 ms — small relative to S1 voxelization.
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    valid_face_mask = mesh.unique_faces() & mesh.nondegenerate_faces()
    mesh.update_faces(valid_face_mask)

    # Re-check after cleanup in case all faces were degenerate.
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(
            "MeshTensors.from_trimesh: mesh is empty after cleanup "
            "(merge_vertices + nondegenerate_faces removed all geometry)"
        )

    device = torch.device(device)
    verts_np = np.asarray(mesh.vertices, dtype=np.float32)
    # ... (existing code unchanged from here)
```

### 2.2 测试改动

**File**: `corep_fast/tests/unit/test_containers.py`  
**Add**: 4 个新测试覆盖 cleanup 路径

```python
def test_mesh_tensors_merges_duplicate_vertices():
    """Two coincident vertices used by separate faces should be welded."""
    verts = np.array([
        [0,0,0], [1,0,0], [0,1,0],     # face 0
        [0,0,0], [1,0,0], [0,1,0.5],   # face 1, shares first 2 verts coordinate-wise
    ], dtype=np.float32)
    faces = np.array([[0,1,2], [3,4,5]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    # After merge_vertices, only 4 unique verts remain (not 6).
    assert mt.vertices.shape[0] == 4

def test_mesh_tensors_removes_unreferenced_vertices():
    verts = np.array([[0,0,0], [1,0,0], [0,1,0], [99,99,99]], dtype=np.float32)
    faces = np.array([[0,1,2]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.vertices.shape[0] == 3   # vertex [99,99,99] dropped

def test_mesh_tensors_removes_duplicate_faces():
    verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
    faces = np.array([[0,1,2], [0,1,2]], dtype=np.int32)  # same face twice
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.faces.shape[0] == 1

def test_mesh_tensors_removes_degenerate_faces():
    verts = np.array([[0,0,0], [1,0,0], [2,0,0], [0,1,0]], dtype=np.float32)
    # face [0,1,2] is collinear (degenerate); face [0,1,3] is valid.
    faces = np.array([[0,1,2], [0,1,3]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.faces.shape[0] == 1
```

### 2.3 不会破坏的现有测试

- `test_e2e_gpu_ab.py`：fast↔custom 都做相同 cleanup，差异不变（仍 ≤0.1%）。
- regression 套件：`run_custom_through_stage` 调用 custom path 已自带 cleanup；fast path 加 cleanup 后只会让两路径更对齐。
- 其他 unit test：所有 fixture 都是程序化生成的干净网格（icosphere、box、open_plane），cleanup 是 no-op。

### 2.4 性能影响

`merge_vertices` 在 trimesh 内部用 KD-tree 做坐标 hash 合并，O(V·log V)。100K 面的网格大约 30–80ms。相对：
- S1 voxelize res=256: ~190ms
- e2e res=256: ~18s

清理开销 < 0.5%，可忽略。

### 2.5 验证步骤（Definition of Done）

1. 跑 `pytest corep_fast/tests/unit/test_containers.py -v` —— 4 个新测试通过。
2. 跑 `pytest corep_fast/tests/ -v`（unit + regression + stages）—— 仍全绿。
3. 在 119 节点跑 e2e regression suite —— V/F ratio 仍在 0.999~1.001 容忍内。
4. 在真实 GLB 数据上对比 pre/post-cleanup 的 V/F count + CD —— 期望差异进一步收敛。
5. 用一个故意脏的输入（重复顶点 + 退化面）跑一遍，验证不再 NaN/崩溃。

---

## 3. 不在本任务范围内的事项

- **不动** s1/s2/s3/s4/s6/s7/s8 任何 stage 实现 —— 单点 patch 即可。
- **不改** centering offset（已 fix）。
- **不动** custom/voxelize.py（保持参考实现不变）。
- 若发现 cleanup 后仍有真实输入残差，转下一份 follow-up（见 `20260416-corep-fast-s1-sat-hardening.md`）。

---

## 4. 参考

- `custom/voxelize.py:26-58`（normalize_mesh 全文）
- `corep_fast/containers.py:56-116`（fast normalize 全文）
- `tmp/test_fast/test_patched_centering.py`（已含 cleanup 的 monkey-patch 验证脚本，可作为实施前的 smoke test）
- commit `d780be8` 的 message（已记载这块 follow-up 是 known gap）
