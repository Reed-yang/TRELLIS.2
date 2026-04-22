# corep 合作者改动研究与 merge 方案

**日期**: 2026-04-21
**作者**: Claude (research) + Reed
**目标分支**: `gpu-pipeline` ← `origin/corep`
**合作者**: lagwein <lagwein@tamu.edu>
**研究 worktree**: `.claude/worktrees/corep-review` (detached @ origin/corep 09cf05a)

---

## 结论（TL;DR）

- corep 相对 gpu-pipeline 的真实引擎分歧**极小**，远小于 diff stat 的表面数字。
- 只有 **1 个文件 1 处冲突**（`corep_fast/stages/s4_face_point.py` 的文档区域），解决方案**直接取 corep 版**，无逻辑丢失。
- corep 的主体贡献是**训练/数据管线工具层新增**（~5000 行纯新文件），与 gpu-pipeline 的 Phase 2 优化工作**正交**。
- **推荐策略**：直接 `git merge origin/corep` 到 gpu-pipeline，手动处理 s4 冲突后回归测试，然后决定 push。
- **唯一需要验证的语义变化**：pipeline.py 在 `_run_custom_s1_to_s7` / `corep_encode` / 新增的 `mesh_to_param` 内加入了 trimesh 规范化（merge_vertices / remove_unreferenced / unique_faces / nondegenerate_faces）。**会改变 e2e 基线**，必须跑回归 golden 验证是否通过。

---

## 1. 分叉历史

### Merge base
- `89a04fd` (2026-04-17) — "docs(s7): investigation + fix writeup for W2 bijective-matching bug"

### corep 相对 merge-base 领先 9 个 commit

```
* 09cf05a Add VAE comparison tools, feat18 scripts, and update training/precompute
* 6cd0bc5 wip: corep_fast optimizations, tests, and feat18 training scripts
*   e866bb7 merge: overfit experiments + profile-deep work from detached HEAD
|\
| * 41976be overfit
| * b05aeca overfit
* fc432ee Merge branch 'gpu-pipeline' (@ 731a939) into corep  — Apr 17 00:16
* 4646913 Merge branch 'gpu-pipeline' (@ 203b254) into corep  — Apr 16 18:28
* 76e763d data_toolkit
* f4b2a70 fix 1.ModuleNotFoundError: No module named 'datasets' 2.fix metadate
```

### gpu-pipeline 相对 merge-base 领先约 30 个 commit

你的 W_SD、W_HG、W_L2L、W_BAF、followup(p2) 系列，外加 post-profile-sync-elim 合并。**corep 最后一次同步 gpu-pipeline 是 `fc432ee`（Apr 17 00:16），只拉到 `731a939`（pre-triton/all），没有拉到后续 P2 followup 工作。**

### 最关键的判断点
- corep 的 `6cd0bc5` commit 在 **fc432ee 之后**，意味着合作者在拉完 pre-triton 后又**继续**做自己的优化。
- 但比对 gpu-pipeline HEAD 后发现：corep 新增的"优化"大部分**与 gpu-pipeline 同期/之后的 P2 工作最终产出相同或完全兼容**（见 §3）。合作者很可能是独立但方向一致地推进，最终只有细微增量（+107 行 s4 VRAM chunking，+13 行 s7 断言）。

---

## 2. 改动全貌（corep vs gpu-pipeline）

### 2.1 引擎核心（需关注）

| 文件 | 差异规模 | 性质 | 冲突? |
|---|---|---|---|
| `corep_fast/stages/s4_face_point.py` | +107 行 | VRAM 分块包装器 | ✅ 1 处冲突（文档区） |
| `corep_fast/stages/s7_rank_assign.py` | +13 行 | 防御性断言 | ❌ auto-merge |
| `corep_fast/pipeline.py` | +275 行 | 主要是纯新增 API（CorepParam/mesh_to_param/param_to_mesh/_unique_to_full_*），外加 2 处 mesh 规范化注入 | ❌ auto-merge |
| `corep_fast/stages/s6_collapse.py` | **0 行** | 完全一致 | — |
| `corep_fast/stages/s7_triton.py` | **0 行** | 完全一致 | — |
| `corep_fast/stages/s8_collapse.py` | **0 行** | 完全一致 | — |
| `corep_fast/utils/persistent_pool.py` | **0 行** | 完全一致（两边独立实现但等价） | — |
| `corep_fast/config.py` | **0 行** | 完全一致 | — |

### 2.2 `custom/` 目录（辅助算法）

| 文件 | 差异规模 | 性质 | 冲突? |
|---|---|---|---|
| `custom/collapse.py` | +129 行 | 新增 `feature_to_mesh()` + 新 imports + `__main__` 改写 | ❌ auto-merge |
| `custom/feature.py` | +281 行 | 新增 `full_to_unique_weights` / `unique_to_full_weights` / `_shift_grid`；注释掉原 collapse 系 import（破循环依赖）；改 `__main__` 测试路径 | ❌ auto-merge |

### 2.3 顶层训练/工具脚本（纯新增，零冲突）

| 文件 | 行数 | 用途 |
|---|---|---|
| `precompute_feat18.py` | 570 | 用 `mesh_to_param` + `param_to_feats` 预计算 18-dim CoReP feature 数据集 |
| `train_finetune_feat18.py` | 995 | 在 ObjaverseXL 1k 子集上 fine-tune SC-VAE，支持 DDP |
| `train_overfit_feat18.py` | 417 | 单 mesh overfit 训练，验证 encode/decode pipeline |
| `train_overfit_shape.py` | 562 | overfit 训练的 shape 分支 |
| `compare_vae_pipelines.py` | 671 | 对比不同 VAE pipeline 评估工具 |
| `compare_vaes.py` | 532 | VAE 模型对比工具 |
| `test.py` | 137 | 调试/验证脚本 |
| `test_preprocess.py` | 159 | 预处理测试 |
| `scripts/precompute_feat18_objaverse_sketchfab.sh` | 78 | shell 包装 |
| `scripts/train_finetune_feat18.sh` | 53 | shell 包装 |

### 2.4 `data_toolkit/` 小修

| 文件 | 行数 | 性质 |
|---|---|---|
| `data_toolkit/datasets/ObjaverseXL.py` | +94（新） | ObjaverseXL 数据集类 |
| `data_toolkit/asset_stats.py` | +13 | 修复 metadata bug |
| `data_toolkit/download.py` | +3 | 修复 ModuleNotFoundError |

### 2.5 测试用例 & 二进制 fixtures

corep 新增的测试文件 gpu-pipeline **都已有同名**（合作者独立实现过）：
- `corep_fast/tests/unit/test_hungarian_batched.py`
- `corep_fast/tests/unit/test_labels_to_list_vectorized.py`
- `corep_fast/tests/unit/test_stage_d_gpu_bfs.py`
- `corep_fast/tests/unit/test_s4_uf.py`
- `corep_fast/tests/unit/test_s6_fastpath_tracer.py`
- `corep_fast/tests/unit/test_persistent_pool.py`

两边应该会 merge 得到一份；验证后若完全一致则无忧。

corep 还新增了二进制 regression fixtures：
- `corep_fast/tests/regression/cpu_worker_optim_goldens/F1_icosphere_s3_r128.pkl` (9.9 MB)
- `corep_fast/tests/regression/cpu_worker_optim_goldens/F2_icosphere_s3_r256.pkl` (39 MB)
- `corep_fast/tests/regression/cpu_worker_optim_goldens/F3_triple_icosphere_r128.pkl` (29 MB)

**共 ~78 MB 二进制**，直接 merge 会进入 git 历史。⚠️ **建议评估是否迁到 Git LFS**。

---

## 3. 冲突详情（只有 1 处）

### `corep_fast/stages/s4_face_point.py`

**位置**：`_count_uturns_gpu_batched_csr` 函数的 docstring 区域（L478-587）。

**冲突原因**：corep 把原 `_count_uturns_gpu_batched_csr` 的"CSR packing + dispatch"实现重命名为 `_count_uturns_gpu_batched_csr_chunk`，并在 `_count_uturns_gpu_batched_csr` 位置放一个新的 VRAM 分块调度包装器。

- **HEAD (corep) 侧**：函数名不变，但 docstring 改成描述新包装器行为，+约 80 行新逻辑（按 segment count 排序 + 分块 + `torch.cuda.empty_cache()` + 逆序还原）。原实现以 `_count_uturns_gpu_batched_csr_chunk` 的名字在下方新增。
- **gpu-pipeline 侧**：保留原 `_count_uturns_gpu_batched_csr` 实现不变。

**解决方案**：**接受 HEAD (corep)** — 新包装器在原实现之上叠加，不删除任何 gpu-pipeline 逻辑。等价于：
```
git checkout --theirs corep_fast/stages/s4_face_point.py
# 然后 diff 验证无误
```

**合并动机**：此分块逻辑解决的是 res=512+ 场景下 `(G, P_MAX, P_MAX)` 内存爆炸，属于对大分辨率实用性的关键补丁。gpu-pipeline 未来做 res≥512 训练会用到。

**风险**：分块逻辑引入额外 `torch.cuda.empty_cache()` 调用（chunks 多时开销大）、`np.argsort` 排序 + 逆序还原的 CPU 开销。在 res≤256 的 benchmark 下走 fast path（`G * P_MAX^2 <= budget`，默认 250M 元素 ≈ 2GB），不会退化。建议：
- 合并后跑一次 3-run median wall time @ res=256 验证 fast path 确实命中、无回归。
- 默认的 `COREP_FAST_S4_UTURN_CHUNK_ELEMS=2.5e8` 在 24GB/48GB GPU 够用；若未来用 A100/H100 再放大。

---

## 4. 需要语义验证的改动

### 4.1 pipeline.py 的 mesh 规范化注入（⚠️ 会改 baseline）

在 3 个位置注入相同 6 行规范化代码：

```python
normalized_mesh = mesh.copy()
normalized_mesh.merge_vertices(merge_tex=True, merge_norm=True)
normalized_mesh.remove_unreferenced_vertices()
mask = normalized_mesh.unique_faces() & normalized_mesh.nondegenerate_faces()
normalized_mesh.update_faces(mask)
mesh = normalized_mesh
```

位置：
- `_run_custom_s1_to_s7`（L91 附近）
- `corep_encode`（L253 附近）
- 新增的 `mesh_to_param`（纯新增函数）

**影响**：对 degenerate / 重复顶点 / 未引用顶点的 mesh，结果会与之前不同。
- ✅ 正面：训练数据清洗，避免下游出错。
- ❌ 风险：如果你的 regression goldens（F1/F2/F3 icosphere）是用未规范化版本生成的，merge 后会 fail。

**应对**：merge 后立刻跑：
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
python -m pytest corep_fast/tests/regression/ -v -x
```
- 若 goldens fail → 两个选择：
  1. 认可新语义，用 corep 的 3 份 fixtures (`F1/F2/F3_*.pkl`) 替换旧 goldens（corep 已附带）。
  2. 回退规范化改动，只保留纯新增 API（`CorepParam`/`mesh_to_param`/`param_to_mesh`），把规范化挪到新 API 内部 — 这是我**推荐的保守路径**，因为能让合作者的训练 pipeline 正常运行（规范化对 robust training 确实有用），同时不污染已被你 benchmark 过的 `corep_encode` 与 `_run_custom_s1_to_s7` 路径。

### 4.2 custom/collapse.py 新增的 top-level import

```python
from feature import unique_to_full_weights
from collapse_face import collapse_face_inner, collapse_face_boundary
from collapse_point import collapse_point_inner, collapse_point_boundary
from utils import load_pickle, save_pickle, fetch_np_array, voxels_to_mesh
```

同时 `custom/feature.py` **注释掉了** `from collapse/collapse_point/collapse_face import ...` — 合作者在打破循环依赖。

**风险**：
- 这些 import 要求 `feature.py`、`utils.py`、`collapse_face.py`、`collapse_point.py` 都在 Python path 下（custom/ 下运行时应无问题）。
- 但若 gpu-pipeline 代码别处仍 `from collapse import reconstruct_mesh, mark_exception`（现在 feature.py 注释掉了这条），可能遇到 **NameError**。

**应对**：merge 后跑 `grep -rn "from collapse import\|from feature import" --include="*.py"` 检查全局使用。

### 4.3 custom/feature.py __main__ 的本地路径

```python
mesh_path = "/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/000-086/6eba14662bd048f9bc1ca10e63b5622f.glb"
resolution = 512
```

硬编码的本机路径。**不影响 merge，但建议单独 commit 改回参数化或移到 `__main__` 外**。

---

## 5. 独立发现：平行工作的收敛性

值得记录：合作者在 corep 分支做的 "W_SD / W_HG / W_L2L 同名优化"（6cd0bc5 提交消息里提到的 "hungarian, UF, BFS, pool, tracer"），**与 gpu-pipeline 已有实现最终合流为几乎等价**。

推测原因：
- 两边都继承了 `pre-triton/all`（731a939）之后 Phase 1 打下的算法骨架。
- 合作者在拉取 `731a939` 之后做的"优化"大概率是**在已有 gpu-pipeline 工作的同名文件基础上修补**（git 语义层面实际相当于"rebase 之后未改动"或"微调"），而不是重写。
- 所以 corep_fast/utils/persistent_pool.py、corep_fast/config.py 等文件 diff 为 0。

这个结论极大降低了 merge 复杂度。

---

## 6. 推荐 merge 方案

### 方案 A（推荐）：直接 3-way merge + 手动处理冲突 + 保守回退规范化

**分三步**：

**Step 1** — 直接 merge：
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git checkout gpu-pipeline
git pull  # 同步 origin/gpu-pipeline
git merge origin/corep --no-commit
# 唯一冲突：corep_fast/stages/s4_face_point.py
git checkout --theirs corep_fast/stages/s4_face_point.py
# 再次确认
git diff HEAD corep_fast/stages/s4_face_point.py | wc -l  # 应 = corep vs gpu-pipeline diff
git add corep_fast/stages/s4_face_point.py
```

**Step 2** — 保守回退 pipeline.py 规范化（如果 §4.1 回归 fail）：

只在 `mesh_to_param` 内保留规范化；从 `_run_custom_s1_to_s7` 和 `corep_encode` 删除那 6 行：

```python
# 保留：
def mesh_to_param(...):
    mesh = trimesh.load(mesh_path, force='mesh')
    normalized_mesh = mesh.copy()
    normalized_mesh.merge_vertices(...)
    ...

# 回退：
def _run_custom_s1_to_s7(...):
    mesh = trimesh.load(mesh_path)  # 不加 normalization
    
def corep_encode(...):
    mesh = trimesh.load(mesh_path)  # 不加 normalization；保留 force='mesh'?
```

注：`corep_encode` 的 `trimesh.load(mesh_path, force='mesh')` 的 `force='mesh'` 参数本身是 corep 加的，属于"可选无害增强"。建议保留 `force='mesh'`，删除规范化。

**Step 3** — 验证 + commit：
```bash
python -m pytest corep_fast/tests/ -v --ignore=corep_fast/tests/regression/cpu_worker_optim_goldens
# 跑 wall bench @ res=256 @ GPU 0 on host 119，3-run median
# 对比 399beb3 (current HEAD) 的数字，偏差 <5% 视为 OK
git commit -m "merge(corep): feat18 training pipeline + s4 VRAM chunking + safety asserts

- Adds CorepParam / mesh_to_param / param_to_mesh for training round-trip API
- Adds full_to_unique_weights / unique_to_full_weights helpers in custom/
- Adds _count_uturns_gpu_batched_csr chunking wrapper for res>=512 VRAM limits
- Adds edge_weights.max() <= W_MAX defensive assert in s7 _build_adjacency_gpu
- Adds train_finetune_feat18 / train_overfit_feat18 / precompute_feat18 training scripts
- Adds compare_vaes / compare_vae_pipelines eval tools
- Adds data_toolkit ObjaverseXL dataset support

Note: mesh normalization in _run_custom_s1_to_s7/corep_encode reverted to
preserve benchmark baseline; kept only inside the new mesh_to_param API."
```

### 方案 B（更保守）：cherry-pick 拆分

拒绝合作者的 2 次 gpu-pipeline merge（4646913、fc432ee），只 cherry-pick 这些 commit：
- `f4b2a70` (data_toolkit 修 bug)
- `76e763d` (data_toolkit)
- `b05aeca` (overfit 新增文件)
- 从 `41976be` 取 custom/feature.py 的 `full_to_unique_weights`/`unique_to_full_weights`、collapse.py 的 `feature_to_mesh`
- 从 `6cd0bc5` 取 pipeline.py 的 `CorepParam`/`mesh_to_param`/`param_to_mesh`、s4 的 chunking、s7 的 assert
- `09cf05a` (compare_vaes + precompute_feat18 等顶层脚本)

优点：每个 cherry-pick 都可单独 commit message 记录意图，历史更清晰。
缺点：工作量大（需要手动拆 41976be 和 6cd0bc5），可能产生更多冲突。

**除非你很介意 merge commit 污染历史，否则 A 更省力。**

### 方案 C（最保守）：全部拒绝，手动搬运

只把 `CorepParam` / `mesh_to_param` / `param_to_mesh` / `full_to_unique` / `unique_to_full` 几个核心函数手动复制过来，外加训练脚本。完全不动 s4/s7/pipeline_edit 现有代码。

适用场景：如果你担心后续 P3 工作会继续大改 s4/s7 且不想 corep 的改动成为历史包袱。

---

## 7. 风险与未决问题

1. **二进制 fixtures 约 78MB** — 是否进 git history 还是走 LFS？
2. **合作者未来还会继续在 corep 上开发吗？** 如果是，merge 后你的分支会和他反向 diverge。建议：merge 完成后 push 到一个约定分支（例如 `corep-sync`），让合作者 rebase 他的后续工作。
3. **`feature.py` 注释掉的 imports** — 需要全局 grep 确认没有破坏现有调用链。
4. **mesh 规范化的正确性** — `merge_vertices(merge_tex=True, merge_norm=True)` 会破坏 vertex 属性（uv、法线），对纯几何管线无害，但若未来需要 texture 会踩坑。
5. **`corep_encode` 的 `force='mesh'` 参数** — trimesh 会把 scene 压扁成单个 mesh；若数据集里有多组件 scene，语义会变。建议保留但要记录到 release note。

---

## 8. 下一步建议

**立即可做**：
- 若你接受方案 A，用上面的 3 步直接推进。我可以进入 gpu-pipeline 主工作目录执行，或者留在本 worktree 里继续准备脚本。

**推荐追加任务**：
1. merge 完成后，跑一次 `test_finetune_feat18.py` smoke 测试（即使只跑 1 step），验证新训练 pipeline 在 gpu-pipeline 代码基上能跑通 — 这是"合作者代码能否接入"的最终证据。
2. 更新 `logs/progress.md` 和 `CLAUDE.MD`（如果你的 CLAUDE.MD 提到分支状态）。
3. 考虑 push 一个 `corep-merged` 分支给合作者作为他后续工作的新起点。

**需要你决策**：
- [ ] 采纳方案 A / B / C？
- [ ] pipeline.py 的 mesh 规范化：保留 / 只保留在 `mesh_to_param` 内 / 全部删除？
- [ ] 二进制 fixtures：直接 merge / 换 LFS / 只挑 F1 保留省空间？
- [ ] merge 在当前 worktree 还是回到主目录执行？

---

*研究方法：detached worktree checkout origin/corep，用 `git diff gpu-pipeline origin/corep`、`git show` 和 dry-run `git merge gpu-pipeline` 交叉验证实际冲突 vs 表面 diff 的差异。*
