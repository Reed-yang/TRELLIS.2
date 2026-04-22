# feat18 VAE finetune — 训练脚本审查 & 8-GPU 启动前置工作

**Date**: 2026-04-22
**Scope**: 评估 `train_finetune_feat18.py` 是否对齐 TRELLIS.2 原生 Shape-VAE 训练 infra；列出 1-node 8-GPU 启动前的前置工作。
**Constraint from user**: 本次训练明确**不使用 render loss**（mask/depth/normal/ssim/lpips 全部不接）。

---

## 0. 关键数据事实（三方独立扫描交叉验证）

| 指标 | 值 | 来源 |
|---|---|---|
| 总样本数 | 41,871 npz | 主扫 |
| 单点 cube (p1≠0, p2=[0,0,0]) | **97.79%** | 131M cubes 主扫 / 69.6M subagent-A / 29.3M subagent-B |
| 双点 cube | 2.21% | 同 |
| p1 严格等于 p2 | 0.70% | 同 |
| p1 mean/std (local) | (0.50, 0.50, 0.50) / 0.228 | 同 |
| p2 mean (含 0) / std | 0.010 / 0.078 | 同 |
| edge_weights 值域 | 整数 0~22 (长尾)，99.1% ∈ {0,1} | 同 |
| face_weights 值域 | 整数 0~5，**99.83% = 0** | 同 |
| edge/face 语义 | **ordinal count** — `corep_fast/stages/s3_edge_weights.py:105` 用 `scatter_add_` 累计光线-三角形交点数 | subagent-B |
| Pretrained input_layer.W[:,0:3] vs [:,3:6] 正交性 | cos-similarity mean ≈ −0.05 | subagent-A |

---

## 1. 18-ch feat 布局 (verified)

`train_overfit_feat18.py:36-44` — 每 voxel 18 维：
```
[point1_xyz(3), point2_xyz(3), edge_weights(6), face_weights(6)]
```
- p1/p2 **asymmetric**: 按 local z 升序排（p1=low z, p2=high z）
- 单点 cube: p2 = [0,0,0] 严格（不是 ≈ p1）
- ≥2 点 cube: 取 farthest pair 再 z-sort
- Points ∈ [0,1] local cube coords
- Edge/face 是 int32 **ordinal count**，不是 categorical

---

## 2. 对比：原生 `ShapeVaeTrainer` (ft-512.json) vs `train_finetune_feat18.py`

| 维度 | 原生 ft-512 | train_finetune_feat18.py | Delta |
|---|---|---|---|
| 输入 ch | 6 (vertex 3 + intersected 3) | 18 | 预期差异 |
| Recon loss | `vertice*0.01 + intersected*0.1` + render | `mse(h, feats)` 全 18ch normalised | 符合 no-render 约束 |
| Subdiv loss | 0.1 × BCE(sum) | 0.1 × BCE(mean) | 等价 |
| KL loss | 1e-6 | 1e-6 ✓ | 一致 |
| Render | mask=1/depth=10/normal=1/ssim=0.2/lpips=0.2 | ❌ | ✅ 按用户要求 |
| Optimizer | AdamW, wd=0 | AdamW, wd=0 ✓ | 一致 |
| **lr** | **1e-5** | **1e-4** | ⚠️ 10× 过高 |
| Max steps | 1,000,000 | 50,000 | 需按 scope 定 |
| Batch/GPU | 4, split=2 (eff=2) | 1, no accum | 信号嘈杂 |
| Grad clip | `AdaptiveGradClipper(max=1, p95)` | 固定 1.0 | warmup 偏紧 |
| **EMA** | `0.9999`，每步更新，分开 ckpt | ❌ 无 | 🔴 阻断 |
| AMP | fp16 inflat_all + master fp32 | bf16 autocast（默认关） | 两种都可 |
| DDP | bucket=128, fu_p=False | 同 ✓ | 一致 |
| **Ckpt** | step 编号 + EMA 分存 + resumable | 覆盖写单文件，无 resume | 🔴 阻断 |
| Eval | held-out val | eval==train | 🟡 无泛化 |
| Aesthetic 过滤 | `min=4.5` | 全量 | 🟡 |
| max_active_voxels | 1,000,000 | `--max_voxels=0` 默认 | 🟡 OOM |

---

## 3. IO warm-start 决策（两个 opus subagent 研究结论）

### 关键数学发现（subagent A）

当前 `load_pretrained_into` (lines 238-253) 在 97.79% 的单点 cube 上：
$$
\text{out} = 0.5 \cdot W_{pre}[:,0:3] \cdot (p_1 - 0.5) + 0.5 \cdot W_{pre}[:,0:3] \cdot (-0.5) + b
$$
= 半强度 vertex signal + **$-0.25 \cdot W_{pre}[:,0:3] \cdot \mathbf{1}$ 死常数 bias**（和原生 intersected 通路几乎正交，**不是**原生 forward 的近似）。

Backbone 用 `LayerNorm32` 抹平 magnitude → **只有方向重要**，warm-start 只在 3/18 方向正确。

Decoder 侧 `output_layer.weight[3:17]=0, bias[3:17]=0` → **15 rows dead-row trap**；且 encoder point2 通路被 zero → decoder point2 rows 也被卡零，形成 chicken-and-egg 依赖。

### 架构拆分（subagent B）

Option: 3-branch encoder (`Linear(3→C0) shared for p1/p2` + `Linear(12→C0) for ef`) + 2-head decoder (`Linear(C_end→3)` × 2 + `Linear(C_end→12)`)。
需要 permutation-invariant MSE 配合。

### 共识 & 决策

| 方案 | 代码量 | 改 repo 组件? | 预期 vs scratch |
|---|---|---|---|
| Option 0: `--no_warmstart_io`（当前默认的 xavier） | 0 | 否 | baseline |
| **Option 1: Partial warm-start (point1 only, 全强度, 无缩放/复制)** | ~10 行，只改 `load_pretrained_into` | 否 | +5-10% |
| Option 2: 三分支架构 + permutation loss | ~130 行，monkey-patch IO 层 | 是 | +5-10%，需 smoke test |

**决定**：采用 **Option 1**（详细 diff 见 §5 的执行清单）。不用 Option 2 —— 违反 `feedback_no_modify_repo` 且需额外验证 permutation loss；不用 Option 0 —— Option 1 的成本只有修一个 bug 的量级。

---

## 4. 阻断级风险（需另外决策）

### R1 — 无 EMA （🔴 high, medium effort）
原生 pipeline 下游加载 EMA 权重。若缺失：(a) 交付 artifact 语义不对；(b) online 权重振荡。

### R2 — 无 resume + 覆盖写单 ckpt （🔴 high, medium effort）
200k+ 步跑预期会 preempt。`torch.save` 覆盖写中断 → 唯一 ckpt 损坏。需原子 save + rolling N + `--resume_from`。

### R3 — lr 1e-4 vs 原生 1e-5 （🟡 zero effort）
Unfreeze 瞬间 backbone 吃 10× 过高梯度，有 catastrophic forgetting 风险。

### R4 — max_voxels=0 默认 （🟡 zero effort）
单文件峰值 499k voxels，P99 可能 ≥1M。8 卡 × bs1 × P99 → 随机 OOM。建议启用 `--bucket_sampler` + `--max_voxels=500000`。

### R5 — eval ≡ train （🟡 low effort）
无 held-out。建议从 41k 切 200 个做 val。

---

## 5. 启动前置清单 (待决策定 scope 后细化)

### 立即必做
1. **跑 `scripts/preprocess-by-rank/compute_stats.py`** 生成 `stats_global.npz`
2. **修复 `load_pretrained_into`** 的 warm-start 数学（Option 1 diff）
3. **lr 对齐到 1e-5** (或加 unfreeze linear warmup)
4. **启用 `--bucket_sampler` + `--max_voxels=500000`**（显存安全）

### Scope 相关（production 必做，research baseline 可跳）
5. **补 EMA 0.9999**（~50 行，仿照 `trellis2/trainers/basic.py:229-241`）
6. **补原子 save + rolling K 历史 ckpt + `--resume_from`**（~80 行）
7. **切 held-out val split**（~20 行，按 sha 哈希分）

### 可选优化
8. AdaptiveGradClipper（~30 行，或直接用 `trellis2/utils/grad_clip_utils.py`）
9. logging cadence 降到 `i_log=100`
10. `--sample_at_step_one` 默认开

---

## 6. 环境事实核对

| 事实 | 数值 |
|---|---|
| `data/*.npz` 数 | 41,871 |
| 单样本 voxel 峰值 | 499,885 |
| stats 文件 | **缺**（需 compute_stats.py） |
| Pretrained | `pretrained/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f.../ckpts/shape_{enc,dec}_next_dc_f16c32_fp16.safetensors` ✓ |
| metadata.csv | 有（sha, aesthetic_score, captions） |
| GPU 位置 | 119 节点 GPU 0-3 idle（但 feedback 说仅 profiling；production 训练地点待定） |
