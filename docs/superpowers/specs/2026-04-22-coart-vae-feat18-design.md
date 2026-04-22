# coart VAE feat18 Finetune — Design Spec

**Date**: 2026-04-22
**Author**: session-level brainstorm with 2× opus subagent cross-validation
**Topic**: 重构 `train_finetune_feat18.py` 为 `coart/` 子包形态，引入三分支 IO 架构 + EMA + atomic resume，面向 1-node 8-GPU production run（目标产出能替代原生 shape VAE 给下游 DiT 使用的 feat18-decoder）

---

## 0. Constraints（来自用户）

- **No render loss**（mask / depth / normal / ssim / lpips 全部不接）
- **1-node 8-GPU**，目标能 resume 过 preempt / OOM
- **Production 定位**：输出 ckpt 要给下游 DiT finetune 加载
- **不改** `trellis2/` 核心组件（`import` 可以，`edit` 不可以）
- **不改** 原 `train_finetune_feat18.py`（保留作为 baseline）
- **新增代码全部放** `coart/` 子包下

---

## 1. Scope

本 spec 只覆盖：
- 新增 `coart/` 子包（含 `coart/vae/` 本 task + `coart/common/` + `coart/data/` 共享层）
- 修复原 warm-start 数学 bug（partial → three-branch independent）
- 补齐 EMA / atomic save / resume 三项 production infra
- 对齐原生 ft-512 config 的关键超参（lr / grad clip / AMP）
- 提供 2×2 ablation flag matrix

不覆盖（future work）：
- Rotation / flip augmentation（需 edge/face channel direction remap）
- Aesthetic filter（先不过滤，看 val 曲线后决定）
- DiT finetune（placeholder 留位）
- fp16_inflat_all AMP path（改用更简单的 bf16 autocast）

---

## 2. Package Layout（task-first）

详见 `logs/decisions_coart_org.md`。核心结构：

```
coart/
├── __init__.py
├── common/                     # task-agnostic infra
│   ├── ema.py
│   ├── checkpoint.py
│   ├── dist_utils.py
│   ├── flex_gemm_patch.py
│   └── logging.py
├── data/                       # 共享数据管道
│   ├── feat18_dataset.py
│   ├── samplers.py
│   └── stats.py
├── vae/                        # 本 task
│   ├── __init__.py
│   ├── config.py
│   ├── io_stems.py
│   ├── build.py
│   ├── loss.py
│   ├── sampling.py
│   ├── train.py
│   └── __main__.py
└── dit/                        # future placeholder
    └── __init__.py             # TODO: reserved for future DiT finetune
```

启动：`torchrun --standalone --nproc_per_node=8 -m coart.vae [args...]`

---

## 3. IO Architecture（核心新增）

### 3.1 `io_arch=three_branch`（default）

**Encoder input stem `Feat18EncIO`**:
```python
h = p1_branch(f[:, 0:3]) + p2_branch(f[:, 3:6]) + ef_branch(f[:, 6:18])
```
- `p1_branch: nn.Linear(3, C0)`（独立，不共享）
- `p2_branch: nn.Linear(3, C0)`（独立，不共享）
- `ef_branch: nn.Linear(12, C0)`

**Decoder output stem `Feat18DecIO`**:
```python
out = concat([p1_head(f), p2_head(f), ef_head(f)])  # 18-dim
```
- `p1_head: nn.Linear(C_end, 3)`
- `p2_head: nn.Linear(C_end, 3)`（独立）
- `ef_head: nn.Linear(C_end, 12)`

### 3.2 `io_arch=monolithic`（ablation option）

保留原生 `sp.SparseLinear(18, C0)` / `sp.SparseLinear(C_end, 18)` 单层结构，作为 ablation baseline。

### 3.3 Warm-start 初始化矩阵

| `--io_arch` | `--warmstart_io` | p1 init | p2 init | ef init |
|---|---|---|---|---|
| three_branch | on（default） | W=`pre_enc.W[:, 0:3]`, b=`pre_enc.b` | W=`pre_enc.W[:, 0:3]`, **b=0** | xavier / 0 |
| three_branch | off | xavier / 0 | xavier / 0 | xavier / 0 |
| monolithic | on | W[:, 0:3]=`pre_enc.W[:, 0:3]`, b=`pre_enc.b`；其余保持 xavier | （同 cols） | xavier |
| monolithic | off | xavier / 0（全部） | - | - |

Decoder 镜像处理（rows 0:2 warm-start point1_head，rows 0:2 warm-start point2_head with bias=0，ef_head xavier）。

### 3.4 Permutation-invariant loss？**不需要**

Independent p1/p2 branches 保留了 z-sort 信号（p1=low z, p2=high z 的数据约定直接映射到两个 branch），不需要 min-over-permutation MSE。这是选 independent over shared-point 的关键 rationale。

### 3.5 Rationale（from subagent 研究）

数据统计（131M cubes 三方独立验证）：
- 97.79% cubes 是单点（p2=[0,0,0]）
- 2.21% 是双点
- p1=p2 严格相等仅 0.70%
- edge 值 99.1% ∈ {0,1}, face 值 99.83% = 0（ordinal count, 非 categorical）

原 `load_pretrained_into` 的 bug：`W[:, 0:3] = 0.5·W_pre[:, 0:3]` + `W[:, 3:6] = 0.5·W_pre[:, 0:3]` 在 97.79% 单点 cube 上产生半强度 vertex signal + 常数死 bias（因为 p2=[0,0,0] 归一化后成 [-0.5]³），**实际上 degrade 而非 improve** 起点。三分支 independent + 全强度 warm-start 修复这个。

Backbone 用 `LayerNorm32`（per-voxel, no running stats）→ input magnitude 无所谓，只有 direction 重要。

---

## 4. Loss Composition

```python
loss = loss_recon + lambda_kl * loss_kl + lambda_subdiv * loss_subdiv
```

- `loss_recon = F.mse_loss(h.feats, x.feats)` 在 normalised 空间（即 `(feats - mean) / std` 后）
- `loss_kl = 0.5 * mean(mu² + exp(logvar) - logvar - 1)`
- `loss_subdiv = mean([BCE_with_logits(sub_pred, sub_gt) for each level])`

**NO render loss**（用户约束）。

**Block-decomposed logging**（三分支副产品，免费）:
- `loss_recon/p1`: MSE on pred[0:3] vs gt[0:3]
- `loss_recon/p2`: MSE on pred[3:6] vs gt[3:6]
- `loss_recon/ef`: MSE on pred[6:18] vs gt[6:18]
- `loss_kl`, `loss_subdiv`, `loss_total`
- `misc/voxels_per_step`, `misc/it_per_s`, `misc/lr`

**权重**:
- `lambda_kl = 1e-6`（对齐原生）
- `lambda_subdiv = 0.1`（对齐原生）

---

## 5. Optimisation

| Item | Value | Rationale |
|---|---|---|
| Optimizer | AdamW, wd=0 | 对齐原生 ft-512 |
| lr | **1e-5** | 对齐原生 ft-512（原脚本 default 1e-4 过高） |
| LR scheduler | None (constant) | 对齐原生 |
| **LR unfreeze warmup** | 500-step linear 0→lr after unfreeze | unfreeze 瞬间 backbone 不被冲 |
| Grad clip | `AdaptiveGradClipper(max_norm=1.0, pct=95)` | 复用 `trellis2/utils/grad_clip_utils.py`；对齐原生；bs=1 + 75× voxel 方差下必需 |
| AMP | **bf16 autocast on** (default `--use_bf16`) | 简单 + Hopper/Ada 数值足够；losses 保持 fp32 |
| `freeze_backbone_steps` | 2000 | 先让 IO + KL stems 学到合理映射，再 unfreeze backbone |
| Batch size / GPU | 1 | 保留；如 loss 噪声大再加 grad_accum=2 |
| Grad accumulation | 0（global batch=8） | 简化；后续 follow-up 再加 |

---

## 6. Data Pipeline

| Item | Value | Notes |
|---|---|---|
| Stats file | `stats_global.npz`（需先跑 `scripts/preprocess-by-rank/compute_stats.py`） | 归一化 ch 6:17 必需 |
| Normalisation | ch 0:5: `x - 0.5`（std=1）；ch 6:17: `(x - μ) / σ` | `coart/data/stats.py` |
| `--max_voxels` | 500000 | 覆盖 P95~P99；防 OOM；deterministic sha-seeded subsampling |
| `--bucket_sampler` | on (sort_mode=shuffle) | 75× voxel 方差下防 straggler |
| Augmentation | integer translation only (±16 default) | rotation/flip 需 edge/face direction remap → TODO future |
| **Val split** | 按 sha 前 8 hex digit → `int(sha[:8], 16) % 200 == k` → k=0 为 val set | deterministic, ~200 samples |
| Aesthetic filter | **None**（先保 41k 全量） | 若 val 曲线劣化再按 `min_aesthetic_score=4.5` 过滤 |
| `num_workers` | 2 per-GPU (total 16) | 防 CPU / RAM 压爆 |

---

## 7. Training Schedule

| Item | Value |
|---|---|
| `--max_steps` | 200000 (production 起点) |
| `--freeze_backbone_steps` | 2000 |
| `--i_log` | 100 (每 100 步 all-reduce + TB write) |
| `--i_save` | 5000 |
| `--i_sample` | 5000 (dump mesh) |
| `--i_val` | 5000 (val MSE 曲线) |
| `--sample_at_step_one` | on (warm-start sanity check) |

---

## 8. Infra（阻断级补齐）

### 8.1 EMA（`coart/common/ema.py`）

- `class EMAModel`:
  - `__init__(model, decay=0.9999)`: store shadow fp32 params
  - `update(model)`: `shadow = decay · shadow + (1 - decay) · live`，每 optimizer step 调用一次
  - `state_dict() / load_state_dict()`
  - `copy_to(model)`: 把 shadow 写回 model.parameters()（用于 eval / checkpoint 导出）
- 仿 `trellis2/trainers/basic.py:229-241` 的做法

### 8.2 Atomic Save + Rolling（`coart/common/checkpoint.py`）

- `atomic_save(obj, path)`:
  1. `torch.save(obj, path + ".tmp")`
  2. `os.replace(path + ".tmp", path)`（原子）
- `save_ckpt(state_dict, output_dir, step, keep_k=3)`:
  - Write `ckpt_step{step:07d}.pt` via `atomic_save`
  - Glob existing `ckpt_step*.pt`, sort by step, delete all but latest K
  - EMA 单独存为 `ckpt_ema0.9999_step{step:07d}.pt`
  - Misc state（optimizer, sampler.epoch, RNG）存为 `misc_step{step:07d}.pt`
- `load_for_resume(output_dir) → dict | None`:
  - Glob `ckpt_step*.pt`，找到最大 step
  - 加载 encoder / decoder / EMA / optimizer / sampler.epoch / RNG
  - 返回 `(state, step)`；不存在则返回 `None`（fresh start）

### 8.3 Resume（`coart/vae/train.py`）

- `--resume_from latest` → 调用 `load_for_resume(output_dir)`
- `--resume_from <explicit_path>` → 直接 load 指定 ckpt（optimizer/sampler/RNG 若有对应 misc 文件则 load，否则 warn + 跳过）
- 恢复顺序：
  1. encoder / decoder state
  2. EMA shadow params
  3. optimizer state
  4. sampler epoch（`sampler.set_epoch(saved_epoch)`）
  5. RNG state（numpy + torch）
  6. step counter
- 支持 mid-training preempt 后无缝继续

### 8.4 Dir 碰撞处理

- Output dir 已存在 + 非空 + 无 `--resume_from` → error: `"output dir already populated; pass --resume_from latest or change --run_tag"`
- Output dir 已存在 + 非空 + `--resume_from latest` → 进入 resume 流程
- Output dir 已存在 + 空 → OK（fresh）
- Output dir 不存在 → 创建

---

## 9. Output Dir Naming

Format: `results/coart_feat18_{YYYYMMDD}_{run_tag}/`

- `{YYYYMMDD}`: 8-digit date, auto-build at run start
- `{run_tag}`: CLI `--run_tag <str>` 必填，regex `^[a-zA-Z0-9_-]+$`
- `--output_dir <path>` 可选 override：显式传入则忽略 auto-build，直接用（主要用于 cross-day resume）

Examples:
- Default: `results/coart_feat18_20260422_three_branch_ws_v0/`
- Ablation: `results/coart_feat18_20260422_three_branch_no_ws/`
- Resume: `--output_dir results/coart_feat18_20260422_three_branch_ws_v0 --resume_from latest`

---

## 10. Ablation Matrix

Four canonical runs（各取一个 run_tag，共享其他 hyperparams）:

| Run | `--io_arch` | `--warmstart_io` | Purpose |
|---|---|---|---|
| A (main) | `three_branch` | `on` | **Default** — architecture + warm-start |
| B | `three_branch` | `off` | Test architecture 单独贡献 |
| C | `monolithic` | `on` | Partial warm-start 修复后的单层 baseline |
| D | `monolithic` | `off` | Pure scratch IO baseline |

4-way loss curve 对比即可回答"是 architecture 还是 warm-start 的功劳"。

---

## 11. CLI Contract

```bash
torchrun --standalone --nproc_per_node=8 -m coart.vae \
    --data_root <path_to_dir_with_data_subdir_and_stats> \
    --stats_path <stats_global.npz> \
    --run_tag <str> \
    [--output_dir <path>] \
    \
    --io_arch {three_branch,monolithic} \
    [--warmstart_io | --no_warmstart_io] \
    \
    --lr 1e-5 \
    --lr_unfreeze_warmup_steps 500 \
    --freeze_backbone_steps 2000 \
    --grad_clip_max 1.0 --grad_clip_pct 95 \
    [--use_bf16] \
    \
    --batch_size 1 --num_workers 2 \
    --max_voxels 500000 \
    [--bucket_sampler] --bucket_sort_mode {shuffle,ascending} \
    --val_split_mod 200 \
    \
    --lambda_kl 1e-6 --lambda_subdiv 0.1 \
    \
    --max_steps 200000 \
    --i_log 100 --i_save 5000 --i_sample 5000 --i_val 5000 \
    [--sample_at_step_one] \
    \
    [--use_ema] --ema_rate 0.9999 \
    --rolling_ckpts 3 \
    --resume_from {none,latest,<path>}
```

---

## 12. Launch Prerequisites（分阶段 checklist）

### Phase 1 — 一次性准备（order-sensitive）
1. 跑 `python scripts/preprocess-by-rank/compute_stats.py --out_dir /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512`（几分钟）
2. 确认 pretrained ckpt 本地可达：`pretrained/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f.../ckpts/shape_{enc,dec}_next_dc_f16c32_fp16.safetensors`
3. 决定训练节点（**pending**：user 尚未指定；不能用 119 节点做 production）

### Phase 2 — 代码实现（本次 scope）
4. 实现 `coart/` 13 个 .py 文件（~500 行 new + ~300 行 copied from 原 script）
5. Smoke test: 单 GPU 跑 10 步 `--io_arch three_branch --warmstart_io`，确认 loss 下降、无 NaN、step-1 mesh dump 合理

### Phase 3 — Production run
6. 8-GPU 8卡跑 `run_tag=three_branch_ws_v0`，max_steps=200000
7. 跑 Ablation B/C/D（各 20-50k 步先对比曲线，必要时继续）
8. Val MSE 收敛后选 best EMA ckpt 导出给下游

---

## 13. Risk / Residual Open Items

1. **训练节点未定**。119 仅 profiling，其他节点待 user 指定。**阻塞 Phase 3**，不阻塞代码实现。
2. **`rolling-3` 历史 ckpt × EMA 双份 × 蛇形 step** 磁盘占用估算：单 ckpt（encoder+decoder）~数 GB，200k 步下总占用可能 ~50 GB。`results/` 所在磁盘需有对应空间。
3. **BucketedDistributedSampler + resume**：恢复时 `sampler.set_epoch(saved_epoch)` 需与保存时一致；rolling ckpt 保留 3 份时不存在 race。需在 `load_for_resume` 里显式 load sampler state。
4. **Multi-point cube 占比仅 2.21%**：p2_branch 梯度信号稀疏，可能学习慢。Block-decomposed log 会让这个可见。如果 `loss_recon/p2` 长期不下降，future work 考虑 focal-style re-weighting。
5. **rotation / flip augmentation 未做**：小数据集（41k）样本多样性靠 translation 单通道增强，可能有欠拟合风险。Val 曲线劣化则加入 rotation（需实现 edge/face 6-ch 方向 remap）。
6. **`dit/` 还是空的**：未来扩展时 `coart.common` 和 `coart.data` 需 review 是否足够抽象（特别是 `feat18_dataset` 是否适合 latent data loader）。

---

## 14. Out of Scope (explicitly)

- Modifying `trellis2/trainers/vae/shape_vae.py`（原生 trainer）
- Modifying `trellis2/models/sc_vaes/*`（原生模型）
- Rewriting `precompute_feat18.py`
- Extending to multi-node training（1 node first；multi-node 是 future work）
- Texture VAE / PBR VAE（scope 外，虽然架构上已留位）
- DiT / flow matching finetune（scope 外，仅 placeholder）
