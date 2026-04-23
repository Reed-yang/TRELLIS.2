# coart/ 包文档

> **coart** 是 TRELLIS.2 中围绕 **corep**（CoReP, Combinatorial Representation）数据做 finetune 训练的子包。首期 task 是基于 corep 18-channel 特征的 Shape-VAE finetune (`coart.vae`)，后续会扩展到 DiT / flow-matching（`coart.dit` 预留）。本包是对原 `train_finetune_feat18.py` 单文件脚本的**模块化重构 + 能力扩展**。

---

## 1. 设计定位

### 1.1 为什么需要 coart

原 `train_finetune_feat18.py`（~1000 行）集 IO stems、Dataset、训练 loop、配置、checkpoint 逻辑于一身，存在：

- **架构耦合**：Dataset / Sampler / 模型构建 / loss / 训练 loop / ckpt / CLI 全部单文件，难以复用
- **功能缺失**：无 EMA、无 resume、单文件 ckpt 覆盖写（preempt → 损坏）、lr 1e-4 过高
- **warm-start bug**：`load_pretrained_into` 的 0.5× 缩放 + 跨列复制，在 97.79% 单点 cube 上实际 degrade 起点（详见 `logs/findings_feat18_vae_launch.md`）

`coart/` 的目标：
- **模块化**：按「训练任务」作为顶层切面（task-first），共享 infra 下沉
- **补齐 production 必备能力**：EMA / 原子 save / rolling-K / resume / 对齐原生 lr
- **修复 warm-start 数学**：改为 independent 三分支 + 全强度复制
- **为后续扩展留位**：`coart.dit`、未来可能的 `coart.tex_vae` 等

### 1.2 与其他模块的分工

| 模块 | 角色 | coart 与它的关系 |
|---|---|---|
| `trellis2/` | 官方 TRELLIS.2 model + trainer 框架 | coart **只 import 不修改**：encoder/decoder 模型、grad_clip 工具、sparse 模块 |
| `corep_fast/` | corep 编码/解码管线（CoReP→mesh） | coart 通过 `feature_to_mesh` / `feats_to_param` 消费其产物 |
| `o_voxel/` | O-Voxel 底层 mesh 转换原语 | 被 trellis2 transitive 使用，coart 不直接 import |
| `precompute_feat18.py` | 离线 precompute feat18 数据 | coart 消费其产出的 `.npz` shards 和 `stats_global.npz` |
| `train_finetune_feat18.py` | 原单文件 baseline | **保留原位不动**，coart 是它的重构升级版 |

---

## 2. 架构设计

### 2.1 Task-first 布局（核心决策）

顶层按「训练任务」切分，不按「代码类型」（没有 `coart/models/`, `coart/trainers/` 这种），共享 infra 下沉到 `coart/common/` + `coart/data/`。

```
coart/
├── common/      # 所有 task 共享的训练 infra
├── data/        # 共享数据管道
├── vae/         # Shape-VAE feat18 finetune task（本期实现）
├── dit/         # 未来 DiT finetune 占位
└── tests/       # 单元测试
```

**为什么 task-first 而不是 type-first**：
- VAE 和 DiT 的 IO / loss / loop 结构差异大，type 切面下每个 task 要跨 4-5 个顶层目录新增，迭代摩擦高
- TRELLIS pipeline 里 VAE → DiT 的接口是**离线 latent 文件**，不存在 live-module-level 耦合
- Research finetune 节奏下，单 task 自包含优于跨模块正交

### 2.2 训练两种 mode（自动分派）

`coart.vae.train` 内部区分两种训练模式，由 `--resume_from` + 是否有已存 ckpt 自动决定：

- **`mode=base`**：从 pretrained TRELLIS.2 权重加载 + 可选 warm-start IO + 执行 `freeze_backbone_steps` 冻结预热。用于**首次训练**。
- **`mode=resume`**：从已有 coart ckpt 加载 encoder/decoder/optimizer/EMA/RNG，**精确恢复**训练时的 trainable 状态（含 unfrozen/step_at_unfreeze）继续训练。用于 **preempt / OOM 后续训**。

两个 mode 代码路径完全分离，不再混在一起，避免了 optimizer param group mismatch 类的 resume bug。

### 2.3 IO 架构（feat18 VAE task 核心改动）

原 TRELLIS.2 shape VAE 的 encoder input_layer / decoder output_layer 是单层 `SparseLinear`（6→C₀ / C_end→7）。corep 的 18-channel 特征（`[point1(3), point2(3), edge_weights(6), face_weights(6)]`）语义上是三块：

| 通道 | 语义 | 数据分布 |
|---|---|---|
| 0:3 point1 xyz | 低-z 代表点（local cube 坐标） | 均匀 [0,1]，97.79% cube 有效 |
| 3:6 point2 xyz | 高-z 代表点，zero 若单点 cube | 2.21% cube 非零 |
| 6:18 edge/face weights | 光线-三角形交点数（ordinal count） | 整数 0~22，99.1% ∈ {0,1} |

本包提供**两种 IO 架构**，通过 `--io_arch` flag 切换：

**`io_arch=three_branch`（推荐默认）** — `coart/vae/io_stems.py`
```python
# Encoder:
h = p1_branch(f[:, 0:3]) + p2_branch(f[:, 3:6]) + ef_branch(f[:, 6:18])
# Decoder:
out = concat[p1_head(f), p2_head(f), ef_head(f)]
```
- p1 / p2 分支**独立不共享**，保留 z-sort 信号 → 不需要 permutation-invariant MSE
- warm-start 时 p1_branch 和 p2_branch **都**从 pretrained vertex 列全强度复制，但 p2 的 bias=0（区分语义角色）

**`io_arch=monolithic`（ablation 用）** — 保留原生单层结构，warm-start 修正为只对 point1 列全强度复制（其余 xavier_uniform）。

完整 warm-start 矩阵（`--warmstart_io` × `--io_arch`）：

| `--io_arch` | `--warmstart_io` | p1 init | p2 init | ef init |
|---|---|---|---|---|
| three_branch | on（默认） | W=`W_pre[:,0:3]`，b=`b_pre` | W=`W_pre[:,0:3]`，**b=0** | xavier / 0 |
| three_branch | off | xavier / 0 | xavier / 0 | xavier / 0 |
| monolithic | on | 仅 cols 0:3 覆盖 | 同（共享 layer） | 同 |
| monolithic | off | xavier / 0 | — | — |

### 2.4 核心训练配置（production 默认）

| 项 | 值 | 说明 |
|---|---|---|
| Optimizer | AdamW, wd=0 | 对齐 TRELLIS.2 原生 ft config |
| lr | **1e-5** | 对齐原生 ft；原 script 的 1e-4 过高，会 unfreeze 瞬间冲坏 backbone |
| LR scheduler | None（常数 lr） | 对齐原生 |
| **LR unfreeze warmup** | 500 步 linear 0→lr（在 unfreeze 后） | 额外保险，backbone 不被梯度冲 |
| **Grad clip** | `AdaptiveGradClipper(p95, max=1.0)` | 复用 `trellis2/utils/grad_clip_utils.py` |
| AMP | bf16 autocast（默认开） | losses 保留 fp32，forward/backward 走 bf16 |
| `freeze_backbone_steps` | 2000 | 前 2k 步只训 IO+KL stems，再解冻 backbone |
| `lambda_kl` / `lambda_subdiv` | 1e-6 / 0.1 | 对齐原生 |
| **Render loss** | **无**（按约束设计） | 去掉 mask/depth/normal/ssim/lpips，仅 MSE+KL+Subdiv |
| EMA | rate=0.9999，每步更新 | **production 必需**：下游 DiT 加载 EMA 权重 |
| Ckpt | atomic save + rolling-K=3 + resume | 防 preempt 中途损坏 |
| Max steps | 200000（production 起点） | smoke 用 10；观察 val 曲线决定是否延长 |

---

## 3. 模块划分

总计 13 个 .py 源文件（不含 `__init__.py`） + 5 个测试文件。

### 3.1 `coart/common/` — task-agnostic 共享 infra

| 文件 | 行数 | 内容 |
|---|---|---|
| `ema.py` | 66 | `EMAModel` 类：fp32 shadow 参数追踪（update / copy_to / state_dict / load_state_dict） |
| `checkpoint.py` | 82 | `atomic_save`（tmp+os.replace）、`save_ckpt`（rolling-K 按 prefix 独立）、`find_latest_ckpt` |
| `dist_utils.py` | 54 | `init_dist` / `unwrap` / `wrap_ddp`（bucket=128MB, find_unused=False）/ `worker_init_fn` |
| `flex_gemm_patch.py` | 51 | `_patch_flex_gemm_frozen_weight_bug` 修 frozen-weight backward 崩溃 |
| `logging.py` | 63 | `CoartTBLogger`：按步 buffer + 按 cadence 批量 all-reduce 写 TB |

### 3.2 `coart/data/` — 共享数据管道

| 文件 | 行数 | 内容 |
|---|---|---|
| `feat18_dataset.py` | 193 | `Feat18Dataset`（支持 `val_split_mod` 按 sha hash 切 train/val）+ `collate_fn` |
| `samplers.py` | 90 | `BucketedDistributedSampler`（按 voxel 数分桶减少 straggler） |
| `stats.py` | 54 | `load_stats` / `normalize` / `denormalize`（per-channel），缺 stats 文件 fallback 到 identity |

### 3.3 `coart/vae/` — Shape-VAE feat18 finetune task

| 文件 | 行数 | 内容 |
|---|---|---|
| `config.py` | 171 | `@dataclass VaeTrainConfig` + argparse（38 CLI 参数），`--run_tag` regex 校验 |
| `io_stems.py` | 73 | `Feat18EncIO`（三分支）、`Feat18DecIO`（三头） |
| `build.py` | 195 | `build_models(io_arch=...)` + `load_pretrained_into(io_arch, warmstart_io)` + 两个 `_apply_warmstart_*` helper |
| `loss.py` | 78 | `compute_vae_loss` 返回 7 key dict（total / recon_p1 / recon_p2 / recon_ef / recon / kl / subdiv） |
| `sampling.py` | 69 | `dump_samples`：eval 时把 decoder 输出 denorm 后走 `feature_to_mesh` 导 .ply |
| `train.py` | 435 | 主训练 loop：base/resume mode 分派 + freeze 调度 + EMA + ckpt + val + mesh dump |
| `__main__.py` | 12 | CLI entry：`python -m coart.vae` / `torchrun ... -m coart.vae` |

### 3.4 `coart/dit/` — 占位

仅含 `__init__.py`（5 行 TODO 注释）。未来新增 DiT finetune 时按 `coart/vae/` 同样布局填充。

### 3.5 `coart/tests/` — 单元测试

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `test_ema.py` | 5 | 初始 shadow、update 数学（drift = (1-decay)·delta）、state_dict roundtrip、copy_to、decay 边界 |
| `test_checkpoint.py` | 7 | atomic save 完整性、无 tmp 泄漏、rolling-K 淘汰、find_latest 边界、multi-prefix 共存 |
| `test_io_stems.py` | 5 | 输出 shape、独立分支、forward 分解等式、decoder 三头独立 |
| `test_build_warmstart.py` | 5 | p1 全强度复制、p2 weight=p1/bias=0、ef 不触碰、decoder 对称 |
| `test_loss.py` | 4 | dict keys、recon 等式、KL 零点、subdiv 空列表 |

全部通过（`pytest coart/tests/`，当前 26/26 green）。

### 3.6 依赖方向（单向，无循环）

```
coart.vae.__main__
  └─ coart.vae.config
  └─ coart.vae.train
       ├─ coart.data.{feat18_dataset, samplers, stats}
       ├─ coart.vae.{build, loss, sampling}
       │    └─ coart.vae.io_stems
       ├─ coart.common.{ema, checkpoint, dist_utils, flex_gemm_patch, logging}
       ├─ trellis2.modules.sparse（read-only）
       ├─ trellis2.models.sc_vaes.sparse_unet_vae（read-only）
       ├─ trellis2.utils.grad_clip_utils（read-only）
       └─ train_overfit_feat18.feature_to_mesh（read-only，原 script）
```

---

## 4. 使用指南

### 4.1 前置准备（一次性）

```bash
# 1. Python 环境：确认 setuptools < 80（triton 兼容）
.venv/bin/pip install "setuptools<80"

# 2. 预训练权重（本地已有）
ls pretrained/models--microsoft--TRELLIS.2-4B/snapshots/*/ckpts/shape_{enc,dec}_next_dc_f16c32_fp16.safetensors

# 3. 预计算 feat18 数据（一次性，耗时数小时-数天取决于数据量）
python precompute_feat18.py ...  # 产出 datasets/.../feat18_512/data/*.npz

# 4. 计算 stats_global.npz（~3 min）
python scripts/preprocess-by-rank/compute_stats.py \
    --out_dir /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512
```

Triton JIT cache 默认写 `<repo_root>/.cache/triton/`（NFS 共享路径，任一节点 warm 后其他节点秒用）。通过 `coart/__init__.py` 里 `os.environ.setdefault("TRITON_CACHE_DIR", ...)` 实现，用户可以 `export TRITON_CACHE_DIR=...` override。

### 4.2 单 GPU 调试 / smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m coart.vae \
    --run_tag smoke_v0 \
    --max_steps 10 --batch_size 1 \
    --max_voxels 100000 --no_bucket_sampler \
    --freeze_backbone_steps 5 \
    --i_log 1 --i_save 10 --i_sample 10 --i_val 10 --sample_at_step_one \
    --use_ema --rolling_ckpts 2
```

`-u` 强制 unbuffered stdout，让 tqdm 实时输出（SSH pipe 环境下不加这个会整段 buffer）。

### 4.3 8-GPU Production 启动（默认 run）

```bash
torchrun --standalone --nproc_per_node=8 -m coart.vae \
    --run_tag three_branch_ws_v0 \
    --stats_path /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz \
    \
    --io_arch three_branch --warmstart_io \
    \
    --lr 1e-5 --lr_unfreeze_warmup_steps 500 \
    --freeze_backbone_steps 2000 \
    --grad_clip_max 1.0 --grad_clip_pct 95 \
    --use_bf16 \
    \
    --batch_size 1 --num_workers 2 \
    --max_voxels 500000 --bucket_sampler \
    --val_split_mod 200 \
    \
    --lambda_kl 1e-6 --lambda_subdiv 0.1 \
    \
    --max_steps 200000 \
    --i_log 100 --i_save 5000 --i_sample 5000 --i_val 5000 \
    --sample_at_step_one \
    \
    --use_ema --ema_rate 0.9999 \
    --rolling_ckpts 3 \
    --resume_from latest
```

输出目录自动构造为 `results/coart_feat18_{YYYYMMDD}_{run_tag}/`。如需显式指定（如跨日 resume），传 `--output_dir <path>` 覆盖。

### 4.4 Ablation Matrix（核心实验）

同 seed / 同 data / 同 lr / 同 steps，只改两个 flag：

| Run | `--io_arch` | `--warmstart_io` | 推荐 `--run_tag` | 目的 |
|---|---|---|---|---|
| **A（默认）** | three_branch | on | `three_branch_ws_v0` | 架构 + warm-start 一起 |
| B | three_branch | off | `three_branch_scratch` | 测架构单独贡献 |
| C | monolithic | on | `mono_partial_ws` | monolithic + partial warm-start |
| D | monolithic | off | `mono_scratch_ctrl` | pure scratch baseline |

跑完后 4 条 `loss_recon/p1`、`loss_recon/p2`、`loss_recon/ef` 曲线对比，区分 "是架构贡献" 还是 "warm-start 贡献"。

### 4.5 Resume 启动

preempt / OOM 后重启时，**相同命令 + `--resume_from latest`**：

```bash
torchrun --standalone --nproc_per_node=8 -m coart.vae \
    --run_tag three_branch_ws_v0 \
    --output_dir results/coart_feat18_20260423_three_branch_ws_v0 \
    --resume_from latest \
    [其余参数同上]
```

Log 会打印：
```
[mode] resume (from step 12345)
[resume] trainable state @ ckpt: unfrozen=True, step_at_unfreeze=2000
[resume] loading .../ckpt_step0012345.pt (step 12345)
[resume] ok — step=12345, epoch=3, unfrozen=True
```

Resume 恢复以下状态：
- encoder / decoder 权重
- optimizer（Adam momentum state）
- EMA shadow 权重（分别为 enc/dec 两份）
- sampler epoch（保持 per-epoch shuffle 顺序）
- numpy / torch RNG state
- step counter
- unfrozen / step_at_unfreeze（trainable 集 + LR warmup 进度）

### 4.6 目录碰撞处理

| 场景 | 行为 |
|---|---|
| `--output_dir` 不存在 | 创建，fresh 训练 |
| `--output_dir` 存在且空 | OK，fresh 训练 |
| `--output_dir` 存在且有 ckpt + `--resume_from latest` | 进 resume 流程 |
| `--output_dir` 存在且有 ckpt + `--resume_from none` | **报错** `"dir already populated; use --resume_from latest or change --run_tag"` |

### 4.7 调试技巧

```bash
# tqdm 实时输出（SSH + pipe 场景下必加）
PYTHONUNBUFFERED=1 python -u -m coart.vae ...

# 指定 triton cache 位置（覆盖默认 NFS 路径）
export TRITON_CACHE_DIR=~/.triton_local_cache

# 单 GPU 跑（不开 DDP），需关掉 bucket_sampler
CUDA_VISIBLE_DEVICES=0 python -m coart.vae \
    --no_bucket_sampler \
    [其余参数...]

# 丢掉 val split（全量 train，无 held-out）
--val_split_mod 0
```

---

## 5. 下游接口（给 DiT / 其他 task 用）

### 5.1 加载 coart 产出的 encoder/decoder（online 权重）

```python
import torch
from coart.vae.build import build_models

encoder, decoder = build_models(io_arch="three_branch")
ck = torch.load("results/coart_feat18_..._v0/ckpt_step0200000.pt",
                map_location="cuda", weights_only=False)
encoder.load_state_dict(ck["encoder"])
decoder.load_state_dict(ck["decoder"])
```

### 5.2 加载 EMA 权重（production 下游推荐）

```python
from coart.common.ema import EMAModel

# 用 fresh model 实例化 EMA（shadow 覆盖会从 ckpt 读）
ema = EMAModel(encoder, decay=0.9999)
ema.load_state_dict(torch.load(
    "results/coart_feat18_..._v0/ema_0.9999_enc_step0200000.pt",
    map_location="cpu", weights_only=False,
))
ema.copy_to(encoder)  # 把 EMA shadow 写回 encoder，后续推理用 EMA 版本
```

**推理时务必使用 EMA 权重**：原生 TRELLIS.2 pipeline 默认加载 EMA 版本，非 EMA 的 raw online 权重在 minimum 附近振荡，下游生成质量不稳定。

### 5.3 Feat18 → latent 导出（DiT 训练数据）

（TODO — DiT finetune 设计时确定接口）

---

## 6. 扩展：新增 training task

新增 task（如 `coart.dit` 或 `coart.tex_vae`）的 recipe：

```bash
# 1. 建目录
mkdir coart/<new_task>

# 2. 按 coart/vae/ 布局填充以下文件：
#    - __init__.py
#    - config.py        (@dataclass NewTaskConfig + argparse)
#    - build.py         (build_models + load_pretrained_into)
#    - io_stems.py      (task-specific IO if needed)
#    - loss.py          (compute_<task>_loss)
#    - sampling.py      (eval-time artifacts)
#    - train.py         (train(cfg))
#    - __main__.py      (CLI entry)

# 3. 共享 infra 直接 import，不复制：
#    from coart.common.ema import EMAModel
#    from coart.common.checkpoint import save_ckpt, find_latest_ckpt
#    from coart.data.feat18_dataset import Feat18Dataset, collate_fn  # 若仍用 feat18
#    ...

# 4. 启动
torchrun ... -m coart.<new_task> --run_tag xxx ...
```

如果新 task 需要新的 dataset 格式，在 `coart/data/` 新增一个模块（如 `coart/data/latent_dataset.py` 给 DiT）；保持按 **用途** 而非 **task** 切分 `data/`，最大化共享面。

`common/` 基本不应 per-task 新增 —— 若确实需要，要考虑是否真共享（如果只有一个 task 用，放到 task 目录内）。

---

## 7. Known Issues & Caveats

1. **Triton JIT cold start 开销大**：首次跑一个新硬件/新 shape 组合会触发 ~20-40 min 的编译 warmup（分散在前 100-300 步）。第二次跑命中 `TRITON_CACHE_DIR` 缓存秒启动。Production 跑预计前 20 min 偏慢，之后稳定。

2. **`setuptools < 80` 约束**：triton 的 AMD backend `driver.py` import setuptools 时依赖已被 setuptools 80+ 删除的 `_distutils_hack`。shared venv 里已降级到 79.0.1。升级 triton 到新版可能解除此约束（未验证）。

3. **Augmentation 只有 integer translation**：rotation / 镜像翻转需要对 edge/face 6 通道做 direction remap（corep 的 unique-cube 约定是方向敏感的），未实现。小数据集（41k 样本）可能有欠拟合风险，观察 val 曲线决定是否补。

4. **Aesthetic filter 未接入**：当前全量 41k 样本喂进训练（`metadata.csv` 里有 `aesthetic_score` 可用）。若 val 曲线劣化可加 `min_aesthetic_score >= 4.5` 过滤（对齐原生 config）。

5. **Multi-point cube 梯度稀疏**：`recon_p2` 通路只对 2.21% cube 有真实梯度信号，可能收敛慢。Block-decomposed log 让这个可见，若长期不下降可考虑 focal-style 加权。

6. **单节点 only**：当前 DDP 走 `torchrun --standalone`，未测试多节点。多节点需要配 `MASTER_ADDR` / `MASTER_PORT` 等 torchrun 参数。

7. **Val pass 是 rank-0 sequential**：不走 DDP，最多 16 个样本，用于 loss 曲线追踪而非精确评估。大规模评估另起 offline 脚本。

8. **Grad accumulation 默认关闭**：`batch_size=1 per GPU × 8 GPU = global batch 8`。原生 ft-512 是 global 16（`batch_size_per_gpu=4, batch_split=2`）。若 loss 信号噪声大，后续可加 `batch_split` 参数启用 grad accumulation。

---

## 8. 相关文档

- 设计决策详情：`docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md`
- 实现 plan：`docs/superpowers/plans/2026-04-22-coart-vae-feat18.md`
- 原脚本对比 findings：`logs/findings_feat18_vae_launch.md`
- 组织决策（task-first rationale）：`logs/decisions_coart_org.md`

---

## 9. 维护提示

- **修 bug 时**：优先在 `coart/` 内修，保持 `trellis2/` 和原 `train_finetune_feat18.py` 不动
- **加参数时**：只改 `coart/vae/config.py`（dataclass + argparse），`train.py` 从 `cfg.*` 读
- **加单元测试时**：放 `coart/tests/test_*.py`，遵循 TDD 风格（先写 failing test，再实现）
- **commit 风格**：conventional commits（`feat(coart): ...` / `fix(coart.vae.loss): ...` / `refactor(coart.vae.train): ...`），commit message 包含 "why + what + test evidence"
