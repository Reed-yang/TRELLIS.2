# TRELLIS.2 训练指南与经验文档

> 基于对代码库的深入研究和实际验证规划，记录关键发现、决策理由和后续可操作事项。

---

## 一、项目概况

TRELLIS.2 是微软开源的 4B 参数 3D 生成模型，核心能力是 **图像 → 3D 资产**（含 PBR 材质）。
其关键创新是 **O-Voxel** 表示——一种无场（field-free）的稀疏体素结构，能处理开放表面、非流形几何和内部封闭结构。

### 当前环境状态（已验证）

| 项目 | 状态 |
|------|------|
| GPU | 8× NVIDIA H100 80GB HBM3 |
| Python | 3.10.12 (venv: `.venv/`) |
| PyTorch | 2.6.0+cu124, CUDA 12.4 |
| flash_attn | 2.7.3 |
| nvdiffrast | 0.4.0 |
| cumesh, flex_gemm, o_voxel | 已安装 |
| transformers, kornia, timm, lpips, tensorboard | 已安装 |
| 预训练模型 | TRELLIS.2-4B, DINOv3, RMBG-2.0 (均在 `pretrained/`) |
| 推理 | 已验证可用 (`example_local.py`) |

**注意**: 没有 conda 环境，使用的是 `.venv/` 虚拟环境，执行训练时需用 `.venv/bin/python`，或先 `source .venv/bin/activate`。

---

## 二、完整训练架构

TRELLIS.2 的训练分为 **4 个阶段**，各阶段可独立训练：

```
Stage 0: SC-VAE Training (两个独立模型，可并行)
│
├── Shape SC-VAE
│   配置: configs/scvae/shape_vae_next_dc_f16c32_fp16.json
│   高分辨率微调: shape_vae_next_dc_f16c32_fp16_ft_512.json
│   输入: mesh dump + dual_grid (O-Voxel 256)
│   输出: Shape encoder + decoder
│   Trainer: ShapeVaeTrainer
│   损失: 重建(vertex/intersected) + 渲染(mask/depth/normal L1+SSIM+LPIPS) + KL散度
│
└── Texture SC-VAE
    配置: configs/scvae/tex_vae_next_dc_f16c32_fp16.json
    高分辨率微调: tex_vae_next_dc_f16c32_fp16_ft_512.json
    输入: pbr dump + pbr_voxels (O-Voxel 256)
    输出: Texture encoder + decoder
    Trainer: PbrVaeTrainer

    ↓ 用训练好的 SC-VAE encoder 编码 latent (encode_shape_latent.py / encode_ss_latent.py)

Stage 1: Sparse Structure Flow Model
    配置: configs/gen/ss_flow_img_dit_1_3B_64_bf16.json
    输入: ss_latent [8, 16, 16, 16] + 条件图像 (512×512)
    输出: 图像 → 稀疏结构生成器
    Trainer: ImageConditionedFlowMatchingCFGTrainer
    条件模型: DinoV3 (facebook/dinov3-vitl16-pretrain-lvd1689m)
    模型参数: 1.3B (1536 channels, 30 blocks, 12 heads)

Stage 2: Shape Flow Model
    配置: configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json
    高分辨率微调: slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json
    输入: shape_latent (SparseTensor) + 条件图像
    输出: 图像 → 形状生成器
    Trainer: ImageConditionedSparseFlowMatchingCFGTrainer

Stage 3: Texture Flow Model
    配置: configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
    高分辨率微调: slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json
    输入: pbr_latent + shape_latent + 条件图像
    输出: 图像+形状 → 纹理生成器
    Trainer: ImageConditionedSparseFlowMatchingCFGTrainer
```

### 阶段间依赖关系

- Stage 0 是独立的，其余阶段都依赖 Stage 0 产出的 encoder
- Stage 1 需要 SS encoder（来自预训练 `microsoft/TRELLIS-image-large`）
- Stage 2 需要 shape latent（由 Stage 0 Shape encoder 编码）
- Stage 3 需要 shape latent + pbr latent（由两个 Stage 0 encoder 编码）
- 所有 Flow Model (Stage 1/2/3) 都需要条件渲染图像

---

## 三、数据准备流水线

### 7 步流水线概览

```
Step 1: 安装依赖        → data_toolkit/setup.sh
Step 2: 初始化 metadata → build_metadata.py {Subset} --root {ROOT}
Step 3: 下载 3D 资产    → download.py {Subset} --root {ROOT}
Step 4: 处理 mesh/PBR   → dump_mesh.py + dump_pbr.py + asset_stats.py
Step 5: 转 O-Voxel      → dual_grid.py + voxelize_pbr.py (CPU)
        ─── 此时可以训练 SC-VAE ───
Step 6: 编码 latent      → encode_shape_latent.py + encode_pbr_latent.py + encode_ss_latent.py (GPU)
Step 7: 渲染条件图像     → render_cond.py (需要 Blender 3.0)
        ─── 此时可以训练 Flow Model ───
```

### 各步骤产出的数据格式

| 步骤 | 产出目录 | 文件格式 | 用途 |
|------|---------|---------|------|
| download | `raw/` | 原始 3D 文件 | 后续处理源 |
| dump_mesh | `mesh_dumps/` | 标准化 mesh | SC-VAE Shape 训练 |
| dump_pbr | `pbr_dumps/` | PBR 纹理 | SC-VAE Texture 训练 |
| dual_grid | `dual_grid_{res}/` | `.vxz` O-Voxel | Shape latent 编码输入 |
| voxelize_pbr | `pbr_voxels_{res}/` | 体素化 PBR | PBR latent 编码输入 |
| encode_shape | `shape_latents/{model}_{res}/` | `.npz` (feats + coords) | Flow Model Stage 2/3 + SS 编码输入 |
| encode_pbr | `pbr_latents/{model}_{res}/` | `.npz` (feats + coords) | Flow Model Stage 3 |
| encode_ss | `ss_latents/{model}_{res}/` | `.npz` (z: [8,16,16,16]) | Flow Model Stage 1 |
| render_cond | `renders_cond/{sha256}/` | PNG + transforms.json | 所有 Flow Model |

### 已知问题

**1. `data_toolkit/build_metadata.py` 缺失 `datasets/` 模块包**。该脚本通过 `importlib.import_module(f'datasets.{sys.argv[1]}')` 加载数据集定义（如 `datasets.ObjaverseXL`），但 `data_toolkit/datasets/` 目录不存在于代码仓库中。这意味着完整的 7 步流水线目前无法直接运行。后续如需走完整流水线，需要：
- 联系作者获取该模块，或
- 根据 `objaverse` 包的 API 逆向实现 `get_metadata()` 和 `add_args()` 函数

**2. DinoV3 模型是 HuggingFace gated repo**。配置中默认的 `facebook/dinov3-vitl16-pretrain-lvd1689m` 需要 HF 登录授权才能下载。解决方案：
- 将模型手动下载到 `pretrained/dinov3/` 目录（已完成）
- 训练配置中将 `image_cond_model.args.model_name` 改为 `pretrained/dinov3`（本地相对路径）
- `DINOv3ViTModel.from_pretrained()` 支持本地路径，无需修改代码
- 参考 `example_local.py` 中的 monkey-patching 方案（推理用）

---

## 四、训练方式与资源需求（实测验证）

### 训练方式：全量参数训练

代码库**只支持全量参数训练（full parameter training）**，不支持 LoRA / PEFT / Adapter。
- 训练日志实测：`Number of trainable parameters: 1292179976`（1.3B 全部可训练）
- 没有参数冻结逻辑（除 DinoV3 条件模型默认 `@torch.no_grad()` 冻结外）
- 如需 LoRA 支持，需自行集成 PEFT 库到 trainer 中

### 并行策略：DDP（非 FSDP）

- 使用 `torch.nn.parallel.DistributedDataParallel`（`trellis2/trainers/basic.py:13`）
- **每张卡持有完整的模型副本**（权重 + 优化器状态 + EMA），DDP 只同步梯度
- **不支持 FSDP**：代码库中无任何 FSDP/FullyShardedDataParallel 引用
- **不支持 DeepSpeed**：无相关集成
- 对于 1.3B 模型 + 80GB 显存的 H100，单卡即可放下完整模型，FSDP 不是必须的

### 显存占用分析（1.3B Flow Model，AMP bf16）

| 组成部分 | 计算方式 | 大小 |
|---------|---------|------|
| 模型权重 (bf16 推理 + fp32 master) | 1.3B × (2+4) bytes | ~7.8 GB |
| AdamW 优化器状态 (fp32 m + v) | 1.3B × 4 × 2 bytes | ~10.3 GB |
| EMA 权重 (fp32) | 1.3B × 4 bytes | ~5.2 GB |
| DinoV3 条件模型 (冻结, ~300M) | ~300M × 4 bytes | ~1.2 GB |
| **静态总计（不含 activation）** | | **~24.5 GB** |
| Activation memory | 取决于 batch_size 和 resolution | ~10-40 GB |

### 各阶段资源需求速查

| 阶段 | 模型规模 | 默认 batch/GPU (split) | 估算显存/卡 | 最低 GPU 配置 |
|------|---------|----------------------|------------|--------------|
| Stage 0: Shape SC-VAE | ~数百M | 8 (split 2) | ~20-30 GB | 1× A100 40GB |
| Stage 0: Texture SC-VAE | ~数百M | 8 (split 2) | ~20-30 GB | 1× A100 40GB |
| Stage 1: SS Flow | 1.3B | 8 (split 4) | ~40-60 GB | 1× A100 80GB |
| Stage 2: Shape Flow 512 | 1.3B + elastic | 8 (split 2) | ~50-70 GB | 1× H100 80GB |
| Stage 2: Shape Flow ft1024 | 1.3B + elastic | 2 (split 1) | ~60-80 GB | 1× H100 80GB |
| Stage 3: Texture Flow 512 | 1.3B + elastic | 8 (split 2) | ~50-70 GB | 1× H100 80GB |
| Stage 3: Texture Flow ft1024 | 1.3B + elastic | 2 (split 1) | ~60-80 GB | 1× H100 80GB |

### 当前环境适配情况（8× H100 80GB）

- **单卡 fine-tune**: 所有阶段都可以在单卡 H100 上完成（Stage 1 已实测验证通过）
- **多卡 DDP 加速**: 8 卡可将训练速度提升约 8×，每卡仍是完整模型副本
- **不需要 FSDP / DeepSpeed**: 1.3B 模型在 80GB 显存中可完整放下（24.5GB 静态 + 充足 activation 空间）
- **Elastic memory**: Stage 2/3 使用 `LinearMemoryController` 动态调整 checkpointing，在显存不足时自动降低 activation 缓存以避免 OOM

### 训练验证结果（2026-03-16 实测）

| 验证项 | 状态 | 详情 |
|--------|------|------|
| 训练启动 | 通过 | 20 样本加载，1.3B 模型初始化，DinoV3 加载成功 |
| Loss 下降 | 通过 | 前 5 步平均 2.36 → 后 5 步平均 1.53，下降 35.2% |
| Checkpoint 保存 | 通过 | step 500 和 1000 各保存 denoiser + EMA + misc |
| 采样可视化 | 通过 | 4 次采样完成（init, step 500, step 1000, final） |
| 训练速度 | — | 7114 steps/h（单卡 H100），1000 步约 8.4 分钟 |

### 如果想降低资源需求（未来改进方向）

| 方案 | 效果 | 实现难度 | 备注 |
|------|------|---------|------|
| 集成 LoRA (PEFT) | 可训练参数降至 ~1-5%，显存降 50%+ | 中等 | 需修改 trainer 集成 `peft` 库 |
| 减小 batch_size + 增大 batch_split | 降低 activation 显存 | 简单 | 修改配置即可，但训练速度下降 |
| 梯度 checkpointing | 用时间换空间 | 简单 | Elastic memory 已部分实现此功能 |
| 冻结部分 transformer blocks | 减少可训练参数和优化器状态 | 简单 | 手动设置 `requires_grad=False` |
| 使用 FSDP | 跨卡分片模型参数 | 较高 | 需要重写 trainer 的并行逻辑 |

---

## 五、决策记录与理由

### 为什么选择 Mini Dataset 直接构造而非走完整流水线？

1. **核心目标是验证训练流程**，不是验证数据准备流程。两者是独立问题。
2. **data_toolkit 有缺失模块**（`datasets/` 包），走完整流水线会被阻塞在 `build_metadata.py`。
3. **render_cond 需要 Blender 3.0**，安装和调试引入额外复杂度。
4. **直接构造数据最快**：Stage 1 训练只需要 3 样东西（metadata.csv + ss_latent + 条件图像），可以在几分钟内生成。

### 为什么选择 Stage 1 (Sparse Structure Flow) 作为验证目标？

1. **数据最简单**：ss_latent 是规则的 dense tensor `[8, 16, 16, 16]`，不涉及 SparseTensor 打包/解包。
2. **最接近生成任务核心**：是推理 pipeline 的第一步生成器。
3. **fine-tune 价值最高**：控制生成的结构拓扑，直接影响最终 3D 质量。
4. **相比 Stage 2/3**：不需要先训练 SC-VAE 来编码 shape/pbr latent，可以用预训练模型的 encoder。

### 为什么验证标准是"1000 步 loss 下降"？

- 纯启动无报错（方案 A）太浅，可能隐藏数据加载、梯度回传等问题。
- 完整采样可视化（方案 C）在 20 个样本的 mini dataset 上意义不大。
- 1000 步观察 loss 趋势是最高效的验证方式，能确认数据加载 → 前向 → 损失计算 → 反向传播 → 参数更新的完整链路。

---

## 六、后续可操作事项

### 短期（训练验证后）

- [ ] **扩大 mini dataset**: 从 20 个样本增加到 100-500 个，观察训练曲线是否更稳定
- [ ] **多卡验证**: 从 `--num_gpus 1` 改为 `--num_gpus 8`，验证 DDP 分布式训练
- [ ] **Checkpoint resume 验证**: 中断训练后用 `--load_dir` 恢复，确认状态一致

### 中期（数据流水线补全）

- [ ] **补全 `data_toolkit/datasets/` 模块**: 实现 `ObjaverseXL.py` 的 `get_metadata()` 和 `add_args()`，使完整 7 步流水线可用
- [ ] **安装 Blender 3.0**: 用于 `render_cond.py` 渲染多视角条件图像
- [ ] **下载 ObjaverseXL 子集**: 用 `--world_size` 控制下载量，先小规模测试
- [ ] **构建真实数据集**: 走完 7 步流水线，生成 1000+ 样本的真实训练集

### 长期（Fine-tune 科研）

- [ ] **Stage 1 Fine-tune**: 在自定义数据上 fine-tune SS Flow Model，控制生成结构
- [ ] **Stage 2 Fine-tune**: fine-tune Shape Flow，改善特定类别的形状生成
- [ ] **Stage 3 Fine-tune**: fine-tune Texture Flow，定制材质风格
- [ ] **高分辨率 fine-tune**: 用 `_ft1024` 配置从 512 分辨率微调到 1024
- [ ] **DinoV3 解冻**: 尝试解冻条件模型参与训练（默认冻结）
- [ ] **SC-VAE 定制**: 如果重建质量不满足需求，fine-tune Shape/Texture SC-VAE

### 配置调整参考

| 想要做什么 | 修改什么 |
|-----------|---------|
| 加载预训练权重 fine-tune | 在配置中加 `"finetune_ckpt"` 字段，或用 `--load_dir` |
| 调整学习率 | `trainer.args.optimizer.args.lr` |
| 修改 batch size | `trainer.args.batch_size_per_gpu` + `batch_split` |
| 切换单卡/多卡 | `--num_gpus N` |
| 多机训练 | `--num_nodes N --node_rank R --master_addr ADDR` |
| 调整 mixed precision | `trainer.args.mix_precision_mode` (amp / inflat_all) |
| 调整 CFG 概率 | `trainer.args.p_uncond` (默认 0.1 = 10% 无条件) |
| 调整采样间隔 | `trainer.args.i_sample` |
| 切换注意力后端 | 环境变量 `ATTN_BACKEND=xformers/flash_attn/sdpa` |

---

## 七、关键文件索引

### 训练入口
- `train.py` — 主训练脚本，支持分布式、checkpoint resume、profiling

### Trainer 实现
- `trellis2/trainers/basic.py` — 基础 Trainer（DDP、混合精度、EMA、梯度裁剪、弹性内存）
- `trellis2/trainers/vae/shape_vae.py` — Shape SC-VAE Trainer
- `trellis2/trainers/vae/pbr_vae.py` — Texture SC-VAE Trainer
- `trellis2/trainers/flow_matching/flow_matching.py` — Flow Matching 基础 Trainer
- `trellis2/trainers/flow_matching/sparse_flow_matching.py` — Sparse Flow Matching
- `trellis2/trainers/flow_matching/mixins/` — CFG、Text/Image 条件 Mixin

### Dataset 实现
- `trellis2/datasets/components.py` — 基础 Dataset + ImageConditionedMixin
- `trellis2/datasets/sparse_structure_latent.py` — SS Latent Dataset (Stage 1)
- `trellis2/datasets/structured_latent_shape.py` — Shape Latent Dataset (Stage 2)
- `trellis2/datasets/structured_latent_svpbr.py` — Texture Latent Dataset (Stage 3)
- `trellis2/datasets/flexi_dual_grid.py` — SC-VAE Shape Dataset (Stage 0)

### 模型定义
- `trellis2/models/sparse_structure_flow.py` — SS Flow Model (Stage 1)
- `trellis2/models/structured_latent_flow.py` — Shape/Texture Flow Model (Stage 2/3)
- `trellis2/models/sc_vaes/fdg_vae.py` — SC-VAE encoder/decoder

### 工具函数
- `trellis2/utils/loss_utils.py` — 损失函数 (L1, SSIM, LPIPS, PSNR)
- `trellis2/utils/data_utils.py` — 数据工具 (ResumableSampler, BalancedResumableSampler)
- `trellis2/utils/elastic_utils.py` — 弹性内存管理
- `trellis2/utils/grad_clip_utils.py` — 自适应梯度裁剪

### 配置文件
- `configs/scvae/` — SC-VAE 训练配置 (4 个)
- `configs/gen/` — Flow Model 训练配置 (5 个)
