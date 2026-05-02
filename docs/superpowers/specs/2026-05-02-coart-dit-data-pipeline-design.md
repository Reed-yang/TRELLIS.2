# coart.dit Data Pipeline (v0, 10K-asset milestone) — Design

**Date**: 2026-05-02
**Author**: siyuan / Claude
**Scope**: 数据 pipeline 端到端 10K-asset 跑通 + manifest 完整。**不包含 DiT trainer 实现/训练**（下一份 spec）。
**Target hardware**: 8–24 GPUs on `host-10-240-99-{117,118,119}`
**Wall-clock budget**: ≤ 10 h for the 10K milestone; design must scale linearly to 50K.

---

## 1. Background — verified facts

### 1.1 Upstream state（已就绪/进行中）

- **`coart.vae` 已 finetune 完毕**（首版 EMA ckpt：`results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_{enc,dec}_step0155000.pt`）。Encoder 输入 18-ch corep 特征 `[p1(3), p2(3), edge(6), face(6)]`；输出 `SparseTensor (N, 32)`。详见 `coart/vae/build.py:49-70`、`coart/vae/io_stems.py:22-47`。
- **Mesh 与 corep feat18 数据基本就绪**：
  - 168,307 assets in `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/`（1.8 TB, sha256+local_path 双索引）
  - 87,414 feat18 npz 在 `feat18_512/data/{sha}.npz`（303 GB, voxelize 仍在进行）
  - top-level `metadata.csv` 含 `sha256, file_identifier, aesthetic_score, captions`
- **`coart.dit/` 目前只有占位 `__init__.py`**（`coart/__init__.py:18-22`、`coart/dit/__init__.py:1-6`）。

### 1.2 原生 TRELLIS-2 训练协议（基线契约，必须对齐）

来自 `configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json`：

| 项 | 值 |
|---|---|
| Dataset class | `ImageConditionedSLatShape`（`trellis2/datasets/structured_latent_shape.py:92`） |
| Cond model | DinoV3 ViT-L/16, `facebook/dinov3-vitl16-pretrain-lvd1689m`, **image_size=512** |
| Cond 加载协议 | `trellis2/datasets/components.py:89-132` `ImageConditionedMixin.get_instance` — 每 step 随机抽 1 view，alpha-bbox crop → resize 512 → α 复合 |
| Latent normalization | `mean[32], std[32]`（写在 dataset args 内，shape latent 训练时归一化） |
| Render 默认 | `data_toolkit/render_cond.py:81` 默认 `--num_cond_views 16`，Blender CYCLES @ 1024 RGBA |
| `min_aesthetic_score` | 4.5 |
| `max_tokens` | 8192 |

> 本 spec 严格对齐这些值，避免与官方 ckpt warm-start 时的分布漂移。

### 1.3 关键发现 — SS-flow 仍可复用，但需坐标系对齐

详细分析见对话记录（2026-05-02 brainstorm）。要点：
- SS-flow + SS-decoder 输出 32³ binary occupancy；与 corep latent 的 32-ch 分布完全解耦。
- corep voxelization (`corep_fast/stages/s1_voxelize.py:20-114`) 与原生 voxelization (`o-voxel/o_voxel/convert/flexible_dual_grid.py`) 算法等价，仅 normalization 差异：
  - 原生：scale=0.99999, mesh in [-0.5, 0.5]³
  - corep：scale=0.947, offset=[0.489, 0.506, 0.513]
- **决策**：SS-flow 全复用；推理时对 32³ coords 做 affine 量化对齐；**但 IoU 验证必须在数据 prep 之前跑一次**，否则训练完后几何漂移会归因错（详见 §6 pre-flight）。

---

## 2. Goals / Non-goals

### Goals
1. 在 ≤ 10 h 内为 10K assets 生产以下 cache：
   - 16-view RGBA 渲染（`renders_cond/{sha}/{v:03d}.png + transforms.json`）
   - 16-view DINOv3 ViT-L/16 features cache（`dino_l16_s512/{sha}.npz`，fp16）
   - 1× shape SLat latent cache（`slat/{vae_tag}/{sha}.npz`，fp16，**版本化**）
2. 输出 single-source-of-truth `manifest.csv`，trainer 直接读。
3. 提供独立的 SS-flow occupancy IoU 验证脚本，作为 pre-flight gate。
4. 设计可线性扩展到 50K，无需重构 schema。
5. 数据 schema 一旦冻结就是下游 trainer 契约，未来不允许 breaking change（slat 版本化通过 path 隔离）。

### Non-goals
1. DiT trainer 实现、训练、eval —— 排到下一份 spec（`docs/superpowers/specs/2026-05-XX-coart-dit-shape-finetune-design.md`）。
2. 修改 `data_toolkit/`、`trellis2/`、`corep_fast/` 任何文件（per user rule）。
3. PBR DiT 数据准备 —— 第一期 hypothesis 是直接复用官方 PBR pipeline，待 shape DiT 跑通后再 verify。
4. 真实照片（非渲染）作为 cond —— 是独立的 distribution-shift 课题。
5. 文本 cond cache —— captions 已在 metadata.csv，未来扩 task 时再加 cache 层。
6. 渲染器替换（EEVEE / NVDIFFRAST）—— samples=64 CYCLES 已能压进预算；改渲染器属于 v1 优化。

---

## 3. Architecture overview

```
                                ┌────────────────────────────────────────┐
                                │  /mnt/novita2/data/video_obj/          │
                                │   ObjaverseXL_sketchfab/               │
                                │   ├── raw/ (1.8 TB mesh, 168K)         │
                                │   ├── metadata.csv (top-level)         │
                                │   └── feat18_512/data/ (87K npz)       │
                                └─────────────┬──────────────────────────┘
                                              │
                pre-flight:                   │
                pick_instances.py + IoU val   │
                                              ▼
                                instances_10k.csv
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
    [stage=render]                  [stage=dino]                    [stage=slat]
    cd data_toolkit && python       coart_cache_dino.py             coart_cache_slat.py
    render_cond.py ObjaverseXL      ├─ load DinoV3 ViT-L/16         ├─ load coart EMA encoder
    --rank R --world_size W         ├─ glob renders_cond/{sha}/*.png ├─ glob feat18_512/{sha}.npz
    (NO modification)               ├─ batch 16 views               ├─ encoder forward → mu
                                    └─ fp16 .npz                    └─ fp16 .npz (versioned by VAE tag)
              │                               │                               │
              ▼                               ▼                               ▼
    renders_cond/{sha}/             dino_l16_s512/{sha}.npz         slat/{vae_tag}/{sha}.npz
      000..015.png + transforms.json
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              ▼
                                  build_manifest.py
                                              ▼
                                       manifest.csv
                                              ▼
                                   [trainer reads — out of scope]
```

**Key design choices**：
- 三 stage 独立 launch，独立 resume，独立 sharding（`--rank/--world_size`，sha hash 模分片）。
- render & dino & slat 可在不同节点同时跑（CPU 重 vs GPU 重错峰）。
- `slat/` 路径包含 `vae_tag`，VAE 升级时新 ckpt 跑出新一份 cache，与旧并存，trainer 可 A/B。
- `manifest.csv` 是 single source of truth，trainer 不直接 glob 输出目录。

---

## 4. Output Layout & Cache Schema

### 4.1 Filesystem layout（**契约，trainer 直接 hardcode 这些路径模板**）

```
/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/
├── instances_10k.csv                # pre-flight 输出，10K sha 列表 + 元数据
├── manifest.csv                     # single source of truth, build_manifest.py 聚合
├── renders_cond/
│   └── {sha}/
│       ├── 000.png ... 015.png      # RGBA 1024×1024
│       └── transforms.json          # 16 frames × {yaw,pitch,radius,fov,transform_matrix}
├── dino_l16_s512/
│   └── {sha}.npz                    # 16 views × ViT-L/16 features
├── slat/
│   └── {vae_tag}/                   # 例：vae_three_branch_ws_v0_ema_s0155000/
│       └── {sha}.npz                # coart EMA encoder mu
├── new_records/                     # render stage per-rank csv shard（来自 data_toolkit 协议）
│   └── part_{rank}.csv
└── logs/
    ├── render_rank{R:02d}.log
    ├── dino_rank{R:02d}.log
    └── slat_{vae_tag}_rank{R:02d}.log
```

### 4.2 Cache 文件 schema（冻结契约）

#### 4.2.1 `renders_cond/{sha}/`（与 `data_toolkit/render_cond.py` 输出一致，零修改）
| File | Format | Content |
|---|---|---|
| `{v:03d}.png` (v=0..15) | RGBA PNG | 1024×1024×4 uint8, Blender CYCLES, alpha = 物体掩膜 |
| `transforms.json` | JSON | `{frames: [{yaw, pitch, radius, fov, file_path, transform_matrix:4x4}]}`. 由 `data_toolkit/blender_script/render_cond.py` 写出 |

#### 4.2.2 `dino_l16_s512/{sha}.npz`

参考 `trellis2/modules/image_feature_extractor.py:81-92` `DinoV3FeatureExtractor.extract_features`：模型直接返回 `hidden_states (B, T, D)`，T 包含 CLS + register + patches 共 1029 tokens for ViT-L/16 @ image_size=512，D=1024。我们 cache 这个原 raw 输出（不拆 CLS / register / patch），保证与 trellis2 trainer online forward 输出 bit-identical。

| Key | dtype | shape | semantics |
|---|---|---|---|
| `features` | **fp16** (np.float16) | (16, T, 1024) | 16 views × T tokens × ViT-L hidden dim. T 由首次 forward 写入 `n_tokens`. 期望 T=1029 |
| `view_idx` | uint8 | (16,) | 0..15 对应 `renders_cond/{sha}/{v:03d}.png` |
| `n_tokens` | int32 (0-d) | — | T，首次 forward 时确定，后续每个 sha 必须一致 |
| `model_id` | str (0-d) | — | `"facebook/dinov3-vitl16-pretrain-lvd1689m"` |
| `image_size` | int32 (0-d) | — | `512` |

存储：每 asset 单文件 npz，`np.savez_compressed` 写 tmp + `os.replace` rename（atomic）。tmp 路径用 `<out_dir>/{sha}.npz.tmp.{pid}.{ts}` 避免多 rank 写同一 sha 的 race。

**Dtype 选择 — fp16 而非 bf16**：bf16 与 numpy 互转需要 `.view(torch.int16).numpy().view(np.uint16)` 之类 bitcast 操作（PyTorch numpy 协议不直接支持 bfloat16），fp16 是 numpy 原生 dtype，npz 保存/加载零摩擦。trainer 端 `torch.from_numpy(arr).to(torch.bfloat16)` 即可转回 bf16 计算。Dynamic range：fp16 最大 ±65504，DINO 输出经 LayerNorm 范围有界（典型 |x| < 50），无溢出风险。

预期单文件大小：16 × 1029 × 1024 × 2 B ≈ **33.7 MB / asset**, 10K total ≈ **340 GB**。

#### 4.2.3 `slat/{vae_tag}/{sha}.npz`
| Key | dtype | shape | semantics |
|---|---|---|---|
| `coords` | int16 | (N, 3) | 32³ cube indices（来自 corep voxelization） |
| `feats` | **fp16** (np.float16) | (N, 32) | encoder mu（**未归一化**，trainer 自己用 dataset args 里的 mean/std 做） |
| `num_voxels` | int32 (0-d) | — | N，方便 manifest 聚合 |
| `vae_ckpt_rel` | str (0-d) | — | `"results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt"` |
| `vae_io_arch` | str (0-d) | — | `"three_branch"`（区分 monolithic ablation） |

设计选择 + 理由：
- **fp16 而非 fp32 / bf16**：fp16 与 fp32 比磁盘减半；与 bf16 比则避免 numpy bitcast。coart latent 受 KL 约束 + dataset normalization 后范围有界，fp16 安全。trainer 读取时 `torch.from_numpy(arr).to(torch.bfloat16)` 转回 bf16 计算。
- **存 mu 不存 sample**：deterministic、可复现；reparameterize 留给 trainer。
- **不存 logvar / KL**：DiT 不消费 posterior 分布。
- **不归一化**：归一化 stats 在 dataset config 里，与 latent 解耦；将来 stats 重算不需要重写 cache。
- **vae_tag 编码**：`vae_<run_tag>_<ema_or_online>_s<step:07d>`。例：`vae_three_branch_ws_v0_ema_s0155000`。

预期单文件大小：avg 5K voxels/asset × (3×2 + 32×2) B ≈ **350 KB / asset**, 10K total ≈ **3.5 GB**（远小于 dino）。

#### 4.2.4 `manifest.csv` 列定义
| Column | dtype | 说明 |
|---|---|---|
| `sha256` | str | primary key |
| `aesthetic_score` | float32 | 来自原 metadata.csv |
| `n_views_rendered` | int8 | 期望 16，缺则 < 16 |
| `render_done` | bool | `n_views_rendered == 16 and transforms.json exists` |
| `dino_done` | bool | `dino_l16_s512/{sha}.npz` 存在且 features.shape[0]==16 |
| `slat_done` | bool | `slat/{vae_tag}/{sha}.npz` 存在 |
| `slat_tag` | str | 当前最新 slat 的 vae_tag（多 vae 版本时取 lex max） |
| `num_voxels` | int32 | slat coords.shape[0] |
| `last_updated` | timestamp | 最后状态更新时刻 |
| `failed_reason` | str | 失败时填，否则空 |

`manifest.csv` 由 `build_manifest.py` 增量构建（不依赖 stage 写）；单进程；< 10 sec for 10K。

#### 4.2.5 `instances_10k.csv` 列定义
继承 `metadata.csv` 全部列 + 增加 `feat18_npz_size_bytes` 列（用于后续 bucketing）。

---

## 5. Stage Scripts & CLI Surface

### 5.1 文件布局

```
scripts/coart_data_v0/
├── README.md                       # quickstart
├── run.sh                          # bash 编排，单一对外入口
├── pick_instances.py               # pre-flight: filter + sample 10K
├── coart_cache_dino.py             # stage=dino
├── coart_cache_slat.py             # stage=slat
└── build_manifest.py               # stage=manifest
```

**新增代码总量**：约 **400-500 LOC**（不含 README）。

### 5.2 `run.sh` API

仿 `scripts/precompute_feat18_objaverse_sketchfab.sh`：

```bash
#!/usr/bin/env bash
# Env vars (with defaults):
#   COART_DATA_ROOT: /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0
#   STAGE: render | dino | slat | manifest
#   INSTANCES: instances_10k.csv (relative to COART_DATA_ROOT)
#   NODES: "117 118 119"
#   NUM_GPUS_PER_NODE: 8
#   VAE_CKPT: (required for STAGE=slat)
#   VAE_TAG:  (required for STAGE=slat)

# usage:
#   STAGE=render NODES="117 118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
#   STAGE=dino   NODES="118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
#   STAGE=slat   VAE_CKPT=... VAE_TAG=... NODES="119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
```

实现要点：
- 计算 `WORLD_SIZE = nodes_count * NUM_GPUS_PER_NODE`，每个节点 ssh 到 `host-10-240-99-${node}`，cd 到项目根，按 SSH 远程执行规则在 `tmp/` 下创建临时 sh 脚本（per `~/.claude/CLAUDE.md` MANDATORY OVERRIDE）。
- 每个 rank 一个 GPU（`CUDA_VISIBLE_DEVICES=$rank`），输出独立 log 文件。
- 不阻塞等待；脚本结尾 `wait` + 退出码聚合。
- 失败 rank 不 kill 其他 rank（`set +e`）；最终打印 per-rank exit code 表。

### 5.3 `pick_instances.py`（pre-flight）

```python
# Usage:
#   python scripts/coart_data_v0/pick_instances.py \
#     --metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
#     --feat18_dir   /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
#     --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
#     --aesthetic_min 4.5 --n 10000 --seed 0 \
#     --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_10k.csv
```

逻辑：
1. Load top-level metadata.csv（sha256, aesthetic_score, captions, file_identifier）。
2. Load raw/metadata.csv（sha256, local_path）— `data_toolkit/render_cond.py` 需要它。
3. Filter `aesthetic_score >= 4.5`（与官方 ckpt 训练 filter 对齐）。
4. Filter `f'{sha}.npz' in os.listdir(feat18_dir)`（必须有 corep feat18）。
5. Filter `local_path` 非空（必须可定位到 mesh 文件）。
6. `df.sample(n, random_state=seed)` → 写出 instances_10k.csv（继承全部列 + `feat18_npz_size_bytes`）。
7. 打印分布统计：aesthetic_score 直方图、feat18 size 分位数、前 10 行 preview。

### 5.4 `coart_cache_dino.py`

CLI:
```
python scripts/coart_data_v0/coart_cache_dino.py \
  --instances <coart_data_root>/instances_10k.csv \
  --renders_dir <coart_data_root>/renders_cond \
  --out_dir    <coart_data_root>/dino_l16_s512 \
  --rank R --world_size W --gpu R%8 \
  [--batch 16] [--limit N]
```

实现 (~150 LOC)：
1. 用 `transformers.AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")` 加载（与官方 config 对齐；trellis2 trainer 会用同一个 model name 在线 forward）。
2. 移到 GPU，eval mode, bf16 autocast。
3. `df = pd.read_csv(instances)`；按 `int(sha[:8], 16) % world_size == rank` 分片（稳定哈希）。
4. Per-asset：glob `renders_cond/{sha}/*.png` → 期望 16 个 → 按 `view_idx` 排序 → preprocess（与 `trellis2/datasets/components.py:113-129` + `trellis2/modules/image_feature_extractor.py:68-70` 严格一致：alpha bbox crop, resize 512 LANCZOS, α 复合 (`image * alpha.unsqueeze(0)`), normalize 用 DINO mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]）。
5. Stack to (16, 3, 512, 512)，single forward → 取 `extract_features` 的 raw `hidden_states`（forward 内部走 bf16 autocast），随后 `.to(torch.float16).cpu().numpy()` 转 fp16，shape = (16, T, 1024) where T 由模型决定（期望 1029）。
6. `np.savez_compressed(tmp_path, features=feats_f16, view_idx=arange(16, dtype=uint8), n_tokens=int32(T), model_id="facebook/dinov3-vitl16-pretrain-lvd1689m", image_size=int32(512))` → atomic rename via `os.replace`。
7. Resume：每个 sha 开始前检查 `out_dir/{sha}.npz` 存在 + `np.load(...)["features"].shape[0] == 16` → 跳过。
8. 进度：tqdm。

预期吞吐：单 GPU ≈ 1.5 sec/asset（16 views forward + I/O），10K / (24 GPU × 0.67 asset/sec) ≈ **10 min**。

### 5.5 `coart_cache_slat.py`

CLI:
```
python scripts/coart_data_v0/coart_cache_slat.py \
  --instances  <coart_data_root>/instances_10k.csv \
  --feat18_dir <dataset_root>/feat18_512/data \
  --vae_ckpt   results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \
  --vae_tag    vae_three_branch_ws_v0_ema_s0155000 \
  --vae_config_json results/coart_feat18_20260423_three_branch_ws_v0/config.json \
  --out_dir    <coart_data_root>/slat \
  --rank R --world_size W --gpu R%8 \
  [--batch 1] [--limit N]
```

实现 (~150 LOC)：
1. Load config.json → 用 `coart.vae.build.build_models(cfg)` 重建 encoder（无 decoder 节省显存）。**API 准确性见 §14 open question #5**。
2. Load EMA enc ckpt → encoder.load_state_dict → cuda + eval + bf16 autocast。
3. Load `stats_global.npz` for normalize（与 trainer 一致）；详见 `coart/data/stats.py:normalize`。
4. Per-asset (按 `int(sha[:8], 16) % world_size == rank` 分片，稳定哈希)：
   - load `feat18_512/data/{sha}.npz` → 拿 `cube_indices, feats, num_boundary`
   - normalize（同 train，调 `coart.data.stats.normalize`）
   - 构 SparseTensor → encoder forward → 取 `mu` (N, 32)
     - **关键**：调用与训练一致的 path（`return_raw=True` if API 支持），且 `sample_posterior=False` 取 mu
   - `np.savez_compressed(tmp, coords=cube_indices.astype(int16), feats=mu.to(torch.float16).cpu().numpy(), num_voxels=int32(N), vae_ckpt_rel=str, vae_io_arch=str)` → `os.replace`
5. Resume：检查 `out_dir/{vae_tag}/{sha}.npz` 存在 → 跳过。

预期吞吐：单 GPU ≈ 0.5 sec/asset（小 mesh）至 2 sec/asset（大 mesh），10K / (8 GPU × 1 asset/sec) ≈ **20 min**。

### 5.6 `build_manifest.py`

```
python scripts/coart_data_v0/build_manifest.py \
  --instances <coart_data_root>/instances_10k.csv \
  --renders_dir <coart_data_root>/renders_cond \
  --dino_dir <coart_data_root>/dino_l16_s512 \
  --slat_root <coart_data_root>/slat \
  --out <coart_data_root>/manifest.csv
```

实现 (~80 LOC)：
1. Load instances_10k.csv。
2. For each sha：
   - Check `renders_cond/{sha}/transforms.json` exists + count `*.png`
   - Check `dino_l16_s512/{sha}.npz` exists
   - Find `slat/*/{sha}.npz` 全部（多 vae 版本），取 lex max 作 `slat_tag`，读 `num_voxels`
3. 写 manifest.csv（atomic）。
4. 打印汇总：% render_done, % dino_done, % slat_done, slat_tag 分布、num_voxels 分位数。

---

## 6. Pre-flight & SS-flow IoU Validation

### 6.1 Pre-flight checklist（T-0，~30 min）

| 步骤 | 命令 / 检查 | 通过条件 |
|---|---|---|
| 1. 磁盘 | `ssh host-10-240-99-117 "df -h /mnt/novita2/data/video_obj"` | ≥ 1.5 TB free |
| 2. VAE ckpt | `ls results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt` | exists |
| 3. DINOv3 ckpt 可拉 | `python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')"` | success |
| 4. Blender 可用 | `ls datasets/blender-3.0.1-linux-x64.tar.xz` 已解压在 `/tmp/blender-3.0.1-linux-x64/blender` | exists |
| 5. `pick_instances.py` | 跑出 instances_10k.csv，10000 行 | success |
| 6. **SS-flow IoU 验证** | 见 §6.2 | res=16 IoU ≥ 0.9 → pass；< 0.9 → 增加 affine 对齐 task；< 0.8 → 升级方案 2（SS-flow finetune），本 spec 标记阻塞 |
| 7. Smoke run | 见 §7 | 100 asset 端到端通过 |

### 6.2 SS-flow occupancy IoU 验证

新脚本 `scripts/coart_compare_occupancy.py`（~150 LOC，独立于数据 pipeline）：

```
python scripts/coart_compare_occupancy.py \
  --golden_dir datasets/coart_golden \
  --resolutions 64,32,16 \
  --out logs/findings_ss_flow_iou.md
```

逻辑：
1. 8 golden assets（`datasets/coart_golden/`），每个 mesh：
   - corep voxelize 在 res=64：`corep_fast.stages.s1_voxelize.voxelize_mesh(...)` → `cubes_corep_64`
   - 原生 voxelize 在 res=64：`o_voxel.convert.mesh_to_flexible_dual_grid(..., resolution=64)` → `cubes_native_64`
2. 各自 max_pool 到 32³ 和 16³（`cube_indices // 2`，去重）。
3. 算每个分辨率的 IoU = `|A ∩ B| / |A ∪ B|`、`|A\B|/|A|`、`|B\A|/|A|`。
4. 写 markdown 报告 `logs/findings_ss_flow_iou.md`，每 asset 一行：
   ```
   | asset      | IoU@64 | IoU@32 | IoU@16 | A\B@16 | B\A@16 |
   ```
5. 终端打印 mean/min/max IoU per resolution + 决策：
   - mean IoU@16 ≥ 0.9 → "PASS：方案 1 直接走，coart.dit 训练数据用 corep-scale GT，推理时仅做仿射量化对齐"
   - mean IoU@16 ∈ [0.8, 0.9] → "PASS WITH WARNING：方案 1 + spec 增加 affine 对齐节"
   - mean IoU@16 < 0.8 → "FAIL：本 spec 阻塞，升级到方案 2（SS-flow finetune）"

预期 wall-clock：res=64 单 mesh ~30 s × 8 assets × 2 voxelizers ≈ 8 min，外加上下采样和报告 < 5 min。**总 ~15 min on 1 GPU**。

---

## 7. Smoke run protocol

在 production launch 之前必跑：

```bash
# 1. pick 100 instances
python scripts/coart_data_v0/pick_instances.py \
  --aesthetic_min 4.5 --n 100 --seed 1 \
  --out <coart_data_root>/instances_smoke100.csv

# 2. render (1 rank, 1 GPU host)
ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  bash scripts/coart_data_v0/run.sh STAGE=render INSTANCES=instances_smoke100.csv NODES='119' NUM_GPUS_PER_NODE=1"

# 3. dino
ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  bash scripts/coart_data_v0/run.sh STAGE=dino  INSTANCES=instances_smoke100.csv NODES='119' NUM_GPUS_PER_NODE=1"

# 4. slat
ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  bash scripts/coart_data_v0/run.sh STAGE=slat  INSTANCES=instances_smoke100.csv \
       VAE_CKPT=results/.../ema_0.9999_enc_step0155000.pt \
       VAE_TAG=vae_three_branch_ws_v0_ema_s0155000 \
       NODES='119' NUM_GPUS_PER_NODE=1"

# 5. manifest
python scripts/coart_data_v0/build_manifest.py \
  --instances <coart_data_root>/instances_smoke100.csv ...

# 6. 端到端 sample test (~30 LOC ad-hoc)
python -c "
import pandas as pd, numpy as np, json
m = pd.read_csv('<coart_data_root>/manifest.csv')
m_done = m[m['render_done'] & m['dino_done'] & m['slat_done']]
print(f'{len(m_done)}/{len(m)} fully cached')
sha = m_done.iloc[0]['sha256']
# load and shape-check
imgs = json.load(open(f'<coart_data_root>/renders_cond/{sha}/transforms.json'))
assert len(imgs['frames']) == 16
dino = np.load(f'<coart_data_root>/dino_l16_s512/{sha}.npz')
assert dino['features'].shape[0] == 16 and dino['features'].shape[2] == 1024  # T determined by model
slat = np.load(f'<coart_data_root>/slat/vae_three_branch_ws_v0_ema_s0155000/{sha}.npz')
assert slat['coords'].shape[1] == 3 and slat['feats'].shape[1] == 32
print('shape contract OK')
"
```

通过条件：
- ≥ 95/100 assets 三 stage 全成功
- shape contract assert 全通过
- 单 stage 端到端时间记录，外推 production 估计是否在 10 h 内

---

## 8. Production runbook (T+0 → T+10h)

时序（基于 24 GPU on 117/118/119）：

| T | Stage | Hosts | Ranks | 预期耗时 | 说明 |
|---|---|---|---|---|---|
| 0:00 | pre-flight + smoke | 119 | 1 | 30-45 min | §6 + §7 |
| 0:45 | **render launch** | 117, 118, 119 | 24 | **5-7 h** | 主 bottleneck（CYCLES samples=64） |
| 4:00 | **dino launch**（render 60% 后开始） | 118, 119 | 16 | ~10 min（per asset 1.5s × 10K / 16） | 可与 render 错峰，dino 处理已完成的 sha |
| 5:00 | **slat launch** | 119 | 8 | ~20 min | 与 render 后段并行 |
| 7:30 | manifest 中检 | local | 1 | 1 min | partial manifest，估算缺失率 |
| ≤ 10:00 | render 完成 | — | — | — | 期望 ≥ 95% sha 成功 |
| 10:00 | manifest 终检 + validation gates | local | 1 | 5 min | §9 |

**渲染 wall-clock 估算**：
- 单 view CYCLES samples=64 @ 1024 ≈ 5-8 s
- 16 views/asset ≈ 80-128 s/asset
- 24 ranks × 3600 s × 7 h = 604,800 rank-sec / asset 100 s avg = ≈ 6048 assets/7h on 24 ranks
- ⚠️ 紧：10K assets 需要 ≥ 95% 成功率 + 一些失败 retry buffer
- **风险预案**：若 4 h 进度 < 30%，立即切 samples=32 重启（spec §10 risk #1）

---

## 9. Validation gates（must all pass to declare done）

| Gate | 命令 / 检查 | 通过阈值 |
|---|---|---|
| G1: 完整率 | `cat manifest.csv | grep -c ',True,True,True,'`（render+dino+slat 三全） | ≥ 9000 / 10000 |
| G2: DINO 16 views | `python -c "import numpy as np; assert np.load('<dino_dir>/<sha>.npz')['features'].shape[0]==16"` × 5 random sha | 5/5 通过 |
| G3: SLat 与 feat18 一致 | 5 random sha：`assert slat.coords.shape[0] == feat18.cube_indices.shape[0]` | 5/5 通过 |
| G4: 端到端读取 | §7 step 6 sample test | shape contract OK |
| G5: SS-flow IoU | `cat logs/findings_ss_flow_iou.md` 中 mean IoU@16 | ≥ 0.9（或文档化的 fallback path） |
| G6: 磁盘 | `df -h` after run | 仍有 ≥ 200 GB free for slat re-runs |

未达 G1：分析 failed_reason 分布，决定是否需要 second pass（针对 mesh 加载 fail 的 sha 单独 retry）。

---

## 10. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| R1: 渲染 wall-clock 超 10 h（CYCLES 慢） | Medium | High | 4h 中检若 < 30% → 切 samples=32 重启；spec §8 已预案。备选：用 EEVEE（需小改 `data_toolkit/blender_script/render_cond.py` 的 engine 参数，**违反 no-modify rule**，需独立讨论） |
| R2: DINOv3 ViT-L/16 ckpt 拉取失败（HF 网络） | Low | High | 预下载到 `pretrained/dinov3-vitl16-pretrain-lvd1689m/`，pass `local_files_only=True` |
| R3: Blender 子进程 segfault on 复杂 mesh | Medium | Medium | `data_toolkit/render_cond.py` 已有 try/except 跳过；`new_records/part_{rank}.csv` 不会包含失败 sha，G1 阈值 90% 已留 buffer |
| R4: SS-flow IoU < 0.8 | Low (基于 corep/o_voxel 算法等价分析) | Critical | spec 阻塞，转入 SS-flow finetune 子项目；本 spec 重新评估 scope |
| R5: 磁盘满 mid-run | Low | High | pre-flight G6 的 1.5 TB free 已留 30% 余量；render PNG 写完后立即可清部分 raw（如 `merged_records/`）— 但**不要清 raw mesh** |
| R6: Sha 哈希分片导致负载不均（大 mesh 集中在某 rank） | Medium | Medium | 用 `sha256` 第一字节 hash 模 world_size 而非 sha 字符串 hash（更均匀）；仍然不完美但不阻塞 |
| R7: Atomic rename 在 NFS 上失败 | Low | Medium | 用 same-fs tmp 路径（`out_dir/.tmp/{sha}.npz` 而非 /tmp/）；`os.replace` 在 ext4/NFS 都是 atomic |
| R8: VAE encoder 重训后 slat cache 全部失效 | Always (设计如此) | Low | 这正是 vae_tag 版本化的目的；新 vae_tag 跑一份新 cache 不影响旧 cache |
| R9: `coart.vae.build.build_models` API 在后续 vae iteration 中变化 | Medium | Medium | `coart_cache_slat.py` 严格走当前 API；任何 breaking change → 同步更新此脚本 |
| R10: trainer 端 dataset class 期望 image cond 是 PNG path 而非 fp16 features | High（这就是 B1 的 trade-off） | Medium | 下一份 spec 明确 trainer dataset class 用 cached features (`torch.from_numpy(arr).to(torch.bfloat16)` 转回 bf16) 而非 online DINO；本 spec 不实现 trainer，仅 lock cache 格式 |

---

## 11. Scope notes — DiT trainer defaults (locked, not implemented)

下一份 spec（`docs/superpowers/specs/2026-05-XX-coart-dit-shape-finetune-design.md`）将 implement，本 spec 在此 lock 默认值便于下游起草：

| 决策点 | Locked default | Source |
|---|---|---|
| Warm-start | 从 `pretrained/models--microsoft--TRELLIS.2-4B/.../ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` 直接 load | 用户确认 |
| Freeze schedule | None（DiT 输入维度未变，无需 freeze stage） | 用户确认 |
| Optimizer | AdamW, betas=(0.9, 0.95), wd=0.01 | 与官方 config 一致 |
| LR | **2e-5**（官方 1e-4 的 1/5），1-2K step linear warmup | finetune 比 pretrain 保守 |
| Max steps | 200K（占位，根据 val 曲线决定延长） | 用户后续确认 |
| EMA | rate=0.9999 | 与官方一致 |
| AMP | bf16 | 与官方一致 |
| Batch | per-GPU 8, batch_split 2 | 与官方一致；24 GPU 时 effective batch=192 |
| `p_uncond` | 0.1 | 与官方一致 |
| Image cond | **从 dino_l16_s512 cache 读 fp16 features**（每 step 随机抽 1 view），用 `torch.from_numpy(arr).to(torch.bfloat16)` 转回 bf16，跳过 online DINO forward | B1 的实现，下游 spec 设计新 dataset class |
| Latent target | 从 `slat/{vae_tag}/{sha}.npz` 加载 (coords, feats)，按 dataset normalization 归一化 | 与 §4.2.3 schema 一致 |

---

## 12. Scaling notes (10K → 50K)

本 spec 设计天然线性扩容：

1. `pick_instances.py --n 50000` 出 `instances_50k.csv`
2. `STAGE=render` 在 50K 上重跑（resume 跳已有 sha，仅渲染新增 40K）
3. `STAGE=dino` 同
4. `STAGE=slat` 同
5. `build_manifest.py --instances instances_50k.csv` 出 manifest
6. 不需要重写任何 schema 或 trainer 代码

预期 50K wall-clock：
- 渲染 5×10K = ~25-35 h on 24 ranks（必然要切 EEVEE 或加节点）
- DINO ~50 min
- SLat ~100 min

> 50K scale 时 Blender 渲染会成为唯一瓶颈，到时再单独讨论是否需要 patch `data_toolkit/blender_script/render_cond.py` 加 EEVEE 选项。

---

## 13. Test plan

新增 unit tests under `coart/tests/` 或 `scripts/coart_data_v0/tests/`（位置待 writing-plans 决定）：

| Test | 描述 | 运行时间 |
|---|---|---|
| `test_pick_instances.py` | 用 hand-crafted 100-row metadata + temp feat18 dir，assert 输出行数、列、aesthetic filter、sha 唯一 | < 1 s |
| `test_cache_dino_shape.py` | mock DinoV3（return random tensors），feed 1 sha 的 16 PNG，assert 输出 npz shape 与 dtype | < 5 s |
| `test_cache_slat_shape.py` | mock encoder（return random `(z, mu, logvar)`），feed 1 feat18 npz，assert 输出 npz schema、coords dtype int16, feats dtype fp16 | < 5 s |
| `test_build_manifest.py` | 构造 mini fs（5 sha 各种缺失组合），assert manifest 列、bool 准确 | < 1 s |
| `test_atomic_write.py` | 触发模拟中断，assert 部分文件不会被读到 | < 2 s |
| `test_compare_occupancy.py` | 在合成 sphere mesh 上跑 corep + o_voxel voxelize，assert IoU > 0.9 | < 30 s |

CPU-only，no GPU dep（mock model）；总 < 1 min。

---

## 14. Open questions（需要在 writing-plans 之前 verify 或决策）

1. ❓ `slat/{vae_tag}/{sha}.npz` 中是否要 **同时** 存 `mu` 与 `logvar`？**默认决策**：只存 `mu`，需要时另跑 `coart_cache_slat.py --include_logvar`。
2. ❓ DINOv3 features 是否要做 **per-view augmentation**？**默认决策**：不在 cache 层做，augmentation 留给 trainer 在 features space 做（如 token dropout）或不做。
3. ❓ Smoke run 是否包含 SS-flow IoU validation？**默认决策**：独立 pre-flight，不串到 smoke。
4. ❓ Blender CYCLES samples 数：**默认决策**：64（中等质量），4h 中检若 render 进度 < 30% 切 32 重启。
5. ❓ **`coart.vae.build` API 准确性 verify**：spec §5.5 假设 `build_models(cfg)` 可独立构建 encoder（不带 decoder）。writing-plans 第一步要 verify 此 API 是否存在；若不存在则需 `build_encoder_only(cfg)` 或者 `build_models(cfg) → 取 .encoder`。这个 verify 是 trivial（< 5 min），不阻塞 spec。
6. ❓ DinoV3 ckpt local cache：spec §10 R2 假设可预下载到 `pretrained/dinov3-vitl16-pretrain-lvd1689m/`。writing-plans 第一步 verify 已下载 or download 步骤要列入 pre-flight。
7. ❓ `hf://datasets/JeffreyXiang/TRELLIS-500K/...` 风格的 ObjaverseXL.py 中 download 路径——目前 metadata 已用 sketchfab subset 完成，render_cond.py 应当从 `raw/metadata.csv` 读 local_path，不再触发下载。writing-plans 时确认这一点（避免 production launch 时不期而至的 HF download）。

---

## 15. Out of band / not in this spec

- DiT trainer 实现 → 下一份 spec
- PBR 端验证 → 下一份 spec（需要 shape DiT 训练完后端到端 sample 看 PBR 质量）
- 真实照片 cond / 文本 cond → 未来扩展
- 渲染器替换 → v1 优化

---

**Spec 完。** 自审 + 用户 review 之后转入 writing-plans。
