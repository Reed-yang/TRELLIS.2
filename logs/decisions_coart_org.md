# `coart/` 子包组织决策

**Date**: 2026-04-22
**Scope**: 在 TRELLIS.2 顶层新增 `coart/` 子包，承载 corep-相关的 finetune 训练（本次是 VAE，后续会扩展到 DiT / texture / pbr 等）。

---

## Layout philosophy: **task-first**

按「训练任务」作为顶层切面（`vae/`, `dit/`, ...），不按「代码类型」（models/, trainers/...）。共享 infra 下沉到 `common/` 和 `data/`。

**为什么不用 type-first（trellis2/ 风格）**：
- VAE 和 DiT 的 IO / loss / loop 差异大，type 切面下每个 task 要跨 4-5 个顶层目录新增，迭代摩擦高
- TRELLIS pipeline 里 VAE → DiT 的接口是**离线 latent 文件**，不存在 live-module-level 耦合
- Research finetune 迭代节奏下，task 自包含优于 type 正交

---

## Final layout

```
coart/
├── __init__.py
│
├── common/                     # 所有 task 共享的训练 infrastructure
│   ├── __init__.py
│   ├── ema.py                  # EMAModel class
│   ├── checkpoint.py           # atomic_save, rolling-K, load_for_resume
│   ├── dist_utils.py           # _init_dist, _wrap_ddp, _unwrap, _worker_init_fn
│   ├── flex_gemm_patch.py      # SubMConv3dFunction frozen-weight backward patch
│   └── logging.py              # TB writer + cadence-controlled all-reduce
│
├── data/                       # 共享数据管道（VAE / DiT 均可用）
│   ├── __init__.py
│   ├── feat18_dataset.py       # Feat18Dataset + collate_fn
│   ├── samplers.py             # BucketedDistributedSampler
│   └── stats.py                # load_stats + normalize/denormalize
│
├── vae/                        # 本次：feat18 Shape-VAE finetune
│   ├── __init__.py
│   ├── config.py               # @dataclass VaeTrainConfig + argparse
│   ├── io_stems.py             # Feat18EncIO, Feat18DecIO (3-branch + monolithic)
│   ├── build.py                # build_models(io_arch), load_pretrained_into(io_arch, warmstart_io)
│   ├── loss.py                 # compute_vae_loss (block-decomposed p1/p2/ef + kl + subdiv)
│   ├── sampling.py             # dump_samples + feature_to_mesh wrapper
│   ├── train.py                # train(config) 主 loop
│   └── __main__.py             # `torchrun ... -m coart.vae`
│
└── dit/                        # 未来 DiT finetune placeholder
    └── __init__.py             # TODO: reserved for future DiT/flow-matching finetune
```

13 个 .py 文件（不含 `__init__.py`） + 1 placeholder 目录（`dit/`）。

---

## 依赖方向（单向，无循环 import）

```
coart.vae.__main__
  └─ coart.vae.config
  └─ coart.vae.train
       ├─ coart.data.feat18_dataset
       ├─ coart.data.samplers
       ├─ coart.data.stats
       ├─ coart.vae.build
       │    └─ coart.vae.io_stems
       ├─ coart.vae.loss
       ├─ coart.vae.sampling
       ├─ coart.common.ema
       ├─ coart.common.checkpoint
       ├─ coart.common.dist_utils
       ├─ coart.common.flex_gemm_patch
       └─ coart.common.logging
```

---

## 启动方式

```bash
# 本次 VAE finetune
torchrun --standalone --nproc_per_node=8 -m coart.vae [args...]

# 未来 DiT finetune
torchrun --standalone --nproc_per_node=8 -m coart.dit [args...]
```

---

## 原 `train_finetune_feat18.py` 归宿

保留原位（不动、不删），作为 baseline 存档。后续所有逻辑迁到 `coart/` 新写，不做 in-place 重构 —— 避免隐性行为漂移，也避免破坏已有 findings/logs 文件里对原脚本的引用。

---

## 未来扩展示意

加一个新 task 只需新建 `coart/<task>/` 子目录 + 5-7 个 .py 文件（config / build / loss / train / __main__ 等），共享 infra 完全复用。`common/` 和 `data/` 零修改。

例如后续可能的扩展：
- `coart/dit_img2shape/` — image-conditioned shape DiT finetune
- `coart/dit_shape2tex/` — shape-conditioned texture DiT finetune
- `coart/tex_vae/` — texture VAE finetune
- `coart/pbr_vae/` — PBR VAE finetune
