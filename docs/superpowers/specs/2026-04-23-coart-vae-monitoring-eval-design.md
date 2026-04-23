# coart.vae 训练监控与评估套件设计

**Date:** 2026-04-23
**Scope:** `coart/` sub-package（不改 `trellis2/`、不改 `scripts/eval/` 既有代码）
**Prereq:** `docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md`（框架已落地，smoke + resume 通过）
**约束优先级:** 质量 > 时间 > ablation 吞吐

---

## 0. 背景与范围

`coart/` 框架（spec 2026-04-22）已完成 17 tasks、22 commits，smoke 跑通。本 spec 关注**真正进入 8-GPU production run 之前**的最后一层补全：训练效率微调、wandb 监控、val deep-eval、与 O-Voxel Layer V baseline 的对标。

**不在本 spec 范围**：
- 模型架构、loss 函数、io_arch、warm-start（已由 2026-04-22 spec 定稿）
- `trellis2/` 内部改动（禁止）
- DiT finetune 阶段监控（placeholder 保留在 `coart/dit/`）
- ablation A/B/C/D 并行调度（因本 run 聚焦质量，ablation 后续再开专题）

**已知现状（截至 2026-04-23）**：
- batch_size=1, num_workers=2, max_voxels=500000, lr=1e-5, max_steps=200000
- loss `= recon + 1e-6·kl + 0.1·subdiv`（**无 render loss**）
- DDP `bucket_cap_mb=128, find_unused_parameters=False`
- logger `CoartTBLogger`（rank-0 TB + all-reduce 平均），wandb **未接入**
- ckpt 实测大小：online+optim 9.3GB/份，EMA(enc+dec) 3.2GB/份；rolling=3 总 37.5GB
- 评估可复用代码：`scripts/eval/eval_metrics.py`（CD/NC/F-score）、`scripts/eval/ovoxel_repr_test.py::_compute_topo_metrics`（组件/边界/Euler/水密）、`train_overfit_feat18.py::feature_to_mesh`（decoder output → trimesh）
- full_ranked.csv 168232 行，按 `pred_enc_s` easy→hard 排序，schema `sha256,local_path,rank,tier,...`
- O-Voxel Layer V 基线（res=512，`results/baseline_experiments/EXP5_full_baseline/geometric_metrics.csv`）：helmet **CD=1.29e-5 NC=0.7731 F@0.005=0.8635**（worst）；spacesuit NC=0.9655；bowl/parallel_planes/icosphere/nested_spheres NC ≥ 0.9995。

---

## 1. 训练效率与 DDP（保守微调）

### 1.1 不动的量

- `batch_size=1`（DDP 8 ranks → effective bs=8）
- `max_voxels=500000`（保留最难 5-10% asset 的原始监督）
- `lr=1e-5`, `freeze_backbone_steps=2000`, `lr_unfreeze_warmup_steps=500`
- AdaptiveGradClipper p95
- loss 权重 `lambda_kl=1e-6`, `lambda_subdiv=0.1`

### 1.2 修改点

**A. DDP wrap 选项**（`coart/common/dist_utils.py::wrap_ddp`）：
```python
DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    bucket_cap_mb=128,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,   # 新增：bucket zero-copy
    broadcast_buffers=False,         # 新增：模型无 BN buffer
)
```

**B. `static_graph=True` 条件启用**（`coart/vae/train.py`）：
- `freeze_backbone` 阶段（step < 2000）`static_graph=False`（梯度图不稳定）
- unfreeze 点 `model._set_static_graph()` 启用
- 为避免 rebuild DDP wrapper，实际做法：**freeze 阶段不用 static graph；unfreeze 时如果观测到 DDP 报 graph mismatch 则退化**

**C. fused AdamW**（`coart/vae/train.py`）：
```python
optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, fused=True)
```
H100 支持；如果 torch 版本报错回退 `fused=False` 且 stderr warn。

### 1.3 预期收益

- 单步 warm time：~0.5-2s → ~0.4-1.8s（省 100-200ms/步）
- 200k 步全程：~30-60 min
- 零风险：任何一项出问题都能独立禁用

### 1.4 度量

在 §2 的 `throughput/` 字段里记录 `step_s, samples_s_per_gpu, voxels_s_per_gpu, avg_batch_voxels`（CUDA event 计时），production run 启动 i_log=100 就能直接读出 warm step/s。

---

## 2. 监控：wandb 主 + TB 备份

### 2.1 接入方式

**config 新增**（`coart/vae/config.py`）：
```python
use_wandb: bool = True
wandb_project: str = "coart-vae"
wandb_mode: str = "online"  # "online" / "offline" / "disabled"
```

**`CoartTBLogger` 扩展**（`coart/common/logging.py`）：
- rank-0 `__init__` 里，如 `use_wandb=True`：
  ```python
  wandb.init(
      project=cfg.wandb_project,
      name=cfg.run_tag,
      config=asdict(cfg),
      tags=[cfg.io_arch,
            "warmstart_io" if cfg.warmstart_io else "scratch",
            f"res{cfg.resolution}"],
      mode=cfg.wandb_mode,
      dir=cfg.output_dir,
  )
  ```
- `flush_if_due`：all-reduce 后 rank-0 既调 `SummaryWriter.add_scalar` 也调 `wandb.log({tag: v}, step=step)`
- 新增 `image(tag, np_array, step)` → `wandb.Image` + 可选 TB `add_image`
- 新增 `object3d(tag, vertices_np, step)` → `wandb.Object3D`
- `close()` 里 `wandb.finish()`
- 任何 wandb 调用 `try/except` 包住，failure 降级为纯 TB，**不 crash 训练**

**依赖安装**：`.venv/bin/pip install wandb`，`WANDB_API_KEY` 由用户设 shell 环境。

### 2.2 Scalar 字段树

| 组 | 字段 | cadence | 产生源 |
|---|---|---|---|
| `train/loss/` | total, recon, recon_p1, recon_p2, recon_ef, kl, subdiv | i_log=100 | `compute_vae_loss()` 返回的 7 key dict |
| `train/sched/` | lr, unfrozen(0/1), steps_since_unfreeze | i_log=100 | optimizer + train.py 状态 |
| `train/grad/` | norm_pre_clip, norm_post_clip, clip_ratio, p95_rolling | i_log=100 | AdaptiveGradClipper 暴露 |
| `throughput/` | step_s, samples_s_per_gpu, voxels_s_per_gpu, avg_batch_voxels | i_log=100 | CUDA event + batch 统计 |
| `val/` | recon_mse, recon_p1, recon_p2, recon_ef | i_val=5000 | 全 226 val asset MSE（已有） |
| `deep_eval/online/mean/` | cd, nc, f005, f01, f05, euler_gap, components_gap, watertight_rate | i_save=5000 | §3 deep-eval |
| `deep_eval/online/per_asset/<name>/` | cd, nc, f005, f01, f05, n_components, euler, n_boundary_edges | i_save=5000 | §3 deep-eval |

EMA 版字段（§3 决定不做 online eval，故 `deep_eval/ema/*` 留 placeholder 不填）。

### 2.3 图像与 3D

`i_save=5000` 时，rank-0 对 `n_dump=2` 个 asset（默认 `helmet` + `val_p95`）：
- `deep_eval/renders/<asset>/online`：4-view normal map 拼图（GT 左、recon 右），2×4 网格 PNG → `wandb.Image`
- （可选）`deep_eval/meshes/<asset>/online`：decode mesh 采样 8192 vertex → `wandb.Object3D`。默认关闭，`cfg.log_3d=False`

### 2.4 Artifact 策略

**ckpt 默认不上传 wandb**。rolling ckpt 在本地 NFS，跨 run 共享由路径本身解决。如需 pinning 某个 ckpt 到 wandb artifact，后续手动 `wandb.save(path)`，不入自动流程。

### 2.5 离线 fallback

节点若无外网，设 `cfg.wandb_mode="offline"`。训练结束后：
```bash
wandb sync results/coart_feat18_20260423_<tag>/wandb/offline-run-*/
```

---

## 3. Deep-eval on 8 golden assets

### 3.1 Golden asset 清单

静态写入 `coart/eval/golden_assets.json`：

| # | name | 来源 | 复杂度 |
|---|---|---|---|
| 1 | `helmet` | `datasets/sketchfab_hard/helmet.glb` | 极高（EXP-5 Layer V NC=0.77） |
| 2 | `triple_sphere` | synthetic（与 EXP-5 `nested_spheres` 同构；由 `scripts/coart_build_golden.py` 生成） | 内部/嵌套 |
| 3 | `val_p10_<sha>` | val split ∩ full_ranked 第 10 百分位 | 低 |
| 4 | `val_p25_<sha>` | val split ∩ full_ranked 第 25 百分位 | 低-中 |
| 5 | `val_p40_<sha>` | val split ∩ full_ranked 第 40 百分位 | 中 |
| 6 | `val_p60_<sha>` | val split ∩ full_ranked 第 60 百分位 | 中-高 |
| 7 | `val_p80_<sha>` | val split ∩ full_ranked 第 80 百分位 | 高 |
| 8 | `val_p95_<sha>` | val split ∩ full_ranked 第 95 百分位 | 极高（对照 helmet） |

"val split ∩ full_ranked" = 同时满足 `_is_val_sha(sha, 200)` 且 sha 在 `datasets/ObjaverseXL_sketchfab/feat18_512/data/` 目录下的 NPZ 文件集合（约 226 个），按 `full_ranked.csv::rank` 升序排列后取百分位。

### 3.2 一次性 setup

脚本 `scripts/coart_build_golden.py`，对每个 golden asset 产出 `datasets/coart_golden/<asset_name>.npz`：
- `cube_indices (N,3) int32, feats (N,18) float32, num_boundary int32` —— feat18 格式，对 helmet / triple_sphere 调用 `precompute_feat18.py` 内部逻辑生成；对 6 个 val asset 直接从 `datasets/ObjaverseXL_sketchfab/feat18_512/data/<sha>.npz` 复制
- `gt_points (100000,3) float32, gt_normals (100000,3) float32` —— 从 GT mesh `sample_points_and_normals()` 采 10 万点缓存
- `gt_topo: {n_components, euler, n_boundary_edges, is_watertight, area}` —— 缓存 GT 拓扑
- metadata: `asset_name, sha, rank, tier, local_path_gt`

GT mesh 来源：
- `helmet`: `datasets/sketchfab_hard/helmet/helmet.glb`（或实际文件位置，setup 时确认）
- `triple_sphere`: synthetic，由 setup 脚本即时生成（`trimesh.creation.icosphere` × 3 不同半径 concentric）
- 6 val asset: `full_ranked.csv` 的 `local_path` + 数据根 `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/`

### 3.3 Layer V baseline 获取

脚本 `scripts/coart_build_baseline.py`，对 8 个 asset 跑 O-Voxel + 原版 SC-VAE roundtrip（复用 `scripts/eval/baseline_exp5_metrics.py::compute_all_metrics`），结果写 `coart/eval/golden_baseline.json`：
```json
{
  "helmet": {"layer_v": {"cd": 1.29e-5, "nc": 0.7731, "f_0.005": 0.8635, "f_0.001": 0.0557, "n_components": 208348, "euler": 4523, "n_boundary_edges": 146102, "is_watertight": false}},
  "triple_sphere": {"layer_v": {"cd": ..., "nc": ..., ...}},
  "val_p10_<sha>": {"layer_v": {...}}
}
```

注：helmet 值来自 EXP-5 CSV `results/baseline_experiments/EXP5_full_baseline/geometric_metrics.csv` 的 `helmet,512,V` 行。`f_0.05` 不在原 CSV schema 内，build_baseline 脚本需统一重新采样 100k 点并用 `f_score_multi([0.005, 0.01, 0.05])` 一次出全部 3 个阈值，保证 baseline 与 deep-eval 的 metric 口径一致。

helmet / triple_sphere 的 Layer V 直接从 EXP-5 CSV 读取已有字段（hardcoded）+ 新采样补 `f_0.05`；6 val asset 从 mesh 开始新跑一遍 Layer V pipeline。

### 3.4 Deep-eval pipeline

`coart/eval/deep_eval.py::run_deep_eval(encoder, decoder, stats, step, logger, cfg)`：

```python
# rank-0 only; other ranks dist.barrier() 等
if rank != 0:
    dist.barrier()
    return

for asset_name, meta in golden.items():
    try:
        d = np.load(f"datasets/coart_golden/{asset_name}.npz")
        cube_indices = torch.from_numpy(d["cube_indices"]).cuda()
        feats_raw = torch.from_numpy(d["feats"]).cuda()
        feats_n = normalize(feats_raw, stats)

        x = SparseTensor(feats=feats_n, coords=cube_indices_with_batch_dim)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, mu, logvar = unwrap(encoder)(x)
            pred = unwrap(decoder)(z, cube_indices_with_batch_dim)
        feats_pred = denormalize(pred.feats.float(), stats).cpu().numpy()
        cube_np = cube_indices.cpu().numpy()

        mesh = feature_to_mesh(feats_pred, cube_np, cfg.resolution)
        if mesh is None:
            logger.scalar(f"deep_eval/online/per_asset/{asset_name}/status_failed", 1.0, step)
            continue

        pts, nrms = sample_points_and_normals(mesh, n=100000)
        cd = chamfer_distance(pts, d["gt_points"])
        nc = normal_consistency(pts, nrms, d["gt_points"], d["gt_normals"])
        fs = f_score_multi(pts, d["gt_points"], thresholds=[0.005, 0.01, 0.05])
        topo = _compute_topo_metrics(mesh)

        per_asset_log = {
            "cd": cd, "nc": nc,
            "f005": fs[0.005], "f01": fs[0.01], "f05": fs[0.05],
            "n_components": topo["n_components"],
            "euler": topo["euler_number"],
            "n_boundary_edges": topo["n_boundary_edges"],
            "is_watertight": float(topo["is_watertight"]),
        }
        for k, v in per_asset_log.items():
            logger.scalar(f"deep_eval/online/per_asset/{asset_name}/{k}", v, step)

        if asset_name in cfg.n_dump_names:  # e.g. ["helmet", "val_p95_..."]
            # metadata stored `local_path_gt` at setup; lazy-load GT trimesh for render
            normal_img = render_normal_4view(mesh, meta["local_path_gt"])
            logger.image(f"deep_eval/renders/{asset_name}/online", normal_img, step)

    except Exception as e:
        logger.scalar(f"deep_eval/online/per_asset/{asset_name}/status_failed", 1.0, step)
        print(f"[deep_eval] {asset_name} failed: {e}", file=sys.stderr)

# mean aggregation over successful assets
# logger.scalar("deep_eval/online/mean/cd", ...)

dist.barrier()
```

时间预算：8 asset × (encode 3-8s + decode 3-8s + feature_to_mesh 5-15s + metrics 1-3s + topo 0.1s) ≈ 100-180s。相对 i_save 间隔 ~2.8h 开销 2-4%。

### 3.5 EMA 策略（瘦身）

`config.py` 加 `rolling_ckpts_ema: int = 1`（默认只保留最新 EMA ckpt）：

- ① EMA shadow update：保留（每步 ~20ms，总 ~67 min）
- ② EMA ckpt save：保留，但 `save_ckpt(prefix="ema_...", keep=cfg.rolling_ckpts_ema)` —— 从 9.6GB 降到 3.2GB 稳态
- ③ EMA deep-eval：**不做**。run 结束后用户手动跑 `python -m coart.eval.eval_ckpt --ckpt ema_0.9999_*.pt` 单独评估

### 3.6 失败处理

单个 asset decode/metric 失败不 crash 训练：
- `try/except` 包住每个 asset
- `deep_eval/online/per_asset/<name>/status_failed = 1` 在 wandb 可视
- `deep_eval/online/mean/*` 计算时跳过 failed asset
- `failed` 连续 3 次 → 写 stderr warning 但训练继续

---

## 4. Baseline 对标与告警

### 4.1 Baseline 参考线（仅展示，不做门槛）

每张 `deep_eval/online/per_asset/<name>/<metric>` chart 叠加两条水平虚线：
- `baseline.layer_r`（O-Voxel round-trip，最差下界）
- `baseline.layer_v`（原 SC-VAE + O-Voxel，目标线）

通过 `wandb.config.baseline = golden_baseline.json` 注入；前端由 wandb reference lines 渲染。

目标层级（仅作为解读参考，**不触发任何自动行为**）：

| 层 | 量化 | 意义 |
|---|---|---|
| Floor | mean NC ≥ 0.90 @ step 20k | 训练没坏 |
| Parity | mean NC ≥ Layer V mean @ step 50-100k | 与原 SC-VAE 打平 |
| Improvement | helmet NC ≥ 0.90, components_gap ≤ 5× Layer V @ step 100-200k | CoReP 表征价值兑现 |

### 4.2 Watchdog（告警不 kill）

三条软告警，写 stderr + `wandb.alert()`，训练继续：

| 条件 | 告警 |
|---|---|
| `grad_norm_pre_clip > 100×p95_baseline` 连续 50 步 | `wandb.alert(title="grad spike", text=f"@step={s} pre_clip={p:.2e} vs p95={b:.2e}", level=WARN)` |
| `train/loss/recon_ef` step 5k-10k 窗口单调上升 | `wandb.alert(title="ef diverge", text=f"@step={s} Δ=+{d:.4f}", level=WARN)` |
| `deep_eval/helmet/online/nc < 0.50 and step >= 10k` | `wandb.alert(title="helmet bad", text=f"@step={s} nc={v:.3f}", level=WARN)` |

触发时额外 dump `results/<run>/watchdog_<cond>_step<N>.json`，包含 trigger 前后 200 步滑窗 loss/grad/lr + 当前 helmet normal map PNG 路径，方便事后离线 debug。

### 4.3 Summary table（wandb 刷新）

每次 i_save 后刷新：

| asset | step | online NC | ΔNC vs Layer V | online CD | ΔCD% | online F@0.005 | Δ |
|---|---|---|---|---|---|---|---|

通过 `wandb.Table` + `wandb.log({"summary": tbl})`。最新一行自动浮到顶。

---

## 5. 文件清单

### 5.1 新增

- `coart/eval/__init__.py`（init package）
- `coart/eval/golden_assets.json`（8 asset 静态清单，由 build_golden 脚本一次性填充）
- `coart/eval/golden_baseline.json`（8 asset × Layer V metric，由 build_baseline 脚本填充）
- `coart/eval/metrics.py`（从 `scripts/eval/eval_metrics.py` + `scripts/eval/ovoxel_repr_test.py` 转导 CD/NC/F-score/topo 函数；实现方式：`sys.path.insert(0, repo/scripts/eval)` 或直接 import）
- `coart/eval/deep_eval.py`（`run_deep_eval()` 入口）
- `coart/eval/watchdog.py`（3 条 watchdog 逻辑 + dump 函数）
- `scripts/coart_build_golden.py`（setup：生成 `datasets/coart_golden/*.npz`）
- `scripts/coart_build_baseline.py`（setup：填 `golden_baseline.json`）
- `coart/tests/test_deep_eval_io.py`（mock encoder/decoder，验证 deep-eval 不 crash 的最小路径）
- `coart/tests/test_watchdog.py`（3 条告警条件的单元测试）

### 5.2 修改

- `coart/vae/config.py`：新增 `use_wandb, wandb_project, wandb_mode, rolling_ckpts_ema, log_3d, n_dump_names`
- `coart/common/logging.py`：加 wandb 接入 + `image()` / `object3d()` 方法
- `coart/common/dist_utils.py::wrap_ddp`：加 `gradient_as_bucket_view=True, broadcast_buffers=False`
- `coart/common/checkpoint.py::save_ckpt`：`keep` 参数调用方可按 prefix 指定（已有，只需调用方改）
- `coart/vae/train.py`：
  - optim 加 `fused=True`
  - unfreeze 点尝试 `_set_static_graph()`
  - deep-eval hook：`if step % cfg.i_save == 0: run_deep_eval(...)`
  - watchdog hook：每 i_log 检查一次
  - EMA save 调用传 `keep=cfg.rolling_ckpts_ema`

### 5.3 依赖

- `.venv/bin/pip install wandb`（已有环境升级）
- `WANDB_API_KEY` env var（用户 shell 配置）

---

## 6. 执行前置 checklist

- [ ] `wandb` 安装成功，rank-0 能 `wandb login`
- [ ] `datasets/coart_golden/*.npz` 生成（`scripts/coart_build_golden.py`）
- [ ] `coart/eval/golden_baseline.json` 填完（`scripts/coart_build_baseline.py`，6 val asset 跑 Layer V，~30 min 单 GPU）
- [ ] smoke test 跑 i_save=20 触发一次 deep-eval 验证：期望 8 asset 都出 metric，无 crash
- [ ] wandb dashboard 验证：面板能看到 train/loss、throughput、val、deep_eval/per_asset
- [ ] watchdog 单元测试过
- [ ] production run 在 120 或指定节点启动，`torchrun --nproc_per_node=8`

节点选择：待 user 指定（119 仅 profiling；120/117 视占用）。

---

## 7. 非目标（明确排除）

- 自动节点选择 / 自动 resume：由 launch script 处理，不在本 spec
- 多 run 对比面板：wandb UI 原生支持，不设计 custom
- 评估样本数扩充（从 8 扩到 100+）：可后续加，此 spec 锁 8
- EMA online deep-eval：用户决定 run 结束后手动跑
- render loss 恢复实验：明确排除
- DiT-side 监控：另立 spec

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| wandb 网络波动 crash 训练 | 中 | 高 | 所有 wandb 调用 `try/except`，降级纯 TB |
| `feature_to_mesh` 在 helmet / val_p95 上 OOM | 中 | 中 | per-asset `try/except` + `status_failed` 标记，mean 剔除 |
| `static_graph=True` 与 DDP 图 mismatch | 低 | 中 | unfreeze 后试启用，报错即禁用 |
| 8 asset 复杂度选错导致 baseline 无对比意义 | 低 | 中 | 采用百分位 stratified + helmet 作最难 anchor |
| fused AdamW 在 torch 版本不兼容 | 低 | 低 | `try/except` 回退 `fused=False` |
| EMA shadow 与 online ckpt 一起 rolling=1 误删 | 低 | 低 | 参数拆分 `rolling_ckpts` vs `rolling_ckpts_ema`，默认 3 / 1 |

---

## 9. 成功判据

本 spec 的实施成功 ≠ 训练跑出好结果。仅评判框架层面：

1. wandb dashboard 有完整 loss/throughput/val/deep_eval 曲线
2. smoke run（i_save=20 强制触发）产出 8 asset 的 per-asset metric
3. watchdog 三条告警在单测里可触发
4. `datasets/coart_golden/*.npz` + `golden_baseline.json` 生成完整
5. production run 启动后 1 小时内 rank-0 的 throughput/step_s 稳定在 < 5s/step，且 wandb 能看到 `deep_eval/online/per_asset/*` 在首个 i_save 步有数
