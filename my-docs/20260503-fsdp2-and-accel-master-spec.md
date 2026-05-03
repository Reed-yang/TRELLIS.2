# FSDP2 集成 + 综合加速 Master Spec (2026-05-03)

> 单 spec 三阶段贯穿：Phase A FSDP2 集成 / Phase B FSDP2 下精细 profiling / Phase C 综合加速 candidate 决策树。
> 接替 `20260502-fsdp2-integration-spec.md`（封存为 v0），合并 `20260503-eltwise-deep-analysis.md` 的发现。
> 实施按 5 wave 推进；每 wave 独立 verify gate；C7 数值改动单独三层 gate。

---

## 1. Goal & Non-goal

### Goal

将 coart DiT 1.3B shape 训练在保持 **数值等价 / ULP-close** 的前提下：

- step time 1.26 s → ≈ 1.10 s（bs=8）；
- 通过 FSDP2 mem 释放推 bs/gpu 8 → 12，effective sample/s 50.8 → ≥ 75（+47%）；
- ETA 2.74 d → ≈ 1.8 d；
- 同时完成 `coart/dit/` 的 monkey-patch 反模式清理，立 `modeling/` + `parallel/` 框架。

### Non-goal

- 不实施 FSDP2 zero3（仅留 design appendix；future scale 模型时再启）；
- 不动 `trellis2/` 任何文件（user 强制规则）；
- 不实施 fp8 / TransformerEngine（超 ROI/effort 比）；
- 不引入数值 dtype 改动（D1/D2/D3 默认 defer，需另启 brainstorm）。

### 总驱动

主：comm-compute overlap + bs↑ 释放 mem 头空间。
副：foundation for future scale（≥ 2.5B model）。

---

## 2. 关键设计决策（brainstorm 已 freeze）

| 决策 | 选项 | 理由 |
|---|---|---|
| FSDP2 模式范围 | ZRO-1 + zero2（zero3 仅 design appendix）| mem 释放够用，zero3 step penalty +5-15% 不值；future 可再启 |
| `mp_policy.reduce_dtype` | **fp32** | bit-equivalent 与 DDP；bf16 reduce 引入 grad 噪声需另议 |
| `mp_policy.param_dtype` | bfloat16 | 与 autocast bf16 一致 |
| `reshard_after_forward` | **False (zero2)** | zero2 选项 |
| Distributed EMA | **shard EMA on each rank**（per-rank 持 local DTensor shard） | mem 1.3 GB/rank vs rank-0-only 5.2 GB |
| Activation checkpoint | **保留 elastic** (`SparseTransformerElasticMixin.with_mem_ratio`) | 已验证；FSDP2 下仅调 `max_mem_ratio_start` 0.5 → 0.5 (zero2) / 0.3 (zero3 if used) |
| C7 fused RMSNorm 实施时机 | W5 最后（FSDP2 后）| FSDP2 hide 不了 RMSNorm（critical compute path），但放最后是为 attribution 干净 |
| `fused_rope_patch.py` | **删除** | 实测 -0.6%（轻微 regression），不值保留 |
| `eltwise_patch.py` / `compile_patch.py` | **删除** | 失败 / deprecated |
| `fused_modulation_patch.py` | **重写为 `modeling/block.py` 内嵌**，不再 monkey-patch | 反模式清理，内容 100% 保留并 default-on |

---

## 3. Code organization principle

### 3.1 反模式审计

`coart/dit/` 当前有 4 个 `*_patch.py`，2 useful（fused_modulation, fused_rope）2 死代码（eltwise, compile）。其中 fused_modulation 是 hot-path 必启的 **monkey-patch trellis2 私有方法 `_forward`**，违反"trellis2 仅作参考"原则。

### 3.2 Target 结构

```
coart/dit/
├── __init__.py               # 仅注册 dataset/trainer 进 trellis2.{datasets,trainers}
├── config.py                 # path/ckpt 常量
├── dataset.py                # CachedImageConditionedSLatShape (extends trellis2)
├── trainer.py                # CachedImage...Trainer (extends trellis2 + dispatch parallel mode)
├── _spawn_helpers.py         # mp.spawn helper（保留，infra fix）
├── modeling/                 # NEW — 重写的模型组件
│   ├── __init__.py           # 注册 CoartImageCondShapeDenoiser 进 trellis2.models
│   ├── denoiser.py           # CoartSLatFlowModel(SLatFlowModel)
│   │                         # CoartElasticSLatFlowModel(SparseTransformerElasticMixin, CoartSLatFlowModel)
│   │                         # 在 __init__ 末尾用 CoartDitBlock 替换 self.blocks，再 load_state_dict 迁移权重
│   ├── block.py              # CoartDitBlock — adapted from
│   │                         #   trellis2/modules/sparse/transformer/modulated.py
│   │                         # 内嵌 fused_modulation（无 monkey patch）；
│   │                         # 引用 CoartSparseMultiHeadRMSNorm
│   └── rmsnorm.py            # CoartSparseMultiHeadRMSNorm
│                             #   - C1: scale → fp32 buffer
│                             #   - C7: forward 用 flash_attn rms_norm_fn (W5 ship)
├── parallel/                 # NEW — FSDP2 / ZRO-1 wiring 隔离
│   ├── __init__.py
│   ├── ddp.py                # super().init_models_and_more 默认路径
│   ├── zro1.py               # _init_zro1 + save consolidate
│   ├── fsdp2.py              # _init_fsdp2 + distributed_ema + dcp_io
│   └── checkpoint.py         # 公共 DCP save/load helpers
└── configs/
    └── coart_dit_shape_512_ft.json
```

### 3.3 实施原则（spec 总则）

1. **优先 inheritance**：`CoartImageCondShapeDenoiser(ModulatedSparseDitImageCondLatents)` 只覆盖 `_build_blocks`。
2. **重写 vs patch 阈值**：当 trellis2 私有方法 ≥ 50 LOC 需修改时 → 直接拷出来改，注释标 `# Adapted from trellis2/<path>:<line> @ <commit-sha>`。
3. **trellis2 公共注册接口可调**（datasets / trainers / models）；私有 `_forward` / 私有属性 → **禁 monkey-patch**。
4. **trainer 适配 hook** 通过 dispatch 到 `coart/dit/parallel/{mode}.py`，`trainer.py` 主体只做 routing（< 30 LOC 改动）。
5. **环境变量 gate 退场**：`COART_FUSE_MODULATION` / `COART_FUSE_ROPE` 等不再需要——重写后默认 enable，旧行为通过 trellis2 的旧 denoiser config 切回。

---

## 4. Phase A — FSDP2 集成 detail

### 4.1 A1 ZRO-1（30 LOC，~1h）

**位置**：`coart/dit/parallel/zro1.py`

```python
def init_with_zro1(trainer, **kwargs):
    # 1. Default DDP init
    trainer._super_init_models(**kwargs)
    # 2. Wrap optimizer
    from torch.distributed.optim import ZeroRedundancyOptimizer
    base_cls = type(trainer.optimizer)        # AdamW
    base_kwargs = trainer.optimizer.defaults
    trainer.optimizer = ZeroRedundancyOptimizer(
        trainer.parallel_model.parameters(),
        optimizer_class=base_cls,
        **base_kwargs,
    )

def consolidate_optim_state_for_save(trainer):
    trainer.optimizer.consolidate_state_dict(to=0)
```

`trainer.save()` 调 `consolidate_optim_state_for_save(self)` 后 rank 0 写。

**收益**：optim state -10 GB/rank（22 → 12 GB static）。Step time ~0%。

### 4.2 A2 FSDP2 zero2（200 LOC，~5h）

#### 4.2.1 Module wrap

`coart/dit/parallel/fsdp2.py`：

```python
def init_with_fsdp2(trainer, *, reshard_after_forward: bool, **kwargs):
    trainer._super_init_models(**kwargs)
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,    # bit-equivalent default
    )
    # Per-block shard — 30 个 CoartDitBlock 各一个 unit
    for blk in trainer.parallel_model.blocks:
        fully_shard(blk, mp_policy=mp,
                    reshard_after_forward=reshard_after_forward)
    # Root wrap — embed/proj/output_proj 等顶层
    fully_shard(trainer.parallel_model, mp_policy=mp,
                reshard_after_forward=reshard_after_forward)
```

#### 4.2.2 Distributed EMA

替换现 rank-0 unshard EMA。在 `_init_fsdp2` 完成后，对每个 ema_rate 创建一组 per-param DTensor shard EMA buffer（与 model param 同 mesh）：

```python
def init_distributed_ema(trainer):
    trainer.ema_states = []
    for rate in trainer.ema_rate:
        ema_shard = {}
        for name, p in trainer.parallel_model.named_parameters():
            # p is DTensor under FSDP2; ema buffer keeps same sharding
            ema_shard[name] = p.detach().clone().to(torch.float32)
        trainer.ema_states.append(ema_shard)

def update_ema(trainer):
    for rate, ema_shard in zip(trainer.ema_rate, trainer.ema_states):
        for name, p in trainer.parallel_model.named_parameters():
            local_p = p.detach().to_local()           # bf16 local shard
            local_ema = ema_shard[name].to_local()    # fp32 local
            local_ema.mul_(rate).add_(local_p.float(), alpha=1.0 - rate)
```

**Save 时**通过 DCP `get_model_state_dict(full_state_dict=True, broadcast_from_rank0=False)` gather 到 rank 0 落盘。

#### 4.2.3 DCP save / load

`coart/dit/parallel/checkpoint.py`：

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)

def save_full_state(model, optimizer, path):
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=opts)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=opts)
    if dist.get_rank() == 0:
        torch.save({"model": model_sd, "optim": optim_sd}, path)

def load_full_state(model, optimizer, path):
    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    full = torch.load(path, map_location="cpu") if dist.get_rank() == 0 else None
    set_model_state_dict(model, full["model"] if full else None, options=opts)
    set_optimizer_state_dict(model, optimizer,
                              full["optim"] if full else None, options=opts)
```

DDP-trained ckpt 通过此路径加载到 FSDP2 model（`broadcast_from_rank0=True` 会重分片到各 rank）。

#### 4.2.4 Trainer 适配 hook

| Hook | 改动 | 原因 |
|---|---|---|
| `init_models_and_more()` | dispatch 到 `parallel/{mode}.py` | 路由 |
| `update_ema()` | FSDP2 mode → distributed EMA | 正确性 |
| `save()` / `load()` / `finetune_from()` | FSDP2 mode → DCP 路径 | DDP↔FSDP2 ckpt 兼容 |
| `check_ddp()` | FSDP2 下 no-op | DDP buffer assert 在 FSDP2 下结构性失败 |
| `run_step()` 的 `no_sync()` | 替换成 `set_requires_gradient_sync(is_last)` | API 不同 |
| `LinearMemoryController.max_mem_ratio_start` | zero2 保 0.5 | spec 默认 |
| `mix_precision_mode` 入口 assert | 必须 == 'amp' | 'inflat_all' 与 FSDP2 不兼容 |

#### 4.2.5 Config schema

`coart/dit/configs/coart_dit_shape_512_ft.json` 增加：

```jsonc
{
  "trainer": {
    "args": {
      "parallel_mode": "fsdp2_zero2",
      "fsdp2": {
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "shard_blocks_individually": true,
        "cpu_offload": false
      }
    }
  }
}
```

`parallel_mode` 取值：`"ddp"` | `"zro1"` | `"fsdp2_zero2"`（zero3 future）。

### 4.3 验证 gate（W2 退场标准）

1. **数值等价**：2-GPU 5 step DDP vs ZRO-1 loss diff < 1e-5；vs FSDP2 zero2 < 1e-4
2. **Ckpt 兼容**：DDP step-100 ckpt → FSDP2 load → step-101 loss 在 DDP step-101 的 ±2σ 内
3. **EMA 正确性**：FSDP2 训 50 step → save → reload → EMA state 与"全量 unshard 计算"对照 fp32 max abs diff < 1e-4
4. **Mem 实测**：zero2 peak ≤ DDP - 12 GB（target -14 GB；2 GB margin 用于 unshard burst noise）
5. **Step time**：
   - zero2 ≤ DDP × 1.05 → 通过；
   - DDP × 1.05 < zero2 ≤ DDP × 1.10 → 通过但 log warning，W3 优先排查 wrap 粒度；
   - zero2 > DDP × 1.10 → 阻塞 W3，与 user 议 reduce_dtype=bf16 ablation 决策

### 4.4 失败 rollback

- A1 失败：`parallel_mode: "ddp"`，0 影响
- A2 失败但 A1 OK：保留 ZRO-1 拿 -10 GB；FSDP2 改动通过 config gate 不启用
- EMA 错误：单独 patch（distributed EMA 是独立模块）
- 重组失败（W2.0 数值非等价）：revert 整个 modeling/ refactor，回到 monkey-patch 形态（保底方案）

---

## 5. Phase B — Profiling protocol

### 5.1 数据收集 sweep（~2h GPU）

| Cell | parallel_mode | bs/gpu | batch_split | active steps | output |
|---|---|---|---|---|---|
| `B1_zero2_bs8` | fsdp2_zero2 | 8 | 2 | 50 | metrics + chrome trace |
| `B2_zero2_bs10` | fsdp2_zero2 | 10 | 2 | 50 | metrics + chrome trace |
| `B3_zero2_bs12` | fsdp2_zero2 | 12 | 2 | 30 | metrics + chrome trace |
| `B4_ddp_bs10` | ddp | 10 | 2 | 30 | 对照（隔离 FSDP2 vs bs↑ 各自 ROI）|

每 cell rank-0 写 chrome trace ~1.2 GB。`profile_dit.py --extra-env` 透传：
- `SPARSE_ATTN_BACKEND=flash_attn_3`（W1 已 ship）
- 不再需要 `COART_FUSE_MODULATION`（已内嵌进 modeling/block.py）

### 5.2 Overlap 分析脚本（commit 进 repo）

新文件：`scripts/profiling/analyze_overlap.py`（~50 LOC）

```
input:  chrome trace .json
output:
  bucket_table.md            # 与 DDP baseline 同格式，便于 diff
  critical_path_timeline.svg # gantt 图：default-stream / nccl-stream
  overlap_summary.json:
    compute_stream_busy_ms
    comm_stream_busy_ms
    comm_hidden_ratio        # 1 - max(0, comm_visible / step_total)
    eltwise_oncritical_ms
    eltwise_offcritical_ms
    optim_step_isolated_ms   # 决定 C6 ROI
```

**核心算法**：trace 取 default-stream + nccl-stream events，sort by start time，逐时段算 stream concurrency。

### 5.3 决策输出

输出 `my-docs/<date>-fsdp2-profile-summary.md`，含：

1. 新 bucket ranking
2. comm overlap 健康度（comm_hidden_ratio）
3. eltwise on-critical-path 实测
4. bs10 vs bs12 throughput trade-off
5. C6 / C7 优先级 revise（按 §6.5 决策树）

### 5.4 失败模式 + 处理

| 现象 | 解读 | 处理 |
|---|---|---|
| comm_hidden_ratio < 70% | wrap 粒度不当 | grouped wrap（5 blocks/组）重测 |
| step time > DDP × 1.10 | DTensor 算子分发慢 / mp fp32 reduce 拖累 | 缩小 wrap 粒度 / 与 user 议 reduce_dtype=bf16 |
| optim_step_isolated_ms < 30 ms | C6 ROI 不存在 | 删 C6，W4 只做 C5 |
| eltwise_oncritical_ms 仍 ≥ 400 ms | C7 priority 不变 | 按计划 W5 ship |
| bs12 mem peak > 75 GB | 余量不够 | 退回 bs10 或评估 F2 selective ckpt |

---

## 6. Phase C — Accel candidate inventory + decision tree

### 6.1 Bit-equivalent（W1 / W4 默认 ship）

| # | 候选 | ROI | LOC | 实施位置 |
|---|---|---|---|---|
| C1 | `SparseMultiHeadRMSNorm.scale` Python float → `register_buffer(fp32)` | -25 ms | 5 | `modeling/rmsnorm.py` |
| C2 | `AdamW(fused=True, foreach=True)` 显式开启验证 | -30~50 ms | 10 | `trainer.py` |
| C3 | FA3 production default-on | +3% sample/s | config | shell + config |
| C4 | NCCL `bucket_cap_mb` 25→50；`gradient_as_bucket_view=True`（DDP only）| -5~10 ms | config | trainer init |
| C5 | bs/gpu 8 → 10/12 sweep | +18~25% sample/s | config | configs |
| C6 | `apply_optim_in_backward`（FSDP2 zero2）| -30~50 ms | 50 | `parallel/fsdp2.py` |
| C8 | `torch._foreach_*` 验证 master-grad chain | -10~20 ms | 10 | `trainer.py` |
| C9 | DataLoader `prefetch_factor` 2→4，`persistent_workers=True` 验证 | -1~3 ms | config | trainer |

**W1 ship**：C1 + C2 + C3 + C4 + C8 + C9
**W4 ship**：C5 + C6（依赖 W2）

### 6.2 ULP-close（W5，需 numerical gate）

| # | 候选 | ROI | LOC | gate |
|---|---|---|---|---|
| **C7** | `CoartSparseMultiHeadRMSNorm` via `flash_attn.ops.rms_norm.rms_norm_fn`（per-head gamma 拆分）| **-110~140 ms (-9~12% step)** | 80 + tests | 三层 gate（见 §7） |
| C10 | AdamW `eps_in_fp32=True`（如 PyTorch 默认非 fp32）| -1~3 ms | 5 | 同 C7 简化版 |

**W5 ship**：C7（C10 视 C2 实测情况补）

### 6.3 Deferred（数值改动较大）

| # | 候选 | 假定 ROI | 为何 defer |
|---|---|---|---|
| D1 | `share_mod` 路径 modulation fp32 → bf16 | -40~60 ms | trainable param dtype 变 |
| D2 | `mp_policy.reduce_dtype = bf16` | -5% step | grad reduce 噪声显著 |
| D3 | fp8 via TransformerEngine | -10~15% | 1000+ LOC |
| D4 | LayerNorm bf16 affine（FSDP2 之外的 LN）| -10 ms | LN bucket 余量小 |

写入 spec **"Future / blocked candidates"** appendix；触发条件：W5 完成后若仍未达 step ≤ 1.10 s 才单独 brainstorm。

### 6.4 算法/架构级（独立 spec，本 spec 仅占位）

| # | 候选 | 假定 ROI | 备注 |
|---|---|---|---|
| F1 | Cross-attn KV cache for static DINO features | -10~30 ms | DINO features 在 timestep 不变 → K/V 可预算 cache 进 dataset |
| F2 | Selective gradient checkpointing | -20~50 ms | mem 余量充足时关 K 个 ckpt |
| F3 | FSDP2 zero3（future scale model）| 释放 -4 GB | mem 紧时启用 |
| F4 | Custom Triton: LN + (1+s)·h + shift fused | -50 ms | modeling/block.py 重写时可一起评估 |
| F5 | Cross-attn 切 FA3（trellis2 dispatch 扩展）| 小 | trellis2 cross-attn 仍 FA2 |

→ 全部 W6+ 独立 brainstorm。

### 6.5 W3 后决策树

```
W3 profile data 回来 → 查 overlap_summary.json：

if comm_hidden_ratio < 70%:
    → grouped wrap 重测 → goto W3 second pass

if eltwise_oncritical_ms 仍 ≥ 400 ms:
    → C7 priority 不变，按计划 W5 ship

if eltwise_oncritical_ms 已被部分 overlap 到 < 300 ms:
    → C7 ROI 下调；若 ROI 仍 > 50 ms 继续 W5
    → 否则把 C7 移到 W6（与 F1 cross-attn KV cache 比较 ROI）

if optim_step_isolated_ms < 30 ms:
    → C6 ROI 不存在，删 C6，W4 只做 C5

if zero2 step time > DDP × 1.10:
    → 触发 wrap 粒度调整 + reduce_dtype=bf16 ablation 决策（升级到与 user 单独讨论）

if bs12 mem peak > 75 GB:
    → 退回 bs10 或评估 F2 selective ckpt
```

---

## 7. C7 numerical gate（三层）

W5 实施 fused RMSNorm 必须依次通过：

1. **ULP test**（unit test，CI 内）：
   - 100 个随机 input shape `[T, num_heads, head_dim]`，T ∈ [256, 8192]，num_heads = 12, head_dim = 128
   - bf16 input + bf16 output；与 baseline `SparseMultiHeadRMSNorm.forward` 比对
   - **gate**：max ULP diff < 16 across all positions

2. **1K-step loss A/B**（1 GPU 短跑）：
   - 同 seed，跑 1K step `parallel_mode=ddp` baseline vs `parallel_mode=ddp + C7`
   - **gate**：cumulative loss diff < 2σ（σ 为 baseline loss curve 的 std over 1K steps）

3. **5K-step trajectory verify**（8 GPU 中跑）：
   - 同 seed 跑 5K step，每 500 step 跑 eval set forward
   - **gate**：eval metric (CD / IoU 等)在 baseline ±5% 内

**任一 gate 失败** → revert C7，spec mark 为 "blocked，需自写 triton kernel"，新 brainstorm。

---

## 8. Wave plan（最终版）

| Wave | 内容 | 期望 cumulative |
|---|---|---|
| **W1**（0.5d）| C1 + C2 + C3 + C4 + C8 + C9（独立无依赖）| step 1.26 → 1.18 (-6%); sample/s 50.8 → 54 (+6%) |
| **W2.0**（0.5d）| Refactor: 删 4 patch + 立 modeling/+parallel/ + fold fused_modulation；5-step bit-exact verify | 0% Δ |
| **W2.1**（0.5d）| A1 ZRO-1 | step ~1.18; mem -10 GB |
| **W2.2**（1d）| A2 FSDP2 zero2 + distributed EMA + DCP save/load | step ~1.18; mem peak ≤ DDP - 12 GB（target -14 GB）|
| **W3**（0.5d）| B profile + overlap analysis + decision tree apply | 0%（测量）|
| **W4**（1d）| C5 (bs=12) + C6 (apply_optim_in_backward) | step 1.40 (bs12) sample/s 68.6 (+35%) |
| **W5**（1.5d）| C7 fused RMSNorm + 三层 gate | step 1.27 (bs12) sample/s 75.6 (+49%) |
| **(W6 可选)** | F1 cross-attn KV cache | step 1.21 sample/s 79.3 (+56%) |

总开发 + verify 工时：~5.5 天 wallclock。

---

## 9. 验证 gate 总表（cross-wave）

| Wave | Gate | 失败处理 |
|---|---|---|
| W1 | 200-step smoke：step ↓ ≥ 3%，loss diff < 1e-7 (bit-equiv) | 单项 revert |
| W2.0 | 5-step bit-exact loss diff < 1e-9（重组数值不变）| revert refactor，保留 monkey-patch 形态 |
| W2.1 | 5-step DDP vs ZRO-1 loss diff < 1e-5；mem -10 GB | revert，回 ddp |
| W2.2 | 5-step FSDP2 vs DDP loss diff < 1e-4；DDP→FSDP2 ckpt step-1 连续；mem peak ≤ DDP-12 GB；step ≤ DDP × 1.05（warning 区间 1.05~1.10）；EMA reload max abs diff < 1e-4 | revert FSDP2，保留 ZRO-1 |
| W3 | trace 完整捕 5 active steps；overlap_summary.json 输出 | 重跑 |
| W4 | bs12 mem peak ≤ 75 GB；C6 后 optim bucket -30~50 ms；loss 等价 | C6 损坏 → revert |
| W5 | C7 三层 gate 全过 | revert C7，触发 D1 / F4 brainstorm |

---

## 10. Future / blocked appendix

- **D1-D4**：dtype 改动类，需独立 brainstorm（数值噪声评估、收敛验证、ablation 设计）
- **F1**：Cross-attn KV cache for static DINO — 高 ROI 候选，需修 dataset emit + block 接口；建议 W5 完成后立即 brainstorm
- **F3 zero3**：当 mem 真紧 / scale ≥ 2.5B model 时启用；本 spec 已为 zero3 留 wrap 路径（`reshard_after_forward` 参数化）
- **F4 Triton fused LN+modulation**：W5 完成后若 step time 仍 > 1.10 s 评估
- **F5 cross-attn FA3**：trellis2 attention dispatch 扩展，需另 PR

---

## 11. 文档版本

- v1.0 (2026-05-03) — brainstorm 落地，待 user review
- 接替 `20260502-fsdp2-integration-spec.md` (v0)，合并 `20260503-eltwise-deep-analysis.md` 发现
