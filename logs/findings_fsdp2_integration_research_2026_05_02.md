# FSDP2 Integration Research — coart DiT (2026-05-02)

> **Recovery report**: rebuilt from a 60-minute research transcript. Source: `/tmp/claude-1001/-mnt-novita2-siyuan-workspace-TRELLIS-2/2a075bcc-c7d6-48c1-9df2-966f7994e578/tasks/aa1668475db17351f.output`. Branch `vae-finetune` @ `413109e`.

## 0. TL;DR

### Baseline (DDP, current `vae-finetune`)
- Per-rank static memory: ~22 GB (fp32 weights 5.20 + fp32 grads 5.20 + AdamW (m+v fp32) 10.40 + DDP buckets ~1).
- Mem peak 58.8 GB / step 2.77 s (anchors). FSDP2 cannot reduce activation portion, only static slice.
- Of 58.8 GB peak: ~22 GB is static, ~37 GB is activation/comm/sparse.

### Expected FSDP2 wins (per-rank, world_size=8)
| Stage | Strategy | Static / rank | Δ vs DDP | Step time impact | BS-per-GPU upper bound |
|-------|----------|---------------|----------|------------------|------------------------|
| **stage-1** `ZRO-1` | optimizer state shard only (`ZeroRedundancyOptimizer`) | 11.7 GB | -10.4 GB | ≈ 0% | +20-30% tokens |
| **stage-2** `SHARD_GRAD_OP` (FSDP2: `fully_shard(..., reshard_after_forward=False)`) | grads + opt sharded; params replicated post-fwd | 7.15 GB | -13.65 GB | ≈ +2-5% | +60-90% tokens |
| **stage-3** `FULL_SHARD` (FSDP2: `reshard_after_forward=True`) | full sharded params/grads/opt | 2.60 GB | -18.2 GB | +5-15% | 2-3× tokens |

### Implementation cost (per-PR, off `vae-finetune`)
- **ZRO-1**: ~30 LOC. Wrap optimizer in `ZeroRedundancyOptimizer`. Save needs `consolidate_state_dict()` first. Risk: low.
- **FSDP2 stage-2 / stage-3**: ~330 LOC across 4 files. Risk: medium.

### Key risks (highest first)
1. **Elastic memory controller × FSDP2 reshard semantics** — `LinearMemoryController` measures `max_memory_allocated` and back-fits a linear `tokens→mem` model. Floating unsharded param footprint will make fit noisy. Mitigation: lower `max_mem_ratio_start` from 0.5 → 0.3 in FSDP2 mode.
2. **EMA on rank-0 only** (`BasicTrainer.update_ema`, basic.py:592) — under FSDP2 `master_params` are local DTensor *shards*, so rank-0-only EMA only sees rank-0's shard. Recommended fix (option B): distributed EMA, all ranks own local shards, gather only at save time.
3. **`AdaptiveGradClipper`** — uses `np.percentile` on host-cached norms; PyTorch 2.6 confirmed `torch.nn.utils.clip_grad_norm_` works on DTensor out of the box, returning identical scalar across ranks. No extra plumbing.
4. **`SparseTransformerElasticMixin.with_mem_ratio`** mutates `self.blocks[i].use_checkpoint` per step. Safe under FSDP2: attribute access pierces the wrapper. No API change.
5. **`mp.spawn` + `mp_policy`** — FSDP2 requires CUDA visible per rank when `fully_shard` is called. `train.py` already does `setup_dist` → `.cuda()` → trainer construct. Order is correct.

### Validation
1. 2-GPU smoke, 5 steps DDP vs FSDP2 stage-3, same warmstart ckpt, expect <1e-4 loss diff with reduce_dtype=fp32.
2. Memory: peak ≈ DDP_peak − 18 GB at stage-3.
3. Save→load round-trip: DDP save → FSDP2 load via `broadcast_from_rank0=True` → step-1 loss matches.

### Recommendation
**Do not jump straight to FULL_SHARD.** Order: (1) ship ZRO-1 as config flag, (2) FSDP2 stage-2 in second PR, (3) promote to stage-3 only if `max_tokens > 12k` or scale-up to ≥2.5B params.

---

## 1. PyTorch 2.6 FSDP2 API speedrun

### 1.1 Single entry point: `fully_shard()`
File: `.venv/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fully_shard.py:51`

```python
def fully_shard(
    module: Union[nn.Module, List[nn.Module]],
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
):
```

Public re-exports from `torch.distributed.fsdp._fully_shard.__init__`: `CPUOffloadPolicy`, `MixedPrecisionPolicy`, `OffloadPolicy`, `FSDPModule`, `fully_shard`, `register_fsdp_forward_method`, `UnshardHandle`.

Idiomatic per-block + root wrap (bottom-up required):
```python
for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy, reshard_after_forward=True)
fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)  # root
```

### 1.2 Stage mapping (FSDP1 enum → FSDP2)
From `torch/distributed/fsdp/api.py:32-69`:

| FSDP1 | FSDP2 equivalent |
|-------|------------------|
| `FULL_SHARD` (=ZeRO-3) | `fully_shard(..., reshard_after_forward=True)` |
| `SHARD_GRAD_OP` (=ZeRO-2) | `fully_shard(..., reshard_after_forward=False)` |
| `NO_SHARD` | DDP fallback |
| `HYBRID_SHARD` | 2D `DeviceMesh` argument |
| `_HYBRID_SHARD_ZERO2` | 2D mesh + `reshard_after_forward=False` |

`reshard_after_forward` also accepts `int` for prefetch-scoped resharding.

### 1.3 Mixed precision policy
File: `_fsdp_api.py:8`. `MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs)`.

> "Unlike autocast, this applies mixed precision at the module level, not op level... low-precision activations are saved for backward and high-to-low-precision casts are incurred only at module boundaries... FSDP works well with module-level mixed precision since it keeps the high-precision sharded parameters in memory anyway."

Recipe to replicate current `mix_precision_mode='amp', mix_precision_dtype='bfloat16'`:
```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # recommended for stability
)
```
**Makes `torch.autocast` redundant for the wrapped DiT** (basic.py:675 amp_context becomes effectively no-op).

### 1.4 State-dict path (DCP)
File: `torch/distributed/checkpoint/state_dict.py`. Recommended FSDP2 idiom:

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
    get_optimizer_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)

# Save: full unsharded weights only on rank 0
sd = get_model_state_dict(model, options=StateDictOptions(
    full_state_dict=True, cpu_offload=True))
if rank == 0:
    torch.save(sd, "model_step.pt")

# Load: read on rank 0, broadcast-shard to others
full_sd = torch.load("model_step.pt") if rank == 0 else None
set_model_state_dict(model, full_sd, options=StateDictOptions(
    full_state_dict=True, broadcast_from_rank0=True))
```

`StateDictOptions` (state_dict.py:105):
- `full_state_dict=True` → all-gather DTensor → plain Tensor.
- `cpu_offload=True` → on rank 0 only when combined with full_state_dict.
- `broadcast_from_rank0=True` → DCP shards and broadcasts during load (DTensor only, not legacy ShardedTensor — FSDP2 is DTensor, OK).

### 1.5 Activation checkpoint compatibility
Confirmed: `trellis2/modules/sparse/transformer/{blocks.py,modulated.py}` uses `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. **Non-reentrant** is the FSDP2-compatible variant. No change required.

### 1.6 Other FSDP2 knobs (FSDPModule API, _fully_shard.py:223+)
- `set_requires_gradient_sync(bool)` — replaces FSDP1 `no_sync()`. Critical for `batch_split`.
- `set_reshard_after_backward(bool)` — keep params unsharded between bwd and next fwd in accumulation.
- `set_is_last_backward(bool)` — explicit hook for last microbatch.
- `set_modules_to_forward_prefetch([...])` — manual prefetch tuning.
- `reshard()` / `unshard(async_op=...)` — explicit sharding control.

### 1.7 Gradient clipping
WebFetch confirmed: `torch.nn.utils.clip_grad_norm_` works on DTensor out of the box, returning identical scalar across ranks. `AdaptiveGradClipper` consumes the scalar; no change required.

---

## 2. trellis2 BasicTrainer × DDP coupling — line by line

### 2.1 Model wrap — `init_models_and_more` (basic.py:202-219)
**Current**:
```python
self.training_models = {
    name: DDP(model, device_ids=[self.local_rank], output_device=self.local_rank,
              bucket_cap_mb=128, find_unused_parameters=False)
    for name, model in self.models.items()
}
```
**FSDP2**:
```python
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for name, model in self.models.items():
    if hasattr(model, "blocks"):
        for b in model.blocks:
            fully_shard(b, mp_policy=mp_policy, reshard_after_forward=reshard_flag)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_flag)
self.training_models = self.models  # FSDP2 mutates in place
```
Risk: low. LOC: ~25.

### 2.2 `master_params` alias (basic.py:222-236)
With `mix_precision_mode='amp'`: `master_params IS model_params` — same nn.Parameter list. Under FSDP2 these are DTensors (sharded). AdamW operates correctly on DTensor. **Optimizer must be constructed AFTER `fully_shard`** (current code already does this — order correct).

`mix_precision_mode='inflat_all'` (basic.py:230) creates a flat fp32 master buffer — **incompatible with FSDP2** (cannot flatten DTensor shards). Add assertion gating. coart config uses `'amp'` so this is fine.

### 2.3 EMA on rank 0 (basic.py:239-240, 592-600)
**Problem**: `master_params` on rank 0 are local DTensor shards. `deepcopy + .mul_().add_()` only EMAs rank-0's shard; other ranks' shards are never EMA'd.

**Recommended fix (option B): distributed EMA**: drop `if self.is_master`, every rank owns its shard's EMA. Mem = 5.2 GB / 8 ranks per ema_rate = 0.65 GB/rank/rate × 2 rates = 1.3 GB/rank. Save: gather EMA shards via `get_model_state_dict` against a temp module pointed at the EMA list.

Risk: medium. LOC: ~60.

### 2.4 `save()` (basic.py:368-421)
Current: rank-0 only, calls `_master_params_to_state_dicts` then `model.state_dict()`. Under DDP returns full replicated tensors.

Under FSDP2: `model.state_dict()` returns DTensor shards. Must use `get_model_state_dict(..., StateDictOptions(full_state_dict=True, cpu_offload=True))`. For optimizer, use `get_optimizer_state_dict`. Threading-based non-blocking saves still work (rank-0 owns the full CPU tensor after DCP offload).

Risk: medium. LOC: ~50.

### 2.5 `load()` (basic.py:319-366) and `finetune_from()` (basic.py:423-466)
Current: `model.load_state_dict(model_ckpt)` per rank.

Under FSDP2:
```python
full_sd = torch.load(path, map_location="cpu") if rank == 0 else None
set_model_state_dict(model, full_sd, options=StateDictOptions(
    full_state_dict=True, broadcast_from_rank0=True))
```
`finetune_from` shape-check stays on rank 0 against the full ckpt, then `set_model_state_dict(strict=False)`. The custom warmstart converter (`scripts/coart_warmstart_dit.py`) saves a plain `state_dict()` — loads cleanly via this API with no conversion.

Risk: medium. LOC: ~40.

### 2.6 `no_sync` for grad accumulation (basic.py:685)
Replace context manager with FSDP2's per-module `set_requires_gradient_sync(is_last)` call:
```python
for i, mb_data in enumerate(data_list):
    is_last = (i == len(data_list) - 1)
    if self.world_size > 1:
        for m in self.training_models.values():
            m.set_requires_gradient_sync(is_last)
    with elastic_controller_context():
        ... loss.backward()
```
Add `set_reshard_after_backward(False)` if `batch_split > 1` to avoid intra-cycle re-gathers.

Risk: low. LOC: ~15.

### 2.7 Gradient clipping (basic.py:704-716)
Native `clip_grad_norm_` works on DTensor. `AdaptiveGradClipper` consumes scalar. The `inflat_all`-specific lines 707-710 are already gated by `mix_precision_mode == 'inflat_all'` branch — they self-skip under FSDP2. LOC: 0.

### 2.8 `ElasticModuleMixin` interaction
`SparseTransformerElasticMixin.with_mem_ratio` (sparse_elastic_mixin.py:13-24) flips `self.blocks[i].use_checkpoint` per step. Attribute pass-through works through FSDP2 wrapper. **No API change.**

`LinearMemoryController.record()` (elastic_utils.py:86-98) measures peak mem; FSDP2's float unsharded param burst makes the linear fit noisier. Mitigation: lower `max_mem_ratio_start` 0.5 → 0.3 in FSDP2 config. Risk: medium for first ~500 steps.

### 2.9 `check_ddp` consistency check (basic.py:602-632)
Current: all-gathers each `master_param`, asserts equal. **Under FSDP2 each rank holds different shard — assertion structurally false.** Two fixes:
- (A) No-op under FSDP2: `if self.parallel_mode.startswith('fsdp2'): return`.
- (B) Use DTensor `full_tensor()` / `redistribute(placements=[Replicate()])`, then all-gather and assert. Expensive (5 GB temp), only at `i_ddpcheck=10000`.

Risk: low. LOC: ~15.

### 2.10 Dataloader (basic.py:271-289)
`ResumableSampler` with `shuffle=True`. Under FSDP2 data parallelism is still needed; FSDP2 only changes parameter sharding. **No change required for FSDP2.** (recovery: needs follow-up read of `trellis2/utils/data_utils.py` to confirm rank-aware behaviour.)

---

## 3. 3-stage switch interface design

### 3.1 Config schema additions
Top-level `parallel_mode` in trainer args:
- `"ddp"` (default, no change)
- `"zro1"` (DDP wrap + `ZeroRedundancyOptimizer`)
- `"fsdp2_zero2"` (`reshard_after_forward=False`)
- `"fsdp2_zero3"` (`reshard_after_forward=True`)

Optional `fsdp2` sub-block:
```json
"fsdp2": {
    "param_dtype": "bfloat16",
    "reduce_dtype": "float32",
    "reshard_after_forward": true,
    "cpu_offload": false,
    "shard_blocks_individually": true
}
```

### 3.2 `init_models_and_more` override (in `coart/dit/trainer.py` to avoid touching trellis2)
```python
def init_models_and_more(self, **kwargs):
    if self.parallel_mode in ("fsdp2_zero2", "fsdp2_zero3"):
        self._init_fsdp2(**kwargs)
    elif self.parallel_mode == "zro1":
        self._init_zro1(**kwargs)
    else:
        super().init_models_and_more(**kwargs)

def _init_fsdp2(self, **kwargs):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy
    cfg = getattr(self, "fsdp2_cfg", {})
    mp = MixedPrecisionPolicy(
        param_dtype=str_to_dtype(cfg.get("param_dtype", "bfloat16")),
        reduce_dtype=str_to_dtype(cfg.get("reduce_dtype", "float32")),
    )
    op = CPUOffloadPolicy() if cfg.get("cpu_offload", False) else OffloadPolicy()
    reshard = (self.parallel_mode == "fsdp2_zero3")
    for name, model in self.models.items():
        if cfg.get("shard_blocks_individually", True) and hasattr(model, "blocks"):
            for b in model.blocks:
                fully_shard(b, mp_policy=mp, offload_policy=op, reshard_after_forward=reshard)
        fully_shard(model, mp_policy=mp, offload_policy=op, reshard_after_forward=reshard)
    self.training_models = self.models
    self.model_params = sum([[p for p in m.parameters() if p.requires_grad] for m in self.models.values()], [])
    assert self.mix_precision_mode in ("amp", None), "FSDP2 requires amp/None"
    self.master_params = self.model_params
    # Distributed EMA (option B from §2.3)
    self.ema_params = [
        [torch.empty_like(p, requires_grad=False).copy_(p.detach()) for p in self.master_params]
        for _ in self.ema_rate
    ]
    self.optimizer = build_optimizer(self.optimizer_config, self.master_params)
    # ... lr_scheduler, elastic, grad_clip same as super
```

### 3.3 ZRO-1 in same switch
```python
def _init_zro1(self, **kwargs):
    super().init_models_and_more(**kwargs)  # DDP + master_params + AdamW
    from torch.distributed.optim import ZeroRedundancyOptimizer
    optimizer_cls = type(self.optimizer)
    state = self.optimizer.state_dict()
    self.optimizer = ZeroRedundancyOptimizer(
        self.master_params,
        optimizer_class=optimizer_cls,
        **self.optimizer_config["args"],
    )
    self.optimizer.load_state_dict(state)
```
For save: `self.optimizer.consolidate_state_dict()` on rank 0 first.
Risk: low. LOC: ~30. Hard win: -10 GB/rank.

### 3.4 `profile_dit` sweep YAML
Append to `scripts/profiling/sweeps/`:
```yaml
- name: parallel_modes_sweep
  matrix:
    parallel_mode: ["ddp", "zro1", "fsdp2_zero2", "fsdp2_zero3"]
    batch_size_per_gpu: [4, 8, 16, 24]
    cuda_visible_devices: "0,1,2,3"
  fixed:
    config: coart/dit/configs/coart_dit_shape_512_ft.json
    max_steps: 50
```

---

## 4. Implementation plan

| # | File | Lines | Change | Test | Est |
|---|------|-------|--------|------|-----|
| 1 | `coart/dit/configs/coart_dit_shape_512_ft.json` | +5 | Add `parallel_mode: "ddp"` | `python train.py --tryrun` | 5m |
| 2 | `coart/dit/trainer.py` | +35 | `_init_zro1()` + dispatch | 2-GPU tryrun zro1, 5 steps, loss vs DDP <1e-5 | 1h |
| 3 | `coart/dit/trainer.py` | +15 | `save()` with `consolidate_state_dict()` for zro1 | save→restart→5 more steps | 30m |
| 4 | `coart/dit/trainer.py` | +120 | `_init_fsdp2()` (per-block wrap, mp_policy, distributed EMA) | 2-GPU tryrun fsdp2_zero3 | 2h |
| 5 | `coart/dit/trainer.py` | +60 | `save()` / `load()` for FSDP2 via DCP | DDP→FSDP2 ckpt round-trip | 2h |
| 6 | `coart/dit/trainer.py` | +30 | `update_ema()` distributed | 50-step EMA save/reload | 1h |
| 7 | `coart/dit/trainer.py` | +15 | `run_step()`: `set_requires_gradient_sync` | `batch_split=2` numeric test | 45m |
| 8 | `coart/dit/trainer.py` | +20 | `check_ddp()` no-op or DTensor `full_tensor()` | `i_ddpcheck=100` runs clean | 30m |
| 9 | `scripts/profiling/sweeps/parallel_modes.yaml` | +30 | Sweep entry | profile_dit runs all 4 modes | 30m |
| 10 | `coart/dit/configs/coart_dit_shape_512_zero3_ft.json` | new | FSDP2_zero3 config + lower `max_mem_ratio_start` | 100-step smoke 8 GPUs | 30m |

Total ~330 LOC, ~9.5h. Split into 3 PRs: (1-3) ZRO-1, (4-7) FSDP2, (8-10) polish + profiling.

---

## 5. Risks + fallbacks

### 5.1 SparseTensor compatibility
`SparseLinear(nn.Linear)` confirmed (sparse/linear.py:10). FSDP2 only sees `nn.Parameter` (dense weights inside). `sp.SparseTensor` flow uses custom autograd.Function but allocates no `nn.Parameter`. Risk: low.

### 5.2 ElasticMixin × FSDP2
Risk is *training stability*, not correctness. Controller will reconverge. Mitigation: `max_mem_ratio_start=0.3` for FSDP2 (planned). Risk: medium, mitigated.

### 5.3 Save/load round-trip
Validation: DDP save → FSDP2 load via `broadcast_from_rank0=True` → loss step-1 must match fp32 precision. Fallback if `set_optimizer_state_dict` is brittle: full-gather `get_optimizer_state_dict` on rank 0 → `torch.save`, load same way.

### 5.4 Numeric divergence DDP vs FSDP2
amp casts at op granularity; FSDP2 mp_policy at module boundary. DDP all-reduces in bf16; FSDP2 reduce-scatters in fp32. Expect <1e-3 relative loss diff after 100 steps. Not a regression.

### 5.5 Fallback path
All changes gated by JSON config `parallel_mode`. Default stays `"ddp"`. Reverting = changing one config key.

---

## 6. Not recommended

- **Do NOT use FSDP1's `FullyShardedDataParallel`** — maintenance-only in 2.6.
- **Do NOT enable `cpu_offload=True`** unless GPU memory forces it; H2D/D2H copies destroy throughput.
- **Do NOT mix `mix_precision_mode='inflat_all'` with FSDP2** — flat-fp32-master pattern incompatible with sharded DTensor. Add fail-fast assertion.
- **Do NOT drop `torch.autocast`** when only partial FSDP2 wrap is in use (we wrap 100% of denoiser, but keep autocast for safety on any unwrapped subm).
- **Do NOT per-block FSDP2 wrap `t_embedder` / `pos_embedder` / `input_layer` / `out_layer`** — they're tiny (<1M params each); fixed per-group overhead. Let root `fully_shard(model)` collect them.
- **Do NOT call `check_ddp()` as written under FSDP2** — assert-fail on shard mismatch. Either no-op or rewrite per §2.9.

---

## 7. Files referenced
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/trainers/basic.py` — BasicTrainer
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/trainers/utils.py` — `make_master_params` (inflat_all)
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/trainers/flow_matching/sparse_flow_matching.py` — `training_losses` (line 100)
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/trainers/flow_matching/mixins/image_conditioned.py` — `_init_image_cond_model`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/utils/dist_utils.py` — `setup_dist`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/utils/elastic_utils.py` — `LinearMemoryController`, `ElasticModuleMixin`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/utils/grad_clip_utils.py` — `AdaptiveGradClipper`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/models/sparse_elastic_mixin.py` — `with_mem_ratio`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/models/structured_latent_flow.py` — `ElasticSLatFlowModel`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/modules/sparse/linear.py` — `SparseLinear(nn.Linear)`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/modules/sparse/transformer/blocks.py` + `modulated.py` — non-reentrant checkpoint
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/coart/dit/trainer.py` — current trainer override surface
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/coart/dit/config.py` — paths, `WARMSTART_HINT`
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/coart/dit/configs/coart_dit_shape_512_ft.json` — current config
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/coart_warmstart_dit.py` — safetensors → trainer ckpt
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/train.py` — entry (mp.spawn + setup_dist)
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/*` — FSDP2
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/lib/python3.10/site-packages/torch/distributed/checkpoint/state_dict.py` — DCP
- `/mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/lib/python3.10/site-packages/torch/distributed/optim/zero_redundancy_optimizer.py` — ZRO-1
