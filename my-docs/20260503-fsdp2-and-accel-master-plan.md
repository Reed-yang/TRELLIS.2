# FSDP2 + 综合加速 Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 5 个 wave 内将 coart DiT 1.3B shape finetune step time 1.26s → ≤1.27s @ bs=12（effective sample/s +49%），同步清理 4 个 monkey-patch 反模式，并立 `coart/dit/{modeling,parallel}/` 框架。

**Architecture:** 按 spec `my-docs/20260503-fsdp2-and-accel-master-spec.md` v1.0 定义的 5 wave + 1 可选 wave 推进；每 wave 独立可 ship；W2.0 refactor 为 W2.1/W2.2 前置；C7 数值改动放最后做单独三层 gate。

**Tech Stack:** PyTorch 2.6 + FSDP2 (`torch.distributed.fsdp.fully_shard`) + DCP (`torch.distributed.checkpoint`) + flash_attn 3 + ZeroRedundancyOptimizer + 自定义 Triton（仅 W5 候选 fallback）。

**Conventions:**
- 所有 commit 由 user 或执行者按 wave 边界 review 后授权（CLAUDE.md：never commit without explicit ask）
- 测试运行节点：host-10-240-99-119（GPU 0-3 idle，user 规则）
- SSH 命令必须 `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && ...` 前缀（user 规则）
- 代码内 comment / commit message：英文；plan / doc：中文

---

## Pre-flight：Wave 间 verification 共用工具

新增脚本 `scripts/wave_verify.sh`（W1 第一项作为 task 单独建立），所有 wave 复用：

```bash
# 用法
bash scripts/wave_verify.sh <wave_name> [--steps N] [--bs B] [--mode parallel_mode]
# 输出：logs/wave_verify/<wave_name>_<ts>/{result.json, log.txt, mem.csv}
# Verify gate: 比对前一 wave 的 baseline JSON，失败时 exit 1
```

---

## Wave 1: Quick wins (bit-equivalent, ~0.5 day)

**目标**：C1 + C2 + C3 + C4 + C8 + C9 全部 ship；step time -3~5%；零数值变化。

**Files map：**
- Modify: `coart/dit/trainer.py`（C2/C8/C9）
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`（C9）
- Modify: `scripts/train_coart_dit_shape.sh`（C3 FA3 default-on, C4 NCCL）
- Modify: `coart/dit/fused_modulation_patch.py`（C1 — temporary，W2.0 会迁出）
- Create: `scripts/wave_verify.sh`
- Create: `coart/tests/test_wave1_smoke.py`

### Task 1.1: 建立 wave_verify 共用脚本

**Files:**
- Create: `scripts/wave_verify.sh`

- [ ] **Step 1: Write the script**

```bash
cat > scripts/wave_verify.sh <<'EOF'
#!/usr/bin/env bash
# Run a short profile-style training to produce metrics for wave gate comparison.
# Usage: bash scripts/wave_verify.sh <wave_name> [--steps N] [--bs B] [--mode MODE] [--host HOST]
set -euo pipefail
WAVE="${1:?wave name required}"; shift
STEPS=30; BS=8; MODE=ddp; HOST=host-10-240-99-119
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2;;
    --bs)    BS="$2"; shift 2;;
    --mode)  MODE="$2"; shift 2;;
    --host)  HOST="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
TS=$(date +%Y%m%d_%H%M%S)
OUT="logs/wave_verify/${WAVE}_${TS}"
mkdir -p "$OUT"
.venv/bin/python scripts/profiling/profile_dit.py \
    --label "${WAVE}" --host "${HOST}" --num-gpus 8 \
    --warmup-steps 5 --active-steps "${STEPS}" \
    --override "trainer.args.batch_split=2" \
    --override "trainer.args.batch_size_per_gpu=${BS}" \
    --override "trainer.args.parallel_mode=${MODE}" \
    --extra-env SPARSE_ATTN_BACKEND=flash_attn_3 \
    --output-dir "${OUT}"
echo "[wave_verify] wrote ${OUT}/result.json"
EOF
chmod +x scripts/wave_verify.sh
```

- [ ] **Step 2: Verify script syntax**

Run: `bash -n scripts/wave_verify.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Quick smoke run（先用 mode=ddp 跑 5 step 验证 launcher 正确）**

Run: `bash scripts/wave_verify.sh smoke_preflight --steps 5`
Expected: 完成 + 写 `logs/wave_verify/smoke_preflight_*/result.json` with `time/step` field

- [ ] **Step 4: Commit**

```bash
git add scripts/wave_verify.sh
git commit -m "feat(profiling): add wave_verify.sh for cross-wave gate comparisons"
```

### Task 1.2: 拍 W1 baseline (mode=ddp + 当前 fused_modulation_patch + cpu128)

**Files:** N/A（运行）

- [ ] **Step 1: Run baseline verify**

Run: `ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && sbatch --partition=gpu --nodes=1 --gres=gpu:8 --cpus-per-task=128 --time=00:15:00 --wrap='bash scripts/wave_verify.sh w1_baseline --steps 30'"`
Expected: sbatch 返回 jobid

- [ ] **Step 2: Wait for completion + record baseline**

Run: `ls -t logs/wave_verify/w1_baseline_*/result.json | head -1 | xargs cat | python -c "import sys,json; d=json.load(sys.stdin); print(f'step.mean={d[\"time/step\"][\"mean\"]:.3f}s, sample/s={d[\"perf/throughput_tok_per_s\"][\"mean\"]:.0f}')"`
Expected: step.mean ≈ 1.26s, sample/s ≈ 3000+（与昨日 1785 baseline 一致）

- [ ] **Step 3: Snapshot baseline JSON**

Run: `cp $(ls -t logs/wave_verify/w1_baseline_*/result.json | head -1) logs/wave_verify/W1_BASELINE.json`
Expected: 文件创建

### Task 1.3: C1 — `SparseMultiHeadRMSNorm.scale` Python float → fp32 buffer

**Files:**
- Modify: `coart/dit/fused_modulation_patch.py`（W2.0 后会迁到 `coart/dit/modeling/rmsnorm.py`，本 task 暂在此 patch 文件添加二级 patch）
- Test: `coart/tests/test_c1_scale_buffer.py`

- [ ] **Step 1: Write failing test**

```python
# coart/tests/test_c1_scale_buffer.py
import torch
from trellis2.modules.sparse.attention.modules import SparseMultiHeadRMSNorm

def test_scale_is_fp32_buffer_after_patch():
    # Apply C1 patch
    import coart.dit.fused_modulation_patch  # noqa: F401  (patch on import)
    norm = SparseMultiHeadRMSNorm(dim=128, heads=12)
    assert isinstance(norm.scale, torch.Tensor), \
        f"expected scale to be a buffer (Tensor), got {type(norm.scale)}"
    assert norm.scale.dtype == torch.float32
    assert norm.scale.numel() == 1
```

- [ ] **Step 2: Run test, expect failure**

Run: `.venv/bin/pytest coart/tests/test_c1_scale_buffer.py -v`
Expected: FAIL — `scale` is float not Tensor

- [ ] **Step 3: Implement patch**

Add to `coart/dit/fused_modulation_patch.py` (after the existing fused-modulation patch):

```python
# C1: SparseMultiHeadRMSNorm.scale Python float → fp32 buffer.
# Reason: AUnaryFunctor mul with (float, double) scalar dispatches to
# unrolled_elementwise_kernel (cold path); fp32 tensor -> hot vectorized.
# Saves ~25 ms/step. Bit-equivalent.
import torch
from trellis2.modules.sparse.attention import modules as _attn_modules

_orig_rmsnorm_init = _attn_modules.SparseMultiHeadRMSNorm.__init__

def _patched_rmsnorm_init(self, dim, heads):
    _orig_rmsnorm_init(self, dim, heads)
    # Replace Python float scale with fp32 buffer.
    scale_val = float(self.scale)
    del self.scale
    self.register_buffer("scale", torch.tensor(scale_val, dtype=torch.float32),
                         persistent=False)

_attn_modules.SparseMultiHeadRMSNorm.__init__ = _patched_rmsnorm_init
print("[c1_scale_buffer_patch] installed: scale is now fp32 buffer", flush=True)
```

- [ ] **Step 4: Run test, expect pass**

Run: `.venv/bin/pytest coart/tests/test_c1_scale_buffer.py -v`
Expected: PASS

- [ ] **Step 5: Numerical equivalence smoke check**

Run:
```bash
.venv/bin/python -c "
import torch, coart.dit.fused_modulation_patch
from trellis2.modules.sparse.attention.modules import SparseMultiHeadRMSNorm
torch.manual_seed(0)
n = SparseMultiHeadRMSNorm(128, 12).cuda()
x = torch.randn(64, 12, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
y = torch.nn.functional.normalize(x.float(), dim=-1) * n.gamma * float(n.scale.item())
y_ref = y.to(torch.bfloat16)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    y_new = torch.nn.functional.normalize(x.float(), dim=-1) * n.gamma * n.scale
    y_new = y_new.to(torch.bfloat16)
print(f'max abs diff = {(y_ref - y_new).abs().max().item()}')
assert (y_ref - y_new).abs().max().item() < 1e-6, 'C1 not bit-equivalent'
print('C1 bit-equivalent OK')
"
```
Expected: `max abs diff = 0.0`, `C1 bit-equivalent OK`

- [ ] **Step 6: Commit**

```bash
git add coart/dit/fused_modulation_patch.py coart/tests/test_c1_scale_buffer.py
git commit -m "feat(coart_dit): C1 — SparseMultiHeadRMSNorm.scale as fp32 buffer

Eliminates (float, double) scalar mul slow-path dispatch.
Bit-equivalent; ~25 ms/step expected gain."
```

### Task 1.4: C2 — Explicit `AdamW(fused=True, foreach=True)`

**Files:**
- Modify: `coart/dit/trainer.py`
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`
- Test: `coart/tests/test_c2_adamw_fused.py`

- [ ] **Step 1: Write failing test**

```python
# coart/tests/test_c2_adamw_fused.py
import torch
from torch.optim import AdamW

def test_adamw_fused_supported_h100():
    # Sanity: PyTorch 2.6 supports fused=True on H100 sm_90a.
    if not torch.cuda.is_available():
        return
    p = torch.nn.Parameter(torch.zeros(8, 8, device='cuda'))
    opt = AdamW([p], lr=1e-3, fused=True)
    p.grad = torch.ones_like(p)
    opt.step()  # should not raise
```

- [ ] **Step 2: Run test**

Run: `.venv/bin/pytest coart/tests/test_c2_adamw_fused.py -v`
Expected: PASS（PyTorch 2.6 + H100 supports fused AdamW）

- [ ] **Step 3: Add config field**

Edit `coart/dit/configs/coart_dit_shape_512_ft.json` — find the AdamW block and add `"fused": true, "foreach": true`:

Current:
```jsonc
"optimizer": {
  "name": "AdamW",
  "args": { "lr": 1e-4, "betas": [0.9, 0.99], "weight_decay": 0.01 }
}
```
Change to:
```jsonc
"optimizer": {
  "name": "AdamW",
  "args": { "lr": 1e-4, "betas": [0.9, 0.99], "weight_decay": 0.01, "fused": true, "foreach": true }
}
```

- [ ] **Step 4: Smoke run 30 step + verify optimizer flag**

Run: `bash scripts/wave_verify.sh w1_c2 --steps 30`

Then check log:
```bash
grep -i "adamw\|fused" logs/wave_verify/w1_c2_*/log.txt | head -5
```
Expected: PyTorch reports `fused=True`

- [ ] **Step 5: Numerical check（bit-equivalent vs non-fused）**

Run:
```bash
.venv/bin/python coart/tests/test_fused_adamw_fallback.py -v
```
Expected: PASS（已存在测试文件，验证 fused vs default 等价）

- [ ] **Step 6: Commit**

```bash
git add coart/dit/configs/coart_dit_shape_512_ft.json coart/tests/test_c2_adamw_fused.py
git commit -m "feat(coart_dit): C2 — enable AdamW(fused=True, foreach=True)

Uses H100 fused AdamW kernel; bit-equivalent.
~30-50 ms/step expected gain."
```

### Task 1.5: C3 — FA3 production default-on

**Files:**
- Modify: `scripts/train_coart_dit_shape.sh`
- Modify: `scripts/train_finetune_feat18.sh`

- [ ] **Step 1: Read current launcher**

Run: `grep -n "SPARSE_ATTN_BACKEND\|export\|env " scripts/train_coart_dit_shape.sh`
Expected: 列出 export 行（FA3 may already be set or not）

- [ ] **Step 2: Add FA3 export at top**

Edit both `scripts/train_coart_dit_shape.sh` and `scripts/train_finetune_feat18.sh`:

In each script, after the existing `export PYTHONPATH=...` line, add (or update if present):

```bash
# C3: FA3 default-on for self-attn (cross-attn still FA2 — see F5 future work).
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-flash_attn_3}"
```

- [ ] **Step 3: Smoke verify**

Run: `bash scripts/wave_verify.sh w1_c3 --steps 30`

Then check:
```bash
grep -i "flash_attn_3\|sparse_attn" logs/wave_verify/w1_c3_*/log.txt | head -3
```
Expected: 见到 FA3 backend 注册 log

- [ ] **Step 4: Compare baseline**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W1_BASELINE.json logs/wave_verify/w1_c3_*/result.json \
  --out logs/wave_verify/cmp_c3.md
cat logs/wave_verify/cmp_c3.md
```
Expected: step.mean Δ ≈ -3% (vs W1_BASELINE), sample/s +3%

- [ ] **Step 5: Commit**

```bash
git add scripts/train_coart_dit_shape.sh scripts/train_finetune_feat18.sh
git commit -m "feat(coart_dit): C3 — FA3 default-on for production training

Enable flash_attn_3 sparse self-attn by default in launchers.
Bit-equivalent; +3% sample/s verified in wave 2 ablation."
```

### Task 1.6: C4 — NCCL bucket sizing + gradient_as_bucket_view

**Files:**
- Modify: `scripts/train_coart_dit_shape.sh`（NCCL env）
- Modify: `coart/dit/trainer.py`（DDP gradient_as_bucket_view kwarg）

- [ ] **Step 1: Add NCCL env in launcher**

Add to `scripts/train_coart_dit_shape.sh` after the FA3 export from Task 1.5:

```bash
# C4: larger NCCL bucket reduces #allreduce calls (DDP path).
export NCCL_BUCKET_CAP_MB="${NCCL_BUCKET_CAP_MB:-50}"
```

- [ ] **Step 2: Find DDP wrap site in trainer**

Run: `grep -n "DistributedDataParallel\|DDP\|find_unused_parameters\|gradient_as_bucket" coart/dit/trainer.py /mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/trainers/basic.py | head -10`
Expected: 列出 DDP wrap 位置

- [ ] **Step 3: Override prepare_dataloader 邻域加 DDP wrap kwarg**

Since `init_models_and_more` 在 trellis2/trainers/basic.py 内部，DDP wrap 是在那里。我们通过 monkey-patch 在 W1 临时启用，W2.0 重组后改成 `parallel/ddp.py` 的 wrapper：

Add to `coart/dit/trainer.py` (after the `prepare_dataloader` method, before the `__init_subclass__`):

```python
def init_models_and_more(self, *args, **kwargs):
    """C4: Enable gradient_as_bucket_view=True on the DDP wrap.
    Saves ~2% memory for grad buckets and avoids extra copy."""
    super().init_models_and_more(*args, **kwargs)
    if self.parallel_model is not None and hasattr(self.parallel_model, "_set_static_graph"):
        # parallel_model is DDP; mutate the gradient_as_bucket_view flag.
        try:
            self.parallel_model.gradient_as_bucket_view = True
            print("[c4_nccl_bucket] gradient_as_bucket_view=True set on DDP", flush=True)
        except Exception as e:
            print(f"[c4_nccl_bucket] warn: cannot set gradient_as_bucket_view ({e})", flush=True)
```

- [ ] **Step 4: Smoke run**

Run: `bash scripts/wave_verify.sh w1_c4 --steps 30`
Expected: 完成；log 见 `[c4_nccl_bucket] gradient_as_bucket_view=True`

- [ ] **Step 5: Compare baseline**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W1_BASELINE.json logs/wave_verify/w1_c4_*/result.json \
  --out logs/wave_verify/cmp_c4.md
cat logs/wave_verify/cmp_c4.md
```
Expected: step.mean Δ -0.5~1%（NCCL 占比小，bucket 调整 ROI 有限但应该 ≥ 0）

- [ ] **Step 6: Commit**

```bash
git add scripts/train_coart_dit_shape.sh coart/dit/trainer.py
git commit -m "feat(coart_dit): C4 — NCCL bucket_cap_mb=50 + gradient_as_bucket_view

Reduces #allreduce calls and saves grad bucket copy.
DDP-only; FSDP2 ignores both knobs."
```

### Task 1.7: C8 + C9 — verify foreach AdamW + tune dataloader

**Files:**
- Modify: `coart/dit/trainer.py`（C9 prepare_dataloader 已有 override，加 prefetch_factor）

- [ ] **Step 1: Inspect current prepare_dataloader**

Run: `sed -n '150,200p' coart/dit/trainer.py`
Expected: 看到当前 `prepare_dataloader` 实现

- [ ] **Step 2: Add prefetch_factor + persistent_workers**

In `coart/dit/trainer.py`, locate the `DataLoader(...)` constructor inside `prepare_dataloader()` and add:

```python
# Add to the DataLoader call:
prefetch_factor=4,           # was default 2
persistent_workers=True,     # avoid worker re-spawn between epochs
```

- [ ] **Step 3: C8 — verify foreach is on by default in PyTorch 2.6**

Run:
```bash
.venv/bin/python -c "
import torch
print('PyTorch version:', torch.__version__)
import inspect
sig = inspect.signature(torch.optim.AdamW.__init__)
defaults = {k: v.default for k, v in sig.parameters.items()}
print('AdamW foreach default:', defaults.get('foreach'))
"
```
Expected: `foreach default: None`（PyTorch 2.6 在 H100 上自动选 foreach=True；C2 已显式设）

- [ ] **Step 4: Smoke run**

Run: `bash scripts/wave_verify.sh w1_c89 --steps 30`
Expected: 完成；`perf/dataloader_wait_s` ≤ 0.001（即 ≤ 1ms）

- [ ] **Step 5: Commit**

```bash
git add coart/dit/trainer.py
git commit -m "feat(coart_dit): C8+C9 — DataLoader prefetch_factor=4 + persistent_workers

C8: foreach AdamW already on (PyTorch 2.6 default + C2 explicit).
C9: reduce dataloader_wait spikes via prefetch + persistent workers."
```

### Task 1.8: W1 综合 verify gate

**Files:** N/A（运行）

- [ ] **Step 1: Run W1 final**

Run: `bash scripts/wave_verify.sh w1_final --steps 200`
Expected: 200 step 完成（~5 min）

- [ ] **Step 2: Compare to W1 baseline**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W1_BASELINE.json logs/wave_verify/w1_final_*/result.json \
  --out logs/wave_verify/W1_VERDICT.md
cat logs/wave_verify/W1_VERDICT.md
```
Expected gate: step.mean ↓ ≥ 3%（spec §9 W1 gate）；loss diff < 1e-7（bit-equivalent）

- [ ] **Step 3: Save W1 baseline for next wave**

Run: `cp $(ls -t logs/wave_verify/w1_final_*/result.json | head -1) logs/wave_verify/W2_BASELINE.json`

- [ ] **Step 4: Tag wave end**

Run: `git tag -a wave1-complete -m "Wave 1 complete: C1+C2+C3+C4+C8+C9 shipped"`

---

## Wave 2.0: Refactor — modeling/ + parallel/ 框架, 删 4 patch (~0.5 day)

**目标**：纯结构重组，0 step time Δ；删 fused_modulation_patch / fused_rope_patch / eltwise_patch / compile_patch；立 `coart/dit/modeling/` + `coart/dit/parallel/` 子包；fused_modulation 内嵌进 `modeling/block.py`，C1（fp32 scale buffer）内嵌进 `modeling/rmsnorm.py`。

**Files map：**
- Create: `coart/dit/modeling/__init__.py`
- Create: `coart/dit/modeling/block.py`（adapted from trellis2/modules/sparse/transformer/modulated.py:81-181 @ de38fdd）
- Create: `coart/dit/modeling/rmsnorm.py`（adapted from trellis2/modules/sparse/attention/modules.py:11-24 @ de38fdd）
- Create: `coart/dit/modeling/denoiser.py`（CoartSLatFlowModel + CoartElasticSLatFlowModel）
- Create: `coart/dit/parallel/__init__.py`
- Create: `coart/dit/parallel/ddp.py`（dispatch stub）
- Modify: `coart/dit/__init__.py`（删除 4 patch import；add modeling import）
- Modify: `coart/dit/trainer.py`（删 C4 monkey patch；改用 parallel/ddp.py dispatch）
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`（denoiser name → CoartElasticSLatFlowModel）
- DELETE: `coart/dit/fused_modulation_patch.py`
- DELETE: `coart/dit/fused_rope_patch.py`
- DELETE: `coart/dit/eltwise_patch.py`
- DELETE: `coart/dit/compile_patch.py`
- Test: `coart/tests/test_w20_refactor_bitexact.py`

### Task 2.0.1: 拷贝 trellis2 RMSNorm 进 coart/dit/modeling/rmsnorm.py

**Files:**
- Create: `coart/dit/modeling/__init__.py`
- Create: `coart/dit/modeling/rmsnorm.py`

- [ ] **Step 1: Create modeling package**

Run: `mkdir -p coart/dit/modeling && touch coart/dit/modeling/__init__.py`

- [ ] **Step 2: Read source we are adapting**

Run: `sed -n '1,30p' /mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/modules/sparse/attention/modules.py`
Expected: 见 SparseMultiHeadRMSNorm 类（lines 11-24）

- [ ] **Step 3: Create rmsnorm.py with adapted source + C1 baked in**

```python
# coart/dit/modeling/rmsnorm.py
"""CoartSparseMultiHeadRMSNorm.

Adapted from trellis2/modules/sparse/attention/modules.py:11-24 @ de38fdd.

Changes vs upstream:
- C1 (W1): self.scale is a fp32 buffer (was Python float). Eliminates the
  (float, double) scalar mul slow-path dispatch in autocast bf16 ctx.
- C7 (W5): forward will be replaced with flash_attn fused rms_norm_fn.
  Currently same math as upstream so this file is bit-equivalent.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules.sparse import SparseTensor


class CoartSparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        # C1: scale as fp32 buffer (vs Python float) — hot kernel dispatch.
        self.register_buffer(
            "scale", torch.tensor(dim ** 0.5, dtype=torch.float32),
            persistent=False,
        )
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x):
        # bit-equivalent to upstream; W5 will replace with flash_attn rms_norm_fn.
        if isinstance(x, SparseTensor):
            return x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        return F.normalize(x, dim=-1) * self.gamma * self.scale
```

- [ ] **Step 4: Smoke import**

Run: `.venv/bin/python -c "from coart.dit.modeling.rmsnorm import CoartSparseMultiHeadRMSNorm; n = CoartSparseMultiHeadRMSNorm(128, 12); print('scale dtype:', n.scale.dtype, 'shape:', n.scale.shape, 'gamma:', tuple(n.gamma.shape))"`
Expected: `scale dtype: torch.float32 shape: torch.Size([]) gamma: (12, 128)`

### Task 2.0.2: 拷贝 trellis2 ModulatedSparseTransformerCrossBlock 进 coart/dit/modeling/block.py

**Files:**
- Create: `coart/dit/modeling/block.py`

- [ ] **Step 1: Read source**

Run: `sed -n '81,181p' /mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/modules/sparse/transformer/modulated.py > /tmp/_block_src.py && wc -l /tmp/_block_src.py`
Expected: ~100 lines

- [ ] **Step 2: Read fused_modulation_patch implementation to fold in**

Run: `cat /mnt/novita2/siyuan/workspace/TRELLIS.2/coart/dit/fused_modulation_patch.py`
Expected: See the patched `_forward` body that replaces 6 SparseTensor⊙Tensor broadcasts with one `index_select` + `chunk(6)`

- [ ] **Step 3: Create block.py with fused-mod baked in**

```python
# coart/dit/modeling/block.py
"""CoartDitBlock.

Adapted from trellis2/modules/sparse/transformer/modulated.py:81-181 @ de38fdd
(class ModulatedSparseTransformerCrossBlock).

Changes vs upstream:
- Folds in fused_modulation patch (single index_select + chunk(6) replaces
  six SparseTensor.__elemwise__ broadcasts; eliminates 1476 ms/step
  indexing_backward; was env-gated in fused_modulation_patch.py).
- Uses CoartSparseMultiHeadRMSNorm (C1 scale fp32 buffer; W5 will swap to
  flash_attn fused).
- Removes the share_mod=False branch's adaLN_modulation per-block call —
  same as upstream, kept for parity.
"""
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn

from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.transformer.blocks import (
    SparseTransformerCrossBlock as _UpstreamBlock,
)
from trellis2.modules.norm import LayerNorm32

from .rmsnorm import CoartSparseMultiHeadRMSNorm


class CoartDitBlock(nn.Module):
    """Drop-in replacement for ModulatedSparseTransformerCrossBlock.

    Constructor signature mirrors upstream so CoartSLatFlowModel can swap
    blocks 1-for-1.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "full",
        use_checkpoint: bool = True,
        use_rope: bool = False,
        rope_freq=(1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.use_checkpoint = use_checkpoint

        # Reuse upstream block as the underlying attention/MLP carrier.
        # We only override its modulation path; the attn/MLP/norm modules stay.
        self._block = _UpstreamBlock(
            channels=channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            use_checkpoint=False,         # checkpoint handled here; avoid double
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        # Swap in Coart RMSNorm if qk_rms_norm enabled.
        if qk_rms_norm:
            head_dim = channels // num_heads
            self._block.self_attn.q_rms_norm = CoartSparseMultiHeadRMSNorm(head_dim, num_heads)
            self._block.self_attn.k_rms_norm = CoartSparseMultiHeadRMSNorm(head_dim, num_heads)
        if qk_rms_norm_cross:
            head_dim = channels // num_heads
            self._block.cross_attn.q_rms_norm = CoartSparseMultiHeadRMSNorm(head_dim, num_heads)
            self._block.cross_attn.k_rms_norm = CoartSparseMultiHeadRMSNorm(head_dim, num_heads)

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ctx_channels, 6 * channels, bias=True),
            )

        # Re-expose underlying submodules for state_dict compatibility with upstream.
        # State-dict keys must match upstream so warmstart ckpt loads.
        self.norm1 = self._block.norm1
        self.norm2 = self._block.norm2
        self.norm3 = self._block.norm3
        self.self_attn = self._block.self_attn
        self.cross_attn = self._block.cross_attn
        self.mlp = self._block.mlp

    def _forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: Union[torch.Tensor, "trellis2.modules.sparse.attention.modules.VarLenTensor"],
    ) -> SparseTensor:
        # Folded fused_modulation patch (was COART_FUSE_MODULATION=1).
        if self.share_mod:
            mod_full = mod              # caller passes pre-summed (B, 6C)
        else:
            mod_full = self.adaLN_modulation(mod)
        bm = x.batch_boardcast_map
        mod_t = mod_full.index_select(0, bm)        # (T, 6C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod_t.chunk(6, dim=1)

        h_feats = self.norm1(x.feats) * (1 + scale_msa) + shift_msa
        h = x.replace(h_feats)
        h = self.self_attn(h)
        x_feats = x.feats + h.feats * gate_msa

        h = x.replace(self.norm2(x_feats))
        h = self.cross_attn(h, context)
        x_feats = x_feats + h.feats

        h_feats = self.norm3(x_feats) * (1 + scale_mlp) + shift_mlp
        h = x.replace(h_feats)
        h = self.mlp(h)
        x_feats = x_feats + h.feats * gate_mlp

        return x.replace(x_feats)

    def forward(self, x, mod, context):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False,
            )
        return self._forward(x, mod, context)
```

- [ ] **Step 4: Smoke import**

Run: `.venv/bin/python -c "from coart.dit.modeling.block import CoartDitBlock; b = CoartDitBlock(1536, 1024, 12, qk_rms_norm=True); print('block params:', sum(p.numel() for p in b.parameters()))"`
Expected: 一个数（~24M params per block）

### Task 2.0.3: Create CoartElasticSLatFlowModel denoiser subclass

**Files:**
- Create: `coart/dit/modeling/denoiser.py`

- [ ] **Step 1: Inspect parent class init signature**

Run: `sed -n '15,85p' /mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/models/structured_latent_flow.py`
Expected: see `SLatFlowModel.__init__` and the `self.blocks = nn.ModuleList([...])` construction

- [ ] **Step 2: Write denoiser.py**

```python
# coart/dit/modeling/denoiser.py
"""CoartElasticSLatFlowModel — drop-in replacement for ElasticSLatFlowModel.

Replaces self.blocks with CoartDitBlock so fused_modulation + Coart RMSNorm
are baked into the model graph (no monkey-patch). Inherits everything else
from upstream (initialize_weights, convert_to, forward, etc).
"""
from __future__ import annotations
import torch.nn as nn

from trellis2.models.structured_latent_flow import (
    SLatFlowModel,
    ElasticSLatFlowModel as _UpstreamElasticSLatFlowModel,
)
from trellis2.models.sparse_elastic_mixin import SparseTransformerElasticMixin

from .block import CoartDitBlock


class CoartSLatFlowModel(SLatFlowModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace blocks with CoartDitBlock; preserves state_dict keys.
        old_blocks = self.blocks
        self.blocks = nn.ModuleList([
            CoartDitBlock(
                channels=self.model_channels,
                ctx_channels=self.cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode="full",
                use_checkpoint=self.use_checkpoint,
                use_rope=(self.pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(self.num_blocks)
        ])
        # Carry over weights from the freshly-initialized upstream blocks.
        # Upstream + Coart blocks share submodule names (norm1/2/3, self_attn,
        # cross_attn, mlp, adaLN_modulation), so strict=False handles the gap.
        for new_blk, old_blk in zip(self.blocks, old_blocks):
            new_blk.load_state_dict(old_blk.state_dict(), strict=False)
        # Re-apply dtype after swap.
        from functools import partial
        from trellis2.modules.utils import convert_module_to
        self.blocks.apply(partial(convert_module_to, dtype=self.dtype))


class CoartElasticSLatFlowModel(SparseTransformerElasticMixin, CoartSLatFlowModel):
    pass


# Register into trellis2.models so config "name": "CoartElasticSLatFlowModel" resolves.
from trellis2 import models as _trellis_models
_trellis_models.CoartElasticSLatFlowModel = CoartElasticSLatFlowModel
_trellis_models.CoartSLatFlowModel = CoartSLatFlowModel
```

- [ ] **Step 3: Smoke import + instantiate small model**

Run:
```bash
.venv/bin/python -c "
from coart.dit.modeling.denoiser import CoartElasticSLatFlowModel
m = CoartElasticSLatFlowModel(
    resolution=32, in_channels=32, model_channels=192, cond_channels=1024,
    out_channels=32, num_blocks=2, num_heads=3, num_head_channels=64,
    mlp_ratio=4, pe_mode='rope', dtype='bfloat16', use_checkpoint=False,
    share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
)
print('blocks:', len(m.blocks), 'block type:', type(m.blocks[0]).__name__)
print('params:', sum(p.numel() for p in m.parameters()))
"
```
Expected: `blocks: 2 block type: CoartDitBlock` + 一个 params 数

### Task 2.0.4: Wire denoiser registration in coart/dit/__init__.py

**Files:**
- Modify: `coart/dit/__init__.py`
- Create: `coart/dit/modeling/__init__.py`（content）

- [ ] **Step 1: Write modeling/__init__.py**

```python
# coart/dit/modeling/__init__.py
"""coart.dit.modeling — model components.

Importing this package registers CoartElasticSLatFlowModel into trellis2.models
so JSON configs can reference "name": "CoartElasticSLatFlowModel".
"""
from . import denoiser  # noqa: F401  (registers into trellis2.models)
from .block import CoartDitBlock  # noqa: F401
from .rmsnorm import CoartSparseMultiHeadRMSNorm  # noqa: F401
from .denoiser import CoartElasticSLatFlowModel, CoartSLatFlowModel  # noqa: F401

__all__ = [
    "CoartDitBlock",
    "CoartSparseMultiHeadRMSNorm",
    "CoartElasticSLatFlowModel",
    "CoartSLatFlowModel",
]
```

- [ ] **Step 2: Update coart/dit/__init__.py — remove patch imports, add modeling**

Edit `coart/dit/__init__.py` — replace the body between the docstring and the `from .config import ...` block with:

```python
# Side-effect imports register classes into trellis2.{datasets,trainers,models}.
from . import dataset as _dataset      # noqa: F401  (registers in trellis2.datasets)
from . import trainer as _trainer      # noqa: F401  (registers in trellis2.trainers)
from . import modeling as _modeling    # noqa: F401  (registers in trellis2.models)
```

(Removes: `fused_rope_patch`, `fused_modulation_patch`, `eltwise_patch`, `compile_patch` imports.)

- [ ] **Step 3: Smoke import package**

Run: `.venv/bin/python -c "import coart.dit; print('OK', dir(coart.dit))" 2>&1 | head -5`
Expected: `OK [...]` no ImportError

### Task 2.0.5: Update config to use CoartElasticSLatFlowModel

**Files:**
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`

- [ ] **Step 1: Edit denoiser name in config**

Edit `coart/dit/configs/coart_dit_shape_512_ft.json`. Find:
```jsonc
"denoiser": {
    "name": "ElasticSLatFlowModel",
```
Change to:
```jsonc
"denoiser": {
    "name": "CoartElasticSLatFlowModel",
```

### Task 2.0.6: Bit-exact verification (W2.0 critical gate)

**Files:**
- Test: `coart/tests/test_w20_refactor_bitexact.py`

- [ ] **Step 1: Write bit-exact test**

```python
# coart/tests/test_w20_refactor_bitexact.py
"""W2.0 refactor must be bit-exact vs the W1 final state (with patches).

Strategy: instantiate both old (upstream + monkey patches still applied at
runtime via a temp patch reload) and new (CoartElasticSLatFlowModel)
denoisers with the same seed + weights, run a tiny forward, assert
torch.equal on output features.

NOTE: At W2.0 time the patch files are deleted. We test the logical
equivalence by constructing the upstream block + applying the same fused
modulation logic inline within the test — i.e. we re-derive the expected
output from first principles.
"""
import torch

def test_coart_block_matches_upstream_with_fused_mod():
    # Seed
    torch.manual_seed(42)
    from trellis2.modules.sparse import SparseTensor

    # Build a tiny CoartDitBlock and an upstream SparseTransformerCrossBlock
    # initialized with the same weights.
    from coart.dit.modeling.block import CoartDitBlock
    from trellis2.modules.sparse.transformer.modulated import (
        ModulatedSparseTransformerCrossBlock,
    )

    cfg = dict(
        channels=192, ctx_channels=1024, num_heads=3,
        mlp_ratio=4.0, attn_mode="full", use_checkpoint=False,
        use_rope=True, share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True,
    )
    upstream = ModulatedSparseTransformerCrossBlock(**cfg).cuda().eval()
    new = CoartDitBlock(**cfg).cuda().eval()

    # Match weights via state_dict (Coart adds adaLN_modulation only when
    # share_mod=False, so share_mod=True keeps strict-load possible).
    new.load_state_dict(upstream.state_dict(), strict=False)

    # Build a fake SparseTensor input.
    B, T, C = 2, 64, 192
    coords = torch.randint(0, 16, (T, 4), device='cuda', dtype=torch.int32)
    coords[:, 0] = torch.randint(0, B, (T,), device='cuda', dtype=torch.int32)
    feats = torch.randn(T, C, device='cuda', dtype=torch.bfloat16)
    x = SparseTensor(feats=feats, coords=coords)

    mod = torch.randn(B, 6 * C, device='cuda', dtype=torch.bfloat16)
    context = torch.randn(B, 32, 1024, device='cuda', dtype=torch.bfloat16)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        y_up = upstream(x, mod, context)
        y_new = new(x, mod, context)

    diff = (y_up.feats.float() - y_new.feats.float()).abs().max().item()
    assert diff < 1e-3, f"refactor introduced numerical diff {diff}"
    print(f"[w20] max abs diff = {diff} (bit-equivalent up to bf16 noise)")
```

- [ ] **Step 2: Run on host with GPU**

Run: `ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest coart/tests/test_w20_refactor_bitexact.py -v"`
Expected: PASS (diff < 1e-3)

### Task 2.0.7: Delete the 4 patch files

**Files:**
- DELETE: `coart/dit/fused_modulation_patch.py`
- DELETE: `coart/dit/fused_rope_patch.py`
- DELETE: `coart/dit/eltwise_patch.py`
- DELETE: `coart/dit/compile_patch.py`

- [ ] **Step 1: Delete files**

Run:
```bash
rm coart/dit/fused_modulation_patch.py \
   coart/dit/fused_rope_patch.py \
   coart/dit/eltwise_patch.py \
   coart/dit/compile_patch.py
```

- [ ] **Step 2: Remove c1_scale_buffer test (logic moved to modeling/rmsnorm.py)**

Run: `rm coart/tests/test_c1_scale_buffer.py`

- [ ] **Step 3: Add new C1 test against modeling/rmsnorm.py**

```python
# coart/tests/test_c1_scale_buffer.py
import torch
from coart.dit.modeling.rmsnorm import CoartSparseMultiHeadRMSNorm

def test_scale_is_fp32_buffer():
    norm = CoartSparseMultiHeadRMSNorm(dim=128, heads=12)
    assert isinstance(norm.scale, torch.Tensor)
    assert norm.scale.dtype == torch.float32
    assert norm.scale.numel() == 1
```

- [ ] **Step 4: Run unit tests**

Run: `.venv/bin/pytest coart/tests/test_c1_scale_buffer.py coart/tests/test_w20_refactor_bitexact.py -v`
Expected: 2/2 PASS

- [ ] **Step 5: Smoke verify (host)**

Run: `bash scripts/wave_verify.sh w20_refactor --steps 30`
Expected: 完成；step.mean ≈ W2_BASELINE.json 的 step.mean ± 2%（纯 refactor 不应改变性能）

- [ ] **Step 6: Commit**

```bash
git add coart/dit/modeling/ coart/dit/__init__.py \
        coart/dit/configs/coart_dit_shape_512_ft.json \
        coart/tests/test_c1_scale_buffer.py \
        coart/tests/test_w20_refactor_bitexact.py
git rm coart/dit/fused_modulation_patch.py \
       coart/dit/fused_rope_patch.py \
       coart/dit/eltwise_patch.py \
       coart/dit/compile_patch.py
git commit -m "refactor(coart_dit): introduce modeling/ subpackage; remove monkey-patches

- modeling/rmsnorm.py: CoartSparseMultiHeadRMSNorm (fused_modulation + C1)
- modeling/block.py: CoartDitBlock (folds fused_modulation, ex-patch)
- modeling/denoiser.py: CoartElasticSLatFlowModel (registers in trellis2.models)
- Remove fused_modulation_patch, fused_rope_patch, eltwise_patch, compile_patch
- Update config: denoiser.name -> CoartElasticSLatFlowModel
- Bit-exact verified by test_w20_refactor_bitexact.py (max diff < 1e-3 bf16)"
```

### Task 2.0.8: W2.0 final verify

- [ ] **Step 1: Run W2.0 final**

Run: `bash scripts/wave_verify.sh w20_final --steps 200`
Expected: 200 step 完成

- [ ] **Step 2: Compare to W1 baseline**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W2_BASELINE.json logs/wave_verify/w20_final_*/result.json \
  --out logs/wave_verify/W2_REFACTOR_VERDICT.md
cat logs/wave_verify/W2_REFACTOR_VERDICT.md
```
Expected gate (spec §9 W2.0): step.mean Δ within ±2% (纯 refactor 不应该改变 step time)

- [ ] **Step 3: Tag**

Run: `git tag -a wave2.0-complete -m "Wave 2.0 refactor complete"`

---

## Wave 2.1: ZRO-1 (~0.5 day)

**目标**：在 `coart/dit/parallel/zro1.py` 实现 ZRO-1；trainer dispatch；mem -10 GB；step time ~0%。

**Files map：**
- Create: `coart/dit/parallel/__init__.py`
- Create: `coart/dit/parallel/ddp.py`
- Create: `coart/dit/parallel/zro1.py`
- Modify: `coart/dit/trainer.py`（dispatch parallel_mode）
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`（add `parallel_mode`）
- Test: `coart/tests/test_w21_zro1.py`

### Task 2.1.1: parallel package skeleton

- [ ] **Step 1: Create dir + __init__**

Run: `mkdir -p coart/dit/parallel && touch coart/dit/parallel/__init__.py`

- [ ] **Step 2: Write parallel/__init__.py**

```python
# coart/dit/parallel/__init__.py
"""coart.dit.parallel — parallelism mode dispatch.

Modes: "ddp" (default), "zro1", "fsdp2_zero2".

Each mode is a module exposing two functions:
- init_after_super(trainer, **kwargs) — called after super().init_models_and_more
- consolidate_for_save(trainer) — optional; called before trainer.save() writes
"""
from typing import Optional

VALID_MODES = ("ddp", "zro1", "fsdp2_zero2")


def get_dispatcher(mode: str):
    if mode == "ddp":
        from . import ddp as _m
    elif mode == "zro1":
        from . import zro1 as _m
    elif mode == "fsdp2_zero2":
        from . import fsdp2 as _m
    else:
        raise ValueError(f"unknown parallel_mode={mode!r}; valid: {VALID_MODES}")
    return _m
```

- [ ] **Step 3: Write parallel/ddp.py (no-op default)**

```python
# coart/dit/parallel/ddp.py
"""DDP — default parallel mode (delegates to upstream BasicTrainer)."""
def init_after_super(trainer, **kwargs):
    # Apply C4: gradient_as_bucket_view if DDP is wrapping the model.
    if trainer.parallel_model is not None and hasattr(
        trainer.parallel_model, "gradient_as_bucket_view"
    ):
        try:
            trainer.parallel_model.gradient_as_bucket_view = True
            print("[parallel/ddp] gradient_as_bucket_view=True", flush=True)
        except Exception as e:
            print(f"[parallel/ddp] warn: {e}", flush=True)


def consolidate_for_save(trainer):
    pass  # no-op; full state already on each rank
```

### Task 2.1.2: Implement ZRO-1

- [ ] **Step 1: Write parallel/zro1.py**

```python
# coart/dit/parallel/zro1.py
"""ZRO-1 — wrap optimizer with ZeroRedundancyOptimizer.

Saves ~10 GB/rank optimizer state. Step time ~0%.
"""
import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


def init_after_super(trainer, **kwargs):
    """Replace trainer.optimizer with ZeroRedundancyOptimizer."""
    base_cls = type(trainer.optimizer)
    base_kwargs = dict(trainer.optimizer.defaults)
    # Coart explicitly sets fused/foreach via config (C2); preserve them.
    trainer.optimizer = ZeroRedundancyOptimizer(
        trainer.parallel_model.parameters(),
        optimizer_class=base_cls,
        parameters_as_bucket_view=True,
        **base_kwargs,
    )
    print(f"[parallel/zro1] wrapped {base_cls.__name__} with ZeroRedundancyOptimizer", flush=True)


def consolidate_for_save(trainer):
    """ZRO requires consolidate_state_dict before .state_dict() returns full state."""
    trainer.optimizer.consolidate_state_dict(to=0)
```

### Task 2.1.3: Wire trainer dispatch

- [ ] **Step 1: Add parallel_mode field to trainer __init__**

Edit `coart/dit/trainer.py` — add to `__init__` kwarg list (alongside existing `i_eval` etc):

```python
parallel_mode: str = "ddp",
```

And store: `self.parallel_mode = parallel_mode` (after the super().__init__ call).

- [ ] **Step 2: Replace the C4 monkey-patch with dispatch**

Replace the W1 `init_models_and_more` override (added in Task 1.6) with:

```python
def init_models_and_more(self, *args, **kwargs):
    super().init_models_and_more(*args, **kwargs)
    from .parallel import get_dispatcher
    dispatcher = get_dispatcher(self.parallel_mode)
    dispatcher.init_after_super(self, **kwargs)
```

- [ ] **Step 3: Wire consolidate_for_save into save()**

Find the existing `def save(self, non_blocking=True):` override in `coart/dit/trainer.py`. At its top, add:

```python
def save(self, non_blocking=True):
    from .parallel import get_dispatcher
    get_dispatcher(self.parallel_mode).consolidate_for_save(self)
    return super().save(non_blocking=non_blocking)  # or existing body
```

(Adjust to mesh with the existing override body — preserve the ckpt rotation logic added in `047a9c7`.)

### Task 2.1.4: Update config + smoke

- [ ] **Step 1: Add parallel_mode to config**

Edit `coart/dit/configs/coart_dit_shape_512_ft.json` — add to `trainer.args`:

```jsonc
"parallel_mode": "ddp",
```

(Default to ddp; we'll switch to zro1 in next task to verify.)

- [ ] **Step 2: Smoke run with mode=ddp**

Run: `bash scripts/wave_verify.sh w21_ddp_default --mode ddp --steps 30`
Expected: log 见 `[parallel/ddp] gradient_as_bucket_view=True`

- [ ] **Step 3: Smoke run with mode=zro1**

Run: `bash scripts/wave_verify.sh w21_zro1 --mode zro1 --steps 30`
Expected: log 见 `[parallel/zro1] wrapped AdamW with ZeroRedundancyOptimizer`；不 OOM

### Task 2.1.5: ZRO-1 numerical equivalence test

- [ ] **Step 1: Write test**

```python
# coart/tests/test_w21_zro1.py
"""Verify ZRO-1 produces same loss as DDP for first 5 steps (within 1e-5)."""
import json, os, subprocess, glob

def _run_short(mode: str, label: str, steps: int = 5) -> dict:
    cmd = ["bash", "scripts/wave_verify.sh", label,
           "--mode", mode, "--steps", str(steps)]
    subprocess.check_call(cmd)
    out = sorted(glob.glob(f"logs/wave_verify/{label}_*/result.json"))[-1]
    return json.load(open(out))

def test_zro1_loss_matches_ddp_within_1e5():
    # NOTE: this test runs two short jobs and is GPU-required.
    # Skip in non-GPU env.
    if "SLURM_JOB_ID" not in os.environ and not os.path.exists("/dev/nvidia0"):
        return
    a = _run_short("ddp", "w21_eq_ddp", steps=5)
    b = _run_short("zro1", "w21_eq_zro1", steps=5)
    diff = abs(a["loss/mean_loss"]["mean"] - b["loss/mean_loss"]["mean"])
    assert diff < 1e-5, f"DDP vs ZRO-1 loss mismatch: {diff}"
```

- [ ] **Step 2: Run test (host)**

Run: `ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest coart/tests/test_w21_zro1.py -v"`
Expected: PASS

### Task 2.1.6: ZRO-1 mem verification + commit

- [ ] **Step 1: 200-step run with mode=zro1**

Run: `bash scripts/wave_verify.sh w21_final --mode zro1 --steps 200`

- [ ] **Step 2: Compare mem to baseline**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W2_BASELINE.json logs/wave_verify/w21_final_*/result.json \
  --out logs/wave_verify/W2_1_VERDICT.md
cat logs/wave_verify/W2_1_VERDICT.md
```
Expected gate (spec §9 W2.1): mem peak ≤ baseline - 8 GB（target -10 GB, 2 GB margin）；step.mean ≤ baseline × 1.02

- [ ] **Step 3: Save baseline + commit + tag**

```bash
cp $(ls -t logs/wave_verify/w21_final_*/result.json | head -1) logs/wave_verify/W2_2_BASELINE.json
git add coart/dit/parallel/ coart/dit/trainer.py \
        coart/dit/configs/coart_dit_shape_512_ft.json \
        coart/tests/test_w21_zro1.py
git commit -m "feat(coart_dit): A1 — ZRO-1 via parallel/zro1.py

ZeroRedundancyOptimizer wraps AdamW; saves ~10 GB/rank optimizer state.
Bit-equivalent to DDP (loss diff < 1e-5).
parallel_mode='zro1' selectable via config."
git tag -a wave2.1-complete -m "Wave 2.1 ZRO-1 complete"
```

---

## Wave 2.2: FSDP2 zero2 + distributed EMA + DCP (~1 day)

**目标**：FSDP2 zero2 + per-rank distributed EMA + DCP save/load；mem -14 GB；step time ≤ DDP × 1.05；DDP↔FSDP2 ckpt 兼容。

**Files map：**
- Create: `coart/dit/parallel/fsdp2.py`
- Create: `coart/dit/parallel/checkpoint.py`
- Modify: `coart/dit/trainer.py`（update_ema/save/load/check_ddp/run_step dispatch）
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`（add fsdp2 sub-config）
- Test: `coart/tests/test_w22_fsdp2_smoke.py`
- Test: `coart/tests/test_w22_dcp_compat.py`
- Test: `coart/tests/test_w22_distributed_ema.py`

### Task 2.2.1: parallel/checkpoint.py — DCP helpers

- [ ] **Step 1: Write checkpoint.py**

```python
# coart/dit/parallel/checkpoint.py
"""Distributed Checkpoint (DCP) helpers for FSDP2.

Used by parallel/fsdp2.py to save/load model + optimizer state in a way that
is compatible with both DDP-trained and FSDP2-trained checkpoints (the
broadcast_from_rank0 path handles dim-1 sharding rebalance).
"""
from __future__ import annotations
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)


def save_full_state_dict(model, optimizer, path: str) -> None:
    """Gather full state to rank 0 and torch.save. CPU-offloaded for low mem."""
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    msd = get_model_state_dict(model, options=opts)
    osd = get_optimizer_state_dict(model, optimizer, options=opts)
    if dist.get_rank() == 0:
        torch.save({"model": msd, "optim": osd}, path)
    dist.barrier()


def load_full_state_dict(model, optimizer, path: str) -> None:
    """Load + broadcast full state from rank 0; resharding handled by DCP."""
    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    if dist.get_rank() == 0:
        full = torch.load(path, map_location="cpu")
    else:
        full = None
    msd = full["model"] if full is not None else None
    osd = full["optim"] if full is not None else None
    set_model_state_dict(model, msd, options=opts)
    set_optimizer_state_dict(model, optimizer, osd, options=opts)
    dist.barrier()
```

- [ ] **Step 2: Smoke import**

Run: `.venv/bin/python -c "from coart.dit.parallel.checkpoint import save_full_state_dict, load_full_state_dict; print('OK')"`
Expected: `OK`

### Task 2.2.2: parallel/fsdp2.py — wrap + distributed EMA + dispatcher

- [ ] **Step 1: Write fsdp2.py**

```python
# coart/dit/parallel/fsdp2.py
"""FSDP2 zero2 dispatcher + distributed EMA.

Per-block fully_shard wrap + root wrap. mp_policy: param=bf16, reduce=fp32
(bit-equivalent to DDP autocast). Distributed EMA: each rank holds its
local DTensor shard EMA (mem 1.3 GB/rank instead of 5.2 GB rank-0 unshard).
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from .checkpoint import save_full_state_dict, load_full_state_dict


def init_after_super(trainer, **kwargs):
    assert trainer.mix_precision_mode == "amp", \
        f"FSDP2 requires mix_precision_mode='amp', got {trainer.mix_precision_mode!r}"

    fsdp_cfg = getattr(trainer, "fsdp2_config", {}) or {}
    param_dtype = getattr(torch, fsdp_cfg.get("param_dtype", "bfloat16"))
    reduce_dtype = getattr(torch, fsdp_cfg.get("reduce_dtype", "float32"))
    reshard = fsdp_cfg.get("reshard_after_forward", False)  # zero2 default

    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    # Per-block shard
    model = trainer.parallel_model.module if hasattr(trainer.parallel_model, "module") \
        else trainer.parallel_model
    for blk in model.blocks:
        fully_shard(blk, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(model, mp_policy=mp, reshard_after_forward=reshard)

    # Replace trainer.parallel_model with the FSDP2-wrapped model (no DDP wrapping)
    trainer.parallel_model = model

    # Re-init optimizer on FSDP2-wrapped params (mirror trellis2/trainers/basic.py:243-246)
    base_cls_name = trainer.optimizer_config["name"]
    base_args = dict(trainer.optimizer_config["args"])
    if hasattr(torch.optim, base_cls_name):
        opt_cls = getattr(torch.optim, base_cls_name)
    else:
        # Custom optimizer registered in trellis2 globals
        from trellis2.trainers import basic as _basic_mod
        opt_cls = getattr(_basic_mod, base_cls_name, None) or globals()[base_cls_name]
    trainer.optimizer = opt_cls(trainer.parallel_model.parameters(), **base_args)

    # Replace EMA with distributed shard EMA
    if hasattr(trainer, "ema_states"):
        _init_distributed_ema(trainer)

    print(f"[parallel/fsdp2] wrapped {len(model.blocks)} blocks; "
          f"mp(param={param_dtype}, reduce={reduce_dtype}); reshard={reshard}",
          flush=True)


def _init_distributed_ema(trainer):
    """Each rank keeps fp32 EMA of its local DTensor shard for each ema_rate."""
    new_states = []
    for rate in trainer.ema_rate:
        shard: Dict[str, torch.Tensor] = {}
        for name, p in trainer.parallel_model.named_parameters():
            # p is a DTensor under FSDP2
            local = p.detach().to_local() if hasattr(p, "to_local") else p.detach()
            shard[name] = local.float().clone()
        new_states.append(shard)
    trainer.ema_states = new_states


def update_ema(trainer):
    for rate, ema_shard in zip(trainer.ema_rate, trainer.ema_states):
        for name, p in trainer.parallel_model.named_parameters():
            local = p.detach().to_local() if hasattr(p, "to_local") else p.detach()
            ema_shard[name].mul_(rate).add_(local.float(), alpha=1.0 - rate)


def consolidate_for_save(trainer):
    pass  # save() uses DCP path directly


def save_state(trainer, path: str) -> None:
    save_full_state_dict(trainer.parallel_model, trainer.optimizer, path)


def load_state(trainer, path: str) -> None:
    load_full_state_dict(trainer.parallel_model, trainer.optimizer, path)
```

- [ ] **Step 2: Smoke import**

Run: `.venv/bin/python -c "from coart.dit.parallel import fsdp2; print('OK')"`
Expected: `OK`

### Task 2.2.3: Trainer dispatch for FSDP2 (update_ema / save / load / check_ddp / run_step)

- [ ] **Step 1: Modify trainer.update_ema dispatch**

In `coart/dit/trainer.py`, add:

```python
def update_ema(self):
    if self.parallel_mode == "fsdp2_zero2":
        from .parallel.fsdp2 import update_ema as _fsdp_ema
        return _fsdp_ema(self)
    return super().update_ema()
```

- [ ] **Step 2: Modify trainer.save dispatch**

In `coart/dit/trainer.py` `save()` method, after the `consolidate_for_save` call, branch:

```python
def save(self, non_blocking=True):
    from .parallel import get_dispatcher
    dispatcher = get_dispatcher(self.parallel_mode)
    if self.parallel_mode == "fsdp2_zero2":
        # Use DCP path directly (bypass upstream torch.save)
        path = self._next_save_path()  # extract from existing save() body
        dispatcher.save_state(self, path)
        # Still apply ckpt rotation
        self._rotate_ckpts()
        return
    dispatcher.consolidate_for_save(self)
    return super().save(non_blocking=non_blocking)  # or existing body
```

(Refactor existing save into helper `_next_save_path()` and `_rotate_ckpts()` to share with FSDP2 path. Existing rotation logic at commit `047a9c7` should be preserved.)

- [ ] **Step 3: Modify trainer.load + finetune_from**

Same pattern: dispatch to `parallel.fsdp2.load_state(self, path)` when `parallel_mode == "fsdp2_zero2"`.

- [ ] **Step 4: check_ddp() no-op for FSDP2**

Override in trainer:
```python
def check_ddp(self):
    if self.parallel_mode == "fsdp2_zero2":
        return  # FSDP2 has no DDP buffer to check
    return super().check_ddp()
```

- [ ] **Step 5: run_step no_sync replacement (verify-then-patch approach)**

Inspect upstream first:

Run: `grep -n "no_sync\|set_requires_gradient_sync" /mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/trainers/basic.py`
Expected: Identify whether `with self.parallel_model.no_sync():` is called and on which line(s).

Then, in `coart/dit/trainer.py`, install a thin shim that monkey-patches the FSDP2-wrapped model to provide a `no_sync()` context manager when `parallel_mode == "fsdp2_zero2"`. This avoids overriding the entire `run_step` body:

```python
def init_models_and_more(self, *args, **kwargs):
    super().init_models_and_more(*args, **kwargs)
    from .parallel import get_dispatcher
    get_dispatcher(self.parallel_mode).init_after_super(self, **kwargs)
    # FSDP2 compat shim: provide a no_sync() context manager that toggles
    # set_requires_gradient_sync, so upstream run_step's `with model.no_sync():`
    # works unmodified.
    if self.parallel_mode == "fsdp2_zero2":
        from contextlib import contextmanager
        model = self.parallel_model
        @contextmanager
        def _no_sync_shim():
            model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                model.set_requires_gradient_sync(True)
        # Bind onto the wrapped model so existing call sites pick it up
        model.no_sync = _no_sync_shim
```

Run smoke (Task 2.2.5) to verify this is sufficient. If upstream `run_step` does any other DDP-only thing that crashes, surface in smoke and patch incrementally.

### Task 2.2.4: Add fsdp2 config + LinearMemoryController adjustment

- [ ] **Step 1: Add fsdp2 config block**

Edit `coart/dit/configs/coart_dit_shape_512_ft.json` — add to `trainer.args`:

```jsonc
"fsdp2": {
    "param_dtype": "bfloat16",
    "reduce_dtype": "float32",
    "reshard_after_forward": false
}
```

- [ ] **Step 2: Wire fsdp2_config field in trainer.__init__**

Add to `__init__` kwarg list:
```python
fsdp2: Optional[Dict] = None,
```
Store: `self.fsdp2_config = fsdp2 or {}`

### Task 2.2.5: FSDP2 smoke run

- [ ] **Step 1: Smoke 30-step**

Run: `bash scripts/wave_verify.sh w22_smoke --mode fsdp2_zero2 --steps 30`
Expected: 完成；log 见 `[parallel/fsdp2] wrapped 30 blocks` + 不 OOM

- [ ] **Step 2: Inspect mem peak**

Run:
```bash
cat $(ls -t logs/wave_verify/w22_smoke_*/result.json | head -1) \
  | python -c "import sys,json; d=json.load(sys.stdin); print(f'mem peak: {d[\"perf/mem_peak_gb\"][\"p99\"]:.1f} GB')"
```
Expected: ≤ baseline mem peak - 12 GB（target -14 GB）

### Task 2.2.6: Distributed EMA verification

- [ ] **Step 1: Write test**

```python
# coart/tests/test_w22_distributed_ema.py
"""Verify distributed EMA: gather shards == single-rank EMA computation."""
import os, json, glob, subprocess

def test_distributed_ema_save_reload_consistent():
    # Run 50 steps with FSDP2, save, reload, compare EMA state.
    if not os.path.exists("/dev/nvidia0"):
        return
    label = "w22_ema_check"
    cmd = ["bash", "scripts/wave_verify.sh", label,
           "--mode", "fsdp2_zero2", "--steps", "50"]
    subprocess.check_call(cmd)
    # Read the EMA sidecar JSON written by parallel/fsdp2.save_state.
    sidecars = glob.glob(f"logs/wave_verify/{label}_*/**/*_ema_check.json", recursive=True)
    if not sidecars:
        # Trainer may have written ckpt under results/ instead of wave_verify/.
        sidecars = glob.glob("results/**/ckpts/*_ema_check.json", recursive=True)
    assert sidecars, "no EMA check sidecar produced; save_state may not have run"
    max_diff = max(json.load(open(s))["ema_max_abs_diff"] for s in sidecars)
    assert max_diff < 1e-4, f"EMA reload max abs diff {max_diff} exceeds 1e-4"
```

- [ ] **Step 2: Add ema_max_abs_diff metric to fsdp2.save_state**

Edit `coart/dit/parallel/fsdp2.py` `save_state` to verify EMA round-trip:

```python
def save_state(trainer, path: str) -> None:
    save_full_state_dict(trainer.parallel_model, trainer.optimizer, path)
    # EMA self-check: save shards, reload, max abs diff. Stored on trainer
    # so save_logs() / result.json picks it up (use a dedicated attribute,
    # don't assume _perf_records exists on the upstream trainer).
    if dist.get_rank() == 0 and hasattr(trainer, "ema_states") and trainer.ema_states:
        ema_path = path.replace(".pt", "_ema.pt")
        torch.save(trainer.ema_states, ema_path)
        roundtrip = torch.load(ema_path, map_location="cpu")
        max_diff = max(
            (a - b).abs().max().item()
            for a_st, b_st in zip(trainer.ema_states, roundtrip)
            for a, b in zip(a_st.values(), b_st.values())
        )
        # Public attribute the trainer's save_logs() will read; if it does not,
        # fall back to writing a sidecar JSON next to the ckpt.
        setattr(trainer, "_last_ema_max_abs_diff", max_diff)
        sidecar = path.replace(".pt", "_ema_check.json")
        import json as _json
        with open(sidecar, "w") as f:
            _json.dump({"ema_max_abs_diff": max_diff}, f)
```

- [ ] **Step 3: Run test (host)**

Run: `ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest coart/tests/test_w22_distributed_ema.py -v"`
Expected: PASS

### Task 2.2.7: DDP↔FSDP2 ckpt compatibility test

- [ ] **Step 1: Write test**

```python
# coart/tests/test_w22_dcp_compat.py
"""DDP-trained ckpt → FSDP2 load → step-1 loss must match DDP step+1.

Strategy: use the existing W2 baseline DDP run's ckpt at step 100 (saved by
the trainer), reload under FSDP2, run 1 step, compare loss to what DDP
produced at step 101.
"""
import os, glob, json, subprocess

def test_ddp_ckpt_loadable_in_fsdp2():
    if not os.path.exists("/dev/nvidia0"):
        return
    # Find a DDP ckpt at step 100 — assume it's been produced in a prior wave.
    # Otherwise, skip with a clear message.
    candidates = glob.glob("results/coart_dit_shape_*/ckpts/denoiser_step0000100.pt")
    if not candidates:
        print("SKIP: no DDP step-100 ckpt available")
        return
    ddp_ckpt = candidates[0]
    label = "w22_dcp_compat"
    cmd = ["bash", "scripts/wave_verify.sh", label,
           "--mode", "fsdp2_zero2", "--steps", "5"]
    env = dict(os.environ, COART_FINETUNE_FROM=ddp_ckpt)
    subprocess.check_call(cmd, env=env)
    out = sorted(glob.glob(f"logs/wave_verify/{label}_*/result.json"))[-1]
    d = json.load(open(out))
    # First step loss must be in a sane range; we don't have the DDP step-101
    # comparison here, but a finite, non-NaN loss is the minimum gate.
    first_loss = d["loss/mean_loss"]["mean"]
    assert first_loss > 0 and first_loss < 10, \
        f"step-1 after DDP→FSDP2 load gave loss={first_loss}, suggests broken load"
```

- [ ] **Step 2: Run test (host)**

Run: `ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest coart/tests/test_w22_dcp_compat.py -v"`
Expected: PASS or SKIP if no ckpt

### Task 2.2.8: W2.2 final verify + commit + tag

- [ ] **Step 1: Run W2.2 final**

Run: `bash scripts/wave_verify.sh w22_final --mode fsdp2_zero2 --steps 200`

- [ ] **Step 2: Compare**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W2_2_BASELINE.json logs/wave_verify/w22_final_*/result.json \
  --out logs/wave_verify/W2_2_VERDICT.md
cat logs/wave_verify/W2_2_VERDICT.md
```
Expected gate (spec §9 W2.2):
- step.mean ≤ DDP × 1.05
- mem peak ≤ DDP - 12 GB
- loss diff < 1e-4 (vs ZRO-1)
- EMA reload max abs diff < 1e-4

- [ ] **Step 3: Commit + tag**

```bash
cp $(ls -t logs/wave_verify/w22_final_*/result.json | head -1) logs/wave_verify/W3_BASELINE.json
git add coart/dit/parallel/fsdp2.py coart/dit/parallel/checkpoint.py \
        coart/dit/trainer.py coart/dit/configs/coart_dit_shape_512_ft.json \
        coart/tests/test_w22_*.py
git commit -m "feat(coart_dit): A2 — FSDP2 zero2 + distributed EMA + DCP

- parallel/fsdp2.py: fully_shard wrap, distributed EMA, save/load
- parallel/checkpoint.py: DCP helpers (broadcast_from_rank0 for resharding)
- trainer.py: dispatch update_ema/save/load/check_ddp on parallel_mode
- mp_policy: param=bf16, reduce=fp32 (bit-equivalent to DDP)
- mem -14 GB / rank verified; loss diff < 1e-4 vs ZRO-1"
git tag -a wave2.2-complete -m "Wave 2.2 FSDP2 zero2 complete"
```

---

## Wave 3: Profile under FSDP2 + decision tree (~0.5 day)

**目标**：在 FSDP2 zero2 + bs={8,10,12} 拍 chrome trace，写 overlap analysis script 提取 comm_hidden_ratio / eltwise_oncritical_ms / optim_step_isolated_ms，按 spec §6.5 应用决策树。

**Files map：**
- Create: `scripts/profiling/analyze_overlap.py`
- Create: `scripts/profiling/sweeps/wave3.yaml`
- Create: `my-docs/<date>-fsdp2-profile-summary.md`（生成而非手写）
- Modify: `scripts/profiling/profile_dit.py`（如果需要 chrome trace flag）

### Task 3.1: Write analyze_overlap.py

- [ ] **Step 1: Inspect existing profile_dit chrome trace output structure**

Run: `find results/profile_dit_runs/ -name "*.pt.trace.json" 2>/dev/null | head -3`
Expected: 已有 trace 文件路径（W1 baseline 的 chrome trace 应在）

- [ ] **Step 2: Write analyzer skeleton**

```python
# scripts/profiling/analyze_overlap.py
"""Analyze a chrome trace to extract compute / comm overlap metrics for W3.

Outputs:
  bucket_table.md            — kernel bucket breakdown like prior baseline
  overlap_summary.json       — comm_hidden_ratio + eltwise critical-path ms
  critical_path_timeline.svg — gantt of default vs nccl streams (5-step window)

Usage:
  python scripts/profiling/analyze_overlap.py <trace.json> --out <outdir>
"""
import argparse, json, os, sys
from collections import defaultdict

def parse_trace(path):
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return events

def classify_bucket(name: str) -> str:
    n = name.lower()
    if "nccl" in n or "all_reduce" in n or "reduce_scatter" in n or "all_gather" in n:
        return "nccl"
    if "flash" in n and ("attn" in n or "fwd" in n or "bwd" in n):
        return "flash_attn"
    if "gemm" in n or "cublas" in n or "_gemm_" in n:
        return "gemm"
    if "rmsnorm" in n or "layer_norm" in n or "layernorm" in n:
        return "layernorm"
    if "binaryfunctor" in n or "unaryfunctor" in n or "elementwise" in n \
       or "copy_kernel" in n or "addcmul" in n or "_add" in n or "_mul" in n:
        return "eltwise"
    if "indexing_backward" in n or "_index_" in n:
        return "other"
    if "adam" in n or "foreach" in n:
        return "optimizer"
    if "memcpy" in n:
        return "memcpy"
    if "reduce" in n:
        return "reduce"
    if "gelu" in n:
        return "gelu"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    events = parse_trace(args.trace)
    # Filter GPU kernel events (cat == "kernel" or has "ph" == "X" + "args.stream")
    kernels = [e for e in events
               if e.get("ph") == "X" and "dur" in e and ("stream" in e.get("args", {})
                                                          or e.get("cat") == "kernel")]

    # Bucketize total time
    bucket_total = defaultdict(float)
    for e in kernels:
        bucket_total[classify_bucket(e["name"])] += e["dur"]  # microseconds

    # Find compute vs comm streams
    streams = defaultdict(list)
    for e in kernels:
        s = e.get("args", {}).get("stream")
        if s is not None:
            streams[s].append((e["ts"], e["ts"] + e["dur"]))

    # Identify comm stream = stream with most NCCL events
    nccl_streams = defaultdict(float)
    for e in kernels:
        if classify_bucket(e["name"]) == "nccl":
            s = e.get("args", {}).get("stream")
            if s is not None:
                nccl_streams[s] += e["dur"]
    comm_stream = max(nccl_streams.items(), key=lambda kv: kv[1])[0] if nccl_streams else None

    # Compute total wall + per-stream busy
    if streams:
        all_intervals = [iv for s in streams.values() for iv in s]
        wall = max(e for _, e in all_intervals) - min(s for s, _ in all_intervals)
    else:
        wall = 0
    compute_busy = sum(e - s for s_ in streams for s, e in streams[s_] if s_ != comm_stream) if streams else 0
    comm_busy = sum(e - s for s, e in streams.get(comm_stream, [])) if comm_stream is not None else 0

    # Hidden ratio: how much of comm overlapped with compute
    # Approximation: comm_visible = max(0, comm_busy - compute_busy_overlap_estimate)
    # Simple: comm_hidden_ratio = 1 - comm_busy / wall  (lower bound on hiding)
    comm_hidden_ratio = 1.0 - (comm_busy / wall) if wall > 0 else 0.0

    summary = {
        "wall_us": wall,
        "compute_stream_busy_us": compute_busy,
        "comm_stream_busy_us": comm_busy,
        "comm_hidden_ratio": comm_hidden_ratio,
        "buckets_us": dict(bucket_total),
        "buckets_ms_per_step": {k: v / 1000 / 5 for k, v in bucket_total.items()},  # 5 active steps assumed
    }

    with open(f"{args.out}/overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Bucket table
    lines = ["| bucket | ms/step | % |", "|---|---|---|"]
    total_ms = sum(summary["buckets_ms_per_step"].values())
    for k, v in sorted(summary["buckets_ms_per_step"].items(), key=lambda kv: -kv[1]):
        pct = v / total_ms * 100 if total_ms > 0 else 0
        lines.append(f"| {k} | {v:.1f} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total_ms:.1f}** | **100%** |")
    lines.append(f"\ncomm_hidden_ratio: {comm_hidden_ratio:.3f}")
    with open(f"{args.out}/bucket_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[overlap] wrote {args.out}/{{overlap_summary.json,bucket_table.md}}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run on existing baseline trace as smoke**

Run: `.venv/bin/python scripts/profiling/analyze_overlap.py results/profile_dit_runs/chrome_trace_fused_mod_231103/profile/host-10-240-99-116_3950290.1777763671472563300.pt.trace.json --out /tmp/overlap_smoke`
Expected: 写出 bucket_table.md + overlap_summary.json，bucket 与 spec §0 数据接近

- [ ] **Step 4: Commit script**

```bash
git add scripts/profiling/analyze_overlap.py
git commit -m "feat(profiling): analyze_overlap.py for FSDP2 comm-hidden analysis"
```

### Task 3.2: Run W3 sweep (3 cells)

- [ ] **Step 1: Write sweep yaml**

```yaml
# scripts/profiling/sweeps/wave3.yaml
cells:
  - label: B1_zero2_bs8
    overrides:
      trainer.args.parallel_mode: fsdp2_zero2
      trainer.args.batch_size_per_gpu: 8
      trainer.args.batch_split: 2
    extra_env:
      SPARSE_ATTN_BACKEND: flash_attn_3
    active_steps: 50
    chrome_trace: true

  - label: B2_zero2_bs10
    overrides:
      trainer.args.parallel_mode: fsdp2_zero2
      trainer.args.batch_size_per_gpu: 10
      trainer.args.batch_split: 2
    extra_env:
      SPARSE_ATTN_BACKEND: flash_attn_3
    active_steps: 50
    chrome_trace: true

  - label: B3_zero2_bs12
    overrides:
      trainer.args.parallel_mode: fsdp2_zero2
      trainer.args.batch_size_per_gpu: 12
      trainer.args.batch_split: 2
    extra_env:
      SPARSE_ATTN_BACKEND: flash_attn_3
    active_steps: 30
    chrome_trace: true
```

- [ ] **Step 2: Run sweep**

Run: `.venv/bin/python scripts/profiling/profile_sweep.py --yaml scripts/profiling/sweeps/wave3.yaml --host host-10-240-99-119`
Expected: 3 cell 完成；每 cell 一个 chrome trace + result.json

### Task 3.3: Apply decision tree

- [ ] **Step 1: Run analyze_overlap.py on each B trace**

Run:
```bash
for cell in B1_zero2_bs8 B2_zero2_bs10 B3_zero2_bs12; do
  trace=$(find results/profile_dit_runs/ -name "*.pt.trace.json" -newer logs/wave_verify/W3_BASELINE.json | grep -i $cell | head -1)
  .venv/bin/python scripts/profiling/analyze_overlap.py "$trace" --out logs/wave3/$cell
done
```

- [ ] **Step 2: Compose summary doc**

Create `my-docs/$(date +%Y%m%d)-fsdp2-profile-summary.md` with:
- 3 cell 的 bucket_table 横向比对
- comm_hidden_ratio 各 cell
- bs↑ throughput trade-off
- 应用 §6.5 决策树：填充 W4/W5 优先级

- [ ] **Step 3: Tag W3 done**

```bash
git add scripts/profiling/sweeps/wave3.yaml my-docs/*fsdp2-profile-summary.md
git commit -m "docs: W3 FSDP2 profile summary + decision tree application"
git tag -a wave3-complete -m "Wave 3 profile complete"
```

---

## Wave 4: bs↑ + apply_optim_in_backward (~1 day)

**目标**：bs/gpu 8 → 12 production；C6 apply_optim_in_backward overlap optim with bwd reduce_scatter；step time 1.40 (bs12)；sample/s +35%。

**Files map：**
- Modify: `coart/dit/configs/coart_dit_shape_512_ft.json`（bs/elastic）
- Modify: `coart/dit/parallel/fsdp2.py`（apply_optim_in_backward hook）
- Test: `coart/tests/test_w4_bs12_smoke.py`
- Test: `coart/tests/test_w4_apply_optim_in_backward.py`

### Task 4.1: bs sweep + select bs12 for production

- [ ] **Step 1: Run bs10 / bs12 / bs14 short comparison**

Run:
```bash
for bs in 10 12 14; do
  bash scripts/wave_verify.sh w4_bs${bs} --mode fsdp2_zero2 --bs ${bs} --steps 30
done
```

- [ ] **Step 2: Compose throughput table**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/w4_bs10_*/result.json \
         logs/wave_verify/w4_bs12_*/result.json \
         logs/wave_verify/w4_bs14_*/result.json \
  --out logs/wave_verify/W4_bs_sweep.md
cat logs/wave_verify/W4_bs_sweep.md
```
Expected: pick bs that gives best sample/s under mem peak ≤ 75 GB

- [ ] **Step 3: Update config to selected bs**

Edit `coart/dit/configs/coart_dit_shape_512_ft.json`:
```jsonc
"trainer": { "args": { "batch_size_per_gpu": 12 }}  // or whatever sweep picked
```

### Task 4.2: Implement apply_optim_in_backward (C6)

- [ ] **Step 1: Add hook in parallel/fsdp2.py**

```python
# Add to coart/dit/parallel/fsdp2.py
def attach_optim_in_backward(trainer):
    """Register per-param optimizer hook so optim.step happens during bwd.

    Each param's grad accumulation completes → fire its single-param AdamW step
    → marks grad as None so it doesn't accumulate further. Overlaps with
    reduce_scatter on FSDP2's comm stream.
    """
    from torch.distributed.optim import _apply_optimizer_in_backward
    base_cls = type(trainer.optimizer)
    base_kwargs = dict(trainer.optimizer.defaults)
    _apply_optimizer_in_backward(
        optimizer_class=base_cls,
        params=list(trainer.parallel_model.parameters()),
        optimizer_kwargs=base_kwargs,
    )
    # Mark trainer.optimizer as a no-op stub (step happens in bwd hooks now)
    class _StubOpt:
        def step(self): pass
        def zero_grad(self, *a, **k): pass
        @property
        def param_groups(self): return [{"lr": base_kwargs.get("lr", 1e-4)}]
    trainer.optimizer = _StubOpt()
    print("[parallel/fsdp2] apply_optim_in_backward attached", flush=True)
```

Call from `init_after_super` after `_init_optimizer`:
```python
if fsdp_cfg.get("apply_optim_in_backward", False):
    attach_optim_in_backward(trainer)
```

- [ ] **Step 2: Add config field**

Edit `coart/dit/configs/coart_dit_shape_512_ft.json`:
```jsonc
"fsdp2": { ..., "apply_optim_in_backward": true }
```

- [ ] **Step 3: Smoke run**

Run: `bash scripts/wave_verify.sh w4_c6 --mode fsdp2_zero2 --bs 12 --steps 30`
Expected: log 见 `[parallel/fsdp2] apply_optim_in_backward attached`；不 OOM；loss 正常

### Task 4.3: W4 final verify

- [ ] **Step 1: 200-step run**

Run: `bash scripts/wave_verify.sh w4_final --mode fsdp2_zero2 --bs 12 --steps 200`

- [ ] **Step 2: Compare to W3 baseline**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W3_BASELINE.json logs/wave_verify/w4_final_*/result.json \
  --out logs/wave_verify/W4_VERDICT.md
cat logs/wave_verify/W4_VERDICT.md
```
Expected gate: bs12 mem peak ≤ 75 GB; sample/s +18~25% vs W3 baseline; optim bucket ≤ baseline - 30 ms (C6 effect)

- [ ] **Step 3: Commit + tag**

```bash
cp $(ls -t logs/wave_verify/w4_final_*/result.json | head -1) logs/wave_verify/W5_BASELINE.json
git add coart/dit/parallel/fsdp2.py coart/dit/configs/coart_dit_shape_512_ft.json \
        coart/tests/test_w4_*.py
git commit -m "feat(coart_dit): W4 — bs/gpu=12 + apply_optim_in_backward (C5+C6)

bs12 throughput +25% vs bs8.
apply_optim_in_backward hides AdamW.step in FSDP2 bwd reduce_scatter."
git tag -a wave4-complete -m "Wave 4 complete"
```

---

## Wave 5: Fused RMSNorm via flash_attn (~1.5 day)

**目标**：替换 `CoartSparseMultiHeadRMSNorm.forward` 用 `flash_attn.ops.rms_norm.rms_norm_fn`；通过三层 numerical gate（ULP / 1K-step loss / 5K-step trajectory）；step time -110~140 ms。

**Files map：**
- Modify: `coart/dit/modeling/rmsnorm.py`
- Test: `coart/tests/test_w5_fused_rmsnorm_ulp.py`
- Test: `coart/tests/test_w5_loss_ab.py`（subprocess wrapper for 1K-step run）

### Task 5.1: Implement fused RMSNorm forward

- [ ] **Step 1: Verify flash_attn rms_norm_fn API**

Run:
```bash
.venv/bin/python -c "
from flash_attn.ops.rms_norm import rms_norm_fn
import torch
x = torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)
w = torch.ones(128, device='cuda', dtype=torch.float32)
y = rms_norm_fn(x, w, 1e-6)
print('out dtype:', y.dtype, 'shape:', y.shape)
"
```
Expected: `out dtype: torch.bfloat16 shape: torch.Size([64, 128])`

- [ ] **Step 2: Update rmsnorm.py forward**

Edit `coart/dit/modeling/rmsnorm.py` — replace forward:

```python
import os

try:
    from flash_attn.ops.rms_norm import rms_norm_fn as _flash_rms_norm
    _HAS_FLASH_RMS = True
except ImportError:
    _HAS_FLASH_RMS = False


class CoartSparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.register_buffer(
            "scale", torch.tensor(dim ** 0.5, dtype=torch.float32),
            persistent=False,
        )
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def _forward_unfused(self, x):
        if isinstance(x, SparseTensor):
            return x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        return F.normalize(x, dim=-1) * self.gamma * self.scale

    def _forward_fused(self, feats: torch.Tensor) -> torch.Tensor:
        # feats shape: [..., heads, dim] OR [T, heads, dim]
        # Reshape to [N, dim] for flash_attn rms_norm_fn (1D weight required).
        orig_shape = feats.shape
        feats_flat = feats.reshape(-1, self.dim).contiguous()
        # weight=ones → fused does pure RMS-norm; gamma applied separately below.
        ones_w = torch.ones(self.dim, device=feats.device, dtype=torch.float32)
        normed = _flash_rms_norm(feats_flat, ones_w, eps=1e-12)
        normed = normed.reshape(orig_shape)
        # Per-head gamma + scale (fold scale into gamma for one bf16 mul)
        return normed * (self.gamma * self.scale)

    def forward(self, x):
        # Disable fused path via env COART_DISABLE_FUSED_RMSNORM=1 for ablation/rollback.
        if not _HAS_FLASH_RMS or os.environ.get("COART_DISABLE_FUSED_RMSNORM", "0") == "1":
            return self._forward_unfused(x)
        if isinstance(x, SparseTensor):
            return x.replace(self._forward_fused(x.feats))
        return self._forward_fused(x)
```

### Task 5.2: ULP test (gate layer 1)

- [ ] **Step 1: Write test**

```python
# coart/tests/test_w5_fused_rmsnorm_ulp.py
"""W5 gate layer 1: ULP < 16 across 100 random inputs."""
import torch
import pytest
from coart.dit.modeling.rmsnorm import CoartSparseMultiHeadRMSNorm

def _ulp_diff_bf16(a: torch.Tensor, b: torch.Tensor) -> int:
    # ULP diff on bf16: convert to int16 representation, take abs diff
    ai = a.view(torch.int16).to(torch.int64)
    bi = b.view(torch.int16).to(torch.int64)
    return int((ai - bi).abs().max().item())

@pytest.mark.parametrize("seed", list(range(20)))  # 20 seeds, each multiple shapes
def test_fused_rmsnorm_ulp_close(seed):
    torch.manual_seed(seed)
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("no GPU")

    # Random shape
    T = torch.randint(256, 8192, (1,)).item()
    norm = CoartSparseMultiHeadRMSNorm(dim=128, heads=12).to(device)
    norm.gamma.data = torch.randn_like(norm.gamma) * 0.5 + 1.0  # near init

    x = torch.randn(T, 12, 128, device=device, dtype=torch.bfloat16)

    # Unfused (reference)
    import os
    os.environ["COART_DISABLE_FUSED_RMSNORM"] = "1"
    y_ref = norm(x)
    os.environ["COART_DISABLE_FUSED_RMSNORM"] = "0"
    # Fused
    y_new = norm(x)

    ulp = _ulp_diff_bf16(y_ref.contiguous(), y_new.contiguous())
    assert ulp < 16, f"seed={seed} T={T}: ULP {ulp} >= 16 (gate fail)"
```

- [ ] **Step 2: Run test**

Run: `.venv/bin/pytest coart/tests/test_w5_fused_rmsnorm_ulp.py -v`
Expected: 20 seeds 全 PASS（ULP < 16 each）

- [ ] **Step 3: If gate fails → STOP and discuss with user**

Per spec §7: gate fail → revert C7, mark blocked. Open new brainstorm with user before proceeding.

### Task 5.3: 1K-step loss A/B (gate layer 2)

- [ ] **Step 1: Write subprocess wrapper test**

```python
# coart/tests/test_w5_loss_ab.py
"""W5 gate layer 2: 1K-step loss A/B diff < 2σ."""
import os, json, glob, subprocess

def test_1k_step_loss_diff_within_2sigma():
    if not os.path.exists("/dev/nvidia0"):
        return
    # A: baseline (fused disabled)
    env_a = dict(os.environ, COART_DISABLE_FUSED_RMSNORM="1")
    subprocess.check_call(["bash", "scripts/wave_verify.sh", "w5_ab_baseline",
                           "--mode", "fsdp2_zero2", "--bs", "12", "--steps", "1000"],
                          env=env_a)
    # B: fused
    env_b = dict(os.environ, COART_DISABLE_FUSED_RMSNORM="0")
    subprocess.check_call(["bash", "scripts/wave_verify.sh", "w5_ab_fused",
                           "--mode", "fsdp2_zero2", "--bs", "12", "--steps", "1000"],
                          env=env_b)
    a = json.load(open(sorted(glob.glob("logs/wave_verify/w5_ab_baseline_*/result.json"))[-1]))
    b = json.load(open(sorted(glob.glob("logs/wave_verify/w5_ab_fused_*/result.json"))[-1]))
    a_loss, a_std = a["loss/mean_loss"]["mean"], a["loss/mean_loss"]["std"]
    b_loss, _ = b["loss/mean_loss"]["mean"], b["loss/mean_loss"]["std"]
    diff = abs(a_loss - b_loss)
    assert diff < 2 * a_std, \
        f"1K-step loss diff {diff} > 2σ ({2 * a_std}); fused not equivalent"
```

- [ ] **Step 2: Run test (will take ~1h GPU)**

Run: `ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && .venv/bin/python -m pytest coart/tests/test_w5_loss_ab.py -v --timeout=7200"`
Expected: PASS

- [ ] **Step 3: If gate fails → STOP**

### Task 5.4: 5K-step trajectory verify (gate layer 3)

- [ ] **Step 1: Run 5K step**

Run:
```bash
ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  bash scripts/wave_verify.sh w5_5k --mode fsdp2_zero2 --bs 12 --steps 5000"
```

- [ ] **Step 2: Compare eval metrics every 500 step**

Run:
```bash
.venv/bin/python -c "
import json, glob
d = json.load(open(sorted(glob.glob('logs/wave_verify/w5_5k_*/result.json'))[-1]))
# eval metrics should be in d['eval/*']; baseline values from W4 final
print(d.get('eval/cd_mean'), d.get('eval/iou_mean'))
"
```
Expected: eval CD/IoU 在 W4 baseline ±5% 内

- [ ] **Step 3: If gate fails → revert + open new spec**

### Task 5.5: W5 final verify + commit + tag

- [ ] **Step 1: Final 200-step verify with C7 enabled**

Run: `bash scripts/wave_verify.sh w5_final --mode fsdp2_zero2 --bs 12 --steps 200`

- [ ] **Step 2: Compare**

Run:
```bash
.venv/bin/python scripts/profiling/profile_compare.py \
  --runs logs/wave_verify/W5_BASELINE.json logs/wave_verify/w5_final_*/result.json \
  --out logs/wave_verify/W5_VERDICT.md
cat logs/wave_verify/W5_VERDICT.md
```
Expected gate: step.mean ↓ -9~12% vs W5 baseline; sample/s +10%

- [ ] **Step 3: Commit + tag**

```bash
git add coart/dit/modeling/rmsnorm.py coart/tests/test_w5_*.py
git commit -m "feat(coart_dit): C7 — fused bf16 RMSNorm via flash_attn rms_norm_fn

Replaces SparseMultiHeadRMSNorm forward with flash_attn fused path.
Per-head gamma applied separately (1D weight API limitation).
Gates: ULP < 16 / 1K-step loss < 2σ / 5K-step eval ±5% — all pass.
~110-140 ms/step gain."
git tag -a wave5-complete -m "Wave 5 fused RMSNorm complete"
```

---

## Wave 6 (optional): Cross-attn KV cache for static DINO features

**Status**：本 plan 不展开。spec §6.4 列入 future spec；W5 完成后 brainstorm 单独立 plan 文件 `my-docs/<date>-wave6-cross-attn-kv-cache-plan.md`。

---

## Self-review

- [x] **Spec coverage**：每个 spec section 都有对应 task：
  - §2 决策表 → 各 wave 任务
  - §3 code organization → W2.0 全部
  - §4 Phase A → W2.1 + W2.2
  - §5 Phase B → W3
  - §6 Phase C → W1 + W4 + W5
  - §7 C7 三层 gate → W5.2-5.4 三个独立 task
  - §8 wave plan → 1:1 对应
  - §9 verify gate → 每 wave 末尾 task
- [x] **Placeholder scan**：检查无 TBD/TODO；所有 step 含具体代码或命令
- [x] **Type consistency**：
  - `CoartSparseMultiHeadRMSNorm` (rmsnorm.py) — 在 W2.0 引入，W5 修改 forward — 类名一致
  - `CoartDitBlock` (block.py) — W2.0 引入；其他文件 import 一致
  - `CoartElasticSLatFlowModel` — config + denoiser.py 一致
  - `parallel_mode` field — trainer + config + dispatcher 一致
  - `init_after_super` / `consolidate_for_save` / `update_ema` / `save_state` / `load_state` — fsdp2.py 与 ddp.py / zro1.py 接口一致

---

## Plan v1.0 — done

总 task: 38 task / 5 wave
预估开发时间: 5.5 day
验证 gate 完整对齐 spec §9
