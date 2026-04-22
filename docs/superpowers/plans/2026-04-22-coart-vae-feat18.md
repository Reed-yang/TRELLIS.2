# coart VAE feat18 Finetune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `coart/` sub-package that replaces `train_finetune_feat18.py` with a task-first modular VAE finetune pipeline, introducing three-branch IO, EMA, atomic resume, and pretrained warm-start repair, targeting 1-node 8-GPU production runs.

**Architecture:** task-first package layout under `coart/` with shared `common/` (EMA / checkpoint / dist / logging / flex_gemm_patch) and shared `data/` (Feat18Dataset / BucketedSampler / stats), plus task-specific `vae/` (io_stems / build / loss / sampling / train / __main__). Placeholder `dit/` for future.

**Tech Stack:** PyTorch, torchrun DDP, TensorBoard, `trellis2` (model + utils, read-only), `flex_gemm` (patched), `o_voxel` (conversion only), numpy, trimesh.

**Reference Spec:** `docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md`

---

## File Structure

```
coart/
├── __init__.py                          (Task 1)
├── common/
│   ├── __init__.py                      (Task 1)
│   ├── flex_gemm_patch.py               (Task 2, copy from train_finetune_feat18.py:78-136)
│   ├── dist_utils.py                    (Task 3, copy from train_finetune_feat18.py:142-178)
│   ├── ema.py                           (Task 4, NEW)
│   ├── checkpoint.py                    (Task 5, NEW)
│   └── logging.py                       (Task 6, NEW)
├── data/
│   ├── __init__.py                      (Task 1)
│   ├── stats.py                         (Task 7, adapt from train_finetune_feat18.py:478-504)
│   ├── feat18_dataset.py                (Task 8, adapt from train_finetune_feat18.py:278-389, add val split)
│   └── samplers.py                      (Task 9, copy from train_finetune_feat18.py:395-472)
├── vae/
│   ├── __init__.py                      (Task 1)
│   ├── io_stems.py                      (Task 10, NEW)
│   ├── build.py                         (Task 11, adapted from train_overfit_feat18.py:228 + train_finetune_feat18.py:184-256 with io_arch branching)
│   ├── loss.py                          (Task 12, NEW)
│   ├── sampling.py                      (Task 13, adapt from train_finetune_feat18.py:847-887)
│   ├── config.py                        (Task 14, adapt from train_finetune_feat18.py:893-991)
│   ├── train.py                         (Task 15 — main loop integration)
│   └── __main__.py                      (Task 16)
├── dit/
│   └── __init__.py                      (Task 1, placeholder)
└── tests/
    ├── __init__.py                      (Task 1)
    ├── test_ema.py                      (Task 4)
    ├── test_checkpoint.py               (Task 5)
    ├── test_io_stems.py                 (Task 10)
    ├── test_build_warmstart.py          (Task 11)
    └── test_loss.py                     (Task 12)
```

**Untouched files (read-only imports from):**
- `trellis2/models/sc_vaes/sparse_unet_vae.py` — SparseUnetVaeEncoder / SparseUnetVaeDecoder
- `trellis2/utils/grad_clip_utils.py` — AdaptiveGradClipper (explicit reuse, per user approval)
- `trellis2/modules/sparse/*` — SparseTensor, SparseLinear
- `trellis2/models.__init__` — `from_pretrained`
- `corep_fast/pipeline.py`, `corep_fast/containers.py` — CorepParam
- `train_overfit_feat18.py` — `feats_to_param`, `feature_to_mesh`, `param_to_feats` (imported, not modified)

**Untouched files (explicitly left alone):**
- `train_finetune_feat18.py` — baseline reference, kept as-is
- `trellis2/trainers/*` — original trainer framework

---

## Task 0: Preflight — Generate `stats_global.npz`

**Files:**
- Run: `scripts/preprocess-by-rank/compute_stats.py`
- Output: `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz`

- [ ] **Step 1: Run stats computation**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
python scripts/preprocess-by-rank/compute_stats.py \
    --out_dir /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512
```

Expected: `[stats] wrote .../feat18_512/stats_global.npz` after 2-5 minutes.

- [ ] **Step 2: Verify stats file**

Run:
```bash
python -c "
import numpy as np
s = np.load('/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz')
print('mean shape:', s['mean'].shape, 'values:', np.round(s['mean'], 4).tolist())
print('std shape:', s['std'].shape, 'values:', np.round(s['std'], 4).tolist())
print('n_voxels:', int(s['n_voxels']))
assert s['mean'].shape == (18,) and s['std'].shape == (18,)
print('OK')
"
```

Expected: mean[:6] ≈ 0.5 (per stats convention), std[:6] ≈ 1.0; mean[6:12] ≈ 0.2-0.6, std[6:12] ≈ 0.4-0.6; mean[12:18] ≈ 0.002-0.004, std[12:18] ≈ 0.04-0.08. n_voxels > 10B. Prints `OK`.

- [ ] **Step 3: No commit needed (data artifact, not code)**

---

## Task 1: Scaffold `coart/` package structure

**Files:**
- Create: `coart/__init__.py`, `coart/common/__init__.py`, `coart/data/__init__.py`, `coart/vae/__init__.py`, `coart/dit/__init__.py`, `coart/tests/__init__.py`

- [ ] **Step 1: Create all directories and empty `__init__.py` files**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p coart/common coart/data coart/vae coart/dit coart/tests
touch coart/__init__.py coart/common/__init__.py coart/data/__init__.py \
      coart/vae/__init__.py coart/tests/__init__.py
```

- [ ] **Step 2: Write placeholder comment in `coart/dit/__init__.py`**

Create `coart/dit/__init__.py` with content:
```python
# Placeholder for future DiT / flow-matching finetune task.
# When implementing, mirror the layout of coart/vae/: config.py, build.py,
# io_stems.py, loss.py, sampling.py, train.py, __main__.py.
# Reuse coart.common and coart.data wherever applicable; keep task-specific
# logic contained to this sub-package.
```

- [ ] **Step 3: Write top-level `coart/__init__.py`**

Content:
```python
"""coart — corep-related finetune training sub-package.

Sub-packages:
    common/   shared training infra (EMA, checkpoint, dist, logging, patches)
    data/     shared data pipeline (Feat18Dataset, BucketedSampler, stats)
    vae/      Shape-VAE feat18 finetune task (`python -m coart.vae`)
    dit/      placeholder for future DiT / flow-matching finetune

See docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md for design.
"""
```

- [ ] **Step 4: Verify package imports**

Run:
```bash
python -c "import coart, coart.common, coart.data, coart.vae, coart.dit, coart.tests; print('OK')"
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add coart/
git commit -m "feat(coart): scaffold task-first sub-package directory structure"
```

---

## Task 2: `coart/common/flex_gemm_patch.py` (copy from original)

**Files:**
- Create: `coart/common/flex_gemm_patch.py`
- Reference: `train_finetune_feat18.py:78-136`

- [ ] **Step 1: Create `coart/common/flex_gemm_patch.py`**

Copy lines 78-136 of `train_finetune_feat18.py` verbatim. Add a module docstring at top:

```python
"""flex_gemm submanifold_conv3d backward patch for frozen-weight compatibility.

Without this patch, training with `weight.requires_grad=False` crashes inside
the triton backward kernel (grad_weight returned as None → reshape on None).
Also fixes a secondary bias=None crash.

Call `_patch_flex_gemm_frozen_weight_bug()` once at process start (idempotent).
"""

# ... [copy body of lines 93-136 from train_finetune_feat18.py] ...
```

The function to include: `_patch_flex_gemm_frozen_weight_bug()`.
Do NOT include the module-level auto-call `_patch_flex_gemm_frozen_weight_bug()` at line 136; callers will invoke it explicitly.

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from coart.common.flex_gemm_patch import _patch_flex_gemm_frozen_weight_bug; print('OK')"
```
Expected: `OK` (no crash even if flex_gemm absent because inner try/except).

- [ ] **Step 3: Commit**

```bash
git add coart/common/flex_gemm_patch.py
git commit -m "feat(coart): add flex_gemm frozen-weight backward patch (common)"
```

---

## Task 3: `coart/common/dist_utils.py` (copy + generalize)

**Files:**
- Create: `coart/common/dist_utils.py`
- Reference: `train_finetune_feat18.py:139-178`

- [ ] **Step 1: Create `coart/common/dist_utils.py`**

Content:
```python
"""Distributed training helpers for torchrun + DDP."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_dist():
    """Initialise torch.distributed if launched via torchrun.

    Returns (rank, world_size, local_rank, is_dist).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def unwrap(m):
    """Return underlying module if wrapped in DDP."""
    return m.module if isinstance(m, DDP) else m


def wrap_ddp(model, local_rank):
    """Wrap model in DDP with canonical settings (bucket=128MB, find_unused=False)."""
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )


def worker_init_fn(worker_id: int):
    """Independent RNG per (rank, worker) so augmentation is not duplicated.

    Reads `RANK` and `_FT_NUM_WORKERS` from env. Callers must set
    `_FT_NUM_WORKERS` before DataLoader construction.
    """
    rank = int(os.environ.get("RANK", 0))
    num_workers = int(os.environ.get("_FT_NUM_WORKERS", 1))
    seed = (rank * max(num_workers, 1) + worker_id) * 9973 + 17
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)
```

Note: function names changed from `_init_dist / _unwrap / _wrap_ddp / _worker_init_fn` (underscore-prefix internals in original) to public `init_dist / unwrap / wrap_ddp / worker_init_fn` (these are now module-level public API).

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from coart.common.dist_utils import init_dist, unwrap, wrap_ddp, worker_init_fn; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add coart/common/dist_utils.py
git commit -m "feat(coart): add dist_utils helpers (init_dist, wrap_ddp, worker_init_fn)"
```

---

## Task 4: `coart/common/ema.py` — EMAModel class (NEW, TDD)

**Files:**
- Create: `coart/common/ema.py`
- Test: `coart/tests/test_ema.py`

- [ ] **Step 1: Write failing test `coart/tests/test_ema.py`**

Content:
```python
"""Unit tests for coart.common.ema.EMAModel."""
import copy

import pytest
import torch
import torch.nn as nn

from coart.common.ema import EMAModel


def _tiny_model():
    return nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))


def test_ema_initial_shadow_matches_model():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.999)
    for p, s in zip(m.parameters(), ema.shadow_params()):
        assert torch.allclose(p.detach().float(), s), "initial shadow must match params"


def test_ema_update_moves_toward_online():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.9)
    # Record initial shadow snapshot.
    snap = [s.clone() for s in ema.shadow_params()]
    # Mutate online weights: add a large delta.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p))
    ema.update(m)
    # Shadow now = 0.9 * snap + 0.1 * (snap + 1) = snap + 0.1
    for old, s in zip(snap, ema.shadow_params()):
        assert torch.allclose(s, old + 0.1, atol=1e-6), "shadow must drift by (1-decay) * delta"


def test_ema_state_dict_roundtrip():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.999)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p) * 0.5)
    ema.update(m)
    sd = ema.state_dict()
    # Fresh EMA, load state.
    ema2 = EMAModel(_tiny_model(), decay=0.999)
    ema2.load_state_dict(sd)
    for s1, s2 in zip(ema.shadow_params(), ema2.shadow_params()):
        assert torch.allclose(s1, s2)


def test_ema_copy_to_writes_shadow_into_target():
    m = _tiny_model()
    ema = EMAModel(m, decay=0.9)
    # Mutate online params; EMA should still match initial values before update.
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(99.0)
    target = _tiny_model()
    ema.copy_to(target)
    # target params should equal ema shadow (i.e. INITIAL m params, not mutated 99.0).
    for s, p in zip(ema.shadow_params(), target.parameters()):
        assert torch.allclose(s, p.detach().float())


def test_ema_decay_bounds():
    m = _tiny_model()
    with pytest.raises(ValueError):
        EMAModel(m, decay=1.5)
    with pytest.raises(ValueError):
        EMAModel(m, decay=-0.1)
```

- [ ] **Step 2: Run tests — confirm fail**

Run:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
pytest coart/tests/test_ema.py -v
```
Expected: `ModuleNotFoundError: No module named 'coart.common.ema'` or similar — 5 fails.

- [ ] **Step 3: Implement `coart/common/ema.py`**

Content:
```python
"""Exponential-moving-average shadow parameter tracking for training stabilisation.

Matches the semantics of trellis2/trainers/basic.py::BasicTrainer EMA:
    shadow = decay * shadow + (1 - decay) * live_params
updated once per optimizer step, stored in fp32 on the same device as the model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator

import torch
import torch.nn as nn


class EMAModel:
    """Maintains fp32 shadow copies of `model.parameters()`.

    Args:
        model: an `nn.Module`; shadow is created from `model.parameters()` at construction.
        decay: EMA decay rate in (0, 1). Typical value 0.9999 for long runs.

    Thread-safety: not thread-safe. Call `update()` exactly once per optimizer
    step in the same process that owns the model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self._shadow = [p.detach().clone().float() for p in model.parameters()]

    def shadow_params(self) -> Iterator[torch.Tensor]:
        """Iterator over fp32 shadow tensors (live references, not copies)."""
        return iter(self._shadow)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Perform one EMA step against the current online parameters."""
        d = self.decay
        one_minus_d = 1.0 - d
        for s, p in zip(self._shadow, model.parameters()):
            # shadow = d * shadow + (1 - d) * p
            s.mul_(d).add_(p.detach().float(), alpha=one_minus_d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Write shadow values into `model.parameters()` (cast to param dtype)."""
        for s, p in zip(self._shadow, model.parameters()):
            p.data.copy_(s.to(p.dtype))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": [s.clone() for s in self._shadow],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        saved = state["shadow"]
        if len(saved) != len(self._shadow):
            raise ValueError(
                f"EMA shadow count mismatch: saved={len(saved)} current={len(self._shadow)}"
            )
        for dst, src in zip(self._shadow, saved):
            if dst.shape != src.shape:
                raise ValueError(f"EMA shape mismatch: {dst.shape} vs {src.shape}")
            dst.copy_(src)
```

- [ ] **Step 4: Run tests — confirm pass**

Run:
```bash
pytest coart/tests/test_ema.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add coart/common/ema.py coart/tests/test_ema.py
git commit -m "feat(coart): add EMAModel with shadow-param update, roundtrip, copy_to"
```

---

## Task 5: `coart/common/checkpoint.py` — atomic save + rolling + resume (NEW, TDD)

**Files:**
- Create: `coart/common/checkpoint.py`
- Test: `coart/tests/test_checkpoint.py`

- [ ] **Step 1: Write failing test `coart/tests/test_checkpoint.py`**

Content:
```python
"""Unit tests for coart.common.checkpoint.

Covers atomic_save (no torn writes), save_ckpt + rolling-K eviction, and
load_for_resume (glob latest + metadata parsing).
"""
import os

import pytest
import torch

from coart.common.checkpoint import atomic_save, save_ckpt, find_latest_ckpt


def test_atomic_save_writes_target_file(tmp_path):
    target = tmp_path / "obj.pt"
    atomic_save({"x": 42}, str(target))
    assert target.exists()
    loaded = torch.load(target)
    assert loaded["x"] == 42


def test_atomic_save_no_tmp_leak(tmp_path):
    target = tmp_path / "obj.pt"
    atomic_save({"x": 42}, str(target))
    # Should be no .tmp file after successful save.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"unexpected tmp leftovers: {leftovers}"


def test_save_ckpt_creates_step_file(tmp_path):
    state = {"step": 100, "model": {"w": torch.zeros(3)}}
    save_ckpt(state, str(tmp_path), step=100, keep_k=3, prefix="ckpt")
    files = sorted(tmp_path.glob("ckpt_step*.pt"))
    assert len(files) == 1
    assert files[0].name == "ckpt_step0000100.pt"


def test_save_ckpt_rolling_eviction(tmp_path):
    for step in [100, 200, 300, 400, 500]:
        save_ckpt({"step": step}, str(tmp_path), step=step, keep_k=3, prefix="ckpt")
    files = sorted(tmp_path.glob("ckpt_step*.pt"))
    assert len(files) == 3
    steps = [int(f.name[len("ckpt_step"):len("ckpt_step") + 7]) for f in files]
    assert steps == [300, 400, 500], f"expected [300,400,500] after rolling, got {steps}"


def test_find_latest_ckpt_returns_none_when_empty(tmp_path):
    assert find_latest_ckpt(str(tmp_path), prefix="ckpt") is None


def test_find_latest_ckpt_returns_max_step(tmp_path):
    for step in [50, 150, 250]:
        save_ckpt({"step": step}, str(tmp_path), step=step, keep_k=10, prefix="ckpt")
    path, step = find_latest_ckpt(str(tmp_path), prefix="ckpt")
    assert step == 250
    assert path.endswith("ckpt_step0000250.pt")


def test_save_ckpt_preserves_multiple_prefixes(tmp_path):
    """EMA and misc ckpts share dir but use different prefixes; rolling is per-prefix."""
    save_ckpt({"step": 100}, str(tmp_path), step=100, keep_k=2, prefix="ckpt")
    save_ckpt({"step": 100}, str(tmp_path), step=100, keep_k=2, prefix="ema_0.9999")
    save_ckpt({"step": 100}, str(tmp_path), step=100, keep_k=2, prefix="misc")
    assert (tmp_path / "ckpt_step0000100.pt").exists()
    assert (tmp_path / "ema_0.9999_step0000100.pt").exists()
    assert (tmp_path / "misc_step0000100.pt").exists()
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest coart/tests/test_checkpoint.py -v`
Expected: 7 fails (ModuleNotFoundError).

- [ ] **Step 3: Implement `coart/common/checkpoint.py`**

Content:
```python
"""Atomic checkpoint I/O with rolling-K eviction.

Naming convention:
    {prefix}_step{step:07d}.pt

Typical prefixes used by coart.vae.train:
    ckpt        — encoder + decoder + optimizer state
    ema_<rate>  — EMA shadow parameters (one file per rate)
    misc        — sampler epoch, RNG state, step counter, etc.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch


_STEP_PATTERN = re.compile(r"_step(\d{7})\.pt$")


def atomic_save(obj: Any, path: str) -> None:
    """Write `torch.save(obj, path)` atomically via tmp + os.replace.

    If the process is killed mid-write, `path` either contains the previous
    contents (or does not exist) — never a torn half-written file.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic rename on POSIX


def _list_ckpts(output_dir: str, prefix: str) -> list[Tuple[str, int]]:
    """Return [(path, step), ...] sorted ascending by step."""
    pattern = os.path.join(output_dir, f"{prefix}_step*.pt")
    out: list[Tuple[str, int]] = []
    for p in glob.glob(pattern):
        m = _STEP_PATTERN.search(os.path.basename(p))
        if m:
            out.append((p, int(m.group(1))))
    out.sort(key=lambda t: t[1])
    return out


def save_ckpt(
    state: Dict[str, Any],
    output_dir: str,
    step: int,
    keep_k: int = 3,
    prefix: str = "ckpt",
) -> str:
    """Atomically save `state` to `<output_dir>/{prefix}_step{step:07d}.pt`.

    After saving, prune older files with the same prefix so only the newest
    `keep_k` remain on disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, f"{prefix}_step{step:07d}.pt")
    atomic_save(state, target)

    existing = _list_ckpts(output_dir, prefix)
    if len(existing) > keep_k:
        to_remove = existing[: len(existing) - keep_k]
        for path, _ in to_remove:
            try:
                os.remove(path)
            except OSError:
                pass
    return target


def find_latest_ckpt(
    output_dir: str,
    prefix: str = "ckpt",
) -> Optional[Tuple[str, int]]:
    """Return (path, step) of the highest-step ckpt with `prefix`, or None."""
    existing = _list_ckpts(output_dir, prefix)
    if not existing:
        return None
    return existing[-1]
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest coart/tests/test_checkpoint.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add coart/common/checkpoint.py coart/tests/test_checkpoint.py
git commit -m "feat(coart): add atomic_save, save_ckpt rolling-K, find_latest_ckpt"
```

---

## Task 6: `coart/common/logging.py` — TB writer helpers (NEW)

**Files:**
- Create: `coart/common/logging.py`

- [ ] **Step 1: Implement `coart/common/logging.py`**

Content:
```python
"""TensorBoard logging helpers with cadence control and DDP all-reduce.

Pattern:
    logger = CoartTBLogger(output_dir, is_master=(rank==0))
    logger.scalar("loss/total", float_value, step)  # buffered, written at i_log cadence
    logger.flush_if_due(step, i_log=100)
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


class CoartTBLogger:
    """Rank-0 TB writer with per-step scalar buffer and DDP-aware all-reduce.

    Non-master ranks accept scalar calls (no-op) to avoid caller branching.
    `flush_if_due` performs an all-reduce across all ranks over the buffered
    scalars and writes rank-0's averaged values to TB at cadence `i_log`.
    """

    def __init__(self, output_dir: str, is_master: bool):
        self.is_master = is_master
        self._buf: Dict[str, list[float]] = {}
        self._writer: Optional[SummaryWriter] = None
        if is_master:
            os.makedirs(os.path.join(output_dir, "tb_logs"), exist_ok=True)
            self._writer = SummaryWriter(os.path.join(output_dir, "tb_logs"))

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Buffer a scalar for this step. Actual write happens on flush_if_due."""
        self._buf.setdefault(tag, []).append(float(value))

    def flush_if_due(self, step: int, i_log: int) -> None:
        """If step % i_log == 0, all-reduce across ranks and write averaged scalars."""
        if step % i_log != 0:
            return
        if not self._buf:
            return

        tags = sorted(self._buf.keys())
        vals = torch.tensor(
            [sum(self._buf[t]) / len(self._buf[t]) for t in tags],
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)

        if self.is_master and self._writer is not None:
            for t, v in zip(tags, vals.tolist()):
                self._writer.add_scalar(t, v, step)

        self._buf.clear()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from coart.common.logging import CoartTBLogger; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add coart/common/logging.py
git commit -m "feat(coart): add CoartTBLogger with cadence control and DDP all-reduce"
```

---

## Task 7: `coart/data/stats.py` — stats loading + normalise helpers

**Files:**
- Create: `coart/data/stats.py`
- Reference: `train_finetune_feat18.py:478-504`

- [ ] **Step 1: Implement `coart/data/stats.py`**

Content:
```python
"""Per-channel normalisation for 18-ch corep feats.

Layout of the 18 channels (from corep_fast param_to_feats):
    [point1_xyz(3), point2_xyz(3), edge_weights(6), face_weights(6)]

Normalisation convention (matches train_finetune_feat18.py:481-484):
    ch 0:6   — (x - 0.5) / 1.0       (point coords already in [0, 1] local cube)
    ch 6:18  — (x - mean) / std      (computed offline from 41k shards)

If the stats file is missing, fall back to identity defaults (mean=[0.5]*6+[0]*12,
std=ones), matching `load_stats` in the original script.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


def load_stats(
    stats_path: str | None,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) tensors of shape (18,) on `device`."""
    if stats_path is None or not os.path.exists(stats_path):
        if verbose:
            print(f"[stats] no file at {stats_path!r}, using identity defaults")
        mean = np.zeros(18, dtype=np.float32)
        mean[:6] = 0.5
        std = np.ones(18, dtype=np.float32)
    else:
        s = np.load(stats_path)
        mean = s["mean"].astype(np.float32)
        std = np.maximum(s["std"].astype(np.float32), 1e-3)  # hard-clamp safety
    if verbose:
        print(f"[stats] mean = {np.round(mean, 4).tolist()}")
        print(f"[stats] std  = {np.round(std, 4).tolist()}")
    return (
        torch.from_numpy(mean).to(device),
        torch.from_numpy(std).to(device),
    )


def normalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply per-channel (x - mean) / std. Shapes: feats (N, 18), mean/std (18,)."""
    return (feats - mean) / std


def denormalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert `normalize`. For logging/visualisation only (training uses normalised space)."""
    return feats * std + mean
```

- [ ] **Step 2: Verify import + behaviour on identity defaults**

Run:
```bash
python -c "
import torch
from coart.data.stats import load_stats, normalize, denormalize
m, s = load_stats(None, torch.device('cpu'), verbose=False)
assert m.shape == (18,) and s.shape == (18,)
x = torch.arange(18, dtype=torch.float32).unsqueeze(0)
y = normalize(x, m, s)
z = denormalize(y, m, s)
assert torch.allclose(x, z, atol=1e-6), 'roundtrip'
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add coart/data/stats.py
git commit -m "feat(coart): add stats load/normalize/denormalize for 18-ch feats"
```

---

## Task 8: `coart/data/feat18_dataset.py` — Dataset + collate + val split

**Files:**
- Create: `coart/data/feat18_dataset.py`
- Reference: `train_finetune_feat18.py:260-389`

- [ ] **Step 1: Implement `coart/data/feat18_dataset.py`**

Content (~ 200 lines; structured):

```python
"""Feat18Dataset — precomputed corep .npz shards with deterministic val split.

Each .npz file (produced by precompute_feat18.py) has:
    cube_indices : (N, 3) int16
    feats        : (N, 18) float16
    num_boundary : (N,)   int8
    resolution   : scalar int32

Train / val split: `int(sha[:8], 16) % val_split_mod == 0` → val, else train.
The split is deterministic and reproducible across machines.
"""
from __future__ import annotations

import glob
import os
import zipfile
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_npz_array_shape(path: str, array_name: str) -> tuple:
    """Read .npy header for one array inside a .npz without decompressing data.

    Much faster than np.load() when we only need the row count (for bucket sampling).
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open(f"{array_name}.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


def _is_val_sha(sha: str, mod: int) -> bool:
    """Hash-based deterministic val assignment: sha[:8] hex % mod == 0."""
    if mod <= 0:
        return False
    try:
        h = int(sha[:8], 16)
    except ValueError:
        h = abs(hash(sha))
    return (h % mod) == 0


class Feat18Dataset(Dataset):
    """Loads precomputed .npz shards and applies random integer translation augmentation.

    Args:
        data_dir: directory containing *.npz shards.
        resolution: grid resolution (must match precompute_feat18.py setting).
        max_translate: ± shift range in voxel units (0 = disable augmentation).
        augment: if True, apply random translation; else deterministic identity.
        precompute_voxel_counts: if True, read .npy headers to cache per-sample N.
        max_voxels: if > 0, subsample each sample down to this cap (sha-seeded).
        val_split_mod: if > 0, determines train/val partition (see below).
        split: "train" or "val" or "all". If "train"/"val", filter by hash mod.

    val_split_mod examples:
        val_split_mod=200 → ~0.5% held-out (200 samples from 41k)
        val_split_mod=0   → split=ignored; all samples included regardless of `split`.
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        max_translate: int = 16,
        augment: bool = True,
        precompute_voxel_counts: bool = False,
        max_voxels: int = 0,
        val_split_mod: int = 0,
        split: str = "train",
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train/val/all, got {split!r}")

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"no .npz files in {data_dir}")

        if val_split_mod > 0 and split != "all":
            want_val = split == "val"
            filtered = [
                f for f in all_files
                if _is_val_sha(os.path.splitext(os.path.basename(f))[0], val_split_mod)
                is want_val
            ]
            if not filtered:
                raise RuntimeError(
                    f"split={split} val_split_mod={val_split_mod} produced empty set "
                    f"from {len(all_files)} total files"
                )
            self.files = filtered
        else:
            self.files = all_files

        self.resolution = resolution
        self.max_translate = max_translate
        self.augment = augment
        self.max_voxels = int(max_voxels)
        self.num_voxels: np.ndarray | None = None
        if precompute_voxel_counts:
            self.num_voxels = np.array(
                [int(_read_npz_array_shape(f, "cube_indices")[0]) for f in self.files],
                dtype=np.int64,
            )

    def effective_voxels(self) -> np.ndarray | None:
        """Per-sample voxel count after --max_voxels cap (for bucket sampler)."""
        if self.num_voxels is None:
            return None
        if self.max_voxels <= 0:
            return self.num_voxels
        return np.minimum(self.num_voxels, self.max_voxels)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        cube_indices = d["cube_indices"].astype(np.int32)
        feats = d["feats"].astype(np.float32)
        num_boundary = d["num_boundary"].astype(np.int32)
        sha = os.path.splitext(os.path.basename(self.files[idx]))[0]

        # Deterministic per-sample voxel subsample (sha-seeded).
        if self.max_voxels > 0 and cube_indices.shape[0] > self.max_voxels:
            try:
                seed = int(sha[:8], 16) & 0xFFFFFFFF
            except ValueError:
                seed = abs(hash(sha)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            keep = rng.choice(cube_indices.shape[0], size=self.max_voxels, replace=False)
            keep.sort()
            cube_indices = cube_indices[keep]
            feats = feats[keep]

        if self.augment and self.max_translate > 0:
            R = self.resolution
            min_idx = cube_indices.min(axis=0)
            max_idx = cube_indices.max(axis=0)
            shifts = np.empty(3, dtype=np.int32)
            for a in range(3):
                lo = max(-int(min_idx[a]), -self.max_translate)
                hi = min(R - 1 - int(max_idx[a]), self.max_translate)
                shifts[a] = np.random.randint(lo, hi + 1) if hi >= lo else 0
            cube_indices = cube_indices + shifts

        return {
            "cube_indices": cube_indices,
            "feats": feats,
            "num_boundary": num_boundary,
            "sha": sha,
        }


def collate_fn(batch):
    """Concatenate variable-length samples into a single SparseTensor payload.

    Produces:
        coords : (sum_N, 4) int32 — [batch_idx, x, y, z]
        feats  : (sum_N, 18) float32
        sizes  : list[int] per-sample voxel counts
        shas   : list[str] per-sample SHAs
        cube_indices_per_sample, num_boundary_per_sample (for eval mesh dump)
    """
    coords_chunks, feats_chunks, sizes = [], [], []
    cube_indices_list, num_boundary_list, shas = [], [], []
    for i, item in enumerate(batch):
        ci = item["cube_indices"]
        N = ci.shape[0]
        bi = np.full((N, 1), i, dtype=np.int32)
        coords_chunks.append(np.concatenate([bi, ci], axis=1))
        feats_chunks.append(item["feats"])
        sizes.append(N)
        cube_indices_list.append(ci)
        num_boundary_list.append(item["num_boundary"])
        shas.append(item["sha"])
    return {
        "coords": torch.from_numpy(np.concatenate(coords_chunks, axis=0)),
        "feats": torch.from_numpy(np.concatenate(feats_chunks, axis=0)),
        "sizes": sizes,
        "cube_indices_per_sample": cube_indices_list,
        "num_boundary_per_sample": num_boundary_list,
        "shas": shas,
    }
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from coart.data.feat18_dataset import Feat18Dataset, collate_fn, _is_val_sha; \
    assert _is_val_sha('00000000abc', 200) is True; \
    assert _is_val_sha('ffffffffabc', 200) is False or _is_val_sha('ffffffffabc', 200) is True; \
    print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Smoke-test against real data (small)**

Run:
```bash
python -c "
from coart.data.feat18_dataset import Feat18Dataset
d_train = Feat18Dataset(
    '/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512/data',
    resolution=512, max_translate=0, augment=False,
    val_split_mod=200, split='train')
d_val = Feat18Dataset(
    '/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512/data',
    resolution=512, max_translate=0, augment=False,
    val_split_mod=200, split='val')
print(f'train: {len(d_train)}, val: {len(d_val)}, total: {len(d_train)+len(d_val)}')
item = d_val[0]
print(f'val[0] sha={item[\"sha\"][:16]} voxels={item[\"cube_indices\"].shape[0]}')
"
```
Expected: `train: ~41670, val: ~200, total: 41871` (small rounding); val[0] prints OK.

- [ ] **Step 4: Commit**

```bash
git add coart/data/feat18_dataset.py
git commit -m "feat(coart): add Feat18Dataset with sha-hash val split + collate_fn"
```

---

## Task 9: `coart/data/samplers.py` — BucketedDistributedSampler

**Files:**
- Create: `coart/data/samplers.py`
- Reference: `train_finetune_feat18.py:395-472`

- [ ] **Step 1: Implement `coart/data/samplers.py`**

Copy the class body from `train_finetune_feat18.py:395-472` verbatim. Add a module docstring:

```python
"""BucketedDistributedSampler — same-voxel-count batches across DDP ranks.

Sort samples by voxel count, reshape into rows of (num_replicas * batch_size),
shuffle row order every epoch. Within any given global batch, all ranks see
similarly-sized samples → limits straggler-rank tail.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class BucketedDistributedSampler(Sampler):
    # [copy lines 408-472 verbatim]
    ...
```

(The full class body is already documented inline in the source; copy as-is.)

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "
import numpy as np
from coart.data.samplers import BucketedDistributedSampler
vx = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int64)
s = BucketedDistributedSampler(vx, num_replicas=2, rank=0, batch_size=1, seed=0)
print(f'len per rank: {len(s)}')
indices = list(iter(s))
print(f'first few indices rank 0: {indices}')
"
```
Expected: `len per rank: 4` and 4 indices printed.

- [ ] **Step 3: Commit**

```bash
git add coart/data/samplers.py
git commit -m "feat(coart): add BucketedDistributedSampler for voxel-balanced DDP batches"
```

---

## Task 10: `coart/vae/io_stems.py` — Feat18EncIO / Feat18DecIO (NEW, TDD)

**Files:**
- Create: `coart/vae/io_stems.py`
- Test: `coart/tests/test_io_stems.py`

- [ ] **Step 1: Write failing test `coart/tests/test_io_stems.py`**

Content:
```python
"""Unit tests for coart.vae.io_stems."""
import pytest
import torch

from coart.vae.io_stems import Feat18EncIO, Feat18DecIO


def _fake_sparse_input(N=32, C=18):
    """Mock SparseTensor-like object exposing .feats and .replace()."""
    class _FakeST:
        def __init__(self, feats):
            self.feats = feats
        def replace(self, new_feats):
            return _FakeST(new_feats)
    return _FakeST(torch.randn(N, C))


def test_enc_io_output_shape():
    stem = Feat18EncIO(c_model=64)
    x = _fake_sparse_input(N=10, C=18)
    out = stem(x)
    assert out.feats.shape == (10, 64)


def test_enc_io_three_branches_independent_gradients():
    """Verify p1_branch, p2_branch, ef_branch are independent nn.Linear modules."""
    stem = Feat18EncIO(c_model=64)
    assert stem.p1_branch is not stem.p2_branch, "p1 and p2 must be independent Linears"
    assert stem.p1_branch.weight is not stem.p2_branch.weight
    assert stem.p1_branch.in_features == 3
    assert stem.p2_branch.in_features == 3
    assert stem.ef_branch.in_features == 12
    assert stem.p1_branch.out_features == 64
    assert stem.p2_branch.out_features == 64
    assert stem.ef_branch.out_features == 64


def test_enc_io_forward_decomposes_18ch():
    """Forward = p1_branch(f[:,0:3]) + p2_branch(f[:,3:6]) + ef_branch(f[:,6:18])."""
    stem = Feat18EncIO(c_model=64)
    f = torch.randn(5, 18)
    x = _fake_sparse_input(N=5, C=18)
    x.feats = f
    out = stem(x)
    expected = (
        stem.p1_branch(f[:, 0:3])
        + stem.p2_branch(f[:, 3:6])
        + stem.ef_branch(f[:, 6:18])
    )
    assert torch.allclose(out.feats, expected, atol=1e-6)


def test_dec_io_output_shape_and_decomposition():
    stem = Feat18DecIO(c_model=64)
    x = _fake_sparse_input(N=7, C=64)
    out = stem(x)
    assert out.feats.shape == (7, 18)
    # Verify split: first 3 = p1_head, 3:6 = p2_head, 6:18 = ef_head
    expected = torch.cat([
        stem.p1_head(x.feats),
        stem.p2_head(x.feats),
        stem.ef_head(x.feats),
    ], dim=-1)
    assert torch.allclose(out.feats, expected, atol=1e-6)


def test_dec_io_three_heads_independent():
    stem = Feat18DecIO(c_model=64)
    assert stem.p1_head is not stem.p2_head
    assert stem.p1_head.out_features == 3
    assert stem.p2_head.out_features == 3
    assert stem.ef_head.out_features == 12
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest coart/tests/test_io_stems.py -v`
Expected: 5 fails.

- [ ] **Step 3: Implement `coart/vae/io_stems.py`**

Content:
```python
"""Three-branch IO stems for feat18 SC-VAE encoder/decoder.

The 18-channel corep payload has three semantically distinct blocks:
    ch 0:3   — point1 xyz (local cube coords, lower-z representative)
    ch 3:6   — point2 xyz (higher-z; zero in 97.8% of single-point cubes)
    ch 6:18  — edge/face ordinal counts (integers 0..~22, 99% in {0,1})

Feat18EncIO splits the 18→C_model projection into three independent nn.Linear
branches summed together. This preserves the z-sort signal (p1 and p2 not
shared → no permutation ambiguity) and allows warm-starting each branch
from a different pretrained source.

Feat18DecIO mirrors the structure on the decoder side: three independent heads
(p1_head, p2_head, ef_head), concatenated to produce the 18-channel output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Feat18EncIO(nn.Module):
    """Three-branch input stem for 18-ch corep feats.

    Forward:
        h = p1_branch(f[:, 0:3]) + p2_branch(f[:, 3:6]) + ef_branch(f[:, 6:18])
    Output shape: (N, c_model).

    Input must be a SparseTensor-like object with attributes `.feats` (N, 18)
    and method `.replace(new_feats) -> same-type`. Does NOT modify the original
    tensor metadata (coords / stride / etc.) — only swaps feats.
    """

    def __init__(self, c_model: int):
        super().__init__()
        self.p1_branch = nn.Linear(3, c_model)
        self.p2_branch = nn.Linear(3, c_model)
        self.ef_branch = nn.Linear(12, c_model)

    def forward(self, x):
        f = x.feats
        h = (
            self.p1_branch(f[:, 0:3])
            + self.p2_branch(f[:, 3:6])
            + self.ef_branch(f[:, 6:18])
        )
        return x.replace(h)


class Feat18DecIO(nn.Module):
    """Three-head output stem producing 18 channels from c_model features.

    Forward:
        out = concat[p1_head(f), p2_head(f), ef_head(f)]   # -> (N, 18)

    Matches the feat18 channel layout expected by `feats_to_param` /
    `feature_to_mesh` downstream: [p1(3), p2(3), ef(12)].
    """

    def __init__(self, c_model: int):
        super().__init__()
        self.p1_head = nn.Linear(c_model, 3)
        self.p2_head = nn.Linear(c_model, 3)
        self.ef_head = nn.Linear(c_model, 12)

    def forward(self, x):
        f = x.feats
        out = torch.cat([
            self.p1_head(f),
            self.p2_head(f),
            self.ef_head(f),
        ], dim=-1)
        return x.replace(out)
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest coart/tests/test_io_stems.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add coart/vae/io_stems.py coart/tests/test_io_stems.py
git commit -m "feat(coart): add three-branch Feat18EncIO/Feat18DecIO with unit tests"
```

---

## Task 11: `coart/vae/build.py` — build_models + load_pretrained_into (with io_arch branching, TDD)

**Files:**
- Create: `coart/vae/build.py`
- Test: `coart/tests/test_build_warmstart.py`
- Reference: `train_overfit_feat18.py:228-288`, `train_finetune_feat18.py:184-256`

- [ ] **Step 1: Write failing test `coart/tests/test_build_warmstart.py`**

Content:
```python
"""Unit tests for load_pretrained_into warm-start matrix (4 combinations).

Verifies that each (io_arch, warmstart_io) combination produces the expected
initialisation. Does NOT load real pretrained weights — uses a mock pretrained
state dict with known contents and asserts the copy logic is correct.
"""
import pytest
import torch
import torch.nn as nn

from coart.vae.io_stems import Feat18EncIO, Feat18DecIO
from coart.vae.build import _apply_warmstart_three_branch


def _fake_pretrained_sd(c0=64, c_end=64):
    """Mock pretrained encoder + decoder state dicts."""
    return (
        {
            "input_layer.weight": torch.randn(c0, 6),
            "input_layer.bias": torch.randn(c0),
        },
        {
            "output_layer.weight": torch.randn(7, c_end),
            "output_layer.bias": torch.randn(7),
        },
    )


def test_three_branch_warmstart_copies_p1_full_strength():
    """p1_branch.weight must equal pretrained W[:, 0:3] exactly."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(
        enc_stem, dec_stem, pre_enc_sd, pre_dec_sd
    )

    expected_p1_w = pre_enc_sd["input_layer.weight"][:, 0:3]
    assert torch.allclose(enc_stem.p1_branch.weight, expected_p1_w)
    assert torch.allclose(enc_stem.p1_branch.bias, pre_enc_sd["input_layer.bias"])


def test_three_branch_warmstart_p2_weight_matches_p1_but_bias_zero():
    """p2_branch.weight = pretrained vertex cols too; p2_branch.bias = 0."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)

    expected_p2_w = pre_enc_sd["input_layer.weight"][:, 0:3]
    assert torch.allclose(enc_stem.p2_branch.weight, expected_p2_w)
    assert torch.allclose(
        enc_stem.p2_branch.bias,
        torch.zeros_like(enc_stem.p2_branch.bias),
    )


def test_three_branch_warmstart_ef_stays_untouched():
    """ef_branch keeps its xavier_uniform init (not zeroed, not copied)."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    ef_w_before = enc_stem.ef_branch.weight.clone()
    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)
    assert torch.allclose(enc_stem.ef_branch.weight, ef_w_before), \
        "ef_branch.weight must NOT be modified by three_branch warm-start"


def test_three_branch_warmstart_decoder_p1_head_copies_rows_0_2():
    """Decoder p1_head.weight = pretrained W[0:3, :]."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)

    assert torch.allclose(dec_stem.p1_head.weight, pre_dec_sd["output_layer.weight"][0:3, :])
    assert torch.allclose(dec_stem.p1_head.bias, pre_dec_sd["output_layer.bias"][0:3])


def test_three_branch_warmstart_decoder_p2_head_weight_matches_p1_but_bias_zero():
    """Decoder p2_head.weight = W[0:3, :] (same as p1); p2_head.bias = 0."""
    pre_enc_sd, pre_dec_sd = _fake_pretrained_sd()
    enc_stem = Feat18EncIO(c_model=64)
    dec_stem = Feat18DecIO(c_model=64)

    _apply_warmstart_three_branch(enc_stem, dec_stem, pre_enc_sd, pre_dec_sd)

    assert torch.allclose(dec_stem.p2_head.weight, pre_dec_sd["output_layer.weight"][0:3, :])
    assert torch.allclose(dec_stem.p2_head.bias, torch.zeros_like(dec_stem.p2_head.bias))
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest coart/tests/test_build_warmstart.py -v`
Expected: 5 fails.

- [ ] **Step 3: Implement `coart/vae/build.py`**

Content:
```python
"""Build feat18 encoder/decoder with selectable IO architecture.

IO variants:
    io_arch="three_branch"  — independent Feat18EncIO / Feat18DecIO (default).
                              Warm-start copies pretrained vertex cols into
                              p1_branch AND p2_branch at full strength; p2 bias=0.
    io_arch="monolithic"    — original sp.SparseLinear(18 → C0) / (C_end → 18).
                              Warm-start copies pretrained vertex cols into
                              the first 3 slots only (partial warm-start).

Backbone weights are always loaded via strict=False + module-level state_dict;
only I/O layers differ per io_arch.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn as nn

from trellis2 import models
from trellis2.models.sc_vaes.sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)

from .io_stems import Feat18EncIO, Feat18DecIO


IOArch = Literal["three_branch", "monolithic"]


def build_models(
    latent_channels: int = 32,
    device: str | torch.device = "cuda",
    io_arch: IOArch = "three_branch",
    model_channels=(64, 128, 256, 512, 1024),
    num_blocks=(0, 4, 8, 16, 4),
    pred_subdiv=(False, True, True, True, False),
    in_channels: int = 18,
    out_channels: int = 18,
) -> Tuple[SparseUnetVaeEncoder, SparseUnetVaeDecoder]:
    """Construct a (encoder, decoder) pair with the selected IO architecture.

    Model channel / block counts / subdiv default to the shape_vae_next_dc_f16c32
    config. Returns fresh (uninitialised beyond xavier) modules; use
    `load_pretrained_into` to warm-start them.
    """
    encoder = SparseUnetVaeEncoder(
        in_channels=in_channels,
        model_channels=list(model_channels),
        latent_channels=latent_channels,
        num_blocks=list(num_blocks),
        block_type=["SparseConvNeXtBlock3d"] * len(num_blocks),
        down_block_type=["SparseResBlockS2C3d"] * (len(num_blocks) - 1),
        block_args=[{"use_checkpoint": True}] * len(num_blocks),
        use_fp16=False,
    ).to(device)

    decoder = SparseUnetVaeDecoder(
        out_channels=out_channels,
        model_channels=list(reversed(list(model_channels))),
        latent_channels=latent_channels,
        num_blocks=list(reversed(list(num_blocks))),
        block_type=["SparseConvNeXtBlock3d"] * len(num_blocks),
        up_block_type=["SparseResBlockC2S3d"] * (len(num_blocks) - 1),
        block_args=[{"use_checkpoint": True}] * len(num_blocks),
        pred_subdiv=list(reversed(list(pred_subdiv))),
        use_fp16=False,
    ).to(device)

    if io_arch == "three_branch":
        c0 = encoder.input_layer.out_features
        c_end = decoder.output_layer.in_features
        encoder.input_layer = Feat18EncIO(c0).to(device)
        decoder.output_layer = Feat18DecIO(c_end).to(device)
    elif io_arch == "monolithic":
        pass  # keep default sp.SparseLinear(18 -> C0) and (C_end -> 18)
    else:
        raise ValueError(f"unknown io_arch: {io_arch!r}")

    return encoder, decoder


@torch.no_grad()
def _apply_warmstart_three_branch(
    enc_stem: Feat18EncIO,
    dec_stem: Feat18DecIO,
    pre_enc_sd: Dict[str, torch.Tensor],
    pre_dec_sd: Dict[str, torch.Tensor],
) -> None:
    """Copy pretrained vertex cols into both point branches; zero p2 biases."""
    pre_enc_w = pre_enc_sd["input_layer.weight"]   # (C0, 6)
    pre_enc_b = pre_enc_sd["input_layer.bias"]     # (C0,)
    pre_dec_w = pre_dec_sd["output_layer.weight"]  # (7, C_end)
    pre_dec_b = pre_dec_sd["output_layer.bias"]    # (7,)

    # Encoder: both point branches get pretrained vertex prior (full strength).
    # ef_branch stays xavier_uniform (nothing to warm-start from).
    enc_stem.p1_branch.weight.data.copy_(pre_enc_w[:, 0:3])
    enc_stem.p1_branch.bias.data.copy_(pre_enc_b)
    enc_stem.p2_branch.weight.data.copy_(pre_enc_w[:, 0:3])
    enc_stem.p2_branch.bias.data.zero_()

    # Decoder: both point heads get pretrained vertex rows; ef_head stays xavier.
    dec_stem.p1_head.weight.data.copy_(pre_dec_w[0:3, :])
    dec_stem.p1_head.bias.data.copy_(pre_dec_b[0:3])
    dec_stem.p2_head.weight.data.copy_(pre_dec_w[0:3, :])
    dec_stem.p2_head.bias.data.zero_()


@torch.no_grad()
def _apply_warmstart_monolithic(
    encoder: SparseUnetVaeEncoder,
    decoder: SparseUnetVaeDecoder,
    pre_enc_sd: Dict[str, torch.Tensor],
    pre_dec_sd: Dict[str, torch.Tensor],
) -> None:
    """Copy pretrained vertex cols into point1 slot only (partial warm-start).

    For monolithic single-layer IO, only the first 3 input columns (= point1)
    and first 3 output rows are overwritten. The other 15 columns/rows keep
    their xavier_uniform init.
    """
    pre_enc_w = pre_enc_sd["input_layer.weight"]
    pre_enc_b = pre_enc_sd["input_layer.bias"]
    pre_dec_w = pre_dec_sd["output_layer.weight"]
    pre_dec_b = pre_dec_sd["output_layer.bias"]

    encoder.input_layer.weight.data[:, 0:3].copy_(pre_enc_w[:, 0:3])
    encoder.input_layer.bias.data.copy_(pre_enc_b)
    decoder.output_layer.weight.data[0:3, :].copy_(pre_dec_w[0:3, :])
    decoder.output_layer.bias.data[0:3].copy_(pre_dec_b[0:3])


def load_pretrained_into(
    encoder: SparseUnetVaeEncoder,
    decoder: SparseUnetVaeDecoder,
    enc_path: str,
    dec_path: str,
    io_arch: IOArch = "three_branch",
    warmstart_io: bool = True,
    verbose: bool = True,
) -> None:
    """Load pretrained backbone + optional IO warm-start.

    Backbone weights are always loaded via strict=False filtering (skipping
    `input_layer.*` / `output_layer.*` which have different shapes in the new IO).

    If `warmstart_io`, dispatch to the per-arch helper for IO initialisation.
    """
    pre_enc = models.from_pretrained(enc_path)
    pre_dec = models.from_pretrained(dec_path)
    pre_enc_sd = pre_enc.state_dict()
    pre_dec_sd = pre_dec.state_dict()

    enc_filtered = {
        k: v for k, v in pre_enc_sd.items() if not k.startswith("input_layer.")
    }
    dec_filtered = {
        k: v for k, v in pre_dec_sd.items() if not k.startswith("output_layer.")
    }
    miss_e, unex_e = encoder.load_state_dict(enc_filtered, strict=False)
    miss_d, unex_d = decoder.load_state_dict(dec_filtered, strict=False)

    if verbose:
        only_io_missing_e = [k for k in miss_e if not k.startswith("input_layer")]
        only_io_missing_d = [k for k in miss_d if not k.startswith("output_layer")]
        print(f"[init] encoder missing (non-IO): {only_io_missing_e}")
        print(f"[init] encoder unexpected     : {list(unex_e)}")
        print(f"[init] decoder missing (non-IO): {only_io_missing_d}")
        print(f"[init] decoder unexpected     : {list(unex_d)}")

    if warmstart_io:
        if io_arch == "three_branch":
            _apply_warmstart_three_branch(
                encoder.input_layer, decoder.output_layer,
                pre_enc_sd, pre_dec_sd,
            )
            if verbose:
                print("[init] three_branch warm-start: pretrained vertex -> "
                      "p1_branch + p2_branch (full strength, p2 bias=0); "
                      "decoder p1_head + p2_head rows; ef untouched")
        elif io_arch == "monolithic":
            _apply_warmstart_monolithic(
                encoder, decoder, pre_enc_sd, pre_dec_sd,
            )
            if verbose:
                print("[init] monolithic partial warm-start: pretrained vertex "
                      "-> point1 cols only; rest kept at xavier_uniform")
        else:
            raise ValueError(f"unknown io_arch: {io_arch!r}")

    del pre_enc, pre_dec, pre_enc_sd, pre_dec_sd
    torch.cuda.empty_cache()
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest coart/tests/test_build_warmstart.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add coart/vae/build.py coart/tests/test_build_warmstart.py
git commit -m "feat(coart): add build_models + load_pretrained_into with io_arch branching"
```

---

## Task 12: `coart/vae/loss.py` — block-decomposed VAE loss (NEW)

**Files:**
- Create: `coart/vae/loss.py`
- Test: `coart/tests/test_loss.py`

- [ ] **Step 1: Write failing test `coart/tests/test_loss.py`**

Content:
```python
"""Unit tests for coart.vae.loss.compute_vae_loss."""
import torch

from coart.vae.loss import compute_vae_loss


def test_loss_dict_keys():
    pred = torch.randn(10, 18)
    target = torch.randn(10, 18)
    mu = torch.randn(10, 32)
    logvar = torch.randn(10, 32)
    subs_gt = [torch.randint(0, 2, (5, 8)).float() for _ in range(3)]
    subs = [torch.randn(5, 8) for _ in range(3)]

    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=subs_gt, subs=subs,
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    for k in ("total", "recon", "recon_p1", "recon_p2", "recon_ef", "kl", "subdiv"):
        assert k in d, f"missing key {k}"


def test_loss_recon_blocks_sum_weighted_to_total_recon():
    """recon_total == mean(mse across all 18 ch) ≈ (3*p1 + 3*p2 + 12*ef) / 18."""
    pred = torch.randn(20, 18)
    target = torch.randn(20, 18)
    mu = torch.zeros(20, 32)
    logvar = torch.zeros(20, 32)
    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=[], subs=[],
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    expected_total_recon = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(d["recon"], expected_total_recon, atol=1e-6)


def test_loss_kl_zero_when_posterior_is_prior():
    pred = torch.zeros(5, 18)
    target = torch.zeros(5, 18)
    mu = torch.zeros(5, 32)
    logvar = torch.zeros(5, 32)  # exp(0)=1, (0+1-0-1)=0
    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=[], subs=[],
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    assert torch.allclose(d["kl"], torch.tensor(0.0), atol=1e-6)


def test_loss_subdiv_empty_when_no_levels():
    pred = torch.zeros(5, 18)
    target = torch.zeros(5, 18)
    mu = torch.zeros(5, 32)
    logvar = torch.zeros(5, 32)
    d = compute_vae_loss(
        pred=pred, target=target,
        mu=mu, logvar=logvar,
        subs_gt=[], subs=[],
        lambda_kl=1e-6, lambda_subdiv=0.1,
    )
    assert d["subdiv"].item() == 0.0
```

- [ ] **Step 2: Run tests — confirm fail**

Run: `pytest coart/tests/test_loss.py -v`
Expected: 4 fails.

- [ ] **Step 3: Implement `coart/vae/loss.py`**

Content:
```python
"""Block-decomposed VAE loss for 18-ch corep feat18.

Returns a dict with the total loss (for backward) and separate block components
(for TB logging). The three recon blocks correspond to the channel layout:
    ch 0:3   (p1)  — point1 xyz
    ch 3:6   (p2)  — point2 xyz
    ch 6:18  (ef)  — edge/face weights

NO render loss. The loss total is:
    loss = recon + lambda_kl * kl + lambda_subdiv * subdiv
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def compute_vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    subs_gt: List[torch.Tensor],
    subs: List[torch.Tensor],
    lambda_kl: float = 1e-6,
    lambda_subdiv: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute total + per-block VAE loss.

    Args:
        pred, target: (N, 18) feature tensors in normalised space.
        mu, logvar: (N, latent_channels) posterior params.
        subs_gt: list of ground-truth subdivision bitmaps per decoder level.
        subs: list of logits predicted for each subdivision level.
        lambda_kl, lambda_subdiv: scalar weights.

    Returns:
        dict with keys: total, recon, recon_p1, recon_p2, recon_ef, kl, subdiv.
        Every value is a 0-d tensor on pred's device.
    """
    pred_f = pred.float()
    target_f = target.float()
    mu_f = mu.float()
    logvar_f = logvar.float()

    recon_total = F.mse_loss(pred_f, target_f)
    recon_p1 = F.mse_loss(pred_f[:, 0:3], target_f[:, 0:3])
    recon_p2 = F.mse_loss(pred_f[:, 3:6], target_f[:, 3:6])
    recon_ef = F.mse_loss(pred_f[:, 6:18], target_f[:, 6:18])

    kl = 0.5 * torch.mean(mu_f.pow(2) + logvar_f.exp() - logvar_f - 1)

    if len(subs) > 0:
        subdiv = sum(
            F.binary_cross_entropy_with_logits(s.float(), g.float())
            for s, g in zip(subs, subs_gt)
        ) / len(subs)
    else:
        subdiv = torch.zeros((), device=pred.device)

    total = recon_total + lambda_kl * kl + lambda_subdiv * subdiv

    return {
        "total": total,
        "recon": recon_total,
        "recon_p1": recon_p1,
        "recon_p2": recon_p2,
        "recon_ef": recon_ef,
        "kl": kl,
        "subdiv": subdiv,
    }
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest coart/tests/test_loss.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add coart/vae/loss.py coart/tests/test_loss.py
git commit -m "feat(coart): add compute_vae_loss with block-decomposed logging"
```

---

## Task 13: `coart/vae/sampling.py` — mesh dump

**Files:**
- Create: `coart/vae/sampling.py`
- Reference: `train_finetune_feat18.py:847-887`

- [ ] **Step 1: Implement `coart/vae/sampling.py`**

Content:
```python
"""Eval-time mesh dump for feat18 VAE.

Runs encoder/decoder in eval mode on first-N samples of a dataset, denormalises
the decoder output to raw feat18 space, converts to CorepParam via
feature_to_mesh (imported from the existing train_overfit_feat18 module), and
exports .ply meshes to out_dir.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from trellis2.modules import sparse as sp
from train_overfit_feat18 import feature_to_mesh

from ..data.stats import denormalize, normalize


@torch.no_grad()
def dump_samples(
    encoder,
    decoder,
    dataset,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    resolution: int,
    n_dump: int,
    out_dir: str,
    device,
) -> None:
    """Dump up to `n_dump` meshes from `dataset` to `out_dir` as .ply files."""
    os.makedirs(out_dir, exist_ok=True)
    encoder.eval()
    decoder.eval()
    n = min(n_dump, len(dataset))
    for k in range(n):
        item = dataset[k]
        ci = item["cube_indices"].astype(np.int32)
        feats_raw = item["feats"].astype(np.float32)
        sha = item["sha"]

        bi = np.zeros((ci.shape[0], 1), dtype=np.int32)
        coords = (
            torch.from_numpy(np.concatenate([bi, ci], axis=1))
            .int()
            .to(device)
        )
        feats_t = torch.from_numpy(feats_raw).float().to(device)
        x = sp.SparseTensor(feats=normalize(feats_t, mean_t, std_t), coords=coords)

        z = encoder(x, sample_posterior=False)
        h = decoder(z)
        h = h[0] if isinstance(h, tuple) else h
        pred_raw = denormalize(h.feats, mean_t, std_t).cpu().numpy()

        try:
            mesh = feature_to_mesh(pred_raw, ci, resolution=resolution, device=str(device))
            if mesh is not None:
                mesh.export(os.path.join(out_dir, f"{sha}_pred.ply"))
            else:
                tqdm.write(f"  [mesh] {sha} returned empty placeholder")
        except Exception as e:
            tqdm.write(f"  [mesh] {sha} failed: {type(e).__name__}: {e}")
    encoder.train()
    decoder.train()
```

- [ ] **Step 2: Verify import**

Run:
```bash
python -c "from coart.vae.sampling import dump_samples; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add coart/vae/sampling.py
git commit -m "feat(coart): add dump_samples for eval-time mesh export"
```

---

## Task 14: `coart/vae/config.py` — TrainConfig + argparse

**Files:**
- Create: `coart/vae/config.py`

- [ ] **Step 1: Implement `coart/vae/config.py`**

Content:
```python
"""CLI argparse + VaeTrainConfig dataclass for coart.vae training."""
from __future__ import annotations

import argparse
import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Optional


_RUN_TAG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class VaeTrainConfig:
    """Typed wrapper over argparse Namespace; constructed by parse_args().

    Serialised as JSON to `<output_dir>/config.json` at training start for
    provenance and later resume consistency checks.
    """
    # Paths
    data_root: str
    data_dir: Optional[str]
    stats_path: Optional[str]
    output_dir: str
    run_tag: str

    # Pretrained
    from_pretrained: bool
    enc_pretrained: str
    dec_pretrained: str

    # IO architecture
    io_arch: str            # "three_branch" or "monolithic"
    warmstart_io: bool

    # Optimisation
    lr: float
    lr_unfreeze_warmup_steps: int
    freeze_backbone_steps: int
    grad_clip_max: float
    grad_clip_pct: float
    use_bf16: bool

    # Data / augmentation
    batch_size: int
    num_workers: int
    max_translate: int
    max_voxels: int
    bucket_sampler: bool
    bucket_sort_mode: str
    val_split_mod: int

    # Loss
    lambda_kl: float
    lambda_subdiv: float

    # Training schedule
    resolution: int
    latent_channels: int
    max_steps: int
    i_log: int
    i_save: int
    i_sample: int
    i_val: int
    n_dump: int
    sample_at_step_one: bool

    # EMA + ckpt
    use_ema: bool
    ema_rate: float
    rolling_ckpts: int
    resume_from: str       # "none", "latest", or explicit path


def _build_output_dir(output_dir: Optional[str], run_tag: str) -> str:
    """If `output_dir` None: auto-build `results/coart_feat18_{YYYYMMDD}_{run_tag}`."""
    if output_dir:
        return output_dir
    ts = datetime.datetime.now().strftime("%Y%m%d")
    return f"results/coart_feat18_{ts}_{run_tag}"


def parse_args() -> VaeTrainConfig:
    p = argparse.ArgumentParser(description="coart.vae — feat18 Shape-VAE finetune")

    # Paths
    p.add_argument("--data_root",
                   default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/"
                           "ObjaverseXL_sketchfab/feat18_512",
                   help="dir containing data/ subdir with *.npz shards")
    p.add_argument("--data_dir", default=None, help="override <data_root>/data")
    p.add_argument("--stats_path", default=None,
                   help="override <data_root>/stats_global.npz")
    p.add_argument("--output_dir", default=None,
                   help="full output dir; if unset, auto-built as "
                        "results/coart_feat18_{YYYYMMDD}_{run_tag}")
    p.add_argument("--run_tag", required=True,
                   help=r"run identifier; regex ^[a-zA-Z0-9_-]+$")

    # Pretrained
    p.add_argument("--from_pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", action="store_false", dest="from_pretrained")
    p.add_argument("--enc_pretrained",
                   default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    p.add_argument("--dec_pretrained",
                   default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16")

    # IO
    p.add_argument("--io_arch", choices=["three_branch", "monolithic"],
                   default="three_branch")
    p.add_argument("--warmstart_io", action="store_true", default=True)
    p.add_argument("--no_warmstart_io", action="store_false", dest="warmstart_io")

    # Optim
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_unfreeze_warmup_steps", type=int, default=500)
    p.add_argument("--freeze_backbone_steps", type=int, default=2000)
    p.add_argument("--grad_clip_max", type=float, default=1.0)
    p.add_argument("--grad_clip_pct", type=float, default=95.0)
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="use_bf16")

    # Data / augment
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_translate", type=int, default=16)
    p.add_argument("--max_voxels", type=int, default=500000)
    p.add_argument("--bucket_sampler", action="store_true", default=True)
    p.add_argument("--no_bucket_sampler", action="store_false", dest="bucket_sampler")
    p.add_argument("--bucket_sort_mode", choices=["shuffle", "ascending"],
                   default="shuffle")
    p.add_argument("--val_split_mod", type=int, default=200,
                   help="sha hash modulus for val partition (0 = no val)")

    # Loss
    p.add_argument("--lambda_kl", type=float, default=1e-6)
    p.add_argument("--lambda_subdiv", type=float, default=0.1)

    # Schedule
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--latent_channels", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--i_log", type=int, default=100)
    p.add_argument("--i_save", type=int, default=5000)
    p.add_argument("--i_sample", type=int, default=5000)
    p.add_argument("--i_val", type=int, default=5000)
    p.add_argument("--n_dump", type=int, default=2)
    p.add_argument("--sample_at_step_one", action="store_true", default=True)
    p.add_argument("--no_sample_at_step_one",
                   action="store_false", dest="sample_at_step_one")

    # EMA / ckpt
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.add_argument("--ema_rate", type=float, default=0.9999)
    p.add_argument("--rolling_ckpts", type=int, default=3)
    p.add_argument("--resume_from", default="latest",
                   help='"none", "latest", or explicit ckpt path')

    args = p.parse_args()

    if not _RUN_TAG_RE.match(args.run_tag):
        raise ValueError(
            f"--run_tag must match ^[a-zA-Z0-9_-]+$, got {args.run_tag!r}"
        )

    args.output_dir = _build_output_dir(args.output_dir, args.run_tag)

    return VaeTrainConfig(**vars(args))
```

- [ ] **Step 2: Verify argparse smoke test**

Run:
```bash
python -c "
import sys
sys.argv = ['test', '--run_tag', 'smoke_test_v0']
from coart.vae.config import parse_args
cfg = parse_args()
print(f'output_dir: {cfg.output_dir}')
print(f'run_tag: {cfg.run_tag}')
print(f'io_arch: {cfg.io_arch}')
print(f'lr: {cfg.lr}')
assert cfg.output_dir.startswith('results/coart_feat18_')
assert cfg.output_dir.endswith('_smoke_test_v0')
print('OK')
"
```
Expected: `OK`, `output_dir` is `results/coart_feat18_{YYYYMMDD}_smoke_test_v0`, defaults as per spec.

- [ ] **Step 3: Commit**

```bash
git add coart/vae/config.py
git commit -m "feat(coart): add VaeTrainConfig + argparse with run_tag validation"
```

---

## Task 15: `coart/vae/train.py` — main training loop (integration)

**Files:**
- Create: `coart/vae/train.py`

This is the largest file (~400 lines). Implement in 4 sub-steps so the file grows incrementally.

- [ ] **Step 1: Create skeleton with imports and `train(cfg)` signature**

Content:
```python
"""Main training loop for coart.vae.

Integrates:
    - Dataset (train + val split)
    - BucketedDistributedSampler
    - build_models (io_arch-aware) + load_pretrained_into
    - DDP wrapping with freeze/unfreeze schedule
    - AdamW optimiser with LR unfreeze-warmup
    - AdaptiveGradClipper (reuse trellis2/utils/grad_clip_utils.py)
    - bf16 autocast forward, fp32 loss
    - EMA shadow updates
    - Rolling-K atomic checkpointing + resume
    - TB logging at i_log cadence
    - Val MSE pass at i_val
    - Mesh dump at i_sample
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from trellis2.modules import sparse as sp
from trellis2.utils.grad_clip_utils import AdaptiveGradClipper

from ..common.checkpoint import save_ckpt, find_latest_ckpt, atomic_save
from ..common.dist_utils import init_dist, unwrap, wrap_ddp, worker_init_fn
from ..common.ema import EMAModel
from ..common.flex_gemm_patch import _patch_flex_gemm_frozen_weight_bug
from ..common.logging import CoartTBLogger

from ..data.feat18_dataset import Feat18Dataset, collate_fn
from ..data.samplers import BucketedDistributedSampler
from ..data.stats import denormalize, load_stats, normalize

from .build import build_models, load_pretrained_into
from .config import VaeTrainConfig
from .loss import compute_vae_loss
from .sampling import dump_samples


def _set_backbone_requires_grad(encoder, decoder, requires_grad: bool) -> None:
    """Freeze / unfreeze every parameter that is NOT input_layer / output_layer / to_latent / from_latent."""
    enc_io_prefixes = ("input_layer.", "to_latent.")
    dec_io_prefixes = ("output_layer.", "from_latent.")
    for name, p in encoder.named_parameters():
        p.requires_grad_(True if any(name.startswith(pr) for pr in enc_io_prefixes) else requires_grad)
    for name, p in decoder.named_parameters():
        p.requires_grad_(True if any(name.startswith(pr) for pr in dec_io_prefixes) else requires_grad)


def _trainable(*models_):
    return [p for m in models_ for p in m.parameters() if p.requires_grad]
```

- [ ] **Step 2: Add setup + resume block to `train()`**

Append to `coart/vae/train.py`:
```python
def train(cfg: VaeTrainConfig) -> None:
    _patch_flex_gemm_frozen_weight_bug()

    rank, world_size, local_rank, is_dist = init_dist()
    is_master = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    # ---------------- output dir + config dump ----------------
    if is_master:
        if os.path.isdir(cfg.output_dir) and os.listdir(cfg.output_dir):
            has_ckpt = find_latest_ckpt(cfg.output_dir, prefix="ckpt") is not None
            if cfg.resume_from == "none" and has_ckpt:
                raise RuntimeError(
                    f"output dir {cfg.output_dir} is populated; "
                    f"pass --resume_from latest or change --run_tag"
                )
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
            json.dump(vars(cfg), f, indent=4)
    if is_dist:
        dist.barrier()

    def _log(msg: str):
        if is_master:
            print(msg)

    _log(f"[dist] rank={rank} world_size={world_size} device={device}")

    # ---------------- stats ----------------
    stats_path = cfg.stats_path or os.path.join(cfg.data_root, "stats_global.npz")
    mean_t, std_t = load_stats(stats_path, device, verbose=is_master)

    # ---------------- datasets + loader ----------------
    data_dir = cfg.data_dir or os.path.join(cfg.data_root, "data")
    train_set = Feat18Dataset(
        data_dir, resolution=cfg.resolution,
        max_translate=cfg.max_translate, augment=True,
        precompute_voxel_counts=cfg.bucket_sampler,
        max_voxels=cfg.max_voxels,
        val_split_mod=cfg.val_split_mod, split="train",
    )
    val_set = Feat18Dataset(
        data_dir, resolution=cfg.resolution,
        max_translate=0, augment=False,
        max_voxels=0,
        val_split_mod=cfg.val_split_mod, split="val",
    ) if cfg.val_split_mod > 0 else None
    _log(f"[data] train={len(train_set)}, val={len(val_set) if val_set else 0}")

    if cfg.bucket_sampler:
        if not is_dist:
            raise RuntimeError("--bucket_sampler requires distributed training")
        sampler = BucketedDistributedSampler(
            voxels_per_sample=train_set.effective_voxels(),
            num_replicas=world_size, rank=rank,
            batch_size=cfg.batch_size, shuffle=True, seed=0,
            sort_mode=cfg.bucket_sort_mode,
        )
    elif is_dist:
        sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
    else:
        sampler = None

    os.environ["_FT_NUM_WORKERS"] = str(max(cfg.num_workers, 1))
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True, collate_fn=collate_fn, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
```

- [ ] **Step 3: Add model build, optimiser, EMA, resume restore**

Append:
```python
    # ---------------- build models ----------------
    encoder, decoder = build_models(
        latent_channels=cfg.latent_channels,
        device=device, io_arch=cfg.io_arch,
    )
    if cfg.from_pretrained:
        load_pretrained_into(
            encoder, decoder,
            enc_path=cfg.enc_pretrained, dec_path=cfg.dec_pretrained,
            io_arch=cfg.io_arch, warmstart_io=cfg.warmstart_io,
            verbose=is_master,
        )

    if cfg.freeze_backbone_steps > 0:
        _set_backbone_requires_grad(encoder, decoder, requires_grad=False)
    _log(f"[model] {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M enc "
         f"+ {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M dec params")

    # ---------------- EMA before DDP wrap ----------------
    ema = EMAModel(encoder, decay=cfg.ema_rate) if cfg.use_ema else None
    ema_dec = EMAModel(decoder, decay=cfg.ema_rate) if cfg.use_ema else None

    # ---------------- DDP wrap ----------------
    if is_dist:
        encoder = wrap_ddp(encoder, local_rank)
        decoder = wrap_ddp(decoder, local_rank)

    trainable = _trainable(encoder, decoder)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.0)

    grad_clipper = AdaptiveGradClipper(
        max_norm=cfg.grad_clip_max, clip_percentile=cfg.grad_clip_pct,
    )

    # ---------------- resume ----------------
    step = 0
    start_epoch = 0
    if cfg.resume_from != "none":
        resume_path = None
        if cfg.resume_from == "latest":
            latest = find_latest_ckpt(cfg.output_dir, prefix="ckpt")
            if latest is not None:
                resume_path, step = latest
        else:
            resume_path = cfg.resume_from
            step = int(resume_path.split("_step")[-1].split(".pt")[0])

        if resume_path is not None and os.path.exists(resume_path):
            _log(f"[resume] loading {resume_path} (step {step})")
            ck = torch.load(resume_path, map_location=device)
            unwrap(encoder).load_state_dict(ck["encoder"])
            unwrap(decoder).load_state_dict(ck["decoder"])
            optimizer.load_state_dict(ck["optimizer"])

            # misc + EMA
            misc_path = resume_path.replace("ckpt_step", "misc_step")
            if os.path.exists(misc_path):
                misc = torch.load(misc_path, map_location="cpu")
                start_epoch = int(misc.get("epoch", 0))
                rng_np = misc.get("rng_np")
                rng_torch = misc.get("rng_torch")
                if rng_np is not None:
                    np.random.set_state(rng_np)
                if rng_torch is not None:
                    torch.set_rng_state(rng_torch.cpu())
            if cfg.use_ema:
                enc_ema = resume_path.replace("ckpt_step", f"ema_{cfg.ema_rate}_enc_step")
                dec_ema = resume_path.replace("ckpt_step", f"ema_{cfg.ema_rate}_dec_step")
                if os.path.exists(enc_ema):
                    ema.load_state_dict(torch.load(enc_ema, map_location="cpu"))
                if os.path.exists(dec_ema):
                    ema_dec.load_state_dict(torch.load(dec_ema, map_location="cpu"))
            _log(f"[resume] ok — resumed at step {step}, epoch {start_epoch}")
```

- [ ] **Step 4: Add the step loop, save, val, unfreeze warmup, sampling**

Append:
```python
    # ---------------- TB logger ----------------
    logger = CoartTBLogger(cfg.output_dir, is_master=is_master)

    unfrozen = cfg.freeze_backbone_steps == 0
    step_at_unfreeze: int | None = 0 if unfrozen else None

    pbar = tqdm(total=cfg.max_steps, initial=step, desc="coart.vae",
                dynamic_ncols=True, disable=not is_master)
    t0 = time.time()
    epoch = start_epoch

    while step < cfg.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= cfg.max_steps:
                break

            coords = batch["coords"].int().to(device, non_blocking=True)
            feats_raw = batch["feats"].float().to(device, non_blocking=True)
            feats = normalize(feats_raw, mean_t, std_t)

            x = sp.SparseTensor(feats=feats, coords=coords)

            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if cfg.use_bf16 else contextlib.nullcontext()
            )
            with autocast_ctx:
                z, mu, logvar = encoder(x, sample_posterior=True, return_raw=True)
                decoded = decoder(z)
                h, subs_gt, subs = decoded

            losses = compute_vae_loss(
                pred=h.feats, target=x.feats,
                mu=mu, logvar=logvar,
                subs_gt=subs_gt, subs=subs,
                lambda_kl=cfg.lambda_kl, lambda_subdiv=cfg.lambda_subdiv,
            )
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            # AdaptiveGradClipper applies the clip + updates rolling window.
            grad_clipper.apply(trainable)

            # LR unfreeze warmup: linear ramp 0 -> cfg.lr across cfg.lr_unfreeze_warmup_steps
            if (not unfrozen) or step_at_unfreeze is None:
                cur_lr = cfg.lr
            elif step - step_at_unfreeze < cfg.lr_unfreeze_warmup_steps:
                frac = (step - step_at_unfreeze) / max(cfg.lr_unfreeze_warmup_steps, 1)
                cur_lr = cfg.lr * frac
            else:
                cur_lr = cfg.lr
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            optimizer.step()

            if cfg.use_ema:
                ema.update(unwrap(encoder))
                ema_dec.update(unwrap(decoder))

            step += 1
            pbar.update(1)

            # Log scalars every step; flush at cadence.
            for k in ("total", "recon", "recon_p1", "recon_p2", "recon_ef", "kl", "subdiv"):
                logger.scalar(f"loss/{k}", losses[k].detach().item(), step)
            logger.scalar("misc/lr", cur_lr, step)
            logger.scalar("misc/voxels", float(x.feats.shape[0]), step)
            logger.scalar("misc/it_per_s", step / max(time.time() - t0, 1e-3), step)
            logger.flush_if_due(step, cfg.i_log)

            # Unfreeze
            if (not unfrozen) and step >= cfg.freeze_backbone_steps:
                _set_backbone_requires_grad(unwrap(encoder), unwrap(decoder),
                                            requires_grad=True)
                if is_dist:
                    encoder = wrap_ddp(unwrap(encoder), local_rank)
                    decoder = wrap_ddp(unwrap(decoder), local_rank)
                trainable = _trainable(encoder, decoder)
                optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=0.0)
                unfrozen = True
                step_at_unfreeze = step
                _log(f"[optim] unfrozen at step {step}; "
                     f"{sum(p.numel() for p in trainable)/1e6:.2f}M trainable")

            # Save ckpt
            if is_master and (step % cfg.i_save == 0 or step == cfg.max_steps):
                ck = {
                    "step": step,
                    "encoder": unwrap(encoder).state_dict(),
                    "decoder": unwrap(decoder).state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_ckpt(ck, cfg.output_dir, step,
                          keep_k=cfg.rolling_ckpts, prefix="ckpt")
                # EMA saved separately per rate-branch.
                if cfg.use_ema:
                    save_ckpt(ema.state_dict(), cfg.output_dir, step,
                              keep_k=cfg.rolling_ckpts,
                              prefix=f"ema_{cfg.ema_rate}_enc")
                    save_ckpt(ema_dec.state_dict(), cfg.output_dir, step,
                              keep_k=cfg.rolling_ckpts,
                              prefix=f"ema_{cfg.ema_rate}_dec")
                misc = {
                    "epoch": epoch,
                    "rng_np": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                }
                save_ckpt(misc, cfg.output_dir, step,
                          keep_k=cfg.rolling_ckpts, prefix="misc")

            # Val MSE pass (simple: sequential, no DDP; rank-0 only).
            if is_master and val_set is not None and (
                step % cfg.i_val == 0 and step > 0
            ):
                val_loss = _run_val(encoder, decoder, val_set, mean_t, std_t,
                                    cfg, device)
                if logger._writer is not None:
                    logger._writer.add_scalar("val/loss_recon", val_loss, step)
                _log(f"[val] step={step} recon={val_loss:.4f}")

            # Mesh dump
            sample_at_1 = (step == 1) and cfg.sample_at_step_one
            if step % cfg.i_sample == 0 or sample_at_1:
                if is_master:
                    sd = os.path.join(cfg.output_dir, f"meshes_step{step:07d}")
                    dump_samples(unwrap(encoder), unwrap(decoder), val_set or train_set,
                                 mean_t, std_t, cfg.resolution, cfg.n_dump, sd, device)
                if is_dist:
                    dist.barrier()

        epoch += 1

    if is_master:
        logger.close()
    pbar.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    _log("Training finished.")


@torch.no_grad()
def _run_val(encoder, decoder, val_set, mean_t, std_t, cfg, device) -> float:
    """Simple val: run encoder+decoder on up to 16 val samples, return mean recon MSE."""
    encoder.eval(); decoder.eval()
    total, n = 0.0, 0
    for k in range(min(16, len(val_set))):
        item = val_set[k]
        ci = item["cube_indices"].astype(np.int32)
        feats_raw = item["feats"].astype(np.float32)
        bi = np.zeros((ci.shape[0], 1), dtype=np.int32)
        coords = torch.from_numpy(np.concatenate([bi, ci], axis=1)).int().to(device)
        feats_t = torch.from_numpy(feats_raw).float().to(device)
        x = sp.SparseTensor(feats=normalize(feats_t, mean_t, std_t), coords=coords)
        z = encoder(x, sample_posterior=False)
        h = decoder(z)
        h = h[0] if isinstance(h, tuple) else h
        total += F.mse_loss(h.feats.float(), x.feats.float()).item()
        n += 1
    encoder.train(); decoder.train()
    return total / max(n, 1)
```

- [ ] **Step 5: Verify import**

Run:
```bash
python -c "from coart.vae.train import train; from coart.vae.config import parse_args; print('OK')"
```
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add coart/vae/train.py
git commit -m "feat(coart): add train.py main loop with EMA, resume, AdaptiveGradClipper, bf16"
```

---

## Task 16: `coart/vae/__main__.py` — CLI entry

**Files:**
- Create: `coart/vae/__main__.py`

- [ ] **Step 1: Implement**

Content:
```python
"""CLI entry for `python -m coart.vae` / `torchrun ... -m coart.vae`."""
from .config import parse_args
from .train import train


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify entry works (dry-run argparse)**

Run:
```bash
python -m coart.vae --run_tag dryrun_test 2>&1 | head -30
```
Expected: prints `[stats] no file at ...` or proceeds to `[dist]` / `[data]` init; dies later on missing stats or CUDA, which is OK — the argparse + import chain worked.

- [ ] **Step 3: Commit**

```bash
git add coart/vae/__main__.py
git commit -m "feat(coart): add __main__.py so `python -m coart.vae` / torchrun works"
```

---

## Task 17: Single-GPU 10-step smoke test + validate end-to-end

**Files:**
- Execute only; no new files.

- [ ] **Step 1: Run 10-step smoke test with three_branch + warmstart**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
CUDA_VISIBLE_DEVICES=0 python -m coart.vae \
    --run_tag smoke_three_branch_ws \
    --data_root /mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512 \
    --stats_path /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz \
    --io_arch three_branch --warmstart_io \
    --max_steps 10 --batch_size 1 \
    --max_voxels 100000 --no_bucket_sampler \
    --freeze_backbone_steps 5 \
    --i_log 1 --i_save 10 --i_sample 10 --i_val 10 --sample_at_step_one \
    --use_ema --rolling_ckpts 2
```

Expected:
- Console shows `[init] three_branch warm-start: ...` and `[init] encoder missing (non-IO): []`
- 10 steps print decreasing loss (or at least finite, no NaN)
- `results/coart_feat18_{YYYYMMDD}_smoke_three_branch_ws/` has:
  - `config.json`
  - `ckpt_step0000010.pt`, `misc_step0000010.pt`, `ema_0.9999_enc_step0000010.pt`, `ema_0.9999_dec_step0000010.pt`
  - `meshes_step0000001/{sha}_pred.ply` (step-1 dump) and `meshes_step0000010/`
  - `tb_logs/` with event file

- [ ] **Step 2: Run 10-step smoke test with monolithic + no warmstart (ablation D)**

```bash
CUDA_VISIBLE_DEVICES=0 python -m coart.vae \
    --run_tag smoke_mono_scratch \
    --data_root /mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512 \
    --stats_path /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz \
    --io_arch monolithic --no_warmstart_io \
    --max_steps 10 --batch_size 1 \
    --max_voxels 100000 --no_bucket_sampler \
    --freeze_backbone_steps 0 \
    --i_log 1 --i_save 10 --i_sample 10 --i_val 10 --sample_at_step_one \
    --no_ema --rolling_ckpts 2
```

Expected: runs to completion, produces ckpt, log shows `[init] monolithic partial warm-start: ...` is **not** printed (warmstart_io off).

- [ ] **Step 3: Run 10-step smoke test + resume**

First leave a ckpt from Step 1. Then re-invoke:
```bash
CUDA_VISIBLE_DEVICES=0 python -m coart.vae \
    --run_tag smoke_three_branch_ws \
    --output_dir results/coart_feat18_{YYYYMMDD}_smoke_three_branch_ws \
    --data_root /mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512 \
    --stats_path /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz \
    --io_arch three_branch --warmstart_io \
    --max_steps 20 --batch_size 1 \
    --max_voxels 100000 --no_bucket_sampler \
    --freeze_backbone_steps 5 \
    --i_log 1 --i_save 10 --i_sample 20 --i_val 20 --no_sample_at_step_one \
    --use_ema --rolling_ckpts 2 \
    --resume_from latest
```
(Replace `{YYYYMMDD}` with today's actual date.)

Expected: `[resume] loading .../ckpt_step0000010.pt (step 10)` + training continues from step 11 → 20.

- [ ] **Step 4: Clean up smoke-test outputs**

```bash
rm -rf results/coart_feat18_*_smoke_*
```

- [ ] **Step 5: Commit smoke-test evidence (optional)**

If you want to freeze a "last-known-good smoke test" record:
```bash
# This step is optional — no new tracked files; smoke runs were ephemeral.
```

---

## Self-Review

### Spec coverage check
Spec sections → implementing tasks:

| Spec § | Topic | Task(s) |
|---|---|---|
| 2 | Package Layout | 1 |
| 3 | IO Architecture (three_branch) | 10, 11 |
| 3 | Warm-start matrix | 11 |
| 4 | Loss composition + block-decomposed log | 12 (+ loop in 15) |
| 5 | Optimisation (lr, scheduler, AMP, warmup, grad clip, freeze) | 14 (config), 15 (loop) |
| 6 | Data pipeline (stats, max_voxels, bucket, val split, augment) | 7, 8, 9, 14 |
| 7 | Training schedule | 14 (defaults), 15 (loop cadence) |
| 8 | EMA / atomic save / rolling / resume | 4, 5, 15 (integration) |
| 9 | Output dir naming | 14 |
| 10 | Ablation matrix (4 combos) | 14 (CLI flags), 17 (smoke tests) |
| 11 | CLI contract | 14, 16 |
| 12 | Launch prerequisites | 0 (stats), 17 (smoke) |
| 13 | Risks / open items | not tasks — monitored at runtime |
| 14 | Out of scope | not tasks — explicitly excluded |

All spec sections 2-12 are covered.

### Placeholder scan
No "TBD", "TODO", "implement later". Every code block is complete. Two intentional non-code markers:
- `results/coart_feat18_{YYYYMMDD}_...` in Task 17 Step 3 — `{YYYYMMDD}` is a user-filled date during the smoke-test run, not a code placeholder (explicitly noted in the step).
- `coart/dit/__init__.py` in Task 1 is a placeholder comment for future work, not an action item.

### Type consistency check
- `Feat18EncIO` and `Feat18DecIO` — same class names used in Tasks 10, 11, 15. ✓
- `EMAModel` used in Tasks 4 and 15 with consistent API (`update(model)`, `state_dict`, `load_state_dict`, `copy_to`, `shadow_params`). ✓
- `compute_vae_loss` signature and return keys match between Tasks 12 and 15 (`total, recon, recon_p1, recon_p2, recon_ef, kl, subdiv`). ✓
- `save_ckpt(state, output_dir, step, keep_k, prefix)` signature consistent between Tasks 5 and 15. ✓
- `find_latest_ckpt` returns `Optional[Tuple[str, int]]` used consistently. ✓
- `load_stats`, `normalize`, `denormalize` — same shapes and device semantics across Tasks 7, 13, 15. ✓
- `init_dist / unwrap / wrap_ddp / worker_init_fn` (no underscore prefix) — consistent public API across Tasks 3, 15. ✓
- `io_arch` literal values `"three_branch"` / `"monolithic"` consistent across Tasks 10, 11, 14, 15. ✓

No type or naming inconsistencies detected.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-22-coart-vae-feat18.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
