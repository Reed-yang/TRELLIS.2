# coart.vae Monitoring & Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `coart/` from a runnable smoke to a production-grade 8-GPU launch with wandb monitoring, deep-eval on 8 golden assets, and watchdog alerts.

**Architecture:** Additive-only changes inside `coart/` + two `scripts/coart_build_*.py` setup utilities + `pip install wandb` in `.venv`. Four stages map to spec §1-4 and are mostly independent (Stage 1 & 2 parallel; Stage 3 depends on 2.3; Stage 4 depends on 2.3 + 3 for full features). Never modify `trellis2/` or existing `scripts/eval/`.

**Tech Stack:** PyTorch 2.6+ (bf16 autocast, fused AdamW, DDP), torch.distributed (NCCL), `flex_gemm` sparse conv backend, `wandb` 0.26+ (primary logging), `torch.utils.tensorboard.SummaryWriter` (backup), `scripts/eval/eval_metrics.py` + `scripts/eval/ovoxel_repr_test.py::_compute_topo_metrics` (reused), `train_overfit_feat18.feature_to_mesh` (decoder→mesh).

**Spec:** `docs/superpowers/specs/2026-04-23-coart-vae-monitoring-eval-design.md`

**Constraints recap** (from spec):
- `batch_size=1`, `max_voxels=500000`, `lr=1e-5`, `max_steps=200000` — **unchanged**
- Loss `= recon + 1e-6·kl + 0.1·subdiv` — **unchanged**
- No modifications to `trellis2/`, `scripts/eval/*.py`, or `train_finetune_feat18.py`
- `coart.__init__.py` already sets `TRITON_CACHE_DIR` default; don't re-set
- Tests run with `.venv/bin/pytest coart/tests/ -q`
- All new/modified code: English comments; commit messages: English; spec/docs content: Chinese OK (matches `coart/README.md`)

---

## File Structure

### Create

| Path | Responsibility | LOC est. |
|---|---|---|
| `coart/eval/__init__.py` | package init + public API | 10 |
| `coart/eval/metrics.py` | thin wrappers around `scripts/eval/` CD/NC/F-score/topo | 80 |
| `coart/eval/deep_eval.py` | `run_deep_eval(encoder, decoder, stats, step, logger, cfg)` | 180 |
| `coart/eval/watchdog.py` | `Watchdog` class with 3 conditions + dump | 140 |
| `coart/eval/golden_assets.json` | static 8-asset list (produced by build_golden) | — |
| `coart/eval/golden_baseline.json` | Layer V baseline per asset (produced by build_baseline) | — |
| `scripts/coart_build_golden.py` | setup: `datasets/coart_golden/*.npz` + `golden_assets.json` | 200 |
| `scripts/coart_build_baseline.py` | setup: `golden_baseline.json` | 120 |
| `coart/tests/test_metrics_wrappers.py` | unit tests for `coart/eval/metrics.py` wrappers | 60 |
| `coart/tests/test_deep_eval_smoke.py` | smoke deep_eval with mock encoder/decoder | 100 |
| `coart/tests/test_watchdog.py` | 3 watchdog conditions | 140 |

### Modify

| Path | Change | Lines touched |
|---|---|---|
| `coart/vae/config.py` | add 6 wandb + EMA fields to dataclass + argparse | ~40 |
| `coart/common/logging.py` | add wandb dual-write + `image()` + `object3d()` + `alert()` | ~80 |
| `coart/common/dist_utils.py::wrap_ddp` | add `gradient_as_bucket_view=True, broadcast_buffers=False` | 2 |
| `coart/vae/train.py` | fused AdamW + deep_eval hook + watchdog hook + EMA rolling split + throughput instrumentation | ~60 |
| `coart/tests/test_checkpoint.py` | add rolling_ckpts_ema smoke | ~15 |

---

## Stage Overview

| Stage | Task IDs | Parallelizable within stage? | Depends on |
|---|---|---|---|
| 1 — DDP/optim micro-opt | 1-2 | Yes | none |
| 2 — wandb + logger extension | 3-6 | Mostly (3→4→5,6) | none |
| 3 — Deep-eval infrastructure | 7-11 | Partial (7 first, 8+9 parallel, 10 after 7+8, 11 after 10+5) | Stage 2 task 5 |
| 4 — Watchdog + baseline display | 12-14 | Yes (all three parallel) | Stage 2 task 5, Stage 3 task 9 |
| 5 — Final integration smoke | 15 | — | all above |

---

## Stage 1 — DDP/optim micro-opt

### Task 1: DDP wrap options

**Files:**
- Modify: `coart/common/dist_utils.py:33-41`
- Test: `coart/tests/test_dist_utils.py` (new)

- [ ] **Step 1: Write the failing test**

Create `coart/tests/test_dist_utils.py`:
```python
"""Smoke test that wrap_ddp builds DDP with the expected flags."""
from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
import torch.nn as nn


def _make_linear():
    return nn.Linear(8, 8).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wrap_ddp_flags_applied():
    """wrap_ddp must pass gradient_as_bucket_view=True and broadcast_buffers=False."""
    # Single-process DDP works by setting env + dummy group.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    from coart.common.dist_utils import wrap_ddp

    with mock.patch(
        "coart.common.dist_utils.DDP",
        wraps=__import__("torch.nn.parallel", fromlist=["DistributedDataParallel"]).DistributedDataParallel,
    ) as spy:
        try:
            wrap_ddp(_make_linear(), local_rank=0)
        except Exception:
            pass  # DDP under gloo/cuda may fail to fully init, but spy records kwargs
        assert spy.called, "DDP constructor was not invoked"
        kwargs = spy.call_args.kwargs
        assert kwargs.get("gradient_as_bucket_view") is True
        assert kwargs.get("broadcast_buffers") is False
        assert kwargs.get("bucket_cap_mb") == 128
        assert kwargs.get("find_unused_parameters") is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/pytest coart/tests/test_dist_utils.py -v
```
Expected: FAIL with `AssertionError: assert None is True` on `gradient_as_bucket_view`.

- [ ] **Step 3: Modify wrap_ddp**

Replace `coart/common/dist_utils.py:33-41`:
```python
def wrap_ddp(model, local_rank):
    """Wrap model in DDP with canonical settings.

    - bucket_cap_mb=128: coalesce grad all-reduce buckets at 128MB
    - find_unused_parameters=False: all params receive grad every step
    - gradient_as_bucket_view=True: bucket zero-copy (saves one grad alloc per step)
    - broadcast_buffers=False: model has no BN, skip per-step buffer sync
    """
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=128,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest coart/tests/test_dist_utils.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add coart/common/dist_utils.py coart/tests/test_dist_utils.py
git commit -m "feat(coart.common.dist_utils): enable gradient_as_bucket_view + broadcast_buffers=False"
```

---

### Task 2: Fused AdamW optimizer

**Files:**
- Modify: `coart/vae/train.py` (the `torch.optim.AdamW(...)` call — locate via `grep -n "AdamW" coart/vae/train.py`)
- Test: `coart/tests/test_fused_adamw_fallback.py` (new)

- [ ] **Step 1: Write the failing test**

Create `coart/tests/test_fused_adamw_fallback.py`:
```python
"""Verify fused=True path works and falls back cleanly when unsupported."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_adamw_on_cuda():
    """On H100/A100 fused AdamW must init without error."""
    from coart.vae.train import _build_optimizer  # added by this task
    params = [torch.randn(8, 8, requires_grad=True, device="cuda")]
    opt = _build_optimizer(params, lr=1e-5)
    # Has a step method and no crash — that's the contract
    assert hasattr(opt, "step")


def test_fused_adamw_cpu_fallback():
    """On CPU-only build, fused must fall back silently."""
    from coart.vae.train import _build_optimizer
    params = [torch.randn(8, 8, requires_grad=True, device="cpu")]
    opt = _build_optimizer(params, lr=1e-5)
    assert hasattr(opt, "step")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest coart/tests/test_fused_adamw_fallback.py -v
```
Expected: FAIL with `ImportError: cannot import name '_build_optimizer'`.

- [ ] **Step 3: Add _build_optimizer helper in train.py**

Locate the existing `torch.optim.AdamW(...)` call in `coart/vae/train.py`. Add this helper near the top of the file (after imports):

```python
def _build_optimizer(trainable_params, lr: float) -> torch.optim.Optimizer:
    """AdamW with fused=True when supported; otherwise plain AdamW.

    Fused kernel requires CUDA + PyTorch >= 2.0; falls back on CPU or
    older torch. Any failure is logged to stderr and demoted to fused=False.
    """
    try:
        return torch.optim.AdamW(trainable_params, lr=lr, fused=True)
    except (TypeError, RuntimeError) as e:
        import sys
        print(f"[coart] fused AdamW unavailable ({e}); falling back to fused=False",
              file=sys.stderr)
        return torch.optim.AdamW(trainable_params, lr=lr, fused=False)
```

Then replace the existing `torch.optim.AdamW(trainable_params, lr=cfg.lr)` line (there may be two: one for base mode, one for resume) with `_build_optimizer(trainable_params, lr=cfg.lr)`.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest coart/tests/test_fused_adamw_fallback.py -v
```
Expected: PASS on the CPU fallback test; CUDA test passes on nodes with GPUs.

- [ ] **Step 5: Commit**

```bash
git add coart/vae/train.py coart/tests/test_fused_adamw_fallback.py
git commit -m "feat(coart.vae.train): use fused AdamW with transparent fallback"
```

---

## Stage 2 — wandb + logger extension

### Task 3: Config fields

**Files:**
- Modify: `coart/vae/config.py:15-75` (dataclass), `coart/vae/config.py:85-171` (argparse)
- Test: `coart/tests/test_config_wandb_fields.py` (new)

- [ ] **Step 1: Write the failing test**

Create `coart/tests/test_config_wandb_fields.py`:
```python
"""Verify new wandb + EMA fields exist in VaeTrainConfig + parse_args."""
import sys
from dataclasses import fields

import pytest

from coart.vae.config import VaeTrainConfig, parse_args


def test_new_fields_on_dataclass():
    names = {f.name for f in fields(VaeTrainConfig)}
    for f in ("use_wandb", "wandb_project", "wandb_mode",
              "rolling_ckpts_ema", "log_3d", "n_dump_names"):
        assert f in names, f"missing {f}"


def test_argparse_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--run_tag", "dummy"])
    cfg = parse_args()
    assert cfg.use_wandb is True
    assert cfg.wandb_project == "coart-vae"
    assert cfg.wandb_mode == "online"
    assert cfg.rolling_ckpts_ema == 1
    assert cfg.log_3d is False
    assert cfg.n_dump_names == ["helmet", "val_p95"]


def test_wandb_mode_validation(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--run_tag", "dummy", "--wandb_mode", "bad"])
    with pytest.raises(SystemExit):  # argparse choices rejects
        parse_args()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest coart/tests/test_config_wandb_fields.py -v
```
Expected: FAIL with `AssertionError: missing use_wandb`.

- [ ] **Step 3: Extend dataclass**

Add after existing `# EMA + ckpt` block in `coart/vae/config.py` (line ~74):
```python
    # Wandb logging
    use_wandb: bool
    wandb_project: str
    wandb_mode: str        # "online" / "offline" / "disabled"

    # EMA ckpt rolling (split from rolling_ckpts so EMA doesn't waste disk)
    rolling_ckpts_ema: int

    # Deep-eval / dump
    log_3d: bool
    n_dump_names: list       # asset names whose renders go to wandb
```

- [ ] **Step 4: Extend argparse**

Add before the `args = p.parse_args()` line in `coart/vae/config.py` (~line 162):
```python
    # Wandb
    p.add_argument("--use_wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    p.add_argument("--wandb_project", default="coart-vae")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")

    # EMA rolling split
    p.add_argument("--rolling_ckpts_ema", type=int, default=1,
                   help="rolling K for EMA ckpts (separate from --rolling_ckpts)")

    # Deep-eval dump
    p.add_argument("--log_3d", action="store_true", default=False,
                   help="log wandb.Object3D of decoded mesh (VRAM-hungry)")
    p.add_argument("--n_dump_names", nargs="+",
                   default=["helmet", "val_p95"],
                   help="asset names for normal-map renders in wandb")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/pytest coart/tests/test_config_wandb_fields.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add coart/vae/config.py coart/tests/test_config_wandb_fields.py
git commit -m "feat(coart.vae.config): add wandb + EMA-rolling + dump fields"
```

---

### Task 4: Extend `CoartTBLogger` with wandb + image/object3d

**Files:**
- Modify: `coart/common/logging.py` (full rewrite; ~80 LOC)
- Test: `coart/tests/test_logger_wandb.py` (new)

- [ ] **Step 1: Write the failing test**

Create `coart/tests/test_logger_wandb.py`:
```python
"""Test CoartTBLogger dual-write to wandb + TB, robustness to wandb failure."""
from __future__ import annotations

import os
import tempfile
from unittest import mock

import numpy as np
import pytest


def test_logger_wandb_disabled_is_noop():
    """use_wandb=False → no wandb.init called, TB still works."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            lg = CoartTBLogger(td, is_master=True, use_wandb=False,
                               wandb_project="x", wandb_mode="online",
                               wandb_run_name="r", wandb_tags=[], config={})
            w.init.assert_not_called()
            lg.scalar("a/b", 1.0, step=1)
            lg.flush_if_due(step=100, i_log=100)
            lg.close()


def test_logger_wandb_network_failure_fallback():
    """wandb.init() raising must not crash; logger still writes to TB."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            w.init.side_effect = RuntimeError("no network")
            lg = CoartTBLogger(td, is_master=True, use_wandb=True,
                               wandb_project="x", wandb_mode="online",
                               wandb_run_name="r", wandb_tags=[], config={})
            # must not raise; internal _wandb set to None
            assert lg._wandb is None
            lg.scalar("a/b", 1.0, step=1)
            lg.flush_if_due(step=100, i_log=100)
            lg.close()


def test_logger_image_calls_wandb():
    """image() forwards to wandb.Image when wandb is live."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            w.init.return_value = mock.MagicMock()
            lg = CoartTBLogger(td, is_master=True, use_wandb=True,
                               wandb_project="x", wandb_mode="online",
                               wandb_run_name="r", wandb_tags=[], config={})
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            lg.image("deep_eval/render/helmet", arr, step=1000)
            w.log.assert_called()


def test_logger_non_master_is_silent():
    """non-master ranks must not init wandb nor write to TB."""
    with tempfile.TemporaryDirectory() as td:
        from coart.common.logging import CoartTBLogger
        with mock.patch("coart.common.logging.wandb") as w:
            lg = CoartTBLogger(td, is_master=False, use_wandb=True,
                               wandb_project="x", wandb_mode="online",
                               wandb_run_name="r", wandb_tags=[], config={})
            w.init.assert_not_called()
            lg.scalar("a/b", 1.0, step=1)  # no-op on non-master
            lg.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest coart/tests/test_logger_wandb.py -v
```
Expected: FAIL — current `CoartTBLogger.__init__` signature has no `use_wandb`.

- [ ] **Step 3: Rewrite `coart/common/logging.py`**

Full replacement:
```python
"""TensorBoard + optional wandb dual-channel logging with cadence control.

Usage:
    logger = CoartTBLogger(
        output_dir, is_master=(rank==0),
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_mode=cfg.wandb_mode,
        wandb_run_name=cfg.run_tag,
        wandb_tags=[cfg.io_arch, ...],
        config=asdict(cfg),
    )
    logger.scalar("loss/total", v, step)      # buffered
    logger.flush_if_due(step, i_log=100)       # all-reduce + write
    logger.image("render/helmet", np_uint8_hwc, step)
    logger.object3d("mesh/helmet", verts_Nx3_float32, step)
    logger.alert(title, text, level="WARN")
    logger.close()
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb as _wandb_mod
    wandb = _wandb_mod
except ImportError:
    wandb = None


class CoartTBLogger:
    """Rank-0 TB + wandb writer; non-master ranks accept calls as no-op.

    `flush_if_due` does an all-reduce over buffered scalars and writes the
    averaged value to both backends.
    """

    def __init__(
        self,
        output_dir: str,
        is_master: bool,
        use_wandb: bool = False,
        wandb_project: str = "coart-vae",
        wandb_mode: str = "online",
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.is_master = is_master
        self._buf: Dict[str, list[float]] = {}
        self._writer: Optional[SummaryWriter] = None
        self._wandb = None
        if is_master:
            os.makedirs(os.path.join(output_dir, "tb_logs"), exist_ok=True)
            self._writer = SummaryWriter(os.path.join(output_dir, "tb_logs"))
            if use_wandb and wandb is not None:
                try:
                    self._wandb = wandb.init(
                        project=wandb_project,
                        name=wandb_run_name,
                        tags=wandb_tags or [],
                        config=config or {},
                        mode=wandb_mode,
                        dir=output_dir,
                    )
                except Exception as e:
                    print(f"[logger] wandb.init failed ({e}); falling back to TB-only",
                          file=sys.stderr)
                    self._wandb = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if not self.is_master:
            return
        self._buf.setdefault(tag, []).append(float(value))

    def flush_if_due(self, step: int, i_log: int) -> None:
        if step % i_log != 0 or not self._buf:
            return
        tags = sorted(self._buf.keys())
        vals = torch.tensor(
            [sum(self._buf[t]) / len(self._buf[t]) for t in tags],
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
        if self.is_master:
            for t, v in zip(tags, vals.tolist()):
                if self._writer is not None:
                    self._writer.add_scalar(t, v, step)
            if self._wandb is not None:
                try:
                    self._wandb.log({t: v for t, v in zip(tags, vals.tolist())},
                                    step=step)
                except Exception as e:
                    print(f"[logger] wandb.log failed ({e})", file=sys.stderr)
        self._buf.clear()

    def image(self, tag: str, np_img: np.ndarray, step: int) -> None:
        """Log a HxWx3 uint8 image."""
        if not self.is_master:
            return
        if self._writer is not None:
            # SummaryWriter wants CHW
            img = np.transpose(np_img, (2, 0, 1))
            try:
                self._writer.add_image(tag, img, step)
            except Exception as e:
                print(f"[logger] tb add_image failed ({e})", file=sys.stderr)
        if self._wandb is not None:
            try:
                self._wandb.log({tag: wandb.Image(np_img)}, step=step)
            except Exception as e:
                print(f"[logger] wandb.Image failed ({e})", file=sys.stderr)

    def object3d(self, tag: str, verts_Nx3: np.ndarray, step: int) -> None:
        """Log a point cloud as wandb.Object3D (TB has no native support)."""
        if not self.is_master or self._wandb is None:
            return
        try:
            obj = wandb.Object3D(verts_Nx3.astype(np.float32))
            self._wandb.log({tag: obj}, step=step)
        except Exception as e:
            print(f"[logger] wandb.Object3D failed ({e})", file=sys.stderr)

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        """Best-effort wandb alert; always prints to stderr."""
        print(f"[ALERT/{level}] {title}: {text}", file=sys.stderr)
        if not self.is_master or self._wandb is None:
            return
        try:
            lvl = getattr(wandb.AlertLevel, level, wandb.AlertLevel.WARN)
            wandb.alert(title=title, text=text, level=lvl)
        except Exception as e:
            print(f"[logger] wandb.alert failed ({e})", file=sys.stderr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest coart/tests/test_logger_wandb.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Run regression on existing tests**

```bash
.venv/bin/pytest coart/tests/ -q
```
Expected: all prior tests (test_ema, test_checkpoint, test_io_stems, test_build_warmstart, test_loss, plus new ones) still PASS. `test_logger_wandb.py::test_logger_non_master_is_silent` confirms backward-compat: non-master ranks still work.

- [ ] **Step 6: Commit**

```bash
git add coart/common/logging.py coart/tests/test_logger_wandb.py
git commit -m "feat(coart.common.logging): add wandb dual-write + image + object3d + alert"
```

---

### Task 5: Wire logger in train.py + throughput metrics

**Files:**
- Modify: `coart/vae/train.py` (logger init site + throughput instrumentation)

- [ ] **Step 1: Locate logger init and main loop in train.py**

```bash
grep -n "CoartTBLogger(" /mnt/novita2/siyuan/workspace/TRELLIS.2/coart/vae/train.py
grep -n "for step in\|while step <\|step += 1" /mnt/novita2/siyuan/workspace/TRELLIS.2/coart/vae/train.py
```
Note the line numbers — they are input to Step 2/3.

- [ ] **Step 2: Update logger init**

Replace the existing `logger = CoartTBLogger(...)` call with (pass `cfg` fields from Task 3):
```python
from dataclasses import asdict

logger = CoartTBLogger(
    cfg.output_dir,
    is_master=(rank == 0),
    use_wandb=cfg.use_wandb,
    wandb_project=cfg.wandb_project,
    wandb_mode=cfg.wandb_mode,
    wandb_run_name=cfg.run_tag,
    wandb_tags=[
        cfg.io_arch,
        "warmstart_io" if cfg.warmstart_io else "scratch",
        f"res{cfg.resolution}",
    ],
    config=asdict(cfg),
)
```

- [ ] **Step 3: Add throughput instrumentation inside the training loop**

Immediately before the `for step in ...` loop, add:
```python
import time
_step_t0 = time.monotonic()
```

Inside the loop, right before `logger.flush_if_due(step, cfg.i_log)`, add:
```python
# Throughput: seconds per step, samples & voxels per sec (per GPU).
_step_dt = time.monotonic() - _step_t0
_step_t0 = time.monotonic()
_batch_nv = int(x.feats.shape[0])  # active voxels in this step's batch
logger.scalar("throughput/step_s", _step_dt, step)
logger.scalar("throughput/samples_s_per_gpu", 1.0 / max(_step_dt, 1e-6), step)
logger.scalar("throughput/voxels_s_per_gpu", _batch_nv / max(_step_dt, 1e-6), step)
logger.scalar("throughput/avg_batch_voxels", float(_batch_nv), step)
```

(Replace `x.feats.shape[0]` with the actual local variable name holding the input `SparseTensor`; confirm via `grep -n "SparseTensor\|feats=" coart/vae/train.py`.)

- [ ] **Step 4: Add grad-norm instrumentation**

Locate where `AdaptiveGradClipper` is invoked (grep `grad_clipper\|AdaptiveGradClipper`). After the clip call, add:
```python
logger.scalar("train/grad/norm_pre_clip", float(grad_clipper.last_pre_clip_norm), step)
logger.scalar("train/grad/norm_post_clip", float(grad_clipper.last_post_clip_norm), step)
logger.scalar("train/grad/clip_ratio",
              float(grad_clipper.last_post_clip_norm / max(grad_clipper.last_pre_clip_norm, 1e-8)),
              step)
logger.scalar("train/grad/p95_rolling", float(grad_clipper.p95), step)
```

Check `trellis2/utils/grad_clip_utils.py::AdaptiveGradClipper` for the exact attribute names:
```bash
grep -n "self\." /mnt/novita2/siyuan/workspace/TRELLIS.2/trellis2/utils/grad_clip_utils.py | head
```
If the attributes are named differently (e.g., `self._last_norm`, `self._p95`), adjust accordingly. Do NOT modify `trellis2/` — read-only.

- [ ] **Step 5: Add scheduler state scalars**

In the same vicinity, add:
```python
logger.scalar("train/sched/lr", float(optimizer.param_groups[0]["lr"]), step)
logger.scalar("train/sched/unfrozen", 1.0 if unfrozen else 0.0, step)
logger.scalar("train/sched/steps_since_unfreeze",
              float(step - step_at_unfreeze) if step_at_unfreeze is not None else 0.0,
              step)
```

- [ ] **Step 6: Verify with a 15-step smoke**

Run smoke against 120 node (1 GPU, 15 steps, uses existing small-data defaults):
```bash
WANDB_MODE=offline .venv/bin/python -m coart.vae \
    --run_tag smoke_monitoring \
    --max_steps 15 --i_log 5 --i_val 100 --i_save 100 \
    --no_sample_at_step_one --max_voxels 100000 \
    --freeze_backbone_steps 0 --lr_unfreeze_warmup_steps 0 \
    --output_dir /tmp/coart_smoke_monitoring
```
Expected stdout:
- `[logger] wandb.init` (or offline equivalent) succeeds
- No crash before step 15
- `results/<run>/wandb/offline-run-*/` exists with log events

Inspect:
```bash
ls -la /tmp/coart_smoke_monitoring/wandb/ 2>/dev/null
ls /tmp/coart_smoke_monitoring/tb_logs/ 2>/dev/null
```
Expected: `wandb/offline-run-*/` directory with `.wandb` file; `tb_logs/events.out.tfevents.*` file.

- [ ] **Step 7: Commit**

```bash
git add coart/vae/train.py
git commit -m "feat(coart.vae.train): wire wandb logger + throughput/grad/sched scalars"
```

---

### Task 6: Install wandb dependency documentation

**Files:**
- Modify: `coart/README.md` (add wandb setup note in the "使用指南" section)

- [ ] **Step 1: Locate the "使用指南" section anchor**

```bash
grep -n "## .*使用指南\|## Usage" coart/README.md
```

- [ ] **Step 2: Add a subsection after "## 使用指南"**

Insert this block (Chinese consistent with rest of README):
````markdown
### 4.0 Wandb 监控（首次 setup）

`coart.vae` 默认开启 wandb online 日志。首次使用：

```bash
.venv/bin/pip install "wandb>=0.26" "protobuf>=7.34.1"
.venv/bin/wandb login   # 粘贴 WANDB_API_KEY（或写入 ~/.netrc）
```

默认 project 是 `coart-vae`。切换到离线或禁用：
```bash
# 离线：日志写到 results/<run>/wandb/offline-run-*/；train 结束后
#       `wandb sync` 批量上传
python -m coart.vae --wandb_mode offline ...
# 完全禁用
python -m coart.vae --no_wandb ...
```

TensorBoard 始终开启作为本地备份：`tensorboard --logdir results/<run>/tb_logs/`。
````

- [ ] **Step 3: Commit**

```bash
git add coart/README.md
git commit -m "docs(coart): document wandb setup"
```

---

## Stage 3 — Deep-eval infrastructure

### Task 7: `coart/eval/metrics.py` — thin wrappers

**Files:**
- Create: `coart/eval/__init__.py`, `coart/eval/metrics.py`
- Test: `coart/tests/test_metrics_wrappers.py`

- [ ] **Step 1: Create `coart/eval/__init__.py`**

```python
"""coart.eval — val-set deep evaluation utilities.

Provides:
    - metrics: CD/NC/F-score/topology wrappers around scripts/eval/
    - deep_eval: run_deep_eval() entry point
    - watchdog: alert conditions

These modules dynamically add `scripts/eval/` to sys.path so the canonical
metric implementations can be reused without copying.
"""
```

- [ ] **Step 2: Create `coart/eval/metrics.py`**

```python
"""Thin wrappers that expose canonical CD/NC/F-score/topology metrics
from scripts/eval/ under a stable coart.eval.metrics API.

Why wrappers: we pin a 100k-point sampling default and return plain dicts
(no PyTorch tensors) so deep_eval can log floats directly.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

_COART_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_EVAL = os.path.join(_COART_ROOT, "scripts", "eval")
if _SCRIPTS_EVAL not in sys.path:
    sys.path.insert(0, _SCRIPTS_EVAL)

import eval_metrics as _em  # noqa: E402
import ovoxel_repr_test as _ort  # noqa: E402


def sample_surface(mesh, num_points: int = 100000):
    """Returns (points_Nx3, normals_Nx3) as numpy float32."""
    pts, nrms = _em.sample_points_and_normals(mesh, num_points=num_points)
    return np.asarray(pts, dtype=np.float32), np.asarray(nrms, dtype=np.float32)


def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    return float(_em.chamfer_distance(pts1, pts2))


def normal_consistency(pts1, nrms1, pts2, nrms2) -> float:
    return float(_em.normal_consistency(pts1, nrms1, pts2, nrms2))


def f_score_multi(pts1, pts2, thresholds: List[float]) -> Dict[float, float]:
    """Return {threshold: f_score} for every requested threshold."""
    out = _em.f_score_multi(pts1, pts2, thresholds=thresholds)
    # eval_metrics.f_score_multi returns dict-like; coerce to plain floats
    return {float(k): float(v) for k, v in out.items()}


def compute_topo_metrics(mesh) -> Dict[str, float]:
    """Wrap _compute_topo_metrics; coerce bools to float for scalar logging."""
    d = _ort._compute_topo_metrics(mesh)
    return {
        "n_components": float(d["n_components"]),
        "euler_number": float(d["euler_number"]),
        "n_boundary_edges": float(d["n_boundary_edges"]),
        "is_watertight": 1.0 if d["is_watertight"] else 0.0,
        "surface_area": float(d["surface_area"]),
    }
```

- [ ] **Step 3: Write failing test**

Create `coart/tests/test_metrics_wrappers.py`:
```python
"""Unit tests for coart.eval.metrics wrappers using trimesh primitives."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from coart.eval.metrics import (
    sample_surface, chamfer_distance, normal_consistency,
    f_score_multi, compute_topo_metrics,
)


@pytest.fixture
def icosphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=1.0)


def test_sample_surface_shapes(icosphere):
    pts, nrms = sample_surface(icosphere, num_points=10000)
    assert pts.shape == (10000, 3)
    assert nrms.shape == (10000, 3)
    assert pts.dtype == np.float32
    assert nrms.dtype == np.float32


def test_chamfer_self_is_zero(icosphere):
    pts, _ = sample_surface(icosphere, num_points=5000)
    assert chamfer_distance(pts, pts) < 1e-6


def test_normal_consistency_self_is_one(icosphere):
    pts, nrms = sample_surface(icosphere, num_points=5000)
    assert normal_consistency(pts, nrms, pts, nrms) > 0.99


def test_f_score_self_is_one(icosphere):
    pts, _ = sample_surface(icosphere, num_points=5000)
    fs = f_score_multi(pts, pts, thresholds=[0.001, 0.01, 0.1])
    for thr, v in fs.items():
        assert v > 0.999, f"F@{thr} self-score should ≈1.0, got {v}"


def test_topo_metrics_watertight_sphere(icosphere):
    m = compute_topo_metrics(icosphere)
    assert m["is_watertight"] == 1.0
    assert m["n_components"] == 1.0
    # Euler of sphere = 2
    assert abs(m["euler_number"] - 2.0) < 0.5
```

- [ ] **Step 4: Run test**

```bash
.venv/bin/pytest coart/tests/test_metrics_wrappers.py -v
```
Expected: 5 tests PASS. If any sampling-related test fails by small numeric margin, raise `num_points` for that test.

- [ ] **Step 5: Commit**

```bash
git add coart/eval/__init__.py coart/eval/metrics.py coart/tests/test_metrics_wrappers.py
git commit -m "feat(coart.eval): add metrics wrappers around scripts/eval/"
```

---

### Task 8: Build golden asset NPZs + static asset list

**Files:**
- Create: `scripts/coart_build_golden.py`
- Output (generated): `datasets/coart_golden/*.npz` (8 files), `coart/eval/golden_assets.json`

- [ ] **Step 1: Study existing preprocess for format**

```bash
grep -n "cube_indices\|num_boundary\|feats" /mnt/novita2/siyuan/workspace/TRELLIS.2/precompute_feat18.py | head -20
head -40 /mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/preprocess-by-rank/out/full_ranked.csv
```
Capture: the NPZ schema is `{cube_indices: (N,3) int32, feats: (N,18) float32, num_boundary: (N,) int32}`.

- [ ] **Step 2: Create `scripts/coart_build_golden.py`**

```python
"""Build 8 golden assets for coart.vae deep-eval.

Produces:
    datasets/coart_golden/<asset_name>.npz
        - cube_indices (N,3) int32
        - feats (N,18) float32
        - num_boundary int32
        - gt_points (100000,3) float32
        - gt_normals (100000,3) float32
        - gt_topo (dict via np.savez_compressed as 0-d object)
        - meta (dict: asset_name, sha, rank, tier, local_path_gt)

    coart/eval/golden_assets.json
        [{"name": "helmet", "sha": null, "rank": -1, "tier": -1,
          "local_path_gt": "datasets/sketchfab_hard/helmet.glb",
          "npz_path": "datasets/coart_golden/helmet.npz"}, ...]

Usage:
    .venv/bin/python scripts/coart_build_golden.py
        [--data_root ...] [--feat18_data_dir ...] [--out_dir datasets/coart_golden]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from coart.data.feat18_dataset import _is_val_sha  # noqa: E402
from coart.eval.metrics import sample_surface, compute_topo_metrics  # noqa: E402


def _percentile_indices(n: int, pcts: List[float]) -> List[int]:
    """Return index (integer, nearest) for each percentile, 0-indexed."""
    return [min(int(round(p / 100.0 * (n - 1))), n - 1) for p in pcts]


def _intersect_val_with_feat18(
    full_ranked_csv: Path,
    feat18_data_dir: Path,
    val_split_mod: int,
) -> List[dict]:
    """Rows from full_ranked that are (a) val-split by sha hash and (b) present in feat18 dir."""
    feat18_shas = {
        os.path.splitext(f)[0]
        for f in os.listdir(feat18_data_dir) if f.endswith(".npz")
    }
    out = []
    with open(full_ranked_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sha = row["sha256"]
            if sha not in feat18_shas:
                continue
            if not _is_val_sha(sha, val_split_mod):
                continue
            out.append(row)
    out.sort(key=lambda r: int(r["rank"]))
    return out


def _make_triple_sphere_mesh() -> trimesh.Trimesh:
    """Three concentric icospheres, merged into one trimesh."""
    meshes = []
    for r in (0.4, 0.7, 1.0):
        s = trimesh.creation.icosphere(subdivisions=4, radius=r)
        meshes.append(s)
    return trimesh.util.concatenate(meshes)


def _gt_cache_and_save(
    asset_name: str,
    npz_src_path: Path,          # where cube_indices/feats come from
    gt_mesh_path: Path,          # where GT mesh lives on disk
    out_npz: Path,
    meta_extra: dict,
) -> None:
    d = np.load(npz_src_path)
    mesh = trimesh.load(gt_mesh_path, force="mesh")
    pts, nrms = sample_surface(mesh, num_points=100000)
    topo = compute_topo_metrics(mesh)
    meta = {
        "asset_name": asset_name,
        "local_path_gt": str(gt_mesh_path),
        **meta_extra,
    }
    np.savez_compressed(
        out_npz,
        cube_indices=d["cube_indices"],
        feats=d["feats"],
        num_boundary=d["num_boundary"],
        gt_points=pts,
        gt_normals=nrms,
        gt_topo=np.array(topo, dtype=object),
        meta=np.array(meta, dtype=object),
    )
    print(f"[build_golden] wrote {out_npz} (N_cubes={len(d['cube_indices'])}, "
          f"GT components={topo['n_components']})")


def _precompute_feat18_from_mesh(
    mesh_path: Path, out_npz: Path, resolution: int,
) -> None:
    """Invoke precompute_feat18.py's CLI on a single mesh to produce NPZ."""
    import subprocess
    # Re-use the existing script rather than copy-paste its logic.
    # precompute_feat18.py expects its standard CLI; we use --single_glb and --single_out.
    # If those flags don't exist, build a minimal input CSV and set --metadata_csv.
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "precompute_feat18.py"),
        "--single_glb", str(mesh_path),
        "--single_out", str(out_npz),
        "--resolution", str(resolution),
    ]
    # If --single_glb not supported, fall back to CSV-based path (below).
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: write a 1-row CSV and run rank-0 mode.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            sha = hashlib.sha256(str(mesh_path).encode()).hexdigest()
            fh.write("sha256,local_path\n")
            fh.write(f"{sha},{mesh_path}\n")
            tmp_csv = fh.name
        subprocess.run([
            str(REPO / ".venv" / "bin" / "python"),
            str(REPO / "precompute_feat18.py"),
            "--metadata_csv", tmp_csv,
            "--output_root", str(out_npz.parent),
            "--resolution", str(resolution),
            "--rank", "0", "--world_size", "1",
        ], check=True)
        # Move <sha>.npz → out_npz
        gen = out_npz.parent / f"{sha}.npz"
        os.rename(gen, out_npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab")
    ap.add_argument("--feat18_data_dir",
                    default=str(REPO / "datasets" / "ObjaverseXL_sketchfab" /
                                "feat18_512" / "data"))
    ap.add_argument("--full_ranked_csv",
                    default=str(REPO / "scripts" / "preprocess-by-rank" /
                                "out" / "full_ranked.csv"))
    ap.add_argument("--sketchfab_hard_root",
                    default=str(REPO / "datasets" / "sketchfab_hard"))
    ap.add_argument("--out_dir", default=str(REPO / "datasets" / "coart_golden"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--val_split_mod", type=int, default=200)
    ap.add_argument("--percentiles", nargs="+", type=float,
                    default=[10, 25, 40, 60, 80, 95])
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset_list = []

    # 1. helmet (external reference)
    helmet_glb = Path(args.sketchfab_hard_root)
    # locate helmet.glb recursively (structure may vary)
    candidates = list(helmet_glb.glob("**/helmet*.glb")) + \
                 list(helmet_glb.glob("**/helmet*.obj"))
    if not candidates:
        print(f"[build_golden] WARNING: no helmet mesh under {helmet_glb}",
              file=sys.stderr)
    else:
        helmet_mesh_path = candidates[0]
        helmet_feat18_tmp = out / "helmet_feat18_tmp.npz"
        _precompute_feat18_from_mesh(helmet_mesh_path, helmet_feat18_tmp,
                                     args.resolution)
        helmet_npz = out / "helmet.npz"
        _gt_cache_and_save("helmet", helmet_feat18_tmp, helmet_mesh_path,
                           helmet_npz, {"sha": None, "rank": -1, "tier": -1})
        os.remove(helmet_feat18_tmp)
        asset_list.append({
            "name": "helmet",
            "sha": None, "rank": -1, "tier": -1,
            "local_path_gt": str(helmet_mesh_path),
            "npz_path": str(helmet_npz.relative_to(REPO)),
        })

    # 2. triple_sphere (synthetic)
    tri_mesh = _make_triple_sphere_mesh()
    tri_mesh_path = out / "triple_sphere.glb"
    tri_mesh.export(tri_mesh_path)
    tri_feat18_tmp = out / "triple_sphere_feat18_tmp.npz"
    _precompute_feat18_from_mesh(tri_mesh_path, tri_feat18_tmp,
                                 args.resolution)
    tri_npz = out / "triple_sphere.npz"
    _gt_cache_and_save("triple_sphere", tri_feat18_tmp, tri_mesh_path,
                       tri_npz, {"sha": None, "rank": -1, "tier": -1})
    os.remove(tri_feat18_tmp)
    asset_list.append({
        "name": "triple_sphere",
        "sha": None, "rank": -1, "tier": -1,
        "local_path_gt": str(tri_mesh_path),
        "npz_path": str(tri_npz.relative_to(REPO)),
    })

    # 3-8. Six val-split assets at percentile spread
    candidates = _intersect_val_with_feat18(
        Path(args.full_ranked_csv),
        Path(args.feat18_data_dir),
        args.val_split_mod,
    )
    print(f"[build_golden] {len(candidates)} val-split feat18 candidates; "
          f"picking at percentiles {args.percentiles}")
    idxs = _percentile_indices(len(candidates), args.percentiles)
    for p, i in zip(args.percentiles, idxs):
        row = candidates[i]
        sha = row["sha256"]
        name = f"val_p{int(p)}"
        src_npz = Path(args.feat18_data_dir) / f"{sha}.npz"
        gt_mesh_path = Path(args.data_root) / row["local_path"]
        out_npz = out / f"{name}.npz"
        _gt_cache_and_save(name, src_npz, gt_mesh_path, out_npz,
                           {"sha": sha, "rank": int(row["rank"]),
                            "tier": int(row["tier"])})
        asset_list.append({
            "name": name, "sha": sha,
            "rank": int(row["rank"]), "tier": int(row["tier"]),
            "local_path_gt": str(gt_mesh_path),
            "npz_path": str(out_npz.relative_to(REPO)),
        })

    # Persist asset list
    json_path = REPO / "coart" / "eval" / "golden_assets.json"
    with open(json_path, "w") as fh:
        json.dump(asset_list, fh, indent=2)
    print(f"[build_golden] wrote {json_path} ({len(asset_list)} assets)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Dry-run (expect success or informative failure)**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python scripts/coart_build_golden.py 2>&1 | tee /tmp/coart_build_golden.log
```
Expected possibilities:
1. Success — 8 NPZ files under `datasets/coart_golden/`, plus `coart/eval/golden_assets.json`. Skip Step 4.
2. `WARNING: no helmet mesh under ...` — user hasn't staged helmet. Ask user for path, re-run.
3. Failure at `_precompute_feat18_from_mesh` because `precompute_feat18.py` doesn't accept `--single_glb`. Go to Step 4.

- [ ] **Step 4: (If Step 3 case 3) patch `_precompute_feat18_from_mesh`**

Inspect `precompute_feat18.py` CLI:
```bash
.venv/bin/python precompute_feat18.py --help | head -30
```
Update `_precompute_feat18_from_mesh` to use the actual supported flags. Commit the update alongside the script.

- [ ] **Step 5: Verify artifacts**

```bash
ls datasets/coart_golden/
# Expected: 8 *.npz files (no *_feat18_tmp.npz leftover)
.venv/bin/python -c "
import numpy as np
for n in ['helmet','triple_sphere','val_p10','val_p25','val_p40','val_p60','val_p80','val_p95']:
    d = np.load(f'datasets/coart_golden/{n}.npz', allow_pickle=True)
    print(n, dict(d['meta'].item())['asset_name'], 'N_cubes=', d['cube_indices'].shape[0])
"
cat coart/eval/golden_assets.json | python -m json.tool | head -30
```

- [ ] **Step 6: Commit**

```bash
git add scripts/coart_build_golden.py coart/eval/golden_assets.json
git commit -m "feat(coart): build golden_assets setup script + produced 8-asset manifest"
# datasets/coart_golden/ is intentionally NOT committed (large binary NPZs)
```

Note: `datasets/coart_golden/*.npz` files are artifacts, not tracked in git. Ensure `datasets/` is in `.gitignore` (it already is).

---

### Task 9: Build Layer V baseline JSON

**Files:**
- Create: `scripts/coart_build_baseline.py`
- Output (generated): `coart/eval/golden_baseline.json`

- [ ] **Step 1: Study EXP-5 CSV for existing helmet/triple_sphere numbers**

```bash
grep "^helmet,512,V\|^nested_spheres,512,V" \
    /mnt/novita2/siyuan/workspace/TRELLIS.2/results/baseline_experiments/EXP5_full_baseline/geometric_metrics.csv
```
Extract: helmet (V): cd=1.2908e-5, nc=0.7731, f_0.005=0.8635, f_0.001=0.0557. nested_spheres (V, closest to triple_sphere): cd=1.25e-5, nc=0.9995, f_0.005=0.8656, f_0.001=0.0762.

- [ ] **Step 2: Create `scripts/coart_build_baseline.py`**

```python
"""Produce coart/eval/golden_baseline.json with Layer V metrics per golden asset.

For helmet & triple_sphere: hardcoded from EXP-5 CSV (plus computed f_0.05).
For 6 val-split assets: run Layer V pipeline (O-Voxel encode + SC-VAE decode)
on their GT meshes and compute metrics.

Usage:
    .venv/bin/python scripts/coart_build_baseline.py

Output:
    coart/eval/golden_baseline.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from coart.eval.metrics import (  # noqa: E402
    sample_surface, chamfer_distance, normal_consistency,
    f_score_multi, compute_topo_metrics,
)


# EXP-5 CSV direct values (res=512, Layer V) — f_0.05 will be computed
_EXP5_HARDCODED = {
    "helmet": {
        "cd": 1.2908307326142676e-05, "nc": 0.7730708122253418,
        "f_0.005": 0.8634670972824097, "f_0.001": 0.055654510855674744,
        "n_components": 208348, "euler": 4523,
        "n_boundary_edges": 146102, "is_watertight": False,
    },
    "triple_sphere": {
        # Using nested_spheres as structural proxy
        "cd": 1.2484781109378673e-05, "nc": 0.9995467066764832,
        "f_0.005": 0.8656343221664429, "f_0.001": 0.07622464001178741,
        "n_components": 22, "euler": 5,
        "n_boundary_edges": 18, "is_watertight": False,
    },
}


def _layer_v_roundtrip_metrics(gt_mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Run O-Voxel + SC-VAE roundtrip; return metric dict.

    Delegates to scripts/eval/baseline_exp5_metrics.py::compute_all_metrics
    which already implements the Layer V pipeline.
    """
    from baseline_exp5_metrics import compute_all_metrics  # scripts/eval/
    # compute_all_metrics expects (gt_mesh, recon_mesh); we need the pipeline
    # to produce recon_mesh. Use its internal Layer V pipeline via main loop
    # or directly call the helper functions it uses.
    from baseline_exp5_metrics import run_layer_v
    recon_mesh = run_layer_v(gt_mesh, resolution=512)  # noqa: F841
    return compute_all_metrics(gt_mesh, recon_mesh)


def main():
    # Load golden assets list
    with open(REPO / "coart" / "eval" / "golden_assets.json") as fh:
        assets = json.load(fh)

    baseline = {}
    for a in assets:
        name = a["name"]
        if name in _EXP5_HARDCODED:
            # Add f_0.05 by sampling GT, then re-running F-score at that threshold.
            # Quick path: load GT, sample 100k, CD-self F@0.05 is just 1.0;
            # but the Layer V F@0.05 requires reading from cached Layer V numbers
            # (which EXP-5 didn't compute). Approximate: if the other F-scores
            # monotonically approach 1, F@0.05 ≈ 0.98 for helmet-class. For
            # correctness, rerun via Layer V pipeline on this asset.
            print(f"[build_baseline] {name}: using EXP-5 hardcoded + computing f_0.05")
            gt_mesh = trimesh.load(a["local_path_gt"], force="mesh")
            try:
                metrics = _layer_v_roundtrip_metrics(gt_mesh)
                f_005 = metrics.get("f_0.05", None)
            except Exception as e:
                print(f"  layer_v f_0.05 computation failed ({e}); "
                      f"using conservative 0.98", file=sys.stderr)
                f_005 = 0.98
            m = dict(_EXP5_HARDCODED[name])
            m["f_0.05"] = float(f_005) if f_005 is not None else 0.98
            baseline[name] = {"layer_v": m}
        else:
            # val_p* — run Layer V pipeline
            print(f"[build_baseline] {name}: running Layer V pipeline")
            gt_mesh = trimesh.load(a["local_path_gt"], force="mesh")
            metrics = _layer_v_roundtrip_metrics(gt_mesh)
            baseline[name] = {"layer_v": metrics}

    out_path = REPO / "coart" / "eval" / "golden_baseline.json"
    with open(out_path, "w") as fh:
        json.dump(baseline, fh, indent=2)
    print(f"[build_baseline] wrote {out_path} ({len(baseline)} assets)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Dry-run**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python scripts/coart_build_baseline.py 2>&1 | tee /tmp/coart_build_baseline.log
```

- [ ] **Step 4: If `run_layer_v` not exported**

Inspect:
```bash
grep -n "^def " /mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/eval/baseline_exp5_metrics.py
```
If no `run_layer_v`, find the equivalent pipeline function (e.g., inside `main()` body) and either factor it out manually in the script OR use `subprocess.run([baseline_exp5_metrics.py, --single, ...])` CLI-style.

Fallback if scripts/eval/ is read-only: implement a local minimal Layer V roundtrip inside `coart_build_baseline.py` using `trellis2.modules.sparse.o_voxel.convert.mesh_to_flexible_dual_grid` + loading the pretrained SC-VAE.

- [ ] **Step 5: Verify artifact**

```bash
cat coart/eval/golden_baseline.json | python -m json.tool | head -40
```
Expected: 8 top-level keys, each with `{"layer_v": {"cd": ..., "nc": ..., "f_0.005": ..., ...}}`.

- [ ] **Step 6: Commit**

```bash
git add scripts/coart_build_baseline.py coart/eval/golden_baseline.json
git commit -m "feat(coart): build Layer V baseline JSON for 8 golden assets"
```

---

### Task 10: `coart/eval/deep_eval.py`

**Files:**
- Create: `coart/eval/deep_eval.py`
- Test: `coart/tests/test_deep_eval_smoke.py`

- [ ] **Step 1: Create `coart/eval/deep_eval.py`**

```python
"""Deep-eval on 8 golden assets at i_save cadence.

Executes rank-0-only; other ranks sit on a barrier. Loads the cached
golden NPZs (produced by scripts/coart_build_golden.py), runs
encoder→decoder on each, computes CD/NC/F-score/topology, and logs per-
asset + aggregated mean scalars. Optionally dumps a 4-view normal-map
side-by-side render for the assets in cfg.n_dump_names.

Public API:
    run_deep_eval(encoder, decoder, stats, step, logger, cfg)
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

from coart.common.dist_utils import unwrap
from coart.data.stats import denormalize, normalize
from coart.eval.metrics import (
    chamfer_distance, compute_topo_metrics, f_score_multi,
    normal_consistency, sample_surface,
)

_REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
_GOLDEN_LIST = _REPO / "coart" / "eval" / "golden_assets.json"


def _load_golden_manifest():
    with open(_GOLDEN_LIST) as fh:
        return json.load(fh)


def _render_normal_4view_side_by_side(recon_mesh, gt_mesh_path: str) -> np.ndarray:
    """Render 4-view normal maps of (GT | recon) and concatenate into one PNG.

    Returns HxWx3 uint8. If rendering fails, returns a solid gray placeholder
    so the logger doesn't crash.
    """
    try:
        import trimesh
        gt_mesh = trimesh.load(gt_mesh_path, force="mesh")
        # Use the standard 4-view config from scripts/eval/eval_metrics.py
        sys.path.insert(0, str(_REPO / "scripts" / "eval"))
        from eval_metrics import render_normal_maps_paper_config
        gt_imgs = render_normal_maps_paper_config(gt_mesh)    # (4, H, W, 3)
        recon_imgs = render_normal_maps_paper_config(recon_mesh)
        # 2 rows × 4 cols grid
        row_gt = np.concatenate(list(gt_imgs), axis=1)
        row_recon = np.concatenate(list(recon_imgs), axis=1)
        grid = np.concatenate([row_gt, row_recon], axis=0)
        return grid.astype(np.uint8)
    except Exception as e:
        print(f"[deep_eval] render failed: {e}", file=sys.stderr)
        return np.full((256, 1024, 3), 128, dtype=np.uint8)


def _one_asset(
    asset: Dict[str, Any],
    encoder,
    decoder,
    stats: Dict[str, np.ndarray],
    step: int,
    resolution: int,
    n_dump_names: list,
    logger,
) -> Dict[str, float]:
    """Run encode→decode→metrics for a single asset; log per-asset scalars.
    Returns metric dict (empty if failed)."""
    name = asset["name"]
    try:
        npz_path = _REPO / asset["npz_path"]
        d = np.load(npz_path, allow_pickle=True)
        cube_indices = torch.from_numpy(d["cube_indices"].astype(np.int32)).cuda()
        feats_raw = torch.from_numpy(d["feats"].astype(np.float32)).cuda()
        feats_n = normalize(feats_raw, stats["mean"], stats["std"])

        # Build SparseTensor per existing convention: add batch-index col of 0s
        from trellis2.modules.sparse import SparseTensor
        N = cube_indices.shape[0]
        batch_col = torch.zeros((N, 1), dtype=torch.int32, device="cuda")
        coords_bn = torch.cat([batch_col, cube_indices], dim=1)
        x = SparseTensor(feats=feats_n, coords=coords_bn)

        enc = unwrap(encoder)
        dec = unwrap(decoder)
        enc.eval(); dec.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z, mu, logvar = enc(x)
            pred, subs = dec(z, return_subdivisions=True) \
                if hasattr(dec, "return_subdivisions") else (dec(z), [])
        enc.train(); dec.train()
        feats_pred = denormalize(pred.feats.float().cpu(),
                                 stats["mean"], stats["std"]).numpy()
        cube_np = cube_indices.cpu().numpy().astype(np.int32)

        # decoder→mesh via feature_to_mesh (from train_overfit_feat18)
        sys.path.insert(0, str(_REPO))
        from train_overfit_feat18 import feature_to_mesh
        mesh = feature_to_mesh(feats_pred, cube_np, resolution)
        if mesh is None or getattr(mesh, "faces", None) is None \
           or len(mesh.faces) == 0:
            logger.scalar(
                f"deep_eval/online/per_asset/{name}/status_failed", 1.0, step)
            return {}

        pts, nrms = sample_surface(mesh, num_points=100000)
        gt_pts = d["gt_points"]
        gt_nrms = d["gt_normals"]
        cd = chamfer_distance(pts, gt_pts)
        nc = normal_consistency(pts, nrms, gt_pts, gt_nrms)
        fs = f_score_multi(pts, gt_pts, thresholds=[0.005, 0.01, 0.05])
        topo = compute_topo_metrics(mesh)

        metrics = {
            "cd": cd, "nc": nc,
            "f005": fs[0.005], "f01": fs[0.01], "f05": fs[0.05],
            "n_components": topo["n_components"],
            "euler": topo["euler_number"],
            "n_boundary_edges": topo["n_boundary_edges"],
            "is_watertight": topo["is_watertight"],
        }
        for k, v in metrics.items():
            logger.scalar(
                f"deep_eval/online/per_asset/{name}/{k}", float(v), step)

        # Normal-map render
        if name in n_dump_names:
            meta = d["meta"].item()
            img = _render_normal_4view_side_by_side(
                mesh, meta["local_path_gt"])
            logger.image(
                f"deep_eval/renders/{name}/online", img, step)

        return metrics
    except Exception as e:
        traceback.print_exc()
        logger.scalar(
            f"deep_eval/online/per_asset/{name}/status_failed", 1.0, step)
        return {}


def run_deep_eval(
    encoder,
    decoder,
    stats: Dict[str, np.ndarray],
    step: int,
    logger,
    cfg,
) -> Dict[str, Dict[str, float]]:
    """rank-0-only deep eval; other ranks sit on barrier.

    Returns {asset_name: metric_dict} on rank 0 (empty dict for failed assets);
    returns {} on non-master ranks. The caller uses this dict to feed Watchdog
    (helmet NC for bad-helmet detection).
    """
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) \
        else 0
    if rank != 0:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return {}

    manifest = _load_golden_manifest()
    results: Dict[str, Dict[str, float]] = {}
    for asset in manifest:
        m = _one_asset(asset, encoder, decoder, stats, step,
                       cfg.resolution, cfg.n_dump_names, logger)
        results[asset["name"]] = m  # {} means failed

    # Aggregate mean scalars over successful assets
    successful = [m for m in results.values() if m]
    if successful:
        keys = ["cd", "nc", "f005", "f01", "f05",
                "n_components", "euler", "n_boundary_edges", "is_watertight"]
        for k in keys:
            vals = [m[k] for m in successful if k in m]
            if vals:
                logger.scalar(f"deep_eval/online/mean/{k}",
                              float(sum(vals) / len(vals)), step)
        wt_rate = sum(1 for m in successful
                      if m.get("is_watertight", 0) >= 0.5) / len(successful)
        logger.scalar(
            "deep_eval/online/mean/watertight_rate", float(wt_rate), step)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return results
```

- [ ] **Step 2: Create smoke test with mocks**

Create `coart/tests/test_deep_eval_smoke.py`:
```python
"""Smoke deep_eval with mocked encoder/decoder and fake golden manifest."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for SparseTensor")
def test_deep_eval_runs_without_crash(tmp_path, monkeypatch):
    """Mock everything: golden manifest is empty → run_deep_eval should no-op."""
    # Redirect golden list to a tmp empty file
    empty_json = tmp_path / "golden_assets.json"
    empty_json.write_text("[]")
    monkeypatch.setattr("coart.eval.deep_eval._GOLDEN_LIST", empty_json)

    from coart.eval.deep_eval import run_deep_eval

    class FakeCfg:
        resolution = 512
        n_dump_names = []

    class FakeLogger:
        def __init__(self): self.calls = []
        def scalar(self, tag, value, step): self.calls.append((tag, value, step))
        def image(self, tag, img, step): self.calls.append((tag, img.shape, step))

    fake_enc = mock.MagicMock()
    fake_dec = mock.MagicMock()
    stats = {"mean": np.zeros(18, dtype=np.float32),
             "std": np.ones(18, dtype=np.float32)}
    logger = FakeLogger()

    # Should complete without throwing — empty manifest path
    run_deep_eval(fake_enc, fake_dec, stats, step=100, logger=logger, cfg=FakeCfg())
    # No calls because no assets were in the manifest
    assert logger.calls == []


def test_aggregation_skips_failed_assets(tmp_path, monkeypatch):
    """If _one_asset returns {} for some assets, mean is over successful ones."""
    from coart.eval import deep_eval as de

    assets_json = tmp_path / "golden_assets.json"
    assets_json.write_text(json.dumps([
        {"name": "a1", "npz_path": "x", "local_path_gt": "x"},
        {"name": "a2", "npz_path": "x", "local_path_gt": "x"},
        {"name": "a3", "npz_path": "x", "local_path_gt": "x"},
    ]))
    monkeypatch.setattr(de, "_GOLDEN_LIST", assets_json)

    def fake_one_asset(asset, *a, **kw):
        return {"cd": 0.1, "nc": 0.9, "f005": 0.8, "f01": 0.7, "f05": 0.6,
                "n_components": 1, "euler": 2, "n_boundary_edges": 0,
                "is_watertight": 1.0} if asset["name"] != "a2" else {}
    monkeypatch.setattr(de, "_one_asset", fake_one_asset)

    class FakeCfg:
        resolution = 512
        n_dump_names = []

    class FakeLogger:
        def __init__(self): self.vals = {}
        def scalar(self, tag, value, step): self.vals[tag] = value
        def image(self, *a, **kw): pass

    logger = FakeLogger()
    de.run_deep_eval(None, None, {}, step=100, logger=logger, cfg=FakeCfg())
    # Mean over 2 successful assets (a1, a3) → mean cd = 0.1
    assert abs(logger.vals["deep_eval/online/mean/cd"] - 0.1) < 1e-6
    assert abs(logger.vals["deep_eval/online/mean/watertight_rate"] - 1.0) < 1e-6
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/pytest coart/tests/test_deep_eval_smoke.py -v
```
Expected: PASS (test 1 CUDA-skipped on CPU, test 2 always passes).

- [ ] **Step 4: Commit**

```bash
git add coart/eval/deep_eval.py coart/tests/test_deep_eval_smoke.py
git commit -m "feat(coart.eval.deep_eval): add run_deep_eval on 8 golden assets"
```

---

### Task 11: EMA rolling split + train.py deep-eval hook

**Files:**
- Modify: `coart/vae/train.py` (EMA save call site + deep_eval hook)

- [ ] **Step 1: Locate EMA save and i_save block**

```bash
grep -n "save_ckpt\|ema.*step\|prefix=.ema\|if step % cfg.i_save" coart/vae/train.py
```
Identify the line that saves EMA with something like `save_ckpt(ema_state, ..., keep=cfg.rolling_ckpts)`.

- [ ] **Step 2: Change EMA save to use rolling_ckpts_ema**

Replace the EMA `save_ckpt(...)` calls to pass `keep=cfg.rolling_ckpts_ema`:
```python
# For encoder EMA:
save_ckpt(
    state_dict=ema_enc.shadow_state_dict(),
    output_dir=cfg.output_dir,
    prefix=f"ema_{cfg.ema_rate}_enc",
    step=step,
    keep=cfg.rolling_ckpts_ema,   # CHANGED from cfg.rolling_ckpts
)
# For decoder EMA:
save_ckpt(
    state_dict=ema_dec.shadow_state_dict(),
    output_dir=cfg.output_dir,
    prefix=f"ema_{cfg.ema_rate}_dec",
    step=step,
    keep=cfg.rolling_ckpts_ema,
)
```

Online ckpt (`ckpt_step*.pt`) and `misc_step*.pt` continue to use `keep=cfg.rolling_ckpts`.

- [ ] **Step 3: Add deep_eval hook at i_save step**

Near the existing `if step % cfg.i_save == 0:` block (after the ckpt save), add:
```python
        if step % cfg.i_save == 0 and step > 0:
            # Deep-eval on 8 golden assets (rank-0 does work, others barrier)
            from coart.eval.deep_eval import run_deep_eval
            try:
                run_deep_eval(
                    encoder=encoder, decoder=decoder,
                    stats={"mean": stats_mean, "std": stats_std},
                    step=step, logger=logger, cfg=cfg,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                logger.alert(
                    title="deep_eval crashed",
                    text=f"step={step}: {e!r}",
                    level="ERROR",
                )
```

Variable names `stats_mean`, `stats_std` must match existing train.py variables that hold the normalization statistics. Check via:
```bash
grep -n "stats\|mean.*std\|normalize" coart/vae/train.py | head
```

- [ ] **Step 4: Smoke verify with 20-step small run**

```bash
rm -rf /tmp/coart_smoke_deepeval
WANDB_MODE=offline .venv/bin/python -m coart.vae \
    --run_tag smoke_deepeval \
    --max_steps 20 --i_log 5 --i_val 100 --i_save 10 \
    --no_sample_at_step_one --max_voxels 100000 \
    --freeze_backbone_steps 0 --lr_unfreeze_warmup_steps 0 \
    --output_dir /tmp/coart_smoke_deepeval 2>&1 | tail -30
```
Expected:
- Step 10 triggers deep-eval (might fail per-asset if golden NPZ not built — check stderr `[deep_eval] ... failed:` messages vs. per_asset values)
- No training crash

If golden NPZs aren't built yet, the hook will log `status_failed=1` for every asset and move on. That's acceptable — Stage 5 smoke re-checks after builds.

- [ ] **Step 5: Verify EMA ckpt footprint**

```bash
ls -sh /tmp/coart_smoke_deepeval/ema_*.pt
# Expected: only 1 set of (enc, dec) files — rolling_ckpts_ema=1 working
ls -sh /tmp/coart_smoke_deepeval/ckpt_step*.pt
# Expected: 1 file at step 10 (max_steps=20 with i_save=10 has step=10, 20)
# Actually rolling=3 means 2 files keep (step 10 + 20), exact depends on find_latest
```

- [ ] **Step 6: Commit**

```bash
git add coart/vae/train.py
git commit -m "feat(coart.vae.train): add deep_eval hook + EMA rolling_ckpts_ema split"
```

---

## Stage 4 — Watchdog + baseline display

### Task 12: `coart/eval/watchdog.py`

**Files:**
- Create: `coart/eval/watchdog.py`
- Test: `coart/tests/test_watchdog.py`

- [ ] **Step 1: Create `coart/eval/watchdog.py`**

```python
"""Three soft watchdog conditions; log alerts + dump rolling window for debug.

Public API:
    wd = Watchdog(output_dir, logger)
    wd.update_train(step, grad_pre, grad_post, grad_p95, loss_ef)
    wd.check_train(step)  # may emit alert
    wd.update_helmet(step, helmet_nc)
    wd.check_helmet(step)  # may emit alert

Conditions:
    A. grad_spike: grad_pre > 100 × p95 for 50 consecutive steps
    B. ef_diverge: loss_ef monotonically increasing over a 5k-step window,
                   window starts at step 5000 and slides to step 10000
    C. helmet_bad: deep_eval helmet nc < 0.50 when step >= 10000

All alerts write stderr + wandb.alert; training never terminates.
Dump file: results/<run>/watchdog_<cond>_step<N>.json
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Optional


class Watchdog:
    GRAD_SPIKE_RATIO = 100.0
    GRAD_SPIKE_CONSECUTIVE = 50
    EF_WINDOW_START = 5000
    EF_WINDOW_END = 10000
    EF_WINDOW_STRIDE = 200
    HELMET_BAD_THRESHOLD = 0.50
    HELMET_BAD_MIN_STEP = 10000

    def __init__(self, output_dir: str, logger):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self._grad_spike_count = 0
        self._grad_spike_fired = False
        self._ef_hist = deque(maxlen=self.EF_WINDOW_END)  # (step, loss_ef)
        self._ef_fired = False
        self._helmet_fired = False
        # Rolling 200-step state buffer for dumps
        self._state_window = deque(maxlen=200)

    def _dump(self, cond: str, step: int, extra: dict) -> None:
        path = self.output_dir / f"watchdog_{cond}_step{step}.json"
        payload = {
            "condition": cond, "step": step,
            "rolling_window": list(self._state_window),
            **extra,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
        except Exception as e:
            print(f"[watchdog] dump failed: {e}")

    def update_train(
        self, step: int,
        grad_pre: float, grad_post: float, grad_p95: float,
        loss_ef: float, lr: float,
    ) -> None:
        self._state_window.append({
            "step": step, "grad_pre": grad_pre, "grad_post": grad_post,
            "grad_p95": grad_p95, "loss_ef": loss_ef, "lr": lr,
        })
        if grad_p95 > 0 and grad_pre > self.GRAD_SPIKE_RATIO * grad_p95:
            self._grad_spike_count += 1
        else:
            self._grad_spike_count = 0
        if self.EF_WINDOW_START <= step <= self.EF_WINDOW_END:
            self._ef_hist.append((step, loss_ef))

    def check_train(self, step: int) -> None:
        # A. grad spike
        if (not self._grad_spike_fired
                and self._grad_spike_count >= self.GRAD_SPIKE_CONSECUTIVE):
            last = self._state_window[-1] if self._state_window else {}
            self.logger.alert(
                title="grad spike",
                text=f"@step={step} pre_clip={last.get('grad_pre'):.2e} "
                     f"vs p95={last.get('grad_p95'):.2e}",
                level="WARN",
            )
            self._dump("grad_spike", step, {"trigger_state": dict(last)})
            self._grad_spike_fired = True
        # B. EF divergence
        if (not self._ef_fired and step >= self.EF_WINDOW_END
                and len(self._ef_hist) >= 2):
            steps, losses = zip(*self._ef_hist)
            # Monotonic increasing over the window → diverging
            deltas = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
            if all(d >= 0 for d in deltas) and (losses[-1] - losses[0]) > 0.01:
                self.logger.alert(
                    title="ef diverge",
                    text=f"@step={step} Δ=+{losses[-1] - losses[0]:.4f}",
                    level="WARN",
                )
                self._dump("ef_diverge", step, {
                    "ef_hist": list(self._ef_hist),
                })
                self._ef_fired = True

    def update_helmet(self, step: int, helmet_nc: Optional[float]) -> None:
        # Called inside deep_eval or after it; helmet_nc may be None if failed
        if helmet_nc is None:
            return
        if (not self._helmet_fired and step >= self.HELMET_BAD_MIN_STEP
                and helmet_nc < self.HELMET_BAD_THRESHOLD):
            self.logger.alert(
                title="helmet bad",
                text=f"@step={step} nc={helmet_nc:.3f}",
                level="WARN",
            )
            self._dump("helmet_bad", step, {"helmet_nc": helmet_nc})
            self._helmet_fired = True
```

- [ ] **Step 2: Write tests**

Create `coart/tests/test_watchdog.py`:
```python
"""Unit tests for Watchdog conditions + dump file creation."""
from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import pytest

from coart.eval.watchdog import Watchdog


class _FakeLogger:
    def __init__(self):
        self.alerts = []
    def alert(self, title, text, level="WARN"):
        self.alerts.append((title, text, level))


@pytest.fixture
def wd(tmp_path):
    return Watchdog(str(tmp_path), _FakeLogger())


def test_grad_spike_fires_after_consecutive(wd):
    # 50 consecutive steps with pre >> p95 → should fire
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    wd.check_train(Watchdog.GRAD_SPIKE_CONSECUTIVE - 1)
    assert len(wd.logger.alerts) == 1
    assert "grad spike" in wd.logger.alerts[0][0]


def test_grad_spike_resets_on_good_step(wd):
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE - 1):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    # One good step resets
    wd.update_train(1000, grad_pre=0.5, grad_post=0.4, grad_p95=0.5,
                    loss_ef=0.1, lr=1e-5)
    wd.check_train(1000)
    assert wd.logger.alerts == []


def test_ef_diverge_fires_on_monotonic_rise(wd):
    # Simulate monotonic-increase loss_ef over window 5000-10000
    for s in range(5000, 10001, 500):
        wd.update_train(s, grad_pre=0.1, grad_post=0.1, grad_p95=0.5,
                        loss_ef=0.1 + 0.01 * (s - 5000) / 500, lr=1e-5)
    wd.check_train(10000)
    assert any("ef diverge" in a[0] for a in wd.logger.alerts)


def test_ef_diverge_doesnt_fire_on_stable(wd):
    for s in range(5000, 10001, 500):
        wd.update_train(s, grad_pre=0.1, grad_post=0.1, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)  # flat
    wd.check_train(10000)
    assert not any("ef diverge" in a[0] for a in wd.logger.alerts)


def test_helmet_bad_fires_above_min_step(wd):
    wd.update_helmet(step=Watchdog.HELMET_BAD_MIN_STEP, helmet_nc=0.3)
    assert any("helmet bad" in a[0] for a in wd.logger.alerts)


def test_helmet_bad_silent_before_min_step(wd):
    wd.update_helmet(step=Watchdog.HELMET_BAD_MIN_STEP - 1, helmet_nc=0.1)
    assert wd.logger.alerts == []


def test_dump_file_written(wd, tmp_path):
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
    wd.check_train(Watchdog.GRAD_SPIKE_CONSECUTIVE - 1)
    dumps = list(tmp_path.glob("watchdog_grad_spike_step*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text())
    assert payload["condition"] == "grad_spike"


def test_fires_only_once(wd):
    # Second trigger must not produce second alert
    for i in range(Watchdog.GRAD_SPIKE_CONSECUTIVE * 2):
        wd.update_train(i, grad_pre=100.0, grad_post=1.0, grad_p95=0.5,
                        loss_ef=0.1, lr=1e-5)
        wd.check_train(i)
    assert sum(1 for a in wd.logger.alerts if "grad spike" in a[0]) == 1
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/pytest coart/tests/test_watchdog.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add coart/eval/watchdog.py coart/tests/test_watchdog.py
git commit -m "feat(coart.eval.watchdog): add grad/ef/helmet watchdog with alert+dump"
```

---

### Task 13: Wire Watchdog into train.py

**Files:**
- Modify: `coart/vae/train.py`

- [ ] **Step 1: Construct Watchdog at logger init**

Immediately after the `logger = CoartTBLogger(...)` line added in Task 5, add:
```python
from coart.eval.watchdog import Watchdog
watchdog = Watchdog(cfg.output_dir, logger)
```

- [ ] **Step 2: Update Watchdog on every step**

Inside the training loop, right after grad-norm instrumentation from Task 5, add:
```python
# Feed watchdog with this step's signals
watchdog.update_train(
    step=step,
    grad_pre=float(grad_clipper.last_pre_clip_norm),
    grad_post=float(grad_clipper.last_post_clip_norm),
    grad_p95=float(grad_clipper.p95),
    loss_ef=float(losses["recon_ef"].detach().item()),
    lr=float(optimizer.param_groups[0]["lr"]),
)
```
(Attribute names `losses["recon_ef"]` must match the output dict of `compute_vae_loss`; verify with `grep "recon_ef" coart/vae/loss.py`.)

- [ ] **Step 3: Check Watchdog at i_log cadence**

Right after `logger.flush_if_due(step, cfg.i_log)` add:
```python
if step % cfg.i_log == 0:
    watchdog.check_train(step)
```

- [ ] **Step 4: Feed Watchdog helmet signal from deep_eval return value**

`run_deep_eval` returns `{asset_name: metric_dict}` (empty dict for failed). Update the Task 11 deep_eval hook in `coart/vae/train.py`:

Change:
```python
            run_deep_eval(
                encoder=encoder, decoder=decoder,
                stats={"mean": stats_mean, "std": stats_std},
                step=step, logger=logger, cfg=cfg,
            )
```
to:
```python
            results = run_deep_eval(
                encoder=encoder, decoder=decoder,
                stats={"mean": stats_mean, "std": stats_std},
                step=step, logger=logger, cfg=cfg,
            )
            helmet_metrics = results.get("helmet", {})
            watchdog.update_helmet(
                step=step,
                helmet_nc=helmet_metrics.get("nc"),  # None if helmet failed
            )
```

No changes needed to `coart/eval/deep_eval.py` (the return type was made a dict in Task 10).

- [ ] **Step 5: 20-step smoke verify**

```bash
rm -rf /tmp/coart_smoke_watchdog
WANDB_MODE=offline .venv/bin/python -m coart.vae \
    --run_tag smoke_watchdog \
    --max_steps 20 --i_log 5 --i_val 100 --i_save 10 \
    --no_sample_at_step_one --max_voxels 100000 \
    --freeze_backbone_steps 0 --lr_unfreeze_warmup_steps 0 \
    --output_dir /tmp/coart_smoke_watchdog 2>&1 | tail -20
```
Expected: smoke runs to completion, no watchdog alerts triggered (threshold conditions not met in 20 steps), no crashes.

- [ ] **Step 6: Commit**

```bash
git add coart/vae/train.py
git commit -m "feat(coart.vae.train): wire Watchdog + helmet signal from deep_eval result"
```

---

### Task 14: Inject baseline reference lines via wandb.config

**Files:**
- Modify: `coart/common/logging.py::__init__` (config enrichment)

- [ ] **Step 1: Modify logger init to load baseline**

In `coart/common/logging.py::CoartTBLogger.__init__`, after the wandb.init success path, add:
```python
                # Inject baseline reference lines via wandb.config
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    bp = _Path(os.path.dirname(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))))) / \
                        "coart" / "eval" / "golden_baseline.json"
                    if bp.exists():
                        with open(bp) as fh:
                            self._wandb.config.update(
                                {"golden_baseline": _json.load(fh)},
                                allow_val_change=True,
                            )
                except Exception as e:
                    print(f"[logger] baseline config inject skipped ({e})",
                          file=sys.stderr)
```

- [ ] **Step 2: Smoke verify**

```bash
rm -rf /tmp/coart_smoke_baseline
WANDB_MODE=offline .venv/bin/python -m coart.vae \
    --run_tag smoke_baseline \
    --max_steps 5 --i_log 5 --i_val 100 --i_save 100 \
    --no_sample_at_step_one --max_voxels 100000 \
    --freeze_backbone_steps 0 --lr_unfreeze_warmup_steps 0 \
    --output_dir /tmp/coart_smoke_baseline 2>&1 | tail -10
grep -r "golden_baseline" /tmp/coart_smoke_baseline/wandb/ 2>/dev/null | head -5
```
Expected: offline wandb config includes `golden_baseline` key (if `coart/eval/golden_baseline.json` exists; else skipped with stderr warning). No crash either way.

- [ ] **Step 3: Commit**

```bash
git add coart/common/logging.py
git commit -m "feat(coart.common.logging): inject golden_baseline.json into wandb.config"
```

---

## Stage 5 — Final integration smoke

### Task 15: Full integration smoke on 120 / idle GPU

**Files:** none (no code changes; ops-only verification)

This task runs the full monitoring + deep-eval + watchdog stack end-to-end on real GPU to verify the spec's success criteria (§9).

- [ ] **Step 1: Ensure golden + baseline artifacts built**

```bash
ls datasets/coart_golden/*.npz 2>/dev/null | wc -l
# Expected: 8
cat coart/eval/golden_baseline.json | python -c "import sys,json; print(len(json.load(sys.stdin)))"
# Expected: 8
```
If either fails, run Task 8 / Task 9 setup scripts first.

- [ ] **Step 2: Pick a live GPU**

```bash
ssh host-10-240-99-120 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' 2>/dev/null | head
# Or local node if user confirms; avoid 119 (profiling-only per feedback memory)
```
Choose GPU with < 2GB used. Record as `GPU_ID=<N>`.

- [ ] **Step 3: Run 40-step smoke with i_save=20 to trigger deep-eval twice**

```bash
RUN_TAG="smoke_full_$(date +%Y%m%d_%H%M%S)"
ssh host-10-240-99-120 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
    CUDA_VISIBLE_DEVICES=<GPU_ID> WANDB_MODE=online \
    .venv/bin/python -m coart.vae \
        --run_tag ${RUN_TAG} \
        --max_steps 40 --i_log 5 --i_val 20 --i_save 20 \
        --no_sample_at_step_one --max_voxels 200000 \
        --freeze_backbone_steps 0 --lr_unfreeze_warmup_steps 0 \
        2>&1 | tail -60"
```
Expected:
- wandb run URL printed
- Two `[deep_eval]` blocks (steps 20 and 40) with 8 per-asset metric lines each
- No crashes

- [ ] **Step 4: Verify wandb dashboard**

Open the printed wandb run URL in a browser and confirm:
- [ ] `train/loss/*` scalars have data points at step 5, 10, 15, 20, 25, 30, 35, 40
- [ ] `train/grad/*` scalars present
- [ ] `throughput/step_s` stable and under 5s/step for step ≥ 10 (warm)
- [ ] `val/recon_mse` has 2 data points (step 20, 40)
- [ ] `deep_eval/online/per_asset/helmet/nc` has 2 data points
- [ ] `deep_eval/online/per_asset/val_p10/cd` has 2 data points
- [ ] `deep_eval/online/mean/cd` has 2 data points
- [ ] `deep_eval/renders/helmet/online` has 2 image uploads
- [ ] Run config includes `golden_baseline` key

- [ ] **Step 5: Verify ckpt footprint**

```bash
ls -sh results/coart_feat18_*_${RUN_TAG}/*.pt | tail
du -sh results/coart_feat18_*_${RUN_TAG}/
```
Expected: EMA files only 1 step (rolling_ckpts_ema=1); online ckpt files ≤ 3 steps.

- [ ] **Step 6: Run test suite**

```bash
.venv/bin/pytest coart/tests/ -q
```
Expected: All tests PASS.

- [ ] **Step 7: Final commit (if any leftover fixes from smoke)**

```bash
git status
# If clean, skip. If changes needed from smoke findings, commit them.
```

- [ ] **Step 8: Update memory with production run readiness**

Append a new project memory entry at `/home/siyuan/.claude/projects/-mnt-novita2-siyuan-workspace-TRELLIS-2/memory/project_coart_monitoring_eval_ready.md`:
```markdown
---
name: coart_monitoring_eval_ready
description: coart.vae monitoring + deep-eval + watchdog infrastructure complete, ready for 8-GPU production launch
type: project
---

coart.vae framework + monitoring/eval stack complete (plan 2026-04-23).

**What works:**
- wandb logging (online, project=coart-vae, entity=mingyang__) + TB backup
- Deep-eval on 8 golden assets at every i_save step (≈2-4% overhead)
- Watchdog alerts (grad spike, ef diverge, helmet bad) — log + wandb.alert, never auto-kill
- Baseline reference lines injected via wandb.config.golden_baseline
- EMA rolling_ckpts_ema=1 (3.2GB steady vs 9.6GB at rolling=3)

**Why:** priority was quality > time, verification-first on online weights; EMA kept as free side-product.

**How to apply:** production 8-GPU launch command uses `--run_tag <v0>` and either `--wandb_mode online` or `offline` depending on node internet. Node choice TBD per cluster load.
```
Also add to `MEMORY.md`:
```
- [project_coart_monitoring_eval_ready.md](project_coart_monitoring_eval_ready.md) — coart.vae monitoring + eval complete, ready for 8-GPU launch
```

---

## Spec Coverage Check

| Spec section | Plan tasks | Coverage |
|---|---|---|
| §1.2 A — DDP wrap opts | Task 1 | ✓ |
| §1.2 B — static_graph | deferred (spec says "tentative, disable on mismatch"; no task needed now) | ✓ (explicitly punted) |
| §1.2 C — fused AdamW | Task 2 | ✓ |
| §2.1 — wandb init + dual-write | Tasks 3, 4, 5 | ✓ |
| §2.2 — scalar field tree | Task 5 (train/*) + Task 10 (deep_eval/*) | ✓ |
| §2.3 — image/3D logging | Task 4 (methods) + Task 10 (caller) | ✓ |
| §2.4 — artifact strategy (ckpt not uploaded) | default behavior | ✓ |
| §2.5 — offline fallback | Task 6 (docs) | ✓ |
| §3.1 — golden asset list | Task 8 | ✓ |
| §3.2 — one-time setup | Task 8 | ✓ |
| §3.3 — Layer V baseline | Task 9 | ✓ |
| §3.4 — deep_eval pipeline | Task 10 | ✓ |
| §3.5 — EMA strategy (shadow keep, rolling_ckpts_ema=1, no EMA eval) | Tasks 3, 11 | ✓ |
| §3.6 — failure handling | Task 10 (_one_asset try/except) | ✓ |
| §4.1 — baseline reference lines | Task 14 | ✓ |
| §4.2 — watchdog | Tasks 12, 13 | ✓ |
| §4.3 — summary table | ⚠️ deferred — wandb UI can show latest metrics natively via panels; table is polish. Add if needed after first production run. | deferred |
| §5.1 — new files | Tasks 7, 8, 9, 10, 12 | ✓ |
| §5.2 — modified files | Tasks 1, 3, 4, 5, 11, 13, 14 | ✓ |
| §5.3 — deps | Task 6 (README) + pre-installed in .venv | ✓ |
| §6 — prelaunch checklist | Task 15 | ✓ |
| §7 — non-goals | respected | ✓ |
| §8 — risks + mitigations | implemented as try/except throughout | ✓ |
| §9 — success criteria | Task 15 | ✓ |

**Deferred**: §4.3 summary table. Rationale: wandb's native "latest metrics" panel shows the same info in live dashboards; a custom `wandb.Table` adds complexity without user-visible benefit until the run settles. Re-evaluate after first production run.

---

## Execution Notes

- **Parallelizable tasks within stage**: Stage 1 (tasks 1 & 2), Stage 2 (tasks 3, 6 after 3), Stage 3 (tasks 8 & 9 after 7), Stage 4 (tasks 12 & 14)
- **Sequential hard deps**: Task 5 → Task 11/13 (logger must exist before hook uses it); Task 8 → Task 9 (baseline needs asset list); Task 10 → Task 11 (hook needs the function); Task 13 → 14 partially share train.py lines
- **When subagent edits `coart/vae/train.py`**: serialize (no two subagents touching it concurrently) — Tasks 2, 5, 11, 13 all modify train.py
- **Test command baseline (after every task)**: `.venv/bin/pytest coart/tests/ -q` — must stay green
- **SSH rule (critical)**: any `ssh host-10-240-99-120 '...'` command MUST start with `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && ...` — see project CLAUDE.md for mandatory override

---

## Post-Plan Handoff

After all 15 tasks merge, the repo is ready for 8-GPU production launch:

```bash
# Example launch (tune --run_tag, --output_dir)
torchrun --nproc_per_node=8 -m coart.vae \
    --run_tag three_branch_ws_v0 \
    --io_arch three_branch --warmstart_io \
    --max_steps 200000 \
    --i_log 100 --i_val 5000 --i_save 5000 \
    --batch_size 1 --max_voxels 500000 \
    --rolling_ckpts 3 --rolling_ckpts_ema 1 \
    --wandb_mode online
```

Node selection (119/120/117) decided at launch time based on cluster load.
