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
from typing import Any, Dict, List, Optional

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
                    print(
                        f"[logger] wandb.init failed ({e}); falling back to TB-only",
                        file=sys.stderr,
                    )
                    self._wandb = None

                # Inject baseline reference data via wandb.config so panels can
                # render horizontal reference lines.
                if self._wandb is not None:
                    try:
                        import json as _json
                        baseline_path = os.path.join(
                            os.path.dirname(os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__)))),
                            "coart", "eval", "golden_baseline.json",
                        )
                        if os.path.exists(baseline_path):
                            with open(baseline_path) as _fh:
                                self._wandb.config.update(
                                    {"golden_baseline": _json.load(_fh)},
                                    allow_val_change=True,
                                )
                    except Exception as e:
                        print(f"[logger] baseline config inject skipped ({e})",
                              file=sys.stderr)

    def scalar(self, tag: str, value: float, step: int) -> None:
        if not self.is_master:
            return
        self._buf.setdefault(tag, []).append(float(value))

    def flush_if_due(self, step: int, i_log: int) -> None:
        """Rank-0-local flush. DO NOT add all_reduce here: not every rank
        has the same set of tags in its buffer (master logs extras like
        deep_eval/render/val scalars), so a collective op would deadlock
        on shape mismatch. Training loss averaging is already handled by
        DDP backward; per-rank values are fine for dashboards.
        """
        if not self.is_master:
            return
        if step % i_log != 0 or not self._buf:
            return
        tags = sorted(self._buf.keys())
        vals = {t: sum(self._buf[t]) / len(self._buf[t]) for t in tags}
        for t, v in vals.items():
            if self._writer is not None:
                self._writer.add_scalar(t, float(v), step)
        if self._wandb is not None:
            try:
                self._wandb.log({t: float(v) for t, v in vals.items()},
                                step=step)
            except Exception as e:
                print(f"[logger] wandb.log failed ({e})", file=sys.stderr)
        self._buf.clear()

    def image(self, tag: str, np_img: np.ndarray, step: int) -> None:
        """Log a HxWx3 uint8 image."""
        if not self.is_master:
            return
        if self._writer is not None:
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
