"""Three soft watchdog conditions; log alerts + dump rolling window for debug.

Public API:
    wd = Watchdog(output_dir, logger)
    wd.update_train(step, grad_pre, grad_post, grad_p95, loss_ef, lr)
    wd.check_train(step)   # may emit alert
    wd.update_helmet(step, helmet_nc)   # from deep_eval return dict

Conditions:
    A. grad_spike: grad_pre > 100 × p95 for 50 consecutive steps
    B. ef_diverge: loss_ef monotonically increasing over window [5000, 10000]
    C. helmet_bad: deep_eval helmet NC < 0.50 when step >= 10000

All alerts write stderr + wandb.alert; training never terminates.
Each condition fires at most once per run; debug dump path:
    results/<run>/watchdog_<cond>_step<N>.json
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional


class Watchdog:
    GRAD_SPIKE_RATIO = 100.0
    GRAD_SPIKE_CONSECUTIVE = 50
    EF_WINDOW_START = 5000
    EF_WINDOW_END = 10000
    HELMET_BAD_THRESHOLD = 0.50
    HELMET_BAD_MIN_STEP = 10000

    def __init__(self, output_dir: str, logger):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self._grad_spike_count = 0
        self._grad_spike_fired = False
        self._ef_hist: deque = deque(maxlen=self.EF_WINDOW_END)
        self._ef_fired = False
        self._helmet_fired = False
        self._state_window: deque = deque(maxlen=200)

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
        if (not self._grad_spike_fired
                and self._grad_spike_count >= self.GRAD_SPIKE_CONSECUTIVE):
            last = self._state_window[-1] if self._state_window else {}
            self.logger.alert(
                title="grad spike",
                text=(f"@step={step} pre_clip={last.get('grad_pre'):.2e} "
                      f"vs p95={last.get('grad_p95'):.2e}"),
                level="WARN",
            )
            self._dump("grad_spike", step, {"trigger_state": dict(last)})
            self._grad_spike_fired = True
        if (not self._ef_fired and step >= self.EF_WINDOW_END
                and len(self._ef_hist) >= 2):
            steps, losses = zip(*self._ef_hist)
            deltas = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
            if all(d >= 0 for d in deltas) and (losses[-1] - losses[0]) > 0.01:
                self.logger.alert(
                    title="ef diverge",
                    text=f"@step={step} Δ=+{losses[-1] - losses[0]:.4f}",
                    level="WARN",
                )
                self._dump(
                    "ef_diverge", step, {"ef_hist": list(self._ef_hist)},
                )
                self._ef_fired = True

    def update_helmet(self, step: int, helmet_nc: Optional[float]) -> None:
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
