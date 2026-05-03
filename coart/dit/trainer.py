"""Cached-features trainer for shape DiT finetune.

Inherits from ``trellis2.trainers.ImageConditionedSparseFlowMatchingCFGTrainer``
and adds:

* ``_init_image_cond_model``: NO-OP (cached DINO features fed by dataset).
* ``encode_image``: identity passthrough.
* ``snapshot_dataset``/``snapshot``: NO-OP (sparse outputs incompatible with
  the upstream dense-tensor probe path).
* ``save``: extra ckpt rotation (keep last_n + every milestone + step 0).
* ``save_logs``: throughput / dataloader wait / mem-peak metrics, wandb
  mirror, and periodic hold-out eval loss.
* ``load_data``: timing wrap to expose dataloader wait time.

Registered into ``trellis2.trainers`` namespace at import time so ``train.py``
can resolve us by name from JSON config.
"""
from __future__ import annotations

import glob
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from trellis2.trainers import ImageConditionedSparseFlowMatchingCFGTrainer

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


__all__ = ["CachedImageConditionedSparseFlowMatchingCFGTrainer"]


class CachedImageConditionedSparseFlowMatchingCFGTrainer(
    ImageConditionedSparseFlowMatchingCFGTrainer
):
    """Image-conditioned sparse FM CFG trainer that consumes cached features."""

    def __init__(
        self,
        *args,
        image_cond_model: Any = None,
        # Extras (not in upstream)
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: str = "online",
        i_eval: int = 2500,
        eval_n: int = 64,
        ckpt_keep_last_n: int = 5,
        ckpt_milestone_every: int = 20000,
        # Override upstream BasicTrainer's hardcoded num_workers (=cpu/gpus = 16/rank).
        # 16x8=128 spawn-method dataloader children re-import trellis2 over NFS,
        # causing 10+ min spawn cascade hang. 4/rank is plenty for cached .npz IO.
        num_workers: Optional[int] = None,
        # W2.1: parallelism dispatch — "ddp" (default) | "zro1" | "fsdp2_zero2".
        parallel_mode: str = "ddp",
        # W2.2: optional FSDP2 sub-config; consumed by parallel/fsdp2.py.
        # Not forwarded to super() — upstream BasicTrainer doesn't accept it.
        fsdp2: Optional[dict] = None,
        **kwargs,
    ):
        # Set BEFORE super().__init__ — BasicTrainer.__init__ calls our
        # prepare_dataloader override, which reads _num_workers_override.
        self._num_workers_override = num_workers

        # W2.2: Stash FSDP2 sub-config BEFORE super().__init__ so that
        # init_models_and_more (called from super) can read it via
        # trainer.fsdp2_config when dispatcher init_after_super runs.
        self.fsdp2_config = fsdp2 or {}

        # NOTE on parallel_mode propagation:
        # Upstream BasicTrainer.__init__ already declares parallel_mode='ddp'
        # as an explicit kwarg and sets self.parallel_mode from it. Our
        # subclass also declares parallel_mode (so we can document it +
        # default it cleanly), but we MUST forward the value to super(),
        # otherwise super resets self.parallel_mode back to its default 'ddp'.
        # Passing it via super().__init__(parallel_mode=parallel_mode, ...)
        # is the single source of truth — super sets self.parallel_mode and
        # then calls init_models_and_more, which reads it.

        # Forward a dummy image_cond_model dict to satisfy the base mixin's
        # __init__ signature. The dict is never read because we override
        # ``_init_image_cond_model`` and ``encode_image`` below.
        super().__init__(
            *args,
            image_cond_model=image_cond_model or {"name": "_unused", "args": {}},
            parallel_mode=parallel_mode,
            **kwargs,
        )

        self.i_eval = i_eval
        self.eval_n = eval_n
        self.ckpt_keep_last_n = ckpt_keep_last_n
        self.ckpt_milestone_every = ckpt_milestone_every

        self._last_load_wait_s = 0.0
        self._eval_batch_cache: Optional[Any] = None  # built lazily

        self._wandb_run = None
        if self.is_master and wandb_project:
            if _wandb is None:
                print("[wandb] not installed — skipping mirror.")
            else:
                try:
                    self._wandb_run = _wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=wandb_run_name or os.path.basename(self.output_dir.rstrip("/")),
                        dir=self.output_dir,
                        config={
                            "output_dir": self.output_dir,
                            "step_at_init": int(self.step),
                            "world_size": int(self.world_size),
                            "batch_size": int(self.batch_size),
                            "batch_size_per_gpu": int(self.batch_size_per_gpu),
                            "batch_split": int(self.batch_split),
                            "ema_rate": list(map(float, self.ema_rate)),
                        },
                        resume="allow",
                        mode=wandb_mode,
                    )
                except Exception as e:
                    # No API key, network down, project quota — never let it
                    # take down the whole DDP training run. tb_logs + log.txt
                    # always work without wandb.
                    print(
                        f"[wandb] init failed ({type(e).__name__}: {e}); "
                        f"continuing with tb-only logging.",
                        flush=True,
                    )
                    self._wandb_run = None

    # -------------------------------------------------------------- model init
    def _init_image_cond_model(self) -> None:
        """No-op: features are already cached on disk + supplied by the dataset."""
        self.image_cond_model = "cached"

    # -------------------------------------------------------------- encode path
    @torch.no_grad()
    def encode_image(self, image):
        """Identity passthrough. ``image`` is already the cached (B, T, 1024) feats."""
        if self.image_cond_model is None:
            self._init_image_cond_model()
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"CachedImageConditionedSparseFlowMatchingCFGTrainer expects the "
                f"dataset to emit pre-extracted feature tensors, got {type(image)}"
            )
        return image

    # ---------------------------------------------------------------- snapshot
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """No-op: SparseTensor is not torch.stack-able."""
        pass

    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """No-op: SparseTensor sample output has no .contiguous()."""
        pass

    # --------------------------------------------------------- dataloader override
    def prepare_dataloader(self, **kwargs):
        """Override upstream basic.py:271 to make num_workers configurable.
        Upstream hardcodes ``ceil(cpu_count / device_count) = 16/rank`` which
        × 8 ranks = 128 spawn-method dataloader children. Each re-imports
        trellis2 + coart.dit over NFS, causing 10+ min spawn cascade hangs.
        For our cached .npz IO, num_workers=4/rank is plenty.
        Honors ``trainer.args.num_workers`` from JSON config; falls back to
        the upstream formula if unset."""
        from trellis2.utils.data_utils import ResumableSampler, cycle as _cycle
        from torch.utils.data import DataLoader as _DataLoader
        nw = self._num_workers_override
        if nw is None:
            import os as _os
            import torch as _torch
            import numpy as _np
            nw = int(_np.ceil(_os.cpu_count() / max(_torch.cuda.device_count(), 1)))
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        # C9: prefetch_factor=4 (was default 2) lets each worker keep more
        # batches queued, smoothing tail-latency on slow .npz reads.
        # persistent_workers=True avoids worker re-spawn between epochs (already
        # implicitly on whenever nw>0; kept explicit for clarity).
        _dl_kwargs = dict(
            batch_size=self.batch_size_per_gpu,
            num_workers=nw,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(nw > 0),
            collate_fn=getattr(self.dataset, "collate_fn", None),
            sampler=self.data_sampler,
        )
        if nw > 0:
            _dl_kwargs["prefetch_factor"] = 4
        self.dataloader = _DataLoader(self.dataset, **_dl_kwargs)
        self.data_iterator = _cycle(self.dataloader)

    # ----------------------------------------------------------- parallel hook
    def init_models_and_more(self, *args, **kwargs):
        """W2.1: dispatch parallel-mode setup after base model + optimizer init.

        - mode="ddp": apply C4 (gradient_as_bucket_view=True) on DDP wraps.
        - mode="zro1": replace optimizer with ZeroRedundancyOptimizer.
        - mode="fsdp2_zero2": (W2.2) — FSDP2 wrap.

        The previous inline C4 monkey-patch lived here; that logic now lives in
        coart/dit/parallel/ddp.py and is invoked via the dispatcher.
        """
        super().init_models_and_more(*args, **kwargs)
        from .parallel import get_dispatcher
        dispatcher = get_dispatcher(self.parallel_mode)
        dispatcher.init_after_super(self, **kwargs)

    # ---------------------------------------------------------------- profiler
    def profile(self, wait=2, warmup=3, active=5):
        """Override broken upstream basic.py:898-911 — original calls
        self.run_step() with no args, but run_step requires data_list.
        Mirror the run() loop's pattern: load_data() then run_step(data_list)."""
        import os as _os
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                _os.path.join(self.output_dir, "profile")),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                data_list = self.load_data()
                self.run_step(data_list)
                prof.step()

    # ---------------------------------------------------------------- load wrap
    def load_data(self):
        """Wrap base load_data to record dataloader wait time."""
        t0 = time.time()
        out = super().load_data()
        if self.is_master:
            self._last_load_wait_s = time.time() - t0
        return out

    # ---------------------------------------------------------------- save_logs
    def save_logs(self):
        # Inject perf metrics into the most recent log entry produced by the
        # base run() loop. These are scalar reads, no GPU sync.
        if self.is_master and len(self.log) > 0:
            entry = self.log[-1][1]
            perf: dict = entry.setdefault("perf", {})
            perf["dataloader_wait_s"] = float(self._last_load_wait_s)
            try:
                step_t = float(entry.get("time", {}).get("step", 0.0))
                input_size = float(entry.get("elastic", {}).get("input_size", 0.0))
                if step_t > 0:
                    perf["throughput_step_per_h"] = 3600.0 / step_t
                    if input_size > 0:
                        perf["throughput_tok_per_s"] = input_size / step_t
            except Exception:
                pass
            perf["mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            perf["mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        # Snapshot what's about to be flushed (super wipes self.log)
        log_snapshot = list(self.log) if self.is_master else []

        super().save_logs()  # writes log.txt + tb scalars; clears self.log

        # Wandb mirror — same scalars that went to tb
        if self._wandb_run and log_snapshot:
            from trellis2.utils.general_utils import dict_any, dict_reduce, dict_flatten
            log_show = [l for _, l in log_snapshot if not dict_any(l, lambda x: np.isnan(x))]
            if log_show:
                merged = dict_reduce(log_show, lambda x: float(np.mean(x)))
                flat = dict_flatten(merged, sep="/")
                self._wandb_run.log(
                    {k: v for k, v in flat.items() if isinstance(v, (int, float))},
                    step=int(self.step),
                )

        # Periodic hold-out eval (rank 0 writes; other ranks compute idempotently)
        if self.i_eval and self.step > 0 and self.step % self.i_eval == 0:
            self._run_eval()

    # ---------------------------------------------------------------- eval
    def _build_eval_batch(self):
        """Pick a stable, fixed N-sample batch from the dataset.

        We pick by quantile of `dataset.loads` so the eval set spans the
        full token-count distribution — this gives more interpretable
        eval/loss across t-bins. Fixed seed → deterministic across resumes.
        """
        n = min(self.eval_n, len(self.dataset))
        loads = np.asarray(self.dataset.loads)
        order = np.argsort(loads)
        # Evenly stride across the load distribution.
        step_stride = max(len(order) // n, 1)
        idx = order[::step_stride][:n].tolist()
        sub = Subset(self.dataset, idx)
        # split_size=None → returns single dict {x_0, cond, ...}
        loader = DataLoader(
            sub,
            batch_size=n,
            num_workers=0,
            shuffle=False,
            collate_fn=partial(self.dataset.collate_fn, split_size=None),
        )
        batch = next(iter(loader))
        from trellis2.utils.data_utils import recursive_to_device
        batch = recursive_to_device(batch, self.device, non_blocking=False)
        return batch

    @torch.no_grad()
    def _run_eval(self):
        """Compute hold-out forward loss + per-bin mse.

        All ranks run the same forward (replicated, identical result) so we
        avoid DDP coordination — no all_reduce, no barrier needed. Cost is
        ~3-5s per call (single batch of `eval_n` samples). At i_eval=2500
        the amortised overhead is well under 0.1%.

        Only rank 0 writes the result.
        """
        try:
            if self._eval_batch_cache is None:
                self._eval_batch_cache = self._build_eval_batch()
            batch = self._eval_batch_cache

            self.denoiser.eval()
            amp_ctx = (
                partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
                if self.mix_precision_mode == "amp"
                else nullcontext
            )
            with amp_ctx():
                terms, _ = self.training_losses(**batch)
            self.denoiser.train()

            if not self.is_master:
                return

            flat = {}
            for k, v in terms.items():
                if isinstance(v, torch.Tensor):
                    flat[k] = v.item()
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            flat[f"{k}/{kk}"] = vv.item()
                        elif isinstance(vv, (int, float)):
                            flat[f"{k}/{kk}"] = float(vv)
                elif isinstance(v, (int, float)):
                    flat[k] = float(v)

            for k, v in flat.items():
                self.writer.add_scalar(f"eval/{k}", v, self.step)
            if self._wandb_run:
                self._wandb_run.log({f"eval/{k}": v for k, v in flat.items()}, step=int(self.step))

            print(
                f"\n[eval @ step {self.step}] "
                f"loss={flat.get('loss', float('nan')):.4f}  "
                f"mse={flat.get('mse', float('nan')):.4f}",
                flush=True,
            )
        except Exception as e:
            if self.is_master:
                print(f"[eval] failed at step {self.step}: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- run_step
    def run_step(self, data_list):
        """Wrap base run_step to inject the per-rank pre-save collective.

        Upstream basic.py:885-886 invokes self.save() ONLY from rank 0 (gated
        by `if self.is_master:` at line 860). But ZeroRedundancyOptimizer's
        consolidate_state_dict is a COLLECTIVE that requires participation
        from all ranks. We therefore cannot put consolidation inside save().

        Hook here: after the actual training step, before the master enters
        its save block, all ranks have already incremented self.step (the
        increment happens in run() between run_step return and the save
        check). Run consolidate on all ranks when the next save will trigger.

        W2.2 extension: for fsdp2_zero2 mode, also (a) update EMA on every
        rank (upstream only updates on rank 0; we want per-rank shard EMA),
        and (b) drive the DCP collective save on all ranks here, then have
        the rank-0 save() override skip the actual write (rotation only).
        """
        # FSDP2: suppress the rank-0-only EMA update inside upstream run_step
        # (we drive the all-rank distributed EMA explicitly below). The flag
        # is honored by our update_ema override.
        if self.parallel_mode == "fsdp2_zero2":
            self._suppress_inner_ema = True
            try:
                result = super().run_step(data_list)
            finally:
                self._suppress_inner_ema = False
            # All ranks update their local EMA shard.
            self.update_ema()
        else:
            result = super().run_step(data_list)

        # self.step has NOT been incremented yet at this point — run() does
        # the +=1 after run_step returns. So check (step + 1).
        if self.world_size > 1 and (self.step + 1) % self.i_save == 0:
            from .parallel import get_dispatcher

            dispatcher = get_dispatcher(self.parallel_mode)
            dispatcher.consolidate_for_save(self)
            # FSDP2: do the DCP collective save here (all ranks). The rank-0
            # save() override is a no-op write (rotation only) in this mode.
            if self.parallel_mode == "fsdp2_zero2":
                from .parallel.fsdp2 import save_state

                next_step = self.step + 1
                path = os.path.join(
                    self.output_dir, "ckpts", f"denoiser_step{next_step:07d}.pt"
                )
                save_state(self, path)
        return result

    # ---------------------------------------------------------------- update_ema
    def update_ema(self):
        """Dispatch EMA update.

        DDP / ZRO: upstream rank-0-only EMA on `master_params`/`ema_params`.
        FSDP2:     per-rank distributed EMA on local DTensor shards.

        For FSDP2 the rank-0-only EMA call inside upstream run_step is
        suppressed via `_suppress_inner_ema`; the explicit per-rank update
        is driven from our run_step override after super() returns.
        """
        if getattr(self, "_suppress_inner_ema", False):
            return
        if self.parallel_mode == "fsdp2_zero2":
            from .parallel.fsdp2 import update_ema as _fsdp_update_ema

            _fsdp_update_ema(self)
            return
        super().update_ema()

    # ---------------------------------------------------------------- check_ddp
    def check_ddp(self):
        """Skip consistency check for FSDP2 — params are sharded DTensors and
        all_gather on master_params would not be meaningful."""
        if self.parallel_mode == "fsdp2_zero2":
            return
        return super().check_ddp()

    # ---------------------------------------------------------------- save
    def save(self, non_blocking=True):
        """Delegate to base then rotate. consolidate_for_save runs from
        run_step on all ranks (see run_step docstring) — save() runs only
        on rank 0 per upstream's `if self.is_master:` gate at basic.py:860.

        For fsdp2_zero2: the actual DCP write happened in run_step on all
        ranks; here on rank 0 we only rotate the on-disk ckpt set.
        """
        if self.parallel_mode == "fsdp2_zero2":
            if self.is_master:
                self._rotate_ckpts()
            return
        super().save(non_blocking=non_blocking)
        if self.is_master:
            self._rotate_ckpts()

    # ---------------------------------------------------------------- load
    def load(self, load_dir, step=0):
        """For fsdp2_zero2 use the DCP path; otherwise upstream tensor load."""
        if self.parallel_mode == "fsdp2_zero2":
            from .parallel.fsdp2 import load_state

            path = os.path.join(load_dir, "ckpts", f"denoiser_step{step:07d}.pt")
            load_state(self, path)
            self.step = step
            return
        return super().load(load_dir, step=step)

    # -------------------------------------------------------- finetune_from
    def finetune_from(self, finetune_ckpt):
        """For fsdp2_zero2 dispatch to DCP load (broadcast_from_rank0 handles
        the DDP→FSDP2 reshard automatically). Otherwise upstream behavior."""
        if self.parallel_mode == "fsdp2_zero2":
            from .parallel.fsdp2 import load_state

            path = (
                finetune_ckpt["denoiser"]
                if isinstance(finetune_ckpt, dict)
                else finetune_ckpt
            )
            load_state(self, path)
            return
        return super().finetune_from(finetune_ckpt)

    def _rotate_ckpts(self):
        """Keep: last N steps + every milestone step + step 0 warmstart."""
        ckpt_dir = os.path.join(self.output_dir, "ckpts")
        if not os.path.isdir(ckpt_dir):
            return
        steps: set = set()
        for fn in os.listdir(ckpt_dir):
            if "_step" in fn and fn.endswith(".pt"):
                try:
                    s = int(fn.rsplit("_step", 1)[1].rsplit(".", 1)[0])
                    steps.add(s)
                except (ValueError, IndexError):
                    pass
        if not steps:
            return
        sorted_steps = sorted(steps)
        keep: set = {0}
        keep.update(sorted_steps[-self.ckpt_keep_last_n :])
        keep.update(s for s in sorted_steps if s % self.ckpt_milestone_every == 0)
        for s in sorted_steps:
            if s in keep:
                continue
            for fn in glob.glob(os.path.join(ckpt_dir, f"*_step{s:07d}.pt")):
                try:
                    os.remove(fn)
                except OSError:
                    pass


# ---------------------------------------------------------------------- registration
def _register() -> None:
    import trellis2.trainers as tt

    setattr(
        tt,
        "CachedImageConditionedSparseFlowMatchingCFGTrainer",
        CachedImageConditionedSparseFlowMatchingCFGTrainer,
    )


_register()
