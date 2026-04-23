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


def _build_optimizer(trainable_params, lr: float) -> torch.optim.Optimizer:
    """AdamW with fused=True when supported; otherwise plain AdamW.

    Fused kernel requires CUDA + PyTorch >= 2.0; falls back on CPU or
    older torch. Any failure is logged to stderr and demoted to fused=False.
    """
    try:
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0, fused=True)
    except (TypeError, RuntimeError) as e:
        import sys
        print(f"[coart] fused AdamW unavailable ({e}); falling back to fused=False",
              file=sys.stderr)
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0, fused=False)


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

    # ---------------- detect mode: base training vs resume ----------------
    # Base mode:    load pretrained TRELLIS.2 weights + warm-start IO + honour
    #               freeze_backbone_steps warmup. Used for the first run.
    # Resume mode:  load a coart ckpt (encoder/decoder/optimizer/EMA/RNG) and
    #               continue from that step. Used after preempt / OOM restart.
    # Auto-detected from `resume_from` and presence of an existing ckpt.
    resume_info = None  # tuple[path, step] | None
    if cfg.resume_from != "none":
        if cfg.resume_from == "latest":
            found = find_latest_ckpt(cfg.output_dir, prefix="ckpt")
            if found is not None:
                resume_info = found
        else:
            st = int(cfg.resume_from.split("_step")[-1].split(".pt")[0])
            resume_info = (cfg.resume_from, st)
    mode = "resume" if resume_info is not None else "base"
    _log(
        f"[mode] {mode}"
        + (f" (from step {resume_info[1]})" if resume_info else "")
    )

    # ---------------- build models ----------------
    encoder, decoder = build_models(
        latent_channels=cfg.latent_channels,
        device=device, io_arch=cfg.io_arch,
    )
    _log(f"[model] {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M enc "
         f"+ {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M dec params")

    # ---------------- mode-specific init ----------------
    # We need to decide the trainable param set BEFORE building the optimizer
    # so that optimizer.load_state_dict() matches saved param groups.
    step = 0
    start_epoch = 0
    if mode == "base":
        # Load pretrained TRELLIS.2 weights + optional warm-start IO stems.
        if cfg.from_pretrained:
            load_pretrained_into(
                encoder, decoder,
                enc_path=cfg.enc_pretrained, dec_path=cfg.dec_pretrained,
                io_arch=cfg.io_arch, warmstart_io=cfg.warmstart_io,
                verbose=is_master,
            )
        # Freeze backbone for the warmup window (train IO + KL stems only).
        if cfg.freeze_backbone_steps > 0:
            _set_backbone_requires_grad(encoder, decoder, requires_grad=False)
            unfrozen = False
            step_at_unfreeze: int | None = None
        else:
            unfrozen = True
            step_at_unfreeze = 0
    else:
        # Resume: peek misc checkpoint to restore the exact trainable state
        # that existed when the ckpt was saved, so optimizer.load_state_dict
        # finds matching param groups.
        resume_path, resume_step = resume_info
        misc_path = resume_path.replace("ckpt_step", "misc_step")
        saved_unfrozen: bool
        saved_step_at_unfreeze: int | None
        if os.path.exists(misc_path):
            # weights_only=False because misc contains numpy RNG state; we trust our own ckpt.
            misc_peek = torch.load(misc_path, map_location="cpu", weights_only=False)
            saved_unfrozen = bool(
                misc_peek.get("unfrozen", resume_step >= cfg.freeze_backbone_steps)
            )
            saved_step_at_unfreeze = misc_peek.get("step_at_unfreeze")
        else:
            saved_unfrozen = resume_step >= cfg.freeze_backbone_steps
            saved_step_at_unfreeze = (
                cfg.freeze_backbone_steps if saved_unfrozen else None
            )
        if not saved_unfrozen:
            _set_backbone_requires_grad(encoder, decoder, requires_grad=False)
        unfrozen = saved_unfrozen
        step_at_unfreeze = saved_step_at_unfreeze
        _log(
            f"[resume] trainable state @ ckpt: unfrozen={unfrozen}, "
            f"step_at_unfreeze={step_at_unfreeze}"
        )

    # ---------------- EMA (before DDP wrap) ----------------
    ema = EMAModel(encoder, decay=cfg.ema_rate) if cfg.use_ema else None
    ema_dec = EMAModel(decoder, decay=cfg.ema_rate) if cfg.use_ema else None

    # ---------------- DDP wrap ----------------
    if is_dist:
        encoder = wrap_ddp(encoder, local_rank)
        decoder = wrap_ddp(decoder, local_rank)

    trainable = _trainable(encoder, decoder)
    optimizer = _build_optimizer(trainable, lr=cfg.lr)

    grad_clipper = AdaptiveGradClipper(
        max_norm=cfg.grad_clip_max, clip_percentile=cfg.grad_clip_pct,
    )

    # ---------------- resume: load state dicts into the freshly-built modules ----------------
    if mode == "resume":
        resume_path, resume_step = resume_info
        _log(f"[resume] loading {resume_path} (step {resume_step})")
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        unwrap(encoder).load_state_dict(ck["encoder"])
        unwrap(decoder).load_state_dict(ck["decoder"])
        optimizer.load_state_dict(ck["optimizer"])
        step = resume_step

        misc_path = resume_path.replace("ckpt_step", "misc_step")
        if os.path.exists(misc_path):
            misc = torch.load(misc_path, map_location="cpu", weights_only=False)
            start_epoch = int(misc.get("epoch", 0))
            rng_np = misc.get("rng_np")
            rng_torch = misc.get("rng_torch")
            if rng_np is not None:
                np.random.set_state(rng_np)
            if rng_torch is not None:
                torch.set_rng_state(rng_torch.cpu())
        if cfg.use_ema:
            enc_ema = resume_path.replace(
                "ckpt_step", f"ema_{cfg.ema_rate}_enc_step"
            )
            dec_ema = resume_path.replace(
                "ckpt_step", f"ema_{cfg.ema_rate}_dec_step"
            )
            if os.path.exists(enc_ema):
                ema.load_state_dict(
                    torch.load(enc_ema, map_location="cpu", weights_only=False)
                )
            if os.path.exists(dec_ema):
                ema_dec.load_state_dict(
                    torch.load(dec_ema, map_location="cpu", weights_only=False)
                )
        _log(f"[resume] ok — step={step}, epoch={start_epoch}, unfrozen={unfrozen}")

    # ---------------- TB + wandb logger ----------------
    from dataclasses import asdict
    logger = CoartTBLogger(
        cfg.output_dir,
        is_master=is_master,
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

    from coart.eval.watchdog import Watchdog
    watchdog = Watchdog(cfg.output_dir, logger)

    pbar = tqdm(total=cfg.max_steps, initial=step, desc="coart.vae",
                dynamic_ncols=True, disable=not is_master)
    t0 = time.time()
    _step_t0 = time.monotonic()
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
            _grad_pre_clip = float(grad_clipper(trainable))
            _grad_p95 = (float(grad_clipper._max_norm)
                         if grad_clipper._max_norm is not None else 0.0)
            _grad_post_clip = (min(_grad_pre_clip, _grad_p95)
                               if _grad_p95 > 0 else _grad_pre_clip)

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

            # Per-component loss scalars (TB + wandb)
            for k in ("total", "recon", "recon_p1", "recon_p2", "recon_ef",
                      "kl", "subdiv"):
                logger.scalar(f"train/loss/{k}",
                              losses[k].detach().item(), step)

            # Throughput
            _step_dt = time.monotonic() - _step_t0
            _step_t0 = time.monotonic()
            _batch_nv = int(x.feats.shape[0])
            logger.scalar("throughput/step_s", _step_dt, step)
            logger.scalar("throughput/samples_s_per_gpu",
                          1.0 / max(_step_dt, 1e-6), step)
            logger.scalar("throughput/voxels_s_per_gpu",
                          _batch_nv / max(_step_dt, 1e-6), step)
            logger.scalar("throughput/avg_batch_voxels", float(_batch_nv), step)

            # Grad-norm / clip stats
            logger.scalar("train/grad/norm_pre_clip", _grad_pre_clip, step)
            logger.scalar("train/grad/norm_post_clip", _grad_post_clip, step)
            logger.scalar(
                "train/grad/clip_ratio",
                _grad_post_clip / max(_grad_pre_clip, 1e-8), step,
            )
            logger.scalar("train/grad/p95_rolling", _grad_p95, step)

            # Scheduler state
            logger.scalar("train/sched/lr", float(cur_lr), step)
            logger.scalar("train/sched/unfrozen",
                          1.0 if unfrozen else 0.0, step)
            logger.scalar(
                "train/sched/steps_since_unfreeze",
                float(step - step_at_unfreeze)
                if step_at_unfreeze is not None else 0.0,
                step,
            )

            # Feed watchdog with this step's signals (cheap)
            watchdog.update_train(
                step=step,
                grad_pre=_grad_pre_clip,
                grad_post=_grad_post_clip,
                grad_p95=_grad_p95,
                loss_ef=float(losses["recon_ef"].detach().item()),
                lr=float(cur_lr),
            )

            logger.flush_if_due(step, cfg.i_log)
            if step % cfg.i_log == 0:
                watchdog.check_train(step)

            # Unfreeze
            if (not unfrozen) and step >= cfg.freeze_backbone_steps:
                _set_backbone_requires_grad(unwrap(encoder), unwrap(decoder),
                                            requires_grad=True)
                if is_dist:
                    encoder = wrap_ddp(unwrap(encoder), local_rank)
                    decoder = wrap_ddp(unwrap(decoder), local_rank)
                trainable = _trainable(encoder, decoder)
                optimizer = _build_optimizer(trainable, lr=cfg.lr)
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
                if cfg.use_ema:
                    save_ckpt(ema.state_dict(), cfg.output_dir, step,
                              keep_k=cfg.rolling_ckpts_ema,
                              prefix=f"ema_{cfg.ema_rate}_enc")
                    save_ckpt(ema_dec.state_dict(), cfg.output_dir, step,
                              keep_k=cfg.rolling_ckpts_ema,
                              prefix=f"ema_{cfg.ema_rate}_dec")
                misc = {
                    "epoch": epoch,
                    "step": step,
                    "unfrozen": unfrozen,
                    "step_at_unfreeze": step_at_unfreeze,
                    "rng_np": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                }
                save_ckpt(misc, cfg.output_dir, step,
                          keep_k=cfg.rolling_ckpts, prefix="misc")

            # Deep-eval on 8 golden assets (rank-0 does work, others barrier)
            if step % cfg.i_save == 0 and step > 0:
                try:
                    from coart.eval.deep_eval import run_deep_eval
                    _deep_results = run_deep_eval(
                        encoder=encoder, decoder=decoder,
                        stats={"mean": mean_t, "std": std_t},
                        step=step, logger=logger, cfg=cfg,
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    if is_master:
                        logger.alert(
                            title="deep_eval crashed",
                            text=f"step={step}: {e!r}",
                            level="ERROR",
                        )
                    _deep_results = {}
                # Feed Watchdog with helmet NC from deep_eval (None if missing)
                _helmet_metrics = _deep_results.get("helmet", {})
                watchdog.update_helmet(
                    step=step,
                    helmet_nc=_helmet_metrics.get("nc") if _helmet_metrics else None,
                )

            # Val MSE pass
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
