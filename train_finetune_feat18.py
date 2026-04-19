"""
Fine-tune the TRELLIS.2 shape SC-VAE on the 1k Objaverse-XL Sketchfab subset
encoded as 18-dim CoReP features.

Differences vs train_overfit_feat18.py:
  * Real Dataset / DataLoader over precomputed .npz shards (see
    precompute_feat18.py) instead of re-running corep_fast every step.
  * Random integer translation augmentation in [-max_translate, +max_translate]
    along each axis, picked per-sample so no voxel is dropped.
  * Per-channel input normalisation using the offline stats produced by
    precompute_feat18.py:  point dims → (x - 0.5),  edge/face dims → (x - μ)/σ.
  * Loads the pretrained TRELLIS.2 shape encoder/decoder and warm-starts the
    new (18-channel) input/output linears from the pretrained vertex pathway
    (channels 0..2). Backbone weights are loaded as-is.
  * Optional freeze-then-unfreeze schedule: train only the I/O + KL stems
    for the first --freeze_backbone_steps, then unfreeze the entire backbone.
  * Reconstruction loss is computed in the NORMALISED feature space so all
    18 channels contribute on a comparable scale.

Usage:
    # 1) Precompute the dataset once (see precompute_feat18.py).
    # 2a) Single GPU:
    python train_finetune_feat18.py \
        --data_root /mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512 \
        --output_dir results/finetune_feat18_1k \
        --max_steps 50000
    # 2b) Multi-GPU (single node, N GPUs) via torchrun:
    torchrun --standalone --nproc_per_node=N train_finetune_feat18.py \
        --data_root ... --output_dir ... --max_steps 50000
    # 2c) Multi-node (example, 2 nodes × 8 GPUs):
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
             --master_addr=$MASTER_IP --master_port=12355 \
             train_finetune_feat18.py ...
    Notes for DDP:
      * --batch_size is PER-GPU (global batch = batch_size × world_size).
      * --num_workers is PER-GPU; keep it modest (e.g. 2–4) on many-GPU nodes.
      * Only rank 0 writes to output_dir (config.json, tb_logs, ckpts, meshes).

NOTE on augmentation:
    Only random integer translation is implemented here. 90° rotations / axis
    flips additionally require remapping the 6 edge_weights and 6 face_weights
    channels (their channel layout is direction-aware in the unique-cube
    convention used by corep_fast). That is left as a TODO; the dataset is
    already large enough (1k) for translation alone to be a useful regulariser.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import trimesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trellis2 import models
from trellis2.modules import sparse as sp

from train_overfit_feat18 import (
    build_models,
    feats_to_param,
    feature_to_mesh,
)


# ─────────────────────── flex_gemm freeze-backbone patch ────────────────────
#
# flex_gemm's submanifold_conv3d has a bug: when weight.requires_grad=False,
# the triton backward kernel returns None for grad_weight, but the wrapper
# `_sparse_submanifold_conv_backward` then unconditionally calls
# `grad_weight.reshape(...)` → AttributeError. This crashes any training that
# freezes backbone conv weights (e.g. our --freeze_backbone_steps warmup).
#
# Workaround: replace `SubMConv3dFunction.backward` with a version that
# temporarily flips weight.requires_grad=True across the internal kernel call
# so the reshape always succeeds, then drops the unwanted grad on return per
# the original semantics. Also fixes a secondary bug where the original
# backward would crash with bias=None (it does `not bias.requires_grad`).


def _patch_flex_gemm_frozen_weight_bug():
    try:
        from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
    except Exception as e:
        print(f"[patch] flex_gemm not importable, skipping patch: {e}")
        return

    if getattr(SubMConv3dFunction, "_trellis_freeze_patch", False):
        return

    def _patched_backward(ctx, grad_output, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        want_input = feats.requires_grad
        want_weight = weight.requires_grad
        want_bias = bias is not None and bias.requires_grad

        w_prev = weight.requires_grad
        if not w_prev:
            weight.requires_grad_(True)
        try:
            grad_input, grad_weight, grad_bias = (
                SubMConv3dFunction._sparse_submanifold_conv_backward(
                    grad_output, feats, neighbor_cache, weight, bias
                )
            )
        finally:
            if not w_prev:
                weight.requires_grad_(False)

        if not want_input:
            grad_input = None
        if not want_weight:
            grad_weight = None
        if not want_bias:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None

    SubMConv3dFunction.backward = staticmethod(_patched_backward)
    SubMConv3dFunction._trellis_freeze_patch = True
    print("[patch] applied flex_gemm SubMConv3dFunction.backward frozen-weight fix")


_patch_flex_gemm_frozen_weight_bug()


# ──────────────────────────── distributed helpers ────────────────────────────


def _init_dist():
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


def _unwrap(m):
    return m.module if isinstance(m, DDP) else m


def _wrap_ddp(model, local_rank):
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )


def _worker_init_fn(worker_id: int):
    """Independent RNG per (rank, worker) so augmentation is not duplicated."""
    rank = int(os.environ.get("RANK", 0))
    num_workers = int(os.environ.get("_FT_NUM_WORKERS", 1))
    seed = (rank * max(num_workers, 1) + worker_id) * 9973 + 17
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)


# ──────────────────────────── pretrained init ────────────────────────────────


@torch.no_grad()
def load_pretrained_into(
    target_enc,
    target_dec,
    enc_path: str,
    dec_path: str,
    warmstart_io: bool = True,
):
    """Copy TRELLIS.2 shape VAE weights into target encoder/decoder.

    The backbone parameters have identical shapes (same model_channels /
    num_blocks / latent_channels / pred_subdiv) and are copied as-is. The
    I/O linears differ:
        encoder.input_layer  : pretrained (C0, 6)   →  target (C0, 18)
        decoder.output_layer : pretrained (7,  Cend) →  target (18, Cend)

    If warmstart_io=True we initialise the new I/O linears from the
    pretrained vertex pathway:
        * encoder: pretrained vertex columns (cols 0..2) are placed into
          BOTH point1 (cols 0..2) and point2 (cols 3..5) of the target,
          scaled by 0.5 so their sum matches the pretrained signal when the
          two points coincide. The 12 weight columns (cols 6..17) start at
          zero, so initially they contribute nothing and the learned
          gradients are responsible for teaching them.
        * decoder: pretrained vertex output rows (rows 0..2) are copied
          into target point1 (rows 0..2). Point2 rows (3..5) and weight
          rows (6..17) start at zero, i.e. early outputs are exactly the
          pretrained vertex prediction in the first 3 channels and zero
          everywhere else.
    Otherwise both layers keep their xavier_uniform initialisation.
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

    miss_e, unex_e = target_enc.load_state_dict(enc_filtered, strict=False)
    miss_d, unex_d = target_dec.load_state_dict(dec_filtered, strict=False)

    only_io_missing_e = [k for k in miss_e if not k.startswith("input_layer")]
    only_io_missing_d = [k for k in miss_d if not k.startswith("output_layer")]
    print(f"[init] encoder missing (non-IO): {only_io_missing_e}")
    print(f"[init] encoder unexpected     : {list(unex_e)}")
    print(f"[init] decoder missing (non-IO): {only_io_missing_d}")
    print(f"[init] decoder unexpected     : {list(unex_d)}")

    if warmstart_io:
        pre_enc_w = pre_enc_sd["input_layer.weight"]   # (C0, 6)
        pre_enc_b = pre_enc_sd["input_layer.bias"]     # (C0,)
        pre_dec_w = pre_dec_sd["output_layer.weight"]  # (7, Cend)
        pre_dec_b = pre_dec_sd["output_layer.bias"]    # (7,)

        target_enc.input_layer.weight.zero_()
        target_enc.input_layer.weight[:, 0:3].copy_(0.5 * pre_enc_w[:, 0:3])
        target_enc.input_layer.weight[:, 3:6].copy_(0.5 * pre_enc_w[:, 0:3])
        target_enc.input_layer.bias.copy_(pre_enc_b)

        target_dec.output_layer.weight.zero_()
        target_dec.output_layer.bias.zero_()
        target_dec.output_layer.weight[0:3, :].copy_(pre_dec_w[0:3, :])
        target_dec.output_layer.bias[0:3].copy_(pre_dec_b[0:3])
        print("[init] warm-started input_layer / output_layer from pretrained vertex pathway")

    del pre_enc, pre_dec, pre_enc_sd, pre_dec_sd
    torch.cuda.empty_cache()


# ──────────────────────────── dataset / loader ───────────────────────────────


class Feat18Dataset(Dataset):
    """Loads precomputed .npz shards and applies random integer translation."""

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        max_translate: int = 16,
        augment: bool = True,
    ):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no .npz files in {data_dir}")
        self.resolution = resolution
        self.max_translate = max_translate
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        cube_indices = d["cube_indices"].astype(np.int32)   # (N, 3)
        feats = d["feats"].astype(np.float32)               # (N, 18)
        num_boundary = d["num_boundary"].astype(np.int32)
        sha = os.path.splitext(os.path.basename(self.files[idx]))[0]

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
    """Concatenate variable-length samples into a single SparseTensor payload."""
    coords_chunks = []
    feats_chunks = []
    sizes = []
    cube_indices_list = []
    num_boundary_list = []
    shas = []
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


# ──────────────────────────── normalisation ──────────────────────────────────


def load_stats(stats_path: str | None, device: torch.device):
    """Return (mean, std) tensors of shape (18,)."""
    if stats_path is None or not os.path.exists(stats_path):
        print(f"[stats] no stats file at {stats_path!r}, using identity defaults")
        mean = np.zeros(18, dtype=np.float32)
        mean[:6] = 0.5
        std = np.ones(18, dtype=np.float32)
    else:
        s = np.load(stats_path)
        mean = s["mean"].astype(np.float32)
        std = s["std"].astype(np.float32)
        # Hard-clamp to safe values just in case the precompute saved bad std.
        std = np.maximum(std, 1e-3)
    print(f"[stats] mean = {np.round(mean, 4).tolist()}")
    print(f"[stats] std  = {np.round(std, 4).tolist()}")
    return (
        torch.from_numpy(mean).to(device),
        torch.from_numpy(std).to(device),
    )


def normalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (feats - mean) / std


def denormalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return feats * std + mean


# ──────────────────────────────── train ──────────────────────────────────────


def _trainable(*models_):
    return [p for m in models_ for p in m.parameters() if p.requires_grad]


def _set_backbone_requires_grad(encoder, decoder, requires_grad: bool):
    """Freeze (or unfreeze) every parameter that is NOT input_layer/output_layer/
    to_latent/from_latent. Those four stems stay trainable in both phases."""
    enc_io_prefixes = ("input_layer.", "to_latent.")
    dec_io_prefixes = ("output_layer.", "from_latent.")
    for name, p in encoder.named_parameters():
        p.requires_grad_(True if any(name.startswith(pr) for pr in enc_io_prefixes) else requires_grad)
    for name, p in decoder.named_parameters():
        p.requires_grad_(True if any(name.startswith(pr) for pr in dec_io_prefixes) else requires_grad)


def train(args):
    rank, world_size, local_rank, is_dist = _init_dist()
    is_master = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    if is_dist:
        dist.barrier()

    def _log(msg: str):
        if is_master:
            print(msg)

    _log(f"[dist] rank={rank} world_size={world_size} local_rank={local_rank} device={device}")

    # ── stats ──
    stats_path = args.stats_path or os.path.join(args.data_root, "stats_rank0.npz")
    mean_t, std_t = load_stats(stats_path, device) if is_master else (None, None)
    if not is_master:
        # Other ranks: load silently to avoid duplicated prints.
        if stats_path is None or not os.path.exists(stats_path):
            mean_np = np.zeros(18, dtype=np.float32); mean_np[:6] = 0.5
            std_np = np.ones(18, dtype=np.float32)
        else:
            s = np.load(stats_path)
            mean_np = s["mean"].astype(np.float32)
            std_np = np.maximum(s["std"].astype(np.float32), 1e-3)
        mean_t = torch.from_numpy(mean_np).to(device)
        std_t = torch.from_numpy(std_np).to(device)

    # ── dataset ──
    data_dir = args.data_dir or os.path.join(args.data_root, "data")
    train_set = Feat18Dataset(
        data_dir,
        resolution=args.resolution,
        max_translate=args.max_translate,
        augment=True,
    )
    eval_set = Feat18Dataset(
        data_dir,
        resolution=args.resolution,
        max_translate=0,
        augment=False,
    )
    _log(f"[data] {len(train_set)} samples in {data_dir}")

    sampler = (
        DistributedSampler(train_set, shuffle=True, drop_last=True)
        if is_dist
        else None
    )
    # Expose num_workers to worker_init_fn via env.
    os.environ["_FT_NUM_WORKERS"] = str(max(args.num_workers, 1))
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )

    # ── model ──
    encoder, decoder = build_models(latent_channels=args.latent_channels, device=device)
    if args.from_pretrained:
        # Load pretrained weights on every rank (identical init → safe).
        # Only master prints; others silence to avoid log duplication.
        if is_master:
            load_pretrained_into(
                encoder, decoder,
                enc_path=args.enc_pretrained,
                dec_path=args.dec_pretrained,
                warmstart_io=args.warmstart_io,
            )
        else:
            import builtins as _b
            _orig_print = _b.print
            _b.print = lambda *a, **k: None
            try:
                load_pretrained_into(
                    encoder, decoder,
                    enc_path=args.enc_pretrained,
                    dec_path=args.dec_pretrained,
                    warmstart_io=args.warmstart_io,
                )
            finally:
                _b.print = _orig_print
    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    _log(f"[model] encoder: {n_params_enc / 1e6:.2f}M params")
    _log(f"[model] decoder: {n_params_dec / 1e6:.2f}M params")

    # ── freeze schedule (must happen BEFORE wrapping in DDP so frozen params
    # are excluded from the reducer) ──
    if args.freeze_backbone_steps > 0:
        _set_backbone_requires_grad(encoder, decoder, requires_grad=False)
        trainable = _trainable(encoder, decoder)
        _log(
            f"[optim] freezing backbone for {args.freeze_backbone_steps} steps "
            f"({sum(p.numel() for p in trainable) / 1e6:.2f}M trainable params)"
        )

    # ── wrap in DDP ──
    if is_dist:
        encoder = _wrap_ddp(encoder, local_rank)
        decoder = _wrap_ddp(decoder, local_rank)

    trainable = _trainable(encoder, decoder)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    writer = (
        SummaryWriter(os.path.join(args.output_dir, "tb_logs"))
        if is_master
        else None
    )
    encoder.train()
    decoder.train()

    step = 0
    unfrozen = args.freeze_backbone_steps == 0
    epoch = 0
    pbar = tqdm(
        total=args.max_steps,
        desc="ft",
        dynamic_ncols=True,
        disable=not is_master,
    )
    t0 = time.time()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.max_steps:
                break

            coords = batch["coords"].int().to(device, non_blocking=True)
            feats_raw = batch["feats"].float().to(device, non_blocking=True)
            feats = normalize(feats_raw, mean_t, std_t)

            x = sp.SparseTensor(feats=feats, coords=coords)
            z, mu, logvar = encoder(x, sample_posterior=True, return_raw=True)
            decoded = decoder(z)
            h, subs_gt, subs = decoded

            # Reconstruction in normalised space — every channel contributes
            # on a roughly unit scale.
            loss_recon = F.mse_loss(h.feats, x.feats)
            loss_kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)
            loss_subdiv = torch.tensor(0.0, device=device)
            for sub_gt, sub in zip(subs_gt, subs):
                loss_subdiv = loss_subdiv + F.binary_cross_entropy_with_logits(
                    sub.feats, sub_gt.float()
                )
            if len(subs) > 0:
                loss_subdiv = loss_subdiv / len(subs)

            loss = (
                loss_recon
                + args.lambda_kl * loss_kl
                + args.lambda_subdiv * loss_subdiv
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                args.grad_clip,
            )
            optimizer.step()

            step += 1

            # All-reduce losses for logging only (doesn't affect training).
            if is_dist:
                log_pack = torch.stack([
                    loss.detach(), loss_recon.detach(),
                    loss_kl.detach(), loss_subdiv.detach(),
                ])
                dist.all_reduce(log_pack, op=dist.ReduceOp.AVG)
                l_total, l_recon, l_kl, l_sub = log_pack.tolist()
            else:
                l_total = loss.item()
                l_recon = loss_recon.item()
                l_kl = loss_kl.item()
                l_sub = loss_subdiv.item()

            pbar.update(1)
            if is_master:
                pbar.set_postfix(
                    loss=f"{l_total:.4f}",
                    recon=f"{l_recon:.4f}",
                    kl=f"{l_kl:.2f}",
                    sub=f"{l_sub:.4f}",
                )
                writer.add_scalar("loss/total", l_total, step)
                writer.add_scalar("loss/recon", l_recon, step)
                writer.add_scalar("loss/kl", l_kl, step)
                writer.add_scalar("loss/subdiv", l_sub, step)
                writer.add_scalar("misc/voxels", float(x.feats.shape[0]), step)
                writer.add_scalar(
                    "misc/it_per_s", step / max(time.time() - t0, 1e-3), step
                )

            # ── unfreeze backbone once warmup is done ──
            if not unfrozen and step >= args.freeze_backbone_steps:
                enc_mod = _unwrap(encoder)
                dec_mod = _unwrap(decoder)
                _set_backbone_requires_grad(enc_mod, dec_mod, requires_grad=True)
                if is_dist:
                    # Rebuild DDP so the reducer picks up the newly-trainable
                    # parameters with find_unused_parameters=False.
                    encoder = _wrap_ddp(enc_mod, local_rank)
                    decoder = _wrap_ddp(dec_mod, local_rank)
                else:
                    encoder, decoder = enc_mod, dec_mod
                trainable = _trainable(encoder, decoder)
                optimizer = torch.optim.AdamW(
                    trainable, lr=args.lr, weight_decay=0.0
                )
                unfrozen = True
                if is_master:
                    tqdm.write(
                        f"[optim] unfroze backbone at step {step} "
                        f"({sum(p.numel() for p in trainable) / 1e6:.2f}M trainable params)"
                    )

            if (step % args.i_save == 0 or step == args.max_steps) and is_master:
                ckpt_path = os.path.join(
                    args.output_dir, f"ckpt_step{step:07d}.pt"
                )
                torch.save(
                    {
                        "step": step,
                        "encoder": _unwrap(encoder).state_dict(),
                        "decoder": _unwrap(decoder).state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                tqdm.write(f"  [save] {ckpt_path}")

            if step % args.i_sample == 0 or step == args.max_steps or step == 1:
                # Only rank 0 dumps meshes; other ranks wait at the barrier.
                if is_master:
                    sample_dir = os.path.join(
                        args.output_dir, f"meshes_step{step:07d}"
                    )
                    os.makedirs(sample_dir, exist_ok=True)
                    _dump_samples(
                        _unwrap(encoder),
                        _unwrap(decoder),
                        eval_set,
                        mean_t,
                        std_t,
                        args,
                        sample_dir,
                        device,
                    )
                if is_dist:
                    dist.barrier()
        epoch += 1

    if is_master:
        writer.close()
    pbar.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    _log("Training finished.")


@torch.no_grad()
def _dump_samples(encoder, decoder, dataset, mean_t, std_t, args, out_dir, device):
    encoder.eval()
    decoder.eval()
    n = min(args.n_dump, len(dataset))
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
        pred_norm = h.feats
        pred_raw = denormalize(pred_norm, mean_t, std_t).cpu().numpy()

        try:
            mesh = feature_to_mesh(
                pred_raw,
                ci,
                resolution=args.resolution,
                device=str(device),
            )
            if mesh is not None:
                mesh.export(os.path.join(out_dir, f"{sha}_pred.ply"))
            else:
                tqdm.write(f"  [mesh] {sha} returned empty placeholder")
        except Exception as e:
            tqdm.write(f"  [mesh] {sha} failed: {type(e).__name__}: {e}")
    encoder.train()
    decoder.train()


# ──────────────────────────────── CLI ─────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_root",
        default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512",
        help="Output of precompute_feat18.py (must contain data/*.npz and stats_rank0.npz).",
    )
    p.add_argument("--data_dir", default=None, help="Override <data_root>/data.")
    p.add_argument("--stats_path", default=None, help="Override <data_root>/stats_rank0.npz.")
    p.add_argument("--output_dir", default="results/finetune_feat18_1k")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--latent_channels", type=int, default=32)

    # ── pretrained init ──
    p.add_argument("--from_pretrained", action="store_true", default=True)
    p.add_argument(
        "--no_pretrained",
        action="store_false",
        dest="from_pretrained",
        help="Train from scratch (skip TRELLIS.2 weights).",
    )
    p.add_argument(
        "--enc_pretrained",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16",
    )
    p.add_argument(
        "--dec_pretrained",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    )
    p.add_argument("--warmstart_io", action="store_true", default=True)
    p.add_argument("--no_warmstart_io", action="store_false", dest="warmstart_io")

    # ── freeze schedule ──
    p.add_argument(
        "--freeze_backbone_steps",
        type=int,
        default=2000,
        help="Train only IO/latent stems for the first N steps, then unfreeze backbone. 0 = always train everything.",
    )

    # ── optimisation ──
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_translate", type=int, default=16)
    p.add_argument("--lambda_kl", type=float, default=1e-6)
    p.add_argument("--lambda_subdiv", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # ── logging ──
    p.add_argument("--i_save", type=int, default=2000)
    p.add_argument("--i_sample", type=int, default=2000)
    p.add_argument("--n_dump", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
