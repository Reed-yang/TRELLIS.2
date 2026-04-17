"""
Overfit the Shape SC-VAE on a single mesh (geometry only, no texture).

Usage:
    python train_overfit_shape.py --mesh path/to/mesh.ply --resolution 256
    python train_overfit_shape.py --mesh path/to/mesh.obj --resolution 256 --max_steps 5000

The script:
  1. Loads the mesh (PLY/OBJ) and normalises it to [-0.5, 0.5].
  2. Creates a mesh_dump pickle and dual-grid .vxz in a temp dataset dir.
  3. Synthesises a minimal metadata.csv so FlexiDualGridDataset can load it.
  4. Builds the Shape VAE encoder/decoder and launches ShapeVaeTrainer
     with settings tuned for single-sample overfitting.
"""

import os
import sys
import json
import pickle
import argparse
import time
import shutil
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import trimesh
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict

from trellis2 import models, datasets, trainers


# ────────────────────────────── data preparation ──────────────────────────────

def prepare_single_mesh_dataset(mesh_path: str, resolution: int, output_dir: str) -> str:
    """
    Convert one mesh file into the directory layout expected by
    FlexiDualGridDataset:
        <output_dir>/mesh_dumps/<sha256>.pickle
        <output_dir>/dual_grid_<resolution>/<sha256>.vxz
        <output_dir>/dual_grid_<resolution>/metadata.csv
        <output_dir>/mesh_dumps/metadata.csv
        <output_dir>/metadata.csv

    Returns the sha256 identifier used.
    """
    sha256 = "overfit_mesh"

    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64))

    # normalise to [-0.5, 0.5]
    vmin = vertices.min(dim=0)[0]
    vmax = vertices.max(dim=0)[0]
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    vertices = (vertices - center) * scale

    # ── mesh dump ──
    mesh_dump_dir = os.path.join(output_dir, "mesh_dumps")
    os.makedirs(mesh_dump_dir, exist_ok=True)
    dump = {"objects": [{"vertices": vertices.numpy(), "faces": faces.numpy()}]}
    with open(os.path.join(mesh_dump_dir, f"{sha256}.pickle"), "wb") as f:
        pickle.dump(dump, f)
    pd.DataFrame([{"sha256": sha256, "mesh_dumped": True}]).to_csv(
        os.path.join(mesh_dump_dir, "metadata.csv"), index=False
    )

    # ── dual grid ──
    dg_dir = os.path.join(output_dir, f"dual_grid_{resolution}")
    os.makedirs(dg_dir, exist_ok=True)
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    dual_vertices_local = dual_vertices * resolution - voxel_indices
    dual_vertices_local = torch.clamp(dual_vertices_local, 0, 1)
    dual_vertices_u8 = (dual_vertices_local * 255).to(torch.uint8)
    intersected_u8 = (
        intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]
    ).to(torch.uint8)

    o_voxel.io.write_vxz(
        os.path.join(dg_dir, f"{sha256}.vxz"),
        voxel_indices,
        {"vertices": dual_vertices_u8, "intersected": intersected_u8},
    )
    num_voxels = len(dual_vertices_u8)
    pd.DataFrame(
        [{"sha256": sha256, "dual_grid_converted": True, "dual_grid_size": num_voxels}]
    ).to_csv(os.path.join(dg_dir, "metadata.csv"), index=False)

    # ── top-level metadata ──
    pd.DataFrame(
        [
            {
                "sha256": sha256,
                "mesh_dumped": True,
                "dual_grid_converted": True,
                "dual_grid_size": num_voxels,
                "aesthetic_score": 10.0,
                "num_faces": int(faces.shape[0]),
            }
        ]
    ).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"[data] mesh normalised – {vertices.shape[0]} verts, {faces.shape[0]} faces")
    print(f"[data] dual grid: {num_voxels} active voxels at resolution {resolution}")
    return sha256


# ────────────────────────────── config builder ────────────────────────────────

def build_overfit_config(resolution: int, max_steps: int, lr: float) -> dict:
    """Return a config dict comparable to shape_vae_next_dc_f16c32_fp16.json
    but tuned for single-mesh overfitting."""
    return {
        "models": {
            "encoder": {
                "name": "FlexiDualGridVaeEncoder",
                "args": {
                    "model_channels": [64, 128, 256, 512, 1024],
                    "latent_channels": 32,
                    "num_blocks": [0, 4, 8, 16, 4],
                    "block_type": [
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                    ],
                    "down_block_type": [
                        "SparseResBlockS2C3d",
                        "SparseResBlockS2C3d",
                        "SparseResBlockS2C3d",
                        "SparseResBlockS2C3d",
                    ],
                    "block_args": [
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                    ],
                    "use_fp16": False,
                },
            },
            "decoder": {
                "name": "FlexiDualGridVaeDecoder",
                "args": {
                    "resolution": resolution,
                    "model_channels": [1024, 512, 256, 128, 64],
                    "latent_channels": 32,
                    "num_blocks": [4, 16, 8, 4, 0],
                    "block_type": [
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                        "SparseConvNeXtBlock3d",
                    ],
                    "up_block_type": [
                        "SparseResBlockC2S3d",
                        "SparseResBlockC2S3d",
                        "SparseResBlockC2S3d",
                        "SparseResBlockC2S3d",
                    ],
                    "block_args": [
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                        {"use_checkpoint": False},
                    ],
                    "use_fp16": False,
                },
            },
        },
        "dataset": {
            "name": "FlexiDualGridDataset",
            "args": {
                "resolution": resolution,
                "max_active_voxels": 10000000,
                "max_num_faces": 10000000,
                "min_aesthetic_score": 0.0,
            },
        },
        "trainer": {
            "name": "ShapeVaeTrainer",
            "args": {
                "max_steps": max_steps,
                "batch_size_per_gpu": 1,
                "batch_split": 1,
                "optimizer": {
                    "name": "AdamW",
                    "args": {"lr": lr, "weight_decay": 0.0},
                },
                "ema_rate": [0.9999],
                "fp16_mode": None,
                "grad_clip": 1.0,
                "i_print": 10,
                "i_log": 10,
                "i_sample": 500,
                "i_save": 500,
                "lambda_subdiv": 0.1,
                "lambda_intersected": 0.1,
                "lambda_vertice": 1e-2,
                # "lambda_mask": 1,
                # "lambda_depth": 10,
                # "lambda_normal": 1,
                "lambda_mask": 0,
                "lambda_depth": 0,
                "lambda_normal": 0,
                "lambda_kl": 1e-6,
                # "lambda_ssim": 0.2,
                # "lambda_lpips": 0.2,
                "lambda_ssim": 0,
                "lambda_lpips": 0,
                # "camera_randomization_config": {"radius_range": [2, 100]},
                "camera_randomization_config": {"radius_range": [1, 1]},
            },
        },
    }


# ────────────────────────────── training entry ────────────────────────────────

def find_ckpt(cfg):
    import glob as _glob
    cfg["load_ckpt"] = None
    if cfg.load_dir != "":
        if cfg.ckpt == "latest":
            files = _glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_*.pt"))
            if files:
                cfg.load_ckpt = max(
                    int(os.path.basename(f).split("step")[-1].split(".")[0])
                    for f in files
                )
        elif cfg.ckpt == "none":
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    import random
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def _save_mesh_ply(mesh, path):
    """Save a trellis2 Mesh object as an ASCII PLY file."""
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().int().numpy()
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


@torch.no_grad()
def _snapshot_with_mesh(trainer, suffix=None, batch_size=None):
    """
    Extended snapshot: runs the normal image snapshot, then additionally
    saves GT and reconstructed meshes as PLY, and o-voxel .vxz files.
    """
    import copy
    from functools import partial
    from contextlib import nullcontext
    from torch.utils.data import DataLoader
    from trellis2.utils.data_utils import recursive_to_device
    from trellis2.modules import sparse as sp

    if batch_size is None:
        batch_size = trainer.snapshot_batch_size

    # Run the original snapshot for rendered images
    trainer.snapshot(suffix=suffix, batch_size=batch_size)

    if not trainer.is_master:
        return

    if suffix is None:
        suffix = f'step{trainer.step:07d}'

    sample_dir = os.path.join(trainer.output_dir, 'samples', suffix)
    os.makedirs(sample_dir, exist_ok=True)

    # Re-run inference to capture mesh objects
    dataloader = DataLoader(
        copy.deepcopy(trainer.dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=trainer.dataset.collate_fn if hasattr(trainer.dataset, 'collate_fn') else None,
    )

    num_samples = batch_size
    gt_meshes = []
    rec_meshes = []
    rec2_meshes = []

    dec = trainer.models['decoder']
    enc = trainer.models['encoder']

    enc.eval()
    amp_context = partial(torch.autocast, device_type='cuda', dtype=trainer.mix_precision_dtype) \
        if trainer.mix_precision_mode == 'amp' else nullcontext

    with amp_context():
        data = next(iter(dataloader))
        args = {k: v[:num_samples] for k, v in data.items()}
        args = recursive_to_device(args, trainer.device)

        z = enc(args['vertices'], args['intersected'])

        # rec: decoder in train mode (uses GT intersected)
        dec.train()
        y_train = dec(z, args['intersected'])
        rec_meshes = y_train[0]

        # rec2: decoder in eval mode (fully predicted)
        z.clear_spatial_cache()
        dec.eval()
        y_eval = dec(z)
        rec2_meshes = y_eval if isinstance(y_eval, list) else y_eval[0]

        gt_meshes = args['mesh']

        # Save o-voxel: re-run decoder without the mesh conversion wrapper
        # We get raw h from the parent class forward
        raw_dec = dec.module if hasattr(dec, 'module') else dec
        raw_parent_forward = type(raw_dec).__mro__[1].forward  # SparseUnetVaeDecoder.forward
        raw_h = raw_parent_forward(raw_dec, z)
        if isinstance(raw_h, tuple):
            raw_h = raw_h[0]

        # Save per-sample o-voxel
        import torch.nn.functional as F
        batch_ids = raw_h.coords[:, 0]
        unique_ids = batch_ids.unique()
        voxel_margin = getattr(raw_dec, 'voxel_margin', 0.5)
        for bid in unique_ids:
            mask = batch_ids == bid
            coords = raw_h.coords[mask][:, 1:].cpu().int()
            feats = raw_h.feats[mask]
            vert_feats = ((1 + 2 * voxel_margin) * torch.sigmoid(feats[..., 0:3]) - voxel_margin)
            dual_verts_u8 = (torch.clamp(vert_feats.detach().cpu(), 0, 1) * 255).to(torch.uint8)
            inter_bits = (feats[..., 3:6].detach().cpu() > 0).to(torch.uint8)
            inter_packed = (inter_bits[:, 0:1] + 2 * inter_bits[:, 1:2] + 4 * inter_bits[:, 2:3])
            vxz_path = os.path.join(sample_dir, f"rec2_b{bid.item()}.vxz")
            o_voxel.io.write_vxz(vxz_path, coords, {"vertices": dual_verts_u8, "intersected": inter_packed})

    enc.train()
    dec.train()

    # Save meshes as PLY
    for i, m in enumerate(gt_meshes):
        _save_mesh_ply(m, os.path.join(sample_dir, f"gt_b{i}.ply"))
    for i, m in enumerate(rec_meshes):
        _save_mesh_ply(m, os.path.join(sample_dir, f"rec_b{i}.ply"))
    for i, m in enumerate(rec2_meshes):
        _save_mesh_ply(m, os.path.join(sample_dir, f"rec2_b{i}.ply"))

    print(f"  Saved {len(gt_meshes)} GT + {len(rec_meshes)} rec + {len(rec2_meshes)} rec2 meshes to {sample_dir}")


def _run_with_tqdm(trainer):
    """
    Replace BasicTrainer.run() with a tqdm-wrapped version so the terminal
    shows a live progress bar with loss information, and saves meshes + o-voxel
    at each snapshot step.
    """
    import torch.distributed as dist

    if trainer.is_master:
        print('\nStarting training...')
        trainer.snapshot_dataset(batch_size=trainer.snapshot_batch_size)

    _snapshot_with_mesh(trainer,
                        suffix='init' if trainer.step == 0 else f'resume_step{trainer.step:07d}',
                        batch_size=trainer.snapshot_batch_size)

    pbar = tqdm(
        initial=trainer.step,
        total=trainer.max_steps,
        desc="training",
        dynamic_ncols=True,
        disable=not trainer.is_master,
    )

    time_elapsed = 0.0
    while trainer.step < trainer.max_steps:
        t0 = time.time()

        data_list = trainer.load_data()
        step_log = trainer.run_step(data_list)

        t1 = time.time()
        time_elapsed += t1 - t0

        trainer.step += 1
        pbar.update(1)

        # Update tqdm postfix with the main loss
        if step_log is not None and trainer.is_master:
            loss_val = step_log.get("loss", None)
            postfix = {}
            if loss_val is not None:
                postfix["loss"] = f"{loss_val['loss']:.4f}"
            pbar.set_postfix(postfix)

        if trainer.parallel_mode == 'ddp' and trainer.world_size > 1 \
                and trainer.i_ddpcheck is not None \
                and trainer.step % trainer.i_ddpcheck == 0:
            trainer.check_ddp()

        if trainer.step % trainer.i_sample == 0:
            _snapshot_with_mesh(trainer)

        if trainer.is_master:
            trainer.log.append((trainer.step, {}))
            trainer.log[-1][1]['time'] = {
                'step': t1 - t0,
                'elapsed': time_elapsed,
            }
            if step_log is not None:
                trainer.log[-1][1].update(step_log)
            if trainer.mix_precision_dtype == torch.float16:
                if trainer.mix_precision_mode == 'amp':
                    trainer.log[-1][1]['scale'] = trainer.scaler.get_scale()
                elif trainer.mix_precision_mode == 'inflat_all':
                    trainer.log[-1][1]['log_scale'] = trainer.log_scale
            if trainer.step % trainer.i_log == 0:
                trainer.save_logs()
            if trainer.step % trainer.i_save == 0:
                trainer.save()

        trainer.check_abort()

    pbar.close()
    _snapshot_with_mesh(trainer, suffix='final', batch_size=trainer.snapshot_batch_size)
    if trainer.world_size > 1:
        dist.barrier()
    if trainer.is_master:
        trainer.writer.close()
        print('Training finished.')


def main_train(local_rank, cfg):
    from trellis2.utils.dist_utils import setup_dist

    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    setup_rng(rank)

    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    print(f"[train] dataset: {dataset}")

    model_dict = {
        name: getattr(models, m.name)(**m.args).cuda()
        for name, m in cfg.models.items()
    }

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt,
    )
    _run_with_tqdm(trainer)


# ──────────────────────────────────── CLI ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Overfit Shape SC-VAE on a single mesh (geometry only)"
    )
    p.add_argument("--mesh", type=str, required=True, help="Path to input mesh (PLY / OBJ)")
    p.add_argument("--resolution", type=int, default=256, help="Voxel grid resolution (default 256)")
    p.add_argument("--output_dir", type=str, default="results/overfit_shape", help="Output directory")
    p.add_argument("--max_steps", type=int, default=1000, help="Total training steps")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--ckpt", type=str, default="latest", help="Checkpoint to resume ('latest', 'none', or step number)")
    p.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── 1. prepare dataset ──
    data_dir = os.path.join(args.output_dir, "dataset")
    prepare_single_mesh_dataset(args.mesh, args.resolution, data_dir)

    # ── 2. build config ──
    config = build_overfit_config(args.resolution, args.max_steps, args.lr)

    # build the data_dir JSON string that FlexiDualGridDataset expects
    data_dir_json = json.dumps(
        {
            "overfit": {
                "base": data_dir,
                "mesh_dump": os.path.join(data_dir, "mesh_dumps"),
                "dual_grid": os.path.join(data_dir, f"dual_grid_{args.resolution}"),
            }
        }
    )

    cfg = edict()
    cfg.update(config)
    cfg.data_dir = data_dir_json
    cfg.output_dir = args.output_dir
    cfg.load_dir = args.output_dir
    cfg.ckpt = args.ckpt
    cfg.num_nodes = 1
    cfg.node_rank = 0
    cfg.num_gpus = args.num_gpus
    cfg.master_addr = "localhost"
    cfg.master_port = "12355"
    cfg.tryrun = False
    cfg.profile = False

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("\n[config]")
    print(json.dumps(cfg.__dict__, indent=4, default=str))

    # ── 3. train ──
    cfg = find_ckpt(cfg)
    if cfg.num_gpus > 1:
        mp.spawn(main_train, args=(cfg,), nprocs=cfg.num_gpus, join=True)
    else:
        main_train(0, cfg)
