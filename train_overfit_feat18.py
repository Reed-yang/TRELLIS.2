"""
Overfit a Sparse VAE on a single (n, 18) feature sample.

The VAE uses SparseUnetVaeEncoder / SparseUnetVaeDecoder to compress
18-channel per-voxel features placed at 3D integer coordinates.
No mesh is involved; no render losses (mask, depth, normal) are used.
Loss = feature reconstruction (MSE) + KL divergence + subdivision BCE.

Usage:
    python train_overfit_feat18.py
    python train_overfit_feat18.py --max_steps 5000 --lr 3e-4
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import trimesh
from trellis2.modules import sparse as sp
from trellis2.models.sc_vaes.sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
    SparseConvNeXtBlock3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)

from corep_fast.pipeline import mesh_to_param, param_to_mesh, CorepParam

# ──────────────────────────────── data ────────────────────────────────────────

def param_to_feats(param: CorepParam) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CorepParam to (cube_indices, feats_18) for the VAE.

    The 18-dim feature per voxel is:
        [point1_xyz(3), point2_xyz(3), edge_weights(6), face_weights(6)]
    For cubes with >= 2 points, the two points that are farthest apart
    (in Euclidean distance) are selected. Cubes with only 1 component
    point get zeros for the second point. Cubes with 0 points get zeros
    for both.

    Points are stored in LOCAL cube coordinates (expected in [0, 1]):
        local = point_global * resolution - cube_index
    where point_global lives in the normalized (0, 1)^3 space.
    """
    N = param.cube_indices.shape[0]
    R = float(param.resolution)
    offsets = param.point_offsets
    n_pts = offsets[1:] - offsets[:-1]
    cube_idx_f = param.cube_indices.astype(np.float32)

    first_2 = np.zeros((N, 6), dtype=np.float32)
    all_local_parts: list[np.ndarray] = []

    # Single-point cubes: store that point as p1.
    mask1 = n_pts == 1
    if mask1.any():
        idx1 = offsets[:-1][mask1]
        p1_only_local = param.point_values[idx1] * R - cube_idx_f[mask1]
        first_2[mask1, :3] = p1_only_local
        all_local_parts.append(p1_only_local)

    # Two-point cubes: both points are trivially the farthest pair.
    mask2 = n_pts == 2
    if mask2.any():
        idx2a = offsets[:-1][mask2]
        p1_local_2 = param.point_values[idx2a] * R - cube_idx_f[mask2]
        p2_local_2 = param.point_values[idx2a + 1] * R - cube_idx_f[mask2]
        first_2[mask2, :3] = p1_local_2
        first_2[mask2, 3:6] = p2_local_2
        all_local_parts.append(p1_local_2)
        all_local_parts.append(p2_local_2)

    # Cubes with >= 3 points: pick the pair with max pairwise distance.
    mask3_idx = np.where(n_pts >= 3)[0]
    for i in mask3_idx:
        start = int(offsets[i])
        n = int(n_pts[i])
        pts_local = param.point_values[start:start + n] * R - cube_idx_f[i]
        diff = pts_local[:, None, :] - pts_local[None, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        a, b = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
        first_2[i, :3] = pts_local[a]
        first_2[i, 3:6] = pts_local[b]
        all_local_parts.append(pts_local)

    # Outlier check: local coordinates should lie in [0, 1].
    tol = 1e-3
    all_local = all_local_parts
    if all_local:
        pts = np.concatenate(all_local, axis=0)
        out_mask = (pts < -tol) | (pts > 1.0 + tol)
        n_out = int(out_mask.any(axis=1).sum())
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        print(
            f"[param_to_feats] local-coord range: "
            f"x=[{lo[0]:.4f},{hi[0]:.4f}] "
            f"y=[{lo[1]:.4f},{hi[1]:.4f}] "
            f"z=[{lo[2]:.4f},{hi[2]:.4f}] "
            f"(total pts={len(pts)})"
        )
        if n_out > 0:
            max_dev = float(np.max(np.abs(pts - np.clip(pts, 0.0, 1.0))))
            print(
                f"[param_to_feats] WARNING: {n_out}/{len(pts)} points fall "
                f"outside [0, 1] (tol={tol}), max deviation = {max_dev:.4e}"
            )

    feats = np.concatenate([
        first_2,
        param.edge_weights.astype(np.float32),
        param.face_weights.astype(np.float32),
    ], axis=1)
    return param.cube_indices, feats


def feats_to_param(
    feats_np: np.ndarray,
    cube_indices: np.ndarray,
    resolution: int,
    num_boundary: np.ndarray | None = None,
) -> CorepParam:
    """Convert (feats_18, cube_indices) back to a CorepParam.

    Reconstructs the CSR point_values/point_offsets from the first 6 dims.
    The first 6 dims encode up to two LOCAL cube-space points; they are
    converted back to global (0, 1)^3 coordinates via:
        point_global = (cube_index + local) / resolution
    If a cube needs more than 2 points during s7 matching, the last point
    is duplicated (handled by param_to_mesh via the CSR structure here
    providing up to 2; s7 Hungarian matching only uses what's available).
    """
    N = len(cube_indices)
    R = float(resolution)
    first_2 = feats_np[:, :6]
    edge_weights = np.round(feats_np[:, 6:12]).astype(np.int32)
    face_weights = np.round(feats_np[:, 12:18]).astype(np.int32)

    cube_idx_f = cube_indices.astype(np.float32)

    point_list = []
    offsets = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        p1_local = first_2[i, :3]
        p2_local = first_2[i, 3:6]
        has_p1 = not np.allclose(p1_local, 0.0, atol=1e-7)
        has_p2 = not np.allclose(p2_local, 0.0, atol=1e-7)
        ci = cube_idx_f[i]
        if has_p1 and has_p2:
            point_list.append((ci + p1_local) / R)
            point_list.append((ci + p2_local) / R)
            offsets[i + 1] = offsets[i] + 2
        elif has_p1:
            point_list.append((ci + p1_local) / R)
            offsets[i + 1] = offsets[i] + 1
        else:
            offsets[i + 1] = offsets[i]

    if point_list:
        point_values = np.stack(point_list, axis=0).astype(np.float32)
    else:
        point_values = np.zeros((0, 3), dtype=np.float32)

    if num_boundary is None:
        num_boundary = np.zeros(N, dtype=np.int32)

    return CorepParam(
        cube_indices=cube_indices.astype(np.int32),
        edge_weights=edge_weights,
        face_weights=face_weights,
        point_values=point_values,
        point_offsets=offsets,
        num_boundary=num_boundary,
        resolution=resolution,
    )


def get_single_data(mesh_path: str, resolution: int = 256, device: str = "cuda"):
    """Encode a mesh file into (cube_indices, feats_18) via corep_fast."""
    param = mesh_to_param(mesh_path, resolution, torch.device(device))
    return param_to_feats(param)


def build_sparse_tensor(coords_np, feats_np, device="cuda"):
    """
    Wrap numpy arrays into a SparseTensor with a batch-index first column.
    """
    coords = torch.from_numpy(coords_np).int()
    feats = torch.from_numpy(feats_np).float()
    batch_idx = torch.zeros(len(coords), 1, dtype=torch.int32)
    coords = torch.cat([batch_idx, coords], dim=1)
    return sp.SparseTensor(feats=feats.to(device), coords=coords.to(device))


def feature_to_mesh(feats_np, cube_indices, resolution, device="cuda"):
    """Reconstruct a trimesh from VAE 18-dim features via corep_fast."""
    param = feats_to_param(feats_np, cube_indices, resolution)
    vertices, faces = param_to_mesh(
        param,
        device=torch.device(device),
        merge_decimals=5,
    )
    v_np = vertices.cpu().numpy()
    f_np = faces.cpu().numpy()
    if f_np.shape[0] == 0:
        return None
    return trimesh.Trimesh(vertices=v_np, faces=f_np)


# ──────────────────────────────── model ───────────────────────────────────────

def build_models(latent_channels=32, device="cuda"):
    """
    Build encoder and decoder with the same U-Net backbone as the shape VAE
    but with 18-channel input/output and no mesh head.
    """
    encoder = SparseUnetVaeEncoder(
        in_channels=18,
        model_channels=[64, 128, 256, 512, 1024],
        latent_channels=latent_channels,
        num_blocks=[0, 4, 8, 16, 4],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        down_block_type=["SparseResBlockS2C3d"] * 4,
        block_args=[{"use_checkpoint": False}] * 5,
        use_fp16=False,
    ).to(device)

    decoder = SparseUnetVaeDecoder(
        out_channels=18,
        model_channels=[1024, 512, 256, 128, 64],
        latent_channels=latent_channels,
        num_blocks=[4, 16, 8, 4, 0],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        up_block_type=["SparseResBlockC2S3d"] * 4,
        block_args=[{"use_checkpoint": False}] * 5,
        use_fp16=False,
        pred_subdiv=True,
    ).to(device)

    return encoder, decoder


# ──────────────────────────── training loop ───────────────────────────────────

@torch.no_grad()
def evaluate(encoder, decoder, coords_np, feats_np, device="cuda"):
    """
    Run encoder → decoder in eval mode (decoder predicts subdivision itself).
    Returns per-feature MSE and the predicted feature tensor.
    """
    encoder.eval()
    decoder.eval()
    x = build_sparse_tensor(coords_np, feats_np, device)
    z = encoder(x, sample_posterior=False)
    decoded = decoder(z)
    if isinstance(decoded, tuple):
        h = decoded[0]
    else:
        h = decoded
    mse = F.mse_loss(h.feats, x.feats).item()
    encoder.train()
    decoder.train()
    return mse, h.feats.detach().cpu().numpy()


def train(args):
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    coords_np, feats_np = get_single_data(args.mesh_path, args.resolution, device)
    print(f"[data] {len(coords_np)} voxels, {feats_np.shape[1]} feature channels")

    encoder, decoder = build_models(latent_channels=args.latent_channels, device=device)
    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[model] encoder: {n_params_enc / 1e6:.2f}M params")
    print(f"[model] decoder: {n_params_dec / 1e6:.2f}M params")

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.0)

    writer = SummaryWriter(os.path.join(args.output_dir, "tb_logs"))

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # ── initial eval ──
    init_mse, _ = evaluate(encoder, decoder, coords_np, feats_np, device)
    print(f"[eval] init  MSE = {init_mse:.6f}")

    encoder.train()
    decoder.train()

    pbar = tqdm(range(1, args.max_steps + 1), desc="training", dynamic_ncols=True)
    for step in pbar:
        x = build_sparse_tensor(coords_np, feats_np, device)

        z, mean, logvar = encoder(x, sample_posterior=True, return_raw=True)
        decoded = decoder(z)
        h, subs_gt, subs = decoded

        loss_recon = F.mse_loss(h.feats, x.feats)

        loss_kl = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)

        loss_subdiv = torch.tensor(0.0, device=device)
        for sub_gt, sub in zip(subs_gt, subs):
            loss_subdiv = loss_subdiv + F.binary_cross_entropy_with_logits(
                sub.feats, sub_gt.float()
            )
        if len(subs) > 0:
            loss_subdiv = loss_subdiv / len(subs)

        loss = loss_recon + args.lambda_kl * loss_kl + args.lambda_subdiv * loss_subdiv

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{loss_recon.item():.6f}",
            kl=f"{loss_kl.item():.2f}",
        )

        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("loss/recon", loss_recon.item(), step)
        writer.add_scalar("loss/kl", loss_kl.item(), step)
        writer.add_scalar("loss/subdiv", loss_subdiv.item(), step)

        if step % args.i_eval == 0 or step == args.max_steps:
            eval_mse, _ = evaluate(encoder, decoder, coords_np, feats_np, device)
            writer.add_scalar("eval/mse", eval_mse, step)
            tqdm.write(f"  [eval] step {step:>6d}  MSE = {eval_mse:.6f}")

        if step % args.i_save == 0 or step == args.max_steps:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_step{step:07d}.pt")
            torch.save(
                {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            tqdm.write(f"  [save] {ckpt_path}")

        if step % 500 == 0:
            mesh_dir = os.path.join(args.output_dir, f"meshes_step{step:07d}")
            os.makedirs(mesh_dir, exist_ok=True)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                x = build_sparse_tensor(coords_np, feats_np, device)
                for sample_idx in range(2):
                    z = encoder(x, sample_posterior=True)
                    decoded = decoder(z)
                    h = decoded[0] if isinstance(decoded, tuple) else decoded
                    pred_feats = h.feats.detach().cpu().numpy()
                    mesh = feature_to_mesh(pred_feats, coords_np, resolution=args.resolution)
                    if mesh is not None:
                        mesh_path = os.path.join(mesh_dir, f"sample_{sample_idx}.ply")
                        mesh.export(mesh_path)
                        tqdm.write(f"  [mesh] saved {mesh_path}")
                    else:
                        tqdm.write(f"  [mesh] sample_{sample_idx} returned None (placeholder)")
            encoder.train()
            decoder.train()

    writer.close()
    print("Training finished.")


# ──────────────────────────────── CLI ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Overfit Sparse VAE on a single (n,18) feature data"
    )
    p.add_argument("--mesh_path", type=str, default="tmp/test_mesh/banana_plant_with_pot.glb")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="results/overfit_feat18")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent_channels", type=int, default=32)
    p.add_argument("--lambda_kl", type=float, default=1e-6)
    p.add_argument("--lambda_subdiv", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--i_eval", type=int, default=200)
    p.add_argument("--i_save", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    coords_np, feats_np = get_single_data(args.mesh_path, args.resolution, "cuda")
    mesh = feature_to_mesh(feats_np, coords_np, resolution=args.resolution, device="cuda")
    mesh.export(os.path.join(args.output_dir, "mesh.ply"))
    train(args)
