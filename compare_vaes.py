"""
Compare the pretrained TRELLIS.2 shape VAE against the fine-tuned feat18 VAE
on a single input mesh, at the mesh level.

For an input mesh, we produce three meshes in tmp/test_vae/<name>/:
    * gt.ply         -- GT after bbox-normalization to [-0.5, 0.5]^3
    * recon_raw.ply  -- Pretrained FlexiDualGrid shape VAE reconstruction
    * recon_ft.ply   -- Fine-tuned feat18 (CoReP) VAE reconstruction
and write metrics.json / metrics.txt comparing each reconstruction to GT.

Metrics (computed after re-normalizing every mesh to the [-0.5, 0.5] unit
cube so scale/position don't dominate):
    * Chamfer distance (bidirectional, mean of squared L2)
    * F-score at several thresholds
    * Normal consistency (mean |cos| of matched normals)
    * Rendered normal-map PSNR / SSIM (skippable with --skip_render)

Usage:
    python compare_vaes.py \
        --mesh_path tmp/test_mesh/banana_plant_with_pot.glb \
        --ft_ckpt  results/finetune_feat18_1k_ddp/ckpt_step0004000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import trimesh

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Default test case: Objaverse-XL Sketchfab sample whose precomputed feat18
# sha256 is 000060a495...cca4277. File stem is a meaningless uuid, so we
# default the output sub-folder to the sha instead of the basename when the
# default mesh is used.
DEFAULT_MESH_PATH = (
    "/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/"
    "raw/hf-objaverse-v1/glbs/000-113/7aca8c05583c48b8a3bfae6d043e331b.glb"
)
DEFAULT_OUT_NAME_FOR_DEFAULT_MESH = (
    "000060a495b381230860ca7315a1b585fabc651cf0833b72b6f481771cca4277"
)

from trellis2 import models
from trellis2.modules import sparse as sp
from scripts.eval.eval_metrics import (
    chamfer_distance,
    compute_rendering_metrics,
    f_score_multi,
    normal_consistency,
    render_normal_maps,
    sample_points_and_normals,
)
from scripts.eval.gap_measurement import (
    load_and_normalize_mesh,
    trimesh_to_trellis_mesh,
)

from train_overfit_feat18 import (
    build_models,
    feats_to_param,
    param_to_feats,
)
from corep_fast.pipeline import mesh_to_param, param_to_mesh


# ────────────────────── Path A: pretrained shape VAE ─────────────────────────


def _load_raw_vae(enc_path: str, dec_path: str, resolution: int, device: torch.device):
    enc = models.from_pretrained(enc_path).eval().to(device)
    dec = models.from_pretrained(dec_path).eval().to(device)
    if hasattr(dec, "set_resolution"):
        dec.set_resolution(resolution)
    return enc, dec


@torch.no_grad()
def _reconstruct_raw_vae(
    tm_mesh: trimesh.Trimesh,
    encoder,
    decoder,
    resolution: int,
    device: torch.device,
) -> trimesh.Trimesh | None:
    """Run the pretrained FlexiDualGrid shape VAE on a bbox-normalized mesh."""
    import o_voxel

    vertices = torch.from_numpy(tm_mesh.vertices.copy()).float()
    faces = torch.from_numpy(tm_mesh.faces.copy()).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )
    if voxel_indices.shape[0] == 0:
        print("[raw_vae] empty voxelization, skipping")
        return None

    dv_local = torch.clamp(dual_vertices * resolution - voxel_indices.float(), 0, 1)
    coords = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices], dim=-1
    )
    vertices_st = sp.SparseTensor(feats=dv_local, coords=coords).to(device)
    intersected_st = vertices_st.replace(intersected.float().to(device))

    z = encoder(vertices_st, intersected_st, sample_posterior=False)
    out = decoder(z)
    mesh_list = out[0] if isinstance(out, tuple) else out
    mesh_trellis = mesh_list[0] if isinstance(mesh_list, list) else mesh_list

    V = mesh_trellis.vertices.detach().cpu().numpy()
    F = mesh_trellis.faces.detach().cpu().numpy()
    if F.shape[0] == 0:
        return None
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


# ─────────────────────── Path B: fine-tuned feat18 VAE ───────────────────────


def _load_stats(stats_path: str, device: torch.device):
    s = np.load(stats_path)
    mean = s["mean"].astype(np.float32)
    std = np.maximum(s["std"].astype(np.float32), 1e-3)
    return (
        torch.from_numpy(mean).to(device),
        torch.from_numpy(std).to(device),
    )


def _load_ft_vae(
    ckpt_path: str, latent_channels: int, device: torch.device
):
    encoder, decoder = build_models(
        latent_channels=latent_channels, device=str(device)
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    miss_e, unex_e = encoder.load_state_dict(ckpt["encoder"], strict=True)
    miss_d, unex_d = decoder.load_state_dict(ckpt["decoder"], strict=True)
    print(
        f"[ft_vae] loaded step={ckpt.get('step', '?')} "
        f"(encoder missing={len(miss_e)} unexpected={len(unex_e)}; "
        f"decoder missing={len(miss_d)} unexpected={len(unex_d)})"
    )
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def _reconstruct_ft_vae(
    mesh_path_for_corep: str,
    encoder,
    decoder,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    resolution: int,
    device: torch.device,
    use_bf16: bool = True,
) -> trimesh.Trimesh | None:
    """Run the fine-tuned feat18 VAE. `mesh_path_for_corep` is a file path
    because `corep_fast.mesh_to_param` loads from disk. The output mesh
    lives in corep_fast's internal [0, 1] frame; caller re-normalizes."""
    param = mesh_to_param(
        mesh_path_for_corep, resolution=resolution, device=device, num_workers=1
    )
    cube_indices_np, feats_raw_np = param_to_feats(param)
    if cube_indices_np.shape[0] == 0:
        print("[ft_vae] empty corep voxelization, skipping")
        return None

    N = cube_indices_np.shape[0]
    batch_col = np.zeros((N, 1), dtype=np.int32)
    coords = torch.from_numpy(np.concatenate([batch_col, cube_indices_np], axis=1)).int().to(device)
    feats_t = torch.from_numpy(feats_raw_np).float().to(device)
    feats_norm = (feats_t - mean_t) / std_t
    x = sp.SparseTensor(feats=feats_norm, coords=coords)

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with autocast_ctx:
        z = encoder(x, sample_posterior=False)
        h = decoder(z)
        h = h[0] if isinstance(h, tuple) else h

    pred_norm = h.feats.float()
    pred_raw = (pred_norm * std_t + mean_t).cpu().numpy()

    param_pred = feats_to_param(pred_raw, cube_indices_np, resolution=resolution)
    print(f"param_pred.face_weights.shape: {param_pred.face_weights.shape}")
    print(f"param_pred.edge_weights.shape: {param_pred.edge_weights.shape}")
    print(f"edgeweightmax: {param_pred.edge_weights.max()}")
    breakpoint()
    verts_t, faces_t = param_to_mesh(param_pred, device=device, merge_decimals=5)
    V = verts_t.detach().cpu().numpy()
    F = faces_t.detach().cpu().numpy()
    if F.shape[0] == 0:
        return None
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


# ────────────────────────────── metrics ─────────────────────────────────────


def _bbox_normalize(tm_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center + scale to fit in [-0.5, 0.5]^3 (matches load_and_normalize_mesh)."""
    v = np.asarray(tm_mesh.vertices, dtype=np.float64)
    if v.shape[0] == 0:
        return tm_mesh
    vmin, vmax = v.min(0), v.max(0)
    center = (vmin + vmax) / 2
    extent = float((vmax - vmin).max())
    if extent == 0:
        return tm_mesh
    scale = 0.99999 / extent
    out = tm_mesh.copy()
    out.vertices = (v - center) * scale
    return out


def _compute_metrics(
    pred_mesh: trimesh.Trimesh | None,
    gt_mesh: trimesh.Trimesh,
    num_points: int,
    thresholds: list[float],
    do_render: bool,
    render_nviews: int,
    render_resolution: int,
    device: torch.device,
) -> dict:
    if pred_mesh is None or pred_mesh.faces.shape[0] == 0:
        return {"error": "empty reconstruction"}

    # Normalize both meshes independently to unit cube so scale/position
    # don't dominate the comparison (same convention as gap_measurement.py).
    pred_n = _bbox_normalize(pred_mesh)
    gt_n = _bbox_normalize(gt_mesh)

    gt_pts, gt_nm = sample_points_and_normals(gt_n, num_points)
    pr_pts, pr_nm = sample_points_and_normals(pred_n, num_points)
    gt_pts, gt_nm = gt_pts.to(device), gt_nm.to(device)
    pr_pts, pr_nm = pr_pts.to(device), pr_nm.to(device)

    result: dict = {}
    result["num_vertices"] = int(pred_mesh.vertices.shape[0])
    result["num_faces"] = int(pred_mesh.faces.shape[0])
    result["chamfer_distance"] = chamfer_distance(pr_pts, gt_pts)
    result["f_score"] = {
        f"{tau:.4f}": v
        for tau, v in f_score_multi(pr_pts, gt_pts, thresholds=thresholds).items()
    }
    result["normal_consistency"] = normal_consistency(pr_pts, pr_nm, gt_pts, gt_nm)

    if do_render:
        try:
            gt_trellis = trimesh_to_trellis_mesh(gt_n)
            pr_trellis = trimesh_to_trellis_mesh(pred_n)
            gt_maps = render_normal_maps(
                gt_trellis, nviews=render_nviews, resolution=render_resolution
            )
            pr_maps = render_normal_maps(
                pr_trellis, nviews=render_nviews, resolution=render_resolution
            )
            rm = compute_rendering_metrics(pr_maps, gt_maps)
            result["render_psnr"] = rm["psnr"]
            result["render_ssim"] = rm["ssim"]
        except Exception as e:
            print(f"[metrics] rendering failed: {type(e).__name__}: {e}")
            result["render_psnr"] = float("nan")
            result["render_ssim"] = float("nan")
    return result


def _format_metrics_txt(summary: dict) -> str:
    lines = []
    lines.append(f"Mesh            : {summary['input_mesh']}")
    lines.append(f"Resolution      : {summary['resolution']}")
    lines.append(f"Output dir      : {summary['output_dir']}")
    lines.append("")
    lines.append(
        f"{'metric':<22} {'raw (pretrained)':>20} {'ft (feat18)':>20}"
    )
    lines.append("-" * 66)

    a = summary["raw_vae"]
    b = summary["ft_vae"]

    def _row(key, fmt="{:>20.6f}", label=None):
        label = label or key
        va = a.get(key)
        vb = b.get(key)
        sa = fmt.format(va) if isinstance(va, (int, float)) else f"{str(va):>20}"
        sb = fmt.format(vb) if isinstance(vb, (int, float)) else f"{str(vb):>20}"
        lines.append(f"{label:<22} {sa} {sb}")

    _row("num_vertices", fmt="{:>20d}")
    _row("num_faces", fmt="{:>20d}")
    _row("chamfer_distance")
    _row("normal_consistency")
    for tau_key in a.get("f_score", {}).keys():
        va = a["f_score"][tau_key]
        vb = b["f_score"].get(tau_key, float("nan"))
        lines.append(
            f"{'f_score@' + tau_key:<22} {va:>20.6f} {vb:>20.6f}"
        )
    if "render_psnr" in a or "render_psnr" in b:
        _row("render_psnr")
        _row("render_ssim")

    lines.append("")
    lines.append(f"raw VAE recon time : {a.get('runtime_s', float('nan')):.2f} s")
    lines.append(f"ft  VAE recon time : {b.get('runtime_s', float('nan')):.2f} s")
    return "\n".join(lines) + "\n"


# ───────────────────────────────── main ──────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh_path",
        default=DEFAULT_MESH_PATH,
        help="Input mesh (.ply/.obj/.glb). Default is the Objaverse-XL Sketchfab "
        "sample whose precomputed feat18 sha256 is "
        f"{DEFAULT_OUT_NAME_FOR_DEFAULT_MESH} "
        "('Golden-armored warrior with wings').",
    )
    p.add_argument("--out_dir", default="tmp/test_vae")
    p.add_argument(
        "--out_name",
        default=None,
        help="Sub-folder name under out_dir. If omitted: use the feat18 sha256 "
        "when the default mesh is used, otherwise the input filename stem.",
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--latent_channels", type=int, default=32)

    p.add_argument(
        "--enc_pretrained",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16",
    )
    p.add_argument(
        "--dec_pretrained",
        default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
    )

    p.add_argument(
        "--ft_ckpt",
        default="results/finetune_feat18_1k_ddp/ckpt_step0004000.pt",
        help="Fine-tuned feat18 VAE checkpoint (produced by train_finetune_feat18.py).",
    )
    p.add_argument(
        "--stats_path",
        default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512/stats_rank0.npz",
        help="Per-channel mean/std used during fine-tuning.",
    )
    p.add_argument("--use_bf16", action="store_true", default=True,
                   help="Wrap ft VAE forward in bf16 autocast (matches training).")
    p.add_argument("--no_bf16", action="store_false", dest="use_bf16")

    p.add_argument("--skip_raw", action="store_true", default=False,
                   help="Skip the pretrained VAE reconstruction (debug only).")
    p.add_argument("--skip_ft", action="store_true", default=False,
                   help="Skip the fine-tuned VAE reconstruction (debug only).")

    p.add_argument("--num_points", type=int, default=10000,
                   help="Surface samples per mesh for CD/F-score/NC.")
    p.add_argument(
        "--f_thresholds",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05],
        help="F-score thresholds in the [-0.5, 0.5] unit-cube frame.",
    )
    p.add_argument("--skip_render", action="store_true", default=False,
                   help="Skip PSNR/SSIM normal-map rendering (faster).")
    p.add_argument("--render_nviews", type=int, default=8)
    p.add_argument("--render_resolution", type=int, default=512)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.out_name:
        sample_name = args.out_name
    elif os.path.abspath(args.mesh_path) == os.path.abspath(DEFAULT_MESH_PATH):
        sample_name = DEFAULT_OUT_NAME_FOR_DEFAULT_MESH
    else:
        sample_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
    out_dir = os.path.join(args.out_dir, sample_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[compare_vaes] input   : {args.mesh_path}")
    print(f"[compare_vaes] out_dir : {out_dir}")
    print(f"[compare_vaes] device  : {device}")

    # ── GT ──
    raw_gt_tm = trimesh.load(args.mesh_path, force="mesh")
    gt_tm = load_and_normalize_mesh(args.mesh_path)
    gt_path = os.path.join(out_dir, "gt.ply")
    gt_tm.export(gt_path)
    print(f"[gt] V={gt_tm.vertices.shape[0]} F={gt_tm.faces.shape[0]} -> {gt_path}")

    # Save the normalized GT as a file too, so corep can load from disk
    # exactly the same geometry both paths see.
    gt_norm_path = os.path.join(out_dir, "_gt_for_corep.ply")
    gt_tm.export(gt_norm_path)

    summary: dict = {
        "input_mesh": args.mesh_path,
        "output_dir": out_dir,
        "resolution": args.resolution,
        "raw_vae": {},
        "ft_vae": {},
    }

    # ── Path A: pretrained shape VAE ──
    if not args.skip_raw:
        print("\n[raw_vae] loading pretrained FlexiDualGrid shape VAE …")
        enc_A, dec_A = _load_raw_vae(
            args.enc_pretrained, args.dec_pretrained, args.resolution, device
        )
        t0 = time.time()
        recon_A = _reconstruct_raw_vae(
            gt_tm, enc_A, dec_A, args.resolution, device
        )
        t_A = time.time() - t0
        recon_A_path = os.path.join(out_dir, "recon_raw.ply")
        if recon_A is not None:
            recon_A.export(recon_A_path)
            print(
                f"[raw_vae] recon V={recon_A.vertices.shape[0]} "
                f"F={recon_A.faces.shape[0]} -> {recon_A_path} ({t_A:.2f}s)"
            )
        else:
            print(f"[raw_vae] empty reconstruction (not saved) ({t_A:.2f}s)")
        del enc_A, dec_A
        torch.cuda.empty_cache()
        summary["raw_vae"]["runtime_s"] = t_A
        summary["raw_vae"]["output_path"] = recon_A_path if recon_A is not None else None
    else:
        recon_A = None

    # ── Path B: fine-tuned feat18 VAE ──
    if not args.skip_ft:
        print("\n[ft_vae] loading fine-tuned feat18 VAE …")
        mean_t, std_t = _load_stats(args.stats_path, device)
        enc_B, dec_B = _load_ft_vae(args.ft_ckpt, args.latent_channels, device)
        t0 = time.time()
        recon_B = _reconstruct_ft_vae(
            gt_norm_path,
            enc_B,
            dec_B,
            mean_t,
            std_t,
            args.resolution,
            device,
            use_bf16=args.use_bf16,
        )
        t_B = time.time() - t0
        recon_B_path = os.path.join(out_dir, "recon_ft.ply")
        if recon_B is not None:
            recon_B.export(recon_B_path)
            print(
                f"[ft_vae] recon V={recon_B.vertices.shape[0]} "
                f"F={recon_B.faces.shape[0]} -> {recon_B_path} ({t_B:.2f}s)"
            )
        else:
            print(f"[ft_vae] empty reconstruction (not saved) ({t_B:.2f}s)")
        del enc_B, dec_B
        torch.cuda.empty_cache()
        summary["ft_vae"]["runtime_s"] = t_B
        summary["ft_vae"]["output_path"] = recon_B_path if recon_B is not None else None
    else:
        recon_B = None

    # ── metrics ──
    print("\n[metrics] computing …")
    if recon_A is not None:
        summary["raw_vae"].update(
            _compute_metrics(
                recon_A, gt_tm, args.num_points, args.f_thresholds,
                do_render=not args.skip_render,
                render_nviews=args.render_nviews,
                render_resolution=args.render_resolution,
                device=device,
            )
        )
    if recon_B is not None:
        summary["ft_vae"].update(
            _compute_metrics(
                recon_B, gt_tm, args.num_points, args.f_thresholds,
                do_render=not args.skip_render,
                render_nviews=args.render_nviews,
                render_resolution=args.render_resolution,
                device=device,
            )
        )

    metrics_json_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    metrics_txt_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write(_format_metrics_txt(summary))

    print(f"\n[compare_vaes] wrote {metrics_json_path}")
    print(f"[compare_vaes] wrote {metrics_txt_path}\n")
    print(_format_metrics_txt(summary))

    if os.path.exists(gt_norm_path):
        os.remove(gt_norm_path)


if __name__ == "__main__":
    main()
