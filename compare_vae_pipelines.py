"""
Batch comparison for TRELLIS.2 shape reconstruction pipelines on the feat18 dataset.

This script samples N examples from the Objaverse feat18 dataset and compares:
  * `ovoxel_roundtrip`: mesh -> O-Voxel -> mesh
  * `raw_vae`: mesh -> O-Voxel -> pretrained TRELLIS.2 shape VAE -> mesh
  * `ft_vae`: feat18 -> fine-tuned feat18 VAE -> mesh
  * `corep_roundtrip`: mesh -> CoReP -> mesh

It writes:
  * per-sample meshes under `<out_dir>/samples/<sha>/`
  * `per_sample.json`
  * `per_sample.csv`
  * `summary.json`
  * `summary.md`

Typical usage:
    python compare_vae_pipelines.py \
        --dataset_root /mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab \
        --feat18_dir /mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/feat18_512 \
        --ft_ckpt results/finetune_feat18_1k/ckpt_step0050000.pt \
        --num_examples 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import trimesh

from trellis2 import models
from trellis2.modules import sparse as sp

from corep_fast.pipeline import mesh_to_param, param_to_mesh
from scripts.eval.eval_metrics import (
    chamfer_distance,
    f_score_multi,
    normal_consistency,
    sample_points_and_normals,
)
from scripts.eval.gap_measurement import load_and_normalize_mesh
from train_overfit_feat18 import build_models, feats_to_param


AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
PIPELINES = (
    "ovoxel_roundtrip",
    "raw_vae",
    "ft_vae",
    "corep_roundtrip",
)


@dataclass
class SampleRecord:
    sha256: str
    mesh_path: str
    feat18_path: str | None


def _resolve_default_pretrained(local_rel: str, hf_rel: str) -> str:
    local_json = f"{local_rel}.json"
    if os.path.exists(local_json):
        return local_rel
    return hf_rel


def _bbox_normalize(tm_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    v = np.asarray(tm_mesh.vertices, dtype=np.float64)
    if v.shape[0] == 0:
        return tm_mesh
    vmin, vmax = v.min(0), v.max(0)
    extent = float((vmax - vmin).max())
    if extent <= 0:
        return tm_mesh
    center = (vmin + vmax) / 2
    out = tm_mesh.copy()
    out.vertices = (v - center) * (0.99999 / extent)
    return out


def _mesh_from_vertices_faces(vertices: torch.Tensor, faces: torch.Tensor) -> trimesh.Trimesh | None:
    v_np = vertices.detach().cpu().numpy()
    f_np = faces.detach().cpu().numpy()
    if f_np.shape[0] == 0:
        return None
    return trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)


def _load_stats(stats_path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = np.maximum(stats["std"].astype(np.float32), 1e-3)
    return torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)


def _load_raw_vae(enc_path: str, dec_path: str, resolution: int, device: torch.device):
    enc = models.from_pretrained(enc_path).eval().to(device)
    dec = models.from_pretrained(dec_path).eval().to(device)
    if hasattr(dec, "set_resolution"):
        dec.set_resolution(resolution)
    return enc, dec


def _load_ft_vae(ckpt_path: str, latent_channels: int, device: torch.device):
    encoder, decoder = build_models(latent_channels=latent_channels, device=str(device))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def _reconstruct_raw_vae(
    gt_mesh_norm: trimesh.Trimesh,
    encoder,
    decoder,
    resolution: int,
    device: torch.device,
) -> trimesh.Trimesh | None:
    import o_voxel

    vertices = torch.from_numpy(gt_mesh_norm.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh_norm.faces.copy()).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=resolution,
        aabb=AABB,
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )
    if voxel_indices.shape[0] == 0:
        return None

    dv_local = torch.clamp(dual_vertices * resolution - voxel_indices.float(), 0, 1)
    coords = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int32), voxel_indices],
        dim=-1,
    )
    vertices_st = sp.SparseTensor(feats=dv_local, coords=coords).to(device)
    intersected_st = vertices_st.replace(intersected.float().to(device))

    z = encoder(vertices_st, intersected_st, sample_posterior=False)
    out = decoder(z)
    mesh_list = out[0] if isinstance(out, tuple) else out
    mesh_trellis = mesh_list[0] if isinstance(mesh_list, list) else mesh_list
    return _mesh_from_vertices_faces(mesh_trellis.vertices, mesh_trellis.faces)


@torch.no_grad()
def _reconstruct_ft_vae_from_feats(
    cube_indices_np: np.ndarray,
    feats_raw_np: np.ndarray,
    num_boundary_np: np.ndarray | None,
    encoder,
    decoder,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    resolution: int,
    device: torch.device,
    use_bf16: bool,
) -> trimesh.Trimesh | None:
    n_voxels = cube_indices_np.shape[0]
    coords = torch.from_numpy(
        np.concatenate([np.zeros((n_voxels, 1), dtype=np.int32), cube_indices_np], axis=1)
    ).to(device=device, dtype=torch.int32)
    feats_t = torch.from_numpy(feats_raw_np).float().to(device)
    feats_norm = (feats_t - mean_t) / std_t
    x = sp.SparseTensor(feats=feats_norm, coords=coords)

    if use_bf16 and device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

    with autocast_ctx:
        z = encoder(x, sample_posterior=False)
        h = decoder(z)
        h = h[0] if isinstance(h, tuple) else h

    pred_raw = (h.feats.float() * std_t + mean_t).detach().cpu().numpy()
    param_pred = feats_to_param(
        pred_raw,
        cube_indices_np,
        resolution=resolution,
        num_boundary=num_boundary_np,
    )
    vertices, faces = param_to_mesh(param_pred, device=device, merge_decimals=5)
    return _mesh_from_vertices_faces(vertices, faces)


@torch.no_grad()
def _ovoxel_roundtrip(
    gt_mesh_norm: trimesh.Trimesh,
    resolution: int,
    device: torch.device,
) -> trimesh.Trimesh | None:
    import o_voxel

    vertices = torch.from_numpy(gt_mesh_norm.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh_norm.faces.copy()).long()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=resolution,
        aabb=AABB,
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )
    if voxel_indices.shape[0] == 0:
        return None

    out_verts, out_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
        voxel_indices.to(device),
        dual_vertices.to(device),
        intersected.to(device),
        split_weight=None,
        grid_size=resolution,
        aabb=AABB,
    )
    mesh = _mesh_from_vertices_faces(out_verts, out_faces)
    if mesh is None:
        return None

    try:
        import cumesh

        cm = cumesh.CuMesh()
        cm.init(out_verts, out_faces)
        cm.fill_holes(max_hole_perimeter=3e-2)
        cm.remove_duplicate_faces()
        cm.repair_non_manifold_edges()
        cm.remove_small_connected_components(1e-5)
        cm.fill_holes(max_hole_perimeter=3e-2)
        cm.unify_face_orientations()
        clean_verts, clean_faces = cm.read()
        mesh = _mesh_from_vertices_faces(clean_verts, clean_faces)
    except Exception:
        pass
    return mesh


@torch.no_grad()
def _corep_roundtrip(
    mesh_path: str,
    resolution: int,
    device: torch.device,
    num_workers: int,
) -> trimesh.Trimesh | None:
    param = mesh_to_param(
        mesh_path,
        resolution=resolution,
        device=device,
        num_workers=num_workers,
    )
    vertices, faces = param_to_mesh(
        param,
        device=device,
        merge_decimals=5,
        num_workers=num_workers,
    )
    return _mesh_from_vertices_faces(vertices, faces)


def _compute_metrics(
    pred_mesh: trimesh.Trimesh | None,
    gt_mesh: trimesh.Trimesh,
    num_points: int,
    thresholds: list[float],
    device: torch.device,
) -> dict[str, Any]:
    if pred_mesh is None or pred_mesh.faces.shape[0] == 0:
        return {"error": "empty reconstruction"}

    pred_n = _bbox_normalize(pred_mesh)
    gt_n = _bbox_normalize(gt_mesh)

    gt_pts, gt_nm = sample_points_and_normals(gt_n, num_points)
    pr_pts, pr_nm = sample_points_and_normals(pred_n, num_points)
    gt_pts = gt_pts.to(device)
    gt_nm = gt_nm.to(device)
    pr_pts = pr_pts.to(device)
    pr_nm = pr_nm.to(device)

    metrics: dict[str, Any] = {
        "num_vertices": int(pred_mesh.vertices.shape[0]),
        "num_faces": int(pred_mesh.faces.shape[0]),
        "chamfer_distance": chamfer_distance(pr_pts, gt_pts),
        "normal_consistency": normal_consistency(pr_pts, pr_nm, gt_pts, gt_nm),
    }
    fs = f_score_multi(pr_pts, gt_pts, thresholds=thresholds)
    for tau, value in fs.items():
        metrics[f"f_score@{tau:.4f}"] = value
    return metrics


def _flatten_for_csv(sample_rows: list[dict[str, Any]], thresholds: list[float]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = ["sha256", "mesh_path"]
    for pipeline in PIPELINES:
        fieldnames.extend(
            [
                f"{pipeline}_runtime_s",
                f"{pipeline}_num_vertices",
                f"{pipeline}_num_faces",
                f"{pipeline}_chamfer_distance",
                f"{pipeline}_normal_consistency",
                f"{pipeline}_error",
            ]
        )
        for tau in thresholds:
            fieldnames.append(f"{pipeline}_f_score@{tau:.4f}")

    rows: list[dict[str, Any]] = []
    for row in sample_rows:
        flat = {"sha256": row["sha256"], "mesh_path": row["mesh_path"]}
        for pipeline in PIPELINES:
            metrics = row["pipelines"].get(pipeline, {})
            flat[f"{pipeline}_runtime_s"] = metrics.get("runtime_s")
            flat[f"{pipeline}_num_vertices"] = metrics.get("num_vertices")
            flat[f"{pipeline}_num_faces"] = metrics.get("num_faces")
            flat[f"{pipeline}_chamfer_distance"] = metrics.get("chamfer_distance")
            flat[f"{pipeline}_normal_consistency"] = metrics.get("normal_consistency")
            flat[f"{pipeline}_error"] = metrics.get("error", "")
            for tau in thresholds:
                flat[f"{pipeline}_f_score@{tau:.4f}"] = metrics.get(f"f_score@{tau:.4f}")
        rows.append(flat)
    return fieldnames, rows


def _aggregate_pipeline(sample_rows: list[dict[str, Any]], pipeline: str) -> dict[str, Any]:
    metric_names: set[str] = set()
    valid = 0
    failed = 0
    for row in sample_rows:
        metrics = row["pipelines"].get(pipeline, {})
        if metrics.get("error"):
            failed += 1
            continue
        valid += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                metric_names.add(key)

    summary: dict[str, Any] = {
        "num_success": valid,
        "num_failed": failed,
        "metrics": {},
    }
    for key in sorted(metric_names):
        values = []
        for row in sample_rows:
            metrics = row["pipelines"].get(pipeline, {})
            value = metrics.get(key)
            if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value):
                values.append(float(value))
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary["metrics"][key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return summary


def _write_summary_md(summary_path: str, config: dict[str, Any], aggregate: dict[str, Any], thresholds: list[float]) -> None:
    lines = []
    lines.append("# VAE Pipeline Comparison")
    lines.append("")
    lines.append(f"- Date: {config['timestamp']}")
    lines.append(f"- Resolution: {config['resolution']}")
    lines.append(f"- Num examples: {config['num_examples']}")
    lines.append(f"- Seed: {config['seed']}")
    lines.append("")
    lines.append("| Pipeline | Success | CD mean | NC mean | Runtime mean (s) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for pipeline in PIPELINES:
        info = aggregate[pipeline]
        metrics = info["metrics"]
        cd = metrics.get("chamfer_distance", {}).get("mean")
        nc = metrics.get("normal_consistency", {}).get("mean")
        rt = metrics.get("runtime_s", {}).get("mean")
        lines.append(
            f"| {pipeline} | {info['num_success']}/{info['num_success'] + info['num_failed']} | "
            f"{'' if cd is None else f'{cd:.6f}'} | "
            f"{'' if nc is None else f'{nc:.6f}'} | "
            f"{'' if rt is None else f'{rt:.3f}'} |"
        )
    lines.append("")
    lines.append("## F-score Means")
    lines.append("")
    lines.append("| Pipeline | " + " | ".join(f"F@{tau:.4f}" for tau in thresholds) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in thresholds) + " |")
    for pipeline in PIPELINES:
        metrics = aggregate[pipeline]["metrics"]
        vals = []
        for tau in thresholds:
            item = metrics.get(f"f_score@{tau:.4f}", {})
            mean = item.get("mean")
            vals.append("" if mean is None else f"{mean:.6f}")
        lines.append(f"| {pipeline} | " + " | ".join(vals) + " |")
    lines.append("")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset_root",
        default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab",
        help="Root containing the metadata CSV and raw meshes.",
    )
    p.add_argument(
        "--metadata_csv",
        default="metadata_first1k.csv",
        help="CSV with `sha256` and `local_path` columns relative to dataset_root.",
    )
    p.add_argument(
        "--feat18_dir",
        default=None,
        help="Directory containing `data/<sha>.npz` and `stats_rank0.npz`. Defaults to <dataset_root>/feat18_<resolution>.",
    )
    p.add_argument(
        "--stats_path",
        default=None,
        help="Override stats path for feat18 normalization. Defaults to <feat18_dir>/stats_rank0.npz.",
    )
    p.add_argument("--ft_ckpt", default="results/finetune_feat18_1k_ddp/ckpt_step0050000.pt")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--latent_channels", type=int, default=32)
    p.add_argument("--num_examples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_points", type=int, default=10000)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument(
        "--f_thresholds",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="results/compare_vae_pipelines")
    p.add_argument("--save_meshes", action="store_true", default=True)
    p.add_argument("--no_save_meshes", action="store_false", dest="save_meshes")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="use_bf16")
    p.add_argument(
        "--enc_pretrained",
        default=_resolve_default_pretrained(
            "pretrained/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16",
            "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16",
        ),
    )
    p.add_argument(
        "--dec_pretrained",
        default=_resolve_default_pretrained(
            "pretrained/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
            "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        ),
    )
    return p.parse_args()


def _select_samples(args) -> list[SampleRecord]:
    metadata_path = os.path.join(args.dataset_root, args.metadata_csv)
    metadata = pd.read_csv(metadata_path)
    if "sha256" not in metadata.columns or "local_path" not in metadata.columns:
        raise RuntimeError(
            f"{metadata_path} must contain `sha256` and `local_path`, got {list(metadata.columns)}"
        )

    feat18_dir = args.feat18_dir or os.path.join(args.dataset_root, f"feat18_{args.resolution}")
    candidates: list[SampleRecord] = []
    for row in metadata.itertuples(index=False):
        mesh_path = os.path.join(args.dataset_root, row.local_path)
        feat18_path = os.path.join(feat18_dir, "data", f"{row.sha256}.npz")
        if not os.path.exists(mesh_path):
            continue
        if not os.path.exists(feat18_path):
            continue
        candidates.append(SampleRecord(row.sha256, mesh_path, feat18_path))

    if len(candidates) < args.num_examples:
        raise RuntimeError(
            f"Only found {len(candidates)} usable samples, need {args.num_examples}."
        )

    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(candidates), size=args.num_examples, replace=False)
    picked.sort()
    return [candidates[int(i)] for i in picked]


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script currently requires a CUDA device.")
    if not os.path.exists(args.ft_ckpt):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.ft_ckpt}")

    feat18_dir = args.feat18_dir or os.path.join(args.dataset_root, f"feat18_{args.resolution}")
    stats_path = args.stats_path or os.path.join(feat18_dir, "stats_rank0.npz")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    samples = _select_samples(args)
    selected_csv = os.path.join(args.out_dir, "selected_samples.csv")
    with open(selected_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sha256", "mesh_path", "feat18_path"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sha256": sample.sha256,
                    "mesh_path": sample.mesh_path,
                    "feat18_path": sample.feat18_path,
                }
            )

    print(f"[compare] selected {len(samples)} samples -> {selected_csv}")
    print("[compare] loading models...")
    mean_t, std_t = _load_stats(stats_path, device)
    raw_encoder, raw_decoder = _load_raw_vae(
        args.enc_pretrained,
        args.dec_pretrained,
        args.resolution,
        device,
    )
    ft_encoder, ft_decoder = _load_ft_vae(args.ft_ckpt, args.latent_channels, device)

    sample_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[compare] {index}/{len(samples)} {sample.sha256[:12]} {sample.mesh_path}")
        sample_dir = os.path.join(args.out_dir, "samples", sample.sha256)
        os.makedirs(sample_dir, exist_ok=True)

        gt_mesh_norm = load_and_normalize_mesh(sample.mesh_path)
        if args.save_meshes:
            gt_mesh_norm.export(os.path.join(sample_dir, "gt.ply"))

        feat_npz = np.load(sample.feat18_path)
        cube_indices_np = feat_npz["cube_indices"].astype(np.int32)
        feats_raw_np = feat_npz["feats"].astype(np.float32)
        num_boundary_np = feat_npz["num_boundary"].astype(np.int32)

        row: dict[str, Any] = {
            "sha256": sample.sha256,
            "mesh_path": sample.mesh_path,
            "pipelines": {},
        }

        pipeline_meshes: dict[str, trimesh.Trimesh | None] = {}

        t0 = time.time()
        pipeline_meshes["ovoxel_roundtrip"] = _ovoxel_roundtrip(gt_mesh_norm, args.resolution, device)
        row["pipelines"]["ovoxel_roundtrip"] = {"runtime_s": time.time() - t0}

        t0 = time.time()
        pipeline_meshes["raw_vae"] = _reconstruct_raw_vae(
            gt_mesh_norm,
            raw_encoder,
            raw_decoder,
            args.resolution,
            device,
        )
        row["pipelines"]["raw_vae"] = {"runtime_s": time.time() - t0}

        t0 = time.time()
        pipeline_meshes["ft_vae"] = _reconstruct_ft_vae_from_feats(
            cube_indices_np,
            feats_raw_np,
            num_boundary_np,
            ft_encoder,
            ft_decoder,
            mean_t,
            std_t,
            args.resolution,
            device,
            use_bf16=args.use_bf16,
        )
        row["pipelines"]["ft_vae"] = {"runtime_s": time.time() - t0}

        t0 = time.time()
        pipeline_meshes["corep_roundtrip"] = _corep_roundtrip(
            sample.mesh_path,
            args.resolution,
            device,
            num_workers=args.num_workers,
        )
        row["pipelines"]["corep_roundtrip"] = {"runtime_s": time.time() - t0}

        for pipeline, mesh in pipeline_meshes.items():
            if args.save_meshes and mesh is not None:
                mesh.export(os.path.join(sample_dir, f"{pipeline}.ply"))
            metrics = _compute_metrics(
                mesh,
                gt_mesh_norm,
                num_points=args.num_points,
                thresholds=args.f_thresholds,
                device=device,
            )
            row["pipelines"][pipeline].update(metrics)

        sample_rows.append(row)
        torch.cuda.empty_cache()

    per_sample_json = os.path.join(args.out_dir, "per_sample.json")
    with open(per_sample_json, "w") as fh:
        json.dump(sample_rows, fh, indent=2)

    fieldnames, csv_rows = _flatten_for_csv(sample_rows, args.f_thresholds)
    per_sample_csv = os.path.join(args.out_dir, "per_sample.csv")
    with open(per_sample_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    aggregate = {pipeline: _aggregate_pipeline(sample_rows, pipeline) for pipeline in PIPELINES}
    summary = {
        "config": {
            "dataset_root": args.dataset_root,
            "metadata_csv": args.metadata_csv,
            "feat18_dir": feat18_dir,
            "stats_path": stats_path,
            "ft_ckpt": args.ft_ckpt,
            "resolution": args.resolution,
            "num_examples": args.num_examples,
            "seed": args.seed,
            "num_points": args.num_points,
            "thresholds": args.f_thresholds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        },
        "aggregate": aggregate,
    }

    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    summary_md = os.path.join(args.out_dir, "summary.md")
    _write_summary_md(summary_md, summary["config"], aggregate, args.f_thresholds)

    print(f"[compare] wrote {per_sample_json}")
    print(f"[compare] wrote {per_sample_csv}")
    print(f"[compare] wrote {summary_json}")
    print(f"[compare] wrote {summary_md}")


if __name__ == "__main__":
    main()
