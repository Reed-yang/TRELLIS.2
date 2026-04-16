"""
Component-Level Evaluation Pipeline.

Phase A: VAE reconstruction baseline (geometric + rendering metrics)
Phase B: DiT generation best-of-16 views (with/without fill_holes)
Phase C: GT injection stage breakdown (5 experimental conditions)

Usage:
    python scripts/eval/component_eval.py --phase a --manifest experiments/component_eval/test_set/manifest.json
    python scripts/eval/component_eval.py --phase b --manifest ... --rank 0 --world_size 8
    python scripts/eval/component_eval.py --phase c --manifest ... --phase_b_csv experiments/component_eval/phase_b/results/per_sample.csv
    python scripts/eval/component_eval.py --phase merge --manifest ... --output_dir experiments/component_eval
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import argparse
import csv
import traceback
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from PIL import Image

from scripts.eval.gap_measurement import (
    load_and_normalize_mesh,
    trimesh_to_trellis_mesh,
    load_vae_models,
    vae_reconstruct,
    load_pipeline,
    _patch_gated_models,
)
from scripts.eval.eval_metrics import (
    sample_points_and_normals,
    trellis_mesh_to_trimesh,
    chamfer_distance,
    f_score_multi,
    normal_consistency,
    find_best_rotation_24_with_icp,
    apply_transform_to_points,
    render_normal_maps_paper_config,
    compute_rendering_metrics,
    compute_lpips,
    F_SCORE_THRESHOLDS,
)

GRID_SIZE = 512


def _load_done_uids(csv_path):
    """Load UIDs already processed from ALL CSV files in the same directory (for resume support).

    Scans all per_sample*.csv files, not just the current rank's file,
    so resume works correctly even when world_size changes between runs.
    """
    done = set()
    results_dir = os.path.dirname(csv_path)
    if not os.path.exists(results_dir):
        return done
    import glob
    for f_path in glob.glob(os.path.join(results_dir, "per_sample*.csv")):
        if os.path.getsize(f_path) > 0:
            with open(f_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("uid"):
                        done.add(row["uid"])
    return done


# ---------------------------------------------------------------------------
# Phase A: VAE Reconstruction Baseline
# ---------------------------------------------------------------------------

def run_phase_a(manifest, output_dir, rank=0, world_size=1):
    """Phase A: SC-VAE encode-decode for each object, evaluate geometry + rendering."""
    results_dir = os.path.join(output_dir, "phase_a", "results")
    meshes_dir = os.path.join(output_dir, "phase_a", "meshes")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)

    # Build CSV field names with multi-threshold f-scores
    fscore_cols = [f"fscore_{t}" for t in F_SCORE_THRESHOLDS]
    fieldnames = [
        "uid", "category", "tier", "face_count",
        "cd_100k", "cd_1m", "nc",
        "psnr", "ssim", "lpips",
    ] + fscore_cols + ["error"]

    csv_suffix = f"_rank{rank}" if world_size > 1 else ""
    csv_path = os.path.join(results_dir, f"per_sample{csv_suffix}.csv")

    # Resume support: skip already-processed UIDs
    done_uids = _load_done_uids(csv_path)
    if done_uids:
        print(f"Resuming: {len(done_uids)} already done, {len(manifest) - len(done_uids)} remaining")

    print("Loading SC-VAE encoder + decoder...")
    encoder, decoder = load_vae_models()

    # Append mode if resuming, write mode if fresh
    mode = 'a' if done_uids else 'w'
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not done_uids:
            writer.writeheader()

        for item in tqdm(manifest, desc="Phase A"):
            uid = item["uid"]
            if uid in done_uids:
                continue
            mesh_path = item["mesh_path"]
            category = item.get("category", "unknown")
            tier = item.get("tier", "unknown")

            row = {k: "" for k in fieldnames}
            row["uid"] = uid
            row["category"] = category
            row["tier"] = tier

            try:
                # Get face count from GT mesh
                gt_trimesh = load_and_normalize_mesh(mesh_path)
                row["face_count"] = len(gt_trimesh.faces)

                # VAE reconstruct
                recon_mesh = vae_reconstruct(mesh_path, encoder, decoder)
                if recon_mesh is None:
                    raise RuntimeError("vae_reconstruct returned None")

                # Save reconstructed mesh
                try:
                    recon_tm = trellis_mesh_to_trimesh(recon_mesh)
                    recon_tm.export(os.path.join(meshes_dir, f"{uid}.obj"))
                except Exception:
                    pass

                gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                recon_trimesh = trellis_mesh_to_trimesh(recon_mesh)

                # Geometric metrics at 100K points
                gt_pts_100k, gt_nrm_100k = sample_points_and_normals(gt_trimesh, 100_000)
                gt_pts_100k, gt_nrm_100k = gt_pts_100k.cuda(), gt_nrm_100k.cuda()
                recon_pts_100k, recon_nrm_100k = sample_points_and_normals(recon_trimesh, 100_000)
                recon_pts_100k, recon_nrm_100k = recon_pts_100k.cuda(), recon_nrm_100k.cuda()

                row["cd_100k"] = chamfer_distance(recon_pts_100k, gt_pts_100k)
                row["nc"] = normal_consistency(recon_pts_100k, recon_nrm_100k, gt_pts_100k, gt_nrm_100k)

                # F-scores at multiple thresholds (using 100K points)
                fscores = f_score_multi(recon_pts_100k, gt_pts_100k)
                for t in F_SCORE_THRESHOLDS:
                    row[f"fscore_{t}"] = fscores[t]

                # Geometric metrics at 1M points (CD only, for precision)
                gt_pts_1m, _ = sample_points_and_normals(gt_trimesh, 1_000_000)
                gt_pts_1m = gt_pts_1m.cuda()
                recon_pts_1m, _ = sample_points_and_normals(recon_trimesh, 1_000_000)
                recon_pts_1m = recon_pts_1m.cuda()
                row["cd_1m"] = chamfer_distance(recon_pts_1m, gt_pts_1m)

                del gt_pts_1m, recon_pts_1m
                torch.cuda.empty_cache()

                # Rendering metrics: 4-view normal maps (paper config)
                try:
                    gt_nmaps = render_normal_maps_paper_config(gt_trellis)
                    recon_nmaps = render_normal_maps_paper_config(recon_mesh)
                    render_m = compute_rendering_metrics(recon_nmaps, gt_nmaps)
                    row["psnr"] = render_m["psnr"]
                    row["ssim"] = render_m["ssim"]
                    row["lpips"] = compute_lpips(recon_nmaps, gt_nmaps)
                except Exception as e:
                    print(f"  [Render Error] {uid}: {e}")
                    row["psnr"] = float('nan')
                    row["ssim"] = float('nan')
                    row["lpips"] = float('nan')

            except Exception as e:
                print(f"  [Phase A Error] {uid}: {e}")
                traceback.print_exc()
                row["error"] = str(e)

            writer.writerow(row)
            csvfile.flush()

            # Free GPU memory
            torch.cuda.empty_cache()

    print(f"Phase A results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Phase B: DiT Generation Best-of-16 Views
# ---------------------------------------------------------------------------

def _get_view_image_path(item, view_idx):
    """Get the path to the conditioning image for a specific view index."""
    renders_dir = item.get("renders_dir")
    uid = item["uid"]
    if renders_dir:
        return os.path.join(renders_dir, uid, f"{view_idx:03d}.png")
    # Fallback: try to infer from image_path by replacing the filename
    image_path = item.get("image_path", "")
    if image_path:
        parent = os.path.dirname(image_path)
        return os.path.join(parent, f"{view_idx:03d}.png")
    return None


def _apply_transform_to_trellis_mesh(trellis_mesh, transform):
    """Apply a 4x4 rigid transform to a trellis mesh (in-place vertices)."""
    R = torch.from_numpy(transform[:3, :3].copy()).float().to(trellis_mesh.vertices.device)
    t = torch.from_numpy(transform[:3, 3].copy()).float().to(trellis_mesh.vertices.device)
    trellis_mesh.vertices = trellis_mesh.vertices @ R.T + t
    return trellis_mesh


def run_phase_b(manifest, output_dir, rank=0, world_size=1):
    """Phase B: 16-view scan, pick best view by CD, compute full metrics."""
    results_dir = os.path.join(output_dir, "phase_b", "results")
    meshes_dir = os.path.join(output_dir, "phase_b", "meshes")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)

    # Build CSV field names
    fscore_filled_cols = [f"fscore_{t}_filled" for t in F_SCORE_THRESHOLDS]
    fscore_raw_cols = [f"fscore_{t}_raw" for t in F_SCORE_THRESHOLDS]
    view_cd_cols = [f"cd_view_{i}" for i in range(16)]
    fieldnames = [
        "uid", "category", "tier", "face_count", "best_view_idx",
        "cd_filled", "cd_raw", "nc_filled", "nc_raw",
        "psnr", "ssim", "lpips",
    ] + fscore_filled_cols + fscore_raw_cols + view_cd_cols + ["error"]

    csv_suffix = f"_rank{rank}" if world_size > 1 else ""
    csv_path = os.path.join(results_dir, f"per_sample{csv_suffix}.csv")

    # Resume support: skip already-processed UIDs
    done_uids = _load_done_uids(csv_path)
    if done_uids:
        print(f"Resuming: {len(done_uids)} already done, {len(manifest) - len(done_uids)} remaining")

    # Load pipeline
    pipeline = load_pipeline()

    mode = 'a' if done_uids else 'w'
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not done_uids:
            writer.writeheader()

        for item in tqdm(manifest, desc="Phase B"):
            uid = item["uid"]
            if uid in done_uids:
                continue
            mesh_path = item["mesh_path"]
            category = item.get("category", "unknown")
            tier = item.get("tier", "unknown")

            row = {k: "" for k in fieldnames}
            row["uid"] = uid
            row["category"] = category
            row["tier"] = tier

            try:
                # Load GT mesh
                gt_trimesh = load_and_normalize_mesh(mesh_path)
                row["face_count"] = len(gt_trimesh.faces)
                gt_pts, gt_nrm = sample_points_and_normals(gt_trimesh, 100_000)
                gt_pts, gt_nrm = gt_pts.cuda(), gt_nrm.cuda()

                # --- 16-view scan: generate mesh for each view, save to disk ---
                view_cds = []
                for vi in range(16):
                    view_path = _get_view_image_path(item, vi)
                    if view_path is None or not os.path.exists(view_path):
                        print(f"  [View {vi}] Image not found: {view_path}")
                        view_cds.append(float('inf'))
                        row[f"cd_view_{vi}"] = float('nan')
                        continue

                    try:
                        image = Image.open(view_path).convert("RGBA")
                        gen_meshes = pipeline.run(image, pipeline_type='512')
                        gen_mesh = gen_meshes[0]

                        # Save mesh to disk for later retrieval
                        gen_tm = trellis_mesh_to_trimesh(gen_mesh)
                        mesh_save_path = os.path.join(meshes_dir, f"{uid}_view{vi:03d}.obj")
                        gen_tm.export(mesh_save_path)

                        # Quick CD with alignment for ranking
                        gen_pts, gen_nrm = sample_points_and_normals(gen_tm, 100_000)
                        gen_pts, gen_nrm = gen_pts.cuda(), gen_nrm.cuda()
                        _, cd_val = find_best_rotation_24_with_icp(gen_pts, gt_pts)
                        view_cds.append(cd_val)
                        row[f"cd_view_{vi}"] = cd_val

                        del gen_mesh, gen_meshes, gen_tm, gen_pts, gen_nrm
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  [View {vi} Error] {uid}: {e}")
                        traceback.print_exc()
                        view_cds.append(float('inf'))
                        row[f"cd_view_{vi}"] = float('nan')
                        torch.cuda.empty_cache()

                # Find best view
                if all(cd == float('inf') for cd in view_cds):
                    raise RuntimeError("All 16 views failed")

                best_idx = int(np.argmin(view_cds))
                row["best_view_idx"] = best_idx

                # --- Full metrics on best view (with fill_holes) ---
                best_mesh_path = os.path.join(meshes_dir, f"{uid}_view{best_idx:03d}.obj")
                best_tm = trimesh.load(best_mesh_path, force="mesh")
                best_pts, best_nrm = sample_points_and_normals(best_tm, 100_000)
                best_pts, best_nrm = best_pts.cuda(), best_nrm.cuda()

                # Align with ICP
                transform, cd_filled = find_best_rotation_24_with_icp(best_pts, gt_pts)
                row["cd_filled"] = cd_filled
                aligned_pts, aligned_nrm = apply_transform_to_points(best_pts, best_nrm, transform)
                row["nc_filled"] = normal_consistency(aligned_pts, aligned_nrm, gt_pts, gt_nrm)

                # F-scores (filled)
                fscores_filled = f_score_multi(aligned_pts, gt_pts)
                for t in F_SCORE_THRESHOLDS:
                    row[f"fscore_{t}_filled"] = fscores_filled[t]

                # Rendering metrics: apply transform to trellis mesh, then render
                try:
                    gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                    gt_nmaps = render_normal_maps_paper_config(gt_trellis)

                    # Apply alignment transform to reconstructed mesh for rendering
                    best_trellis = trimesh_to_trellis_mesh(best_tm)
                    best_trellis = _apply_transform_to_trellis_mesh(best_trellis, transform)
                    pred_nmaps = render_normal_maps_paper_config(best_trellis)

                    render_m = compute_rendering_metrics(pred_nmaps, gt_nmaps)
                    row["psnr"] = render_m["psnr"]
                    row["ssim"] = render_m["ssim"]
                    row["lpips"] = compute_lpips(pred_nmaps, gt_nmaps)
                except Exception as e:
                    print(f"  [Render Error] {uid}: {e}")
                    row["psnr"] = float('nan')
                    row["ssim"] = float('nan')
                    row["lpips"] = float('nan')

                del best_pts, best_nrm, aligned_pts, aligned_nrm
                torch.cuda.empty_cache()

                # --- Re-run best view WITHOUT fill_holes ---
                best_view_path = _get_view_image_path(item, best_idx)
                try:
                    from trellis2.representations.mesh.base import Mesh as BaseMesh
                    original_fill_holes = BaseMesh.fill_holes
                    BaseMesh.fill_holes = lambda self, *a, **kw: None
                    try:
                        image = Image.open(best_view_path).convert("RGBA")
                        raw_meshes = pipeline.run(image, pipeline_type='512')
                        raw_mesh = raw_meshes[0]
                    finally:
                        BaseMesh.fill_holes = original_fill_holes

                    raw_tm = trellis_mesh_to_trimesh(raw_mesh)
                    raw_tm.export(os.path.join(meshes_dir, f"{uid}_best_raw.obj"))

                    raw_pts, raw_nrm = sample_points_and_normals(raw_tm, 100_000)
                    raw_pts, raw_nrm = raw_pts.cuda(), raw_nrm.cuda()
                    _, cd_raw = find_best_rotation_24_with_icp(raw_pts, gt_pts)
                    row["cd_raw"] = cd_raw

                    transform_raw, _ = find_best_rotation_24_with_icp(raw_pts, gt_pts)
                    aligned_raw_pts, aligned_raw_nrm = apply_transform_to_points(raw_pts, raw_nrm, transform_raw)
                    row["nc_raw"] = normal_consistency(aligned_raw_pts, aligned_raw_nrm, gt_pts, gt_nrm)

                    fscores_raw = f_score_multi(aligned_raw_pts, gt_pts)
                    for t in F_SCORE_THRESHOLDS:
                        row[f"fscore_{t}_raw"] = fscores_raw[t]

                    del raw_mesh, raw_meshes, raw_tm, raw_pts, raw_nrm
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  [Raw mesh Error] {uid}: {e}")
                    traceback.print_exc()
                    row["cd_raw"] = float('nan')
                    row["nc_raw"] = float('nan')
                    for t in F_SCORE_THRESHOLDS:
                        row[f"fscore_{t}_raw"] = float('nan')

            except Exception as e:
                print(f"  [Phase B Error] {uid}: {e}")
                traceback.print_exc()
                row["error"] = str(e)

            writer.writerow(row)
            csvfile.flush()
            torch.cuda.empty_cache()

    print(f"Phase B results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Phase C: GT Injection Stage Breakdown
# ---------------------------------------------------------------------------

def _select_phase_c_samples(phase_b_csv, manifest, n_per_tier=33):
    """
    Select 100 objects from Phase B results: 33 per tier,
    spanning CD distribution (11 bottom quartile, 11 median, 11 top quartile).
    """
    import pandas as pd
    df = pd.read_csv(phase_b_csv)

    # Merge tier info from manifest
    manifest_df = pd.DataFrame(manifest)
    if "tier" in manifest_df.columns:
        df = df.merge(manifest_df[["uid", "tier"]], on="uid", how="left", suffixes=("", "_manifest"))
        if "tier_manifest" in df.columns:
            df["tier"] = df["tier"].fillna(df["tier_manifest"])
            df.drop(columns=["tier_manifest"], inplace=True)

    # Filter out failed samples
    df = df[df["cd_filled"].notna() & (df["cd_filled"] != "")]
    df["cd_filled"] = pd.to_numeric(df["cd_filled"], errors="coerce")
    df = df.dropna(subset=["cd_filled"])

    selected_uids = []
    tiers = df["tier"].unique()
    n_per_quartile = n_per_tier // 3

    for tier_val in sorted(tiers):
        tier_df = df[df["tier"] == tier_val].sort_values("cd_filled")
        n = len(tier_df)
        if n == 0:
            continue

        # Bottom quartile (best), median region, top quartile (worst)
        q1_end = max(1, n // 4)
        q2_start = max(1, n // 4)
        q2_end = min(n, 3 * n // 4)
        q3_start = min(n - 1, 3 * n // 4)

        bottom = tier_df.iloc[:q1_end]
        middle = tier_df.iloc[q2_start:q2_end]
        top = tier_df.iloc[q3_start:]

        # Evenly sample from each region
        selected_uids.extend(bottom.head(n_per_quartile)["uid"].tolist())
        if len(middle) > 0:
            step = max(1, len(middle) // n_per_quartile)
            selected_uids.extend(middle.iloc[::step].head(n_per_quartile)["uid"].tolist())
        selected_uids.extend(top.tail(n_per_quartile)["uid"].tolist())

    # Deduplicate while preserving order
    seen = set()
    unique_uids = []
    for uid in selected_uids:
        if uid not in seen:
            seen.add(uid)
            unique_uids.append(uid)

    print(f"Phase C: selected {len(unique_uids)} samples from {len(tiers)} tiers")
    return unique_uids


def _get_gt_structure(mesh_path, grid_size=512):
    """
    Get GT sparse structure coords from mesh via o_voxel.
    Returns coords tensor [N, 4] with batch dim prepended.
    """
    import o_voxel

    tm_mesh = load_and_normalize_mesh(mesh_path)
    vertices = torch.from_numpy(tm_mesh.vertices.copy()).float()
    faces = torch.from_numpy(tm_mesh.faces.copy()).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    # Convert voxel_indices to sparse structure coords format
    # Pipeline uses coords with batch dim: [batch_idx, x, y, z]
    # The sparse structure operates at resolution 32 (ss_res for 512 pipeline)
    # Quantize from grid_size to ss_res=32
    ss_res = 32
    ratio = grid_size // ss_res
    ss_coords = voxel_indices // ratio
    ss_coords_with_batch = torch.cat(
        [torch.zeros(ss_coords.shape[0], 1, dtype=torch.int), ss_coords],
        dim=-1,
    )
    # Deduplicate
    ss_coords_with_batch = ss_coords_with_batch.unique(dim=0)

    return ss_coords_with_batch, voxel_indices, dual_vertices, intersected


def _get_gt_shape_latent(mesh_path, encoder, grid_size=512):
    """
    Encode GT mesh to shape latent via SC-VAE encoder.
    Returns SparseTensor (the latent z).
    """
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    tm_mesh = load_and_normalize_mesh(mesh_path)
    vertices = torch.from_numpy(tm_mesh.vertices.copy()).float()
    faces = torch.from_numpy(tm_mesh.faces.copy()).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    dv_local = dual_vertices * grid_size - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)

    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices],
        dim=-1,
    )
    vertices_st = SparseTensor(feats=dv_local, coords=coords_with_batch)
    intersected_st = vertices_st.replace(intersected.float())

    with torch.no_grad():
        z = encoder(vertices_st.cuda(), intersected_st.cuda())

    return z


def _load_tex_encoder():
    """Load texture/material SC-VAE encoder if available."""
    import trellis2.models as models

    enc_path = "pretrained/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
    if not os.path.exists(f"{enc_path}.json"):
        enc_path = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"

    try:
        tex_encoder = models.from_pretrained(enc_path).eval().cuda()
        print("Texture encoder loaded successfully.")
        return tex_encoder
    except Exception as e:
        print(f"WARNING: Could not load texture encoder: {e}")
        print("Conditions C3 and C4 (GT material) will be skipped.")
        return None


def _get_gt_tex_latent(mesh_path, tex_encoder, shape_slat, pipeline, grid_size=512):
    """
    Get GT texture/material latent.
    This is experimental and may require debugging.

    The texture encoder expects PBR voxel data, which we don't have for
    arbitrary GT meshes. This function attempts a best-effort approach:
    encode the mesh with uniform material properties.

    Returns SparseTensor (tex latent) or None on failure.
    """
    from trellis2.modules.sparse import SparseTensor

    # Use the shape_slat coords to construct a simple material input
    # Material encoder expects [base_color(3), metallic(1), roughness(1), alpha(1)] = 6 channels
    # Use default PBR values: base_color=(0.8, 0.8, 0.8), metallic=0, roughness=0.5, alpha=1.0
    # Normalized to [-1, 1] range (as in encode_pbr_latent.py: feats / 255 * 2 - 1)
    n_voxels = shape_slat.coords.shape[0]
    default_base_color = torch.full((n_voxels, 3), 0.6)  # (0.8 * 2 - 1 = 0.6)
    default_metallic = torch.full((n_voxels, 1), -1.0)    # (0 * 2 - 1 = -1.0)
    default_roughness = torch.full((n_voxels, 1), 0.0)    # (0.5 * 2 - 1 = 0.0)
    default_alpha = torch.full((n_voxels, 1), 1.0)        # (1.0 * 2 - 1 = 1.0)
    feats = torch.cat([default_base_color, default_metallic, default_roughness, default_alpha], dim=-1)

    tex_input = SparseTensor(feats=feats.float(), coords=shape_slat.coords.clone())

    try:
        with torch.no_grad():
            z_tex = tex_encoder(tex_input.cuda())
        return z_tex
    except Exception as e:
        print(f"  WARNING: GT texture encoding failed: {e}")
        traceback.print_exc()
        return None


def _run_dit_with_injection(pipeline, image, condition_name, mesh_path,
                            shape_encoder, tex_encoder,
                            gt_structure_coords=None,
                            gt_shape_latent=None,
                            gt_tex_latent=None):
    """
    Run the DiT pipeline with selective GT injection.

    Replicates the pipeline's run() flow for pipeline_type='512',
    replacing specific stage outputs with GT-derived data.

    Args:
        pipeline: Trellis2ImageTo3DPipeline
        image: PIL Image (RGBA, preprocessed)
        condition_name: name for logging (e.g. "C1", "C2")
        mesh_path: path to GT mesh (for GT data extraction if needed)
        shape_encoder: SC-VAE shape encoder (for GT shape latent)
        tex_encoder: SC-VAE texture encoder (for GT tex latent, may be None)
        gt_structure_coords: pre-computed GT structure coords [N, 4] or None
        gt_shape_latent: pre-computed GT shape latent (SparseTensor) or None
        gt_tex_latent: pre-computed GT tex latent (SparseTensor) or None

    Returns:
        trellis2.representations.MeshWithVoxel or None
    """
    from trellis2.modules.sparse import SparseTensor

    print(f"    [{condition_name}] Starting pipeline with GT injection...")

    # Step 1: Get conditioning
    torch.manual_seed(42)
    cond_512 = pipeline.get_cond([image], 512)

    # Step 2: Sparse structure
    if gt_structure_coords is not None:
        print(f"    [{condition_name}] Using GT structure ({gt_structure_coords.shape[0]} voxels)")
        coords = gt_structure_coords.to(pipeline.device)
    else:
        print(f"    [{condition_name}] Using DiT structure")
        coords = pipeline.sample_sparse_structure(cond_512, 32, num_samples=1)

    # Step 3: Shape latent
    if gt_shape_latent is not None:
        print(f"    [{condition_name}] Using GT shape latent ({gt_shape_latent.feats.shape})")
        shape_slat = gt_shape_latent
        # Ensure coords match - GT shape latent coords might differ from structure coords
        # The shape latent has its own coordinate system from the encoder
    else:
        print(f"    [{condition_name}] Using DiT shape latent")
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models['shape_slat_flow_model_512'],
            coords, {}
        )

    # Step 4: Texture latent
    if gt_tex_latent is not None:
        print(f"    [{condition_name}] Using GT texture latent ({gt_tex_latent.feats.shape})")
        tex_slat = gt_tex_latent
    else:
        print(f"    [{condition_name}] Using DiT texture latent")
        tex_slat = pipeline.sample_tex_slat(
            cond_512, pipeline.models['tex_slat_flow_model_512'],
            shape_slat, {}
        )

    # Step 5: Decode
    torch.cuda.empty_cache()
    try:
        out_mesh = pipeline.decode_latent(shape_slat, tex_slat, 512)
        print(f"    [{condition_name}] Decode successful, {len(out_mesh)} mesh(es)")
        return out_mesh[0]
    except Exception as e:
        print(f"    [{condition_name}] Decode failed: {e}")
        traceback.print_exc()
        return None


def run_phase_c(manifest, output_dir, phase_b_csv, rank=0, world_size=1):
    """Phase C: GT injection experiments on selected samples."""
    results_dir = os.path.join(output_dir, "phase_c", "results")
    meshes_dir = os.path.join(output_dir, "phase_c", "meshes")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(meshes_dir, exist_ok=True)

    # Select samples from Phase B
    selected_uids = _select_phase_c_samples(phase_b_csv, manifest)

    # Filter manifest to selected
    uid_set = set(selected_uids)
    selected_manifest = [item for item in manifest if item["uid"] in uid_set]
    # Preserve Phase C selection order
    uid_to_item = {item["uid"]: item for item in selected_manifest}
    selected_manifest = [uid_to_item[uid] for uid in selected_uids if uid in uid_to_item]

    # Shard for multi-GPU
    if world_size > 1:
        start = len(selected_manifest) * rank // world_size
        end = len(selected_manifest) * (rank + 1) // world_size
        selected_manifest = selected_manifest[start:end]

    print(f"Phase C: processing {len(selected_manifest)} samples (rank {rank}/{world_size})")

    # Load models
    print("Loading SC-VAE shape encoder + decoder...")
    shape_encoder, shape_decoder = load_vae_models()

    print("Loading texture encoder (optional)...")
    tex_encoder = _load_tex_encoder()
    has_tex_encoder = tex_encoder is not None

    print("Loading DiT pipeline...")
    pipeline = load_pipeline()

    # Conditions
    conditions = ["baseline", "C1", "C2"]
    if has_tex_encoder:
        conditions.extend(["C3", "C4"])
    else:
        print("Skipping C3/C4 (no texture encoder available)")

    # Build CSV field names
    cond_metric_cols = []
    for cond in conditions:
        cond_metric_cols.extend([
            f"cd_{cond}", f"nc_{cond}",
        ])
        for t in F_SCORE_THRESHOLDS:
            cond_metric_cols.append(f"fscore_{t}_{cond}")

    fieldnames = ["uid", "category", "tier", "face_count"] + cond_metric_cols + ["error"]

    csv_suffix = f"_rank{rank}" if world_size > 1 else ""
    csv_path = os.path.join(results_dir, f"per_sample{csv_suffix}.csv")

    # Read Phase B results to get best view index for each uid
    import pandas as pd
    phase_b_df = pd.read_csv(phase_b_csv)
    uid_to_best_view = {}
    for _, pb_row in phase_b_df.iterrows():
        try:
            uid_to_best_view[pb_row["uid"]] = int(pb_row["best_view_idx"])
        except (ValueError, KeyError):
            pass

    # Resume support: skip already-processed UIDs
    done_uids = _load_done_uids(csv_path)
    if done_uids:
        print(f"Resuming: {len(done_uids)} already done, {len(selected_manifest) - len(done_uids)} remaining")

    mode = 'a' if done_uids else 'w'
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not done_uids:
            writer.writeheader()

        for item in tqdm(selected_manifest, desc="Phase C"):
            uid = item["uid"]
            if uid in done_uids:
                continue
            mesh_path = item["mesh_path"]
            category = item.get("category", "unknown")
            tier = item.get("tier", "unknown")

            row = {k: "" for k in fieldnames}
            row["uid"] = uid
            row["category"] = category
            row["tier"] = tier

            try:
                # Load GT mesh
                gt_trimesh = load_and_normalize_mesh(mesh_path)
                row["face_count"] = len(gt_trimesh.faces)
                gt_pts, gt_nrm = sample_points_and_normals(gt_trimesh, 100_000)
                gt_pts, gt_nrm = gt_pts.cuda(), gt_nrm.cuda()

                # Get conditioning image (best view from Phase B or first view)
                best_view = uid_to_best_view.get(uid, 0)
                view_path = _get_view_image_path(item, best_view)
                if view_path is None or not os.path.exists(view_path):
                    # Fallback to image_path
                    view_path = item.get("image_path")
                if view_path is None or not os.path.exists(view_path):
                    raise RuntimeError(f"No conditioning image found for {uid}")

                image = Image.open(view_path).convert("RGBA")
                image = pipeline.preprocess_image(image)

                # Pre-compute GT data
                print(f"  [{uid}] Computing GT structure and latents...")
                gt_coords, voxel_indices, dual_vertices, intersected_data = _get_gt_structure(mesh_path, GRID_SIZE)
                gt_shape_latent = _get_gt_shape_latent(mesh_path, shape_encoder, GRID_SIZE)

                gt_tex_latent = None
                if has_tex_encoder:
                    gt_tex_latent = _get_gt_tex_latent(mesh_path, tex_encoder, gt_shape_latent, pipeline, GRID_SIZE)

                # Run each condition
                for cond_name in conditions:
                    print(f"  [{uid}] Running condition: {cond_name}")

                    try:
                        if cond_name == "baseline":
                            # Full DiT, no GT injection
                            gen_mesh = _run_dit_with_injection(
                                pipeline, image, cond_name, mesh_path,
                                shape_encoder, tex_encoder,
                            )
                        elif cond_name == "C1":
                            # GT structure + DiT shape + DiT material
                            gen_mesh = _run_dit_with_injection(
                                pipeline, image, cond_name, mesh_path,
                                shape_encoder, tex_encoder,
                                gt_structure_coords=gt_coords,
                            )
                        elif cond_name == "C2":
                            # GT structure + GT shape + DiT material
                            gen_mesh = _run_dit_with_injection(
                                pipeline, image, cond_name, mesh_path,
                                shape_encoder, tex_encoder,
                                gt_structure_coords=gt_coords,
                                gt_shape_latent=gt_shape_latent,
                            )
                        elif cond_name == "C3":
                            # DiT structure + DiT shape + GT material
                            gen_mesh = _run_dit_with_injection(
                                pipeline, image, cond_name, mesh_path,
                                shape_encoder, tex_encoder,
                                gt_tex_latent=gt_tex_latent,
                            )
                        elif cond_name == "C4":
                            # GT structure + DiT shape + GT material
                            gen_mesh = _run_dit_with_injection(
                                pipeline, image, cond_name, mesh_path,
                                shape_encoder, tex_encoder,
                                gt_structure_coords=gt_coords,
                                gt_tex_latent=gt_tex_latent,
                            )
                        else:
                            continue

                        if gen_mesh is None:
                            raise RuntimeError(f"Condition {cond_name} returned None")

                        # Save mesh
                        try:
                            gen_tm = trellis_mesh_to_trimesh(gen_mesh)
                            gen_tm.export(os.path.join(meshes_dir, f"{uid}_{cond_name}.obj"))
                        except Exception:
                            pass

                        # Evaluate
                        gen_tm = trellis_mesh_to_trimesh(gen_mesh)
                        gen_pts, gen_nrm = sample_points_and_normals(gen_tm, 100_000)
                        gen_pts, gen_nrm = gen_pts.cuda(), gen_nrm.cuda()

                        transform, cd_val = find_best_rotation_24_with_icp(gen_pts, gt_pts)
                        row[f"cd_{cond_name}"] = cd_val

                        aligned_pts, aligned_nrm = apply_transform_to_points(gen_pts, gen_nrm, transform)
                        row[f"nc_{cond_name}"] = normal_consistency(aligned_pts, aligned_nrm, gt_pts, gt_nrm)

                        fscores = f_score_multi(aligned_pts, gt_pts)
                        for t in F_SCORE_THRESHOLDS:
                            row[f"fscore_{t}_{cond_name}"] = fscores[t]

                        del gen_mesh, gen_tm, gen_pts, gen_nrm, aligned_pts, aligned_nrm
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  [{uid}] Condition {cond_name} failed: {e}")
                        traceback.print_exc()
                        row[f"cd_{cond_name}"] = float('nan')
                        row[f"nc_{cond_name}"] = float('nan')
                        for t in F_SCORE_THRESHOLDS:
                            row[f"fscore_{t}_{cond_name}"] = float('nan')
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [Phase C Error] {uid}: {e}")
                traceback.print_exc()
                row["error"] = str(e)

            writer.writerow(row)
            csvfile.flush()
            torch.cuda.empty_cache()

    print(f"Phase C results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Merge Phase: Combine rank CSV files
# ---------------------------------------------------------------------------

def run_merge(output_dir, phase_name):
    """Merge rank CSV files into a single per_sample.csv."""
    import glob as globmod

    results_dir = os.path.join(output_dir, phase_name, "results")
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    rank_files = sorted(globmod.glob(os.path.join(results_dir, "per_sample_rank*.csv")))
    if not rank_files:
        print(f"No rank files found in {results_dir}")
        # Check if single-GPU file exists
        single = os.path.join(results_dir, "per_sample.csv")
        if os.path.exists(single):
            print(f"Single-GPU file already exists: {single}")
        return

    print(f"Merging {len(rank_files)} rank files for phase {phase_name}...")

    merged_path = os.path.join(results_dir, "per_sample.csv")
    all_rows = []
    fieldnames = None

    for rf in rank_files:
        with open(rf, 'r') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for r in reader:
                all_rows.append(r)

    with open(merged_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Merged {len(all_rows)} rows into {merged_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Component-Level Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Phase A (VAE reconstruction):
    python scripts/eval/component_eval.py --phase a --manifest manifest.json

  Phase B (DiT best-of-16, multi-GPU):
    python scripts/eval/component_eval.py --phase b --manifest manifest.json --rank 0 --world_size 8

  Phase C (GT injection):
    python scripts/eval/component_eval.py --phase c --manifest manifest.json --phase_b_csv phase_b/results/per_sample.csv

  Merge rank files:
    python scripts/eval/component_eval.py --phase merge --output_dir experiments/component_eval
        """,
    )
    parser.add_argument("--phase", type=str, required=True,
                        choices=["a", "b", "c", "merge"],
                        help="Pipeline phase: a (VAE), b (DiT), c (GT injection), merge")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to test set manifest.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/component_eval",
                        help="Output directory for results")
    parser.add_argument("--phase_b_csv", type=str, default=None,
                        help="Path to Phase B per_sample.csv (required for Phase C)")
    parser.add_argument("--rank", type=int, default=0,
                        help="Worker rank for multi-GPU (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of workers for multi-GPU")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate")
    parser.add_argument("--grid_size", type=int, default=512,
                        help="O-Voxel grid resolution")
    args = parser.parse_args()

    global GRID_SIZE
    GRID_SIZE = args.grid_size

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase == "merge":
        # Merge all phases that have rank files
        for phase_name in ["phase_a", "phase_b", "phase_c"]:
            run_merge(args.output_dir, phase_name)
        return

    # Load manifest
    if args.manifest is None:
        parser.error("--manifest is required for phases a, b, c")

    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.max_samples:
        manifest = manifest[:args.max_samples]

    # Shard for multi-GPU (Phase A and B)
    if args.phase in ("a", "b") and args.world_size > 1:
        start = len(manifest) * args.rank // args.world_size
        end = len(manifest) * (args.rank + 1) // args.world_size
        manifest = manifest[start:end]
        print(f"Shard: rank {args.rank}, samples {start}-{end} ({len(manifest)} total)")

    if args.phase == "a":
        run_phase_a(manifest, args.output_dir, args.rank, args.world_size)
    elif args.phase == "b":
        run_phase_b(manifest, args.output_dir, args.rank, args.world_size)
    elif args.phase == "c":
        if args.phase_b_csv is None:
            parser.error("--phase_b_csv is required for Phase C")
        run_phase_c(manifest, args.output_dir, args.phase_b_csv, args.rank, args.world_size)


if __name__ == "__main__":
    main()
