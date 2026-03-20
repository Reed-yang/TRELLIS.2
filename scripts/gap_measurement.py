"""
Gap Measurement: Quantify quality gap between SC-VAE reconstruction and DiT generation.

Usage:
    python scripts/gap_measurement.py --manifest experiments/gap_measurement/pilot_data/manifest.json
    python scripts/gap_measurement.py --manifest ... --path_a_only   # VAE reconstruction only
    python scripts/gap_measurement.py --manifest ... --path_b_only   # DiT generation only
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import csv
import torch
import numpy as np
import trimesh
from tqdm import tqdm
from PIL import Image

from scripts.eval_metrics import (
    sample_points_and_normals,
    trellis_mesh_to_trimesh,
    chamfer_distance,
    f_score,
    normal_consistency,
    render_normal_maps,
    compute_rendering_metrics,
)

NUM_SAMPLE_POINTS = 10000
GRID_SIZE = 512
F_SCORE_THRESHOLD = 0.01
RENDER_NVIEWS = 8
RENDER_RESOLUTION = 512


# ---------------------------------------------------------------------------
# Mesh loading and normalization
# ---------------------------------------------------------------------------

def load_and_normalize_mesh(mesh_path):
    """Load a mesh with trimesh, normalize to [-0.5, 0.5]."""
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = mesh.vertices.astype(np.float64)
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    mesh.vertices = (vertices - center) * scale
    return mesh


def trimesh_to_trellis_mesh(tm_mesh):
    """Convert trimesh.Trimesh to trellis2.representations.Mesh (on CUDA)."""
    from trellis2.representations import Mesh as TrellisMesh
    return TrellisMesh(
        vertices=torch.from_numpy(tm_mesh.vertices.copy()).float().cuda(),
        faces=torch.from_numpy(tm_mesh.faces.copy()).int().cuda(),
    )


# ---------------------------------------------------------------------------
# Reference image rendering (headless, using NVDiffRast)
# ---------------------------------------------------------------------------

def render_reference_from_gt(mesh_path, output_path, resolution=512):
    """
    Render a reference image from GT mesh using NVDiffRast (works headless).
    Falls back gracefully if rendering fails.
    """
    try:
        from trellis2.utils.render_utils import render_snapshot

        tm_mesh = load_and_normalize_mesh(mesh_path)
        trellis_mesh = trimesh_to_trellis_mesh(tm_mesh)

        # Render from a good viewpoint (single view, close, wide fov)
        result = render_snapshot(
            trellis_mesh,
            resolution=resolution,
            nviews=1,
            r=2, fov=40,
            return_types=["normal"],
        )

        # Convert normal map to RGB image and save
        normal_img = result["normal"][0]  # [H, W, 3] uint8
        img = Image.fromarray(normal_img)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"  Failed to render reference image: {e}")
        return False


# ---------------------------------------------------------------------------
# Path A: VAE Reconstruction
# ---------------------------------------------------------------------------

def load_vae_models():
    """Load pretrained SC-VAE encoder and decoder."""
    import trellis2.models as models

    # Try local path first, then HF Hub
    enc_path = "pretrained/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    dec_path = "pretrained/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

    # Check if local files exist (with .json extension)
    if not os.path.exists(f"{enc_path}.json"):
        enc_path = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    if not os.path.exists(f"{dec_path}.json"):
        dec_path = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

    encoder = models.from_pretrained(enc_path).eval().cuda()
    decoder = models.from_pretrained(dec_path).eval().cuda()
    decoder.set_resolution(GRID_SIZE)
    return encoder, decoder


def vae_reconstruct(mesh_path, encoder, decoder):
    """
    Path A: GT mesh -> O-Voxel -> SC-VAE encode -> decode -> reconstructed mesh.

    Returns:
        trellis2.representations.Mesh on CUDA, or None on failure
    """
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    # Load and normalize
    tm_mesh = load_and_normalize_mesh(mesh_path)
    vertices = torch.from_numpy(tm_mesh.vertices.copy()).float()
    faces = torch.from_numpy(tm_mesh.faces.copy()).long()

    # Mesh -> O-Voxel
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=GRID_SIZE,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    # Prepare encoder input.
    # Use float directly without uint8 quantization (same as trellis2_texturing.py:211)
    dv_local = dual_vertices * GRID_SIZE - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)

    # Build SparseTensor with batch dim prepended: [batch_idx, x, y, z]
    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices],
        dim=-1,
    )
    vertices_st = SparseTensor(
        feats=dv_local,
        coords=coords_with_batch,
    )
    # intersected from o_voxel.convert is already [N, 3] bool
    intersected_st = vertices_st.replace(intersected.float())

    # Encode -> decode
    with torch.no_grad():
        z = encoder(vertices_st.cuda(), intersected_st.cuda())
        recon_meshes = decoder(z)

    if isinstance(recon_meshes, list):
        return recon_meshes[0]
    return recon_meshes


# ---------------------------------------------------------------------------
# Path B: DiT Generation
# ---------------------------------------------------------------------------

_pipeline = None


def _patch_gated_models():
    """Redirect gated HF models (DINOv3, RMBG) to local pretrained paths."""
    pretrained_dir = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet

    original_dinov3_init = DinoV3FeatureExtractor.__init__
    original_birefnet_init = BiRefNet.__init__

    def patched_dinov3_init(self, model_name, image_size=512):
        if "dinov3" in model_name:
            local_path = os.path.join(pretrained_dir, "dinov3")
            if os.path.exists(local_path):
                model_name = local_path
        original_dinov3_init(self, model_name, image_size)

    def patched_birefnet_init(self, model_name="ZhengPeng7/BiRefNet"):
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        if "RMBG" in model_name or "BiRefNet" in model_name:
            local_path = os.path.join(pretrained_dir, "rmbg2")
            if os.path.exists(local_path):
                model_name = local_path
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=False
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    DinoV3FeatureExtractor.__init__ = patched_dinov3_init
    BiRefNet.__init__ = patched_birefnet_init


def load_pipeline():
    """Load pretrained Trellis2 image-to-3D pipeline (singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    _patch_gated_models()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Try local path first
    model_path = "pretrained/TRELLIS.2-4B"
    if not os.path.exists(os.path.join(model_path, "pipeline.json")):
        model_path = "microsoft/TRELLIS.2-4B"

    _pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    _pipeline.to("cuda")
    return _pipeline


def dit_generate(image_path):
    """
    Path B: test image -> full pipeline -> generated mesh.

    Returns:
        trellis2.representations.Mesh on CUDA, or None on failure
    """
    pipeline = load_pipeline()
    image = Image.open(image_path).convert("RGBA")
    meshes = pipeline.run(image, pipeline_type='512')
    return meshes[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_single(gt_mesh_path, recon_mesh, gen_mesh, gt_trimesh=None):
    """
    Evaluate one sample: compare reconstructed and generated meshes against GT.

    Returns:
        dict with all metrics for both paths
    """
    if gt_trimesh is None:
        gt_trimesh = load_and_normalize_mesh(gt_mesh_path)

    gt_points, gt_normals = sample_points_and_normals(gt_trimesh, NUM_SAMPLE_POINTS)
    gt_points, gt_normals = gt_points.cuda(), gt_normals.cuda()

    result = {}
    gt_normals_maps = None  # cached across Path A and Path B

    # Path A metrics
    if recon_mesh is not None:
        recon_trimesh = trellis_mesh_to_trimesh(recon_mesh)
        recon_points, recon_normals = sample_points_and_normals(recon_trimesh, NUM_SAMPLE_POINTS)
        recon_points, recon_normals = recon_points.cuda(), recon_normals.cuda()
        result["vae_cd"] = chamfer_distance(recon_points, gt_points)
        result["vae_fscore"] = f_score(recon_points, gt_points, threshold=F_SCORE_THRESHOLD)
        result["vae_nc"] = normal_consistency(recon_points, recon_normals, gt_points, gt_normals)

        # Rendering metrics: render GT and recon from same views
        try:
            if gt_normals_maps is None:
                gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                gt_normals_maps = render_normal_maps(gt_trellis, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            recon_normals_maps = render_normal_maps(recon_mesh, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            render_metrics = compute_rendering_metrics(recon_normals_maps, gt_normals_maps)
            result["vae_psnr"] = render_metrics["psnr"]
            result["vae_ssim"] = render_metrics["ssim"]
        except Exception as e:
            print(f"    Rendering metrics failed: {e}")
            result["vae_psnr"] = float('nan')
            result["vae_ssim"] = float('nan')
    else:
        result["vae_cd"] = float('nan')
        result["vae_fscore"] = float('nan')
        result["vae_nc"] = float('nan')
        result["vae_psnr"] = float('nan')
        result["vae_ssim"] = float('nan')

    # Path B metrics
    if gen_mesh is not None:
        gen_trimesh = trellis_mesh_to_trimesh(gen_mesh)
        gen_points, gen_normals = sample_points_and_normals(gen_trimesh, NUM_SAMPLE_POINTS)
        gen_points, gen_normals = gen_points.cuda(), gen_normals.cuda()
        result["dit_cd"] = chamfer_distance(gen_points, gt_points)
        result["dit_fscore"] = f_score(gen_points, gt_points, threshold=F_SCORE_THRESHOLD)
        result["dit_nc"] = normal_consistency(gen_points, gen_normals, gt_points, gt_normals)

        try:
            if gt_normals_maps is None:
                gt_trellis = trimesh_to_trellis_mesh(gt_trimesh)
                gt_normals_maps = render_normal_maps(gt_trellis, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            gen_normals_maps = render_normal_maps(gen_mesh, nviews=RENDER_NVIEWS, resolution=RENDER_RESOLUTION)
            render_metrics = compute_rendering_metrics(gen_normals_maps, gt_normals_maps)
            result["dit_psnr"] = render_metrics["psnr"]
            result["dit_ssim"] = render_metrics["ssim"]
        except Exception as e:
            print(f"    Rendering metrics failed: {e}")
            result["dit_psnr"] = float('nan')
            result["dit_ssim"] = float('nan')
    else:
        result["dit_cd"] = float('nan')
        result["dit_fscore"] = float('nan')
        result["dit_nc"] = float('nan')
        result["dit_psnr"] = float('nan')
        result["dit_ssim"] = float('nan')

    return result


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(results, output_path):
    """Generate summary markdown from per-sample results."""
    metrics = ["cd", "fscore", "nc", "psnr", "ssim"]

    lines = ["# Gap Measurement Results\n"]
    lines.append(f"**Samples evaluated:** {len(results)}\n")
    lines.append(f"**Resolution:** {GRID_SIZE}^3\n")
    lines.append(f"**Points sampled:** {NUM_SAMPLE_POINTS}\n")
    lines.append(f"**F-score threshold:** {F_SCORE_THRESHOLD}\n\n")

    lines.append("## Aggregate Metrics\n")
    lines.append("| Metric | VAE Recon (mean+-std) | DiT Gen (mean+-std) | Gap (DiT - VAE) |")
    lines.append("|--------|----------------------|---------------------|-----------------|\n")

    for m in metrics:
        vae_vals = [r.get(f"vae_{m}", float('nan')) for r in results]
        dit_vals = [r.get(f"dit_{m}", float('nan')) for r in results]
        vae_vals = [v for v in vae_vals if not np.isnan(v)]
        dit_vals = [v for v in dit_vals if not np.isnan(v)]

        if vae_vals:
            vae_str = f"{np.mean(vae_vals):.6f} +- {np.std(vae_vals):.6f}"
        else:
            vae_str = "N/A"
        if dit_vals:
            dit_str = f"{np.mean(dit_vals):.6f} +- {np.std(dit_vals):.6f}"
        else:
            dit_str = "N/A"
        if vae_vals and dit_vals:
            gap = np.mean(dit_vals) - np.mean(vae_vals)
            gap_str = f"{gap:+.6f}"
        else:
            gap_str = "N/A"

        lines.append(f"| {m.upper()} | {vae_str} | {dit_str} | {gap_str} |")

    lines.append("\n## Interpretation\n")
    lines.append("- **CD/F-score gap small** -> VAE is the ceiling, prioritize SC-VAE improvements")
    lines.append("- **CD/F-score gap large** -> DiT is the bottleneck, prioritize DiT optimization")
    lines.append("- **NC gap large but CD gap small** -> DiT struggles with surface normals specifically\n")

    # Decision
    vae_cds = [r.get("vae_cd", float('nan')) for r in results]
    dit_cds = [r.get("dit_cd", float('nan')) for r in results]
    vae_cds = [v for v in vae_cds if not np.isnan(v)]
    dit_cds = [v for v in dit_cds if not np.isnan(v)]
    if vae_cds and dit_cds:
        vae_mean = np.mean(vae_cds)
        dit_mean = np.mean(dit_cds)
        if vae_mean > 0:
            ratio = dit_mean / vae_mean
            lines.append(f"**DiT CD / VAE CD ratio:** {ratio:.2f}x\n")
            if ratio > 2.0:
                lines.append("**-> DECISION: DiT is significantly worse than VAE upper bound. Prioritize DiT optimization.**\n")
            elif ratio > 1.5:
                lines.append("**-> DECISION: Moderate gap. Consider improving both DiT and VAE.**\n")
            else:
                lines.append("**-> DECISION: DiT is close to VAE upper bound. Prioritize SC-VAE improvements.**\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Summary saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gap Measurement: VAE reconstruction vs DiT generation")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to pilot data manifest.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement/results",
                        help="Output directory for results")
    parser.add_argument("--path_a_only", action="store_true",
                        help="Only run Path A (VAE reconstruction)")
    parser.add_argument("--path_b_only", action="store_true",
                        help="Only run Path B (DiT generation)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate")
    parser.add_argument("--grid_size", type=int, default=512,
                        help="O-Voxel grid resolution")
    args = parser.parse_args()

    global GRID_SIZE
    GRID_SIZE = args.grid_size

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vae_reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "dit_generations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "rendered_refs"), exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.max_samples:
        manifest = manifest[:args.max_samples]

    print(f"Evaluating {len(manifest)} samples...")

    # Load models
    encoder, decoder = None, None
    if not args.path_b_only:
        print("Loading SC-VAE encoder + decoder...")
        encoder, decoder = load_vae_models()

    # Process each sample
    all_results = []
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    fieldnames = ["uid", "category",
                  "vae_cd", "vae_fscore", "vae_nc", "vae_psnr", "vae_ssim",
                  "dit_cd", "dit_fscore", "dit_nc", "dit_psnr", "dit_ssim",
                  "vae_error", "dit_error"]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(manifest, desc="Gap Measurement"):
            uid = item["uid"]
            mesh_path = item["mesh_path"]
            image_path = item.get("image_path")
            category = item.get("category", "unknown")

            row = {"uid": uid, "category": category, "vae_error": "", "dit_error": ""}

            # Path A: VAE reconstruction
            recon_mesh = None
            if not args.path_b_only:
                try:
                    recon_mesh = vae_reconstruct(mesh_path, encoder, decoder)
                except Exception as e:
                    print(f"  [Path A Error] {uid}: {e}")
                    row["vae_error"] = str(e)

            # Before Path B: check if image is a gray placeholder, re-render from GT if so
            if not args.path_a_only and image_path:
                try:
                    img = Image.open(image_path)
                    pixels = np.array(img)
                    if pixels.std() < 5:  # nearly uniform = placeholder
                        print(f"  Re-rendering reference image for {uid}...")
                        rendered_path = os.path.join(args.output_dir, "rendered_refs", f"{uid}.png")
                        if render_reference_from_gt(mesh_path, rendered_path):
                            image_path = rendered_path
                except Exception:
                    pass

            # Path B: DiT generation
            gen_mesh = None
            if not args.path_a_only and image_path and os.path.exists(image_path):
                try:
                    gen_mesh = dit_generate(image_path)
                except Exception as e:
                    print(f"  [Path B Error] {uid}: {e}")
                    row["dit_error"] = str(e)

            # Save intermediate meshes for debugging
            if recon_mesh is not None:
                try:
                    recon_tm = trellis_mesh_to_trimesh(recon_mesh)
                    recon_tm.export(os.path.join(args.output_dir, "vae_reconstructions", f"{uid}.obj"))
                except Exception:
                    pass
            if gen_mesh is not None:
                try:
                    gen_tm = trellis_mesh_to_trimesh(gen_mesh)
                    gen_tm.export(os.path.join(args.output_dir, "dit_generations", f"{uid}.obj"))
                except Exception:
                    pass

            # Evaluate
            try:
                metrics = evaluate_single(mesh_path, recon_mesh, gen_mesh)
                row.update(metrics)
            except Exception as e:
                print(f"  [Eval Error] {uid}: {e}")

            all_results.append(row)
            writer.writerow(row)
            csvfile.flush()

            # Free GPU memory
            del recon_mesh, gen_mesh
            torch.cuda.empty_cache()

    # Generate summary
    summary_path = os.path.join(args.output_dir, "summary.md")
    generate_summary(all_results, summary_path)

    # Print quick stats
    print(f"\nResults saved to {csv_path}")
    print(f"Summary saved to {summary_path}")
    vae_successes = sum(1 for r in all_results if not np.isnan(r.get("vae_cd", float('nan'))))
    dit_successes = sum(1 for r in all_results if not np.isnan(r.get("dit_cd", float('nan'))))
    print(f"Path A successes: {vae_successes}/{len(all_results)}")
    print(f"Path B successes: {dit_successes}/{len(all_results)}")


if __name__ == "__main__":
    main()
