"""
Baseline Experiments Orchestrator.

Runs shared O-Voxel encoding + VAE inference for each (model, resolution),
caches intermediates, then dispatches to per-experiment analysis.

Each (model, resolution) job runs in a subprocess with isolated CUDA context
(following ovoxel_repr_test.py pattern).

Usage:
    python scripts/eval/baseline_experiments.py --step 1                    # EXP-5 Layer R only
    python scripts/eval/baseline_experiments.py --step 2                    # VAE inference + cache
    python scripts/eval/baseline_experiments.py --step 3                    # EXP-1/2/3/5-V from cache
    python scripts/eval/baseline_experiments.py --step 4                    # EXP-4 CoReP (elastic)
    python scripts/eval/baseline_experiments.py --step all                  # Everything
    python scripts/eval/baseline_experiments.py --step summary              # Generate summary.md
    python scripts/eval/baseline_experiments.py --step all --num-gpus 8     # 8-GPU parallel
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# HuggingFace cache lives in pretrained/ (shared novita storage)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(
    os.path.dirname(__file__), '..', '..', 'pretrained'))

import json
import time
import argparse
import subprocess
import torch
import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOLUTIONS = [512, 1024]
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
OUTPUT_ROOT = "results/baseline_experiments"
NUM_SAMPLE_POINTS = 100_000
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python')

# Sample registry
SAMPLES = {
    "helmet": {
        "source": "datasets/sketchfab_hard/early_medieval_nasal_helmet.glb",
    },
    "bugatti": {
        "source": "datasets/sketchfab_hard/bugatti-eb110-super-sport-1992-by-alexka.zip",
    },
    "spacesuit": {
        "source": "datasets/sketchfab_hard/franz-viehbocks-sokol-space-suit.zip",
    },
    "bowl": {
        "source": "results/baseline_experiments/data/bowl.obj",
    },
    "parallel_planes": {
        "source_template": "results/baseline_experiments/data/parallel_planes_d2.0_res{res}.obj",
    },
    "nested_spheres": {
        "source": "results/baseline_experiments/data/nested_spheres.obj",
    },
    "icosphere": {
        "source": "results/baseline_experiments/data/icosphere.obj",
    },
}


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_gt_mesh(model_id, resolution=None):
    """Load and normalize a GT mesh to [-0.5, 0.5]."""
    info = SAMPLES[model_id]
    if "source_template" in info:
        source = info["source_template"].format(res=resolution)
    else:
        source = info["source"]

    # Reuse ovoxel_repr_test loading logic for zip/glb files
    if source.endswith('.zip'):
        from scripts.eval.ovoxel_repr_test import _extract_mesh_from_zip
        source = _extract_mesh_from_zip(source, model_id)

    loaded = trimesh.load(source)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    vertices = mesh.vertices.astype(np.float64)
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    mesh.vertices = (vertices - center) * scale
    return mesh


# ---------------------------------------------------------------------------
# VAE model loading
# ---------------------------------------------------------------------------

def load_vae_models(resolution=512):
    """Load pretrained SC-VAE encoder and decoder."""
    import trellis2.models as models

    enc_path = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    dec_path = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

    encoder = models.from_pretrained(enc_path).eval().cuda()
    decoder = models.from_pretrained(dec_path).eval().cuda()
    decoder.set_resolution(resolution)
    return encoder, decoder


# ---------------------------------------------------------------------------
# Subprocess worker dispatch
# ---------------------------------------------------------------------------

def _dispatch_worker(worker_name, model_id, resolution, gpu_id, extra_args=None):
    """Run a worker function in subprocess with isolated CUDA context."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_HUB_CACHE"] = os.path.join(PROJECT_ROOT, "pretrained")

    cmd = [
        PYTHON, os.path.abspath(__file__),
        "--worker", worker_name,
        "--model-id", model_id,
        "--resolution", str(resolution),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=900, cwd=PROJECT_ROOT,
        )
        # Parse JSON result from stdout (last line)
        for line in reversed(result.stdout.strip().split('\n')):
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"model_id": model_id, "resolution": resolution,
                "error": f"No JSON. stderr: {result.stderr[-500:] if result.stderr else 'none'}"}
    except subprocess.TimeoutExpired:
        return {"model_id": model_id, "resolution": resolution, "error": "Timeout (900s)"}
    except Exception as e:
        return {"model_id": model_id, "resolution": resolution, "error": str(e)}


def run_parallel(worker_name, models, resolutions, num_gpus, extra_args=None):
    """Dispatch all (model, resolution) jobs in parallel across GPUs."""
    jobs = [(m, r) for m in models for r in resolutions]
    print(f"\nDispatching {len(jobs)} jobs [{worker_name}] across {num_gpus} GPUs...")

    results = []
    for batch_start in range(0, len(jobs), num_gpus):
        batch = jobs[batch_start:batch_start + num_gpus]
        procs = []
        for i, (model_id, res) in enumerate(batch):
            gpu_id = i % num_gpus
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["HF_HUB_CACHE"] = os.path.join(PROJECT_ROOT, "pretrained")
            cmd = [
                PYTHON, os.path.abspath(__file__),
                "--worker", worker_name,
                "--model-id", model_id,
                "--resolution", str(res),
            ]
            if extra_args:
                cmd.extend(extra_args)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, cwd=PROJECT_ROOT,
            )
            procs.append((model_id, res, proc))

        # Wait for batch
        for model_id, res, proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=900)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                results.append({"model_id": model_id, "resolution": res, "error": "Timeout"})
                print(f"  [{model_id}@{res}] TIMEOUT")
                continue

            parsed = None
            for line in reversed(stdout.strip().split('\n')):
                if line.startswith('{'):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if parsed is None:
                err = stderr[-300:].strip() if stderr else "no output"
                parsed = {"model_id": model_id, "resolution": res, "error": err}

            if parsed.get("error"):
                print(f"  [{model_id}@{res}] FAILED: {parsed['error'][:100]}")
            else:
                print(f"  [{model_id}@{res}] OK")
            results.append(parsed)

    return results


# ---------------------------------------------------------------------------
# Worker: VAE cache
# ---------------------------------------------------------------------------

def worker_vae_cache(model_id, resolution):
    """Run O-Voxel + VAE inference, save all intermediates to cache."""
    import warnings
    warnings.filterwarnings("ignore")

    import o_voxel
    import torch.nn.functional as F
    from trellis2.modules.sparse import SparseTensor
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

    gt_mesh = load_gt_mesh(model_id, resolution)
    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    os.makedirs(cache_dir, exist_ok=True)
    gt_mesh.export(os.path.join(cache_dir, "gt.obj"))

    # O-Voxel encode (CPU)
    vertices = torch.from_numpy(gt_mesh.vertices.copy()).float()
    faces = torch.from_numpy(gt_mesh.faces.copy()).long()
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=resolution, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )

    # Save QEF output
    torch.save({
        'coords': voxel_indices,
        'dual_vertices': dual_vertices,
        'intersected': intersected,
    }, os.path.join(cache_dir, "qef.pt"))

    # Prepare SparseTensor for VAE
    dv_local = dual_vertices * resolution - voxel_indices.float()
    dv_local = torch.clamp(dv_local, 0, 1)
    coords_with_batch = torch.cat(
        [torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int), voxel_indices], dim=-1,
    )
    vertices_st = SparseTensor(feats=dv_local, coords=coords_with_batch)
    intersected_st = vertices_st.replace(intersected.float())

    # VAE encode → decode (get raw 7ch before activation)
    encoder, decoder = load_vae_models(resolution)

    with torch.no_grad():
        z = encoder(vertices_st.cuda(), intersected_st.cuda())

        # Get raw 7ch output by calling parent class forward
        h_raw = SparseUnetVaeDecoder.forward(decoder, z)

        # Apply activations
        margin = decoder.voxel_margin
        dec_verts = (1 + 2 * margin) * torch.sigmoid(h_raw.feats[..., 0:3]) - margin
        dec_intersected_logits = h_raw.feats[..., 3:6]
        dec_intersected = (dec_intersected_logits > 0).float()
        dec_split_weight = F.softplus(h_raw.feats[..., 6:7])

    # Save decoder output
    torch.save({
        'dec_verts': dec_verts.cpu(),
        'dec_intersected_logits': dec_intersected_logits.cpu(),
        'dec_intersected': dec_intersected.cpu(),
        'dec_split_weight': dec_split_weight.cpu(),
        'coords': h_raw.coords.cpu(),
        'voxel_margin': margin,
    }, os.path.join(cache_dir, "decoder.pt"))

    print(json.dumps({"model_id": model_id, "resolution": resolution,
                       "n_voxels": int(voxel_indices.shape[0]), "status": "ok"}))


# ---------------------------------------------------------------------------
# Worker: Layer R (O-Voxel roundtrip, no VAE)
# ---------------------------------------------------------------------------

def worker_layer_r(model_id, resolution):
    """Layer R: O-Voxel roundtrip — no VAE needed."""
    import warnings
    warnings.filterwarnings("ignore")
    from scripts.eval.baseline_exp5_metrics import run_layer_r
    result = run_layer_r(model_id, resolution)
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Worker: Run all experiments from cache
# ---------------------------------------------------------------------------

def worker_experiments(model_id, resolution):
    """Run EXP-1/2/3/5-V from cached VAE output."""
    import warnings
    warnings.filterwarnings("ignore")

    cache_dir = os.path.join(OUTPUT_ROOT, "cache", f"{model_id}_{resolution}")
    if not os.path.exists(os.path.join(cache_dir, "decoder.pt")):
        print(json.dumps({"model_id": model_id, "resolution": resolution,
                          "error": "No VAE cache"}))
        return

    from scripts.eval.baseline_exp1_topology import run_exp1
    from scripts.eval.baseline_exp2_ablation import run_exp2
    from scripts.eval.baseline_exp3_tricks import run_exp3
    from scripts.eval.baseline_exp5_metrics import run_layer_v

    results = {}
    try:
        results['exp1'] = run_exp1(model_id, resolution)
    except Exception as e:
        print(f"  EXP-1 failed: {e}", file=sys.stderr)

    try:
        results['layer_v'] = run_layer_v(model_id, resolution)
    except Exception as e:
        print(f"  Layer V failed: {e}", file=sys.stderr)

    try:
        results['exp2'] = run_exp2(model_id, resolution)
    except Exception as e:
        print(f"  EXP-2 failed: {e}", file=sys.stderr)

    try:
        results['exp3'] = run_exp3(model_id, resolution)
    except Exception as e:
        print(f"  EXP-3 failed: {e}", file=sys.stderr)

    print(json.dumps({"model_id": model_id, "resolution": resolution, "status": "ok"}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline Experiments Orchestrator")
    parser.add_argument("--step", default="all",
                        choices=["1", "2", "3", "4", "6", "all", "summary"])
    parser.add_argument("--models", nargs="+", default=list(SAMPLES.keys()))
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    parser.add_argument("--num-gpus", type=int, default=8)
    # Worker args (internal)
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()

    # Worker mode (subprocess)
    if args.worker:
        if args.worker == "vae_cache":
            worker_vae_cache(args.model_id, args.resolution)
        elif args.worker == "layer_r":
            worker_layer_r(args.model_id, args.resolution)
        elif args.worker == "experiments":
            worker_experiments(args.model_id, args.resolution)
        return

    # Dispatcher mode
    models = [m for m in args.models if m in SAMPLES]
    step = args.step

    # Step 1: EXP-5 Layer R
    if step in ("1", "all"):
        print("=" * 60)
        print("STEP 1: EXP-5 Layer R (O-Voxel roundtrip, no VAE)")
        print("=" * 60)
        run_parallel("layer_r", models, args.resolutions, args.num_gpus)

    # Step 2: VAE inference + cache
    if step in ("2", "all"):
        print("\n" + "=" * 60)
        print("STEP 2: VAE Inference + Cache")
        print("=" * 60)
        run_parallel("vae_cache", models, args.resolutions, args.num_gpus)

    # Step 3: EXP-1/2/3/5-V from cache
    if step in ("3", "all"):
        print("\n" + "=" * 60)
        print("STEP 3: EXP-1 + EXP-5 Layer V + EXP-2 + EXP-3 (from cache)")
        print("=" * 60)
        run_parallel("experiments", models, args.resolutions, args.num_gpus)

    # Step 4: EXP-4 CoReP (elastic, no GPU needed)
    if step in ("4", "all"):
        print("\n" + "=" * 60)
        print("STEP 4: EXP-4 CoReP Diagnosis (elastic)")
        print("=" * 60)
        from scripts.eval.baseline_exp4_corep import run_exp4
        for model_id in models:
            for res in args.resolutions:
                try:
                    run_exp4(model_id, res)
                except Exception as e:
                    print(f"  [{model_id}@{res}] CoReP failed: {e}")

    # Step 6: EXP-6 CoReP Full Reconstruction (elastic, no GPU needed)
    if step in ("6", "all"):
        print("\n" + "=" * 60)
        print("STEP 6: EXP-6 CoReP Full Reconstruction (Layer C)")
        print("=" * 60)
        from scripts.eval.baseline_exp6_corep_recon import run_exp6
        for model_id in models:
            for res in args.resolutions:
                try:
                    run_exp6(model_id, res)
                except Exception as e:
                    import traceback
                    print(f"  [{model_id}@{res}] CoReP recon failed: {e}")
                    traceback.print_exc()

    # Summary
    if step in ("summary", "all"):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        from scripts.eval.baseline_summary import generate_summary
        generate_summary()


if __name__ == "__main__":
    main()
