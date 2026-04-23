"""Deep-eval on 8 golden assets at i_save cadence.

Executes rank-0-only; other ranks sit on a barrier. Loads the cached
golden NPZs (produced by scripts/coart_build_golden.py), runs
encoder→decoder on each, computes CD/NC/F-score/topology, and logs per-
asset + aggregated mean scalars. Optionally dumps a 4-view normal-map
side-by-side render for the assets in cfg.n_dump_names.

Public API:
    run_deep_eval(encoder, decoder, stats, step, logger, cfg)
        -> Dict[asset_name, Dict[metric, float]]
        Empty dict per asset means that asset failed; outer dict is empty on
        non-master ranks.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

from coart.common.dist_utils import unwrap
from coart.data.stats import denormalize, normalize
from coart.eval.metrics import (
    chamfer_distance,
    compute_topo_metrics,
    f_score_multi,
    normal_consistency,
    sample_surface,
)

_REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
_GOLDEN_LIST = _REPO / "coart" / "eval" / "golden_assets.json"


def _load_golden_manifest():
    with open(_GOLDEN_LIST) as fh:
        return json.load(fh)


def _to_trellis_mesh(mesh):
    """Convert trimesh.Trimesh → trellis2.representations.Mesh on CUDA."""
    import torch as _t
    from trellis2.representations import Mesh as TrellisMesh
    v = _t.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).cuda()
    f = _t.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).cuda()
    return TrellisMesh(vertices=v, faces=f)


def _render_normal_4view_side_by_side(recon_mesh, gt_mesh_path: str) -> np.ndarray:
    """Render 4-view normal maps of (GT | recon) and concatenate into one PNG.

    Returns HxWx3 uint8. If rendering fails, returns a solid gray placeholder
    so the logger doesn't crash. Input `recon_mesh` is a trimesh.Trimesh;
    scripts/eval renderer expects trellis2.representations.Mesh, so convert.
    """
    try:
        import trimesh
        gt_tri = trimesh.load(gt_mesh_path, force="mesh")
        sys.path.insert(0, str(_REPO / "scripts" / "eval"))
        from eval_metrics import render_normal_maps_paper_config
        gt_imgs = render_normal_maps_paper_config(_to_trellis_mesh(gt_tri))
        recon_imgs = render_normal_maps_paper_config(_to_trellis_mesh(recon_mesh))
        # Each img is a CHW float tensor in [0,1]; stack to HW(3) uint8 grid
        def _t2np(t):
            return (t.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        row_gt = np.concatenate([_t2np(t) for t in gt_imgs], axis=1)
        row_recon = np.concatenate([_t2np(t) for t in recon_imgs], axis=1)
        grid = np.concatenate([row_gt, row_recon], axis=0)
        return grid
    except Exception as e:
        print(f"[deep_eval] render failed: {e}", file=sys.stderr)
        return np.full((256, 1024, 3), 128, dtype=np.uint8)


def _one_asset(
    asset: Dict[str, Any],
    encoder,
    decoder,
    stats_mean: torch.Tensor,
    stats_std: torch.Tensor,
    step: int,
    resolution: int,
    n_dump_names: list,
    logger,
) -> Dict[str, float]:
    name = asset["name"]
    try:
        npz_path = _REPO / asset["npz_path"]
        d = np.load(npz_path, allow_pickle=True)
        cube_indices = torch.from_numpy(d["cube_indices"].astype(np.int32)).cuda()
        feats_raw = torch.from_numpy(d["feats"].astype(np.float32)).cuda()
        feats_n = normalize(feats_raw, stats_mean, stats_std)

        from trellis2.modules import sparse as sp
        N = cube_indices.shape[0]
        batch_col = torch.zeros((N, 1), dtype=torch.int32, device="cuda")
        coords_bn = torch.cat([batch_col, cube_indices], dim=1)
        x = sp.SparseTensor(feats=feats_n, coords=coords_bn)

        enc = unwrap(encoder)
        dec = unwrap(decoder)
        was_training = enc.training
        enc.eval(); dec.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = enc(x, sample_posterior=False)
                pred = dec(z)
                pred = pred[0] if isinstance(pred, tuple) else pred
        finally:
            if was_training:
                enc.train(); dec.train()

        feats_pred_raw = denormalize(
            pred.feats.float().cpu(),
            stats_mean.cpu(),
            stats_std.cpu(),
        ).numpy()
        cube_np = cube_indices.cpu().numpy().astype(np.int32)

        sys.path.insert(0, str(_REPO))
        from train_overfit_feat18 import feature_to_mesh
        mesh = feature_to_mesh(feats_pred_raw, cube_np, resolution)
        if (mesh is None
                or getattr(mesh, "faces", None) is None
                or len(mesh.faces) == 0):
            logger.scalar(
                f"deep_eval/online/per_asset/{name}/status_failed", 1.0, step,
            )
            return {}

        pts, nrms = sample_surface(mesh, num_points=100000)
        gt_pts = d["gt_points"]
        gt_nrms = d["gt_normals"]
        cd = chamfer_distance(pts, gt_pts)
        nc = normal_consistency(pts, nrms, gt_pts, gt_nrms)
        fs = f_score_multi(pts, gt_pts, thresholds=[0.005, 0.01, 0.05])
        topo = compute_topo_metrics(mesh)

        metrics = {
            "cd": cd, "nc": nc,
            "f005": fs[0.005], "f01": fs[0.01], "f05": fs[0.05],
            "n_components": topo["n_components"],
            "euler": topo["euler_number"],
            "n_boundary_edges": topo["n_boundary_edges"],
            "is_watertight": topo["is_watertight"],
        }
        for k, v in metrics.items():
            logger.scalar(
                f"deep_eval/online/per_asset/{name}/{k}", float(v), step,
            )

        if name in n_dump_names:
            meta = d["meta"].item()
            img = _render_normal_4view_side_by_side(mesh, meta["local_path_gt"])
            logger.image(
                f"deep_eval/renders/{name}/online", img, step,
            )

        return metrics
    except Exception as e:
        traceback.print_exc()
        logger.scalar(
            f"deep_eval/online/per_asset/{name}/status_failed", 1.0, step,
        )
        return {}
    finally:
        # Defensive: release any CUDA arenas held by the failed asset so the
        # next asset / training step starts from a clean slate.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_deep_eval(
    encoder,
    decoder,
    stats: Dict[str, torch.Tensor],
    step: int,
    logger,
    cfg,
) -> Dict[str, Dict[str, float]]:
    """rank-0-only deep eval; other ranks sit on barrier.

    Returns {asset_name: metric_dict} on rank 0 (empty dict for failed assets);
    returns {} on non-master ranks. The caller uses this dict to feed Watchdog
    (helmet NC for bad-helmet detection).
    """
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) \
        else 0
    _has_dist = dist.is_available() and dist.is_initialized()

    if rank != 0:
        if _has_dist:
            dist.barrier()
        return {}

    # All rank-0 logic wrapped in try/finally so barrier ALWAYS fires, even on
    # unhandled exception (manifest load, aggregation bug, etc). Without this,
    # a single rank-0 bug would deadlock DDP forever.
    results: Dict[str, Dict[str, float]] = {}
    try:
        if not _GOLDEN_LIST.exists():
            print(f"[deep_eval] {_GOLDEN_LIST} missing; skipping deep-eval",
                  file=sys.stderr)
            return {}

        first_step = int(getattr(cfg, "first_deep_eval_step", 0))
        if step < first_step:
            print(f"[deep_eval] step={step} < first_deep_eval_step={first_step}; "
                  f"skipping", file=sys.stderr)
            return {}

        manifest = _load_golden_manifest()
        stats_mean = stats["mean"]
        stats_std = stats["std"]
        for asset in manifest:
            m = _one_asset(
                asset, encoder, decoder, stats_mean, stats_std,
                step, cfg.resolution, cfg.n_dump_names, logger,
            )
            results[asset["name"]] = m

        successful = [m for m in results.values() if m]
        if successful:
            keys = ["cd", "nc", "f005", "f01", "f05",
                    "n_components", "euler", "n_boundary_edges", "is_watertight"]
            for k in keys:
                vals = [m[k] for m in successful if k in m]
                if vals:
                    logger.scalar(
                        f"deep_eval/online/mean/{k}",
                        float(sum(vals) / len(vals)),
                        step,
                    )
            wt_rate = sum(
                1 for m in successful if m.get("is_watertight", 0) >= 0.5
            ) / len(successful)
            logger.scalar(
                "deep_eval/online/mean/watertight_rate",
                float(wt_rate),
                step,
            )
    except Exception as e:
        traceback.print_exc()
        print(f"[deep_eval] run_deep_eval rank-0 crashed at step {step}: {e!r}",
              file=sys.stderr)
    finally:
        if _has_dist:
            dist.barrier()
    return results
