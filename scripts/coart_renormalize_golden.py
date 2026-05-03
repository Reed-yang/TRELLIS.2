"""Re-normalize gt_points/gt_normals in existing coart_golden NPZs in-place.

Why: gt_points were originally sampled from the raw glb without normalization
(see coart_build_golden.py history), while pred mesh and Layer-V baseline
both live in canonical spaces. This produced a ~9-order-of-magnitude blowup
in deep_eval/online/mean/cd on assets whose source glb had non-unit scale
(val_p25 center ~(2491, 937, -54), extent ~1500).

What: for each asset listed in coart/eval/golden_assets.json, reload the
source glb at meta['local_path_gt'], apply EXP-5 normalization, resample
100k points+normals, and atomically rewrite the NPZ. Preserves the
precomputed cube_indices / feats / num_boundary unchanged -- only the
ground-truth point cloud and associated metadata are replaced.

Safety: the original NPZ is moved to
    datasets/coart_golden/.backup_<timestamp>/<name>.npz
before the new file is written. If the source glb is missing or fails to
load, the asset is skipped and the original NPZ is left untouched.

Usage:
    .venv/bin/python scripts/coart_renormalize_golden.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from coart.eval.metrics import compute_topo_metrics, sample_surface  # noqa: E402
from coart.eval.normalization import normalize_mesh_exp5_inplace  # noqa: E402


def _load_mesh_as_trimesh(path: Path) -> trimesh.Trimesh:
    """Load glb/obj; concatenate Scene geometries into a single Trimesh."""
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"{path}: Scene has no Trimesh geometry")
        m = trimesh.util.concatenate(geoms)
    return m


def _renormalize_one(asset: dict, backup_dir: Path) -> dict:
    """Return a status dict. Never raises; reports per-asset failures."""
    name = asset["name"]
    npz_path = REPO / asset["npz_path"]
    gt_mesh_path = Path(asset["local_path_gt"])

    if not npz_path.exists():
        return {"name": name, "status": "skip_no_npz", "path": str(npz_path)}
    if not gt_mesh_path.exists():
        return {"name": name, "status": "skip_no_mesh", "mesh": str(gt_mesh_path)}

    try:
        mesh = _load_mesh_as_trimesh(gt_mesh_path)
    except Exception as e:
        return {"name": name, "status": "load_failed", "err": repr(e)}

    try:
        bbox_min, bbox_max = normalize_mesh_exp5_inplace(mesh)
        pts, nrms = sample_surface(mesh, num_points=100000)
        topo = compute_topo_metrics(mesh)
    except Exception as e:
        return {"name": name, "status": "normalize_failed", "err": repr(e)}

    # Read existing arrays to preserve (only gt_points / gt_normals / gt_topo / meta change).
    with np.load(npz_path, allow_pickle=True) as d:
        old_meta = d["meta"].item() if "meta" in d.files else {}
        payload = {
            "cube_indices": d["cube_indices"],
            "feats": d["feats"],
            "num_boundary": d["num_boundary"],
        }

    new_meta = {
        **old_meta,
        "asset_name": name,
        "local_path_gt": str(gt_mesh_path),
        "bbox_min_raw": bbox_min.tolist(),
        "bbox_max_raw": bbox_max.tolist(),
        "normalization": "exp5",
    }

    # Backup then atomically replace. np.savez_compressed auto-appends ".npz"
    # to the output path, so use a tmp stem that ends up as <name>.tmp.npz
    # and then os.replace() to the target.
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npz_path, backup_dir / npz_path.name)

    tmp_stem = npz_path.with_suffix("")  # strips .npz -> <dir>/<name>
    tmp_stem_str = str(tmp_stem) + ".tmp"
    tmp_path = Path(tmp_stem_str + ".npz")  # what savez_compressed actually writes
    np.savez_compressed(
        tmp_stem_str,
        cube_indices=payload["cube_indices"],
        feats=payload["feats"],
        num_boundary=payload["num_boundary"],
        gt_points=pts,
        gt_normals=nrms,
        gt_topo=np.array(topo, dtype=object),
        meta=np.array(new_meta, dtype=object),
    )
    os.replace(tmp_path, npz_path)

    return {
        "name": name,
        "status": "ok",
        "range_x": (float(pts[:, 0].min()), float(pts[:, 0].max())),
        "range_y": (float(pts[:, 1].min()), float(pts[:, 1].max())),
        "range_z": (float(pts[:, 2].min()), float(pts[:, 2].max())),
        "bbox_raw_extent": (bbox_max - bbox_min).tolist(),
    }


def main():
    asset_list_path = REPO / "coart" / "eval" / "golden_assets.json"
    if not asset_list_path.exists():
        print(f"[renormalize] ERROR: {asset_list_path} missing", file=sys.stderr)
        sys.exit(1)

    with open(asset_list_path) as fh:
        assets = json.load(fh)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = REPO / "datasets" / "coart_golden" / f".backup_{stamp}"
    print(f"[renormalize] {len(assets)} assets, backup -> {backup_dir}")

    summary = []
    for a in assets:
        r = _renormalize_one(a, backup_dir)
        summary.append(r)
        if r["status"] == "ok":
            print(f"  OK   {r['name']:15s} "
                  f"x[{r['range_x'][0]:+.3f},{r['range_x'][1]:+.3f}] "
                  f"y[{r['range_y'][0]:+.3f},{r['range_y'][1]:+.3f}] "
                  f"z[{r['range_z'][0]:+.3f},{r['range_z'][1]:+.3f}] "
                  f"raw_extent={[round(v, 2) for v in r['bbox_raw_extent']]}")
        else:
            print(f"  SKIP {r['name']:15s} status={r['status']}  {r.get('err', r.get('mesh', ''))}")

    ok = sum(1 for r in summary if r["status"] == "ok")
    print(f"[renormalize] done: {ok}/{len(summary)} assets renormalized; "
          f"backups in {backup_dir if ok else '(not created)'}")


if __name__ == "__main__":
    main()
