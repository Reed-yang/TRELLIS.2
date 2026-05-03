"""Build 8 golden assets for coart.vae deep-eval.

Produces:
    datasets/coart_golden/<asset_name>.npz
        - cube_indices (N,3) int16 (from precompute_feat18)
        - feats (N,18) float16
        - num_boundary int8
        - gt_points (100000,3) float32
        - gt_normals (100000,3) float32
        - gt_topo (dict via np.savez as 0-d object)
        - meta (dict: asset_name, sha, rank, tier, local_path_gt)

    coart/eval/golden_assets.json
        [{"name": "helmet", "sha": null, "rank": -1, "tier": -1,
          "local_path_gt": "...glb",
          "npz_path": "datasets/coart_golden/helmet.npz"}, ...]

Usage:
    .venv/bin/python scripts/coart_build_golden.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import trimesh

REPO = Path("/mnt/novita2/siyuan/workspace/TRELLIS.2")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from coart.data.feat18_dataset import _is_val_sha  # noqa: E402
from coart.eval.metrics import compute_topo_metrics, sample_surface  # noqa: E402
from coart.eval.normalization import normalize_mesh_exp5_inplace  # noqa: E402


def _percentile_indices(n: int, pcts: List[float]) -> List[int]:
    return [min(int(round(p / 100.0 * (n - 1))), n - 1) for p in pcts]


def _intersect_val_with_feat18(
    full_ranked_csv: Path,
    feat18_data_dir: Path,
    val_split_mod: int,
) -> List[dict]:
    feat18_shas = {
        os.path.splitext(f)[0]
        for f in os.listdir(feat18_data_dir) if f.endswith(".npz")
    }
    out = []
    with open(full_ranked_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sha = row["sha256"]
            if sha not in feat18_shas:
                continue
            if not _is_val_sha(sha, val_split_mod):
                continue
            out.append(row)
    out.sort(key=lambda r: int(r["rank"]))
    return out


def _make_triple_sphere_mesh() -> trimesh.Trimesh:
    meshes = []
    for r in (0.4, 0.7, 1.0):
        s = trimesh.creation.icosphere(subdivisions=4, radius=r)
        meshes.append(s)
    return trimesh.util.concatenate(meshes)


def _gt_cache_and_save(
    asset_name: str,
    feat18_src_npz: Path,
    gt_mesh_path: Path,
    out_npz: Path,
    meta_extra: dict,
) -> None:
    d = np.load(feat18_src_npz)
    mesh = trimesh.load(gt_mesh_path, force="mesh")
    # Handle Scene (multi-geometry glb) by merging into a single Trimesh; some
    # Objaverse assets (e.g. val_p25) load as Scene and without this the
    # subsequent vertex operations fail.
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"{gt_mesh_path}: Scene has no Trimesh geometry")
        mesh = trimesh.util.concatenate(geoms)
    # Canonicalize gt to EXP-5 [-0.5, 0.5]^3 space BEFORE sampling. Matches
    # Layer-V baseline space (golden_baseline.json) and matches the pred-space
    # transform applied in coart/eval/deep_eval.py.
    bbox_min, bbox_max = normalize_mesh_exp5_inplace(mesh)
    pts, nrms = sample_surface(mesh, num_points=100000)
    topo = compute_topo_metrics(mesh)
    meta = {
        "asset_name": asset_name,
        "local_path_gt": str(gt_mesh_path),
        "bbox_min_raw": bbox_min.tolist(),
        "bbox_max_raw": bbox_max.tolist(),
        "normalization": "exp5",
        **meta_extra,
    }
    np.savez_compressed(
        out_npz,
        cube_indices=d["cube_indices"],
        feats=d["feats"],
        num_boundary=d["num_boundary"],
        gt_points=pts,
        gt_normals=nrms,
        gt_topo=np.array(topo, dtype=object),
        meta=np.array(meta, dtype=object),
    )
    print(f"[build_golden] wrote {out_npz} (N_cubes={len(d['cube_indices'])}, "
          f"GT components={topo['n_components']}, "
          f"gt range x[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}])")


def _precompute_feat18_via_csv(
    mesh_path: Path, out_npz: Path, resolution: int,
) -> None:
    """Invoke precompute_feat18.py on one mesh using a tmp CSV.

    precompute_feat18.py expects --dataset_root + --metadata_csv(sha256,local_path).
    We build a 1-row CSV where local_path is relative to mesh_path's parent.
    """
    dataset_root = mesh_path.parent
    rel_path = mesh_path.name
    sha = hashlib.sha256(str(mesh_path).encode()).hexdigest()
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write("sha256,local_path\n")
        fh.write(f"{sha},{rel_path}\n")
        tmp_csv = fh.name
    tmp_outdir = out_npz.parent / f"_feat18_tmp_{asset_tag(mesh_path)}"
    tmp_outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "precompute_feat18.py"),
        "--dataset_root", str(dataset_root),
        "--metadata_csv", tmp_csv,
        "--resolution", str(resolution),
        "--out_dir", str(tmp_outdir),
        "--rank", "0", "--world_size", "1",
    ]
    print(f"[build_golden] precompute: {mesh_path.name} -> {tmp_outdir}")
    subprocess.run(cmd, check=True)
    gen = tmp_outdir / "data" / f"{sha}.npz"
    if not gen.exists():
        raise RuntimeError(
            f"precompute_feat18 did not produce {gen}; check {tmp_outdir}"
        )
    shutil.move(str(gen), str(out_npz))
    shutil.rmtree(tmp_outdir, ignore_errors=True)
    os.unlink(tmp_csv)


def asset_tag(path: Path) -> str:
    return path.stem[:24].replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab")
    ap.add_argument("--feat18_data_dir",
                    default=str(REPO / "datasets" / "ObjaverseXL_sketchfab" /
                                "feat18_512" / "data"))
    ap.add_argument("--full_ranked_csv",
                    default=str(REPO / "scripts" / "preprocess-by-rank" /
                                "out" / "full_ranked.csv"))
    ap.add_argument("--sketchfab_hard_root",
                    default=str(REPO / "datasets" / "sketchfab_hard"))
    ap.add_argument("--out_dir",
                    default=str(REPO / "datasets" / "coart_golden"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--val_split_mod", type=int, default=200)
    ap.add_argument("--percentiles", nargs="+", type=float,
                    default=[10, 25, 40, 60, 80, 95])
    ap.add_argument("--skip_precompute", action="store_true",
                    help="reuse any existing helmet/triple_sphere feat18 npz "
                         "under out_dir instead of re-running precompute_feat18")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset_list = []

    # 1. helmet — search sketchfab_hard recursively
    helmet_root = Path(args.sketchfab_hard_root)
    helmet_candidates = (list(helmet_root.glob("**/*helmet*.glb"))
                         + list(helmet_root.glob("**/*helmet*.obj"))
                         + list(helmet_root.glob("**/*nasal*.glb")))
    if not helmet_candidates:
        print(f"[build_golden] WARNING: no helmet mesh under {helmet_root}",
              file=sys.stderr)
    else:
        helmet_mesh = helmet_candidates[0]
        helmet_npz = out / "helmet.npz"
        if helmet_npz.exists() and args.skip_precompute:
            print(f"[build_golden] helmet: reusing {helmet_npz}")
        else:
            tmp_feat18 = out / "_helmet_feat18.npz"
            _precompute_feat18_via_csv(helmet_mesh, tmp_feat18, args.resolution)
            _gt_cache_and_save(
                "helmet", tmp_feat18, helmet_mesh, helmet_npz,
                {"sha": None, "rank": -1, "tier": -1},
            )
            os.remove(tmp_feat18)
        asset_list.append({
            "name": "helmet",
            "sha": None, "rank": -1, "tier": -1,
            "local_path_gt": str(helmet_mesh),
            "npz_path": str(helmet_npz.relative_to(REPO)),
        })

    # 2. triple_sphere (synthetic)
    tri_mesh_path = out / "triple_sphere.glb"
    tri_npz = out / "triple_sphere.npz"
    if not (tri_npz.exists() and args.skip_precompute):
        tri_mesh = _make_triple_sphere_mesh()
        tri_mesh.export(tri_mesh_path)
        tmp_feat18 = out / "_triple_sphere_feat18.npz"
        _precompute_feat18_via_csv(tri_mesh_path, tmp_feat18, args.resolution)
        _gt_cache_and_save(
            "triple_sphere", tmp_feat18, tri_mesh_path, tri_npz,
            {"sha": None, "rank": -1, "tier": -1},
        )
        os.remove(tmp_feat18)
    else:
        print(f"[build_golden] triple_sphere: reusing {tri_npz}")
    asset_list.append({
        "name": "triple_sphere",
        "sha": None, "rank": -1, "tier": -1,
        "local_path_gt": str(tri_mesh_path),
        "npz_path": str(tri_npz.relative_to(REPO)),
    })

    # 3-8. Six val-split assets at percentile spread
    candidates = _intersect_val_with_feat18(
        Path(args.full_ranked_csv),
        Path(args.feat18_data_dir),
        args.val_split_mod,
    )
    print(f"[build_golden] {len(candidates)} val-split feat18 candidates; "
          f"picking at percentiles {args.percentiles}")
    idxs = _percentile_indices(len(candidates), args.percentiles)
    for p, i in zip(args.percentiles, idxs):
        row = candidates[i]
        sha = row["sha256"]
        name = f"val_p{int(p)}"
        src_npz = Path(args.feat18_data_dir) / f"{sha}.npz"
        gt_mesh_path = Path(args.data_root) / row["local_path"]
        out_npz = out / f"{name}.npz"
        if out_npz.exists() and args.skip_precompute:
            print(f"[build_golden] {name}: reusing {out_npz}")
        else:
            _gt_cache_and_save(
                name, src_npz, gt_mesh_path, out_npz,
                {"sha": sha, "rank": int(row["rank"]),
                 "tier": int(row["tier"])},
            )
        asset_list.append({
            "name": name, "sha": sha,
            "rank": int(row["rank"]), "tier": int(row["tier"]),
            "local_path_gt": str(gt_mesh_path),
            "npz_path": str(out_npz.relative_to(REPO)),
        })

    json_path = REPO / "coart" / "eval" / "golden_assets.json"
    with open(json_path, "w") as fh:
        json.dump(asset_list, fh, indent=2)
    print(f"[build_golden] wrote {json_path} ({len(asset_list)} assets)")


if __name__ == "__main__":
    main()
