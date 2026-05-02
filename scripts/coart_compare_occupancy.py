#!/usr/bin/env python
"""SS-flow occupancy IoU validation: corep_fast voxelize vs original o_voxel,
on golden assets, at res=64/32/16. Decision gate per spec section 6.2.

Usage:
  python scripts/coart_compare_occupancy.py \
    --golden_dir datasets/coart_golden \
    --resolutions 64,32,16 \
    --out logs/findings_ss_flow_iou.md
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys
from typing import Optional

# Make corep_fast / o_voxel importable when invoked via `python scripts/foo.py`.
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import trimesh


def downsample_cubes(cubes: np.ndarray, factor: int) -> np.ndarray:
    """Max-pool 3D occupancy to lower resolution. cubes: (N, 3) int. Returns
    deduplicated downsampled coords."""
    pooled = (cubes.astype(np.int64) // factor).astype(np.int32)
    return np.unique(pooled, axis=0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Set IoU on (N,3) int cube coords."""
    sa = {tuple(r) for r in a}
    sb = {tuple(r) for r in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def voxelize_corep(mesh_path: str, resolution: int) -> np.ndarray:
    """Return (N, 3) int corep cube indices in [0, resolution)^3.

    Verified API (Step A):
      corep_fast.stages.s1_voxelize.s1_voxelize(mesh: MeshTensors,
          resolution: int, device: torch.device) -> CubeBatch
      MeshTensors.from_trimesh(mesh, resolution, device='cpu')
    """
    import torch
    # VERIFY: import inside fn so tests can patch / skip if extension missing.
    from corep_fast.stages import s1_voxelize as s1_mod
    from corep_fast.containers import MeshTensors

    m = trimesh.load(mesh_path, force="mesh")
    # VERIFY: MeshTensors.from_trimesh signature is (mesh, resolution, device='cpu').
    # corep_fast normalizes vertices internally to [0,1]^3 (per s1_voxelize docstring),
    # so caller does NOT need to pre-center / pre-scale the trimesh here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = MeshTensors.from_trimesh(m, resolution=resolution, device=device)
    # VERIFY: top-level callable is `s1_voxelize.s1_voxelize` (module + function same name).
    cb = s1_mod.s1_voxelize(mt, resolution, device)
    # VERIFY: CubeBatch.cube_indices is (N, 3) int32 per CubeBatch dataclass docstring.
    cubes = cb.cube_indices
    return cubes.detach().cpu().numpy().astype(np.int32)


def voxelize_native(mesh_path: str, resolution: int) -> np.ndarray:
    """Return (N, 3) int original (o_voxel) surface cube indices.

    Verified API (Step A):
      o_voxel.convert.flexible_dual_grid.mesh_to_flexible_dual_grid(
          vertices, faces, voxel_size=None, grid_size=None, aabb=None, ...)
        -> (coords, dual_vertices, intersected_flag)
    """
    import torch
    from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid

    m = trimesh.load(mesh_path, force="mesh")
    # native pipeline expects mesh in [-0.5, 0.5]^3 (matches o_voxel default conventions)
    extent = float((m.vertices.max(0) - m.vertices.min(0)).max())
    if extent <= 0:
        return np.zeros((0, 3), dtype=np.int32)
    scale = 0.99999 / extent
    center = (m.vertices.max(0) + m.vertices.min(0)) / 2
    verts_np = (np.asarray(m.vertices, dtype=np.float32) - center) * scale  # → [-0.5, 0.5]
    faces_np = np.asarray(m.faces, dtype=np.int32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts = torch.from_numpy(verts_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    # VERIFY: pass grid_size=resolution (int broadcasts to [R,R,R]); aabb=[-0.5,0.5]^3.
    aabb = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    out = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=resolution, aabb=aabb,
    )
    # VERIFY: return tuple is (coords, dual_vertices, intersected_flag); coords is (N, 3).
    coords = out[0] if isinstance(out, (tuple, list)) else out
    return coords.detach().cpu().numpy().astype(np.int32)


def compare_one(mesh_path: str, resolutions: list[int]) -> dict:
    out = {"asset": os.path.basename(mesh_path)}
    base_res = max(resolutions)
    cubes_corep = voxelize_corep(mesh_path, base_res)
    cubes_native = voxelize_native(mesh_path, base_res)
    for res in resolutions:
        factor = base_res // res
        a = downsample_cubes(cubes_corep, factor) if factor > 1 else cubes_corep
        b = downsample_cubes(cubes_native, factor) if factor > 1 else cubes_native
        sa = {tuple(r) for r in a}
        sb = {tuple(r) for r in b}
        out[f"iou_{res}"] = iou(a, b)
        out[f"a_minus_b_{res}"] = len(sa - sb)
        out[f"b_minus_a_{res}"] = len(sb - sa)
        out[f"n_corep_{res}"] = len(a)
        out[f"n_native_{res}"] = len(b)
    return out


def run(golden_dir: str, resolutions: list[int], out: str) -> None:
    paths = sorted(
        glob.glob(os.path.join(golden_dir, "*.glb"))
        + glob.glob(os.path.join(golden_dir, "*", "*.glb"))
    )
    if not paths:
        raise FileNotFoundError(f"no .glb in {golden_dir}")

    rows = []
    for p in paths:
        try:
            rows.append(compare_one(p, resolutions))
        except Exception as e:
            print(f"[iou] FAIL {p}: {e}", file=sys.stderr)
            rows.append({"asset": os.path.basename(p), "error": str(e)})

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("# SS-flow Occupancy IoU Report\n\n")
        cols = (
            ["asset"]
            + [f"iou_{r}" for r in resolutions]
            + [f"n_corep_{r}" for r in resolutions]
            + [f"n_native_{r}" for r in resolutions]
        )
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                    for c in cols
                )
                + " |\n"
            )

        # decision
        valid = [r for r in rows if "error" not in r]
        if valid:
            min_res = min(resolutions)
            mean_iou_min = sum(r[f"iou_{min_res}"] for r in valid) / len(valid)
            f.write(f"\n**Mean IoU @ {min_res}^3: {mean_iou_min:.4f}**\n\n")
            if mean_iou_min >= 0.9:
                verdict = "PASS — scheme 1 directly, only inference-time affine quantization alignment"
            elif mean_iou_min >= 0.8:
                verdict = "PASS WITH WARNING — scheme 1 + spec adds affine alignment section"
            else:
                verdict = "FAIL — blocked, upgrade to scheme 2 (SS-flow finetune)"
            f.write(f"**Verdict: {verdict}**\n")
            print(f"[iou] mean_iou_{min_res}={mean_iou_min:.4f} -> {verdict}")
        else:
            print("[iou] no valid rows; check .glb paths and voxelize APIs", file=sys.stderr)
            sys.exit(1)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden_dir", required=True)
    p.add_argument("--resolutions", default="64,32,16")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    run(args.golden_dir, [int(x) for x in args.resolutions.split(",")], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
