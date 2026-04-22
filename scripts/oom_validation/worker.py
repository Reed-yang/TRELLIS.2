"""OOM validation worker.

Runs corep_encode (s1-s7) on a single mesh, captures peak VRAM / host RSS /
per-stage walls / success status, and writes a JSON payload. Designed to be
launched in parallel on GPU 0/1/2/3 by run_4gpu_117.sh — each process sets
CUDA_VISIBLE_DEVICES=<rank> itself and then sees the chosen card as cuda:0.

Usage:
    python worker.py --label easy-40GiB \
        --glb_path datasets/...glb \
        --device cuda:0 \
        --out_json results/easy-40GiB.json

Add --dry_run to skip the actual encode and write a stub JSON (used by
smoke_test.sh to validate the runner before the impl lands).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path


def _first_12_hex(glb_path: str) -> str:
    """Return first 12 hex chars of the GLB filename (pre-.glb), matches the
    sha256-prefix convention in the OOM historical log."""
    name = Path(glb_path).stem
    # Filenames are 32-char hex; take the first 12
    return name[:12] if len(name) >= 12 else name


def _peak_host_rss_mb() -> float:
    """Peak resident set size of this process, in MiB.
    On Linux ru_maxrss is in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _stub_payload(args: argparse.Namespace) -> dict:
    """Build a deterministic stub JSON payload for --dry_run mode."""
    return {
        "label": args.label,
        "glb_path": args.glb_path,
        "device": args.device,
        "sha256_prefix": _first_12_hex(args.glb_path),
        "resolution": args.resolution,
        "success": True,
        "status": "ok",
        "error_msg": "",
        "wall_load_s": 0.0,
        "wall_encode_s": 0.0,
        "stage_walls_s": {
            "s1_voxelize": 0.0,
            "s2_components": 0.0,
            "s3_edge_weights": 0.0,
            "s4_face_point": 0.0,
            "s6_collapse": 0.0,
            "s7_rank_assign": 0.0,
        },
        "peak_vram_mb": 0.0,
        "peak_host_rss_mb": _peak_host_rss_mb(),
        "cube_count": 0,
        "num_components_total": 0,
        "num_boundary_total": 0,
        "dry_run": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OOM validation worker (single mesh → single GPU)")
    parser.add_argument("--label", required=True, help="Human label, e.g. 'easy-40GiB'")
    parser.add_argument("--glb_path", required=True, help="Path to input .glb (relative to repo root)")
    parser.add_argument("--device", default="cuda:0", help="Torch device string (default cuda:0)")
    parser.add_argument("--out_json", required=True, help="Output JSON path")
    parser.add_argument("--resolution", type=int, default=512, help="Voxel resolution (default 512)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip encode and write stub JSON (smoke-test mode)")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = _stub_payload(args)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[dry_run] label={args.label} wrote stub → {out_json}")
        return 0

    # Resolve mesh path against CWD (caller should run from repo root).
    glb_abs = str(Path(args.glb_path).resolve())
    if not Path(glb_abs).exists():
        payload = {
            "label": args.label,
            "glb_path": args.glb_path,
            "device": args.device,
            "sha256_prefix": _first_12_hex(args.glb_path),
            "resolution": args.resolution,
            "success": False,
            "status": "other_error:FileNotFoundError",
            "error_msg": f"GLB not found: {glb_abs}",
            "wall_load_s": None,
            "wall_encode_s": None,
            "stage_walls_s": {},
            "peak_vram_mb": None,
            "peak_host_rss_mb": _peak_host_rss_mb(),
            "cube_count": None,
            "num_components_total": None,
            "num_boundary_total": None,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[fail] {args.label}: GLB missing at {glb_abs}")
        return 2

    # Deferred imports so --help / --dry_run don't require torch.
    import torch
    import trimesh
    from corep_fast.containers import MeshTensors
    from corep_fast.pipeline import corep_encode
    from corep_fast.profiling.harness import ProfilingCollector

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ----- Load wall: trimesh.load + MeshTensors.from_trimesh -----
    t_load0 = time.perf_counter()
    status = "ok"
    error_msg = ""
    wall_encode = None
    peak_vram_mb = None
    stage_walls_s: dict[str, float] = {}
    cube_count = None
    num_components_total = None
    num_boundary_total = None
    success = False

    try:
        mesh = trimesh.load(glb_abs, force="mesh")
        # corep_encode itself normalizes + builds MeshTensors, so we only need
        # the wall for mesh load here. We still construct MeshTensors once to
        # expose any load-time failure (mirrors what 168k precompute does).
        _mt_probe = MeshTensors.from_trimesh(mesh, args.resolution, device=device)
        del _mt_probe
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_load = time.perf_counter() - t_load0
    except Exception as exc:  # noqa: BLE001
        wall_load = time.perf_counter() - t_load0
        status = f"other_error:{type(exc).__name__}"
        error_msg = (str(exc) or repr(exc))[:300]
        payload = {
            "label": args.label,
            "glb_path": args.glb_path,
            "device": args.device,
            "sha256_prefix": _first_12_hex(args.glb_path),
            "resolution": args.resolution,
            "success": False,
            "status": status,
            "error_msg": error_msg,
            "wall_load_s": wall_load,
            "wall_encode_s": None,
            "stage_walls_s": {},
            "peak_vram_mb": None,
            "peak_host_rss_mb": _peak_host_rss_mb(),
            "cube_count": None,
            "num_components_total": None,
            "num_boundary_total": None,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[fail:load] {args.label}: {status} {error_msg}")
        return 3

    # ----- Reset VRAM counters + encode -----
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    pc = ProfilingCollector(mesh_name=Path(glb_abs).name, resolution=args.resolution, impl="corep_fast")

    t_enc0 = time.perf_counter()
    try:
        batch = corep_encode(
            mesh_path=glb_abs,
            resolution=args.resolution,
            device=device,
            collector=pc,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_encode = time.perf_counter() - t_enc0

        # Extract correctness signals. .item() is fine here — only done after
        # successful encode and outside any hot loop.
        try:
            cube_count = int(batch.cube_indices.shape[0])
            num_components_total = int(batch.num_components.sum().item())
            num_boundary_total = int(batch.num_boundary.sum().item())
        except Exception:  # noqa: BLE001
            pass

        success = True
        status = "ok"
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        wall_encode = time.perf_counter() - t_enc0
        status = "oom"
        error_msg = (str(exc) or repr(exc))[:300]
    except Exception as exc:  # noqa: BLE001
        wall_encode = time.perf_counter() - t_enc0
        status = f"other_error:{type(exc).__name__}"
        error_msg = (str(exc) or repr(exc))[:300]
        # Also dump traceback to stderr for the .log file
        traceback.print_exc(file=sys.stderr)
    finally:
        if device.type == "cuda":
            try:
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            except Exception:  # noqa: BLE001
                peak_vram_mb = None

        # Stage walls from ProfilingCollector (empty dict if nothing recorded)
        try:
            for name, rec in pc._records.items():
                stage_walls_s[name] = float(rec.wall_time_s)
        except Exception:  # noqa: BLE001
            stage_walls_s = {}

    payload = {
        "label": args.label,
        "glb_path": args.glb_path,
        "device": args.device,
        "sha256_prefix": _first_12_hex(args.glb_path),
        "resolution": args.resolution,
        "success": success,
        "status": status,
        "error_msg": error_msg,
        "wall_load_s": wall_load,
        "wall_encode_s": wall_encode,
        "stage_walls_s": stage_walls_s,
        "peak_vram_mb": peak_vram_mb,
        "peak_host_rss_mb": _peak_host_rss_mb(),
        "cube_count": cube_count,
        "num_components_total": num_components_total,
        "num_boundary_total": num_boundary_total,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    # One-line summary for log grep
    summary = (
        f"[{args.label}] status={status} "
        f"encode={wall_encode:.2f}s " if wall_encode is not None else f"[{args.label}] status={status} encode=NA "
    )
    if wall_encode is None:
        summary = f"[{args.label}] status={status} encode=NA "
    else:
        summary = f"[{args.label}] status={status} encode={wall_encode:.2f}s "
    summary += (
        f"peak_vram={peak_vram_mb:.0f}MB " if peak_vram_mb is not None else "peak_vram=NA "
    )
    summary += f"peak_rss={_peak_host_rss_mb():.0f}MB "
    summary += f"cubes={cube_count}" if cube_count is not None else "cubes=NA"
    print(summary)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
