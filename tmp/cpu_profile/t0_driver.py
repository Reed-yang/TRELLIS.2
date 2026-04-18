"""T0 experiment driver: measure corep_pipeline e2e wall-time.

Usage:
    CUDA_VISIBLE_DEVICES=3 python tmp/cpu_profile/t0_driver.py --mode default --trials 3
    CUDA_VISIBLE_DEVICES=4 python tmp/cpu_profile/t0_driver.py --mode serial --trials 3
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import trimesh


# Inlined serial replacement for multiprocessing.Pool.
class SerialPool:
    def __init__(self, processes=None, *a, **kw):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, items, *a, **kw):
        return [fn(x) for x in items]

    def imap(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)

    def imap_unordered(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)

    def starmap(self, fn, items, *a, **kw):
        return [fn(*x) for x in items]

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


def make_fixture(path: str):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    mesh.export(path)


def maybe_force_serial():
    """Monkeypatch multiprocessing.Pool to run serially. Must run BEFORE
    importing corep_fast. Also patches multiprocessing.pool.Pool for full
    coverage (submodule must be imported explicitly)."""
    import multiprocessing as mp
    import multiprocessing.pool as mp_pool
    mp.Pool = SerialPool
    mp_pool.Pool = SerialPool


def patch_stage_pools():
    """Replace already-imported `_Pool` / `Pool` aliases in each stage module."""
    for mod_name in (
        "corep_fast.stages.s4_face_point",
        "corep_fast.stages.s6_collapse",
        "corep_fast.stages.s7_rank_assign",
        "corep_fast.stages.s8_collapse",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for alias in ("_Pool", "Pool"):
            if hasattr(mod, alias):
                setattr(mod, alias, SerialPool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["default", "serial"], required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256)
    args = p.parse_args()

    if args.mode == "serial":
        maybe_force_serial()

    # Import AFTER monkeypatch
    from corep_fast.pipeline import corep_pipeline

    if args.mode == "serial":
        patch_stage_pools()

    device = torch.device("cuda:0")
    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = str(Path(tmp) / "fixture.ply")
        make_fixture(mesh_path)

        # Warmup (not measured)
        warm = corep_pipeline(mesh_path, args.resolution, device)
        torch.cuda.synchronize()
        _, v_warm, f_warm = warm
        V_count = int(v_warm.shape[0]) if hasattr(v_warm, "shape") else -1
        F_count = int(f_warm.shape[0]) if hasattr(f_warm, "shape") else -1
        print(f"[warmup] mode={args.mode} V={V_count} F={F_count}", flush=True)

        samples = []
        last_V = last_F = -1
        for i in range(args.trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, v, f = corep_pipeline(mesh_path, args.resolution, device)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
            last_V = int(v.shape[0]) if hasattr(v, "shape") else -1
            last_F = int(f.shape[0]) if hasattr(f, "shape") else -1
            print(
                f"[trial {i+1}/{args.trials}] mode={args.mode} "
                f"wall={samples[-1]:.3f}s V={last_V} F={last_F}",
                flush=True,
            )

        sorted_samples = sorted(samples)
        median = sorted_samples[len(sorted_samples) // 2]
        result = {
            "mode": args.mode,
            "resolution": args.resolution,
            "trials": args.trials,
            "samples_sec": samples,
            "sorted_samples_sec": sorted_samples,
            "median_sec": median,
            "min_sec": sorted_samples[0],
            "max_sec": sorted_samples[-1],
            "V_count": last_V,
            "F_count": last_F,
        }
        out = Path(f"tmp/cpu_profile/t0_{args.mode}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[write] {out}")


if __name__ == "__main__":
    main()
