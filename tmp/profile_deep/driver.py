"""CoReP deep profiling driver — layer 0 (nsys-wrapped), layer 1/3 (torch.profiler).

Usage:
    python tmp/profile_deep/driver.py --layer 1 --res 256
    nsys profile ... python tmp/profile_deep/driver.py --layer 0 --res 256

Pipeline call pattern mirrors tmp/e2e_profile_m2.py verbatim.
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

# MUST patch before any corep_fast import
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tmp.profile_deep import monkeypatch_nvtx
monkeypatch_nvtx.apply_stage_nvtx()
# NOTE: apply_substage_events() intentionally DISABLED — Task 5 smoke check
# (commit 3af0d99) measured 41.3% patch overhead at res=128, far above the
# 10% decision-gate threshold. Per Plan Appendix A, fall back to NVTX-only
# for Tasks 6-11. Sub-stage breakdown in s4/s6/s7 will rely on NVTX ranges
# + torch.profiler kernel trace analysis (Task 8/9) rather than explicit
# Event timing. See results/smoke_overhead.log.
# monkeypatch_nvtx.apply_substage_events()  # DISABLED

import numpy as np
import torch

from tmp.profile_deep.build_icosphere import build_icosphere_subdiv3


RESULTS_DIR = REPO_ROOT / "tmp/profile_deep/results"


def set_deterministic():
    torch.manual_seed(42)
    np.random.seed(42)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def run_pipeline(mesh, res: int, device: torch.device) -> dict:
    """Run the full corep_fast pipeline end-to-end, return per-stage wall-time.

    Call pattern mirrors tmp/e2e_profile_m2.py verbatim. NVTX ranges (from
    monkeypatch) mark each stage; we still record explicit per-stage walltime
    for the summary JSON.
    """
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign
    from corep_fast.stages.s8_collapse import decode_from_cubebatch

    t = {}

    gc.collect(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    mt = MeshTensors.from_trimesh(mesh, res, device=device)
    torch.cuda.synchronize()
    t["load"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s1_voxelize(mt, res, device)
    torch.cuda.synchronize(); t["s1"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s2_components(batch, mt)
    torch.cuda.synchronize(); t["s2"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s3_edge_weights(batch, mt)
    torch.cuda.synchronize(); t["s3"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s4_face_point(batch, mt)
    torch.cuda.synchronize(); t["s4"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s6_collapse(batch)
    torch.cuda.synchronize(); t["s6"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    batch = s7_rank_assign(batch)
    torch.cuda.synchronize(); t["s7"] = time.perf_counter() - t0

    gc.collect(); t0 = time.perf_counter()
    v, f = decode_from_cubebatch(batch, merge_decimals=5)
    torch.cuda.synchronize(); t["s8"] = time.perf_counter() - t0

    t["s1_s7"] = sum(t[k] for k in ("s1", "s2", "s3", "s4", "s6", "s7"))
    t["e2e"] = t["load"] + t["s1_s7"] + t["s8"]
    t["_cubes"] = batch.num_cubes
    t["_V"] = int(v.shape[0])
    t["_F"] = int(f.shape[0])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["0", "1", "3"], required=True,
                    help="0 = naked run (nsys wraps externally); 1/3 = torch.profiler")
    ap.add_argument("--res", type=int, required=True)
    args = ap.parse_args()

    set_deterministic()
    mesh = build_icosphere_subdiv3()
    device = torch.device("cuda:0")

    # Warmup
    print(f"[warmup] pipeline at res={args.res}...")
    gc.collect()
    torch.cuda.empty_cache()
    _ = run_pipeline(mesh, args.res, device)
    torch.cuda.synchronize()

    # Reset accumulators (discard warmup events)
    monkeypatch_nvtx._SUBSTAGE_TIMINGS.clear()
    gc.collect()
    torch.cuda.empty_cache()

    out_prefix = RESULTS_DIR / f"layer{args.layer}_res{args.res}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    header = {
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nvidia_smi": nvidia_smi_snapshot(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "layer": args.layer,
        "resolution": args.res,
        "seeds": {"torch": 42, "numpy": 42},
    }

    print(f"[measure] layer={args.layer} res={args.res}...")
    t_stages: dict = {}
    pipeline_error: BaseException | None = None
    try:
        if args.layer == "0":
            t_stages = run_pipeline(mesh, args.res, device)
        else:
            trace_path = str(out_prefix) + "_trace.json"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
                record_shapes=True,
                profile_memory=False,
                with_stack=True,
            ) as prof:
                t_stages = run_pipeline(mesh, args.res, device)
                prof.step()
            print(f"[write] {trace_path}")
    except BaseException as e:
        pipeline_error = e
        print(f"[error] pipeline raised: {type(e).__name__}: {e}")
    finally:
        substage = monkeypatch_nvtx.dump_substage_timings()
        summary_path = str(out_prefix) + "_summary.json"
        with open(summary_path, "w") as f:
            # Serialize: drop per_call_ms lists (keep count + total) to stay small.
            serializable_sub = {
                k: {"count": v["count"], "total_ms": v["total_ms"]}
                for k, v in substage.items()
            }
            payload = {
                "header": header,
                "stage_walltime_sec": t_stages,
                "substage_ms": serializable_sub,
            }
            if pipeline_error is not None:
                payload["pipeline_error"] = f"{type(pipeline_error).__name__}: {pipeline_error}"
            json.dump(payload, f, indent=2)
        print(f"[write] {summary_path}")
        print(f"[done] e2e = {t_stages.get('e2e', -1):.3f}s")

    if pipeline_error is not None:
        raise pipeline_error


if __name__ == "__main__":
    main()
