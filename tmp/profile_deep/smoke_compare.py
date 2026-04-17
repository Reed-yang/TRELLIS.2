"""Run the pipeline patched and unpatched, compare walltime."""
import gc
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from tmp.profile_deep.build_icosphere import build_icosphere_subdiv3


def _pipeline_once(mesh, res: int, device: torch.device):
    """Single pipeline run, no timing. Caller handles timing + sync."""
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign
    from corep_fast.stages.s8_collapse import decode_from_cubebatch

    mt = MeshTensors.from_trimesh(mesh, res, device=device)
    batch = s1_voxelize(mt, res, device)
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    batch = s6_collapse(batch)
    batch = s7_rank_assign(batch)
    return decode_from_cubebatch(batch, merge_decimals=5)


def _time_reps(mesh, res: int, reps: int, device: torch.device) -> list[float]:
    # Warmup
    _pipeline_once(mesh, res, device)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        _pipeline_once(mesh, res, device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def run_unpatched(mesh, res: int, reps: int = 3) -> list[float]:
    device = torch.device("cuda:0")
    return _time_reps(mesh, res, reps, device)


def run_patched(mesh, res: int, reps: int = 3) -> list[float]:
    from tmp.profile_deep import monkeypatch_nvtx
    monkeypatch_nvtx.apply_stage_nvtx()
    monkeypatch_nvtx.apply_substage_events()
    device = torch.device("cuda:0")
    return _time_reps(mesh, res, reps, device)


def main():
    mesh = build_icosphere_subdiv3()
    res = 128

    mode = sys.argv[1]  # "patched" or "unpatched"
    if mode == "patched":
        times = run_patched(mesh, res)
    elif mode == "unpatched":
        times = run_unpatched(mesh, res)
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        sys.exit(2)

    import statistics
    print(f"{mode} median={statistics.median(times):.3f}s  all={times}")


if __name__ == "__main__":
    main()
