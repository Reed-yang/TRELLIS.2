"""Per-fixture runner used by test_cpu_worker_optim.py.

Invoked as a subprocess so that each fixture runs in a fresh Python process
with a clean CUDA state. Without this isolation, the pipeline's output on
the triple-concentric icosphere (F3) drifts ~0.9 in V positions depending
on which fixture ran before it in the same process (cuDNN/cuBLAS autotune
cache + CUDA workspace pool are per-process and workload-dependent).

Usage:
    PYTHONHASHSEED=0 python _cpu_worker_optim_runner.py \\
        --fixture F1 --resolution 128 --mesh-name icosphere_s3 \\
        --output /tmp/out.pkl
"""
import argparse
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# SerialPool monkeypatch BEFORE corep_fast import
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool


class _SerialPool:
    def __init__(self, processes=None, *a, **kw):
        self.processes = processes
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, items, *a, **kw): return [fn(x) for x in items]
    def imap(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)
    def imap_unordered(self, fn, items, *a, **kw):
        for x in items:
            yield fn(x)
    def starmap(self, fn, items, *a, **kw): return [fn(*x) for x in items]
    def close(self): pass
    def terminate(self): pass
    def join(self): pass


_mp.Pool = _SerialPool
_mp_pool.Pool = _SerialPool

# CUBLAS workspace must be set BEFORE torch.cuda initializes to force a
# deterministic allocator layout across processes. Ref:
# https://docs.nvidia.com/cuda/cublas/#cublasApi_reproducibility
import os as _os_setup
_os_setup.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import trimesh

# Disable cuDNN autotune (picks different algorithms per-process) and
# request deterministic cuDNN paths. Full torch.use_deterministic_algorithms
# would raise on ops that have no deterministic CUDA kernel — we enable it
# in warn-only mode so the pipeline can still run if it hits one.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

from corep_fast.pipeline import corep_pipeline


def _triple_icosphere() -> trimesh.Trimesh:
    parts = []
    for r in (1.00, 1.01, 1.02):
        parts.append(trimesh.creation.icosphere(subdivisions=3, radius=r * 0.4))
    return trimesh.util.concatenate(parts)


MESH_FACTORIES = {
    ("icosphere_s3", 128): lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.4),
    ("icosphere_s3", 256): lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.4),
    ("triple_icosphere", 128): _triple_icosphere,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--mesh-name", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    factory = MESH_FACTORIES[(args.mesh_name, args.resolution)]
    mesh = factory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        mesh_path = str(Path(tmp) / "fixture.ply")
        mesh.export(mesh_path)
        _, v, f = corep_pipeline(mesh_path, args.resolution, device)

    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.cpu().numpy()

    result = {
        "V_count": int(v.shape[0]),
        "F_count": int(f.shape[0]),
        "V": v.astype(np.float64),
        "F": f.astype(np.int64),
    }
    with open(args.output, "wb") as fh:
        pickle.dump(result, fh)
    print(f"[runner:{args.fixture}] V={result['V_count']}, F={result['F_count']}, wrote {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
