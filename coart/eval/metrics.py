"""Thin wrappers that expose canonical CD/NC/F-score/topology metrics
from scripts/eval/ under a stable coart.eval.metrics API.

Why wrappers: we pin a 100k-point sampling default and return plain dicts
(no PyTorch tensors) so deep_eval can log floats directly.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import torch

_COART_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_EVAL = os.path.join(_COART_ROOT, "scripts", "eval")
if _SCRIPTS_EVAL not in sys.path:
    sys.path.insert(0, _SCRIPTS_EVAL)

import eval_metrics as _em  # noqa: E402
import ovoxel_repr_test as _ort  # noqa: E402


def _to_torch(x):
    """Accept numpy or torch; return torch float32 on CUDA if available else CPU."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
    else:
        t = x.float()
    return t.cuda() if torch.cuda.is_available() else t


def sample_surface(mesh, num_points: int = 100000):
    """Returns (points_Nx3, normals_Nx3) as numpy float32."""
    pts, nrms = _em.sample_points_and_normals(mesh, num_points=num_points)
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    if isinstance(nrms, torch.Tensor):
        nrms = nrms.cpu().numpy()
    return np.asarray(pts, dtype=np.float32), np.asarray(nrms, dtype=np.float32)


def chamfer_distance(pts1, pts2) -> float:
    return float(_em.chamfer_distance(_to_torch(pts1), _to_torch(pts2)))


def normal_consistency(pts1, nrms1, pts2, nrms2) -> float:
    return float(_em.normal_consistency(
        _to_torch(pts1), _to_torch(nrms1), _to_torch(pts2), _to_torch(nrms2),
    ))


def f_score_multi(pts1, pts2, thresholds: List[float]) -> Dict[float, float]:
    """Return {threshold: f_score} for every requested threshold."""
    out = _em.f_score_multi(_to_torch(pts1), _to_torch(pts2),
                            thresholds=thresholds)
    return {float(k): float(v) for k, v in out.items()}


def compute_topo_metrics(mesh) -> Dict[str, float]:
    """Wrap _compute_topo_metrics; coerce bools to float for scalar logging."""
    d = _ort._compute_topo_metrics(mesh)
    return {
        "n_components": float(d["n_components"]),
        "euler_number": float(d["euler_number"]),
        "n_boundary_edges": float(d["n_boundary_edges"]),
        "is_watertight": 1.0 if d["is_watertight"] else 0.0,
        "surface_area": float(d["surface_area"]),
    }
