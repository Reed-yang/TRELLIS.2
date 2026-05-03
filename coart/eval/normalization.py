"""Canonical coordinate-system helpers for deep-eval.

The deep-eval metric path (Chamfer, F-score, normal consistency) compares
a predicted mesh against gt_points + Layer-V baseline values. These three
signals historically lived in three different coordinate systems:

    pred mesh       : CoReP [0,1]^3 offset space
        MeshTensors.from_trimesh (corep_fast/containers.py:74-90)
        scale = 0.947 / max_extent, offset = (0.489, 0.506, 0.513)

    gt_points       : raw glb coordinates (original Objaverse units)
        coart_build_golden.py originally sampled without any normalization

    Layer-V baseline: EXP-5 canonical [-0.5, 0.5]^3 space
        baseline_experiments.load_gt_mesh (scripts/eval/baseline_experiments.py:75-99)
        scale = 0.99999 / max_extent, centered

All three use isotropic bbox scaling with the same (center, max_extent), so
they are topologically equivalent and related by a constant affine. This
module exposes the canonical EXP-5 space and the pred->EXP-5 affine, so
gt_points, pred mesh, and baseline all land in the same frame for metric
computation.
"""
from __future__ import annotations

import numpy as np

# CoReP normalization constants (corep_fast/containers.py::MeshTensors.from_trimesh).
COREP_SCALE_FACTOR = 0.947
COREP_OFFSET = np.array([0.489, 0.506, 0.513], dtype=np.float32)

# EXP-5 canonical normalization (baseline_experiments.py::load_gt_mesh).
EXP5_SCALE_FACTOR = 0.99999

# Affine scale from CoReP [0,1]^3 to EXP-5 [-0.5, 0.5]^3. Because both
# normalizations share the same (center, max_extent) of the same mesh,
# only the scale factor and offset differ -- this is a single constant.
COREP_TO_EXP5_SCALE = EXP5_SCALE_FACTOR / COREP_SCALE_FACTOR  # ~= 1.05596


def normalize_mesh_exp5_inplace(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Apply EXP-5 canonical normalization to ``mesh.vertices`` in place.

    Returns ``(bbox_min, bbox_max)`` of the pre-normalized vertices so the
    caller can persist them as metadata.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = float((vmax - vmin).max())
    if extent == 0.0:
        raise ValueError("normalize_mesh_exp5_inplace: degenerate mesh with zero extent")
    scale = EXP5_SCALE_FACTOR / extent
    mesh.vertices = (verts - center) * scale
    return vmin.astype(np.float32), vmax.astype(np.float32)


def corep_to_exp5_vertices(v_corep: np.ndarray) -> np.ndarray:
    """Map vertices from CoReP [0,1]^3 offset space to EXP-5 [-0.5, 0.5]^3."""
    return (v_corep - COREP_OFFSET) * COREP_TO_EXP5_SCALE
