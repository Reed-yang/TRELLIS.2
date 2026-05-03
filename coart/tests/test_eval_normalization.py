"""Guardrails for the CoReP<->EXP-5 coordinate-system bridge.

Both the CoReP pipeline and the EXP-5 baseline pipeline apply isotropic
bbox scaling with the same (center, max_extent). The affine between them
must stay a single constant; if either scale factor (COREP_SCALE_FACTOR,
EXP5_SCALE_FACTOR) or the offset (COREP_OFFSET) ever drifts, deep_eval
CD/NC/F-score metrics silently become wrong. These tests fail loudly in
that case.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from coart.eval.normalization import (
    COREP_OFFSET,
    COREP_SCALE_FACTOR,
    COREP_TO_EXP5_SCALE,
    EXP5_SCALE_FACTOR,
    corep_to_exp5_vertices,
    normalize_mesh_exp5_inplace,
)


def test_exp5_normalize_puts_mesh_in_unit_cube():
    """Any mesh, regardless of original scale/position, must land in [-0.5, 0.5]^3."""
    mesh = trimesh.creation.box(
        extents=(1e-3, 1e3, 3.14),
    ).apply_translation([1234.5, -567.25, 89.125])
    bbox_min, bbox_max = normalize_mesh_exp5_inplace(mesh)
    v = np.asarray(mesh.vertices)
    assert v.min() >= -0.5 - 1e-5, f"min {v.min()} violates [-0.5, 0.5]"
    assert v.max() <= 0.5 + 1e-5, f"max {v.max()} violates [-0.5, 0.5]"
    # Longest axis must hit (approximately) full span; 0.99999 safety margin.
    extent = v.max(axis=0) - v.min(axis=0)
    assert abs(extent.max() - EXP5_SCALE_FACTOR) < 1e-5
    # bbox metadata captured the pre-normalization range.
    np.testing.assert_allclose(bbox_min, np.array([1234.5 - 5e-4, -567.25 - 500, 89.125 - 1.57]), atol=1e-3)
    np.testing.assert_allclose(bbox_max, np.array([1234.5 + 5e-4, -567.25 + 500, 89.125 + 1.57]), atol=1e-3)


def test_exp5_normalize_rejects_degenerate_mesh():
    """Zero-extent mesh must raise rather than divide by zero."""
    mesh = trimesh.Trimesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        process=False,
    )
    with pytest.raises(ValueError, match="degenerate"):
        normalize_mesh_exp5_inplace(mesh)


def test_corep_to_exp5_matches_direct_normalization():
    """For any raw verts, (corep_to_exp5 ∘ corep_normalize) must equal
    (exp5_normalize ∘ identity). That's the whole point of the bridge --
    if drift sneaks in, deep_eval's pred mesh is in the wrong frame."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((1000, 3)).astype(np.float64) * 10 + np.array([42, -7, 3.14])

    vmin, vmax = raw.min(axis=0), raw.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = float((vmax - vmin).max())

    v_corep = (raw - center) * (COREP_SCALE_FACTOR / extent) + COREP_OFFSET
    v_exp5_direct = (raw - center) * (EXP5_SCALE_FACTOR / extent)
    v_exp5_via_corep = corep_to_exp5_vertices(v_corep.astype(np.float32))

    np.testing.assert_allclose(v_exp5_via_corep, v_exp5_direct, atol=1e-5)


def test_corep_to_exp5_scale_constant_is_correct():
    """The scale constant is the single source of pred-side correctness."""
    assert COREP_TO_EXP5_SCALE == pytest.approx(
        EXP5_SCALE_FACTOR / COREP_SCALE_FACTOR,
    )
    # Rough sanity: CoReP uses 0.947 margin, EXP-5 uses 0.99999 -> ratio ~1.056.
    assert 1.05 < COREP_TO_EXP5_SCALE < 1.06


def test_corep_offset_centers_on_half_cube():
    """COREP_OFFSET is the center the CoReP pipeline shifts meshes to.
    It must be near (0.5, 0.5, 0.5); if either axis drifts outside [0, 1]
    then pred mesh vertices could go negative or > 1, which changes the
    downstream affine result."""
    assert np.all(COREP_OFFSET >= 0.4)
    assert np.all(COREP_OFFSET <= 0.6)
