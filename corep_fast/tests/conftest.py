"""
Shared pytest fixtures for corep_fast tests.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def custom_dir(project_root) -> Path:
    return project_root / 'custom'


@pytest.fixture
def test_mesh_dir(project_root) -> Path:
    return project_root / 'tmp' / 'test_mesh'


# ---------------------------------------------------------------------------
# Synthetic mesh fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_mesh() -> trimesh.Trimesh:
    """Unit cube — 12 triangles, 8 vertices, watertight."""
    return trimesh.creation.box(extents=[1., 1., 1.])


@pytest.fixture
def icosphere_mesh() -> trimesh.Trimesh:
    """Icosphere subdivisions=2 — closed, smooth, 80 faces."""
    return trimesh.creation.icosphere(subdivisions=2, radius=0.4)


@pytest.fixture
def open_plane_mesh() -> trimesh.Trimesh:
    """Flat square — 2 triangles, 4 boundary edges."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


@pytest.fixture
def cube_mesh_tensors(cube_mesh) -> MeshTensors:
    return MeshTensors.from_trimesh(cube_mesh, resolution=64, device='cpu')


@pytest.fixture
def icosphere_mesh_tensors(icosphere_mesh) -> MeshTensors:
    return MeshTensors.from_trimesh(icosphere_mesh, resolution=64, device='cpu')
