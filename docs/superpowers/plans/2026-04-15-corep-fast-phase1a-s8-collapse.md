# CoReP-Fast Phase 1a: s8_collapse Torch Rewrite + Hybrid Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `s8_collapse` (the #1 bottleneck at 48% of pipeline runtime) as a pure-Torch stage, and build a hybrid pipeline orchestrator that can mix `custom/` and `corep_fast/` stages transparently.

**Architecture:** The hybrid pipeline orchestrator (`corep_fast/pipeline.py`) chains custom/ stages s1-s7 with the new corep_fast s8, using interop bridges at the boundary. The s8 Torch rewrite replaces all Python dict/set hotspots (vertex_to_index, seen_faces, per-edge Python loops) with vectorized `torch.sort`, `torch.unique_consecutive`, and `torch.unique` operations.

**Tech Stack:** Python 3.10+, Torch 2.x, NumPy, scipy, trimesh, pytest. No Triton, no CUDA C++ in Phase 1a.

**Spec:** `docs/superpowers/specs/2026-04-15-corep-fast-stage1-design.md` section 6.8 (s8_collapse Torch strategy)

**Prerequisite:** Phase 0 complete (88 tests passing). All Phase 0 infrastructure is available: `CubeBatch`, `MeshTensors`, interop bridges, profiling harness, topology equivalence checker, A/B rig.

**Critical correctness constraint:** The s8 Torch output must produce topology-equivalent meshes to custom/ — vertex counts within tolerance, face counts matching exactly. We validate via A/B comparison on the baseline evaluation set.

---

## File Structure

```
corep_fast/
├── pipeline.py                              # Task 1 — hybrid pipeline orchestrator
├── stages/
│   ├── __init__.py                          # Task 2 — update: export s8_collapse
│   └── s8_collapse.py                       # Tasks 2-6 — Torch s8 rewrite
└── tests/
    └── unit/
        ├── test_pipeline.py                 # Task 7
        └── test_s8_collapse.py              # Tasks 2-6 (test-first for each task)
```

**Key decisions locked by this structure:**

- `s8_collapse.py` contains all helper functions as module-level functions (not a class) — individually testable, following the `stage_forward()` public signature from spec section 6
- `pipeline.py` is the sole entry point for end-to-end runs; it imports from both `custom/` (via sys.path) and `corep_fast/`
- Tests mirror source layout: `test_s8_collapse.py` covers Tasks 2-6, `test_pipeline.py` covers Tasks 1, 7

---

## Task 1: Hybrid Pipeline Orchestrator

**Purpose:** A single function that runs the full CoReP pipeline using custom/ for stages s1-s7 and corep_fast/ for s8 (or any configured mix). This enables incremental migration: each stage can be independently toggled between custom/ and corep_fast/.

- [ ] **Step 0 (test):** Write `corep_fast/tests/unit/test_pipeline.py` with a smoke test that calls the pipeline orchestrator with a synthetic mesh and verifies it produces a PLY file.

`corep_fast/tests/unit/test_pipeline.py`:
```python
"""Unit tests for corep_fast/pipeline.py — hybrid pipeline orchestrator."""
import os
import tempfile

import pytest
import trimesh

from corep_fast.pipeline import run_hybrid_pipeline, PipelineConfig


class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig()
        # By default, s1-s7 use custom/, s8 uses corep_fast
        assert cfg.s8_impl == 'corep_fast'
        assert cfg.s1_to_s7_impl == 'custom'

    def test_all_custom(self):
        cfg = PipelineConfig(s8_impl='custom')
        assert cfg.s8_impl == 'custom'

    def test_invalid_impl_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(s8_impl='invalid')


class TestPipelineSmokeTest:
    """Requires custom/ to be importable. Skip if not available."""

    @pytest.fixture
    def simple_mesh_path(self, tmp_path):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        path = tmp_path / "test_sphere.ply"
        mesh.export(str(path))
        return str(path)

    def test_pipeline_produces_ply_custom_only(self, simple_mesh_path, tmp_path):
        """Run full pipeline with custom/ for all stages (baseline)."""
        cfg = PipelineConfig(s8_impl='custom')
        out_path = str(tmp_path / "out_custom.ply")
        result = run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_path,
            config=cfg,
        )
        assert os.path.exists(result)
        # Verify PLY is loadable
        loaded = trimesh.load(result)
        assert loaded.vertices.shape[0] > 0
        assert loaded.faces.shape[0] > 0
```

**Expected:** Test fails with `ModuleNotFoundError: No module named 'corep_fast.pipeline'`.

- [ ] **Step 1 (implement):** Create `corep_fast/pipeline.py`.

`corep_fast/pipeline.py`:
```python
"""
Hybrid pipeline orchestrator for CoReP processing.

Runs the full CoReP pipeline, mixing custom/ and corep_fast/ stages.
Default configuration: s1-s7 from custom/, s8 from corep_fast/.

Usage:
    from corep_fast.pipeline import run_hybrid_pipeline, PipelineConfig

    # Default: custom/ s1-s7 + corep_fast/ s8
    ply_path = run_hybrid_pipeline("mesh.ply", resolution=512, output_path="out.ply")

    # All custom/ (baseline comparison)
    cfg = PipelineConfig(s8_impl='custom')
    ply_path = run_hybrid_pipeline("mesh.ply", resolution=512, output_path="out.ply", config=cfg)
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from corep_fast.profiling.harness import ProfilingCollector, stage_timer


_VALID_IMPLS = frozenset({'custom', 'corep_fast'})


@dataclass
class PipelineConfig:
    """Configuration for the hybrid pipeline.

    Fields:
        s1_to_s7_impl: Implementation for stages 1-7. Only 'custom' supported in Phase 1a.
        s8_impl: Implementation for stage 8. 'custom' or 'corep_fast'.
        merge_decimals: Vertex welding precision (number of decimal places).
        profiling: Whether to collect per-stage timings.
    """
    s1_to_s7_impl: str = 'custom'
    s8_impl: str = 'corep_fast'
    merge_decimals: int = 5
    profiling: bool = False

    def __post_init__(self) -> None:
        if self.s1_to_s7_impl not in _VALID_IMPLS:
            raise ValueError(
                f"s1_to_s7_impl must be one of {sorted(_VALID_IMPLS)}, "
                f"got {self.s1_to_s7_impl!r}"
            )
        if self.s8_impl not in _VALID_IMPLS:
            raise ValueError(
                f"s8_impl must be one of {sorted(_VALID_IMPLS)}, "
                f"got {self.s8_impl!r}"
            )


def _ensure_custom_importable() -> None:
    """Add custom/ to sys.path if not already there."""
    project_root = str(Path(__file__).resolve().parents[1])
    custom_dir = os.path.join(project_root, 'custom')
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)


def _run_custom_s1_to_s7(
    mesh_path: str,
    resolution: int,
    output_dir: str,
    collector: Optional[ProfilingCollector] = None,
) -> list[dict]:
    """Run custom/ stages s1 through s7, returning the final list-of-dict registers."""
    _ensure_custom_importable()

    from voxelize import voxelize
    from feature_volume import feature_volume
    from feature_edge import feature_edge
    from feature_face import feature_face
    from feature_point import feature_point
    from collapse_face import collapse_face_inner, collapse_face_boundary
    from collapse_point import collapse_point_inner, collapse_point_boundary
    from collapse import mark_exception
    from utils import fetch_np_array

    mesh = trimesh.load(mesh_path)
    pc = collector or ProfilingCollector()

    with stage_timer('s1_voxelize', pc):
        norm_mesh, boundaries, face_regs, bnd_regs, nm_regs = \
            voxelize(mesh, output_dir, resolution)

    with stage_timer('s2_feature_volume', pc):
        face_regs, bnd_regs = feature_volume(
            face_regs, bnd_regs, norm_mesh, boundaries, output_dir)

    with stage_timer('s3_feature_edge', pc):
        face_regs = feature_edge(norm_mesh, resolution, face_regs, output_dir, debug=False)

    with stage_timer('s4_feature_face', pc):
        face_regs = feature_face(
            norm_mesh, resolution, face_regs, boundaries, bnd_regs, output_dir, debug=False)

    with stage_timer('s4_feature_point', pc):
        face_regs = feature_point(norm_mesh, resolution, face_regs, output_dir, debug=False)

    # Split inner vs boundary
    inner_mask = fetch_np_array(face_regs, 'num_boundary') == 0
    boundary_mask = ~inner_mask
    inner_regs = [face_regs[i] for i in range(len(face_regs)) if inner_mask[i]]
    boundary_regs_split = [face_regs[i] for i in range(len(face_regs)) if boundary_mask[i]]

    with stage_timer('s6_collapse_face', pc):
        solved_u, ambig_u, unsolv_u = collapse_face_inner(inner_regs)
        solved_b, ambig_b, unsolv_b = collapse_face_boundary(boundary_regs_split)

    with stage_timer('s7_collapse_point', pc):
        point_regs = collapse_point_inner(solved_u, resolution, debug=False,
                                          output_directory=output_dir)
        point_regs_bnd = collapse_point_boundary(solved_b, resolution, debug=False,
                                                  output_directory=output_dir)

    exception_regs = mark_exception([*ambig_u, *unsolv_u, *ambig_b, *unsolv_b])
    all_regs = [*point_regs, *point_regs_bnd, *exception_regs]

    return all_regs


def _run_custom_s8(
    resolution: int,
    all_regs: list[dict],
    output_path: str,
    collector: Optional[ProfilingCollector] = None,
) -> str:
    """Run custom/ stage 8 (collapse → PLY)."""
    _ensure_custom_importable()
    from collapse import reconstruct_mesh

    pc = collector or ProfilingCollector()
    with stage_timer('s8_collapse', pc):
        reconstruct_mesh(resolution, all_regs, output_filepath=output_path)

    return output_path


def _run_corep_fast_s8(
    resolution: int,
    all_regs: list[dict],
    output_path: str,
    merge_decimals: int = 5,
    collector: Optional[ProfilingCollector] = None,
) -> str:
    """Run corep_fast/ stage 8 (Torch collapse → PLY)."""
    from corep_fast.stages.s8_collapse import s8_collapse_to_ply

    pc = collector or ProfilingCollector()
    with stage_timer('s8_collapse', pc):
        s8_collapse_to_ply(
            resolution=resolution,
            cube_data_list=all_regs,
            output_filepath=output_path,
            merge_decimals=merge_decimals,
        )

    return output_path


def run_hybrid_pipeline(
    mesh_path: str,
    resolution: int,
    output_path: str,
    config: Optional[PipelineConfig] = None,
    collector: Optional[ProfilingCollector] = None,
) -> str:
    """
    Run the full CoReP pipeline on a single mesh.

    Args:
        mesh_path: Path to input .ply mesh.
        resolution: Voxel grid resolution.
        output_path: Where to write the output .ply file.
        config: Pipeline configuration. Defaults to custom/ s1-s7 + corep_fast/ s8.
        collector: Optional profiling collector for timing.

    Returns:
        Path to the output PLY file.
    """
    if config is None:
        config = PipelineConfig()

    # Create a temp directory for intermediate stage outputs
    tmp_dir = tempfile.mkdtemp(prefix='corep_hybrid_')
    try:
        # Stages 1-7: always custom/ in Phase 1a
        all_regs = _run_custom_s1_to_s7(
            mesh_path, resolution, tmp_dir, collector=collector,
        )

        # Stage 8: configurable
        if config.s8_impl == 'custom':
            return _run_custom_s8(
                resolution, all_regs, output_path, collector=collector,
            )
        else:
            return _run_corep_fast_s8(
                resolution, all_regs, output_path,
                merge_decimals=config.merge_decimals,
                collector=collector,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 2 (verify):** Run the test.

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_pipeline.py::TestPipelineConfig -xvs
```

**Expected:** `TestPipelineConfig` passes (3 tests). `TestPipelineSmokeTest` fails because `s8_collapse` module does not yet exist, but we won't run it yet.

- [ ] **Step 3 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/pipeline.py corep_fast/tests/unit/test_pipeline.py && \
  git commit -m "feat(corep_fast): add hybrid pipeline orchestrator with PipelineConfig"
```

---

## Task 2: Global Edge Enumeration (s8 kernel 1/4)

**Purpose:** For every occupied cube, enumerate its 12 axis-aligned global edges, encode them as int64 for sorting, determine unique edges, and compute edge ownership via lexicographic argmin. This replaces the Python `task_generator()` loop in custom/collapse.py.

The global edge encoding scheme from `custom/collapse.py::get_global_edge()`:
- Local edge 0 `(0,1)` → `('x', ix, iy, iz)` — Bottom Front
- Local edge 1 `(1,2)` → `('z', ix+1, iy, iz)` — Bottom Right
- Local edge 2 `(2,3)` → `('x', ix, iy, iz+1)` — Bottom Back
- Local edge 3 `(3,0)` → `('z', ix, iy, iz)` — Bottom Left
- Local edge 4 `(4,5)` → `('x', ix, iy+1, iz)` — Top Front
- Local edge 5 `(5,6)` → `('z', ix+1, iy+1, iz)` — Top Right
- Local edge 6 `(6,7)` → `('x', ix, iy+1, iz+1)` — Top Back
- Local edge 7 `(7,4)` → `('z', ix, iy+1, iz)` — Top Left
- Local edge 8 `(0,4)` → `('y', ix, iy, iz)` — Front-Left Vertical
- Local edge 9 `(1,5)` → `('y', ix+1, iy, iz)` — Front-Right Vertical
- Local edge 10 `(2,6)` → `('y', ix+1, iy, iz+1)` — Back-Right Vertical
- Local edge 11 `(3,7)` → `('y', ix, iy, iz+1)` — Back-Left Vertical

The `get_surrounding_cubes()` function gives the 4 neighbors in cyclic order:
- axis `'x'`: `(nx, ny, nz), (nx, ny-1, nz), (nx, ny-1, nz-1), (nx, ny, nz-1)`
- axis `'y'`: `(nx, ny, nz), (nx-1, ny, nz), (nx-1, ny, nz-1), (nx, ny, nz-1)`
- axis `'z'`: `(nx, ny, nz), (nx-1, ny, nz), (nx-1, ny-1, nz), (nx, ny-1, nz)`

The ownership condition in custom/ is: `min(active_neighbors) == idx`. In Torch, this becomes: for each unique global edge, find the minimum cube_hash among its occupied neighbors.

- [ ] **Step 0 (test):** Write tests for global edge enumeration.

`corep_fast/tests/unit/test_s8_collapse.py` (initial):
```python
"""Unit tests for corep_fast/stages/s8_collapse.py — Torch s8 rewrite."""
import pytest
import torch

from corep_fast.stages.s8_collapse import (
    compute_global_edge_keys,
    enumerate_unique_edges,
    build_edge_neighbor_table,
    compute_edge_ownership,
)


# ---- Fixtures ----

@pytest.fixture
def single_cube():
    """One cube at (5, 5, 5), resolution 64."""
    cube_indices = torch.tensor([[5, 5, 5]], dtype=torch.int32)
    resolution = 64
    return cube_indices, resolution


@pytest.fixture
def two_adjacent_cubes():
    """Two cubes sharing an edge: (5,5,5) and (5,5,6), resolution 64.
    These share the X-axis edge at global coords (5, 5, 6) and others."""
    cube_indices = torch.tensor([[5, 5, 5], [5, 5, 6]], dtype=torch.int32)
    resolution = 64
    return cube_indices, resolution


@pytest.fixture
def four_cubes_around_edge():
    """Four cubes sharing a single Y-axis edge at (5,5,5):
    (4,5,4), (5,5,4), (4,5,5), (5,5,5)
    These are the 4 neighbors returned by get_surrounding_cubes('y', 5, 5, 5)."""
    cube_indices = torch.tensor([
        [4, 5, 4], [5, 5, 4], [4, 5, 5], [5, 5, 5]
    ], dtype=torch.int32)
    resolution = 64
    return cube_indices, resolution


# ---- Tests for compute_global_edge_keys ----

class TestComputeGlobalEdgeKeys:
    def test_single_cube_produces_12_edges(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        # Each cube generates 12 global edge keys
        assert keys.shape == (12,)
        assert cube_ids.shape == (12,)
        assert local_ids.shape == (12,)
        # All cube_ids should be 0 (single cube)
        assert (cube_ids == 0).all()
        # local_ids should be 0..11
        assert local_ids.tolist() == list(range(12))

    def test_two_cubes_produce_24_edges(self, two_adjacent_cubes):
        cube_indices, resolution = two_adjacent_cubes
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        assert keys.shape == (24,)

    def test_global_keys_are_int64(self, single_cube):
        cube_indices, resolution = single_cube
        keys, _, _ = compute_global_edge_keys(cube_indices, resolution)
        assert keys.dtype == torch.int64

    def test_adjacent_cubes_share_edges(self, two_adjacent_cubes):
        """Two adjacent cubes must share some global edge keys."""
        cube_indices, resolution = two_adjacent_cubes
        keys, cube_ids, _ = compute_global_edge_keys(cube_indices, resolution)
        keys_0 = set(keys[cube_ids == 0].tolist())
        keys_1 = set(keys[cube_ids == 1].tolist())
        shared = keys_0 & keys_1
        # Cubes (5,5,5) and (5,5,6) are adjacent in Z — they share 4 edges
        assert len(shared) >= 4


# ---- Tests for enumerate_unique_edges ----

class TestEnumerateUniqueEdges:
    def test_single_cube_has_12_unique_edges(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        assert unique_keys.shape[0] == 12  # isolated cube, all edges unique

    def test_shared_edges_reduce_count(self, two_adjacent_cubes):
        cube_indices, resolution = two_adjacent_cubes
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        # 24 total entries, some deduplicated
        assert unique_keys.shape[0] < 24
        assert edge_id_per_entry.shape == keys.shape

    def test_four_cubes_reduce_more(self, four_cubes_around_edge):
        cube_indices, resolution = four_cubes_around_edge
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        # 4 cubes * 12 = 48 entries, shared edges reduce the unique count
        assert unique_keys.shape[0] < 48


# ---- Tests for build_edge_neighbor_table ----

class TestBuildEdgeNeighborTable:
    def test_single_cube_all_edges_have_one_neighbor(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        # table.neighbor_counts: (num_unique_edges,) — each should be 1
        assert (table.neighbor_counts == 1).all()

    def test_four_cubes_shared_edge_has_four_neighbors(self, four_cubes_around_edge):
        """The Y-axis edge at (5,5,5) is shared by all 4 cubes."""
        cube_indices, resolution = four_cubes_around_edge
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        # At least one edge should have 4 neighbors
        assert (table.neighbor_counts == 4).any()


# ---- Tests for compute_edge_ownership ----

class TestComputeEdgeOwnership:
    def test_single_cube_owns_all_edges(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        owner_cube = compute_edge_ownership(table, cube_indices)
        # Single cube, it owns all 12 edges
        assert (owner_cube == 0).all()

    def test_shared_edge_owned_by_lex_smallest(self, four_cubes_around_edge):
        """Edge ownership should go to the cube with smallest (ix,iy,iz)."""
        cube_indices, resolution = four_cubes_around_edge
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        owner_cube = compute_edge_ownership(table, cube_indices)
        # For the fully-shared edge, owner should be cube 0 = (4,5,4) — lex smallest
        max_neighbors_edge = table.neighbor_counts.argmax()
        assert owner_cube[max_neighbors_edge].item() == 0
```

**Expected:** Tests fail with `ModuleNotFoundError: No module named 'corep_fast.stages.s8_collapse'`.

- [ ] **Step 1 (implement):** Create `corep_fast/stages/s8_collapse.py` with the edge enumeration functions.

`corep_fast/stages/s8_collapse.py`:
```python
"""
Stage 8: Collapse — Torch-accelerated mesh reconstruction from CoReP cube data.

Replaces custom/collapse.py with vectorized Torch operations. The critical
optimization is replacing Python dict/set vertex dedup with torch.unique,
and replacing per-edge Python iteration with batched sort + unique_consecutive.

Reference: spec section 6.8 (s8_collapse Torch strategy).

Public API:
    s8_collapse_to_ply(resolution, cube_data_list, output_filepath, merge_decimals)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Edge offset table: for each local edge 0..11, the (axis, dx, dy, dz) offset
# applied to (ix, iy, iz) to produce the global edge coordinate.
#
# axis encoding: 0 = X, 1 = Y, 2 = Z
# Derived from custom/collapse.py::get_global_edge().
# ---------------------------------------------------------------------------

# (axis, dx, dy, dz) for local edges 0..11
_EDGE_OFFSET_TABLE = torch.tensor([
    # local 0: ('x', ix, iy, iz)       → axis=0, dx=0, dy=0, dz=0
    [0, 0, 0, 0],
    # local 1: ('z', ix+1, iy, iz)     → axis=2, dx=1, dy=0, dz=0
    [2, 1, 0, 0],
    # local 2: ('x', ix, iy, iz+1)     → axis=0, dx=0, dy=0, dz=1
    [0, 0, 0, 1],
    # local 3: ('z', ix, iy, iz)       → axis=2, dx=0, dy=0, dz=0
    [2, 0, 0, 0],
    # local 4: ('x', ix, iy+1, iz)     → axis=0, dx=0, dy=1, dz=0
    [0, 0, 1, 0],
    # local 5: ('z', ix+1, iy+1, iz)   → axis=2, dx=1, dy=1, dz=0
    [2, 1, 1, 0],
    # local 6: ('x', ix, iy+1, iz+1)   → axis=0, dx=0, dy=1, dz=1
    [0, 0, 1, 1],
    # local 7: ('z', ix, iy+1, iz)     → axis=2, dx=0, dy=1, dz=0
    [2, 0, 1, 0],
    # local 8: ('y', ix, iy, iz)       → axis=1, dx=0, dy=0, dz=0
    [1, 0, 0, 0],
    # local 9: ('y', ix+1, iy, iz)     → axis=1, dx=1, dy=0, dz=0
    [1, 1, 0, 0],
    # local 10: ('y', ix+1, iy, iz+1)  → axis=1, dx=1, dy=0, dz=1
    [1, 1, 0, 1],
    # local 11: ('y', ix, iy, iz+1)    → axis=1, dx=0, dy=0, dz=1
    [1, 0, 0, 1],
], dtype=torch.int32)  # (12, 4)


# ---------------------------------------------------------------------------
# Neighbor offset table: for each axis, the 4 neighbor cube offsets.
# From custom/collapse.py::get_surrounding_cubes().
#
# axis X: (nx, ny, nz), (nx, ny-1, nz), (nx, ny-1, nz-1), (nx, ny, nz-1)
#   → offsets from (nx, ny, nz): (0,0,0), (0,-1,0), (0,-1,-1), (0,0,-1)
# axis Y: (nx, ny, nz), (nx-1, ny, nz), (nx-1, ny, nz-1), (nx, ny, nz-1)
#   → offsets: (0,0,0), (-1,0,0), (-1,0,-1), (0,0,-1)
# axis Z: (nx, ny, nz), (nx-1, ny, nz), (nx-1, ny-1, nz), (nx, ny-1, nz)
#   → offsets: (0,0,0), (-1,0,0), (-1,-1,0), (0,-1,0)
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = torch.tensor([
    # axis 0 (X)
    [[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]],
    # axis 1 (Y)
    [[0, 0, 0], [-1, 0, 0], [-1, 0, -1], [0, 0, -1]],
    # axis 2 (Z)
    [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
], dtype=torch.int32)  # (3, 4, 3)


# ---------------------------------------------------------------------------
# Reverse local edge table: for each (axis, neighbor_position 0..3), which
# local edge of that neighbor maps to this shared global edge.
# Derived from custom/collapse.py::get_local_edge() inverted.
#
# For axis X (edge_axis=0):
#   neighbor (dy=0, dz=0) → local_edge 6 (Top-Back)      — position 0 in _NEIGHBOR_OFFSETS
#                            Wait, the offsets for axis X are (0,0,0), (0,-1,0), (0,-1,-1), (0,0,-1)
#   Position 0: offset (0,0,0) = (nx, ny, nz). In get_local_edge: dy=ny-min_y=1, dz=nz-min_z=1
#     → axis X, dy=1, dz=1 → local_edge 0 (Bottom-Front)
#   Position 1: offset (0,-1,0) = (nx, ny-1, nz). dy=ny-1-min_y=0, dz=nz-min_z=1
#     → axis X, dy=0, dz=1 → local_edge 2 (Bottom-Back)
#   Position 2: offset (0,-1,-1) = (nx, ny-1, nz-1). dy=0, dz=0
#     → axis X, dy=0, dz=0 → local_edge 6 (Top-Back)
#   Position 3: offset (0,0,-1) = (nx, ny, nz-1). dy=1, dz=0
#     → axis X, dy=1, dz=0 → local_edge 4 (Top-Front)
#
# For axis Y (edge_axis=1):
#   Position 0: (0,0,0) → dx=1, dz=1 → local_edge 3 (Bottom-Left)
#   Position 1: (-1,0,0) → dx=0, dz=1 → local_edge 1 (Bottom-Right)
#   Position 2: (-1,0,-1) → dx=0, dz=0 → local_edge 5 (Top-Right)
#   Position 3: (0,0,-1) → dx=1, dz=0 → local_edge 7 (Top-Left)
#
# For axis Z (edge_axis=2):
#   Position 0: (0,0,0) → dx=1, dy=1 → local_edge 8 (Front-Left)
#   Position 1: (-1,0,0) → dx=0, dy=1 → local_edge 9 (Front-Right)
#   Position 2: (-1,-1,0) → dx=0, dy=0 → local_edge 10 (Back-Right)
#   Position 3: (0,-1,0) → dx=1, dy=0 → local_edge 11 (Back-Left)
# ---------------------------------------------------------------------------

_REVERSE_LOCAL_EDGE = torch.tensor([
    # axis 0 (X): positions 0,1,2,3
    [0, 2, 6, 4],
    # axis 1 (Y): positions 0,1,2,3
    [3, 1, 5, 7],
    # axis 2 (Z): positions 0,1,2,3
    [8, 9, 10, 11],
], dtype=torch.int32)  # (3, 4)


# ---------------------------------------------------------------------------
# Negative-direction local edges (ranks need flipping).
# From custom/collapse.py: "Edges 2, 6 (-X) and 3, 7 (-Y) go in the
# negative direction."
# ---------------------------------------------------------------------------

_NEGATIVE_DIRECTION_EDGES = frozenset({2, 6, 3, 7})


# ---------------------------------------------------------------------------
# EdgeNeighborTable: CSR-like table mapping unique edges → their neighbor cubes
# ---------------------------------------------------------------------------

@dataclass
class EdgeNeighborTable:
    """
    Mapping from unique global edges to their neighboring occupied cubes.

    Fields:
        neighbor_counts: (E,) int32 — number of occupied neighbors per unique edge
        neighbor_cube_ids: (E, 4) int32 — cube indices (into cube_indices array),
            padded with -1 for absent neighbors
        neighbor_positions: (E, 4) int32 — position 0..3 in the cyclic neighbor order,
            padded with -1
        neighbor_local_edges: (E, 4) int32 — local edge index (0..11) for each neighbor,
            padded with -1
        edge_axes: (E,) int32 — axis (0=X, 1=Y, 2=Z) of each unique edge
    """
    neighbor_counts: torch.Tensor
    neighbor_cube_ids: torch.Tensor
    neighbor_positions: torch.Tensor
    neighbor_local_edges: torch.Tensor
    edge_axes: torch.Tensor


def compute_global_edge_keys(
    cube_indices: torch.Tensor,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For every occupied cube, compute the 12 global edge keys.

    Each global edge is encoded as a single int64:
        key = axis * R^3 + nx * R^2 + ny * R + nz
    where (axis, nx, ny, nz) is the global edge identifier and R = resolution + 2
    (to ensure non-negative coordinates after offsets).

    Args:
        cube_indices: (N, 3) int32 — (ix, iy, iz) per cube.
        resolution: int — grid resolution.

    Returns:
        keys: (N*12,) int64 — global edge key per (cube, local_edge) pair.
        cube_ids: (N*12,) int32 — which cube each entry belongs to.
        local_ids: (N*12,) int32 — which local edge (0..11).
    """
    device = cube_indices.device
    N = cube_indices.shape[0]
    table = _EDGE_OFFSET_TABLE.to(device)  # (12, 4)

    # Expand cube_indices: (N, 1, 3) + offsets (1, 12, 3) → (N, 12, 3)
    ix = cube_indices[:, 0:1].to(torch.int64)  # (N, 1)
    iy = cube_indices[:, 1:2].to(torch.int64)
    iz = cube_indices[:, 2:3].to(torch.int64)

    # offsets for each local edge
    axes = table[:, 0].to(torch.int64)    # (12,)
    dx = table[:, 1].to(torch.int64)      # (12,)
    dy = table[:, 2].to(torch.int64)
    dz = table[:, 3].to(torch.int64)

    # Global edge coordinates: (N, 12)
    nx = ix + dx.unsqueeze(0)   # (N, 12)
    ny = iy + dy.unsqueeze(0)
    nz = iz + dz.unsqueeze(0)
    ax = axes.unsqueeze(0).expand(N, 12)  # (N, 12)

    # Encode as int64 key. Use R = resolution + 2 for headroom.
    R = int(resolution) + 2
    keys = ax * (R * R * R) + nx * (R * R) + ny * R + nz  # (N, 12)

    # Flatten
    keys = keys.reshape(-1)                    # (N*12,)

    # cube_ids: which cube each entry belongs to
    cube_ids = torch.arange(N, dtype=torch.int32, device=device).unsqueeze(1).expand(N, 12).reshape(-1)

    # local_ids: which local edge
    local_ids = torch.arange(12, dtype=torch.int32, device=device).unsqueeze(0).expand(N, 12).reshape(-1)

    return keys, cube_ids, local_ids


def enumerate_unique_edges(
    keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deduplicate global edge keys.

    Args:
        keys: (M,) int64 — all global edge keys from compute_global_edge_keys.

    Returns:
        unique_keys: (E,) int64 — sorted unique edge keys.
        edge_id_per_entry: (M,) int64 — maps each input entry to its unique edge index.
    """
    sorted_keys, sort_idx = torch.sort(keys)
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)

    # Map back to original order
    edge_id_per_entry = torch.empty_like(keys, dtype=torch.int64)
    edge_id_per_entry[sort_idx] = inverse.to(torch.int64)

    return unique_keys, edge_id_per_entry


def build_edge_neighbor_table(
    edge_id_per_entry: torch.Tensor,
    cube_ids: torch.Tensor,
    local_ids: torch.Tensor,
    num_unique_edges: int,
) -> EdgeNeighborTable:
    """
    Build a table mapping each unique edge to its occupied neighbor cubes.

    For each unique edge, up to 4 cubes may share it. We record which cubes,
    what their local edge index is, and their position (0..3) in the cyclic
    neighbor ordering.

    Args:
        edge_id_per_entry: (M,) int64 — unique edge index per entry.
        cube_ids: (M,) int32 — cube index per entry.
        local_ids: (M,) int32 — local edge (0..11) per entry.
        num_unique_edges: int — total number of unique edges.

    Returns:
        EdgeNeighborTable with fields filled.
    """
    device = edge_id_per_entry.device
    E = num_unique_edges
    M = edge_id_per_entry.shape[0]

    # Sort by edge_id to group neighbors of the same edge
    sort_idx = torch.argsort(edge_id_per_entry)
    sorted_edge_ids = edge_id_per_entry[sort_idx]
    sorted_cube_ids = cube_ids[sort_idx]
    sorted_local_ids = local_ids[sort_idx]

    # Count neighbors per edge
    neighbor_counts = torch.zeros(E, dtype=torch.int32, device=device)
    neighbor_counts.scatter_add_(
        0,
        sorted_edge_ids.to(torch.int64),
        torch.ones(M, dtype=torch.int32, device=device),
    )

    # Build padded (E, 4) tables
    neighbor_cube_ids = torch.full((E, 4), -1, dtype=torch.int32, device=device)
    neighbor_local_edges = torch.full((E, 4), -1, dtype=torch.int32, device=device)
    neighbor_positions = torch.full((E, 4), -1, dtype=torch.int32, device=device)

    # Determine edge axes from the first entry of each edge
    edge_axes = torch.zeros(E, dtype=torch.int32, device=device)

    # We need a per-edge slot counter. Use a scatter approach.
    # First, compute the offset of each entry within its edge group.
    # edges_start[e] = index of first entry with edge_id == e
    # For each entry in sorted order, its position within the group.
    ones = torch.ones(M, dtype=torch.int64, device=device)
    cumcount = torch.zeros(E, dtype=torch.int64, device=device)

    # Sequential fill — this is the bottleneck but only runs once
    # and M is typically < 1M for resolution 256.
    # For a fully vectorized version we would use segment_csr, but
    # a simple Python loop over unique edges is acceptable for Phase 1a.
    offsets = torch.zeros(E + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(neighbor_counts.to(torch.int64), dim=0)

    table_offset = _EDGE_OFFSET_TABLE.to(device)

    for e_idx in range(E):
        lo = int(offsets[e_idx].item())
        hi = int(offsets[e_idx + 1].item())
        count = hi - lo
        if count == 0:
            continue

        # Extract entries for this edge
        e_cube_ids = sorted_cube_ids[lo:hi]
        e_local_ids = sorted_local_ids[lo:hi]

        # Axis from the first entry's local edge
        first_local = int(e_local_ids[0].item())
        axis = int(table_offset[first_local, 0].item())
        edge_axes[e_idx] = axis

        for slot in range(min(count, 4)):
            cid = int(e_cube_ids[slot].item())
            lid = int(e_local_ids[slot].item())
            neighbor_cube_ids[e_idx, slot] = cid
            neighbor_local_edges[e_idx, slot] = lid
            # Position: determine from which position in _REVERSE_LOCAL_EDGE
            # this local edge corresponds to.
            rev = _REVERSE_LOCAL_EDGE[axis]
            pos = (rev == lid).nonzero(as_tuple=False)
            if pos.numel() > 0:
                neighbor_positions[e_idx, slot] = pos[0, 0].item()

    return EdgeNeighborTable(
        neighbor_counts=neighbor_counts,
        neighbor_cube_ids=neighbor_cube_ids,
        neighbor_positions=neighbor_positions,
        neighbor_local_edges=neighbor_local_edges,
        edge_axes=edge_axes,
    )


def compute_edge_ownership(
    table: EdgeNeighborTable,
    cube_indices: torch.Tensor,
) -> torch.Tensor:
    """
    For each unique edge, determine which cube owns it (lexicographic minimum).

    Custom/ uses: min(active_neighbors) == idx, comparing tuples lexicographically.
    We replicate this by computing cube_hash = ix * R^2 + iy * R + iz for each
    neighbor and taking argmin.

    Args:
        table: EdgeNeighborTable from build_edge_neighbor_table.
        cube_indices: (N, 3) int32.

    Returns:
        owner_cube_id: (E,) int32 — cube index of the owner for each unique edge.
            -1 if edge has no neighbors (shouldn't happen for occupied cubes).
    """
    device = cube_indices.device
    E = table.neighbor_counts.shape[0]
    R = int(cube_indices.max().item()) + 2

    # Compute hash for all cubes
    cube_hash = (
        cube_indices[:, 0].to(torch.int64) * R * R
        + cube_indices[:, 1].to(torch.int64) * R
        + cube_indices[:, 2].to(torch.int64)
    )  # (N,)

    # For each edge's neighbor slots, look up hash. Pad with MAX for absent slots.
    MAX_HASH = torch.iinfo(torch.int64).max
    neighbor_hashes = torch.full((E, 4), MAX_HASH, dtype=torch.int64, device=device)

    # Fill in valid slots
    valid = table.neighbor_cube_ids >= 0  # (E, 4)
    valid_cids = table.neighbor_cube_ids.clamp(min=0).to(torch.int64)  # (E, 4)
    neighbor_hashes[valid] = cube_hash[valid_cids[valid]]

    # Argmin per edge
    min_slot = neighbor_hashes.argmin(dim=1)  # (E,)
    owner_cube_id = torch.gather(table.neighbor_cube_ids, 1, min_slot.unsqueeze(1)).squeeze(1)

    return owner_cube_id
```

- [ ] **Step 2 (update stages/__init__.py):**

`corep_fast/stages/__init__.py`:
```python
"""
CoReP-Fast stage implementations.

Phase 1a: s8_collapse (Torch rewrite of custom/collapse.py).
"""
```

- [ ] **Step 3 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py -xvs -k "TestCompute or TestEnumerate or TestBuildEdge or TestComputeEdge"
```

**Expected:** All 11 tests pass.

- [ ] **Step 4 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/stages/__init__.py corep_fast/stages/s8_collapse.py \
          corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): s8 global edge enumeration + ownership (kernel 1/4)"
```

---

## Task 3: Shared-Edge Geometry Processing (s8 kernel 2/4)

**Purpose:** For each owned edge, gather the 4 neighboring cubes' loop data, group component_points by normalized rank on this edge, compute projection points as the mean, and emit fan triangles. This is the core geometry function replacing `process_shared_edge_geometry()`.

**Key algorithm details from custom/collapse.py:**
1. For each neighbor cube at position `pos`, find its local edge for this shared edge
2. Iterate over that cube's `sorted_loops` — for each loop that crosses the local edge, extract the rank from `loop_data['rank'][i]`
3. Normalize rank: edges 2, 6, 3, 7 go in negative direction, so `normalized_rank = (W - 1) - rank`
4. Group points by normalized rank: `points_by_rank[rank][uv] = point`
5. The UV positions for the 4 neighbors are `(0,0), (1,0), (1,1), (0,1)` in cyclic order
6. For each rank group with all 4 points present, create 4 fan triangles connecting projection point to neighbors

- [ ] **Step 0 (test):** Add geometry processing tests.

Append to `corep_fast/tests/unit/test_s8_collapse.py`:
```python
from corep_fast.stages.s8_collapse import process_shared_edges_batch


class TestProcessSharedEdgesBatch:
    """Test the shared-edge geometry processing."""

    def _make_simple_cube_data(self):
        """Create 4 cubes around a Y-axis edge, each with 1 loop crossing the shared edge.

        Cubes: (4,5,4), (5,5,4), (4,5,5), (5,5,5)
        Shared Y-axis edge at global (5,5,5).

        For axis Y neighbors:
          pos 0: (5,5,5) → local_edge 3
          pos 1: (4,5,5) → local_edge 1
          pos 2: (4,5,4) → local_edge 5
          pos 3: (5,5,4) → local_edge 7

        Each cube has 1 loop crossing its local edge at rank 0, with a component_point.
        """
        cube_data_list = [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'loop': [3, 12, 1], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': False,
                'num_components': 1,
            },
        ]
        return cube_data_list

    def test_four_cubes_produce_fan_triangles(self):
        """With all 4 neighbors present at rank 0, we should get 4 fan triangles."""
        data = self._make_simple_cube_data()
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # 4 component_points + 1 projection_point = 5 vertices (before welding)
        assert verts.shape[0] >= 5
        # 4 fan triangles
        assert faces.shape[0] >= 4
        assert faces.shape[1] == 3

    def test_two_cubes_produce_no_fan(self):
        """With only 2 neighbors, rank group has 2 points — no fan triangles (need 4)."""
        data = self._make_simple_cube_data()[:2]
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # No faces because we need 4 neighbors for fan generation
        assert faces.shape[0] == 0

    def test_empty_input_returns_empty(self):
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=[],
            merge_decimals=5,
        )
        assert verts.shape == (0, 3)
        assert faces.shape == (0, 3)

    def test_projection_point_is_mean(self):
        """The projection point should be the mean of the 4 component points."""
        data = self._make_simple_cube_data()
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # Expected mean: (0.04+0.05+0.04+0.05)/4=0.045, 0.05, (0.04+0.04+0.05+0.05)/4=0.045
        expected_proj = torch.tensor([0.045, 0.05, 0.045])
        # Find the projection point — it should be one of the vertices
        found = False
        for i in range(verts.shape[0]):
            if torch.allclose(verts[i], expected_proj, atol=1e-5):
                found = True
                break
        assert found, f"Projection point {expected_proj} not found in vertices:\n{verts}"
```

**Expected:** Tests fail with `ImportError: cannot import name 'process_shared_edges_batch'`.

- [ ] **Step 1 (implement):** Add `process_shared_edges_batch` to `s8_collapse.py`.

This function operates on the Python list-of-dict format (same as custom/) but internally vectorizes the heavy work. The full Torch-native path using CubeBatch CSR is a future optimization.

Append to `corep_fast/stages/s8_collapse.py`:
```python
def process_shared_edges_batch(
    resolution: int,
    cube_data_list: list[dict],
    merge_decimals: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process all shared edges across all cubes and return (vertices, faces).

    This replaces the full iteration loop in custom/collapse.py::generate_global_mesh().
    It uses the same algorithm but collects all results into tensors.

    Args:
        resolution: Grid resolution.
        cube_data_list: List of dicts from custom/ s7 output.
        merge_decimals: Vertex welding precision.

    Returns:
        vertices: (V, 3) float32 — welded vertex coordinates.
        faces: (F, 3) int32 — triangle faces (vertex indices).
    """
    if not cube_data_list:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    # 1. Build cube_map: cube_indices_tuple → list[pruned_dict]
    cube_map: dict[tuple, list[dict]] = {}
    for data in cube_data_list:
        idx = data.get('cube_indices')
        if idx is None:
            continue
        idx = tuple(idx) if not isinstance(idx, tuple) else idx

        loops = []
        for loop in data.get('sorted_loops', []):
            loops.append({
                'component_point': loop.get('component_point'),
                'rank': loop.get('rank', []),
                'loop': loop.get('loop', []),
            })

        pruned = {
            'cube_indices': idx,
            'sorted_loops': loops,
            'edge_weights': data.get('edge_weights', [0] * 18),
            'exception': data.get('exception', False),
            'num_components': data.get('num_components', 0),
        }

        if idx not in cube_map:
            cube_map[idx] = []
        cube_map[idx].append(pruned)

    # 2. Enumerate owned edges (using the same task_generator logic as custom/)
    all_verts: list[tuple] = []
    all_tris: list[tuple[tuple, tuple, tuple]] = []

    def get_grid_input(indices_list):
        grid = []
        valid_count = 0
        for idx in indices_list:
            if idx in cube_map:
                grid.append(cube_map[idx])
                valid_count += 1
            else:
                grid.append([{'cube_indices': idx, 'sorted_loops': []}])
        return grid, valid_count

    for idx in cube_map:
        x, y, z = idx
        local_edges = [
            ('X', x, y, z), ('X', x, y+1, z), ('X', x, y, z+1), ('X', x, y+1, z+1),
            ('Y', x, y, z), ('Y', x+1, y, z), ('Y', x, y, z+1), ('Y', x+1, y, z+1),
            ('Z', x, y, z), ('Z', x+1, y, z), ('Z', x, y+1, z), ('Z', x+1, y+1, z),
        ]

        for axis, a, b, c in local_edges:
            if axis == 'X':
                if not (0 <= a < resolution and 1 <= b < resolution and 1 <= c < resolution):
                    continue
                neighbors = [(a, b-1, c-1), (a, b, c-1), (a, b-1, c), (a, b, c)]
            elif axis == 'Y':
                if not (1 <= a < resolution and 0 <= b < resolution and 1 <= c < resolution):
                    continue
                neighbors = [(a-1, b, c-1), (a, b, c-1), (a-1, b, c), (a, b, c)]
            elif axis == 'Z':
                if not (1 <= a < resolution and 1 <= b < resolution and 0 <= c < resolution):
                    continue
                neighbors = [(a-1, b-1, c), (a, b-1, c), (a-1, b, c), (a, b, c)]
            else:
                continue

            active_neighbors = [n for n in neighbors if n in cube_map]
            if not active_neighbors:
                continue

            if min(active_neighbors) == idx:
                grid, count = get_grid_input(neighbors)
                if count >= 2:
                    verts, tris = _process_shared_edge_geometry(grid)
                    all_verts.extend(verts)
                    all_tris.extend(tris)

    # 3. Vertex welding + face dedup using Torch
    if not all_tris:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    return _weld_and_dedup(all_verts, all_tris, merge_decimals)


def _process_shared_edge_geometry(
    grid_2x2_lists: list[list[dict]],
) -> tuple[list[tuple], list[tuple]]:
    """
    Process a 2x2 grid of cubes sharing an edge.

    Replicates custom/collapse.py::process_shared_edge_geometry() exactly,
    but returns raw vertex/triangle tuples for downstream Torch welding.

    Args:
        grid_2x2_lists: 4 lists of dicts for the 4 neighbor positions.

    Returns:
        (new_vertices, triangles) where each triangle is (pt0, pt1, pt2)
        as coordinate tuples.
    """
    import collections

    # 1. Deduce edge axis
    valid_indices = []
    for cube_list in grid_2x2_lists:
        if cube_list and 'cube_indices' in cube_list[0]:
            valid_indices.append(cube_list[0]['cube_indices'])

    if not valid_indices:
        return [], []

    min_idx = [min(idx[i] for idx in valid_indices) for i in range(3)]
    max_idx = [max(idx[i] for idx in valid_indices) for i in range(3)]

    edge_axis = -1
    for i in range(3):
        if min_idx[i] == max_idx[i]:
            edge_axis = i
            break

    if edge_axis == -1:
        edge_axis = 2

    def get_local_edge(dx, dy, dz):
        if edge_axis == 2:
            if dx == 0 and dy == 0: return 10
            if dx == 1 and dy == 0: return 11
            if dx == 0 and dy == 1: return 9
            if dx == 1 and dy == 1: return 8
        elif edge_axis == 0:
            if dy == 0 and dz == 0: return 6
            if dy == 1 and dz == 0: return 4
            if dy == 0 and dz == 1: return 2
            if dy == 1 and dz == 1: return 0
        elif edge_axis == 1:
            if dx == 0 and dz == 0: return 5
            if dx == 1 and dz == 0: return 7
            if dx == 0 and dz == 1: return 1
            if dx == 1 and dz == 1: return 3
        return -1

    # 2. Extract points by rank
    points_by_rank = collections.defaultdict(dict)
    exception_points = {}
    cube_by_uv = {}
    shared_edge_weight = 0

    for cube_list in grid_2x2_lists:
        if not cube_list:
            continue
        idx = cube_list[0]['cube_indices']
        dx = idx[0] - min_idx[0]
        dy = idx[1] - min_idx[1]
        dz = idx[2] - min_idx[2]

        local_edge = get_local_edge(dx, dy, dz)
        if local_edge == -1:
            continue

        if edge_axis == 2:   uv = (dx, dy)
        elif edge_axis == 0: uv = (dy, dz)
        elif edge_axis == 1: uv = (dx, dz)

        cube_by_uv[uv] = cube_list

        for d in cube_list:
            w = d.get('edge_weights', [0]*18)[local_edge]
            if w > 0:
                shared_edge_weight = max(shared_edge_weight, w)

            if d.get('exception') is True:
                pt = None
                if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
                    pt = d['sorted_loops'][0].get('component_point')
                elif d.get('component_points') and len(d['component_points']) > 0:
                    pt = d['component_points'][0]
                elif d.get('component_point'):
                    pt = d.get('component_point')
                if pt is not None:
                    exception_points[uv] = pt
                continue

            for loop_data in d.get('sorted_loops', []):
                edges = loop_data.get('loop', [])
                ranks = loop_data.get('rank', [])

                for i, edge in enumerate(edges):
                    if edge == local_edge:
                        rank = ranks[i]

                        if local_edge in (2, 6, 3, 7):
                            W = d.get('edge_weights', [0]*18)[local_edge]
                            normalized_rank = (W - 1) - rank
                        else:
                            normalized_rank = rank

                        pt = loop_data.get('component_point')
                        if pt is not None:
                            points_by_rank[normalized_rank][uv] = pt

    # 3. Conditional promotion of disconnected cubes to exceptions
    all_have_components = len(cube_by_uv) == 4 and all(
        any(d.get('num_components', 0) > 0 for d in cl) for cl in cube_by_uv.values()
    )
    all_incomplete = all(len(pt_map) < 4 for pt_map in points_by_rank.values())

    if all_have_components and shared_edge_weight > 0 and all_incomplete:
        neighbors_pairs = [((0,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0))]
        new_exception_uvs = set()

        for uv1, uv2 in neighbors_pairs:
            connected = False
            for pt_map in points_by_rank.values():
                if uv1 in pt_map and uv2 in pt_map:
                    connected = True
                    break
            if not connected:
                if uv1 in cube_by_uv: new_exception_uvs.add(uv1)
                if uv2 in cube_by_uv: new_exception_uvs.add(uv2)

        for uv in new_exception_uvs:
            if uv not in exception_points:
                pt = None
                for d in cube_by_uv[uv]:
                    if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
                        pt = d['sorted_loops'][0].get('component_point')
                    elif d.get('component_points') and len(d['component_points']) > 0:
                        pt = d['component_points'][0]
                    elif d.get('component_point'):
                        pt = d.get('component_point')
                    if pt is not None:
                        break
                if pt is not None:
                    exception_points[uv] = pt
                for pt_map in points_by_rank.values():
                    pt_map.pop(uv, None)

        empty_ranks = [r for r, pm in points_by_rank.items() if not pm]
        for r in empty_ranks:
            del points_by_rank[r]

    # 4. Distribute exceptions as wildcards
    for rank, pt_map in points_by_rank.items():
        for uv, exc_pt in exception_points.items():
            if uv not in pt_map:
                pt_map[uv] = exc_pt

    if not points_by_rank and len(exception_points) > 0:
        points_by_rank[0] = exception_points

    # 5. Emit triangles
    new_vertices = []
    triangles = []
    neighbors = [(0,0), (1,0), (1,1), (0,1)]

    for rank, pt_map in sorted(points_by_rank.items()):
        rank_pts = list(pt_map.values())
        if not rank_pts:
            continue

        avg_x = sum(p[0] for p in rank_pts) / len(rank_pts)
        avg_y = sum(p[1] for p in rank_pts) / len(rank_pts)
        avg_z = sum(p[2] for p in rank_pts) / len(rank_pts)
        proj_pt = (avg_x, avg_y, avg_z)
        new_vertices.append(proj_pt)

        if len(rank_pts) == 4:
            triangles.append((proj_pt, pt_map[neighbors[0]], pt_map[neighbors[1]]))
            triangles.append((proj_pt, pt_map[neighbors[1]], pt_map[neighbors[2]]))
            triangles.append((proj_pt, pt_map[neighbors[2]], pt_map[neighbors[3]]))
            triangles.append((proj_pt, pt_map[neighbors[3]], pt_map[neighbors[0]]))

    return new_vertices, triangles
```

- [ ] **Step 2 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestProcessSharedEdgesBatch -xvs
```

**Expected:** All 4 tests pass.

- [ ] **Step 3 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): s8 shared-edge geometry processing (kernel 2/4)"
```

---

## Task 4: Vertex Welding + Face Dedup (s8 kernel 3/4)

**Purpose:** Replace the Python `vertex_to_index` dict and `seen_faces` set with `torch.unique` — this is the key optimization that eliminates the two critical Python hotspots in custom/collapse.py.

**Algorithm:**
1. Collect all triangle vertices into a `(T*3, 3)` float32 tensor
2. Round to `merge_decimals` precision → `torch.round(v * 10^d) / 10^d`
3. `torch.unique(rounded, dim=0, return_inverse=True)` → welded vertex table + reindex
4. Remap face indices, remove degenerate triangles (3 distinct vertices)
5. Canonical face rotation (rotate so min vertex is first) → `torch.unique(canonical, dim=0)` → deduped faces

- [ ] **Step 0 (test):** Add vertex welding tests.

Append to `corep_fast/tests/unit/test_s8_collapse.py`:
```python
from corep_fast.stages.s8_collapse import _weld_and_dedup


class TestWeldAndDedup:
    def test_basic_welding(self):
        """Vertices that round to the same value should be merged."""
        verts = [
            (0.100001, 0.200001, 0.300001),
            (0.100002, 0.200002, 0.300002),  # same after rounding to 4 decimals
            (0.5, 0.5, 0.5),
        ]
        tris = [
            (verts[0], verts[1], verts[2]),
        ]
        v, f = _weld_and_dedup(verts, tris, merge_decimals=4)
        # After welding, verts[0] and verts[1] merge → 2 unique vertices
        assert v.shape[0] == 2
        # But the face becomes degenerate (v0==v1), so it's removed
        assert f.shape[0] == 0

    def test_non_degenerate_preserved(self):
        """Three distinct vertices form a valid triangle."""
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        tris = [(verts[0], verts[1], verts[2])]
        v, f = _weld_and_dedup(verts, tris, merge_decimals=5)
        assert v.shape[0] == 3
        assert f.shape[0] == 1

    def test_duplicate_faces_removed(self):
        """Same triangle appearing twice should be deduplicated."""
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        tris = [
            (verts[0], verts[1], verts[2]),
            (verts[1], verts[2], verts[0]),  # same triangle, rotated
        ]
        v, f = _weld_and_dedup(verts, tris, merge_decimals=5)
        assert v.shape[0] == 3
        assert f.shape[0] == 1

    def test_different_faces_preserved(self):
        """Two distinct triangles should both be kept."""
        verts = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        ]
        tris = [
            (verts[0], verts[1], verts[2]),
            (verts[3], verts[4], verts[5]),
        ]
        v, f = _weld_and_dedup(verts, tris, merge_decimals=5)
        assert v.shape[0] == 4  # (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        assert f.shape[0] == 2

    def test_empty_input(self):
        v, f = _weld_and_dedup([], [], merge_decimals=5)
        assert v.shape == (0, 3)
        assert f.shape == (0, 3)

    def test_output_dtypes(self):
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        tris = [(verts[0], verts[1], verts[2])]
        v, f = _weld_and_dedup(verts, tris, merge_decimals=5)
        assert v.dtype == torch.float32
        assert f.dtype == torch.int32
```

**Expected:** Tests fail with `ImportError: cannot import name '_weld_and_dedup'`.

- [ ] **Step 1 (implement):** Add `_weld_and_dedup` to `s8_collapse.py`.

Append to `corep_fast/stages/s8_collapse.py`:
```python
def _weld_and_dedup(
    all_verts: list[tuple],
    all_tris: list[tuple],
    merge_decimals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vertex welding + face deduplication using torch.unique.

    This replaces the Python dict vertex_to_index and set seen_faces from
    custom/collapse.py::generate_global_mesh() with O(N log N) sort-based dedup.

    Args:
        all_verts: List of (x, y, z) coordinate tuples (unused directly — vertices
                   are extracted from triangle tuples).
        all_tris: List of ((x0,y0,z0), (x1,y1,z1), (x2,y2,z2)) triangle tuples.
        merge_decimals: Number of decimal places for vertex rounding.

    Returns:
        vertices: (V, 3) float32 — unique vertex coordinates.
        faces: (F, 3) int32 — triangle face indices.
    """
    if not all_tris:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )

    # 1. Flatten all triangle vertices into (T*3, 3) tensor
    T = len(all_tris)
    flat_verts = torch.zeros((T * 3, 3), dtype=torch.float64)
    for i, tri in enumerate(all_tris):
        for j, pt in enumerate(tri):
            flat_verts[i * 3 + j, 0] = pt[0]
            flat_verts[i * 3 + j, 1] = pt[1]
            flat_verts[i * 3 + j, 2] = pt[2]

    # 2. Round for welding
    scale = 10.0 ** merge_decimals
    rounded = torch.round(flat_verts * scale)

    # 3. Unique vertices via torch.unique on rounded coordinates
    unique_rounded, inverse_indices = torch.unique(rounded, dim=0, return_inverse=True)

    # 4. Recover actual coordinates: take the first occurrence of each unique rounded vertex
    num_unique = unique_rounded.shape[0]
    unique_verts = torch.zeros((num_unique, 3), dtype=torch.float32)
    # Use scatter to fill with first-seen coordinates (order from inverse)
    seen = torch.zeros(num_unique, dtype=torch.bool)
    for i in range(flat_verts.shape[0]):
        uid = int(inverse_indices[i].item())
        if not seen[uid]:
            unique_verts[uid] = flat_verts[i].float()
            seen[uid] = True

    # 5. Build face index array
    face_indices = inverse_indices.reshape(T, 3).to(torch.int32)

    # 6. Remove degenerate faces (where any two vertices are the same)
    v0 = face_indices[:, 0]
    v1 = face_indices[:, 1]
    v2 = face_indices[:, 2]
    non_degenerate = (v0 != v1) & (v1 != v2) & (v2 != v0)
    face_indices = face_indices[non_degenerate]

    if face_indices.shape[0] == 0:
        return unique_verts, torch.zeros((0, 3), dtype=torch.int32)

    # 7. Canonical face rotation: rotate so minimum vertex index is first
    face_long = face_indices.to(torch.int64)
    min_val, _ = face_long.min(dim=1, keepdim=True)

    # For each face, find which position has the minimum
    is_min = face_long == min_val  # (F, 3)
    # Take the first True position
    min_pos = is_min.to(torch.int64).argmax(dim=1)  # (F,)

    # Rotate: shift so min_pos becomes position 0
    arange3 = torch.arange(3, device=face_long.device).unsqueeze(0)  # (1, 3)
    shifted = (arange3 + min_pos.unsqueeze(1)) % 3  # (F, 3)
    canonical = torch.gather(face_long, 1, shifted)  # (F, 3)

    # 8. Deduplicate canonical faces
    unique_canonical, unique_idx = torch.unique(canonical, dim=0, return_inverse=True)

    # Keep only the first occurrence of each unique canonical face
    # We want to preserve the original winding, so use the first occurrence index
    F_unique = unique_canonical.shape[0]
    first_occurrence = torch.zeros(F_unique, dtype=torch.int64, device=face_long.device)
    seen_face = torch.zeros(F_unique, dtype=torch.bool, device=face_long.device)
    for i in range(face_indices.shape[0]):
        uid = int(unique_idx[i].item())
        if not seen_face[uid]:
            first_occurrence[uid] = i
            seen_face[uid] = True

    deduped_faces = face_indices[first_occurrence].to(torch.int32)

    return unique_verts, deduped_faces
```

- [ ] **Step 2 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestWeldAndDedup -xvs
```

**Expected:** All 6 tests pass.

- [ ] **Step 3 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): s8 vertex welding + face dedup via torch.unique (kernel 3/4)"
```

---

## Task 5: PLY Writer + Public API (s8 kernel 4/4)

**Purpose:** Wire everything together: the public `s8_collapse_to_ply()` function that the pipeline calls. Takes the same inputs as `custom/collapse.py::reconstruct_mesh()` — a resolution, list of cube dicts, and output path — and writes a PLY file.

- [ ] **Step 0 (test):** Add PLY output tests.

Append to `corep_fast/tests/unit/test_s8_collapse.py`:
```python
import os
import tempfile
import trimesh

from corep_fast.stages.s8_collapse import s8_collapse_to_ply


class TestS8CollapseToPly:
    def _make_cube_data_four_around_edge(self):
        """4 cubes around a Y-axis edge with complete rank 0 data."""
        return [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'loop': [3, 12, 1], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': False,
                'num_components': 1,
            },
        ]

    def test_produces_valid_ply(self, tmp_path):
        data = self._make_cube_data_four_around_edge()
        out_path = str(tmp_path / "test_output.ply")
        result = s8_collapse_to_ply(
            resolution=64,
            cube_data_list=data,
            output_filepath=out_path,
        )
        assert os.path.exists(result)
        mesh = trimesh.load(result)
        assert mesh.vertices.shape[0] > 0
        assert mesh.faces.shape[0] > 0

    def test_empty_input_produces_empty_ply(self, tmp_path):
        out_path = str(tmp_path / "empty.ply")
        result = s8_collapse_to_ply(
            resolution=64,
            cube_data_list=[],
            output_filepath=out_path,
        )
        assert os.path.exists(result)

    def test_ply_header_format(self, tmp_path):
        data = self._make_cube_data_four_around_edge()
        out_path = str(tmp_path / "header_check.ply")
        s8_collapse_to_ply(resolution=64, cube_data_list=data, output_filepath=out_path)
        with open(out_path, 'r') as f:
            lines = f.readlines()
        assert lines[0].strip() == 'ply'
        assert lines[1].strip() == 'format ascii 1.0'
        assert 'element vertex' in lines[2]
        assert 'end_header' in ''.join(lines[:10])
```

**Expected:** Tests fail with `ImportError`.

- [ ] **Step 1 (implement):** Add `s8_collapse_to_ply` and PLY writer to `s8_collapse.py`.

Append to `corep_fast/stages/s8_collapse.py`:
```python
def _write_ply_ascii(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    output_filepath: str,
) -> None:
    """
    Write vertices and faces to an ASCII PLY file.

    Args:
        vertices: (V, 3) float32.
        faces: (F, 3) int32.
        output_filepath: Destination path.
    """
    V = vertices.shape[0]
    F = faces.shape[0]

    verts_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy() if F > 0 else np.zeros((0, 3), dtype=np.int32)

    with open(output_filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {V}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {F}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        for i in range(V):
            f.write(f"{verts_np[i, 0]} {verts_np[i, 1]} {verts_np[i, 2]}\n")

        # Faces
        for i in range(F):
            f.write(f"3 {faces_np[i, 0]} {faces_np[i, 1]} {faces_np[i, 2]}\n")


def s8_collapse_to_ply(
    resolution: int,
    cube_data_list: list[dict],
    output_filepath: str = "output_mesh.ply",
    merge_decimals: int = 5,
) -> str:
    """
    Public API: Reconstruct mesh from cube data and write PLY.

    Drop-in replacement for custom/collapse.py::reconstruct_mesh().

    This function performs:
    1. reform_intersection_data (extract loops + component_points)
    2. extract_original_cube_edges (filter to edges 0-11)
    3. generate_global_mesh (edge enumeration, geometry, welding, PLY write)

    But steps 1-2 are unnecessary for the sorted_loops format (which already has
    component_point and loop/rank data), so we skip them when sorted_loops is present.

    Args:
        resolution: Grid resolution.
        cube_data_list: List of cube dicts from s7 output.
        output_filepath: Where to write the PLY.
        merge_decimals: Vertex welding precision.

    Returns:
        The output filepath.
    """
    # If cube_data_list uses the old 'loops' + 'component_points' format (not sorted_loops),
    # run reform + extract first
    needs_reform = False
    for data in cube_data_list:
        if 'sorted_loops' not in data and 'loops' in data:
            needs_reform = True
            break

    if needs_reform:
        cube_data_list = _reform_and_extract(cube_data_list)

    # Run geometry processing
    vertices, faces = process_shared_edges_batch(
        resolution=resolution,
        cube_data_list=cube_data_list,
        merge_decimals=merge_decimals,
    )

    # Write PLY
    _write_ply_ascii(vertices, faces, output_filepath)

    return output_filepath


def _reform_and_extract(cube_data_list: list[dict]) -> list[dict]:
    """
    Convert old-format cube data (with 'loops' + 'component_points') to sorted_loops format.

    Replicates custom/collapse.py::reform_intersection_data() +
    extract_original_cube_edges() but maps to sorted_loops format.
    """
    for data in cube_data_list:
        if 'sorted_loops' in data:
            continue

        loops = data.get('loops', [])
        points = data.get('component_points', [])
        num_loops = data.get('num_loops', 0)

        sorted_loops = []
        if num_loops == len(points):
            for i in range(num_loops):
                sorted_loops.append({
                    'component_point': list(points[i]) if isinstance(points[i], (list, tuple)) else points[i],
                    'loop': loops[i] if i < len(loops) else [],
                    'rank': [-1] * len(loops[i]) if i < len(loops) else [],
                })

        data['sorted_loops'] = sorted_loops

    return cube_data_list
```

- [ ] **Step 2 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestS8CollapseToPly -xvs
```

**Expected:** All 3 tests pass.

- [ ] **Step 3 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): s8 PLY writer + s8_collapse_to_ply public API (kernel 4/4)"
```

---

## Task 6: Exception Cube Handling

**Purpose:** Verify that exception cubes (AMBIGUOUS, UNSOLVABLE, BUDGET_EXCEEDED) are handled correctly. Exception cubes use their first component_point as a wildcard that fills into any rank group where the cube is an absent neighbor.

The exception logic is already implemented in `_process_shared_edge_geometry()` (Task 3), which faithfully replicates custom/collapse.py. This task adds dedicated tests to verify correctness.

- [ ] **Step 0 (test):** Add exception cube tests.

Append to `corep_fast/tests/unit/test_s8_collapse.py`:
```python
class TestExceptionCubeHandling:
    """Test that exception cubes inject their component_point as wildcards."""

    def _make_data_with_exception(self):
        """3 normal cubes + 1 exception cube around a shared edge.

        Normal cubes have loops crossing the shared edge at rank 0.
        Exception cube has exception=True and a component_point.
        """
        return [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False,
                'num_components': 1,
            },
            {
                # Exception cube — its component_point should fill in as wildcard
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': True,
                'num_components': 1,
            },
        ]

    def test_exception_fills_missing_slot(self):
        """With 3 normal + 1 exception, we should still get fan triangles
        because the exception point fills the 4th slot."""
        data = self._make_data_with_exception()
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # Should produce fan triangles (exception fills the 4th position)
        assert faces.shape[0] >= 4

    def test_all_exceptions_produce_fan(self):
        """4 exception cubes — all have component_points, should group at rank 0
        and form fan if 4 points available."""
        data = [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': True,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': True,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': True,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': True,
                'num_components': 1,
            },
        ]
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # All exceptions → grouped at rank 0 with 4 points → 4 fan triangles
        assert faces.shape[0] >= 4
```

- [ ] **Step 1 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestExceptionCubeHandling -xvs
```

**Expected:** All 2 tests pass (exception handling is already in `_process_shared_edge_geometry`).

- [ ] **Step 2 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "test(corep_fast): add exception cube handling tests for s8_collapse"
```

---

## Task 7: A/B Integration Validation

**Purpose:** Run the full hybrid pipeline with custom/ s1-s7 + corep_fast/ s8, compare the output PLY against custom/-only output. This is the definitive correctness test.

- [ ] **Step 0 (test):** Add A/B integration test.

Append to `corep_fast/tests/unit/test_pipeline.py`:
```python
class TestABComparison:
    """A/B comparison: custom/ s8 vs corep_fast/ s8 on the same s1-s7 output."""

    @pytest.fixture
    def simple_mesh_path(self, tmp_path):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        path = tmp_path / "test_sphere.ply"
        mesh.export(str(path))
        return str(path)

    def test_mesh_vertex_count_within_tolerance(self, simple_mesh_path, tmp_path):
        """Both implementations should produce similar vertex counts."""
        # Run custom/ for everything
        cfg_custom = PipelineConfig(s8_impl='custom')
        out_custom = str(tmp_path / "out_custom.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_custom,
            config=cfg_custom,
        )

        # Run custom/ s1-s7 + corep_fast/ s8
        cfg_fast = PipelineConfig(s8_impl='corep_fast')
        out_fast = str(tmp_path / "out_fast.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_fast,
            config=cfg_fast,
        )

        mesh_custom = trimesh.load(out_custom)
        mesh_fast = trimesh.load(out_fast)

        # Vertex count should match exactly (same algorithm, same data)
        assert mesh_custom.vertices.shape[0] == mesh_fast.vertices.shape[0], \
            f"Vertex count mismatch: custom={mesh_custom.vertices.shape[0]}, fast={mesh_fast.vertices.shape[0]}"

    def test_mesh_face_count_matches(self, simple_mesh_path, tmp_path):
        """Both implementations should produce the same face count."""
        cfg_custom = PipelineConfig(s8_impl='custom')
        out_custom = str(tmp_path / "out_custom.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_custom,
            config=cfg_custom,
        )

        cfg_fast = PipelineConfig(s8_impl='corep_fast')
        out_fast = str(tmp_path / "out_fast.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_fast,
            config=cfg_fast,
        )

        mesh_custom = trimesh.load(out_custom)
        mesh_fast = trimesh.load(out_fast)

        assert mesh_custom.faces.shape[0] == mesh_fast.faces.shape[0], \
            f"Face count mismatch: custom={mesh_custom.faces.shape[0]}, fast={mesh_fast.faces.shape[0]}"

    def test_profiled_pipeline_records_s8(self, simple_mesh_path, tmp_path):
        """Profiling collector should record s8_collapse timing."""
        from corep_fast.profiling.harness import ProfilingCollector

        cfg = PipelineConfig(s8_impl='corep_fast')
        out = str(tmp_path / "out_profiled.ply")
        pc = ProfilingCollector(mesh_name='test', resolution=64, impl='hybrid')
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out,
            config=cfg,
            collector=pc,
        )

        assert 's8_collapse' in pc
        assert pc['s8_collapse'].wall_time_s > 0
```

- [ ] **Step 1 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_pipeline.py::TestABComparison -xvs
```

**Expected:** All 3 tests pass. The vertex and face counts should match because `corep_fast/stages/s8_collapse.py::_process_shared_edge_geometry()` replicates the exact same algorithm as `custom/collapse.py::process_shared_edge_geometry()`.

**If tests fail:** Debug by comparing the intermediate outputs:
```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -c "
from corep_fast.pipeline import run_hybrid_pipeline, PipelineConfig
import trimesh

mesh_path = '/tmp/test_sphere.ply'
trimesh.creation.icosphere(subdivisions=2, radius=0.4).export(mesh_path)

r1 = run_hybrid_pipeline(mesh_path, 64, '/tmp/custom.ply', PipelineConfig(s8_impl='custom'))
r2 = run_hybrid_pipeline(mesh_path, 64, '/tmp/fast.ply', PipelineConfig(s8_impl='corep_fast'))

m1 = trimesh.load(r1)
m2 = trimesh.load(r2)
print(f'Custom: V={m1.vertices.shape[0]}, F={m1.faces.shape[0]}')
print(f'Fast:   V={m2.vertices.shape[0]}, F={m2.faces.shape[0]}')
"
```

- [ ] **Step 2 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/tests/unit/test_pipeline.py && \
  git commit -m "test(corep_fast): add A/B integration test — custom vs corep_fast s8"
```

---

## Task 8: Performance Benchmark

**Purpose:** Run a profiled comparison of custom/ s8 vs corep_fast/ s8 on the baseline evaluation set and verify speedup.

- [ ] **Step 0 (test):** Add a benchmark test that reports timing.

Append to `corep_fast/tests/unit/test_pipeline.py`:
```python
class TestPerformanceBenchmark:
    """Performance comparison — not strict assertions, but reports speedup."""

    @pytest.fixture
    def simple_mesh_path(self, tmp_path):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
        path = tmp_path / "bench_sphere.ply"
        mesh.export(str(path))
        return str(path)

    def test_report_speedup(self, simple_mesh_path, tmp_path, capsys):
        """Run both implementations and print timing comparison."""
        from corep_fast.profiling.harness import ProfilingCollector
        import time

        resolution = 128

        # Custom
        cfg_custom = PipelineConfig(s8_impl='custom')
        out_custom = str(tmp_path / "bench_custom.ply")
        pc_custom = ProfilingCollector(mesh_name='bench', resolution=resolution, impl='custom')
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=resolution,
            output_path=out_custom,
            config=cfg_custom,
            collector=pc_custom,
        )

        # corep_fast
        cfg_fast = PipelineConfig(s8_impl='corep_fast')
        out_fast = str(tmp_path / "bench_fast.ply")
        pc_fast = ProfilingCollector(mesh_name='bench', resolution=resolution, impl='corep_fast')
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=resolution,
            output_path=out_fast,
            config=cfg_fast,
            collector=pc_fast,
        )

        t_custom = pc_custom['s8_collapse'].wall_time_s
        t_fast = pc_fast['s8_collapse'].wall_time_s
        speedup = t_custom / t_fast if t_fast > 0 else float('inf')

        print(f"\n--- s8_collapse Benchmark ---")
        print(f"  Custom:     {t_custom:.3f}s")
        print(f"  corep_fast: {t_fast:.3f}s")
        print(f"  Speedup:    {speedup:.1f}x")

        mesh_c = trimesh.load(out_custom)
        mesh_f = trimesh.load(out_fast)
        print(f"  Custom mesh:     V={mesh_c.vertices.shape[0]}, F={mesh_c.faces.shape[0]}")
        print(f"  corep_fast mesh: V={mesh_f.vertices.shape[0]}, F={mesh_f.faces.shape[0]}")

        # No strict speedup assertion in Phase 1a — the torch.unique optimization
        # is the critical path and will show benefit at higher resolution.
        # At low resolution (128), overhead may dominate.
        assert t_custom > 0
        assert t_fast > 0
```

- [ ] **Step 1 (verify):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_pipeline.py::TestPerformanceBenchmark -xvs
```

**Expected:** Test passes and prints timing comparison. At low resolution, speedup may be modest or even negative due to Python overhead in the geometry processing loop. The significant speedup will come at higher resolutions (512+) where vertex_to_index dict is the bottleneck.

- [ ] **Step 2 (commit):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/tests/unit/test_pipeline.py && \
  git commit -m "test(corep_fast): add s8 performance benchmark test"
```

---

## Task 9: Full Test Suite Regression

**Purpose:** Run all existing Phase 0 tests plus new Phase 1a tests to ensure nothing is broken.

- [ ] **Step 0 (verify all tests):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/ -xvs --tb=short
```

**Expected:** All Phase 0 tests (88) + Phase 1a tests pass. No regressions.

- [ ] **Step 1 (final commit with test count):**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/ --tb=short -q 2>&1 | tail -5
```

Record the test count. Expected: ~110+ tests passing.

---

## Implementation Notes

### Why the geometry loop is still Python (for now)

The `_process_shared_edge_geometry()` function is a faithful port of custom/collapse.py's logic — it runs in Python, iterating over cubes and their loops. This is intentional for Phase 1a:

1. **Correctness first:** Matching custom/ behavior exactly is the priority. A Torch-native geometry kernel would require restructuring the entire algorithm.
2. **The real bottleneck is vertex welding:** The profiling shows that the `vertex_to_index` dict and `seen_faces` set are the critical hotspots. The `_weld_and_dedup()` function replaces both with `torch.unique`, which is the key optimization.
3. **Phase 1b will vectorize geometry:** Once A/B equivalence is confirmed, the geometry loop can be rewritten to use CubeBatch CSR fields and batched operations.

### Edge encoding scheme

Global edge keys are encoded as `axis * R^3 + nx * R^2 + ny * R + nz` where `R = resolution + 2`. The +2 headroom prevents negative coordinates after applying neighbor offsets (minimum offset is -1). This encoding is bijective for all valid grid coordinates.

### Vertex welding precision

The default `merge_decimals=5` matches custom/collapse.py. This means vertices within 1e-5 of each other (after rounding) are merged. At resolution 512, the cube side length is ~1/512 ≈ 0.002, so merge_decimals=5 is well within a single cube's interior.

### Exception cube contract

Exception cubes are marked by `{'exception': True}` in the cube data dict. Their first component_point is used as a wildcard that fills into any rank group where the cube's UV position is absent. This allows disconnected geometry to be stitched into the mesh even when the formal loop/rank structure is broken.

The "conditional promotion" logic detects cubes that have components but fail to connect to their neighbors on a shared edge, and promotes them to exception status dynamically. This is a critical correctness feature that prevents holes in the output mesh.
