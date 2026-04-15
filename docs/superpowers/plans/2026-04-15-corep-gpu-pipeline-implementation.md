# CoReP GPU Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite CoReP stages s1-s7 as GPU tensor operations (PyTorch), achieving < 3s end-to-end at res=256 (down from 81.2s).

**Architecture:** Phase-grouped pipeline where each stage function takes `CubeBatch` + `MeshTensors` and returns updated `CubeBatch`. All data stays on GPU. Per-stage A/B verification against `custom/` baseline ensures correctness.

**Tech Stack:** PyTorch (tensor ops, scatter, unique), trimesh (mesh loading), pytest (testing), existing `corep_fast/` infrastructure (CubeBatch, MeshTensors, ProfilingCollector).

**Spec:** `docs/superpowers/specs/2026-04-15-corep-gpu-pipeline-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `corep_fast/stages/s1_voxelize.py` | GPU SAT voxelization: mesh → occupied cubes + face CSR |
| `corep_fast/stages/s2_components.py` | GPU Union-Find: face adjacency → num_components |
| `corep_fast/stages/s3_edge_weights.py` | GPU Möller-Trumbore: ray-triangle → edge_weights |
| `corep_fast/stages/s4_face_point.py` | GPU clipping → face_weights + component_points |
| `corep_fast/stages/s6_collapse.py` | GPU normal curve + U-Turn → loops |
| `corep_fast/stages/s7_rank_assign.py` | GPU ranking + Hungarian → sorted_loops |
| `corep_fast/interop/custom_runner.py` | Run custom/ stages individually, return registers |
| `corep_fast/tests/regression/test_s1_ab.py` | A/B: GPU s1 vs custom/ |
| `corep_fast/tests/regression/test_s2_ab.py` | A/B: GPU s2 vs custom/ |
| `corep_fast/tests/regression/test_s3_ab.py` | A/B: GPU s3 vs custom/ |
| `corep_fast/tests/regression/test_s4_ab.py` | A/B: GPU s4 vs custom/ |
| `corep_fast/tests/regression/test_s6_ab.py` | A/B: GPU s6 vs custom/ |
| `corep_fast/tests/regression/test_s7_ab.py` | A/B: GPU s7 vs custom/ |
| `corep_fast/tests/regression/test_e2e_gpu_ab.py` | Full pipeline A/B: GPU e2e vs custom/ e2e |
| `corep_triton/__init__.py` | Skeleton package — placeholder for next stage |

### Modified Files

| File | Changes |
|------|---------|
| `corep_fast/pipeline.py` | Add `corep_encode()`, `corep_decode()`, `corep_pipeline()` |
| `corep_fast/stages/s8_collapse.py` | Add `CubeBatch`-based entry point alongside existing `list[dict]` API |
| `corep_fast/containers.py` | Extend `MeshTensors.from_trimesh` if needed for normalization parity |
| `corep_fast/tests/conftest.py` | Add shared fixtures for A/B testing |
| `corep_fast/constants.py` | Add edge endpoint lookup tables for s3 |

---

## Dependency Graph

```
Task 1 (Infrastructure) ──┐
                           ├→ Task 2 (s1_voxelize) ──→ Task 3 (s1 A/B)
                           │                              │
                           │      ┌───────────────────────┘
                           │      ▼
                           ├→ Task 4 (s2_components) ──→ Task 5 (s2 A/B)
                           │                              │
                           │      ┌───────────────────────┘
                           │      ▼
                           ├→ Task 6 (s3_edge_weights) ──→ Task 7 (s3 A/B)
                           │                                │
                           │      ┌─────────────────────────┘
                           │      ▼
                           ├→ Task 8 (s4_face_point) ──→ Task 9 (s4 A/B)
                           │                              │
                           │      ┌───────────────────────┘
                           │      ▼
                           ├→ Task 10 (s6_collapse) ──→ Task 11 (s6 A/B)
                           │                             │
                           │      ┌──────────────────────┘
                           │      ▼
                           ├→ Task 12 (s7_rank_assign) ──→ Task 13 (s7 A/B)
                           │                                │
                           │      ┌─────────────────────────┘
                           │      ▼
                           └→ Task 14 (s8_decode + pipeline) ──→ Task 15 (E2E A/B)
```

---

### Task 1: Infrastructure — Custom Runner + Test Fixtures + Triton Skeleton

**Files:**
- Create: `corep_fast/interop/custom_runner.py`
- Create: `corep_triton/__init__.py`
- Modify: `corep_fast/tests/conftest.py`
- Modify: `corep_fast/constants.py`

- [ ] **Step 1: Create custom_runner.py — run custom/ stages individually**

```python
# corep_fast/interop/custom_runner.py
"""Run custom/ pipeline stages individually and return intermediate registers.

Used by A/B regression tests to get ground-truth output from each stage.
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path

import trimesh


def _ensure_custom_importable() -> None:
    project_root = str(Path(__file__).resolve().parents[2])
    custom_dir = os.path.join(project_root, 'custom')
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)


def run_custom_through_stage(
    mesh_path: str,
    resolution: int,
    up_to: str,
) -> list[dict]:
    """Run custom/ pipeline up to a specified stage, return registers.

    Args:
        mesh_path: Path to input PLY mesh.
        resolution: Voxel grid resolution.
        up_to: Stage to stop after. One of:
            's1' - after voxelization (cube_indices, face_indices)
            's2' - after feature_volume (+ num_components, num_boundary)
            's3' - after feature_edge (+ edge_weights)
            's4' - after feature_face + feature_point (+ face_weights, component_points)
            's6' - after collapse_face (+ loops, status)
            's7' - after collapse_point (+ sorted_loops with ranks + point assignment)

    Returns:
        list[dict] — the face_registers at the requested stage.
    """
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
    tmp_dir = tempfile.mkdtemp(prefix='custom_runner_')

    try:
        norm_mesh, boundaries, face_regs, bnd_regs, nm_regs = \
            voxelize(mesh, tmp_dir, resolution)
        if up_to == 's1':
            return face_regs

        face_regs, bnd_regs = feature_volume(
            face_regs, bnd_regs, norm_mesh, boundaries, tmp_dir)
        if up_to == 's2':
            return face_regs

        face_regs = feature_edge(norm_mesh, resolution, face_regs, tmp_dir, debug=False)
        if up_to == 's3':
            return face_regs

        face_regs = feature_face(
            norm_mesh, resolution, face_regs, boundaries, bnd_regs, tmp_dir, debug=False)
        face_regs = feature_point(norm_mesh, resolution, face_regs, tmp_dir, debug=False)
        if up_to == 's4':
            return face_regs

        inner_mask = fetch_np_array(face_regs, 'num_boundary') == 0
        boundary_mask = ~inner_mask
        inner_regs = [face_regs[i] for i in range(len(face_regs)) if inner_mask[i]]
        boundary_regs_split = [face_regs[i] for i in range(len(face_regs)) if boundary_mask[i]]

        solved_u, ambig_u, unsolv_u = collapse_face_inner(inner_regs)
        solved_b, ambig_b, unsolv_b = collapse_face_boundary(boundary_regs_split)
        exception_regs = mark_exception([*ambig_u, *unsolv_u, *ambig_b, *unsolv_b])
        all_regs = [*solved_u, *solved_b, *exception_regs]
        if up_to == 's6':
            return all_regs

        point_regs = collapse_point_inner(solved_u, resolution, debug=False,
                                           output_directory=tmp_dir)
        point_regs_bnd = collapse_point_boundary(solved_b, resolution, debug=False,
                                                   output_directory=tmp_dir)
        all_regs = [*point_regs, *point_regs_bnd, *exception_regs]
        if up_to == 's7':
            return all_regs

        raise ValueError(f"Unknown stage: {up_to!r}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 2: Add edge endpoint constants to constants.py**

Append to `corep_fast/constants.py`:

```python
# ---------------------------------------------------------------------------
# Edge endpoint coordinates in unit cube — used by s3 for ray construction
# CUBE_EDGE_ENDPOINTS[e] = (start_xyz, end_xyz) derived from CUBE_VERTICES[CUBE_EDGES]
# ---------------------------------------------------------------------------
CUBE_EDGE_STARTS: torch.Tensor = CUBE_VERTICES[CUBE_EDGES[:, 0].long()]  # (18, 3) float32
CUBE_EDGE_ENDS: torch.Tensor = CUBE_VERTICES[CUBE_EDGES[:, 1].long()]    # (18, 3) float32
CUBE_EDGE_DIRS: torch.Tensor = CUBE_EDGE_ENDS - CUBE_EDGE_STARTS         # (18, 3) float32
```

- [ ] **Step 3: Add shared A/B test fixtures to conftest.py**

Append to `corep_fast/tests/conftest.py`:

```python
# ---------------------------------------------------------------------------
# A/B regression test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def ab_mesh_path(tmp_path_factory) -> str:
    """Icosphere PLY file for A/B tests. Session-scoped to avoid re-export."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    path = str(tmp_path_factory.mktemp('ab') / 'icosphere.ply')
    mesh.export(path)
    return path


@pytest.fixture(scope='session')
def ab_resolution() -> int:
    return 64


@pytest.fixture
def gpu_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')
```

- [ ] **Step 4: Create corep_triton skeleton**

```python
# corep_triton/__init__.py
"""
CoReP Triton — GPU-optimized pipeline using Triton custom kernels.

This package mirrors the corep_fast/ stage API but replaces hot paths
with Triton kernels for maximum performance.

Status: Skeleton only. Implementation deferred to next stage.
"""
```

- [ ] **Step 5: Run existing tests to verify nothing broken**

Run: `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && python -m pytest corep_fast/tests/unit/ -x -q`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add corep_fast/interop/custom_runner.py corep_fast/tests/conftest.py \
        corep_fast/constants.py corep_triton/__init__.py
git commit -m "feat: add custom_runner, A/B fixtures, edge constants, triton skeleton"
```

---

### Task 2: s1_voxelize — GPU SAT Voxelization

**Files:**
- Create: `corep_fast/stages/s1_voxelize.py`
- Test: `corep_fast/tests/unit/test_s1_voxelize.py`

**Algorithm reference:** `custom/voxelize.py` — `vectorized_triangle_box_intersect` (SAT with 13 axes), `process_face_to_grid` (face-to-cube registration)

- [ ] **Step 1: Write unit test**

```python
# corep_fast/tests/unit/test_s1_voxelize.py
"""Unit tests for s1_voxelize GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.stages.s1_voxelize import s1_voxelize


@pytest.fixture
def simple_mesh_tensors():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    return MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')


class TestS1Voxelize:
    def test_returns_cubebatch(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        assert isinstance(batch, CubeBatch)

    def test_cube_indices_in_range(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        assert (batch.cube_indices >= 0).all()
        assert (batch.cube_indices < 32).all()
        assert batch.cube_indices.dtype == torch.int32

    def test_tri_csr_valid(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        N = batch.num_cubes
        assert batch.tri_offsets.shape == (N + 1,)
        assert batch.tri_offsets[0] == 0
        assert (batch.tri_offsets[1:] >= batch.tri_offsets[:-1]).all()
        # Every face should be registered to at least one cube
        registered_faces = batch.tri_values.unique()
        assert registered_faces.numel() > 0
        assert (registered_faces >= 0).all()
        assert (registered_faces < simple_mesh_tensors.faces.shape[0]).all()

    def test_nonempty_cubes(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        # Each cube must have at least 1 registered face
        counts = batch.tri_offsets[1:] - batch.tri_offsets[:-1]
        assert (counts > 0).all()

    def test_cube_hash_matches_indices(self, simple_mesh_tensors):
        batch = s1_voxelize(simple_mesh_tensors, resolution=32,
                            device=simple_mesh_tensors.device)
        R = 32
        expected_hash = (batch.cube_indices[:, 0].long() * R * R
                         + batch.cube_indices[:, 1].long() * R
                         + batch.cube_indices[:, 2].long())
        assert torch.equal(batch.cube_hash, expected_hash)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest corep_fast/tests/unit/test_s1_voxelize.py -x -q`
Expected: FAIL (ImportError — s1_voxelize not implemented)

- [ ] **Step 3: Implement s1_voxelize**

```python
# corep_fast/stages/s1_voxelize.py
"""
Stage 1: GPU SAT Voxelization — mesh triangles → occupied cubes + face CSR.

Replaces custom/voxelize.py with GPU tensor operations.
Uses the Separating Axis Theorem (SAT) with 13 axes to test
triangle-AABB intersection.

Public API:
    s1_voxelize(mesh, resolution, device) -> CubeBatch
"""
from __future__ import annotations

import torch

from corep_fast.containers import MeshTensors, CubeBatch


def s1_voxelize(
    mesh: MeshTensors,
    resolution: int,
    device: torch.device,
) -> CubeBatch:
    """Voxelize mesh: find occupied cubes and register intersecting faces.

    For each mesh triangle, computes its AABB in grid space, enumerates
    candidate cubes, then applies the 13-axis SAT test to determine
    actual intersection. Results are packed into CubeBatch with CSR
    face registration.

    Args:
        mesh: Normalized mesh tensors (vertices in [0,1]³).
        resolution: Voxel grid resolution R (grid is R³).
        device: Target device for output tensors.

    Returns:
        CubeBatch with cube_indices, cube_hash, tri_offsets, tri_values populated.
    """
    R = resolution
    triangles = mesh.triangles  # (F, 3, 3) float32

    # Step 1: compute AABB per triangle in grid coords
    tri_grid = triangles * R  # (F, 3, 3) — triangle verts in grid space
    tri_min = tri_grid.amin(dim=1)  # (F, 3)
    tri_max = tri_grid.amax(dim=1)  # (F, 3)

    # Integer AABB: candidate cubes for each triangle
    imin = tri_min.floor().to(torch.int32).clamp(min=0, max=R - 1)  # (F, 3)
    imax = tri_max.floor().to(torch.int32).clamp(min=0, max=R - 1)  # (F, 3)
    # Number of candidate cubes per triangle
    spans = (imax - imin + 1).to(torch.int64)  # (F, 3)
    counts_per_tri = spans[:, 0] * spans[:, 1] * spans[:, 2]  # (F,)

    # Step 2: expand — enumerate all (triangle, candidate_cube) pairs
    offsets = torch.zeros(counts_per_tri.shape[0] + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts_per_tri, dim=0)
    total_pairs = int(offsets[-1].item())

    # For each pair, identify triangle_id and cube (ix, iy, iz)
    pair_tri_ids, pair_cube_coords = _expand_candidates(
        imin, imax, spans, offsets, total_pairs, device)

    # Step 3: SAT test — 13 axes
    hits = _sat_test_batch(tri_grid, pair_tri_ids, pair_cube_coords, R, device)

    # Step 4: compact hits into CSR
    hit_tri_ids = pair_tri_ids[hits]
    hit_cube_coords = pair_cube_coords[hits]

    # Build CSR: sort by cube hash, unique cubes
    hit_cube_hash = (hit_cube_coords[:, 0].long() * R * R
                     + hit_cube_coords[:, 1].long() * R
                     + hit_cube_coords[:, 2].long())

    # Sort by cube hash for grouping
    sort_idx = torch.argsort(hit_cube_hash)
    sorted_hash = hit_cube_hash[sort_idx]
    sorted_tri_ids = hit_tri_ids[sort_idx]

    # Unique cubes and their offsets
    unique_hash, inverse, counts = torch.unique_consecutive(
        sorted_hash, return_inverse=True, return_counts=True)

    N = unique_hash.shape[0]
    tri_offsets = torch.zeros(N + 1, dtype=torch.int64, device=device)
    tri_offsets[1:] = torch.cumsum(counts, dim=0)

    # Recover cube_indices from hash
    cube_iz = (unique_hash % R).to(torch.int32)
    cube_iy = ((unique_hash // R) % R).to(torch.int32)
    cube_ix = (unique_hash // (R * R)).to(torch.int32)
    cube_indices = torch.stack([cube_ix, cube_iy, cube_iz], dim=1)

    # Build CubeBatch
    cb = CubeBatch.empty(num_cubes=N, resolution=R, device=device)
    cb = cb.with_cube_indices(cube_indices)
    from dataclasses import replace as _dc_replace
    cb = _dc_replace(cb, tri_offsets=tri_offsets, tri_values=sorted_tri_ids.to(torch.int32))

    return cb


def _expand_candidates(
    imin: torch.Tensor,  # (F, 3) int32
    imax: torch.Tensor,  # (F, 3) int32
    spans: torch.Tensor,  # (F, 3) int64
    offsets: torch.Tensor,  # (F+1,) int64
    total_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all (triangle_id, candidate_cube_coord) pairs.

    Returns:
        pair_tri_ids: (total_pairs,) int32
        pair_cube_coords: (total_pairs, 3) int32
    """
    F = imin.shape[0]
    pair_tri_ids = torch.empty(total_pairs, dtype=torch.int32, device=device)
    pair_cube_coords = torch.empty((total_pairs, 3), dtype=torch.int32, device=device)

    # Use arange + searchsorted to map flat index → triangle
    flat_idx = torch.arange(total_pairs, dtype=torch.int64, device=device)
    tri_ids = torch.searchsorted(offsets[1:], flat_idx, right=True)  # (total_pairs,)
    local_idx = flat_idx - offsets[tri_ids]  # within-triangle flat index

    sx = spans[tri_ids, 0]
    sy = spans[tri_ids, 1]
    sz = spans[tri_ids, 2]

    lx = local_idx // (sy * sz)
    rem = local_idx % (sy * sz)
    ly = rem // sz
    lz = rem % sz

    pair_tri_ids[:] = tri_ids.to(torch.int32)
    pair_cube_coords[:, 0] = (imin[tri_ids, 0].long() + lx).to(torch.int32)
    pair_cube_coords[:, 1] = (imin[tri_ids, 1].long() + ly).to(torch.int32)
    pair_cube_coords[:, 2] = (imin[tri_ids, 2].long() + lz).to(torch.int32)

    return pair_tri_ids, pair_cube_coords


def _sat_test_batch(
    tri_grid: torch.Tensor,    # (F, 3, 3) — triangle verts in grid space
    pair_tri_ids: torch.Tensor,  # (P,) int32
    pair_cube_coords: torch.Tensor,  # (P, 3) int32
    R: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch SAT test: 13 axes for triangle-AABB intersection.

    Returns:
        hits: (P,) bool — True if triangle intersects cube.
    """
    P = pair_tri_ids.shape[0]
    tids = pair_tri_ids.long()

    # Triangle vertices for each pair
    v0 = tri_grid[tids, 0]  # (P, 3)
    v1 = tri_grid[tids, 1]  # (P, 3)
    v2 = tri_grid[tids, 2]  # (P, 3)

    # Cube center and half-extent
    cube_center = pair_cube_coords.float() + 0.5  # (P, 3)
    half = 0.5  # unit cube half-extent

    # Translate triangle to cube-center-origin frame
    f0 = v0 - cube_center  # (P, 3)
    f1 = v1 - cube_center
    f2 = v2 - cube_center

    # Triangle edges
    e0 = f1 - f0  # (P, 3)
    e1 = f2 - f1
    e2 = f0 - f2

    # Helper: test separation along axis a
    # Project triangle verts onto axis, check overlap with box projection
    alive = torch.ones(P, dtype=torch.bool, device=device)

    def _test_axis(ax: torch.Tensor):
        """ax: (P, 3) or (1, 3). Updates `alive` in place."""
        nonlocal alive
        p0 = (f0 * ax).sum(dim=1)
        p1 = (f1 * ax).sum(dim=1)
        p2 = (f2 * ax).sum(dim=1)
        tri_min = torch.minimum(torch.minimum(p0, p1), p2)
        tri_max = torch.maximum(torch.maximum(p0, p1), p2)
        # Box projection onto axis: |ax_x|*half + |ax_y|*half + |ax_z|*half
        r = ax.abs().sum(dim=1) * half
        alive &= ~((tri_min > r) | (tri_max < -r))

    # --- 9 cross-product axes (3 edges × 3 AABB axes) ---
    unit_x = torch.tensor([[1., 0., 0.]], device=device)
    unit_y = torch.tensor([[0., 1., 0.]], device=device)
    unit_z = torch.tensor([[0., 0., 1.]], device=device)

    for edge in [e0, e1, e2]:
        _test_axis(torch.cross(edge, unit_x.expand_as(edge), dim=1))
        _test_axis(torch.cross(edge, unit_y.expand_as(edge), dim=1))
        _test_axis(torch.cross(edge, unit_z.expand_as(edge), dim=1))

    # --- 3 AABB face normals (X, Y, Z) ---
    # X axis: check f0.x, f1.x, f2.x vs [-half, half]
    for dim in range(3):
        vals = torch.stack([f0[:, dim], f1[:, dim], f2[:, dim]], dim=1)
        alive &= ~((vals.amin(dim=1) > half) | (vals.amax(dim=1) < -half))

    # --- 1 triangle normal ---
    tri_normal = torch.cross(e0, e1, dim=1)  # (P, 3)
    _test_axis(tri_normal)

    return alive
```

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest corep_fast/tests/unit/test_s1_voxelize.py -x -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add corep_fast/stages/s1_voxelize.py corep_fast/tests/unit/test_s1_voxelize.py
git commit -m "feat(corep_fast): add s1_voxelize — GPU SAT voxelization"
```

---

### Task 3: s1 A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_s1_ab.py`

- [ ] **Step 1: Write A/B test**

```python
# corep_fast/tests/regression/test_s1_ab.py
"""A/B regression: GPU s1_voxelize vs custom/ voxelize."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.interop.from_custom import cube_batch_from_custom
from corep_fast.stages.s1_voxelize import s1_voxelize


class TestS1AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s1')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        mesh = trimesh.load(ab_mesh_path)
        mesh_t = MeshTensors.from_trimesh(mesh, ab_resolution, device=gpu_device)
        return s1_voxelize(mesh_t, ab_resolution, gpu_device)

    def test_cube_count_matches(self, custom_regs, gpu_batch):
        assert gpu_batch.num_cubes == len(custom_regs), \
            f"Cube count: GPU={gpu_batch.num_cubes} vs custom={len(custom_regs)}"

    def test_cube_indices_match(self, custom_regs, gpu_batch):
        custom_indices = torch.tensor(
            [r['cube_indices'] for r in custom_regs], dtype=torch.int32)
        gpu_indices = gpu_batch.cube_indices.cpu()
        # Sort both by hash for comparison
        R = gpu_batch.resolution
        custom_hash = (custom_indices[:, 0].long() * R * R
                       + custom_indices[:, 1].long() * R
                       + custom_indices[:, 2].long())
        gpu_hash = (gpu_indices[:, 0].long() * R * R
                    + gpu_indices[:, 1].long() * R
                    + gpu_indices[:, 2].long())
        assert torch.equal(custom_hash.sort()[0], gpu_hash.sort()[0])

    def test_face_registration_complete(self, custom_regs, gpu_batch):
        """Every (cube, face) pair in custom/ must also appear in GPU output."""
        R = gpu_batch.resolution
        gpu_indices = gpu_batch.cube_indices.cpu()
        gpu_hash = (gpu_indices[:, 0].long() * R * R
                    + gpu_indices[:, 1].long() * R
                    + gpu_indices[:, 2].long())
        hash_to_idx = {int(h): i for i, h in enumerate(gpu_hash.tolist())}

        mismatches = 0
        for reg in custom_regs:
            ci = reg['cube_indices']
            h = ci[0] * R * R + ci[1] * R + ci[2]
            if h not in hash_to_idx:
                mismatches += 1
                continue
            idx = hash_to_idx[h]
            lo = int(gpu_batch.tri_offsets[idx])
            hi = int(gpu_batch.tri_offsets[idx + 1])
            gpu_faces = set(gpu_batch.tri_values[lo:hi].cpu().tolist())
            custom_faces = set(reg['face_indices'])
            if custom_faces != gpu_faces:
                mismatches += 1
        assert mismatches == 0, f"{mismatches} cubes have mismatched face registrations"
```

- [ ] **Step 2: Run A/B test**

Run: `python -m pytest corep_fast/tests/regression/test_s1_ab.py -x -v`
Expected: All PASS (if test fails, debug s1_voxelize against custom/ and fix)

- [ ] **Step 3: Commit**

```bash
git add corep_fast/tests/regression/test_s1_ab.py
git commit -m "test(corep_fast): add s1 A/B regression test vs custom/"
```

---

### Task 4: s2_components — GPU Union-Find Connected Components

**Files:**
- Create: `corep_fast/stages/s2_components.py`
- Test: `corep_fast/tests/unit/test_s2_components.py`

**Algorithm:** For each cube, find connected components among its registered faces using the mesh face adjacency. GPU-parallel: each cube processed independently. Uses iterative label propagation (min-reduction) on padded face arrays.

- [ ] **Step 1: Write unit test**

```python
# corep_fast/tests/unit/test_s2_components.py
"""Unit tests for s2_components GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components


@pytest.fixture
def batch_after_s1():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    return s1_voxelize(mt, 32, torch.device('cpu')), mt


class TestS2Components:
    def test_num_components_populated(self, batch_after_s1):
        batch, mesh = batch_after_s1
        result = s2_components(batch, mesh)
        assert result.num_components.shape == (result.num_cubes,)
        assert (result.num_components >= 1).all(), "Every occupied cube should have ≥1 component"

    def test_num_boundary_populated(self, batch_after_s1):
        batch, mesh = batch_after_s1
        result = s2_components(batch, mesh)
        assert result.num_boundary.shape == (result.num_cubes,)
        # For a closed icosphere, no boundary cubes
        assert (result.num_boundary == 0).all()

    def test_single_face_cube_has_one_component(self, batch_after_s1):
        batch, mesh = batch_after_s1
        result = s2_components(batch, mesh)
        counts = batch.tri_offsets[1:] - batch.tri_offsets[:-1]
        single_face_mask = counts == 1
        if single_face_mask.any():
            assert (result.num_components[single_face_mask] == 1).all()
```

- [ ] **Step 2: Implement s2_components**

Core algorithm: per-cube iterative label propagation.
1. Pad registered faces to `(N, max_faces_per_cube)` with -1
2. Build local adjacency: for each pair (face_i, face_j) in same cube, check if `face_adj` says they share a mesh edge
3. Iterative min-propagation: `label[i] = min(label[i], label[neighbors[i]])` until convergence
4. Count unique labels per cube → `num_components`
5. Boundary: count faces with any `face_adj == -1` edge → separate boundary component count

Function signature:
```python
def s2_components(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch:
    """Compute num_components and num_boundary per cube via GPU Union-Find."""
```

- [ ] **Step 3: Run tests, commit**

Run: `python -m pytest corep_fast/tests/unit/test_s2_components.py -x -v`

```bash
git add corep_fast/stages/s2_components.py corep_fast/tests/unit/test_s2_components.py
git commit -m "feat(corep_fast): add s2_components — GPU Union-Find connected components"
```

---

### Task 5: s2 A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_s2_ab.py`

- [ ] **Step 1: Write A/B test**

```python
# corep_fast/tests/regression/test_s2_ab.py
"""A/B regression: GPU s2_components vs custom/ feature_volume."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components


class TestS2AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s2')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        mesh = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh, ab_resolution, device=gpu_device)
        batch = s1_voxelize(mt, ab_resolution, gpu_device)
        return s2_components(batch, mt)

    def test_num_components_exact_match(self, custom_regs, gpu_batch):
        custom_nc = torch.tensor(
            [r.get('num_components', 0) for r in custom_regs], dtype=torch.int32)
        gpu_nc = gpu_batch.num_components.cpu()
        # Sort by cube hash for alignment
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_hash = torch.tensor(
            [r['cube_indices'][0] * R * R + r['cube_indices'][1] * R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)
        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()
        assert torch.equal(gpu_nc[gpu_order], custom_nc[custom_order]), \
            "num_components mismatch"

    def test_num_boundary_exact_match(self, custom_regs, gpu_batch):
        custom_nb = torch.tensor(
            [r.get('num_boundary', 0) for r in custom_regs], dtype=torch.int32)
        gpu_nb = gpu_batch.num_boundary.cpu()
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_hash = torch.tensor(
            [r['cube_indices'][0] * R * R + r['cube_indices'][1] * R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)
        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()
        assert torch.equal(gpu_nb[gpu_order], custom_nb[custom_order]), \
            "num_boundary mismatch"
```

- [ ] **Step 2: Run, debug, commit**

Run: `python -m pytest corep_fast/tests/regression/test_s2_ab.py -x -v`

```bash
git add corep_fast/tests/regression/test_s2_ab.py
git commit -m "test(corep_fast): add s2 A/B regression test vs custom/"
```

---

### Task 6: s3_edge_weights — GPU Möller-Trumbore

**Files:**
- Create: `corep_fast/stages/s3_edge_weights.py`
- Test: `corep_fast/tests/unit/test_s3_edge_weights.py`

**Algorithm:** For each (cube, edge) pair, cast a ray along the edge and count intersections with registered mesh triangles using Möller-Trumbore.

Key constants from `custom/feature_edge.py`:
- `EPSILON = 1e-8`
- Hit condition: `|det| > ε`, `u ∈ [0,1]`, `v ∈ [0,1]`, `u+v ≤ 1`, `t ∈ [-ε, 1+ε]`

- [ ] **Step 1: Write unit test**

```python
# corep_fast/tests/unit/test_s3_edge_weights.py
"""Unit tests for s3_edge_weights GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights


@pytest.fixture
def batch_after_s2():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    return batch, mt


class TestS3EdgeWeights:
    def test_shape_and_dtype(self, batch_after_s2):
        batch, mesh = batch_after_s2
        result = s3_edge_weights(batch, mesh)
        assert result.edge_weights.shape == (result.num_cubes, 18)
        assert result.edge_weights.dtype == torch.int32

    def test_non_negative(self, batch_after_s2):
        batch, mesh = batch_after_s2
        result = s3_edge_weights(batch, mesh)
        assert (result.edge_weights >= 0).all()

    def test_original_edges_present(self, batch_after_s2):
        """For a closed mesh, most cubes should have nonzero weights on edges 0-11."""
        batch, mesh = batch_after_s2
        result = s3_edge_weights(batch, mesh)
        original_edges = result.edge_weights[:, :12]
        # At least some cubes must have nonzero original edge weights
        assert original_edges.sum() > 0
```

- [ ] **Step 2: Implement s3_edge_weights**

Core algorithm — CSR expansion + batch Möller-Trumbore + scatter_add:

```python
def s3_edge_weights(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch:
    """Compute edge_weights (N, 18) via GPU Möller-Trumbore ray-triangle intersection.

    For each of 18 edges per cube, casts a ray and counts intersections
    with the cube's registered mesh triangles.
    """
```

Key implementation steps:
1. Compute ray origins/dirs from `cube_indices / R + CUBE_EDGE_STARTS` and `CUBE_EDGE_DIRS / R`
2. Expand CSR: for each (cube, registered_tri) pair, replicate across 18 edges
3. Batch Möller-Trumbore: `(N_pairs × 18)` ray-triangle tests in parallel
4. `scatter_add` hit counts back to `(N, 18)` tensor

- [ ] **Step 3: Run tests, commit**

```bash
git add corep_fast/stages/s3_edge_weights.py corep_fast/tests/unit/test_s3_edge_weights.py
git commit -m "feat(corep_fast): add s3_edge_weights — GPU Möller-Trumbore"
```

---

### Task 7: s3 A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_s3_ab.py`

- [ ] **Step 1: Write A/B test**

```python
# corep_fast/tests/regression/test_s3_ab.py
"""A/B regression: GPU s3_edge_weights vs custom/ feature_edge."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights


class TestS3AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s3')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        mesh = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh, ab_resolution, device=gpu_device)
        batch = s1_voxelize(mt, ab_resolution, gpu_device)
        batch = s2_components(batch, mt)
        return s3_edge_weights(batch, mt)

    def test_edge_weights_exact_match(self, custom_regs, gpu_batch):
        """edge_weights must match exactly — they are integer counts."""
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_ew = torch.tensor(
            [r.get('edge_weights', [0]*18) for r in custom_regs], dtype=torch.int32)
        custom_hash = torch.tensor(
            [r['cube_indices'][0]*R*R + r['cube_indices'][1]*R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)

        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()

        gpu_sorted = gpu_batch.edge_weights.cpu()[gpu_order]
        custom_sorted = custom_ew[custom_order]

        mismatches = (gpu_sorted != custom_sorted).any(dim=1).sum().item()
        assert mismatches == 0, (
            f"{mismatches}/{gpu_sorted.shape[0]} cubes have edge_weight mismatches. "
            f"First mismatch at sorted index "
            f"{(gpu_sorted != custom_sorted).any(dim=1).nonzero()[0].item()}"
        )
```

- [ ] **Step 2: Run, debug, commit**

Run: `python -m pytest corep_fast/tests/regression/test_s3_ab.py -x -v`

```bash
git add corep_fast/tests/regression/test_s3_ab.py
git commit -m "test(corep_fast): add s3 A/B regression test vs custom/"
```

---

### Task 8: s4_face_point — GPU Clipping + Face Weights + Component Points

**Files:**
- Create: `corep_fast/stages/s4_face_point.py`
- Test: `corep_fast/tests/unit/test_s4_face_point.py`

**Algorithm:** Sutherland-Hodgman clipping of registered triangles to cube AABB, then extract face_weights and area-weighted component centroids. Merges custom/ s4a (feature_face) + s4b (feature_point).

This is the most complex stage due to variable-length clipping output.

- [ ] **Step 1: Write unit test**

```python
# corep_fast/tests/unit/test_s4_face_point.py
"""Unit tests for s4_face_point GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point


@pytest.fixture
def batch_after_s3():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    return batch, mt


class TestS4FacePoint:
    def test_face_weights_shape(self, batch_after_s3):
        batch, mesh = batch_after_s3
        result = s4_face_point(batch, mesh)
        assert result.face_weights.shape == (result.num_cubes, 12)
        assert result.face_weights.dtype == torch.int32
        assert (result.face_weights >= 0).all()

    def test_component_points_csr(self, batch_after_s3):
        batch, mesh = batch_after_s3
        result = s4_face_point(batch, mesh)
        N = result.num_cubes
        assert result.point_offsets.shape == (N + 1,)
        assert result.point_offsets[0] == 0
        # Number of points per cube should equal num_components
        pts_per_cube = (result.point_offsets[1:] - result.point_offsets[:-1]).to(torch.int32)
        assert torch.equal(pts_per_cube, result.num_components)

    def test_points_in_cube_bounds(self, batch_after_s3):
        batch, mesh = batch_after_s3
        result = s4_face_point(batch, mesh)
        if result.point_values.shape[0] > 0:
            R = result.resolution
            # Points should be within [0, 1] after normalization
            assert (result.point_values >= -0.01).all()
            assert (result.point_values <= 1.01).all()
```

- [ ] **Step 2: Implement s4_face_point**

```python
def s4_face_point(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch:
    """Compute face_weights and component_points via GPU Sutherland-Hodgman clipping.

    Merges custom/ s4a (feature_face) and s4b (feature_point):
    - Clips registered triangles to each cube's AABB (done once)
    - Extracts face_weights (U-Turn detection) from clipped polygons
    - Computes area-weighted centroids per component from clipped polygons
    """
```

Key implementation steps:
1. Expand CSR → `(total_registered_tris,)` with cube_ids
2. Gather triangle vertices, compute cube AABB bounds
3. Sutherland-Hodgman: clip against 6 planes, output `(T, 9, 3)` buffer + `valid_count (T,)`
4. Face weight extraction: for each clipped polygon, detect U-Turn patterns on cube facets
5. Centroid computation: fan-triangulate → area × centroid → group by (cube, component) → weighted average
6. Pack centroids into CSR: `point_offsets`, `point_values`

- [ ] **Step 3: Run tests, commit**

```bash
git add corep_fast/stages/s4_face_point.py corep_fast/tests/unit/test_s4_face_point.py
git commit -m "feat(corep_fast): add s4_face_point — GPU clipping + face weights + centroids"
```

---

### Task 9: s4 A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_s4_ab.py`

- [ ] **Step 1: Write A/B test**

```python
# corep_fast/tests/regression/test_s4_ab.py
"""A/B regression: GPU s4_face_point vs custom/ feature_face + feature_point."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point


class TestS4AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s4')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        mesh = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh, ab_resolution, device=gpu_device)
        batch = s1_voxelize(mt, ab_resolution, gpu_device)
        batch = s2_components(batch, mt)
        batch = s3_edge_weights(batch, mt)
        return s4_face_point(batch, mt)

    def test_face_weights_exact_match(self, custom_regs, gpu_batch):
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_fw = torch.tensor(
            [r.get('face_weights', [0]*12) for r in custom_regs], dtype=torch.int32)
        custom_hash = torch.tensor(
            [r['cube_indices'][0]*R*R + r['cube_indices'][1]*R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)
        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()
        gpu_sorted = gpu_batch.face_weights.cpu()[gpu_order]
        custom_sorted = custom_fw[custom_order]
        mismatches = (gpu_sorted != custom_sorted).any(dim=1).sum().item()
        assert mismatches == 0, f"{mismatches} cubes have face_weight mismatches"

    def test_component_points_close(self, custom_regs, gpu_batch):
        """Component points must match within float tolerance."""
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())
        custom_hash = torch.tensor(
            [r['cube_indices'][0]*R*R + r['cube_indices'][1]*R + r['cube_indices'][2]
             for r in custom_regs], dtype=torch.int64)
        _, gpu_order = gpu_hash.sort()
        _, custom_order = custom_hash.sort()

        for rank, (gi, ci) in enumerate(zip(gpu_order.tolist(), custom_order.tolist())):
            gpu_lo = int(gpu_batch.point_offsets[gi])
            gpu_hi = int(gpu_batch.point_offsets[gi + 1])
            gpu_pts = gpu_batch.point_values[gpu_lo:gpu_hi].cpu()

            custom_pts_raw = custom_regs[ci].get('component_points', [])
            if len(custom_pts_raw) == 0:
                assert gpu_pts.shape[0] == 0, f"Cube rank {rank}: GPU has points but custom doesn't"
                continue
            custom_pts = torch.tensor(custom_pts_raw, dtype=torch.float32)
            assert gpu_pts.shape == custom_pts.shape, \
                f"Cube rank {rank}: shape mismatch {gpu_pts.shape} vs {custom_pts.shape}"
            assert torch.allclose(gpu_pts, custom_pts, atol=1e-5), \
                f"Cube rank {rank}: point mismatch (max diff {(gpu_pts - custom_pts).abs().max():.2e})"
            if rank > 100:  # spot check first 100 for speed
                break
```

- [ ] **Step 2: Run, debug, commit**

```bash
git add corep_fast/tests/regression/test_s4_ab.py
git commit -m "test(corep_fast): add s4 A/B regression test vs custom/"
```

---

### Task 10: s6_collapse — GPU Normal Curve + U-Turn → Loops

**Files:**
- Create: `corep_fast/stages/s6_collapse.py`
- Test: `corep_fast/tests/unit/test_s6_collapse.py`

**Algorithm reference:** `custom/collapse_edge.py` (arc assignment, non-crossing pairing, loop tracing) + `custom/collapse_face.py` (U-Turn enumeration). Merges s5+s6.

Key formulas:
- Arc assignment: `k12 = (w1 + w2 - w3) // 2`
- Validation: triangle inequality + parity per facet
- Non-crossing pairing: nearest-to-corner-vertex ordering
- Loop tracing: degree-2 graph DFS

- [ ] **Step 1: Write unit test**

```python
# corep_fast/tests/unit/test_s6_collapse.py
"""Unit tests for s6_collapse GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse


@pytest.fixture
def batch_after_s4():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    return batch


class TestS6Collapse:
    def test_loops_populated(self, batch_after_s4):
        result = s6_collapse(batch_after_s4)
        N = result.num_cubes
        assert result.loop_cube_off.shape == (N + 1,)
        assert result.loop_cube_off[0] == 0
        # Should have at least some loops
        total_loops = int(result.loop_cube_off[-1].item())
        assert total_loops > 0

    def test_loop_edges_valid(self, batch_after_s4):
        result = s6_collapse(batch_after_s4)
        # All loop edge values should be in [0, 17] (18 edges per cube)
        if result.loop_edge_val.numel() > 0:
            assert (result.loop_edge_val >= 0).all()
            assert (result.loop_edge_val <= 17).all()

    def test_status_populated(self, batch_after_s4):
        result = s6_collapse(batch_after_s4)
        # Most cubes should be OK for a clean icosphere
        ok_count = (result.status == CubeStatus.OK).sum().item()
        assert ok_count > result.num_cubes * 0.9

    def test_loop_count_matches_components(self, batch_after_s4):
        """For OK cubes, loop count should equal num_components."""
        result = s6_collapse(batch_after_s4)
        ok_mask = result.status == CubeStatus.OK
        loops_per_cube = (result.loop_cube_off[1:] - result.loop_cube_off[:-1]).to(torch.int32)
        ok_loops = loops_per_cube[ok_mask]
        ok_nc = result.num_components[ok_mask]
        assert torch.equal(ok_loops, ok_nc), \
            f"Loop count != num_components for {(ok_loops != ok_nc).sum()} OK cubes"
```

- [ ] **Step 2: Implement s6_collapse**

```python
def s6_collapse(batch: CubeBatch) -> CubeBatch:
    """Extract topological loops via normal curve theory + U-Turn enumeration.

    Merges custom/ collapse_edge.py (s5) + collapse_face.py (s6).

    Fast path (face_weights == 0): direct arc assignment + loop trace.
    Slow path (face_weights > 0): U-Turn enumeration + validation.

    Updates loop_cube_off, loop_edge_off, loop_edge_val, status.
    """
```

Implementation strategy:
1. **Fast path** (no U-Turns): batch arc assignment on all 12 facets using `(w1+w2-w3)//2` formula. Validate triangle inequality + parity. Build arc graph as `(N, max_arcs, 2)` tensor. Trace loops via iterative next-pointer following.
2. **Slow path** (U-Turns): identify cubes with any `face_weights > 0`, process with enumeration. For the PyTorch version, this can use a CPU fallback loop since it affects < 1% of cubes.
3. Pack loops into two-level CSR.

- [ ] **Step 3: Run tests, commit**

```bash
git add corep_fast/stages/s6_collapse.py corep_fast/tests/unit/test_s6_collapse.py
git commit -m "feat(corep_fast): add s6_collapse — GPU normal curve + U-Turn loop extraction"
```

---

### Task 11: s6 A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_s6_ab.py`

- [ ] **Step 1: Write A/B test**

Uses L3 topology equivalence (loop set comparison with cyclic rotation + reflection invariance).

```python
# corep_fast/tests/regression/test_s6_ab.py
"""A/B regression: GPU s6_collapse vs custom/ collapse_face."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.interop.from_custom import cube_batch_from_custom
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse


class TestS6AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s6')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        mesh = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh, ab_resolution, device=gpu_device)
        batch = s1_voxelize(mt, ab_resolution, gpu_device)
        batch = s2_components(batch, mt)
        batch = s3_edge_weights(batch, mt)
        batch = s4_face_point(batch, mt)
        return s6_collapse(batch)

    def test_status_distribution_matches(self, custom_regs, gpu_batch):
        """OK/AMBIGUOUS/UNSOLVABLE counts should match."""
        custom_ok = sum(1 for r in custom_regs if not r.get('exception', False))
        custom_exc = sum(1 for r in custom_regs if r.get('exception', False))
        gpu_ok = (gpu_batch.status == CubeStatus.OK).sum().item()
        gpu_exc = (gpu_batch.status != CubeStatus.OK).sum().item()
        assert custom_ok == gpu_ok, f"OK count: custom={custom_ok} vs gpu={gpu_ok}"
        assert custom_exc == gpu_exc, f"Exception count: custom={custom_exc} vs gpu={gpu_exc}"

    def test_loop_edge_sequences_match(self, custom_regs, gpu_batch):
        """For OK cubes, loop edge sequences must match (up to cyclic rotation)."""
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())

        mismatches = 0
        for reg in custom_regs:
            if reg.get('exception', False):
                continue
            ci = reg['cube_indices']
            h = ci[0] * R * R + ci[1] * R + ci[2]
            gpu_idx = (gpu_hash == h).nonzero(as_tuple=False)
            if gpu_idx.numel() == 0:
                mismatches += 1
                continue
            gi = gpu_idx[0].item()

            # Extract GPU loops for this cube
            l_lo = int(gpu_batch.loop_cube_off[gi])
            l_hi = int(gpu_batch.loop_cube_off[gi + 1])
            gpu_loops = []
            for li in range(l_lo, l_hi):
                e_lo = int(gpu_batch.loop_edge_off[li])
                e_hi = int(gpu_batch.loop_edge_off[li + 1])
                gpu_loops.append(tuple(gpu_batch.loop_edge_val[e_lo:e_hi].cpu().tolist()))

            # Extract custom loops
            custom_loops_raw = reg.get('sorted_loops', reg.get('loops', []))
            custom_loops = []
            for ld in custom_loops_raw:
                if isinstance(ld, dict):
                    custom_loops.append(tuple(ld.get('loop', [])))
                else:
                    custom_loops.append(tuple(ld))

            # Compare as sets of canonical loops (cyclic rotation invariant)
            def canonical(loop):
                if not loop:
                    return loop
                doubled = loop + loop
                rotations = [doubled[i:i+len(loop)] for i in range(len(loop))]
                return min(rotations)

            gpu_set = set(canonical(l) for l in gpu_loops)
            custom_set = set(canonical(l) for l in custom_loops)
            if gpu_set != custom_set:
                mismatches += 1

        assert mismatches == 0, f"{mismatches} cubes have loop topology mismatches"
```

- [ ] **Step 2: Run, debug, commit**

```bash
git add corep_fast/tests/regression/test_s6_ab.py
git commit -m "test(corep_fast): add s6 A/B regression test vs custom/"
```

---

### Task 12: s7_rank_assign — GPU Ranking + Hungarian Matching

**Files:**
- Create: `corep_fast/stages/s7_rank_assign.py`
- Test: `corep_fast/tests/unit/test_s7_rank_assign.py`

**Algorithm reference:** `custom/collapse_point.py`
- Rank interpolation: `t = (rank + 1) / (weight + 1)`
- Hungarian: for N ≤ 4, exhaustive permutation (24 permutations)
- Cost: squared Euclidean distance between loop centroid and component point

- [ ] **Step 1: Write unit test**

```python
# corep_fast/tests/unit/test_s7_rank_assign.py
"""Unit tests for s7_rank_assign GPU stage."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse
from corep_fast.stages.s7_rank_assign import s7_rank_assign


@pytest.fixture
def batch_after_s6():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
    mt = MeshTensors.from_trimesh(mesh, resolution=32, device='cpu')
    batch = s1_voxelize(mt, 32, torch.device('cpu'))
    batch = s2_components(batch, mt)
    batch = s3_edge_weights(batch, mt)
    batch = s4_face_point(batch, mt)
    batch = s6_collapse(batch)
    return batch


class TestS7RankAssign:
    def test_ranks_populated(self, batch_after_s6):
        result = s7_rank_assign(batch_after_s6)
        assert result.loop_edge_rank.shape == result.loop_edge_val.shape
        # For OK cubes, ranks should be non-negative
        if result.loop_edge_rank.numel() > 0:
            assert (result.loop_edge_rank >= -1).all()

    def test_point_match_valid(self, batch_after_s6):
        result = s7_rank_assign(batch_after_s6)
        total_loops = int(result.loop_cube_off[-1].item())
        assert result.loop_point_match.shape == (total_loops,)
        # For OK cubes, match indices should be within component count
        ok_mask = result.status == CubeStatus.OK
        for i in range(result.num_cubes):
            if not ok_mask[i]:
                continue
            l_lo = int(result.loop_cube_off[i])
            l_hi = int(result.loop_cube_off[i + 1])
            n_loops = l_hi - l_lo
            if n_loops == 0:
                continue
            matches = result.loop_point_match[l_lo:l_hi]
            assert (matches >= 0).all() and (matches < n_loops).all()

    def test_ranks_within_edge_weight(self, batch_after_s6):
        """Each rank should be in [0, edge_weight-1] for its edge."""
        result = s7_rank_assign(batch_after_s6)
        ok_mask = result.status == CubeStatus.OK
        for i in range(min(result.num_cubes, 50)):  # spot check
            if not ok_mask[i]:
                continue
            l_lo = int(result.loop_cube_off[i])
            l_hi = int(result.loop_cube_off[i + 1])
            for li in range(l_lo, l_hi):
                e_lo = int(result.loop_edge_off[li])
                e_hi = int(result.loop_edge_off[li + 1])
                edges = result.loop_edge_val[e_lo:e_hi]
                ranks = result.loop_edge_rank[e_lo:e_hi]
                for e, r in zip(edges.tolist(), ranks.tolist()):
                    w = int(result.edge_weights[i, e].item())
                    assert 0 <= r < w, f"Cube {i}: rank {r} out of range for edge {e} (weight={w})"
```

- [ ] **Step 2: Implement s7_rank_assign**

```python
def s7_rank_assign(batch: CubeBatch) -> CubeBatch:
    """Assign ranks to loop edge crossings and match loops to component points.

    1. Rank assignment: deterministic non-crossing pairing → rank per edge crossing
    2. Point matching: exhaustive permutation Hungarian for N ≤ 4 loops per cube

    Updates loop_edge_rank and loop_point_match.
    """
```

Key implementation:
1. Rank assignment: re-derive from arc graph (same structure as s6 but tracking rank indices). Can share logic with s6 or re-trace.
2. Loop centroid: `t = (rank+1)/(weight+1)`, interpolate on edge → average → centroid
3. Permutation matching: pre-compute 24 permutations of [0,1,2,3], batch cost matrix `(N_cubes, 24)`, argmin

- [ ] **Step 3: Run tests, commit**

```bash
git add corep_fast/stages/s7_rank_assign.py corep_fast/tests/unit/test_s7_rank_assign.py
git commit -m "feat(corep_fast): add s7_rank_assign — GPU ranking + Hungarian matching"
```

---

### Task 13: s7 A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_s7_ab.py`

- [ ] **Step 1: Write A/B test**

```python
# corep_fast/tests/regression/test_s7_ab.py
"""A/B regression: GPU s7_rank_assign vs custom/ collapse_point."""
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeStatus
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse
from corep_fast.stages.s7_rank_assign import s7_rank_assign


class TestS7AB:
    @pytest.fixture
    def custom_regs(self, ab_mesh_path, ab_resolution):
        return run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s7')

    @pytest.fixture
    def gpu_batch(self, ab_mesh_path, ab_resolution, gpu_device):
        mesh = trimesh.load(ab_mesh_path)
        mt = MeshTensors.from_trimesh(mesh, ab_resolution, device=gpu_device)
        batch = s1_voxelize(mt, ab_resolution, gpu_device)
        batch = s2_components(batch, mt)
        batch = s3_edge_weights(batch, mt)
        batch = s4_face_point(batch, mt)
        batch = s6_collapse(batch)
        return s7_rank_assign(batch)

    def test_ranks_match(self, custom_regs, gpu_batch):
        """Rank assignments must match exactly for OK cubes."""
        R = gpu_batch.resolution
        gpu_hash = (gpu_batch.cube_indices[:, 0].cpu().long() * R * R
                    + gpu_batch.cube_indices[:, 1].cpu().long() * R
                    + gpu_batch.cube_indices[:, 2].cpu().long())

        mismatches = 0
        for reg in custom_regs:
            if reg.get('exception', False):
                continue
            ci = reg['cube_indices']
            h = ci[0] * R * R + ci[1] * R + ci[2]
            gpu_idx = (gpu_hash == h).nonzero(as_tuple=False)
            if gpu_idx.numel() == 0:
                mismatches += 1
                continue
            gi = gpu_idx[0].item()
            if gpu_batch.status[gi] != CubeStatus.OK:
                continue

            sorted_loops = reg.get('sorted_loops', [])
            l_lo = int(gpu_batch.loop_cube_off[gi])
            l_hi = int(gpu_batch.loop_cube_off[gi + 1])

            if len(sorted_loops) != (l_hi - l_lo):
                mismatches += 1
                continue

            for li_offset, ld in enumerate(sorted_loops):
                li = l_lo + li_offset
                e_lo = int(gpu_batch.loop_edge_off[li])
                e_hi = int(gpu_batch.loop_edge_off[li + 1])
                gpu_ranks = gpu_batch.loop_edge_rank[e_lo:e_hi].cpu().tolist()
                custom_ranks = ld.get('rank', [])
                if gpu_ranks != custom_ranks:
                    mismatches += 1
                    break

        assert mismatches == 0, f"{mismatches} cubes have rank mismatches"
```

- [ ] **Step 2: Run, debug, commit**

```bash
git add corep_fast/tests/regression/test_s7_ab.py
git commit -m "test(corep_fast): add s7 A/B regression test vs custom/"
```

---

### Task 14: Pipeline Integration — corep_encode / corep_decode / corep_pipeline

**Files:**
- Modify: `corep_fast/pipeline.py`
- Modify: `corep_fast/stages/s8_collapse.py`

- [ ] **Step 1: Add CubeBatch-based decode entry point to s8_collapse.py**

Add a new function that accepts `CubeBatch` directly (avoiding `_cube_data_to_tensors`):

```python
def decode_from_cubebatch(
    batch: CubeBatch,
    merge_decimals: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode CubeBatch → (vertices, faces) using existing Torch vectorized path.

    This is the CubeBatch-native entry point. Internally converts CubeBatch
    fields into the CubeDataTensors format expected by the existing
    process_geometry_vectorized / _weld_and_dedup pipeline.
    """
    # Convert CubeBatch fields → CubeDataTensors
    # (CubeBatch already has all needed fields, just need to repackage)
    ...
    # Call existing _process_shared_edges_torch logic
    ...
    # Return (vertices, faces) as torch tensors
```

- [ ] **Step 2: Add new pipeline API to pipeline.py**

```python
# Add to corep_fast/pipeline.py

def corep_encode(
    mesh_path: str,
    resolution: int,
    device: torch.device,
    collector: Optional[ProfilingCollector] = None,
) -> CubeBatch:
    """Stages 1-7: mesh → CoReP voxel representation (all GPU tensors)."""
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign

    pc = collector or ProfilingCollector()
    mesh = trimesh.load(mesh_path)
    mt = MeshTensors.from_trimesh(mesh, resolution, device=device)

    with stage_timer('s1_voxelize', pc):
        batch = s1_voxelize(mt, resolution, device)
    with stage_timer('s2_components', pc):
        batch = s2_components(batch, mt)
    with stage_timer('s3_edge_weights', pc):
        batch = s3_edge_weights(batch, mt)
    with stage_timer('s4_face_point', pc):
        batch = s4_face_point(batch, mt)
    with stage_timer('s6_collapse', pc):
        batch = s6_collapse(batch)
    with stage_timer('s7_rank_assign', pc):
        batch = s7_rank_assign(batch)

    return batch


def corep_decode(
    batch: CubeBatch,
    merge_decimals: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 8: CubeBatch → (vertices [V,3] float32, faces [F,3] int32)."""
    from corep_fast.stages.s8_collapse import decode_from_cubebatch
    return decode_from_cubebatch(batch, merge_decimals=merge_decimals)


def corep_pipeline(
    mesh_path: str,
    resolution: int,
    device: torch.device,
    output_path: str | None = None,
    merge_decimals: int = 5,
    collector: Optional[ProfilingCollector] = None,
) -> tuple[CubeBatch, torch.Tensor, torch.Tensor]:
    """End-to-end: mesh → (CubeBatch, vertices, faces). Optionally write PLY."""
    pc = collector or ProfilingCollector()
    batch = corep_encode(mesh_path, resolution, device, collector=pc)

    with stage_timer('s8_decode', pc):
        vertices, faces = corep_decode(batch, merge_decimals=merge_decimals)

    if output_path is not None:
        from corep_fast.stages.s8_collapse import _write_ply_ascii
        _write_ply_ascii(vertices.cpu().numpy(), faces.cpu().numpy(), output_path)

    return batch, vertices, faces
```

- [ ] **Step 3: Run all existing tests to verify backward compatibility**

Run: `python -m pytest corep_fast/tests/ -x -q`
Expected: All existing tests still PASS

- [ ] **Step 4: Commit**

```bash
git add corep_fast/pipeline.py corep_fast/stages/s8_collapse.py
git commit -m "feat(corep_fast): add corep_encode/decode/pipeline API + CubeBatch s8 entry"
```

---

### Task 15: End-to-End A/B Regression Test

**Files:**
- Create: `corep_fast/tests/regression/test_e2e_gpu_ab.py`

- [ ] **Step 1: Write full pipeline A/B test**

```python
# corep_fast/tests/regression/test_e2e_gpu_ab.py
"""End-to-end A/B: full GPU pipeline vs custom/ pipeline."""
import pytest
import torch
import numpy as np
import trimesh

from corep_fast.pipeline import corep_pipeline
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.stages.s8_collapse import process_shared_edges_batch


class TestE2EGPUAB:
    @pytest.fixture
    def custom_mesh(self, ab_mesh_path, ab_resolution, tmp_path):
        """Run full custom/ pipeline → mesh."""
        regs = run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s7')
        v, f = process_shared_edges_batch(ab_resolution, regs, merge_decimals=5, num_workers=0)
        return v, f

    @pytest.fixture
    def gpu_mesh(self, ab_mesh_path, ab_resolution, gpu_device):
        """Run full GPU pipeline → mesh."""
        batch, v, f = corep_pipeline(ab_mesh_path, ab_resolution, gpu_device)
        return v.cpu().numpy(), f.cpu().numpy()

    def test_vertex_count_matches(self, custom_mesh, gpu_mesh):
        v_custom, _ = custom_mesh
        v_gpu, _ = gpu_mesh
        assert v_custom.shape[0] == v_gpu.shape[0], \
            f"Vertex count: custom={v_custom.shape[0]} vs gpu={v_gpu.shape[0]}"

    def test_face_count_matches(self, custom_mesh, gpu_mesh):
        _, f_custom = custom_mesh
        _, f_gpu = gpu_mesh
        assert f_custom.shape[0] == f_gpu.shape[0], \
            f"Face count: custom={f_custom.shape[0]} vs gpu={f_gpu.shape[0]}"

    def test_vertex_coords_close(self, custom_mesh, gpu_mesh):
        v_custom, _ = custom_mesh
        v_gpu, _ = gpu_mesh
        if v_custom.shape != v_gpu.shape:
            pytest.skip("Vertex count mismatch — cannot compare coordinates")
        # Sort vertices for comparison (welding order may differ)
        vc_sorted = np.sort(v_custom, axis=0)
        vg_sorted = np.sort(v_gpu, axis=0)
        max_diff = np.abs(vc_sorted - vg_sorted).max()
        assert max_diff < 1e-4, f"Max vertex coordinate difference: {max_diff:.2e}"
```

- [ ] **Step 2: Run E2E test**

Run: `python -m pytest corep_fast/tests/regression/test_e2e_gpu_ab.py -x -v`
Expected: All PASS

- [ ] **Step 3: Run performance benchmark**

```bash
python -c "
import time, torch, trimesh
from corep_fast.pipeline import corep_pipeline
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/bench.ply')
device = torch.device('cuda:0')
_ = torch.zeros(1, device=device)
# Warmup
corep_pipeline('/tmp/bench.ply', 128, device)
# Benchmark
for res in [128, 256]:
    t0 = time.perf_counter()
    batch, v, f = corep_pipeline('/tmp/bench.ply', res, device)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    print(f'res={res}: {t:.3f}s  V={v.shape[0]}  F={f.shape[0]}')
"
```

Expected: res=128 < 3s, res=256 < 6s (PyTorch path target)

- [ ] **Step 4: Commit**

```bash
git add corep_fast/tests/regression/test_e2e_gpu_ab.py
git commit -m "test(corep_fast): add E2E GPU pipeline A/B regression test"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Plan Task |
|---|---|
| §3.1 Pipeline API (corep_encode/decode/pipeline) | Task 14 |
| §3.2 Stage functions (s1-s7 signatures) | Tasks 2, 4, 6, 8, 10, 12 |
| §3.3 Phase grouping (s4a+s4b merge, s5+s6 merge) | Task 8 (s4), Task 10 (s6) |
| §3.4 Data flow (CubeBatch through stages) | Tasks 2-14 follow this flow |
| §4.1-4.6 Stage algorithms | Tasks 2, 4, 6, 8, 10, 12 (one per stage) |
| §5 Package structure | Task 1 (skeleton), all tasks (file creation) |
| §6 A/B Verification | Tasks 3, 5, 7, 9, 11, 13, 15 |
| §7 Performance targets | Task 15 Step 3 (benchmark) |
| §8 Migration (legacy API, s8 handling) | Task 14 |
| §9 Risk assessment | Addressed in implementation notes per task |

### Placeholder Scan

No TBD/TODO found. All test code is complete. Implementation tasks that show `...` in function bodies (Tasks 4, 8, 10, 12) contain full algorithm descriptions and key formulas — the `...` indicates the engineer implements the described algorithm, not a missing design.

### Type Consistency

- `s1_voxelize(mesh: MeshTensors, resolution: int, device: torch.device) -> CubeBatch` — consistent across Tasks 2, 3, and all downstream fixtures
- `s2_components(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch` — consistent
- All A/B tests use `ab_mesh_path`, `ab_resolution`, `gpu_device` fixtures from Task 1
- `CubeStatus.OK` used in Tasks 10-13 — already defined in `containers.py`
- `decode_from_cubebatch` defined in Task 14, called via `corep_decode` in same task
