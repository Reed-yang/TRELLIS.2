# CoReP-Fast Phase 0: Infrastructure Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the infrastructure layer of `corep_fast/` — data containers, configuration, profiling harness, topology equivalence checker, interop bridges, and baseline runner — with **no Torch stage rewrites yet**, culminating in a complete baseline profile run of `custom/` on the baseline evaluation set (`baseline_custom_v0.json`).

**Architecture:** Pure Python + dataclasses + lightweight Torch tensor containers. No CUDA kernel work in Phase 0. The final artifact is a runnable profiling harness that (a) instruments every stage of the current `custom/` CoReP pipeline and (b) produces the ground-truth performance numbers that will drive Phase 1 stage rewriting priority.

**Tech Stack:** Python 3.10+, Torch 2.x, `torch_scatter` (new dependency), NumPy, scipy, trimesh, pytest, orjson (already in use). No Triton, no pybind11, no CUDA C++.

**Spec:** `docs/superpowers/specs/2026-04-15-corep-fast-stage1-design.md` §4 (Directory Structure), §5 (Data Containers), §7 (Profiling Harness), §14 (Appendix constants + interop contract)

**Scope note:** Phase 1 (8 stage Torch rewrites) and Phase 2 (multi-GPU runner + validation) are **separate plans**, to be written after Phase 0 completes — because spec §7.7 requires that Phase 1 stage priority be determined by the actual profile data from Task 16 of this plan, not by pre-guessed ordering.

---

## File Structure

```
corep_fast/                              # project root, parallel with custom/
├── __init__.py                          # Task 1  — empty export surface for Phase 0
├── constants.py                         # Task 2  — cube topology tables
├── config.py                            # Task 3  — Mode enum, StageConfig, BackendConfig
├── containers.py                        # Tasks 4, 5 — MeshTensors + CubeBatch dataclasses
│
├── geometry/                            # Task 1  — package skeleton only (kernels in Phase 1)
│   └── __init__.py
│
├── stages/                              # Task 1  — package skeleton only (implementations in Phase 1)
│   └── __init__.py
│
├── interop/
│   ├── __init__.py                      # Task 1
│   ├── from_custom.py                   # Task 10 — list-of-dict → CubeBatch
│   └── to_custom.py                     # Task 11 — CubeBatch → list-of-dict
│
├── profiling/
│   ├── __init__.py                      # Task 1
│   ├── harness.py                       # Task 6  — stage_timer + ProfilingCollector
│   ├── topology_equivalence.py          # Tasks 7, 8, 9 — 5-layer equivalence checker
│   ├── ab_rig.py                        # Task 12 — single-mesh A/B runner
│   ├── baseline_runner.py               # Task 13 — driver for baseline eval set
│   └── report_builder.py                # Task 14 — JSON → Markdown summary
│
├── distributed/                         # Phase 2 placeholder
│   └── __init__.py                      # Task 1
│
├── scripts/                             # Phase 2 placeholder
│   └── __init__.py                      # Task 1
│
└── tests/
    ├── __init__.py                      # Task 1
    ├── conftest.py                      # Task 15 — shared pytest fixtures
    ├── unit/
    │   ├── __init__.py                  # Task 1
    │   ├── test_constants.py            # Task 2  — topology table sanity checks
    │   ├── test_config.py               # Task 3
    │   ├── test_containers.py           # Tasks 4, 5
    │   ├── test_harness.py              # Task 6
    │   ├── test_topology_equivalence.py # Tasks 7, 8, 9
    │   ├── test_interop.py              # Tasks 10, 11
    │   ├── test_ab_rig.py               # Task 12
    │   └── test_report_builder.py       # Task 14
    └── regression/
        └── __init__.py                  # Task 1

profiling/runs/                          # Task 16 — output directory for baseline profile runs
└── baseline_custom_v0.json              # Task 16 — Step 0 baseline profile
```

**Key decisions locked by this structure:**

- `corep_fast/` lives at project root, parallel to `custom/` — per spec §0.1 "user direct instruction"
- Phase 0 creates **all package skeletons** (including `geometry/`, `stages/`, `distributed/`, `scripts/`) so Phase 1 tasks can drop files in without restructuring
- `tests/unit/` mirrors source layout 1:1 — one `test_X.py` per module
- `profiling/runs/` is **gitignored by default** (contains large JSON outputs); baseline profile results are session-local per user but the `baseline_custom_v0.json` file path is referenced by name throughout Phase 1

---

## Task 1: Directory Skeleton + Dependency Pin

**Files:**
- Create: `corep_fast/__init__.py`
- Create: `corep_fast/geometry/__init__.py`
- Create: `corep_fast/stages/__init__.py`
- Create: `corep_fast/interop/__init__.py`
- Create: `corep_fast/profiling/__init__.py`
- Create: `corep_fast/distributed/__init__.py`
- Create: `corep_fast/scripts/__init__.py`
- Create: `corep_fast/tests/__init__.py`
- Create: `corep_fast/tests/unit/__init__.py`
- Create: `corep_fast/tests/regression/__init__.py`
- Create: `corep_fast/requirements-phase0.txt`
- Test: `corep_fast/tests/unit/test_package_import.py`

- [ ] **Step 1: Create the directory skeleton**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p corep_fast/{geometry,stages,interop,profiling,distributed,scripts,tests/unit,tests/regression}
mkdir -p profiling/runs
```

- [ ] **Step 2: Create all empty `__init__.py` files**

For each of the following paths, create an empty file (zero bytes):

```
corep_fast/__init__.py
corep_fast/geometry/__init__.py
corep_fast/stages/__init__.py
corep_fast/interop/__init__.py
corep_fast/profiling/__init__.py
corep_fast/distributed/__init__.py
corep_fast/scripts/__init__.py
corep_fast/tests/__init__.py
corep_fast/tests/unit/__init__.py
corep_fast/tests/regression/__init__.py
```

Use the Write tool to create each as empty string content `""`.

- [ ] **Step 3: Create `corep_fast/requirements-phase0.txt`**

```txt
# corep_fast Phase 0 additional dependencies
# All other deps (torch, numpy, scipy, trimesh, orjson, pytest) are already in the project .venv.
torch_scatter>=2.1.2
```

- [ ] **Step 4: Install torch_scatter into the project venv**

Run:

```bash
.venv/bin/pip install -r corep_fast/requirements-phase0.txt
```

Expected output: `Successfully installed torch_scatter-2.x.x` (or "Requirement already satisfied"). If the installation fails with a build error, check that the installed `torch` version matches a `torch_scatter` wheel by running `.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"` and installing the matching wheel from `https://data.pyg.org/whl/torch-<ver>+cu<ver>.html`.

- [ ] **Step 5: Verify torch_scatter imports correctly**

Run:

```bash
.venv/bin/python -c "import torch_scatter; print(torch_scatter.__version__)"
```

Expected: A version number like `2.1.2` or similar. If it prints an ImportError, stop and ask the user before proceeding.

- [ ] **Step 6: Write the package-import smoke test**

Create `corep_fast/tests/unit/test_package_import.py`:

```python
"""Smoke test: all corep_fast subpackages import without error."""


def test_corep_fast_root_imports():
    import corep_fast  # noqa: F401


def test_corep_fast_geometry_imports():
    import corep_fast.geometry  # noqa: F401


def test_corep_fast_stages_imports():
    import corep_fast.stages  # noqa: F401


def test_corep_fast_interop_imports():
    import corep_fast.interop  # noqa: F401


def test_corep_fast_profiling_imports():
    import corep_fast.profiling  # noqa: F401


def test_corep_fast_distributed_imports():
    import corep_fast.distributed  # noqa: F401


def test_torch_scatter_available():
    import torch_scatter
    assert hasattr(torch_scatter, 'segment_csr')
```

- [ ] **Step 7: Run the smoke test**

Run:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python -m pytest corep_fast/tests/unit/test_package_import.py -v
```

Expected: 7 tests pass.

- [ ] **Step 8: Update `.gitignore` for `profiling/runs/`**

Append to the existing `.gitignore`:

```gitignore

# CoReP-Fast profiling outputs (large JSON, session-local)
profiling/runs/
```

Verify existing `docs` ignore is still present (the spec was force-added with `-f`).

- [ ] **Step 9: Commit**

```bash
git add -f corep_fast/__init__.py corep_fast/geometry/__init__.py corep_fast/stages/__init__.py \
           corep_fast/interop/__init__.py corep_fast/profiling/__init__.py \
           corep_fast/distributed/__init__.py corep_fast/scripts/__init__.py \
           corep_fast/tests/__init__.py corep_fast/tests/unit/__init__.py \
           corep_fast/tests/regression/__init__.py \
           corep_fast/requirements-phase0.txt \
           corep_fast/tests/unit/test_package_import.py \
           .gitignore
git commit -m "$(cat <<'EOF'
feat(corep_fast): create phase-0 package skeleton and pin torch_scatter

- Create corep_fast/ package tree with all subpackage placeholders
- Pin torch_scatter >= 2.1.2 as the only new Phase 0 dependency
- Add smoke test verifying all subpackages import
- Ignore profiling/runs/ for large session-local profile outputs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `corep_fast/constants.py` — Cube Topology Tables

**Files:**
- Create: `corep_fast/constants.py`
- Test: `corep_fast/tests/unit/test_constants.py`

**Reference:** spec §14.A; `custom/collapse_edge.py` (TRIANGLES, EDGE_VERTS), `custom/ARCHITECTURE.md` §13 (Cube Geometry Reference)

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_constants.py`:

```python
"""Unit tests for corep_fast/constants.py cube topology tables."""
import torch

from corep_fast import constants as C


def test_num_edges_is_18():
    assert C.NUM_EDGES == 18


def test_num_facets_is_12():
    assert C.NUM_FACETS == 12


def test_num_vertices_is_8():
    assert C.NUM_VERTICES == 8


def test_cube_vertices_shape_and_dtype():
    assert isinstance(C.CUBE_VERTICES, torch.Tensor)
    assert C.CUBE_VERTICES.shape == (8, 3)
    assert C.CUBE_VERTICES.dtype == torch.float32
    # Corner (0,0,0) exists
    assert torch.equal(C.CUBE_VERTICES[0], torch.tensor([0., 0., 0.]))
    # Corner (1,1,1) exists
    assert torch.equal(C.CUBE_VERTICES[6], torch.tensor([1., 1., 1.]))


def test_cube_edges_shape_and_values():
    assert isinstance(C.CUBE_EDGES, torch.Tensor)
    assert C.CUBE_EDGES.shape == (18, 2)
    assert C.CUBE_EDGES.dtype == torch.int32
    # Edge 0 connects vertex 0 and 1 (spec §14.A)
    assert C.CUBE_EDGES[0].tolist() == [0, 1]
    # Edge 12 is the bottom diagonal (0,2)
    assert C.CUBE_EDGES[12].tolist() == [0, 2]
    # Edge 17 is the left face diagonal (0,7)
    assert C.CUBE_EDGES[17].tolist() == [0, 7]


def test_cube_facets_shape_and_values():
    """12 triangular facets, each defined by 3 edge indices."""
    assert isinstance(C.CUBE_FACETS, torch.Tensor)
    assert C.CUBE_FACETS.shape == (12, 3)
    assert C.CUBE_FACETS.dtype == torch.int32
    # T0: bottom half 1 — edges (0, 1, 12)
    assert C.CUBE_FACETS[0].tolist() == [0, 1, 12]
    # T11: left half 2 — edges (7, 8, 17)
    assert C.CUBE_FACETS[11].tolist() == [7, 8, 17]


def test_cube_edges_all_reference_valid_vertices():
    assert C.CUBE_EDGES.min() >= 0
    assert C.CUBE_EDGES.max() < 8


def test_cube_facets_all_reference_valid_edges():
    assert C.CUBE_FACETS.min() >= 0
    assert C.CUBE_FACETS.max() < 18


def test_edge_share_factors_shape():
    """Each edge has a cross-cube sharing factor (4 for axis edges, 2 for diagonals)."""
    assert C.EDGE_SHARE_FACTORS.shape == (18,)
    assert C.EDGE_SHARE_FACTORS.dtype == torch.int32
    # Axis edges 0-11: shared by 4 cubes
    assert torch.all(C.EDGE_SHARE_FACTORS[:12] == 4)
    # Diagonal edges 12-17: shared by 2 cubes
    assert torch.all(C.EDGE_SHARE_FACTORS[12:] == 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_constants.py -v
```

Expected: ModuleNotFoundError or AttributeError on first test (`C.NUM_EDGES` not defined).

- [ ] **Step 3: Write the constants module**

Create `corep_fast/constants.py`:

```python
"""
Cube topology constants for CoReP representation.

All tables are Torch tensors pinned to CPU by default.  Per-device copies are
materialized lazily by the geometry / stages modules via `.to(device)`.

Reference:
    - custom/ARCHITECTURE.md §13 (Cube Geometry Reference)
    - spec §14.A (Appendix CoReP Constants Reference)
"""
import torch


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

NUM_VERTICES: int = 8
NUM_EDGES: int = 18           # 12 axis-aligned + 6 face diagonals
NUM_FACETS: int = 12          # 2 triangles per cube face × 6 faces


# ---------------------------------------------------------------------------
# Cube vertices  (unit cube [0,1]³, indexed as in custom/ARCHITECTURE.md §13.1)
# ---------------------------------------------------------------------------
#
#         7 ─────────── 6          Y (up)
#        /|            /|          │
#       / |           / |          │
#      4 ─────────── 5  |          └──── X (right)
#      |  |          |  |         /
#      |  3 ─────────|── 2       Z (front)
#      | /           | /
#      |/            |/
#      0 ─────────── 1
#
CUBE_VERTICES: torch.Tensor = torch.tensor([
    [0., 0., 0.],   # 0
    [1., 0., 0.],   # 1
    [1., 1., 0.],   # 2
    [0., 1., 0.],   # 3
    [0., 0., 1.],   # 4
    [1., 0., 1.],   # 5
    [1., 1., 1.],   # 6
    [0., 1., 1.],   # 7
], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Cube edges  (18 total = 12 axis-aligned + 6 face diagonals)
# Each row is (start_vertex, end_vertex).
# Matches the ordering in custom/ARCHITECTURE.md §13.2.
# ---------------------------------------------------------------------------
CUBE_EDGES: torch.Tensor = torch.tensor([
    # 12 axis-aligned edges, shared by 4 cubes each
    [0, 1],   #  0 — bottom front
    [1, 2],   #  1 — bottom right
    [2, 3],   #  2 — bottom back
    [3, 0],   #  3 — bottom left
    [4, 5],   #  4 — top front
    [5, 6],   #  5 — top right
    [6, 7],   #  6 — top back
    [7, 4],   #  7 — top left
    [0, 4],   #  8 — front-left vertical
    [1, 5],   #  9 — front-right vertical
    [2, 6],   # 10 — back-right vertical
    [3, 7],   # 11 — back-left vertical
    # 6 face diagonals, shared by 2 cubes each
    [0, 2],   # 12 — bottom diagonal
    [4, 6],   # 13 — top diagonal
    [1, 4],   # 14 — front face diagonal
    [1, 6],   # 15 — right face diagonal
    [2, 7],   # 16 — back face diagonal
    [0, 7],   # 17 — left face diagonal
], dtype=torch.int32)


# ---------------------------------------------------------------------------
# Cube triangulated facets  (12 triangles = 2 per cube face × 6 faces)
# Each row is a tuple of 3 edge indices defining the triangle.
# Matches custom/ARCHITECTURE.md §13.3.
# ---------------------------------------------------------------------------
CUBE_FACETS: torch.Tensor = torch.tensor([
    [ 0,  1, 12],   # T0  — bottom half 1
    [ 2,  3, 12],   # T1  — bottom half 2
    [ 4,  5, 13],   # T2  — top half 1
    [ 6,  7, 13],   # T3  — top half 2
    [ 0,  8, 14],   # T4  — front half 1
    [ 4,  9, 14],   # T5  — front half 2
    [ 1, 10, 15],   # T6  — right half 1
    [ 5,  9, 15],   # T7  — right half 2
    [ 2, 11, 16],   # T8  — back half 1
    [ 6, 10, 16],   # T9  — back half 2
    [ 3, 11, 17],   # T10 — left half 1
    [ 7,  8, 17],   # T11 — left half 2
], dtype=torch.int32)


# ---------------------------------------------------------------------------
# Cross-cube edge sharing factors
# Axis edges are shared by 4 cubes (four-way rotational symmetry around the
# edge); face diagonals are shared by 2 cubes (the two cubes adjacent to that
# face).  Used by Stage 8 deduplication accounting.
# ---------------------------------------------------------------------------
EDGE_SHARE_FACTORS: torch.Tensor = torch.cat([
    torch.full((12,), 4, dtype=torch.int32),
    torch.full((6,),  2, dtype=torch.int32),
])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_constants.py -v
```

Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/constants.py corep_fast/tests/unit/test_constants.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add cube topology constants module

Centralize the CoReP cube geometry tables (vertices, 18 edges, 12
triangulated facets, cross-cube share factors) as Torch tensors, matching
custom/ARCHITECTURE.md §13 and spec §14.A.  Replaces the ad-hoc tables
duplicated across custom/collapse_edge.py, custom/collapse_point.py, and
custom/collapse_volume.py.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `corep_fast/config.py` — Mode Enum, StageConfig, BackendConfig

**Files:**
- Create: `corep_fast/config.py`
- Test: `corep_fast/tests/unit/test_config.py`

**Reference:** spec §9.2 (Debug Modes), §10.2 (Backend Dispatch)

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_config.py`:

```python
"""Unit tests for corep_fast/config.py."""
import pytest

from corep_fast import config as cf


def test_mode_enum_values():
    assert cf.Mode.PRODUCTION.value == 'production'
    assert cf.Mode.DEBUG.value == 'debug'
    assert cf.Mode.STRICT.value == 'strict'


def test_set_and_get_mode_default_is_production():
    cf.set_mode(cf.Mode.PRODUCTION)
    assert cf.get_mode() == cf.Mode.PRODUCTION


def test_set_and_get_mode_roundtrip():
    cf.set_mode(cf.Mode.DEBUG)
    assert cf.get_mode() == cf.Mode.DEBUG
    cf.set_mode(cf.Mode.PRODUCTION)
    assert cf.get_mode() == cf.Mode.PRODUCTION


def test_stage_config_defaults():
    sc = cf.StageConfig.default()
    assert sc.chunk_size_bytes == 512 * 1024 * 1024  # 512 MiB
    assert sc.prod_k_threshold == 100_000
    assert sc.max_poly_verts == 12


def test_stage_config_override():
    sc = cf.StageConfig(chunk_size_bytes=1 << 30, prod_k_threshold=50_000, max_poly_verts=16)
    assert sc.chunk_size_bytes == 1 << 30
    assert sc.prod_k_threshold == 50_000
    assert sc.max_poly_verts == 16


def test_backend_config_all_torch_default():
    bc = cf.BackendConfig.all_torch()
    assert bc.s1 == 'torch'
    assert bc.s2 == 'torch'
    assert bc.s3 == 'torch'
    assert bc.s4_face == 'torch'
    assert bc.s4_point == 'torch'
    assert bc.s5 == 'torch'
    assert bc.s6 == 'torch'
    assert bc.s7 == 'torch'
    assert bc.s8 == 'torch'


def test_backend_config_rejects_invalid_value():
    with pytest.raises((ValueError, AssertionError)):
        cf.BackendConfig(
            s1='fortran', s2='torch', s3='torch', s4_face='torch',
            s4_point='torch', s5='torch', s6='torch', s7='torch', s8='torch',
        )


def test_backend_config_partial_override():
    bc = cf.BackendConfig.all_torch()
    bc2 = bc.with_override(s6='triton')
    assert bc2.s6 == 'triton'
    assert bc2.s1 == 'torch'
    # Original is not mutated
    assert bc.s6 == 'torch'
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_config.py -v
```

Expected: ModuleNotFoundError or AttributeError (config module not yet present).

- [ ] **Step 3: Write the config module**

Create `corep_fast/config.py`:

```python
"""
Runtime configuration for corep_fast.

Three logical config pieces:

1. Mode (PRODUCTION / DEBUG / STRICT) — global process-wide debug level.
2. StageConfig — per-stage numerical tunables (chunk sizes, thresholds, budgets).
3. BackendConfig — which backend (torch | triton) each stage uses.  In Phase 0
   and all of Stage 1, every stage uses 'torch'.  Stage 2 will flip individual
   fields to 'triton'.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import Literal


# ---------------------------------------------------------------------------
# Mode (debug level)
# ---------------------------------------------------------------------------

class Mode(enum.Enum):
    PRODUCTION = 'production'   # default: skip invariant checks, fastest
    DEBUG      = 'debug'        # enable invariant checks, dump per-stage pickles
    STRICT     = 'strict'       # DEBUG + per-stage A/B vs custom/ (very slow)


_GLOBAL_MODE: Mode = Mode.PRODUCTION


def set_mode(mode: Mode) -> None:
    """Set the global debug mode for this process."""
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode


def get_mode() -> Mode:
    """Return the current global debug mode."""
    return _GLOBAL_MODE


# ---------------------------------------------------------------------------
# StageConfig (numerical tunables)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageConfig:
    """
    Numerical tunables shared across stages.

    Fields:
        chunk_size_bytes: peak working memory budget for Stage 1 voxelize
            candidate-pair expansion.  512 MiB is the default; increase on
            high-VRAM GPUs.
        prod_k_threshold: maximum Cartesian product size for Stage 6
            combinatorial enumeration.  Cubes exceeding this are marked
            BUDGET_EXCEEDED and delegated to the Stage 8 exception path.
            Matches custom/collapse_face.py:260 (100000).
        max_poly_verts: maximum padded polygon vertex count after
            Sutherland-Hodgman clipping in Stage 4.  12 is sufficient because
            a triangle clipped against 6 AABB planes has at most 9 vertices.
    """
    chunk_size_bytes: int = 512 * 1024 * 1024
    prod_k_threshold: int = 100_000
    max_poly_verts: int = 12

    @classmethod
    def default(cls) -> 'StageConfig':
        return cls()


# ---------------------------------------------------------------------------
# BackendConfig (per-stage torch | triton)
# ---------------------------------------------------------------------------

Backend = Literal['torch', 'triton']
_VALID_BACKENDS: frozenset[str] = frozenset({'torch', 'triton'})


@dataclass(frozen=True)
class BackendConfig:
    """Per-stage backend selection.  All stages are 'torch' in Stage 1."""
    s1: Backend
    s2: Backend
    s3: Backend
    s4_face: Backend
    s4_point: Backend
    s5: Backend
    s6: Backend
    s7: Backend
    s8: Backend

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if value not in _VALID_BACKENDS:
                raise ValueError(
                    f"BackendConfig field {field_name!r} has invalid value {value!r}; "
                    f"expected one of {sorted(_VALID_BACKENDS)}"
                )

    @classmethod
    def all_torch(cls) -> 'BackendConfig':
        return cls(
            s1='torch', s2='torch', s3='torch', s4_face='torch',
            s4_point='torch', s5='torch', s6='torch', s7='torch', s8='torch',
        )

    def with_override(self, **kwargs: Backend) -> 'BackendConfig':
        """Return a new BackendConfig with the given fields overridden."""
        return replace(self, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_config.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/config.py corep_fast/tests/unit/test_config.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add Mode, StageConfig, BackendConfig

Global debug mode (PRODUCTION/DEBUG/STRICT), frozen dataclass StageConfig
for numerical tunables (chunk sizes, combinatorial budgets, polygon vertex
cap), and frozen BackendConfig for per-stage backend dispatch (torch-only
in Stage 1, triton plug-in points for Stage 2).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `corep_fast/containers.py` — `MeshTensors` Dataclass

**Files:**
- Create: `corep_fast/containers.py` (MeshTensors portion; CubeBatch added in Task 5)
- Test: `corep_fast/tests/unit/test_containers.py` (MeshTensors portion)

**Reference:** spec §5.2

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_containers.py`:

```python
"""Unit tests for corep_fast/containers.py MeshTensors (Task 4) and CubeBatch (Task 5)."""
import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors


# ---------------------------------------------------------------------------
# MeshTensors
# ---------------------------------------------------------------------------

def _make_cube_trimesh() -> trimesh.Trimesh:
    """A unit cube centered at origin — 12 triangles, 8 vertices, watertight."""
    return trimesh.creation.box(extents=[1., 1., 1.])


def _make_open_plane_trimesh() -> trimesh.Trimesh:
    """Flat square made of 2 triangles — 4 boundary edges, no non-manifold."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_mesh_tensors_from_trimesh_cube():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert isinstance(mt.vertices, torch.Tensor)
    assert mt.vertices.dtype == torch.float32
    assert mt.vertices.shape == (8, 3)
    assert mt.faces.dtype == torch.int32
    assert mt.faces.shape == (12, 3)
    assert mt.triangles.shape == (12, 3, 3)
    assert mt.face_normals.shape == (12, 3)
    assert mt.face_adj.shape == (12, 3)
    assert mt.resolution == 64
    assert str(mt.device) == 'cpu'


def test_mesh_tensors_cube_has_no_boundaries():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.boundaries.shape[0] == 0


def test_mesh_tensors_open_plane_has_4_boundaries():
    mesh = _make_open_plane_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.boundaries.shape == (4, 2, 3)
    assert mt.boundary_face_ids.shape == (4,)
    assert mt.boundary_face_ids.dtype == torch.int32


def test_mesh_tensors_normalization_puts_mesh_in_unit_cube():
    mesh = _make_cube_trimesh()
    # before normalization the box is centered at origin with extent 1
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert mt.vertices.min() >= 0.0 - 1e-6
    assert mt.vertices.max() <= 1.0 + 1e-6


def test_mesh_tensors_triangles_match_faces_gather():
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    gathered = mt.vertices[mt.faces.long()]
    assert torch.allclose(gathered, mt.triangles)


def test_mesh_tensors_face_adj_cube_is_dense():
    """Every face of a cube shares an edge with exactly 3 neighbors — no -1 entries."""
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    assert (mt.face_adj >= 0).all()


def test_mesh_tensors_face_adj_open_plane_has_boundary_gaps():
    """Open plane has 2 faces sharing 1 edge — 5 of 6 adjacency slots are -1."""
    mesh = _make_open_plane_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')
    neg_count = (mt.face_adj == -1).sum().item()
    assert neg_count == 4  # 2 faces × 3 edges - 2 shared = 4 unshared


def test_mesh_tensors_validate_rejects_empty():
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))
    with pytest.raises(ValueError):
        MeshTensors.from_trimesh(empty, resolution=64, device='cpu')


def test_mesh_tensors_to_cuda_device_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    mesh = _make_cube_trimesh()
    mt = MeshTensors.from_trimesh(mesh, resolution=64, device='cuda')
    assert mt.vertices.is_cuda
    assert mt.faces.is_cuda
    assert mt.triangles.is_cuda
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_containers.py -v
```

Expected: ImportError (MeshTensors not defined in containers.py yet).

- [ ] **Step 3: Write `containers.py` — MeshTensors portion**

Create `corep_fast/containers.py`:

```python
"""
Data containers for corep_fast.

MeshTensors: normalized input mesh held as device tensors.
CubeBatch:   all cubes of a single mesh, dense-where-possible + CSR-where-ragged.

Reference: spec §5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import trimesh


# ---------------------------------------------------------------------------
# MeshTensors
# ---------------------------------------------------------------------------

@dataclass
class MeshTensors:
    """Input mesh after normalization to [0,1]³, held as device tensors."""

    # Geometry
    vertices: torch.Tensor              # (V, 3)      float32
    faces: torch.Tensor                 # (F, 3)      int32    triangle → vertex indices
    triangles: torch.Tensor             # (F, 3, 3)   float32  pre-gathered vertices[faces]
    face_normals: torch.Tensor          # (F, 3)      float32  unit normals

    # Topology derived from faces
    face_adj: torch.Tensor              # (F, 3)      int32    neighbor face per edge, -1 if boundary

    # Open boundaries (edges belonging to exactly one face)
    boundaries: torch.Tensor            # (B, 2, 3)   float32  segment endpoints
    boundary_face_ids: torch.Tensor     # (B,)        int32

    # Non-manifold features
    nm_edges: torch.Tensor              # (M, 2, 3)   float32  edges shared by >2 faces
    nm_vertices: torch.Tensor           # (P, 3)      float32  bowtie vertices

    # Normalization metadata
    center: torch.Tensor                # (3,)        float32  centroid before normalization
    scale: float                        # uniform scale applied
    resolution: int                     # voxel grid resolution

    # Device handle
    device: torch.device

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_trimesh(
        cls,
        mesh: trimesh.Trimesh,
        resolution: int,
        device: str | torch.device = 'cpu',
    ) -> 'MeshTensors':
        """
        Build a MeshTensors from a trimesh.Trimesh.  Applies normalization
        (center + uniform scale to [0,1]³ with a tiny safety margin) matching
        custom/voxelize.py's `normalize_mesh` semantics.
        """
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise ValueError(
                f"MeshTensors.from_trimesh: input mesh is empty "
                f"(V={mesh.vertices.shape[0]}, F={mesh.faces.shape[0]})"
            )

        device = torch.device(device)
        verts_np = np.asarray(mesh.vertices, dtype=np.float32)
        faces_np = np.asarray(mesh.faces, dtype=np.int32)

        # --- Normalization: center, scale to fit in [0,1]³ with a margin ---
        bbox_min = verts_np.min(axis=0)
        bbox_max = verts_np.max(axis=0)
        center_np = 0.5 * (bbox_min + bbox_max)
        extent = float((bbox_max - bbox_min).max())
        if extent == 0.0:
            raise ValueError("MeshTensors.from_trimesh: degenerate mesh with zero extent")
        # Apply 0.947 safety margin matching custom/voxelize.py
        scale = 0.947 / extent
        verts_np = (verts_np - center_np) * scale + 0.5  # map into [0.0265, 0.9735]

        verts = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)
        faces = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32)
        triangles = verts[faces.long()]                                       # (F, 3, 3)
        e1 = triangles[:, 1] - triangles[:, 0]                                # (F, 3)
        e2 = triangles[:, 2] - triangles[:, 0]
        normals = torch.cross(e1, e2, dim=-1)                                 # (F, 3)
        norms = torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-20)
        face_normals = normals / norms

        # --- Face adjacency (via trimesh) ---
        face_adj = _build_face_adj(mesh, faces_np.shape[0], device=device)

        # --- Boundaries (edges in exactly one face) ---
        boundaries, boundary_face_ids = _extract_boundaries(mesh, verts_np, device=device)

        # --- Non-manifold edges and vertices ---
        nm_edges, nm_vertices = _extract_non_manifolds(mesh, verts_np, device=device)

        return cls(
            vertices=verts,
            faces=faces,
            triangles=triangles,
            face_normals=face_normals,
            face_adj=face_adj,
            boundaries=boundaries,
            boundary_face_ids=boundary_face_ids,
            nm_edges=nm_edges,
            nm_vertices=nm_vertices,
            center=torch.from_numpy(center_np).to(device=device, dtype=torch.float32),
            scale=scale,
            resolution=resolution,
            device=device,
        )


# ---------------------------------------------------------------------------
# Helpers used by MeshTensors.from_trimesh
# ---------------------------------------------------------------------------

def _build_face_adj(
    mesh: trimesh.Trimesh,
    num_faces: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build (F, 3) int32 face adjacency table: entry [i, j] is the index of the
    face sharing edge j of face i, or -1 if that edge is a boundary.

    Uses trimesh.face_adjacency (pairs of adjacent face ids) plus
    trimesh.face_adjacency_edges (edges, each as (v_a, v_b)).
    """
    adj = np.full((num_faces, 3), -1, dtype=np.int32)
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int32)   # (E, 2)
    edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int32)  # (E, 2)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    for (fa, fb), (va, vb) in zip(pairs, edges):
        # Find which slot in face fa contains edge (va, vb)
        for slot in range(3):
            u, v = faces[fa, slot], faces[fa, (slot + 1) % 3]
            if {int(u), int(v)} == {int(va), int(vb)}:
                adj[fa, slot] = fb
                break
        for slot in range(3):
            u, v = faces[fb, slot], faces[fb, (slot + 1) % 3]
            if {int(u), int(v)} == {int(va), int(vb)}:
                adj[fb, slot] = fa
                break

    return torch.from_numpy(adj).to(device=device, dtype=torch.int32)


def _extract_boundaries(
    mesh: trimesh.Trimesh,
    verts_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (boundaries, boundary_face_ids):
        boundaries:         (B, 2, 3) float32  endpoints of open-boundary edges
        boundary_face_ids:  (B,)      int32    the single face each boundary belongs to
    """
    # trimesh.edges_unique_length and edges_face give us the info we need.
    # A boundary edge belongs to exactly one face.
    faces = np.asarray(mesh.faces, dtype=np.int32)
    # Build edge → face count
    edge_face_map: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for (u, v) in [(a, b), (b, c), (c, a)]:
            key = (int(min(u, v)), int(max(u, v)))
            edge_face_map.setdefault(key, []).append(fi)

    bnd_segs: list[np.ndarray] = []
    bnd_face_ids: list[int] = []
    for (u, v), fids in edge_face_map.items():
        if len(fids) == 1:
            seg = np.stack([verts_np[u], verts_np[v]], axis=0)
            bnd_segs.append(seg)
            bnd_face_ids.append(fids[0])

    if bnd_segs:
        bnd_arr = np.stack(bnd_segs, axis=0).astype(np.float32)
        bnd_ids = np.asarray(bnd_face_ids, dtype=np.int32)
    else:
        bnd_arr = np.zeros((0, 2, 3), dtype=np.float32)
        bnd_ids = np.zeros((0,), dtype=np.int32)

    return (
        torch.from_numpy(bnd_arr).to(device=device, dtype=torch.float32),
        torch.from_numpy(bnd_ids).to(device=device, dtype=torch.int32),
    )


def _extract_non_manifolds(
    mesh: trimesh.Trimesh,
    verts_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (nm_edges, nm_vertices):
        nm_edges:    (M, 2, 3) float32  edges shared by >2 faces
        nm_vertices: (P, 3)    float32  bowtie vertices (pinch points)
    """
    faces = np.asarray(mesh.faces, dtype=np.int32)
    edge_face_count: dict[tuple[int, int], int] = {}
    for (a, b, c) in faces:
        for (u, v) in [(a, b), (b, c), (c, a)]:
            key = (int(min(u, v)), int(max(u, v)))
            edge_face_count[key] = edge_face_count.get(key, 0) + 1

    nm_pairs = [k for k, v in edge_face_count.items() if v > 2]
    if nm_pairs:
        nm_arr = np.stack(
            [np.stack([verts_np[u], verts_np[v]], axis=0) for (u, v) in nm_pairs],
            axis=0,
        ).astype(np.float32)
    else:
        nm_arr = np.zeros((0, 2, 3), dtype=np.float32)

    # Bowtie vertices: not currently detected by trimesh API alone.  We leave
    # nm_vertices empty in Phase 0; Stage 1 s1_voxelize can extend this later
    # if topology_equivalence demands it.
    nm_verts = np.zeros((0, 3), dtype=np.float32)

    return (
        torch.from_numpy(nm_arr).to(device=device, dtype=torch.float32),
        torch.from_numpy(nm_verts).to(device=device, dtype=torch.float32),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_containers.py -v -k MeshTensors
```

Expected: 9 MeshTensors tests pass (the CUDA test may skip if no GPU is available locally; that's fine).

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/containers.py corep_fast/tests/unit/test_containers.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add MeshTensors dataclass with trimesh constructor

Normalized input-mesh container: vertices, faces, pre-gathered triangles,
face normals, face adjacency, open-boundary edges, and non-manifold edges,
all on a single torch device.  Matches custom/voxelize.py normalization
semantics (center + 0.947 safety margin fit into [0,1]³).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `corep_fast/containers.py` — `CubeBatch` Dataclass

**Files:**
- Modify: `corep_fast/containers.py` (append `CubeBatch` and CSR helpers)
- Modify: `corep_fast/tests/unit/test_containers.py` (append `CubeBatch` tests)

**Reference:** spec §5.3, §5.4

- [ ] **Step 1: Append the failing CubeBatch tests**

Add to `corep_fast/tests/unit/test_containers.py`:

```python
# ---------------------------------------------------------------------------
# CubeBatch
# ---------------------------------------------------------------------------
from corep_fast.containers import CubeBatch, CubeStatus


def _make_empty_cube_batch(N: int = 3, device: str = 'cpu') -> CubeBatch:
    return CubeBatch.empty(num_cubes=N, resolution=64, device=torch.device(device))


def test_cube_batch_empty_shapes():
    cb = _make_empty_cube_batch(N=5)
    assert cb.num_cubes == 5
    assert cb.cube_indices.shape == (5, 3)
    assert cb.cube_indices.dtype == torch.int32
    assert cb.cube_hash.shape == (5,)
    assert cb.cube_hash.dtype == torch.int64
    assert cb.edge_weights.shape == (5, 18)
    assert cb.edge_weights.dtype == torch.int32
    assert cb.face_weights.shape == (5, 12)
    assert cb.status.shape == (5,)
    assert cb.num_components.shape == (5,)


def test_cube_batch_empty_csr_offsets():
    cb = _make_empty_cube_batch(N=5)
    assert cb.tri_offsets.shape == (6,)
    assert cb.tri_offsets.dtype == torch.int64
    assert torch.equal(cb.tri_offsets, torch.zeros(6, dtype=torch.int64))
    assert cb.tri_values.shape == (0,)
    assert cb.bnd_offsets.shape == (6,)
    assert cb.bnd_values.shape == (0,)
    assert cb.point_offsets.shape == (6,)


def test_cube_batch_status_enum():
    assert CubeStatus.OK == 0
    assert CubeStatus.AMBIGUOUS == 1
    assert CubeStatus.UNSOLVABLE == 2
    assert CubeStatus.BUDGET_EXCEEDED == 3


def test_cube_batch_set_tri_csr():
    """Test populating the tri_values/tri_offsets CSR from a list of per-cube lists."""
    cb = _make_empty_cube_batch(N=3)
    per_cube_tris = [
        torch.tensor([5, 7, 9], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    ]
    cb = cb.set_tri_csr(per_cube_tris)
    assert torch.equal(cb.tri_offsets, torch.tensor([0, 3, 3, 4], dtype=torch.int64))
    assert torch.equal(cb.tri_values, torch.tensor([5, 7, 9, 2], dtype=torch.int32))


def test_cube_batch_get_tri_for_cube():
    """Test slicing a single cube's triangles via CSR offsets."""
    cb = _make_empty_cube_batch(N=3)
    per_cube_tris = [
        torch.tensor([5, 7, 9], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    ]
    cb = cb.set_tri_csr(per_cube_tris)
    assert torch.equal(cb.get_tris(0), torch.tensor([5, 7, 9], dtype=torch.int32))
    assert cb.get_tris(1).shape == (0,)
    assert torch.equal(cb.get_tris(2), torch.tensor([2], dtype=torch.int32))


def test_cube_batch_num_loops_is_zero_when_empty():
    cb = _make_empty_cube_batch(N=3)
    assert cb.num_loops == 0
    assert cb.num_loop_edges == 0


def test_cube_batch_cube_hash_roundtrip():
    """Verify cube_hash encodes (ix, iy, iz) reversibly within resolution bounds."""
    cb = _make_empty_cube_batch(N=3)
    indices = torch.tensor([[1, 2, 3], [0, 0, 0], [63, 63, 63]], dtype=torch.int32)
    cb = cb.with_cube_indices(indices)
    expected_hashes = torch.tensor(
        [1 * 64 * 64 + 2 * 64 + 3, 0, 63 * 64 * 64 + 63 * 64 + 63],
        dtype=torch.int64,
    )
    assert torch.equal(cb.cube_hash, expected_hashes)


def test_cube_batch_invariants_check_csr_monotone():
    cb = _make_empty_cube_batch(N=3)
    cb.invariants_check('empty')  # should not raise

    # Corrupt the tri_offsets and verify the check catches it
    bad_offsets = torch.tensor([0, 5, 3, 7], dtype=torch.int64)  # non-monotone
    cb_bad = cb.with_tri_offsets(bad_offsets)
    with pytest.raises(AssertionError):
        cb_bad.invariants_check('test_bad')
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_containers.py -v -k CubeBatch
```

Expected: ImportError (`CubeBatch`, `CubeStatus` not defined).

- [ ] **Step 3: Append `CubeBatch` + helpers to `corep_fast/containers.py`**

Append to `corep_fast/containers.py`:

```python
# ---------------------------------------------------------------------------
# CubeStatus (int enum, stored as int32 tensor)
# ---------------------------------------------------------------------------

class CubeStatus:
    """
    Per-cube status codes.  Stored as int32 in CubeBatch.status.
    Defined as a plain class with int attributes (not enum.IntEnum) because
    we compare them to tensor values directly.
    """
    OK:               int = 0
    AMBIGUOUS:        int = 1
    UNSOLVABLE:       int = 2
    BUDGET_EXCEEDED:  int = 3


# ---------------------------------------------------------------------------
# CubeBatch
# ---------------------------------------------------------------------------

@dataclass
class CubeBatch:
    """All occupied cubes of a single mesh, with per-stage features accumulated."""

    # Spatial identity
    cube_indices: torch.Tensor     # (N, 3)   int32
    cube_hash:    torch.Tensor     # (N,)     int64

    # Stage 1 CSR registries
    tri_offsets: torch.Tensor      # (N+1,)   int64
    tri_values:  torch.Tensor      # (T,)     int32
    bnd_offsets: torch.Tensor      # (N+1,)   int64
    bnd_values:  torch.Tensor      # (B,)     int32
    nm_offsets:  torch.Tensor      # (N+1,)   int64
    nm_values:   torch.Tensor      # (M,)     int32

    # Stage 2
    num_components: torch.Tensor   # (N,)     int32
    num_boundary:   torch.Tensor   # (N,)     int32

    # Stage 3
    edge_weights: torch.Tensor     # (N, 18)  int32

    # Stage 4
    face_weights:  torch.Tensor    # (N, 12)  int32
    point_offsets: torch.Tensor    # (N+1,)   int64
    point_values:  torch.Tensor    # (P, 3)   float32

    # Stage 5-6 — loops as two-level CSR
    loop_cube_off:  torch.Tensor   # (N+1,)   int64
    loop_edge_off:  torch.Tensor   # (L+1,)   int64
    loop_edge_val:  torch.Tensor   # (E,)     int32
    loop_edge_rank: torch.Tensor   # (E,)     int32

    # Stage 6-7 — loop ↔ point matching (-1 if unmatched)
    loop_point_match: torch.Tensor # (L,)     int32

    # Per-cube status
    status: torch.Tensor           # (N,)     int32

    # Bookkeeping
    device:     torch.device
    resolution: int

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def empty(cls, num_cubes: int, resolution: int, device: torch.device) -> 'CubeBatch':
        """Create a CubeBatch with all tensors zero-initialized for `num_cubes` cubes."""
        N = num_cubes
        zeros_i32 = lambda shape: torch.zeros(shape, dtype=torch.int32, device=device)
        zeros_i64 = lambda shape: torch.zeros(shape, dtype=torch.int64, device=device)
        zeros_f32 = lambda shape: torch.zeros(shape, dtype=torch.float32, device=device)

        return cls(
            cube_indices=zeros_i32((N, 3)),
            cube_hash=zeros_i64((N,)),

            tri_offsets=zeros_i64((N + 1,)),
            tri_values=zeros_i32((0,)),
            bnd_offsets=zeros_i64((N + 1,)),
            bnd_values=zeros_i32((0,)),
            nm_offsets=zeros_i64((N + 1,)),
            nm_values=zeros_i32((0,)),

            num_components=zeros_i32((N,)),
            num_boundary=zeros_i32((N,)),

            edge_weights=zeros_i32((N, 18)),
            face_weights=zeros_i32((N, 12)),

            point_offsets=zeros_i64((N + 1,)),
            point_values=zeros_f32((0, 3)),

            loop_cube_off=zeros_i64((N + 1,)),
            loop_edge_off=zeros_i64((1,)),
            loop_edge_val=zeros_i32((0,)),
            loop_edge_rank=torch.full((0,), -1, dtype=torch.int32, device=device),

            loop_point_match=torch.zeros((0,), dtype=torch.int32, device=device),

            status=zeros_i32((N,)),

            device=device,
            resolution=resolution,
        )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    @property
    def num_cubes(self) -> int:
        return int(self.cube_indices.shape[0])

    @property
    def num_loops(self) -> int:
        # loop_cube_off[-1] is the total loop count
        return int(self.loop_cube_off[-1].item()) if self.loop_cube_off.numel() > 0 else 0

    @property
    def num_loop_edges(self) -> int:
        return int(self.loop_edge_val.shape[0])

    # -----------------------------------------------------------------------
    # Mutators (return new CubeBatch — never mutate in place)
    # -----------------------------------------------------------------------

    def with_cube_indices(self, indices: torch.Tensor) -> 'CubeBatch':
        """Replace cube_indices and recompute cube_hash."""
        assert indices.shape == (self.num_cubes, 3)
        assert indices.dtype == torch.int32
        res = self.resolution
        hash_vals = (
            indices[:, 0].to(torch.int64) * res * res
            + indices[:, 1].to(torch.int64) * res
            + indices[:, 2].to(torch.int64)
        )
        return _replace_fields(self, cube_indices=indices, cube_hash=hash_vals)

    def with_tri_offsets(self, offsets: torch.Tensor) -> 'CubeBatch':
        return _replace_fields(self, tri_offsets=offsets)

    def set_tri_csr(self, per_cube_tris: list[torch.Tensor]) -> 'CubeBatch':
        """
        Populate tri_offsets + tri_values from a Python list of per-cube tensors.
        """
        assert len(per_cube_tris) == self.num_cubes
        offsets = torch.zeros(self.num_cubes + 1, dtype=torch.int64, device=self.device)
        lengths = torch.tensor(
            [t.numel() for t in per_cube_tris], dtype=torch.int64, device=self.device,
        )
        offsets[1:] = torch.cumsum(lengths, dim=0)
        if sum(t.numel() for t in per_cube_tris) > 0:
            values = torch.cat(per_cube_tris).to(dtype=torch.int32, device=self.device)
        else:
            values = torch.zeros((0,), dtype=torch.int32, device=self.device)
        return _replace_fields(self, tri_offsets=offsets, tri_values=values)

    def get_tris(self, cube_idx: int) -> torch.Tensor:
        """Return the triangle indices registered to cube `cube_idx`."""
        lo = int(self.tri_offsets[cube_idx].item())
        hi = int(self.tri_offsets[cube_idx + 1].item())
        return self.tri_values[lo:hi]

    # -----------------------------------------------------------------------
    # Invariant checker (debug mode)
    # -----------------------------------------------------------------------

    def invariants_check(self, stage: str) -> None:
        """Raise AssertionError if any structural invariant is violated."""
        N = self.num_cubes

        assert self.cube_indices.shape == (N, 3), f"[{stage}] cube_indices shape"
        assert self.cube_indices.dtype == torch.int32, f"[{stage}] cube_indices dtype"

        # CSR offset monotonicity
        for name, off in [
            ('tri_offsets', self.tri_offsets),
            ('bnd_offsets', self.bnd_offsets),
            ('nm_offsets', self.nm_offsets),
            ('point_offsets', self.point_offsets),
            ('loop_cube_off', self.loop_cube_off),
        ]:
            assert off.shape == (N + 1,), f"[{stage}] {name} shape"
            diffs = off[1:] - off[:-1]
            assert (diffs >= 0).all(), f"[{stage}] {name} is non-monotone"
            assert off[0].item() == 0, f"[{stage}] {name}[0] must be 0"

        # Value length matches offsets[-1]
        assert self.tri_values.shape[0] == int(self.tri_offsets[-1].item()), \
            f"[{stage}] tri_values size mismatch"
        assert self.bnd_values.shape[0] == int(self.bnd_offsets[-1].item()), \
            f"[{stage}] bnd_values size mismatch"

        # Per-cube fixed-dim features
        assert self.edge_weights.shape == (N, 18), f"[{stage}] edge_weights shape"
        assert self.face_weights.shape == (N, 12), f"[{stage}] face_weights shape"
        assert self.status.shape == (N,), f"[{stage}] status shape"
        assert (self.edge_weights >= 0).all(), f"[{stage}] negative edge_weights"
        assert (self.face_weights >= 0).all(), f"[{stage}] negative face_weights"


# ---------------------------------------------------------------------------
# Internal helper: replace a subset of dataclass fields immutably
# ---------------------------------------------------------------------------

def _replace_fields(cb: CubeBatch, **updates) -> CubeBatch:
    from dataclasses import replace as _dc_replace
    return _dc_replace(cb, **updates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_containers.py -v
```

Expected: All MeshTensors tests (from Task 4) still pass, and all 8 new CubeBatch tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/containers.py corep_fast/tests/unit/test_containers.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add CubeBatch dataclass with CSR helpers

CubeBatch holds all cubes of a single mesh with:
- dense (N, *) tensors for fixed-dim features (edge_weights[18], etc.)
- two-level CSR (offsets + values) for variable-length data (triangles per
  cube, points per cube, loops per cube, edges per loop)
- immutable-style mutators (with_*, set_*_csr) that return new instances
- invariants_check() for debug-mode structural validation

Also add CubeStatus constants (OK/AMBIGUOUS/UNSOLVABLE/BUDGET_EXCEEDED).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `corep_fast/profiling/harness.py` — Stage Timer + ProfilingCollector

**Files:**
- Create: `corep_fast/profiling/harness.py`
- Test: `corep_fast/tests/unit/test_harness.py`

**Reference:** spec §7.1

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_harness.py`:

```python
"""Unit tests for corep_fast/profiling/harness.py."""
import json
import time
from pathlib import Path

import pytest
import torch

from corep_fast.profiling.harness import ProfilingCollector, stage_timer


def test_collector_starts_empty():
    pc = ProfilingCollector()
    assert pc.num_stages_recorded == 0
    assert pc.total_wall_time_s == 0.0


def test_stage_timer_records_wall_time():
    pc = ProfilingCollector()
    with stage_timer('sleep50ms', pc):
        time.sleep(0.05)
    assert pc.num_stages_recorded == 1
    rec = pc['sleep50ms']
    assert rec.wall_time_s >= 0.04
    assert rec.wall_time_s < 0.5  # sanity upper bound


def test_stage_timer_records_multiple_stages():
    pc = ProfilingCollector()
    with stage_timer('stage_a', pc):
        time.sleep(0.01)
    with stage_timer('stage_b', pc):
        time.sleep(0.01)
    assert pc.num_stages_recorded == 2
    assert 'stage_a' in pc
    assert 'stage_b' in pc


def test_stage_timer_records_exception_cleanly():
    pc = ProfilingCollector()
    with pytest.raises(ValueError):
        with stage_timer('failing', pc):
            raise ValueError("oops")
    # Even on failure, timing should be recorded
    assert 'failing' in pc


def test_collector_total_wall_time():
    pc = ProfilingCollector()
    with stage_timer('a', pc):
        time.sleep(0.01)
    with stage_timer('b', pc):
        time.sleep(0.01)
    assert pc.total_wall_time_s >= 0.015


def test_collector_to_json_roundtrip(tmp_path: Path):
    pc = ProfilingCollector(mesh_name='sphere.ply', resolution=256, impl='custom')
    with stage_timer('s1_voxelize', pc):
        time.sleep(0.005)
    with stage_timer('s6_collapse_face', pc):
        time.sleep(0.005)
    out_path = tmp_path / "profile.json"
    pc.save_json(out_path)
    # Reload
    loaded = json.loads(out_path.read_text())
    assert loaded['mesh'] == 'sphere.ply'
    assert loaded['resolution'] == 256
    assert loaded['impl'] == 'custom'
    assert 's1_voxelize' in loaded['stages']
    assert 's6_collapse_face' in loaded['stages']
    assert loaded['stages']['s1_voxelize']['wall_time_s'] > 0
    assert 'total_wall_time_s' in loaded


def test_substage_timer_nested():
    """Substages within a stage should be recorded as nested entries."""
    pc = ProfilingCollector()
    with stage_timer('s6_collapse_face', pc):
        with stage_timer('algebraic_pruning', pc, parent='s6_collapse_face'):
            time.sleep(0.005)
        with stage_timer('enumeration', pc, parent='s6_collapse_face'):
            time.sleep(0.005)
    s6 = pc['s6_collapse_face']
    assert 'substages' in s6.extra
    assert 'algebraic_pruning' in s6.extra['substages']
    assert 'enumeration' in s6.extra['substages']


def test_stage_timer_records_gpu_mem_when_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pc = ProfilingCollector()
    with stage_timer('gpu_alloc', pc):
        big = torch.zeros(1024 * 1024 * 16, dtype=torch.float32, device='cuda')  # 64 MiB
        del big
    rec = pc['gpu_alloc']
    # Peak should be at least the allocation size
    assert rec.peak_gpu_mem_bytes >= 0  # may be 0 if caching allocator reused memory
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_harness.py -v
```

Expected: ImportError (harness module not defined).

- [ ] **Step 3: Write the harness module**

Create `corep_fast/profiling/harness.py`:

```python
"""
Profiling harness: GPU-fenced stage timer + result collector.

Usage
-----
    pc = ProfilingCollector(mesh_name='sphere.ply', resolution=1024, impl='corep_fast')
    with stage_timer('s1_voxelize', pc):
        ...
    with stage_timer('s6_collapse_face', pc):
        with stage_timer('algebraic_pruning', pc, parent='s6_collapse_face'):
            ...
    pc.save_json('profiling/runs/sphere_1024.json')

Reference: spec §7.1.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Record + Collector
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """One timing record for a single stage (or substage)."""
    wall_time_s: float = 0.0
    peak_gpu_mem_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  # holds substages, notes, etc.


@dataclass
class ProfilingCollector:
    """
    Holds all timing records for one pipeline invocation on one mesh.

    Fields `mesh_name`, `resolution`, `impl` are metadata passed through to
    the JSON output unchanged.
    """
    mesh_name: str = ''
    resolution: int = 0
    impl: str = ''          # 'custom' or 'corep_fast' or 'ab'
    _records: dict[str, StageRecord] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        wall_time_s: float,
        peak_gpu_mem_bytes: int = 0,
        parent: Optional[str] = None,
    ) -> None:
        """
        Record a stage's timing.  If `parent` is given, the record is nested
        inside the parent stage's `extra['substages']` dict rather than being
        a top-level stage.
        """
        rec = StageRecord(
            wall_time_s=wall_time_s, peak_gpu_mem_bytes=peak_gpu_mem_bytes,
        )
        if parent is None:
            self._records[stage] = rec
        else:
            if parent not in self._records:
                # Parent not yet recorded (nested context exited first); create stub.
                self._records[parent] = StageRecord()
            self._records[parent].extra.setdefault('substages', {})[stage] = \
                {'wall_time_s': wall_time_s, 'peak_gpu_mem_bytes': peak_gpu_mem_bytes}

    @property
    def num_stages_recorded(self) -> int:
        return len(self._records)

    @property
    def total_wall_time_s(self) -> float:
        return sum(r.wall_time_s for r in self._records.values())

    def __contains__(self, stage: str) -> bool:
        return stage in self._records

    def __getitem__(self, stage: str) -> StageRecord:
        return self._records[stage]

    def save_json(self, out_path: str | Path) -> None:
        """Persist to a JSON file matching the schema in spec §7.1."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'mesh': self.mesh_name,
            'resolution': self.resolution,
            'impl': self.impl,
            'stages': {
                name: {
                    'wall_time_s': rec.wall_time_s,
                    'peak_gpu_mem_bytes': rec.peak_gpu_mem_bytes,
                    **({'substages': rec.extra['substages']}
                       if 'substages' in rec.extra else {}),
                }
                for name, rec in self._records.items()
            },
            'total_wall_time_s': self.total_wall_time_s,
            'total_peak_gpu_mem_bytes': max(
                (r.peak_gpu_mem_bytes for r in self._records.values()),
                default=0,
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Stage timer context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def stage_timer(
    name: str,
    collector: ProfilingCollector,
    *,
    parent: Optional[str] = None,
):
    """
    GPU-fenced timing context manager.  Synchronizes CUDA before entering and
    exiting to avoid measuring asynchronous kernel launches instead of actual
    compute.  Records wall time and peak GPU memory (if CUDA is available).
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
        mem_before = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_before = 0

    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda_available:
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() - mem_before
        else:
            peak_mem = 0
        wall = time.perf_counter() - t0
        collector.record(
            stage=name,
            wall_time_s=wall,
            peak_gpu_mem_bytes=max(peak_mem, 0),
            parent=parent,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_harness.py -v
```

Expected: 8 tests pass (`test_stage_timer_records_gpu_mem_when_cuda` may skip if no GPU available).

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/profiling/harness.py corep_fast/tests/unit/test_harness.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add profiling harness with stage_timer + collector

GPU-fenced context manager that records wall time and peak GPU memory per
stage, with nested substage support via the `parent=` argument.  Collector
serializes to a JSON schema matching spec §7.1, ready to drive the A/B
rig and baseline_runner in later tasks.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `corep_fast/profiling/topology_equivalence.py` — L1 Integer Fields Layer

**Files:**
- Create: `corep_fast/profiling/topology_equivalence.py` (L1 portion; L2 in Task 8; L3/4/5 in Task 9)
- Test: `corep_fast/tests/unit/test_topology_equivalence.py` (L1 portion)

**Reference:** spec §7.2 (5-layer checker), §14.C (canonical forms)

- [ ] **Step 1: Write the failing L1 test**

Create `corep_fast/tests/unit/test_topology_equivalence.py`:

```python
"""Unit tests for corep_fast/profiling/topology_equivalence.py."""
import pytest
import torch

from corep_fast.containers import CubeBatch
from corep_fast.profiling.topology_equivalence import (
    EquivalenceReport,
    check_layer1_integer_fields,
    check_topology_equivalence,
)


def _make_cube_batch_with_fields(num_cubes=3, device='cpu') -> CubeBatch:
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device(device))
    # Populate some fields deterministically
    cb.num_components[:] = torch.tensor([1, 2, 1], dtype=torch.int32)
    cb.num_boundary[:] = torch.tensor([0, 0, 1], dtype=torch.int32)
    cb.edge_weights[:] = torch.arange(num_cubes * 18, dtype=torch.int32).reshape(num_cubes, 18) % 4
    cb.face_weights[:] = torch.arange(num_cubes * 12, dtype=torch.int32).reshape(num_cubes, 12) % 3
    cb.status[:] = torch.zeros(num_cubes, dtype=torch.int32)
    return cb


def test_l1_equivalence_passes_on_identical_batches():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is True
    assert report.num_cubes == 3
    assert len(report.mismatches) == 0


def test_l1_equivalence_fails_on_num_components_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.num_components[1] = 3  # diverge
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'num_components' for m in report.mismatches)
    assert report.mismatches[0].cube_idx == 1
    assert report.mismatches[0].value_a == 2
    assert report.mismatches[0].value_b == 3


def test_l1_equivalence_fails_on_edge_weights_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.edge_weights[0, 5] = 99  # diverge at cube 0, edge 5
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'edge_weights' for m in report.mismatches)


def test_l1_equivalence_fails_on_face_weights_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.face_weights[2, 7] = 99
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'face_weights' for m in report.mismatches)


def test_l1_equivalence_fails_on_status_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.status[0] = 2  # UNSOLVABLE
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'status' for m in report.mismatches)


def test_l1_equivalence_fails_on_shape_mismatch():
    cb_a = _make_cube_batch_with_fields(num_cubes=3)
    cb_b = _make_cube_batch_with_fields(num_cubes=4)
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    # Shape mismatch is reported as a single "shape" mismatch
    assert any(m.field == 'shape' for m in report.mismatches)


def test_top_level_check_layer1_only_runs_when_shapes_match():
    """The top-level check_topology_equivalence should short-circuit on shape mismatch."""
    cb_a = _make_cube_batch_with_fields(num_cubes=3)
    cb_b = _make_cube_batch_with_fields(num_cubes=4)
    report = check_topology_equivalence(cb_a, cb_b, layers=['l1'])
    assert not report.all_passed()
    assert report.layer1.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_topology_equivalence.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the L1 portion of topology_equivalence**

Create `corep_fast/profiling/topology_equivalence.py`:

```python
"""
5-layer topology equivalence checker between two CubeBatches.

Layers (see spec §7.2):
    L1 — integer fields: num_components, num_boundary, edge_weights,
         face_weights, status  (exact equality)
    L2 — loop set per cube, compared via canonical form
    L3 — rank assignment on loop edges
    L4 — loop ↔ component_point matching
    L5 — 3D component point coordinates (rtol=1e-4, atol=1e-6)

L1 is implemented in this task.  L2 in Task 8, L3/L4/L5 in Task 9.

The top-level entry point `check_topology_equivalence` runs the requested
layers in order and fails fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from corep_fast.containers import CubeBatch


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass
class Mismatch:
    """A single equivalence violation."""
    field: str                  # 'num_components', 'edge_weights', 'loop_set', etc.
    cube_idx: int               # which cube diverged (0 for shape-level)
    value_a: object = None
    value_b: object = None
    detail: str = ''


@dataclass
class LayerReport:
    layer_name: str
    passed: bool = True
    num_cubes: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    def add_mismatch(self, m: Mismatch) -> None:
        self.passed = False
        self.mismatches.append(m)


@dataclass
class EquivalenceReport:
    layer1: Optional[LayerReport] = None
    layer2: Optional[LayerReport] = None
    layer3: Optional[LayerReport] = None
    layer4: Optional[LayerReport] = None
    layer5: Optional[LayerReport] = None

    def all_passed(self) -> bool:
        return all(
            (r is None or r.passed)
            for r in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5)
        )

    def pretty_print(self) -> str:
        lines = ["Topology Equivalence Report"]
        for r in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5):
            if r is None:
                continue
            status = "PASS" if r.passed else f"FAIL ({len(r.mismatches)} mismatches)"
            lines.append(f"  {r.layer_name}: {status}")
            if not r.passed and r.mismatches:
                first = r.mismatches[0]
                lines.append(
                    f"    First mismatch: cube {first.cube_idx}, field {first.field!r}, "
                    f"A={first.value_a!r}, B={first.value_b!r}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 1: integer fields
# ---------------------------------------------------------------------------

_L1_FIELDS = [
    'num_components',
    'num_boundary',
    'edge_weights',
    'face_weights',
    'status',
]


def check_layer1_integer_fields(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    """
    Exact equality check on all integer per-cube fields.  Shape mismatch is
    reported as a single `field='shape'` mismatch with cube_idx=0.
    """
    report = LayerReport(layer_name='L1 integer fields', num_cubes=cb_a.num_cubes)

    if cb_a.num_cubes != cb_b.num_cubes:
        report.add_mismatch(Mismatch(
            field='shape',
            cube_idx=0,
            value_a=cb_a.num_cubes,
            value_b=cb_b.num_cubes,
            detail=f'num_cubes differs: {cb_a.num_cubes} vs {cb_b.num_cubes}',
        ))
        return report

    for fname in _L1_FIELDS:
        ta = getattr(cb_a, fname).cpu()
        tb = getattr(cb_b, fname).cpu()
        if ta.shape != tb.shape:
            report.add_mismatch(Mismatch(
                field=fname,
                cube_idx=0,
                value_a=tuple(ta.shape),
                value_b=tuple(tb.shape),
                detail=f'{fname} shape differs',
            ))
            continue
        if not torch.equal(ta, tb):
            # Find first cube with a divergent value
            if ta.dim() == 1:
                diff_mask = (ta != tb)
                cube_idx = int(diff_mask.nonzero(as_tuple=False)[0].item())
                va = int(ta[cube_idx].item())
                vb = int(tb[cube_idx].item())
            else:  # 2-D like edge_weights[N,18]
                diff_mask = (ta != tb).any(dim=-1)
                cube_idx = int(diff_mask.nonzero(as_tuple=False)[0].item())
                va = ta[cube_idx].tolist()
                vb = tb[cube_idx].tolist()
            report.add_mismatch(Mismatch(
                field=fname, cube_idx=cube_idx, value_a=va, value_b=vb,
            ))
    return report


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

_ALL_LAYERS = ['l1', 'l2', 'l3', 'l4', 'l5']


def check_topology_equivalence(
    cb_a: CubeBatch,
    cb_b: CubeBatch,
    layers: list[str] = None,
) -> EquivalenceReport:
    """
    Run the requested layers in order.  On the first failing layer, subsequent
    layers are skipped (left as None in the report) because downstream
    equivalence checks are meaningless when upstream structure is broken.
    """
    if layers is None:
        layers = _ALL_LAYERS

    report = EquivalenceReport()

    if 'l1' in layers:
        report.layer1 = check_layer1_integer_fields(cb_a, cb_b)
        if not report.layer1.passed:
            return report

    # L2/L3/L4/L5 added in Tasks 8 and 9
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_topology_equivalence.py -v
```

Expected: 7 L1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/profiling/topology_equivalence.py corep_fast/tests/unit/test_topology_equivalence.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add L1 integer-field topology equivalence checker

First of five layers in the topology equivalence checker (spec §7.2):
exact per-cube equality on num_components, num_boundary, edge_weights[18],
face_weights[12], and status.  Shape mismatches short-circuit.  Mismatch
report surfaces the first diverging cube idx + old/new value pair.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Loop Canonical Form + L2 Loop Set Equivalence

**Files:**
- Modify: `corep_fast/profiling/topology_equivalence.py` (add canonical form + L2)
- Modify: `corep_fast/tests/unit/test_topology_equivalence.py` (add L2 tests)

**Reference:** spec §7.2 L2, §14.C (canonical forms)

- [ ] **Step 1: Append L2 tests**

Append to `corep_fast/tests/unit/test_topology_equivalence.py`:

```python
# ---------------------------------------------------------------------------
# Layer 2: Loop canonical form + loop set equivalence
# ---------------------------------------------------------------------------
from corep_fast.profiling.topology_equivalence import (
    canonicalize_loop,
    canonicalize_loop_set,
    check_layer2_loop_structure,
)


def test_canonicalize_loop_rotates_to_min_start():
    """Canonical form rotates so the minimum edge is first."""
    loop = [5, 7, 11, 3]
    canon = canonicalize_loop(loop)
    assert canon == (3, 5, 7, 11)


def test_canonicalize_loop_prefers_forward_direction():
    """If forward and reversed start the same, forward wins."""
    loop = [3, 5, 7, 11]  # minimum 3 at index 0; reversed is [11,7,5,3] -> rot to [3,11,7,5]
    canon = canonicalize_loop(loop)
    assert canon == (3, 5, 7, 11)  # forward beats reversed because (3,5) < (3,11)


def test_canonicalize_loop_picks_reversed_when_smaller():
    loop = [3, 11, 7, 5]  # forward canon: (3,11,7,5); reversed canon: (3,5,7,11)
    canon = canonicalize_loop(loop)
    assert canon == (3, 5, 7, 11)


def test_canonicalize_loop_single_edge():
    assert canonicalize_loop([7]) == (7,)


def test_canonicalize_loop_empty_is_error():
    with pytest.raises(ValueError):
        canonicalize_loop([])


def test_canonicalize_loop_set_sorts_loops():
    loops = [[5, 7, 11], [2, 4, 6]]
    canon = canonicalize_loop_set(loops)
    assert canon == ((2, 4, 6), (5, 7, 11))


def test_canonicalize_loop_set_handles_rotation_and_reflection_per_loop():
    loops = [[7, 11, 5], [6, 4, 2]]
    canon = canonicalize_loop_set(loops)
    # Loop 0 canon: (5, 7, 11); Loop 1 canon: (2, 4, 6)
    assert canon == ((2, 4, 6), (5, 7, 11))


def _cube_batch_with_loops(
    loops_per_cube: list[list[list[int]]],
    num_cubes: int = None,
    device: str = 'cpu',
) -> CubeBatch:
    """Helper: build a CubeBatch whose loop CSR contains the given nested list."""
    if num_cubes is None:
        num_cubes = len(loops_per_cube)
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device(device))

    # Build two-level CSR
    loop_cube_off = [0]
    loop_edge_off = [0]
    loop_edge_val = []
    for cube_loops in loops_per_cube:
        loop_cube_off.append(loop_cube_off[-1] + len(cube_loops))
        for loop in cube_loops:
            loop_edge_off.append(loop_edge_off[-1] + len(loop))
            loop_edge_val.extend(loop)

    cb.loop_cube_off = torch.tensor(loop_cube_off, dtype=torch.int64)
    cb.loop_edge_off = torch.tensor(loop_edge_off, dtype=torch.int64)
    cb.loop_edge_val = torch.tensor(loop_edge_val, dtype=torch.int32)
    cb.loop_edge_rank = torch.full((len(loop_edge_val),), -1, dtype=torch.int32)
    return cb


def test_l2_equivalence_passes_on_identical_loops():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]], [[2, 4, 6]]])
    cb_b = _cube_batch_with_loops([[[5, 7, 11, 3]], [[2, 4, 6]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert report.passed


def test_l2_equivalence_passes_on_rotated_loops():
    """Rotated loop is equivalent via canonical form."""
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]]])
    cb_b = _cube_batch_with_loops([[[3, 5, 7, 11]]])  # rotated to start at min
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert report.passed


def test_l2_equivalence_passes_on_reversed_loops():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]]])
    cb_b = _cube_batch_with_loops([[[3, 11, 7, 5]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert report.passed


def test_l2_equivalence_fails_on_different_loop_sets():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]]])
    cb_b = _cube_batch_with_loops([[[5, 7, 11, 4]]])  # 3 → 4
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].cube_idx == 0
    assert report.mismatches[0].field == 'loop_set'


def test_l2_equivalence_fails_on_different_loop_count():
    cb_a = _cube_batch_with_loops([[[1, 2, 3], [4, 5, 6]]])
    cb_b = _cube_batch_with_loops([[[1, 2, 3]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert not report.passed
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_topology_equivalence.py -v -k "canonicalize or l2"
```

Expected: ImportError (canonicalize_loop / check_layer2_loop_structure not defined).

- [ ] **Step 3: Append L2 + canonical form helpers to topology_equivalence.py**

Append to `corep_fast/profiling/topology_equivalence.py`:

```python
# ---------------------------------------------------------------------------
# Loop canonical form (spec §14.C)
# ---------------------------------------------------------------------------

def canonicalize_loop(edges: list[int] | torch.Tensor) -> tuple[int, ...]:
    """
    Return the canonical form of a single loop (list of edge indices).

    1. Rotate so the minimum edge index is at position 0.
    2. Pick the lexicographically smaller of the forward and reversed rotations.
    """
    if isinstance(edges, torch.Tensor):
        seq = edges.tolist()
    else:
        seq = list(edges)
    if not seq:
        raise ValueError("canonicalize_loop: empty loop")

    n = len(seq)
    # Forward canonical: start at the minimum element
    min_idx = min(range(n), key=lambda i: seq[i])
    fwd = tuple(seq[(min_idx + k) % n] for k in range(n))

    # Reversed canonical: reverse, then start at the minimum element
    rev = list(reversed(seq))
    min_idx_r = min(range(n), key=lambda i: rev[i])
    rev_canon = tuple(rev[(min_idx_r + k) % n] for k in range(n))

    return fwd if fwd <= rev_canon else rev_canon


def canonicalize_loop_set(loops: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """
    Return the canonical form of a set of loops:
    1. Canonicalize each loop individually
    2. Sort the resulting canonical tuples lexicographically
    """
    canon_each = [canonicalize_loop(l) for l in loops]
    return tuple(sorted(canon_each))


# ---------------------------------------------------------------------------
# Layer 2: Loop set equivalence
# ---------------------------------------------------------------------------

def _extract_loops_for_cube(cb: CubeBatch, cube_idx: int) -> list[list[int]]:
    """Return the loops (as Python lists of edge indices) for a given cube."""
    loop_lo = int(cb.loop_cube_off[cube_idx].item())
    loop_hi = int(cb.loop_cube_off[cube_idx + 1].item())
    loops = []
    for l_idx in range(loop_lo, loop_hi):
        e_lo = int(cb.loop_edge_off[l_idx].item())
        e_hi = int(cb.loop_edge_off[l_idx + 1].item())
        loops.append(cb.loop_edge_val[e_lo:e_hi].tolist())
    return loops


def check_layer2_loop_structure(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    """
    Per-cube loop set equivalence via canonical form.  Assumes L1 has passed
    (same num_cubes).
    """
    report = LayerReport(layer_name='L2 loop structure', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        loops_a = _extract_loops_for_cube(cb_a, cube_idx)
        loops_b = _extract_loops_for_cube(cb_b, cube_idx)

        if len(loops_a) != len(loops_b):
            report.add_mismatch(Mismatch(
                field='loop_count',
                cube_idx=cube_idx,
                value_a=len(loops_a),
                value_b=len(loops_b),
            ))
            continue

        canon_a = canonicalize_loop_set(loops_a) if loops_a else ()
        canon_b = canonicalize_loop_set(loops_b) if loops_b else ()
        if canon_a != canon_b:
            report.add_mismatch(Mismatch(
                field='loop_set',
                cube_idx=cube_idx,
                value_a=canon_a,
                value_b=canon_b,
            ))

    return report
```

Also update `check_topology_equivalence` to wire L2:

```python
    # Replace the L2 stub with:
    if 'l2' in layers:
        report.layer2 = check_layer2_loop_structure(cb_a, cb_b)
        if not report.layer2.passed:
            return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_topology_equivalence.py -v
```

Expected: All 7 L1 tests + 5 canonical form tests + 5 L2 tests = 17 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/profiling/topology_equivalence.py corep_fast/tests/unit/test_topology_equivalence.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add L2 loop set equivalence with canonical form

canonicalize_loop normalizes a single loop via (rotate to min) + (pick
lexicographically smaller of forward/reversed), per spec §14.C.
canonicalize_loop_set sorts the per-loop canonical tuples to make the
loop set itself canonical.  check_layer2_loop_structure uses these to
compare loop sets per cube across two CubeBatches, tolerant to any
rotation/reflection of individual loops.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: L3/L4/L5 Extensions — Rank, Point Matching, Coordinates

**Files:**
- Modify: `corep_fast/profiling/topology_equivalence.py`
- Modify: `corep_fast/tests/unit/test_topology_equivalence.py`

**Reference:** spec §7.2 L3–L5

- [ ] **Step 1: Append L3/L4/L5 tests**

Append to `corep_fast/tests/unit/test_topology_equivalence.py`:

```python
from corep_fast.profiling.topology_equivalence import (
    check_layer3_rank_assignment,
    check_layer4_point_matching,
    check_layer5_point_coordinates,
)


def _cube_batch_with_ranks_and_points(
    loops_per_cube: list[list[list[int]]],
    ranks_per_cube: list[list[list[int]]],
    match_per_cube: list[list[int]],
    points_per_cube: list[list[list[float]]],
) -> CubeBatch:
    """Build CubeBatch with loops + ranks + point matching + points."""
    num_cubes = len(loops_per_cube)
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device('cpu'))

    # Two-level CSR for loops
    loop_cube_off = [0]
    loop_edge_off = [0]
    loop_edge_val = []
    loop_edge_rank = []
    loop_point_match = []
    point_offsets = [0]
    point_values = []

    for ci in range(num_cubes):
        cube_loops = loops_per_cube[ci]
        cube_ranks = ranks_per_cube[ci]
        cube_match = match_per_cube[ci]
        cube_points = points_per_cube[ci]

        loop_cube_off.append(loop_cube_off[-1] + len(cube_loops))
        for li, loop in enumerate(cube_loops):
            loop_edge_off.append(loop_edge_off[-1] + len(loop))
            loop_edge_val.extend(loop)
            loop_edge_rank.extend(cube_ranks[li])
        loop_point_match.extend(cube_match)
        point_offsets.append(point_offsets[-1] + len(cube_points))
        point_values.extend(cube_points)

    cb.loop_cube_off = torch.tensor(loop_cube_off, dtype=torch.int64)
    cb.loop_edge_off = torch.tensor(loop_edge_off, dtype=torch.int64)
    cb.loop_edge_val = torch.tensor(loop_edge_val, dtype=torch.int32)
    cb.loop_edge_rank = torch.tensor(loop_edge_rank, dtype=torch.int32)
    cb.loop_point_match = torch.tensor(loop_point_match, dtype=torch.int32)
    cb.point_offsets = torch.tensor(point_offsets, dtype=torch.int64)
    if point_values:
        cb.point_values = torch.tensor(point_values, dtype=torch.float32)
    else:
        cb.point_values = torch.zeros((0, 3), dtype=torch.float32)
    return cb


# ── L3 tests ──

def test_l3_passes_identical_ranks():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    report = check_layer3_rank_assignment(cb_a, cb_b)
    assert report.passed


def test_l3_fails_on_rank_diff():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 1, 0]]],  # rank[1] changed
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    report = check_layer3_rank_assignment(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].field == 'loop_edge_rank'


# ── L4 tests ──

def test_l4_passes_identical_matching():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[0, 1]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[0, 1]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    report = check_layer4_point_matching(cb_a, cb_b)
    assert report.passed


def test_l4_fails_on_match_swap():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[0, 1]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[1, 0]],  # swapped
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    report = check_layer4_point_matching(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].field == 'loop_point_match'


# ── L5 tests ──

def test_l5_passes_within_tolerance():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.50005, 0.49998, 0.50001]]],  # within rtol=1e-4
    )
    report = check_layer5_point_coordinates(cb_a, cb_b)
    assert report.passed


def test_l5_fails_outside_tolerance():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.6, 0.5, 0.5]]],  # 0.1 off — outside tolerance
    )
    report = check_layer5_point_coordinates(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].field == 'point_values'


def test_full_5layer_check_passes():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    report = check_topology_equivalence(cb_a, cb_b, layers=['l1', 'l2', 'l3', 'l4', 'l5'])
    assert report.all_passed()
```

- [ ] **Step 2: Run tests to verify L3/L4/L5 tests fail**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_topology_equivalence.py -v -k "l3 or l4 or l5 or full_5"
```

Expected: ImportError (check_layer3/4/5 not defined).

- [ ] **Step 3: Append L3/L4/L5 implementations to topology_equivalence.py**

Append to `corep_fast/profiling/topology_equivalence.py`:

```python
# ---------------------------------------------------------------------------
# Layer 3: Rank assignment
# ---------------------------------------------------------------------------

def check_layer3_rank_assignment(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    """Exact equality of loop_edge_rank across all loops of all cubes."""
    report = LayerReport(layer_name='L3 rank assignment', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        loop_lo_a = int(cb_a.loop_cube_off[cube_idx].item())
        loop_hi_a = int(cb_a.loop_cube_off[cube_idx + 1].item())
        loop_lo_b = int(cb_b.loop_cube_off[cube_idx].item())
        loop_hi_b = int(cb_b.loop_cube_off[cube_idx + 1].item())

        for li_a, li_b in zip(range(loop_lo_a, loop_hi_a), range(loop_lo_b, loop_hi_b)):
            e_lo_a = int(cb_a.loop_edge_off[li_a].item())
            e_hi_a = int(cb_a.loop_edge_off[li_a + 1].item())
            e_lo_b = int(cb_b.loop_edge_off[li_b].item())
            e_hi_b = int(cb_b.loop_edge_off[li_b + 1].item())

            rank_a = cb_a.loop_edge_rank[e_lo_a:e_hi_a]
            rank_b = cb_b.loop_edge_rank[e_lo_b:e_hi_b]
            if rank_a.shape != rank_b.shape or not torch.equal(rank_a, rank_b):
                report.add_mismatch(Mismatch(
                    field='loop_edge_rank',
                    cube_idx=cube_idx,
                    value_a=rank_a.tolist(),
                    value_b=rank_b.tolist(),
                ))
                break  # one mismatch per cube is enough
    return report


# ---------------------------------------------------------------------------
# Layer 4: Loop ↔ point matching
# ---------------------------------------------------------------------------

def check_layer4_point_matching(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    """Exact equality of loop_point_match per cube."""
    report = LayerReport(layer_name='L4 point matching', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        loop_lo_a = int(cb_a.loop_cube_off[cube_idx].item())
        loop_hi_a = int(cb_a.loop_cube_off[cube_idx + 1].item())
        loop_lo_b = int(cb_b.loop_cube_off[cube_idx].item())
        loop_hi_b = int(cb_b.loop_cube_off[cube_idx + 1].item())

        match_a = cb_a.loop_point_match[loop_lo_a:loop_hi_a]
        match_b = cb_b.loop_point_match[loop_lo_b:loop_hi_b]

        if match_a.shape != match_b.shape or not torch.equal(match_a, match_b):
            report.add_mismatch(Mismatch(
                field='loop_point_match',
                cube_idx=cube_idx,
                value_a=match_a.tolist(),
                value_b=match_b.tolist(),
            ))
    return report


# ---------------------------------------------------------------------------
# Layer 5: Component point coordinates (float tolerance)
# ---------------------------------------------------------------------------

_L5_RTOL = 1e-4
_L5_ATOL = 1e-6


def check_layer5_point_coordinates(
    cb_a: CubeBatch,
    cb_b: CubeBatch,
    rtol: float = _L5_RTOL,
    atol: float = _L5_ATOL,
) -> LayerReport:
    """
    Per-cube comparison of component_point 3D coordinates within (rtol, atol).
    """
    report = LayerReport(layer_name='L5 point coordinates', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        lo_a = int(cb_a.point_offsets[cube_idx].item())
        hi_a = int(cb_a.point_offsets[cube_idx + 1].item())
        lo_b = int(cb_b.point_offsets[cube_idx].item())
        hi_b = int(cb_b.point_offsets[cube_idx + 1].item())

        pts_a = cb_a.point_values[lo_a:hi_a]
        pts_b = cb_b.point_values[lo_b:hi_b]

        if pts_a.shape != pts_b.shape:
            report.add_mismatch(Mismatch(
                field='point_values',
                cube_idx=cube_idx,
                value_a=tuple(pts_a.shape),
                value_b=tuple(pts_b.shape),
                detail='point count differs',
            ))
            continue

        if pts_a.numel() > 0 and not torch.allclose(pts_a.cpu(), pts_b.cpu(), rtol=rtol, atol=atol):
            max_diff = (pts_a.cpu() - pts_b.cpu()).abs().max().item()
            report.add_mismatch(Mismatch(
                field='point_values',
                cube_idx=cube_idx,
                value_a=pts_a.tolist(),
                value_b=pts_b.tolist(),
                detail=f'max abs diff = {max_diff:.2e}',
            ))
    return report
```

Also update `check_topology_equivalence` to wire L3/L4/L5:

```python
    if 'l3' in layers:
        report.layer3 = check_layer3_rank_assignment(cb_a, cb_b)
        if not report.layer3.passed:
            return report

    if 'l4' in layers:
        report.layer4 = check_layer4_point_matching(cb_a, cb_b)
        if not report.layer4.passed:
            return report

    if 'l5' in layers:
        report.layer5 = check_layer5_point_coordinates(cb_a, cb_b)
```

- [ ] **Step 4: Run all topology tests**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_topology_equivalence.py -v
```

Expected: All 27 tests pass (7 L1 + 7 canonical/L2 + 5 L2 + 2 L3 + 2 L4 + 2 L5 + 2 integration).

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/profiling/topology_equivalence.py corep_fast/tests/unit/test_topology_equivalence.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): complete 5-layer topology equivalence checker (L3-L5)

L3: exact per-loop rank assignment equality.
L4: exact loop-to-point matching index equality.
L5: point coordinate floating-point comparison with rtol=1e-4, atol=1e-6.

The top-level check_topology_equivalence now chains all 5 layers with
fail-fast: each subsequent layer is skipped if an upstream layer fails.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `corep_fast/interop/from_custom.py` — list-of-dict → CubeBatch

**Files:**
- Create: `corep_fast/interop/from_custom.py`
- Test: `corep_fast/tests/unit/test_interop.py`

**Reference:** spec §5.5, §14.D

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_interop.py`:

```python
"""Unit tests for corep_fast/interop/from_custom.py and to_custom.py."""
import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.interop.from_custom import cube_batch_from_custom


def _make_fake_face_registers() -> list[dict]:
    """Simulate custom/'s output: a list of dicts, one per cube."""
    return [
        {
            'cube_indices': (10, 20, 30),
            'face_indices': [0, 1, 2],
            'num_components': 1,
            'num_boundary': 0,
            'edge_weights': list(range(18)),
            'face_weights': list(range(12)),
            'component_points': [[0.5, 0.5, 0.5]],
        },
        {
            'cube_indices': (11, 20, 30),
            'face_indices': [3, 4],
            'num_components': 2,
            'num_boundary': 1,
            'edge_weights': [1] * 18,
            'face_weights': [0] * 12,
            'component_points': [[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]],
        },
    ]


def _make_mesh_tensors() -> MeshTensors:
    mesh = trimesh.creation.box(extents=[1., 1., 1.])
    return MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')


def test_from_custom_basic():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.num_cubes == 2
    assert torch.equal(cb.cube_indices[0], torch.tensor([10, 20, 30], dtype=torch.int32))
    assert torch.equal(cb.cube_indices[1], torch.tensor([11, 20, 30], dtype=torch.int32))


def test_from_custom_tri_csr():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    tris_0 = cb.get_tris(0)
    assert torch.equal(tris_0, torch.tensor([0, 1, 2], dtype=torch.int32))
    tris_1 = cb.get_tris(1)
    assert torch.equal(tris_1, torch.tensor([3, 4], dtype=torch.int32))


def test_from_custom_scalar_fields():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.num_components[0].item() == 1
    assert cb.num_components[1].item() == 2
    assert cb.num_boundary[0].item() == 0
    assert cb.num_boundary[1].item() == 1


def test_from_custom_edge_weights():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.edge_weights[0].tolist() == list(range(18))
    assert cb.edge_weights[1].tolist() == [1] * 18


def test_from_custom_face_weights():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.face_weights[0].tolist() == list(range(12))
    assert cb.face_weights[1].tolist() == [0] * 12


def test_from_custom_component_points():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    # Cube 0 has 1 point, cube 1 has 2 points
    assert cb.point_offsets.tolist() == [0, 1, 3]
    assert cb.point_values.shape == (3, 3)
    assert torch.allclose(cb.point_values[0], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(cb.point_values[2], torch.tensor([0.7, 0.7, 0.7]))


def test_from_custom_missing_optional_fields():
    """If fields like loops or edge_weights are absent, they default to zeros."""
    regs = [{'cube_indices': (5, 5, 5), 'face_indices': [0]}]
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')
    assert cb.num_cubes == 1
    assert cb.edge_weights[0].sum().item() == 0  # default zeros
    assert cb.num_components[0].item() == 0


def test_from_custom_include_subset():
    """Only populate specified fields; others get sentinels/zeros."""
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, include={'cube_indices', 'face_indices'}, device='cpu')
    assert cb.num_cubes == 2
    # tri CSR should be populated
    assert cb.tri_values.shape[0] > 0
    # edge_weights should be zero (not populated)
    assert cb.edge_weights.sum().item() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_interop.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write from_custom.py**

Create `corep_fast/interop/from_custom.py`:

```python
"""
Convert custom/'s list-of-dict face_registers into a corep_fast CubeBatch.

Reference: spec §14.D.
"""
from __future__ import annotations

from typing import Optional

import torch

from corep_fast.containers import CubeBatch, MeshTensors


def cube_batch_from_custom(
    face_registers: list[dict],
    mesh: MeshTensors,
    *,
    include: Optional[set[str]] = None,
    device: str | torch.device = 'cpu',
) -> CubeBatch:
    """
    Build a CubeBatch from custom/'s output format.

    Args:
        face_registers: list of dicts (one per cube) as produced by
            custom/feature.py stages.  Required keys: 'cube_indices'.
            Optional keys: 'face_indices', 'num_components', 'num_boundary',
            'edge_weights', 'face_weights', 'component_points', 'loops',
            'sorted_loops', 'status'.
        mesh: MeshTensors (used for resolution; not otherwise consumed here).
        include: if provided, only populate these field groups.  Unspecified
            groups get default zeros.  Valid values: 'cube_indices',
            'face_indices', 'num_components', 'num_boundary', 'edge_weights',
            'face_weights', 'component_points', 'loops', 'status'.
        device: target device for the CubeBatch tensors.

    Returns:
        CubeBatch with the requested fields populated.
    """
    device = torch.device(device)
    N = len(face_registers)
    cb = CubeBatch.empty(num_cubes=N, resolution=mesh.resolution, device=device)

    _all = include is None

    # --- cube_indices ---
    if _all or 'cube_indices' in include:
        indices = torch.tensor(
            [r['cube_indices'] for r in face_registers],
            dtype=torch.int32, device=device,
        )
        cb = cb.with_cube_indices(indices)

    # --- tri CSR (face_indices) ---
    if _all or 'face_indices' in include:
        per_cube = []
        for r in face_registers:
            fids = r.get('face_indices', [])
            per_cube.append(torch.tensor(fids, dtype=torch.int32, device=device))
        cb = cb.set_tri_csr(per_cube)

    # --- scalar integer fields ---
    if _all or 'num_components' in include:
        cb.num_components = torch.tensor(
            [r.get('num_components', 0) for r in face_registers],
            dtype=torch.int32, device=device,
        )

    if _all or 'num_boundary' in include:
        cb.num_boundary = torch.tensor(
            [r.get('num_boundary', 0) for r in face_registers],
            dtype=torch.int32, device=device,
        )

    if _all or 'status' in include:
        cb.status = torch.tensor(
            [r.get('status', 0) for r in face_registers],
            dtype=torch.int32, device=device,
        )

    # --- edge_weights ---
    if _all or 'edge_weights' in include:
        ew_list = [r.get('edge_weights', [0] * 18) for r in face_registers]
        cb.edge_weights = torch.tensor(ew_list, dtype=torch.int32, device=device)

    # --- face_weights ---
    if _all or 'face_weights' in include:
        fw_list = [r.get('face_weights', [0] * 12) for r in face_registers]
        cb.face_weights = torch.tensor(fw_list, dtype=torch.int32, device=device)

    # --- component_points (CSR) ---
    if _all or 'component_points' in include:
        offsets = [0]
        all_pts = []
        for r in face_registers:
            pts = r.get('component_points', [])
            offsets.append(offsets[-1] + len(pts))
            all_pts.extend(pts)
        cb.point_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        if all_pts:
            cb.point_values = torch.tensor(all_pts, dtype=torch.float32, device=device)
        else:
            cb.point_values = torch.zeros((0, 3), dtype=torch.float32, device=device)

    # --- loops (two-level CSR) ---
    if _all or 'loops' in include:
        loop_cube_off = [0]
        loop_edge_off = [0]
        loop_edge_val = []
        loop_edge_rank = []
        loop_point_match = []

        for r in face_registers:
            loops = r.get('sorted_loops', r.get('loops', []))
            loop_cube_off.append(loop_cube_off[-1] + len(loops))
            for loop_data in loops:
                if isinstance(loop_data, dict):
                    # sorted_loops format: {'loop': [...], 'rank': [...]}
                    edges = loop_data.get('loop', [])
                    ranks = loop_data.get('rank', [-1] * len(edges))
                else:
                    # plain loop format: [e1, e2, ...]
                    edges = list(loop_data) if not isinstance(loop_data, list) else loop_data
                    ranks = [-1] * len(edges)
                loop_edge_off.append(loop_edge_off[-1] + len(edges))
                loop_edge_val.extend(edges)
                loop_edge_rank.extend(ranks)
            # point matching
            match = r.get('loop_point_match', list(range(len(loops))))
            loop_point_match.extend(match)

        cb.loop_cube_off = torch.tensor(loop_cube_off, dtype=torch.int64, device=device)
        cb.loop_edge_off = torch.tensor(loop_edge_off, dtype=torch.int64, device=device)
        cb.loop_edge_val = torch.tensor(loop_edge_val, dtype=torch.int32, device=device)
        cb.loop_edge_rank = torch.tensor(loop_edge_rank, dtype=torch.int32, device=device)
        cb.loop_point_match = torch.tensor(loop_point_match, dtype=torch.int32, device=device)

    return cb
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_interop.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/interop/from_custom.py corep_fast/tests/unit/test_interop.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add interop/from_custom.py (list-of-dict → CubeBatch)

Converts custom/'s face_registers output format into a CubeBatch with
proper CSR offsets.  Supports selective field population via `include=`
argument, enabling partial A/B tests where only upstream stages have been
rewritten.  Handles both sorted_loops (dict w/ 'loop'+'rank') and plain
loop (list of edge ints) formats.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `corep_fast/interop/to_custom.py` — CubeBatch → list-of-dict

**Files:**
- Create: `corep_fast/interop/to_custom.py`
- Modify: `corep_fast/tests/unit/test_interop.py` (append roundtrip tests)

- [ ] **Step 1: Append roundtrip tests**

Append to `corep_fast/tests/unit/test_interop.py`:

```python
from corep_fast.interop.to_custom import custom_from_cube_batch


def test_roundtrip_from_custom_to_custom():
    """Convert list-of-dict → CubeBatch → list-of-dict and verify structure."""
    regs_in = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs_in, mt, device='cpu')
    regs_out = custom_from_cube_batch(cb, mt)

    assert len(regs_out) == 2
    assert regs_out[0]['cube_indices'] == (10, 20, 30)
    assert regs_out[1]['cube_indices'] == (11, 20, 30)
    assert regs_out[0]['face_indices'] == [0, 1, 2]
    assert regs_out[1]['face_indices'] == [3, 4]
    assert regs_out[0]['num_components'] == 1
    assert regs_out[1]['num_components'] == 2
    assert regs_out[0]['edge_weights'] == list(range(18))
    assert regs_out[1]['edge_weights'] == [1] * 18


def test_roundtrip_preserves_component_points():
    regs_in = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs_in, mt, device='cpu')
    regs_out = custom_from_cube_batch(cb, mt)

    # Cube 0: 1 point
    pts0 = regs_out[0]['component_points']
    assert len(pts0) == 1
    assert abs(pts0[0][0] - 0.5) < 1e-5

    # Cube 1: 2 points
    pts1 = regs_out[1]['component_points']
    assert len(pts1) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_interop.py -v -k "roundtrip"
```

Expected: ImportError (to_custom not yet written).

- [ ] **Step 3: Write to_custom.py**

Create `corep_fast/interop/to_custom.py`:

```python
"""
Convert a CubeBatch back into custom/'s list-of-dict format.

Used for:
1. Partial A/B runs where custom/ consumes a corep_fast stage output.
2. Dumping CubeBatch snapshots for offline analysis with existing custom/ tooling.

Reference: spec §14.D.
"""
from __future__ import annotations

import torch

from corep_fast.containers import CubeBatch, MeshTensors


def custom_from_cube_batch(
    cb: CubeBatch,
    mesh: MeshTensors,
) -> list[dict]:
    """
    Convert a CubeBatch into custom/'s list-of-dict format.

    Returns a list of dicts (one per cube) with keys matching custom/ conventions:
    'cube_indices', 'face_indices', 'num_components', 'num_boundary',
    'edge_weights', 'face_weights', 'component_points', 'status'.
    """
    result = []
    for i in range(cb.num_cubes):
        d: dict = {}

        # cube_indices — as a plain Python tuple
        d['cube_indices'] = tuple(cb.cube_indices[i].cpu().tolist())

        # face_indices — from tri CSR
        lo = int(cb.tri_offsets[i].item())
        hi = int(cb.tri_offsets[i + 1].item())
        d['face_indices'] = cb.tri_values[lo:hi].cpu().tolist()

        # scalar fields
        d['num_components'] = int(cb.num_components[i].item())
        d['num_boundary'] = int(cb.num_boundary[i].item())
        d['status'] = int(cb.status[i].item())

        # edge_weights — list of 18 ints
        d['edge_weights'] = cb.edge_weights[i].cpu().tolist()

        # face_weights — list of 12 ints
        d['face_weights'] = cb.face_weights[i].cpu().tolist()

        # component_points — from point CSR
        p_lo = int(cb.point_offsets[i].item())
        p_hi = int(cb.point_offsets[i + 1].item())
        d['component_points'] = cb.point_values[p_lo:p_hi].cpu().tolist()

        # loops — from two-level CSR
        l_lo = int(cb.loop_cube_off[i].item())
        l_hi = int(cb.loop_cube_off[i + 1].item())
        loops = []
        for l_idx in range(l_lo, l_hi):
            e_lo = int(cb.loop_edge_off[l_idx].item())
            e_hi = int(cb.loop_edge_off[l_idx + 1].item())
            edges = cb.loop_edge_val[e_lo:e_hi].cpu().tolist()
            ranks = cb.loop_edge_rank[e_lo:e_hi].cpu().tolist()
            loops.append({'loop': edges, 'rank': ranks})
        if loops:
            d['sorted_loops'] = loops

        result.append(d)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_interop.py -v
```

Expected: 10 tests pass (8 from_custom + 2 roundtrip).

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/interop/to_custom.py corep_fast/tests/unit/test_interop.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add interop/to_custom.py (CubeBatch → list-of-dict)

Inverse of from_custom.py: extracts CSR data back into custom/'s
flat list-of-dict format.  Roundtrip from_custom → to_custom preserves
cube_indices, face_indices, scalar fields, edge/face weights, and
component_points.  Used for partial A/B tests and debug snapshots.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `corep_fast/profiling/ab_rig.py` — Single-mesh A/B Runner

**Files:**
- Create: `corep_fast/profiling/ab_rig.py`
- Test: `corep_fast/tests/unit/test_ab_rig.py`

**Reference:** spec §7.3

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_ab_rig.py`:

```python
"""Unit tests for corep_fast/profiling/ab_rig.py."""
from unittest import mock

import pytest
import torch

from corep_fast.profiling.ab_rig import ABReport, ab_run_from_cube_batches


def _make_identical_pair():
    from corep_fast.containers import CubeBatch
    from corep_fast.profiling.harness import ProfilingCollector
    cb = CubeBatch.empty(num_cubes=2, resolution=64, device=torch.device('cpu'))
    pc = ProfilingCollector()
    return cb, cb, pc, pc


def test_ab_report_all_passed_on_identical():
    cb_a, cb_b, pc_a, pc_b = _make_identical_pair()
    report = ab_run_from_cube_batches(cb_a, cb_b, pc_a, pc_b)
    assert report.equivalence.all_passed()


def test_ab_report_speedup_calculation():
    from corep_fast.profiling.harness import ProfilingCollector, StageRecord
    pc_a = ProfilingCollector()
    pc_a._records['s1_voxelize'] = StageRecord(wall_time_s=10.0)
    pc_b = ProfilingCollector()
    pc_b._records['s1_voxelize'] = StageRecord(wall_time_s=1.0)

    from corep_fast.containers import CubeBatch
    cb = CubeBatch.empty(num_cubes=2, resolution=64, device=torch.device('cpu'))
    report = ab_run_from_cube_batches(cb, cb, pc_a, pc_b)
    assert report.speedups_per_stage['s1_voxelize'] == pytest.approx(10.0, rel=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_ab_rig.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write ab_rig.py**

Create `corep_fast/profiling/ab_rig.py`:

```python
"""
A/B comparison rig: run custom/ and corep_fast/ on the same mesh and
compare topology equivalence + speedup.

Reference: spec §7.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from corep_fast.containers import CubeBatch
from corep_fast.profiling.harness import ProfilingCollector
from corep_fast.profiling.topology_equivalence import (
    EquivalenceReport,
    check_topology_equivalence,
)


@dataclass
class ABReport:
    """Result of a single-mesh A/B comparison."""
    equivalence: EquivalenceReport
    custom_timings: ProfilingCollector
    fast_timings: ProfilingCollector
    speedups_per_stage: dict[str, float] = field(default_factory=dict)
    speedup_total: float = 1.0

    def summary_str(self) -> str:
        lines = [
            "A/B Report",
            f"  Equivalence: {'PASS' if self.equivalence.all_passed() else 'FAIL'}",
            f"  Total speedup: {self.speedup_total:.1f}×",
        ]
        for stage, sp in sorted(self.speedups_per_stage.items()):
            lines.append(f"    {stage}: {sp:.1f}×")
        return "\n".join(lines)


def ab_run_from_cube_batches(
    cb_custom: CubeBatch,
    cb_fast: CubeBatch,
    timings_custom: ProfilingCollector,
    timings_fast: ProfilingCollector,
    layers: list[str] | None = None,
) -> ABReport:
    """
    Compare two CubeBatches (one from custom/, one from corep_fast/) and two
    sets of profiling timings.  Returns an ABReport with equivalence + speedup.

    This is the "inner" function that does not run either pipeline — it only
    compares pre-computed results.  The "outer" orchestrator that actually
    calls both pipelines is baseline_runner.py (Task 13).
    """
    equiv = check_topology_equivalence(cb_custom, cb_fast, layers=layers)

    # Compute per-stage speedup
    speedups: dict[str, float] = {}
    for stage_name in timings_custom._records:
        t_custom = timings_custom._records[stage_name].wall_time_s
        t_fast = timings_fast._records.get(stage_name)
        if t_fast is not None and t_fast.wall_time_s > 0:
            speedups[stage_name] = t_custom / t_fast.wall_time_s
        elif t_custom > 0:
            speedups[stage_name] = float('inf')

    total_custom = timings_custom.total_wall_time_s
    total_fast = timings_fast.total_wall_time_s
    total_speedup = total_custom / total_fast if total_fast > 0 else float('inf')

    return ABReport(
        equivalence=equiv,
        custom_timings=timings_custom,
        fast_timings=timings_fast,
        speedups_per_stage=speedups,
        speedup_total=total_speedup,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_ab_rig.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/profiling/ab_rig.py corep_fast/tests/unit/test_ab_rig.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add A/B comparison rig (profiling/ab_rig.py)

Inner comparison function: takes two pre-computed CubeBatches + timing
collectors, runs topology equivalence check, and computes per-stage
speedup ratios.  The outer orchestrator that invokes both custom/ and
corep_fast/ pipelines will be added in baseline_runner.py (Task 13).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `corep_fast/profiling/baseline_runner.py` — Baseline Eval Set Driver

**Files:**
- Create: `corep_fast/profiling/baseline_runner.py`

**Reference:** spec §7.4

This module is a CLI driver that orchestrates running the `custom/` CoReP pipeline on a set of meshes, collecting per-stage timings via `stage_timer`, and saving the results as JSON.  It is the script that produces `baseline_custom_v0.json` in Task 16.

- [ ] **Step 1: Write the baseline_runner module**

Create `corep_fast/profiling/baseline_runner.py`:

```python
"""
Baseline runner: drive the custom/ CoReP pipeline on a set of meshes,
collect per-stage timings, and save results as JSON.

Usage:
    .venv/bin/python -m corep_fast.profiling.baseline_runner \
        --impl custom \
        --meshes results/baseline_experiments/data/*.ply \
        --resolution 512 \
        --output profiling/runs/baseline_custom_v0.json

Reference: spec §7.4.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh


def _run_custom_pipeline(mesh_path: str, resolution: int) -> dict:
    """
    Run custom/ pipeline on a single mesh and return timing + status dict.

    Imports custom/ modules dynamically, following the same sequence as
    custom/feature.py:  voxelize → feature_volume → feature_edge →
    feature_face → feature_point → collapse_face_{inner,boundary} →
    collapse_point_{inner,boundary} → reconstruct_mesh.
    """
    from corep_fast.profiling.harness import ProfilingCollector, stage_timer

    # Add custom/ to sys.path if needed
    project_root = str(Path(__file__).resolve().parents[2])
    custom_dir = os.path.join(project_root, 'custom')
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)

    # Dynamic imports to avoid polluting the module-level namespace
    from voxelize import voxelize
    from feature_volume import feature_volume
    from feature_edge import feature_edge
    from feature_face import feature_face
    from feature_point import feature_point
    from collapse_face import collapse_face_inner, collapse_face_boundary
    from collapse_point import collapse_point_inner, collapse_point_boundary
    from collapse import reconstruct_mesh, mark_exception
    from utils import fetch_np_array

    pc = ProfilingCollector(mesh_name=os.path.basename(mesh_path),
                            resolution=resolution, impl='custom')
    mesh = trimesh.load(mesh_path)

    import tempfile
    output_dir = tempfile.mkdtemp(prefix='corep_baseline_')

    try:
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

        with stage_timer('s8_collapse', pc):
            reconstruct_mesh(resolution, all_regs,
                             output_filepath=os.path.join(output_dir, 'out.ply'))

    finally:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    return {
        'mesh': os.path.basename(mesh_path),
        'resolution': resolution,
        'status': 'ok',
        'num_cubes': len(face_regs),
        'timings': pc,
    }


def run_baseline(
    mesh_paths: list[str],
    resolution: int,
    impl: str,
    output_path: str,
) -> None:
    """
    Run the specified impl on all meshes and save aggregated results.
    """
    results = []
    for mesh_path in mesh_paths:
        print(f"[baseline_runner] Processing {os.path.basename(mesh_path)} @ {resolution}...")
        t0 = time.time()
        try:
            if impl == 'custom':
                r = _run_custom_pipeline(mesh_path, resolution)
            else:
                raise NotImplementedError(f"impl={impl!r} not yet supported (Phase 1)")
            r['total_wall_s'] = time.time() - t0
            results.append(r)
            print(f"  → OK in {r['total_wall_s']:.1f}s, {r['num_cubes']} cubes")
        except Exception as e:
            tb = traceback.format_exc()
            results.append({
                'mesh': os.path.basename(mesh_path),
                'resolution': resolution,
                'status': 'error',
                'error': str(e),
                'traceback': tb,
            })
            print(f"  → ERROR: {e}")

    # Serialize
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != 'timings'}
        if 'timings' in r:
            pc: 'ProfilingCollector' = r['timings']
            entry['stages'] = {
                name: {'wall_time_s': rec.wall_time_s, 'peak_gpu_mem_bytes': rec.peak_gpu_mem_bytes}
                for name, rec in pc._records.items()
            }
            entry['total_pipeline_s'] = pc.total_wall_time_s
        serializable.append(entry)

    out.write_text(json.dumps(serializable, indent=2))
    print(f"\n[baseline_runner] Saved {len(serializable)} results to {out}")


def main():
    parser = argparse.ArgumentParser(description='CoReP baseline runner')
    parser.add_argument('--impl', required=True, choices=['custom', 'corep_fast', 'ab'])
    parser.add_argument('--meshes', nargs='+', required=True, help='Glob patterns or paths')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    # Expand globs
    all_paths = []
    for pattern in args.meshes:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"Warning: no files matched {pattern!r}")
        all_paths.extend(expanded)

    if not all_paths:
        parser.error("No mesh files found")

    run_baseline(all_paths, args.resolution, args.impl, args.output)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Quick smoke-test the module can at least parse**

Run:

```bash
.venv/bin/python -m corep_fast.profiling.baseline_runner --help
```

Expected: argparse usage message printed, no import errors.

- [ ] **Step 3: Commit**

```bash
git add -f corep_fast/profiling/baseline_runner.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add profiling/baseline_runner.py CLI driver

Orchestrates running the custom/ CoReP pipeline on a set of meshes with
stage_timer instrumentation, producing a JSON profile result per mesh.
This is the script that will generate baseline_custom_v0.json (Phase 0
Step 0) to establish the performance ground-truth before any Torch rewrites.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `corep_fast/profiling/report_builder.py` — JSON → Markdown Summary

**Files:**
- Create: `corep_fast/profiling/report_builder.py`
- Test: `corep_fast/tests/unit/test_report_builder.py`

- [ ] **Step 1: Write the failing test**

Create `corep_fast/tests/unit/test_report_builder.py`:

```python
"""Unit tests for corep_fast/profiling/report_builder.py."""
import json
from pathlib import Path

from corep_fast.profiling.report_builder import build_markdown_summary


def test_build_markdown_summary_basic(tmp_path: Path):
    data = [
        {
            'mesh': 'sphere.ply',
            'resolution': 256,
            'status': 'ok',
            'num_cubes': 100,
            'total_pipeline_s': 5.0,
            'stages': {
                's1_voxelize': {'wall_time_s': 2.0, 'peak_gpu_mem_bytes': 1000},
                's6_collapse_face': {'wall_time_s': 2.5, 'peak_gpu_mem_bytes': 2000},
            },
        },
    ]
    json_path = tmp_path / 'profile.json'
    json_path.write_text(json.dumps(data))
    md = build_markdown_summary(str(json_path))
    assert '# CoReP Profiling Summary' in md
    assert 'sphere.ply' in md
    assert 's1_voxelize' in md
    assert '2.00' in md or '2.0' in md
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_report_builder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write report_builder.py**

Create `corep_fast/profiling/report_builder.py`:

```python
"""
Build Markdown summary from profiling JSON output.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_markdown_summary(json_path: str) -> str:
    """Read a baseline_runner JSON and produce a Markdown summary string."""
    data = json.loads(Path(json_path).read_text())
    lines = ["# CoReP Profiling Summary", ""]

    # Header table
    lines.append("| Mesh | Resolution | Cubes | Total (s) | Status |")
    lines.append("|------|-----------|------:|----------:|--------|")
    for entry in data:
        mesh = entry.get('mesh', '?')
        res = entry.get('resolution', '?')
        cubes = entry.get('num_cubes', '?')
        total = entry.get('total_pipeline_s', 0)
        status = entry.get('status', '?')
        lines.append(f"| {mesh} | {res} | {cubes} | {total:.2f} | {status} |")

    # Per-stage breakdown
    lines.extend(["", "## Per-Stage Breakdown", ""])
    for entry in data:
        if entry.get('status') != 'ok':
            continue
        mesh = entry.get('mesh', '?')
        res = entry.get('resolution', '?')
        stages = entry.get('stages', {})
        total = entry.get('total_pipeline_s', 1)

        lines.append(f"### {mesh} @ {res}")
        lines.append("")
        lines.append("| Stage | Time (s) | % of Total | Peak GPU (MB) |")
        lines.append("|-------|--------:|-----------:|--------------:|")
        for sname, sdata in sorted(stages.items()):
            wall = sdata.get('wall_time_s', 0)
            pct = (wall / total * 100) if total > 0 else 0
            mem = sdata.get('peak_gpu_mem_bytes', 0) / (1024 * 1024)
            lines.append(f"| {sname} | {wall:.2f} | {pct:.1f}% | {mem:.1f} |")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_report_builder.py -v
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add -f corep_fast/profiling/report_builder.py corep_fast/tests/unit/test_report_builder.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add profiling/report_builder.py (JSON → Markdown)

Reads baseline_runner JSON output and produces a formatted Markdown
summary with per-mesh totals and per-stage time breakdown (wall time,
% of total, peak GPU memory).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Test Infrastructure — `conftest.py` + Fixtures

**Files:**
- Create: `corep_fast/tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

Create `corep_fast/tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Verify conftest loads**

Run:

```bash
.venv/bin/python -m pytest corep_fast/tests/ --collect-only 2>&1 | head -30
```

Expected: Test collection succeeds; fixtures are discovered.

- [ ] **Step 3: Commit**

```bash
git add -f corep_fast/tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(corep_fast): add tests/conftest.py with shared fixtures

Project path fixtures (project_root, custom_dir, test_mesh_dir) and
synthetic mesh fixtures (cube_mesh, icosphere_mesh, open_plane_mesh) with
pre-built MeshTensors variants for convenient unit test authoring.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Execute Step 0 — Run Baseline Profile on `custom/`

**Files:**
- Output: `profiling/runs/baseline_custom_v0.json`

**Reference:** spec §7.4 Step 0, §7.7

This is the **most important non-code task** in Phase 0.  It runs `custom/` end-to-end on the baseline evaluation set and produces the ground-truth performance numbers that will determine Phase 1 stage priority.

- [ ] **Step 1: Identify the baseline evaluation meshes**

Run:

```bash
ls results/baseline_experiments/data/*.ply 2>/dev/null || echo "No .ply found — check path"
```

If no `.ply` files exist, first run the model generator:

```bash
.venv/bin/python scripts/eval/baseline_generate_models.py --output-dir results/baseline_experiments/data
```

- [ ] **Step 2: Run baseline_runner on custom/ at resolution 256 (quick sanity)**

Run:

```bash
.venv/bin/python -m corep_fast.profiling.baseline_runner \
    --impl custom \
    --meshes results/baseline_experiments/data/*.ply \
    --resolution 256 \
    --output profiling/runs/baseline_custom_256_sanity.json
```

Expected: JSON output with per-mesh per-stage timings.  Look for `"status": "ok"` on all meshes.  If any fail, debug before proceeding.

- [ ] **Step 3: Run baseline_runner on custom/ at resolution 512**

Run:

```bash
.venv/bin/python -m corep_fast.profiling.baseline_runner \
    --impl custom \
    --meshes results/baseline_experiments/data/*.ply \
    --resolution 512 \
    --output profiling/runs/baseline_custom_512.json
```

This will take significantly longer.  Monitor progress in the terminal.

- [ ] **Step 4: Run baseline_runner on custom/ at resolution 1024**

Run:

```bash
.venv/bin/python -m corep_fast.profiling.baseline_runner \
    --impl custom \
    --meshes results/baseline_experiments/data/*.ply \
    --resolution 1024 \
    --output profiling/runs/baseline_custom_1024.json
```

This is the target resolution for Stage 1 performance claims.  It may take hours depending on mesh complexity.  Use the SSH/slurm approach from `run_baseline_116.sh` if running on a remote node.

- [ ] **Step 5: Build the Markdown summary**

Run:

```bash
.venv/bin/python -c "
from corep_fast.profiling.report_builder import build_markdown_summary
for res in [256, 512, 1024]:
    path = f'profiling/runs/baseline_custom_{res}.json'
    try:
        md = build_markdown_summary(path)
        out = f'profiling/runs/baseline_custom_{res}_summary.md'
        open(out, 'w').write(md)
        print(f'Wrote {out}')
    except FileNotFoundError:
        print(f'Skipping {path} (not found)')
"
```

- [ ] **Step 6: Copy the 512 result as the canonical v0 baseline**

```bash
cp profiling/runs/baseline_custom_512.json profiling/runs/baseline_custom_v0.json
```

(512 is the canonical resolution because custom/ at 1024 may be very slow; 512 gives usable data for prioritization while 1024 results are optional bonus data.)

- [ ] **Step 7: Review the summary and identify the top 3 hottest stages**

Read `profiling/runs/baseline_custom_512_summary.md` carefully.  Note:
- Which stage takes the most % of total time?
- Is it consistent across meshes, or does one mesh dominate?
- Is `s6_collapse_face` always the top stage, or is `s1_voxelize` or `s4_feature_face` sometimes bigger?

This data directly determines the Phase 1 implementation priority in Task 17.

---

## Task 17: Document Phase 1 Priority from Baseline Profile

**Files:**
- Create: `docs/superpowers/plans/phase1-priority-decision.md`

**Reference:** spec §7.7

- [ ] **Step 1: Write the priority document**

Based on Task 16 results, create `docs/superpowers/plans/phase1-priority-decision.md`:

```markdown
# Phase 1 Stage Rewriting Priority

> Decision date: YYYY-MM-DD
> Based on: profiling/runs/baseline_custom_v0.json (resolution 512)

## Stage Time Rankings (from baseline profile)

| Rank | Stage | Avg Wall Time (s) | % of Total | Decision |
|------|-------|-------------------:|-----------:|----------|
| 1 | s? | ?.? | ?.?% | Rewrite first |
| 2 | s? | ?.? | ?.?% | Rewrite second |
| ... | | | | |

## Final Phase 1 Task Order

1. `s?` — [reason from profile data]
2. `s?` — [reason]
3. ...

## Notes

- [Any surprises from the profile data]
- [Any meshes that are outliers]
```

Fill in the actual numbers from the baseline profile.

- [ ] **Step 2: Commit the priority document**

```bash
git add -f docs/superpowers/plans/phase1-priority-decision.md
git commit -m "$(cat <<'EOF'
docs: record Phase 1 stage priority from baseline profile

Stage rewriting order determined by actual profiling data from
baseline_custom_v0.json, not pre-guessed ordering.  This is the
input to the Phase 1 implementation plan.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Write Phase 1 plan (separate plan document)**

With the priority established, invoke `superpowers:writing-plans` again to produce the Phase 1 implementation plan.  The plan file should be saved to `docs/superpowers/plans/2026-XX-XX-corep-fast-phase1-stage-rewrites.md`, with task ordering matching the priority document above.

---

## Phase 0 Exit Criteria

All of the following must be true before Phase 0 is considered complete and Phase 1 can begin:

1. ✅ `corep_fast/` directory skeleton created with all subpackages
2. ✅ `torch_scatter` installed and importable
3. ✅ `constants.py` has all 18 edges, 12 facets, 8 vertices, share factors
4. ✅ `config.py` has Mode, StageConfig, BackendConfig
5. ✅ `containers.py` has MeshTensors (with from_trimesh) and CubeBatch (with CSR ops)
6. ✅ `profiling/harness.py` has stage_timer + ProfilingCollector
7. ✅ `profiling/topology_equivalence.py` has all 5 layers passing unit tests
8. ✅ `interop/from_custom.py` and `interop/to_custom.py` pass roundtrip tests
9. ✅ `profiling/ab_rig.py` has ABReport + comparison function
10. ✅ `profiling/baseline_runner.py` can run custom/ pipeline with timing
11. ✅ `profiling/report_builder.py` produces Markdown summary from JSON
12. ✅ All unit tests pass: `.venv/bin/python -m pytest corep_fast/tests/ -v`
13. ✅ `baseline_custom_v0.json` exists and contains valid per-stage timings
14. ✅ Phase 1 priority document written based on actual profile data
