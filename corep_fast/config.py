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
import os
from dataclasses import dataclass, replace
from typing import Literal


# ---------------------------------------------------------------------------
# M2 feature flags (set via env vars for easy toggle in profile runs)
# ---------------------------------------------------------------------------

# P1: Skip dict intermediate in s8 (direct-tensor path).
USE_DIRECT_TENSOR_S8 = os.environ.get('COREP_FAST_S8_DIRECT_TENSOR', '1') == '1'

# P2: GPU-accelerated face_weights in s4.
USE_GPU_FW_S4 = os.environ.get('COREP_FAST_S4_GPU_FW', '1') == '1'

# P3: Build grids directly from CubeBatch tensors in s8 (no cube_map dict).
USE_DIRECT_GRIDS_S8 = os.environ.get('COREP_FAST_S8_DIRECT_GRIDS', '1') == '1'


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
