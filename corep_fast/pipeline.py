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
