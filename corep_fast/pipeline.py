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


# ---------------------------------------------------------------------------
# OOM retry wrapper (2026-04-21 throughput + VRAM rescue track)
# ---------------------------------------------------------------------------

_S4_BUDGET_ENV = "COREP_FAST_S4_UTURN_CHUNK_ELEMS"
_S4_BUDGET_DEFAULT = 250_000_000


def _retry_with_shrinking_budget(stage_fn, batch, **kwargs):
    """Run `stage_fn(batch, **kwargs)`. On torch.cuda.OutOfMemoryError, shrink
    the s4 U-turn chunk budget 4x and retry up to twice. Third OOM propagates.

    Only active when corep_fast.config.VRAM_RESCUE is True; otherwise a direct
    passthrough (preserves legacy behaviour for non-rescue runs).
    """
    import gc as _gc
    import os as _os
    import torch as _torch
    from corep_fast.config import VRAM_RESCUE as _VRAM_RESCUE

    if not _VRAM_RESCUE:
        return stage_fn(batch, **kwargs)

    for attempt in range(3):
        try:
            return stage_fn(batch, **kwargs)
        except _torch.cuda.OutOfMemoryError:
            if attempt == 2:
                raise  # third strike -- driver will write .failed sentinel
            try:
                current = int(_os.environ.get(_S4_BUDGET_ENV, str(_S4_BUDGET_DEFAULT)))
            except ValueError:
                current = _S4_BUDGET_DEFAULT
            new = max(1, current // 4)
            _os.environ[_S4_BUDGET_ENV] = str(new)
            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()


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

    normalized_mesh = mesh.copy()
    normalized_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    normalized_mesh.remove_unreferenced_vertices()
    mask = normalized_mesh.unique_faces() & normalized_mesh.nondegenerate_faces()
    normalized_mesh.update_faces(mask)
    mesh = normalized_mesh

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


# ---------------------------------------------------------------------------
# Pure corep_fast encode / decode / pipeline API
# ---------------------------------------------------------------------------

def corep_encode(
    mesh_path: str,
    resolution: int,
    device: 'torch.device',
    collector: Optional[ProfilingCollector] = None,
    num_workers: int | None = None,
) -> 'CubeBatch':
    """Stages 1-7: mesh → CoReP voxel representation (all GPU tensors).

    Args:
        mesh_path: Path to input mesh file.
        resolution: Voxel grid resolution.
        device: PyTorch device for GPU tensors.
        collector: Optional profiling collector.
        num_workers: Worker count for multiprocessing stages (s4/s6/s7).
            None = auto (cpu_count // 2).

    Returns:
        CubeBatch with all stages populated.
    """
    import torch
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign

    pc = collector or ProfilingCollector()
    mesh = trimesh.load(mesh_path, force='mesh')

    normalized_mesh = mesh.copy()
    normalized_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    normalized_mesh.remove_unreferenced_vertices()
    mask = normalized_mesh.unique_faces() & normalized_mesh.nondegenerate_faces()
    normalized_mesh.update_faces(mask)
    mesh = normalized_mesh
    
    mt = MeshTensors.from_trimesh(mesh, resolution, device=device)

    # NOTE: no PersistentWorkerPool here. Each stage creates its own temp Pool
    # to allow fork-inherited shared data (COW) without double-fork overhead.
    with stage_timer('s1_voxelize', pc):
        batch = s1_voxelize(mt, resolution, device)
    with stage_timer('s2_components', pc):
        batch = s2_components(batch, mt)
    with stage_timer('s3_edge_weights', pc):
        batch = s3_edge_weights(batch, mt)
    with stage_timer('s4_face_point', pc):
        batch = _retry_with_shrinking_budget(
            s4_face_point, batch, mesh=mt, num_workers=num_workers)
    with stage_timer('s6_collapse', pc):
        batch = _retry_with_shrinking_budget(
            s6_collapse, batch, num_workers=num_workers)
    with stage_timer('s7_rank_assign', pc):
        batch = _retry_with_shrinking_budget(
            s7_rank_assign, batch, num_workers=num_workers)

    return batch


def corep_decode(
    batch: 'CubeBatch',
    merge_decimals: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 8: CubeBatch → (vertices [V,3] float32, faces [F,3] int32)."""
    import torch
    from corep_fast.stages.s8_collapse import decode_from_cubebatch
    return decode_from_cubebatch(batch, merge_decimals=merge_decimals)


@dataclass
class CorepParam:
    """Minimal representation of a CoReP-encoded mesh.

    Contains only the data needed to reconstruct the mesh via s6+s7+s8,
    with edge/face weights stored in compact (unique) form.
    """
    cube_indices: np.ndarray       # (N, 3)   int32
    edge_weights: np.ndarray       # (N, 6)   int32  — unique weights
    face_weights: np.ndarray       # (N, 6)   int32  — unique weights
    point_values: np.ndarray       # (P, 3)   float32
    point_offsets: np.ndarray      # (N+1,)   int64
    num_boundary: np.ndarray       # (N,)     int32
    resolution: int


def mesh_to_param(
    mesh_path: str,
    resolution: int,
    device: 'torch.device',
    collector: Optional[ProfilingCollector] = None,
    num_workers: int | None = None,
) -> CorepParam:
    """Encode a mesh into the minimal CoReP parameter representation.

    Runs s1-s4 (voxelize, components, edge weights, face/point weights)
    and extracts the compact representation. Does NOT run s6/s7 since
    those are re-derivable from the returned parameters.

    Args:
        mesh_path: Path to input mesh file (.ply, .obj, .glb, etc.).
        resolution: Voxel grid resolution.
        device: PyTorch device for GPU tensors.
        collector: Optional profiling collector.
        num_workers: Worker count for multiprocessing stages.

    Returns:
        CorepParam with all fields populated.
    """
    import torch
    from corep_fast.containers import MeshTensors
    from corep_fast.stages.s1_voxelize import s1_voxelize
    from corep_fast.stages.s2_components import s2_components
    from corep_fast.stages.s3_edge_weights import s3_edge_weights
    from corep_fast.stages.s4_face_point import s4_face_point

    pc = collector or ProfilingCollector()
    mesh = trimesh.load(mesh_path, force='mesh')

    normalized_mesh = mesh.copy()
    normalized_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    normalized_mesh.remove_unreferenced_vertices()
    mask = normalized_mesh.unique_faces() & normalized_mesh.nondegenerate_faces()
    normalized_mesh.update_faces(mask)
    mesh = normalized_mesh
    
    mt = MeshTensors.from_trimesh(mesh, resolution, device=device)

    with stage_timer('s1_voxelize', pc):
        batch = s1_voxelize(mt, resolution, device)
    with stage_timer('s2_components', pc):
        batch = s2_components(batch, mt)
    with stage_timer('s3_edge_weights', pc):
        batch = s3_edge_weights(batch, mt)
    with stage_timer('s4_face_point', pc):
        batch = _retry_with_shrinking_budget(
            s4_face_point, batch, mesh=mt, num_workers=num_workers)

    # Extract and compact the representation
    ew_full = batch.edge_weights.cpu().numpy()   # (N, 18) int32
    fw_full = batch.face_weights.cpu().numpy()   # (N, 12) int32

    # Unique edge indices: 0, 3, 8, 12, 14, 17
    ew_unique = ew_full[:, [0, 3, 8, 12, 14, 17]]
    # Unique face indices: 0, 1, 4, 5, 10, 11
    fw_unique = fw_full[:, [0, 1, 4, 5, 10, 11]]

    return CorepParam(
        cube_indices=batch.cube_indices.cpu().numpy(),
        edge_weights=ew_unique,
        face_weights=fw_unique,
        point_values=batch.point_values.cpu().numpy(),
        point_offsets=batch.point_offsets.cpu().numpy(),
        num_boundary=batch.num_boundary.cpu().numpy(),
        resolution=resolution,
    )


def _unique_to_full_edge_weights(unique: np.ndarray, cube_indices: np.ndarray, res: int) -> np.ndarray:
    """Reconstruct (N, 18) full edge weights from (N, 6) unique weights."""
    x, y, z = cube_indices[:, 0], cube_indices[:, 1], cube_indices[:, 2]
    U = np.zeros((res, res, res, 6), dtype=unique.dtype)
    U[x, y, z] = unique

    E_X, E_Y, E_Z = U[..., 0], U[..., 1], U[..., 2]
    D_XY, D_XZ, D_YZ = U[..., 3], U[..., 4], U[..., 5]

    def shift(grid, dx, dy, dz):
        # Return s where s[x,y,z] = grid[x+dx, y+dy, z+dz], zero-padded at
        # out-of-bounds positions. Previous implementation copied the original
        # value at the boundary, which wrongly leaked the current cube's
        # weights into non-existent neighbors.
        s = np.zeros_like(grid)
        sx, sy, sz = grid.shape[:3]
        s[:sx - dx, :sy - dy, :sz - dz] = grid[dx:, dy:, dz:]
        return s

    F = np.zeros((res, res, res, 18), dtype=unique.dtype)
    F[..., 0] = E_X
    F[..., 1] = shift(E_Y, 1, 0, 0)
    F[..., 2] = shift(E_X, 0, 1, 0)
    F[..., 3] = E_Y
    F[..., 4] = shift(E_X, 0, 0, 1)
    F[..., 5] = shift(E_Y, 1, 0, 1)
    F[..., 6] = shift(E_X, 0, 1, 1)
    F[..., 7] = shift(E_Y, 0, 0, 1)
    F[..., 8] = E_Z
    F[..., 9] = shift(E_Z, 1, 0, 0)
    F[..., 10] = shift(E_Z, 1, 1, 0)
    F[..., 11] = shift(E_Z, 0, 1, 0)
    F[..., 12] = D_XY
    F[..., 13] = shift(D_XY, 0, 0, 1)
    F[..., 14] = D_XZ
    F[..., 15] = shift(D_YZ, 1, 0, 0)
    F[..., 16] = shift(D_XZ, 0, 1, 0)
    F[..., 17] = D_YZ
    return F[x, y, z]


def _unique_to_full_face_weights(unique: np.ndarray, cube_indices: np.ndarray, res: int) -> np.ndarray:
    """Reconstruct (N, 12) full face weights from (N, 6) unique weights."""
    x, y, z = cube_indices[:, 0], cube_indices[:, 1], cube_indices[:, 2]
    U = np.zeros((res, res, res, 6), dtype=unique.dtype)
    U[x, y, z] = unique

    T_Bot1, T_Bot2 = U[..., 0], U[..., 1]
    T_Frt1, T_Frt2 = U[..., 2], U[..., 3]
    T_Lft1, T_Lft2 = U[..., 4], U[..., 5]

    def shift(grid, dx, dy, dz):
        # Return s where s[x,y,z] = grid[x+dx, y+dy, z+dz], zero-padded at
        # out-of-bounds positions. Previous implementation copied the original
        # value at the boundary, which wrongly leaked the current cube's
        # weights into non-existent neighbors.
        s = np.zeros_like(grid)
        sx, sy, sz = grid.shape[:3]
        s[:sx - dx, :sy - dy, :sz - dz] = grid[dx:, dy:, dz:]
        return s

    F = np.zeros((res, res, res, 12), dtype=unique.dtype)
    F[..., 0] = T_Bot1
    F[..., 1] = T_Bot2
    F[..., 2] = shift(T_Bot1, 0, 0, 1)
    F[..., 3] = shift(T_Bot2, 0, 0, 1)
    F[..., 4] = T_Frt1
    F[..., 5] = T_Frt2
    F[..., 6] = shift(T_Lft1, 1, 0, 0)
    F[..., 7] = shift(T_Lft2, 1, 0, 0)
    F[..., 8] = shift(T_Frt1, 0, 1, 0)
    F[..., 9] = shift(T_Frt2, 0, 1, 0)
    F[..., 10] = T_Lft1
    F[..., 11] = T_Lft2
    return F[x, y, z]


def param_to_mesh(
    param: CorepParam,
    device: 'torch.device',
    merge_decimals: int = 5,
    collector: Optional[ProfilingCollector] = None,
    num_workers: int | None = None,
) -> tuple:
    """Reconstruct a mesh from the minimal CoReP parameter representation.

    Rebuilds a CubeBatch from the compact parameters, then re-runs
    s6 (face collapse) → s7 (rank assign) → s8 (decode) to produce
    the output mesh.

    Args:
        param: CorepParam from mesh_to_param.
        device: PyTorch device.
        merge_decimals: Vertex welding precision.
        collector: Optional profiling collector.
        num_workers: Worker count for multiprocessing stages.

    Returns:
        (vertices, faces) tuple:
            vertices: (V, 3) float32
            faces: (F, 3) int32
    """
    import torch
    from corep_fast.containers import CubeBatch
    from corep_fast.stages.s6_collapse import s6_collapse
    from corep_fast.stages.s7_rank_assign import s7_rank_assign

    pc = collector or ProfilingCollector()
    N = param.cube_indices.shape[0]
    res = param.resolution

    # Expand unique weights back to full weights
    ew_full = _unique_to_full_edge_weights(param.edge_weights, param.cube_indices, res)
    fw_full = _unique_to_full_face_weights(param.face_weights, param.cube_indices, res)

    # Build cube_hash
    ci = param.cube_indices.astype(np.int64)
    cube_hash = ci[:, 0] * res * res + ci[:, 1] * res + ci[:, 2]

    zeros_i32 = lambda shape: torch.zeros(shape, dtype=torch.int32, device=device)
    zeros_i64 = lambda shape: torch.zeros(shape, dtype=torch.int64, device=device)

    batch = CubeBatch(
        cube_indices=torch.from_numpy(param.cube_indices).to(device=device, dtype=torch.int32),
        cube_hash=torch.from_numpy(cube_hash).to(device=device, dtype=torch.int64),

        # S1 CSR registries — not needed by s6/s7/s8, fill with empty
        tri_offsets=zeros_i64((N + 1,)),
        tri_values=zeros_i32((0,)),
        bnd_offsets=zeros_i64((N + 1,)),
        bnd_values=zeros_i32((0,)),
        nm_offsets=zeros_i64((N + 1,)),
        nm_values=zeros_i32((0,)),

        num_components=zeros_i32((N,)),
        num_boundary=torch.from_numpy(param.num_boundary).to(device=device, dtype=torch.int32),

        edge_weights=torch.from_numpy(ew_full).to(device=device, dtype=torch.int32),
        face_weights=torch.from_numpy(fw_full).to(device=device, dtype=torch.int32),

        point_offsets=torch.from_numpy(param.point_offsets).to(device=device, dtype=torch.int64),
        point_values=torch.from_numpy(param.point_values).to(device=device, dtype=torch.float32),

        # S5-S7 outputs — will be populated by s6_collapse and s7_rank_assign
        loop_cube_off=zeros_i64((N + 1,)),
        loop_edge_off=zeros_i64((1,)),
        loop_edge_val=zeros_i32((0,)),
        loop_edge_rank=torch.full((0,), -1, dtype=torch.int32, device=device),
        loop_point_match=torch.zeros((0,), dtype=torch.int32, device=device),

        status=zeros_i32((N,)),

        comp_face_off=zeros_i64((N + 1,)),
        comp_face_val=zeros_i32((0,)),
        uturn_assignment=torch.full((N, 12, 3), -1, dtype=torch.int32, device=device),

        device=device,
        resolution=res,
    )

    with stage_timer('s6_collapse', pc):
        batch = s6_collapse(batch, num_workers=num_workers)
    with stage_timer('s7_rank_assign', pc):
        batch = s7_rank_assign(batch, num_workers=num_workers)
    with stage_timer('s8_decode', pc):
        vertices, faces = corep_decode(batch, merge_decimals=merge_decimals)

    return vertices, faces


def corep_pipeline(
    mesh_path: str,
    resolution: int,
    device: 'torch.device',
    output_path: str | None = None,
    merge_decimals: int = 5,
    collector: Optional[ProfilingCollector] = None,
    num_workers: int | None = None,
) -> tuple:
    """End-to-end: mesh → (CubeBatch, vertices, faces). Optionally write PLY."""
    import torch
    pc = collector or ProfilingCollector()
    batch = corep_encode(mesh_path, resolution, device, collector=pc,
                         num_workers=num_workers)

    with stage_timer('s8_decode', pc):
        vertices, faces = corep_decode(batch, merge_decimals=merge_decimals)

    if output_path is not None:
        from corep_fast.stages.s8_collapse import _write_ply_ascii
        _write_ply_ascii(vertices, faces, output_path)

    return batch, vertices, faces
