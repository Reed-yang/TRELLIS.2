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
    adj = np.full((num_faces, 3), -1, dtype=np.int32)
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int32)   # (E, 2)
    edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int32)  # (E, 2)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    for (fa, fb), (va, vb) in zip(pairs, edges):
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
    faces = np.asarray(mesh.faces, dtype=np.int32)
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

    nm_verts = np.zeros((0, 3), dtype=np.float32)

    return (
        torch.from_numpy(nm_arr).to(device=device, dtype=torch.float32),
        torch.from_numpy(nm_verts).to(device=device, dtype=torch.float32),
    )


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

    # s2 output — component-face CSR (populated by s2_components)
    comp_face_off:   torch.Tensor  # (N+1,) int64 — CSR offsets for comp_face_val
    comp_face_val:   torch.Tensor  # (tot_CF,) int32 — flat face ids grouped by (cube, component)

    # s6 output — U-turn assignment (populated by s6_collapse)
    uturn_assignment: torch.Tensor  # (N, 12, 3) int32 — per-facet (u1,u2,u3); -1 for fast-path

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

            comp_face_off=zeros_i64((N + 1,)),
            comp_face_val=zeros_i32((0,)),
            uturn_assignment=torch.full((N, 12, 3), -1, dtype=torch.int32, device=device),

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
        """Populate tri_offsets + tri_values from a Python list of per-cube tensors."""
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

        # comp_face CSR
        assert self.comp_face_off.shape == (N + 1,), f"[{stage}] comp_face_off shape"
        assert self.comp_face_off.dtype == torch.int64, f"[{stage}] comp_face_off dtype"
        diffs = self.comp_face_off[1:] - self.comp_face_off[:-1]
        assert (diffs >= 0).all(), f"[{stage}] comp_face_off is non-monotone"
        assert self.comp_face_off[0].item() == 0, f"[{stage}] comp_face_off[0] must be 0"
        assert self.comp_face_val.shape[0] == int(self.comp_face_off[-1].item()), \
            f"[{stage}] comp_face_val size mismatch"

        # uturn_assignment
        assert self.uturn_assignment.shape == (N, 12, 3), f"[{stage}] uturn_assignment shape"
        assert self.uturn_assignment.dtype == torch.int32, f"[{stage}] uturn_assignment dtype"


# ---------------------------------------------------------------------------
# Internal helper: replace a subset of dataclass fields immutably
# ---------------------------------------------------------------------------

def _replace_fields(cb: CubeBatch, **updates) -> CubeBatch:
    from dataclasses import replace as _dc_replace
    return _dc_replace(cb, **updates)
