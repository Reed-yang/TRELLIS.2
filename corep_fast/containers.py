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
