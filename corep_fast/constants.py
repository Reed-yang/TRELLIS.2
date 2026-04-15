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
# ---------------------------------------------------------------------------
EDGE_SHARE_FACTORS: torch.Tensor = torch.cat([
    torch.full((12,), 4, dtype=torch.int32),
    torch.full((6,),  2, dtype=torch.int32),
])
