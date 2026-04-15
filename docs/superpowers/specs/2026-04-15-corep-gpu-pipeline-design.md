# CoReP GPU Pipeline — Full End-to-End Rewrite Design

> **Date:** 2026-04-15
> **Branch:** `gpu-pipeline` (from `siyuan-dev` + triton-s8 cherry-picks)
> **Supersedes:** corep-fast-stage1-design, corep-fast-stage2-* (those focused on s8 only)

---

## 1. Motivation

Profiling of the full CoReP pipeline (icosphere subdiv=3, res=256) reveals:

| Component | Time | % of e2e |
|-----------|------|----------|
| s1-s7 (custom/ Python) | 77.2s | **95.0%** |
| s8 (Torch vectorized) | 4.0s | 5.0% |
| **Total** | **81.2s** | 100% |

All prior optimization work (Torch vectorization, Triton analysis, CuMesh evaluation) targeted s8, which is only 5% of the pipeline. The true bottleneck is s1-s7's Python implementation: per-cube Python loops over 275K cubes with dict-based data passing.

**Goal:** Rewrite the entire s1-s7 pipeline as GPU tensor operations, targeting **< 3s end-to-end** (PyTorch path) and **< 1s** (Triton path), down from 81.2s.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Pipeline interface** | `corep_encode()` → CubeBatch, `corep_decode()` → mesh | CoReP voxel representation (CubeBatch) is a first-class output for VAE training |
| **Verification** | Per-stage A/B against custom/ | Each GPU stage must match custom/ output exactly (integers) or within tolerance (floats) |
| **Tech route** | PyTorch first, Triton replace hot paths later | PyTorch version is iteration-friendly; Triton is performance-optimized |
| **Dual path** | `corep_fast/` (PyTorch) + `corep_triton/` (Triton) | Both long-term coexist; iteration happens on PyTorch, Triton tracks after |
| **Stage grouping** | Phase-grouped (4 phases) | Merge s4a+s4b (shared clipping), merge s5+s6 (normal curve + U-Turn) |
| **custom/ handling** | Unmodified, GT baseline only | Never modify; used exclusively for A/B correctness verification |

---

## 3. Architecture

### 3.1 Pipeline API

```python
def corep_encode(
    mesh_path: str, resolution: int, device: torch.device,
    collector: ProfilingCollector | None = None,
) -> CubeBatch:
    """Stages 1-7: mesh → CoReP voxel representation (all GPU tensors)."""

def corep_decode(
    batch: CubeBatch, merge_decimals: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 8: CubeBatch → (vertices [V,3] float32, faces [F,3] int32)."""

def corep_pipeline(
    mesh_path: str, resolution: int, device: torch.device,
    output_path: str | None = None,
    merge_decimals: int = 5,
) -> tuple[CubeBatch, torch.Tensor, torch.Tensor]:
    """End-to-end: mesh → (CubeBatch, vertices, faces). Optionally write PLY."""
```

### 3.2 Stage Functions

Each stage has a uniform signature: takes `CubeBatch` (+ `MeshTensors` where needed), returns updated `CubeBatch`. All tensors stay on GPU throughout.

```python
def s1_voxelize(mesh: MeshTensors, resolution: int, device: torch.device) -> CubeBatch
def s2_components(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch
def s3_edge_weights(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch
def s4_face_point(batch: CubeBatch, mesh: MeshTensors) -> CubeBatch
def s6_collapse(batch: CubeBatch) -> CubeBatch
def s7_rank_assign(batch: CubeBatch) -> CubeBatch
# s8_decode is corep_decode() above
```

### 3.3 Phase Grouping

```
Phase 1: Geometry Registration
  s1_voxelize    — mesh triangles → occupied cubes + face CSR registration
  s2_components  — face adjacency DFS → num_components, num_boundary

Phase 2: Feature Extraction
  s3_edge_weights — Möller-Trumbore ray-triangle → edge_weights (N, 18)
  s4_face_point   — Sutherland-Hodgman clipping → face_weights (N, 12) + component_points
                    (merges custom/ s4a + s4b; clipping done once, both outputs extracted)

Phase 3: Topology Solve
  s6_collapse     — normal curve arc assignment + U-Turn enumeration → loops
                    (merges custom/ s5 + s6)
  s7_rank_assign  — deterministic rank assignment + Hungarian point matching → sorted_loops

Phase 4: Mesh Decode
  s8_decode       — existing Torch vectorized path (edge geometry + welding)
```

### 3.4 Data Flow

```
Input: mesh.ply + resolution
  │
  ▼
MeshTensors.from_trimesh()          # vertices, faces, triangles, face_adj on GPU
  │
  ▼
s1_voxelize ─────────────────────── CubeBatch created:
  │                                   cube_indices (N,3), cube_hash (N,)
  │                                   tri_offsets (N+1,), tri_values (T,)
  ▼
s2_components ───────────────────── CubeBatch updated:
  │                                   num_components (N,), num_boundary (N,)
  ▼
s3_edge_weights ─────────────────── CubeBatch updated:
  │                                   edge_weights (N,18)
  ▼
s4_face_point ───────────────────── CubeBatch updated:
  │                                   face_weights (N,12)
  │                                   point_offsets (N+1,), point_values (P,3)
  ▼
s6_collapse ─────────────────────── CubeBatch updated:
  │                                   loop_cube_off (N+1,), loop_edge_off (L+1,)
  │                                   loop_edge_val (E,), status (N,)
  ▼
s7_rank_assign ──────────────────── CubeBatch updated:
  │                                   loop_edge_rank (E,), loop_point_match (L,)
  ▼
s8_decode ───────────────────────── Output:
                                      vertices (V,3) float32, faces (F,3) int32
```

---

## 4. Stage Algorithms

### 4.1 s1_voxelize — SAT Triangle-AABB Test

**Input:** MeshTensors (triangles `(F,3,3)` on GPU), resolution
**Output:** CubeBatch with `cube_indices`, `tri_offsets`, `tri_values`

Algorithm:
1. Per triangle: compute AABB → enumerate candidate cubes within AABB
2. GPU parallel SAT (Separating Axis Theorem) with 13 axes:
   - 1 triangle normal, 3 cube face normals, 9 edge×edge cross products
   - Each (triangle, candidate_cube) pair tested independently
3. Compact hits → CSR format: `tri_offsets[cube_i]` / `tri_values[flat]`
4. `cube_indices` = unique occupied cube coordinates via `torch.unique`

Implementation notes:
- Variable candidate count per triangle → "expand then compact" pattern:
  prefix-sum candidate counts → allocate flat buffer → parallel fill → SAT test → compact hits
- Output CSR: sort (cube_hash, face_idx) pairs → `torch.unique_consecutive` → offsets

### 4.2 s2_components — GPU Union-Find

**Input:** CubeBatch (`tri_offsets`, `tri_values`), MeshTensors (`face_adj`)
**Output:** `num_components` `(N,)`, `num_boundary` `(N,)`

Algorithm:
1. Pre-compute global face adjacency tensor from mesh: `face_adj (F, 3)` int32
2. Per cube: initialize label[i] = i for each registered face
3. For each pair of faces registered to same cube: if mesh-adjacent → union labels
4. Count distinct labels per cube → `num_components`
5. Boundary detection: face edges with `face_adj == -1` → boundary component count

Implementation notes:
- Per-cube face count is small (typically < 20) → pad to fixed max, use masked operations
- Union-Find as iterative label propagation: `label[i] = min(label[i], label[adj[i]])` repeated until convergence
- Convergence guaranteed in O(diameter) iterations; for small subgraphs, 3-5 iterations suffice

### 4.3 s3_edge_weights — Möller-Trumbore Ray-Triangle Intersection

**Input:** CubeBatch (`cube_indices`, `tri_offsets`, `tri_values`), MeshTensors (`triangles`)
**Output:** `edge_weights (N, 18)` int32

Algorithm:
1. Construct ray batch: N cubes × 18 edges → ray origins `(N,18,3)` + directions `(18,3)`
   - Edge endpoints derived from `cube_indices / resolution + CUBE_VERTICES[CUBE_EDGES]`
2. For each (cube, edge): iterate registered triangles via CSR, count intersections
   - Möller-Trumbore test: compute determinant, barycentric coords (u,v), parameter t
   - Hit condition: `|det| > ε`, `u ∈ [0,1]`, `v ∈ [0,1]`, `u+v ≤ 1`, `t ∈ [-ε, 1+ε]`
3. `edge_weights[cube, edge]` = hit count

Implementation notes:
- CSR iteration strategy: expand to `(total_registered_tris × 18)` flat pairs → batch Möller-Trumbore → `scatter_add` by (cube, edge) back to `(N, 18)`
- This avoids padding and handles variable tri-per-cube naturally
- Memory: for res=256, ~275K cubes × avg ~5 tris = ~1.4M pairs × 18 edges = ~25M tests, each ~30 FLOPs → trivial on H100

### 4.4 s4_face_point — Clipping + Face Weights + Component Points

**Input:** CubeBatch (`cube_indices`, `tri_offsets`, `tri_values`, `num_components`), MeshTensors
**Output:** `face_weights (N, 12)`, `point_offsets (N+1,)`, `point_values (P, 3)`

Algorithm:
1. Sutherland-Hodgman clipping: for each (cube, registered_triangle), clip triangle to cube AABB
   - 6 clip planes (xmin, xmax, ymin, ymax, zmin, zmax) applied sequentially
   - Input: triangle (3 vertices) → Output: clipped polygon (3-7 vertices)
   - Pre-allocate `(T_total, 9, 3)` buffer with `valid_count (T_total,)` tracking actual vertex count
2. Face weights extraction:
   - For each of 12 cube facets: count U-Turn events from clipped polygon edge patterns
   - U-Turn = a mesh surface segment that enters and exits through the same cube edge
3. Component points:
   - Fan-triangulate each clipped polygon from vertex 0
   - Area-weighted centroid per triangle: `area = 0.5 * ||cross(e1, e2)||`, `centroid = (v0+v1+v2)/3`
   - Group by component (from s2 labels) → weighted average per component
   - Pack into CSR: `point_offsets[cube]`, `point_values[flat]`

Implementation notes:
- Clipping produces variable-length output → fixed 9-vertex buffer per input triangle (Sutherland-Hodgman on a convex 6-plane clip region with a triangle input produces at most 9 vertices). Use `valid_count` tensor to track actual vertex count per clipped polygon.
- Merging s4a+s4b: clipping is done once, face_weights and centroids both extracted from same clipped data
- This eliminates the ~60% redundant clipping work that exists in custom/ (s4a and s4b clip independently)

### 4.5 s6_collapse — Normal Curve Theory + U-Turn Enumeration

**Input:** CubeBatch (`edge_weights`, `face_weights`)
**Output:** `loop_cube_off`, `loop_edge_off`, `loop_edge_val`, `status`

Algorithm:
1. Arc assignment (per cube, per facet): for facet with edges (w1, w2, w3):
   - `k12 = (w1+w2-w3)/2`, `k23 = (w2+w3-w1)/2`, `k31 = (w3+w1-w2)/2`
   - Validate: triangle inequality, parity, non-negative
   - Invalid → `status = UNSOLVABLE`
2. U-Turn enumeration (cubes with any face_weight > 0):
   - For each facet: enumerate `(u1, u2, u3)` with `u_i ∈ [0, face_weight[facet_edge_i]]`
   - Adjusted weights: `w'_i = w_i - 2*u_i`
   - Validate adjusted weights → collect valid local assignments
   - Global Cartesian product over 12 facets → loop configurations
   - 0 solutions → UNSOLVABLE, 1 → OK, ≥2 → AMBIGUOUS
3. Arc graph + loop tracing:
   - Nodes: `(edge_idx, point_idx)` for each edge crossing
   - Edges: arcs from k-assignments + U-Turn intra-edge arcs
   - Every node has degree 2 → DFS yields disjoint closed loops
4. Pack loops into two-level CSR

Implementation notes:
- **Fast path (no U-Turns)**: majority of cubes have `face_weights == 0` → skip enumeration, direct arc assignment + loop trace. Fully batchable on GPU.
- **Slow path (with U-Turns)**: small minority of cubes. Process separately, potentially with less parallelism but bounded combinatorial work per cube.
- Loop tracing on GPU: each cube's arc graph fits in a fixed-size array (max 18 edges × max weight per edge, bounded). Use per-thread local arrays.
- Inner vs boundary split: `num_boundary == 0` mask → inner cubes use standard enumeration, boundary cubes use extended version with open-path arcs.

### 4.6 s7_rank_assign — Ranking + Hungarian Matching

**Input:** CubeBatch (loops from s6, `edge_weights`, `point_values`)
**Output:** `loop_edge_rank`, `loop_point_match`

Algorithm:
1. Rank assignment (deterministic):
   - For each edge crossed by multiple loops: assign rank [0, W-1] based on non-crossing pairing rule
   - "Nearest to corner vertex" ordering at each cube corner → unique rank per crossing
   - This is a derivation from the arc graph, not an optimization problem
2. Point-to-loop matching:
   - Compute loop centroids: for each loop, interpolate edge crossings to 3D positions, average
     - Position on edge: `p = edge_start + (rank+1)/(weight+1) * (edge_end - edge_start)`
   - For N ≤ 4 loops per cube: exhaustive permutation matching
     - Pre-compute all 24 permutations of [0,1,2,3]
     - Distance sum per permutation: `sum(||loop_centroid[i] - point[perm[i]]||²)`
     - Best permutation = argmin → `loop_point_match`
   - GPU batch: `(N_cubes, 24)` distance sums → `argmin(dim=1)`

Implementation notes:
- Rank assignment is the same algorithm as loop tracing (s6) but tracking rank indices. Can be fused with s6's loop trace to avoid redundant graph traversal.
- Permutation matching for N < 4: mask unused slots in the 4-permutation table.

---

## 5. Package Structure

```
corep_fast/                              # PyTorch implementation (primary)
├── pipeline.py                          # corep_encode / corep_decode / corep_pipeline
│                                        # + legacy run_hybrid_pipeline (backward compat)
├── containers.py                        # CubeBatch, MeshTensors (existing, extended)
├── constants.py                         # cube topology constants (existing, extended)
├── stages/
│   ├── __init__.py
│   ├── s1_voxelize.py                   # NEW: SAT voxelization
│   ├── s2_components.py                 # NEW: GPU Union-Find
│   ├── s3_edge_weights.py               # NEW: Möller-Trumbore
│   ├── s4_face_point.py                 # NEW: clipping + face_weights + points
│   ├── s6_collapse.py                   # NEW: normal curve + U-Turn
│   ├── s7_rank_assign.py                # NEW: ranking + Hungarian
│   └── s8_decode.py                     # RENAMED from s8_collapse.py, trimmed
├── interop/
│   ├── from_custom.py                   # existing + per-stage conversion extensions
│   └── to_custom.py                     # existing
├── profiling/
│   ├── harness.py                       # existing stage_timer
│   ├── ab_rig.py                        # existing + per-stage comparison extensions
│   ├── topology_equivalence.py          # existing
│   └── baseline_runner.py               # existing
└── tests/
    ├── unit/                            # per-stage unit tests
    │   ├── test_s1_voxelize.py
    │   ├── test_s2_components.py
    │   ├── test_s3_edge_weights.py
    │   ├── test_s4_face_point.py
    │   ├── test_s6_collapse.py
    │   ├── test_s7_rank_assign.py
    │   └── ...existing tests...
    └── regression/
        ├── test_s1_ab.py                # A/B: GPU s1 vs custom/ voxelize
        ├── test_s2_ab.py                # A/B: GPU s2 vs custom/ feature_volume
        ├── test_s3_ab.py                # A/B: GPU s3 vs custom/ feature_edge
        ├── test_s4_ab.py                # A/B: GPU s4 vs custom/ feature_face+point
        ├── test_s6_ab.py                # A/B: GPU s6 vs custom/ collapse_face
        ├── test_s7_ab.py                # A/B: GPU s7 vs custom/ collapse_point
        └── test_e2e_ab.py               # existing end-to-end mesh comparison

corep_triton/                            # Triton implementation (future)
├── __init__.py
├── kernels/                             # Triton kernel source files
│   ├── moller_trumbore.py               # s3 hot path
│   ├── polygon_clip.py                  # s4 hot path
│   └── normal_curve.py                  # s6 hot path
└── stages/                              # Same signatures as corep_fast/stages/
    └── (populated later when profiling identifies hot paths)
```

---

## 6. A/B Verification Framework

### 6.1 Verification Layers

| Layer | Compared Fields | Tolerance | When |
|-------|----------------|-----------|------|
| **L0 Shape** | All tensor shapes | exact | Every stage |
| **L1 Integer** | edge_weights, face_weights, num_components, num_boundary, loop edges, ranks, status | exact | Every stage |
| **L2 Float** | component_points, loop centroids | atol=1e-5 | s4, s7 |
| **L3 Topology** | Loop set equivalence (cyclic rotation + reflection invariant) | canonical form | s6, s7 |
| **L4 End-to-end** | Output mesh V count, F count, vertex coordinates | V/F exact, coords atol=1e-4 | e2e |

### 6.2 Test Infrastructure

```python
# Helper: run custom/ pipeline to a specific stage
def run_custom_through_stage(mesh_path: str, resolution: int, up_to: str) -> list[dict]:
    """Run custom/ stages sequentially, return registers after 'up_to' stage.
    
    up_to: 's1' | 's2' | 's3' | 's4a' | 's4b' | 's6' | 's7'
    """

# Helper: compare GPU CubeBatch vs custom/ list[dict]
def assert_stage_match(gpu_batch: CubeBatch, custom_regs: list[dict], stage: str):
    """Dispatch to appropriate L0-L3 checks based on stage."""
```

### 6.3 Test Fixtures

- **icosphere subdiv=2, res=64**: fast smoke test (~1s custom/)
- **icosphere subdiv=3, res=128**: standard regression (covers all topology cases)
- **icosphere subdiv=3, res=256**: full-scale validation (275K cubes, covers edge cases)

---

## 7. Performance Targets

| Milestone | s1-s7 Time | s8 Time | e2e | vs Current |
|-----------|-----------|---------|-----|-----------|
| Current (custom/ + s8 Torch) | 77.2s | 4.0s | 81.2s | 1.0x |
| **Phase 1: PyTorch s1-s7** | **~2-5s** | **~1s** | **~3-6s** | **14-27x** |
| Phase 2: Triton hot paths | ~0.3-1s | ~0.03s | ~0.3-1s | 81-270x |

### Per-stage estimates (PyTorch path, res=256)

| Stage | Current (custom/) | Target (PyTorch GPU) | Speedup |
|-------|-------------------|---------------------|---------|
| s1_voxelize | 2.2s | ~0.05s | ~44x |
| s2_components | 3.7s | ~0.2s | ~19x |
| s3_edge_weights | 18.8s | ~0.1s | ~188x |
| s4_face_point | 26.7s (s4a+s4b) | ~0.5s | ~53x |
| s6_collapse | 13.6s | ~0.2s | ~68x |
| s7_rank_assign | 12.1s | ~0.1s | ~121x |
| s8_decode | 4.0s | ~1.0s (with OPT-1/2) | ~4x |

---

## 8. Migration & Compatibility

### 8.1 Existing API Preservation

`run_hybrid_pipeline()` remains available for backward compatibility:

```python
def run_hybrid_pipeline(mesh_path, resolution, output_path, config=None, collector=None):
    """Legacy API. Delegates to corep_encode + corep_decode internally."""
```

### 8.2 CubeBatch as Exchange Format

`CubeBatch` is the single exchange format between:
- `corep_encode()` output
- `corep_decode()` input
- VAE training target (downstream consumer)
- A/B verification (via `interop/from_custom.py`)

No changes to `CubeBatch` field definitions are needed — the existing dataclass already covers all stage outputs.

### 8.3 s8_decode Handling

The existing `s8_collapse.py` is renamed to `s8_decode.py` with:
- Torch vectorized path preserved (already 15.8x vs custom/)
- Python fallback path removed (full GPU pipeline eliminates candidate edge cases)
- Input changes from `list[dict]` to `CubeBatch` (the `_cube_data_to_tensors` conversion becomes unnecessary since data is already tensors)

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| s4 clipping variable-length output on GPU | High — most complex stage to implement | Pre-allocate fixed 7-vertex buffer; validate max vertex count empirically |
| s6 U-Turn combinatorial explosion for high face_weights | Medium — could stall GPU threads | Split fast/slow paths; most cubes have 0 U-Turns |
| s2 Union-Find convergence on GPU | Low — bounded by small subgraph diameter | Iterative label propagation with convergence check; fallback to CPU for outliers |
| Float precision differences GPU vs CPU | Low — affects L2 comparisons | Use atol=1e-5 for float fields; edge_weights/face_weights are integer-exact |
| Memory pressure at high resolution | Medium — res=256 already uses 7.2GB for s8 alone | Monitor per-stage peak; s1-s7 GPU should be more memory-efficient than materializing all Python dicts |
