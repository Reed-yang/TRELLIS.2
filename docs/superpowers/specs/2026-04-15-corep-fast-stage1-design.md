# CoReP-Fast Stage 1 Design: Torch-vectorized Rewrite of `custom/` CoReP Pipeline

> Date: 2026-04-15
> Goal: Deliver `corep_fast/`, a production-grade rewrite of the CoReP representation pipeline that runs end-to-end on CUDA via Torch tensor vectorization, targeting **order-of-magnitude throughput gains for batch dataset generation**, while keeping `custom/` as an untouched reference implementation for correctness validation.
> Prerequisite: `custom/ARCHITECTURE.md`, `papers/20260315-Native_and_Compact_Structured_Latents_for_3D_Generation/full_text.md` §A.1 / §B, `memory/project_baseline_experiments.md`

---

## 0. Summary

CoReP (Compact Representation Pipeline) is an enhanced surface representation built on top of TRELLIS.2's O-Voxel. It encodes 18 per-cube edge crossing counts, 12 per-cube triangulated facet weights, connected-component counts, and component-level branch points — strictly more expressive than O-Voxel's binary edge flags. The current implementation in `custom/` is 6966 lines of pure Python + NumPy, **written for mathematical correctness, not performance**. Profiling is not yet available, but inspection shows:

- `custom/voxelize.py` (Stage 1) is fully single-threaded; per-triangle Python for-loop
- `custom/feature_*.py` (Stages 2–4) use `multiprocessing.Pool` for inter-cube parallelism, but inner math mixes vectorized NumPy with Python loops (edge dimension, polygon clipping, O(n²) node dedup)
- `custom/collapse_face.py` (Stage 6) is **fully single-threaded** with **combinatorial enumeration** up to O(W³)^12 (inner variant, capped at 100K) and O(W⁶)^12 (boundary variant, weakly capped)
- `custom/collapse.py` (Stage 8) is single-threaded with a streaming PLY writer

The decision is a **two-stage acceleration roadmap**:

- **Stage 1** (this document): NumPy→Torch vectorization + CUDA tensor operations + per-mesh GPU, orchestrated by `multiprocessing` across 8 GPUs. Delivered as an independent `corep_fast/` package; `custom/` is not modified.
- **Stage 2** (future, separate design): Triton kernels (same stack as the paper's FlexGEMM backend) and/or custom CUDA extensions targeting the remaining hot kernels identified by Stage 1's profiling data.

This document specifies Stage 1 exclusively.

### 0.1 Key Design Decisions (confirmed with user)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Use case | **Batch dataset generation** (VAE fine-tune data prep) | Throughput-optimized, per-mesh latency insensitive |
| Optimization depth ceiling | **Layer B** (Torch vectorization) now; Layer C (Triton) deferred to Stage 2 | De-risk first, let Stage 1 data drive Stage 2 kernel selection |
| Numerical consistency | **Topology equivalence** (not bit-exact) | Allows float32 GPU, parallel reduction reordering, tie-breaking differences |
| Hardware | **Single node, 8 GPUs** (`host-10-240-99-116`, same as baseline experiments) | Aligned with existing infra |
| Code layout | **New top-level `corep_fast/` directory**; `custom/` untouched | "Source is algorithm textbook, production in `corep_fast/`" — user quote |
| Cross-mesh GPU batching | **Not in Stage 1** (deferred to Stage 2) | Segmented ragged batching complexity belongs with Triton work |
| Implementation strategy | **Path A — full waterfall rewrite** of all 8 stages + Profiling-First sequencing | No mixed Python/Torch fallbacks; single consistent tech stack for Stage 2 friendliness |
| Profiling reference set | **Reuse baseline experiments evaluation set** | Aligned with `results/baseline_experiments/` |

### 0.2 Explicit Non-goals of Stage 1

Stage 1 will **not** do any of the following, no matter how tempting they become during implementation. If any of these turn out to be blocking, the implementation must pause and escalate for a design change:

1. Triton kernels (anywhere)
2. CUDA C++ native extensions (anywhere)
3. Cross-mesh GPU batching (multiple meshes sharing a single kernel launch)
4. Replacing `scipy.optimize.linear_sum_assignment` with a GPU Hungarian solver
5. Reusing `o-voxel/src/hash/` CUDA hash table in Stage 8
6. Modifying any file under `custom/`, `trellis2/`, `o-voxel/`
7. Introducing new Python dependencies other than `torch_scatter`
8. Optimizing PLY writer I/O (keep Python streaming writer)
9. Multi-node distributed execution
10. AMD GPU support
11. Per-mesh custom tuning outside the baseline evaluation set

---

## 1. Context: CoReP Pipeline Current State

### 1.1 Pipeline Stages

Based on `custom/feature.py` (entry point), the CoReP pipeline has 8 sequential stages:

| # | File | Output | LoC | Parallelism Today | Hotspot |
|---|------|--------|-----|-------------------|---------|
| 1 | `voxelize.py` | face_registers + boundary_registers (pickle) | 889 | **None** | Per-triangle Python for-loop at L210-242 |
| 2 | `feature_volume.py` | +num_components, +num_boundary | 330 | `Pool` (bs=10000) | DFS per cube (already light) |
| 3 | `feature_edge.py` | +edge_weights[18] | 259 | `Pool` | Möller-Trumbore — outer Python loop over 18 edges |
| 4a | `feature_face.py` | +face_weights[12] | 907 | `Pool` | **O(n²) point dedup** around L403-407 (`np.linalg.norm < 1e-8` pairwise); Sutherland-Hodgman per-polygon Python loop |
| 4b | `feature_point.py` | +component_points | 362 | `Pool` | Per-polygon Python Sutherland-Hodgman clip |
| 5 | `collapse_edge.py` | +loops (inner cubes) | 226 | `Pool` | Light; Python DFS on 18-node fixed graph |
| 6 | `collapse_face.py` | +loops (all cubes, U-Turn/boundary) | 737 | **None** | **O(W³)^12 ∧ O(W⁶)^12** combinatorial enumeration |
| 7 | `collapse_point.py` | +rank, +loop↔point matching | 1132 | `Pool` | Python DFS + O(k²) loop matching + scipy Hungarian |
| 8 | `collapse.py` | Final PLY mesh | 1172 | **None** | Shared-edge geometry + streaming PLY writer |

Total: **~6300 LoC of algorithmic code** excluding visualization and utilities.

### 1.2 Cross-cube Dependencies

| Stage | Locality |
|-------|----------|
| 1 — 7 | **All embarrassingly parallel across cubes**, with per-cube inputs/outputs only |
| 8 | **Cross-cube** — each global shared edge is shared by up to 4 neighboring cubes; resolved by lexicographic ownership |

### 1.3 TRELLIS.2 Reusable Infrastructure (not used by Stage 1)

| Asset | Path | Note |
|-------|------|------|
| CUDA hash table (Murmur3 + linear probing) | `o-voxel/src/hash/{hash.cu,hash.cuh,api.h}` | Useful for Stage 8 shared-edge lookup; **deferred to Stage 2** |
| Z-order / Hilbert encoding CUDA | `o-voxel/src/serialize/*.cu` | Low priority for CoReP |
| `flex_gemm` Triton sparse conv backend | pip package (paper §B FlexGEMM) | Only relevant for VAE convolutions, not CoReP |
| `SparseTensor` (N, 4) int32 container | `trellis2/modules/sparse/basic.py` | Not directly reused — `CubeBatch` fulfills the same role internally |

### 1.4 Paper Context (§A.1 / §B)

- §A.1 Algorithm 1 (Mesh→O-Voxel native conversion): uses per-triangle, per-edge loops in C++ with QEF accumulation. Paper claims **"a few seconds on a single CPU"** for the native O-Voxel conversion. This is the upper-bound target for Stage 1+2 combined — CoReP is strictly more complex so we do not expect to match it in Stage 1 alone.
- §B FlexGEMM: Triton-based sparse convolution backend with Masked Implicit GEMM + Gray code ordering + Split-K. Paper reports 2× speedup over torchsparse/spconv/fvdb. **This applies to the SC-VAE convolutional training path, not CoReP topology extraction.** It is useful as technology-stack reference for Stage 2, not an implementation dependency for Stage 1.

---

## 2. Two-Stage Roadmap

| | **Stage 1** (this spec) | **Stage 2** (future spec) |
|---|--|--|
| Tech stack | NumPy → Torch vectorization + CUDA tensor ops | Triton kernels + optional C++ pybind11 |
| Parallelism | `mp` across 8 GPUs; per-mesh GPU, per-cube batching within mesh | Per-kernel Triton fusion + optional cross-mesh segmented batching |
| Deliverables | `corep_fast/` drop-in pipeline, profiling harness, A/B validation, multi-GPU runner, baseline evaluation set results | Triton kernel library, kernel-level benchmarks, optional CUDA hash reuse |
| Exit criteria | 100% topology-equivalent output on baseline eval set; order-of-magnitude end-to-end speedup; automated identification of Stage 2 hot kernels | "Few seconds / mesh at 1024³" throughput target (paper §1 baseline); diminishing returns |
| Scope boundary | **`custom/` read-only**; no new deps except `torch_scatter` | Stage 1 containers and public API remain frozen; Triton plugs in behind the same `stage_forward()` signatures |

---

## 3. Stage 1 Objectives, Scope, Exit Criteria

### 3.1 Objectives

1. Deliver `corep_fast/` as a **complete, self-contained, drop-in** replacement of the 8-stage CoReP pipeline
2. All main compute paths run on CUDA via Torch tensor operations — no Python for-loops in the inner math
3. Preserve `custom/` as read-only reference ("algorithm textbook + A/B ground truth")
4. Ship a profiling harness that produces per-stage, per-substage wall-time + peak GPU memory data
5. Ship an A/B rig that runs `custom/` and `corep_fast/` side-by-side and validates topology equivalence
6. Ship a multi-GPU runner aligned with `scripts/eval/run_baseline_116.sh` patterns
7. Produce an automatically-generated Stage 2 Triton kernel candidate list as Stage 1's last artifact

### 3.2 Scope

**In scope:**
- Torch rewrite of all 8 stages: `s1_voxelize`, `s2_feature_volume`, `s3_feature_edge`, `s4_feature_face`, `s4_feature_point`, `s5_collapse_edge`, `s6_collapse_face`, `s7_collapse_point`, `s8_collapse`
- Unified Tensor data containers (`MeshTensors`, `CubeBatch`)
- Profiling harness + topology equivalence checker + A/B rig
- Multi-GPU task distribution (mesh-level, dynamic queue)
- Baseline evaluation set end-to-end A/B report
- Regression test set (pytest) with 3–5 representative meshes
- Interop bridges (`from_custom.py` / `to_custom.py`) to enable per-stage isolation debugging

**Out of scope:** see §0.2 explicit non-goals.

### 3.3 Exit Criteria

All four must be satisfied:

1. **Functional completeness**: All 8 stages have corresponding `corep_fast/stages/s{N}_*.py` implementations with matching public signatures
2. **Topology equivalence**: On the baseline evaluation set (≥14 meshes across 7 models × 2 resolutions), **100% of samples pass the topology equivalence checker** (see §7)
3. **Performance**: End-to-end wall time per mesh on a single H100/A100 shows **order-of-magnitude improvement** (specific numerical target set after §7.4 Step 0 baseline profiling produces `baseline_custom_v0.json`)
4. **Stage 2 handoff artifact**: `profiling/stage2_triton_candidates.md` auto-generated from the final profiling run, listing 3–5 kernel-level hot paths ranked by remaining time

---

## 4. Directory Structure

```
corep_fast/                              # project root, parallel with custom/
├── __init__.py                          # public API surface
├── pipeline.py                          # top-level entry: run_corep_fast(mesh_path, resolution, out_dir, config)
├── containers.py                        # MeshTensors, CubeBatch dataclasses
├── constants.py                         # EDGE_VERTS, TRIANGLES, cube topology tables
├── config.py                            # StageConfig, BackendConfig, Mode enum
│
├── geometry/                            # cross-stage reusable geometric kernels
│   ├── __init__.py
│   ├── sat.py                           # triangle-AABB separating axis test
│   ├── moller_trumbore.py               # ray-triangle intersection
│   ├── sutherland_hodgman.py            # polygon clipping against planes
│   ├── segment_aabb.py                  # segment-AABB SAT (for boundary edges)
│   └── topology_ops.py                  # normal-curve arc assignment, loop tracing (shared between s5/s6/s7)
│
├── stages/
│   ├── __init__.py
│   ├── s1_voxelize.py
│   ├── s2_feature_volume.py
│   ├── s3_feature_edge.py
│   ├── s4_feature_face.py
│   ├── s4_feature_point.py
│   ├── s5_collapse_edge.py              # inner cube normal-curve loop extraction
│   ├── s6_collapse_face.py              # U-Turn + boundary combinatorial resolution
│   ├── s7_collapse_point.py             # loop rank sorting + Hungarian point assignment
│   └── s8_collapse.py                   # global mesh stitching + PLY writer
│
├── interop/                             # bidirectional bridges with custom/
│   ├── __init__.py
│   ├── from_custom.py                   # custom/'s list-of-dict → CubeBatch
│   └── to_custom.py                     # CubeBatch → list-of-dict (for debug / partial A/B)
│
├── profiling/
│   ├── __init__.py
│   ├── harness.py                       # stage_timer context manager + ProfilingCollector
│   ├── topology_equivalence.py          # 5-layer equivalence checker (§7.2)
│   ├── ab_rig.py                        # side-by-side custom/ vs corep_fast/ runner
│   ├── baseline_runner.py               # baseline eval set driver
│   ├── stage2_candidate_generator.py    # auto-generates Stage 2 Triton kernel candidate list
│   └── report_builder.py                # JSON → Markdown/HTML summary
│
├── distributed/
│   ├── __init__.py
│   ├── mesh_worker.py                   # per-GPU worker process entry
│   ├── task_queue.py                    # mp.Queue wrapper with resume support
│   └── watchdog.py                      # worker health monitoring + auto-restart
│
├── tests/
│   ├── unit/
│   │   ├── test_containers.py
│   │   ├── test_geometry_sat.py
│   │   ├── test_geometry_moller_trumbore.py
│   │   ├── test_geometry_sutherland_hodgman.py
│   │   ├── test_s1_voxelize.py
│   │   ├── test_s2_feature_volume.py
│   │   ├── test_s3_feature_edge.py
│   │   ├── test_s4_feature_face.py
│   │   ├── test_s4_feature_point.py
│   │   ├── test_s5_collapse_edge.py
│   │   ├── test_s6_collapse_face.py
│   │   ├── test_s7_collapse_point.py
│   │   └── test_s8_collapse.py
│   ├── integration/
│   │   └── test_full_pipeline_small.py
│   └── regression/
│       ├── conftest.py
│       ├── fixtures/                    # 3-5 mesh paths + expected outputs
│       └── test_full_pipeline_equivalence.py
│
└── scripts/
    ├── run_parallel.py                  # multi-GPU runner (slurm + SSH fallback)
    ├── run_baseline_ab.sh               # wrapper: run A/B on baseline eval set
    ├── profile_stage_by_stage.py        # profiling harness driver
    └── plot_speedups.py                 # matplotlib bar chart from JSON profiling output
```

**Rationale for the `s{N}_` file-name prefix**: makes it trivial to align profiling reports, log files, and Stage 2 kernel candidate lists with the original `feature.py` pipeline ordering. `s4_feature_face.py` and `s4_feature_point.py` share the stage number because Stage 4 is conceptually a dual step (face weights and component points computed from the same clipped polygon data).

---

## 5. Data Containers

### 5.1 Design Principles

1. **One container per mesh.** `CubeBatch` holds all cubes of a single mesh as GPU tensors. Cross-mesh parallelism is handled at the `distributed/` layer, not inside the container.
2. **Dense where possible, CSR where ragged.** Fixed-dimension per-cube quantities (e.g., `edge_weights[18]`, `face_weights[12]`) are dense `(N, *)` tensors. Variable-length per-cube quantities (triangles registered to a cube, loops per cube, edges per loop) use two-level CSR (offset + values), compatible with `torch_scatter` segmented reductions.
3. **Frozen fields.** The dataclass field list is finalized at Stage 1 and **must not change in Stage 2**. Stage 2 may change internal storage layout (e.g., reorder for cache friendliness) but not the public dataclass interface.
4. **No Python-level dicts or lists.** Every piece of state is a `torch.Tensor`, allowing kernel fusion and zero-copy interop with downstream Torch ops.

### 5.2 `MeshTensors`

```python
# corep_fast/containers.py

@dataclass
class MeshTensors:
    """Input mesh after normalization to [0,1]³, held as device tensors."""

    # Geometry
    vertices:     torch.Tensor   # (V, 3)      float32
    faces:        torch.Tensor   # (F, 3)      int32    triangle → vertex indices
    triangles:    torch.Tensor   # (F, 3, 3)   float32  pre-gathered vertices[faces]
    face_normals: torch.Tensor   # (F, 3)      float32  unit normals

    # Topology derived from faces
    face_adj:     torch.Tensor   # (F, 3)      int32    neighboring face per edge, -1 if boundary

    # Open boundaries (edges belonging to exactly one face)
    boundaries:   torch.Tensor   # (B, 2, 3)   float32  segment endpoints in [0,1]³
    boundary_face_ids: torch.Tensor  # (B,)    int32    which face the boundary edge came from

    # Non-manifold features
    nm_edges:     torch.Tensor   # (M, 2, 3)   float32  edges shared by >2 faces
    nm_vertices:  torch.Tensor   # (P, 3)      float32  bowtie vertices

    # Normalization metadata (for reconstructing world coordinates)
    center:       torch.Tensor   # (3,)        float32  mesh centroid before normalization
    scale:        float                        # uniform scale applied
    resolution:   int                          # voxel grid resolution (e.g., 1024)

    # Device handle
    device:       torch.device

    def validate(self) -> None:
        """Fail fast on malformed inputs. Raised as CorepFastInputError."""
        ...
```

### 5.3 `CubeBatch`

```python
@dataclass
class CubeBatch:
    """All occupied cubes of a single mesh, with per-stage features accumulated."""

    # ─── Spatial identity ────────────────────────────────────────────
    cube_indices: torch.Tensor   # (N, 3)   int32    grid (ix, iy, iz)
    # Flat hash of cube_indices for fast neighbor lookup in Stage 8
    cube_hash:    torch.Tensor   # (N,)     int64    ix * res² + iy * res + iz

    # ─── Stage 1 outputs: ragged per-cube triangle / boundary registries (CSR) ───
    tri_offsets:  torch.Tensor   # (N+1,)   int64    cumulative triangle counts
    tri_values:   torch.Tensor   # (T,)     int32    indices into MeshTensors.faces
    bnd_offsets:  torch.Tensor   # (N+1,)   int64
    bnd_values:   torch.Tensor   # (B,)     int32    indices into MeshTensors.boundaries
    nm_offsets:   torch.Tensor   # (N+1,)   int64
    nm_values:    torch.Tensor   # (M,)     int32    indices into MeshTensors.nm_edges

    # ─── Stage 2 outputs: scalar topology features ───────────────────
    num_components: torch.Tensor  # (N,)    int32
    num_boundary:   torch.Tensor  # (N,)    int32

    # ─── Stage 3 outputs: edge crossing counts ───────────────────────
    edge_weights:   torch.Tensor  # (N, 18) int32

    # ─── Stage 4 outputs: face weights + component points ────────────
    face_weights:   torch.Tensor  # (N, 12) int32
    # Component points in ragged CSR
    point_offsets:  torch.Tensor  # (N+1,)   int64
    point_values:   torch.Tensor  # (P, 3)   float32

    # ─── Stage 5–6 outputs: loops (two-level CSR) ────────────────────
    # cube → loops (cube level offsets)
    loop_cube_off:  torch.Tensor  # (N+1,)       int64    #loops per cube
    # loop → edges (loop level offsets, indexed against L_total = loop_cube_off[-1])
    loop_edge_off:  torch.Tensor  # (L+1,)       int64    #edges per loop
    # Edge values
    loop_edge_val:  torch.Tensor  # (E,)         int32    edge idx in [0, 18)
    # Stage 7 adds rank alongside the edge values
    loop_edge_rank: torch.Tensor  # (E,)         int32    rank ≥ 0, -1 if unassigned

    # ─── Stage 6/7 outputs: loop-to-point matching ───────────────────
    # For each loop (L total), which point (index into point_values of the same cube) is it matched to
    loop_point_match: torch.Tensor  # (L,) int32, -1 if unmatched (exception cube)

    # ─── Status per cube ─────────────────────────────────────────────
    # 0 = OK, 1 = AMBIGUOUS (multiple solutions), 2 = UNSOLVABLE (no solution)
    # 3 = BUDGET_EXCEEDED (s6 cube with prod(K_i) > threshold)
    status: torch.Tensor         # (N,)     int32

    # ─── Bookkeeping ────────────────────────────────────────────────
    device: torch.device
    resolution: int

    # ─── Invariant checks (debug mode only) ─────────────────────────
    def invariants_check(self, stage: str) -> None:
        """Verify per-stage invariants. Raises CorepFastInvariantError on violation."""
        ...

    # ─── Size / shape introspection ─────────────────────────────────
    @property
    def num_cubes(self) -> int: ...
    @property
    def num_loops(self) -> int: ...
    @property
    def num_loop_edges(self) -> int: ...
```

### 5.4 Ragged CSR Convention

All two-level ragged data uses cumulative offset arrays (PyTorch and `torch_scatter` convention), with the sentinel invariant `offsets[0] == 0 and offsets[-1] == total_count`:

```
cube i → tri_values[tri_offsets[i] : tri_offsets[i+1]]
loop l → loop_edge_val[loop_edge_off[l] : loop_edge_off[l+1]]
loop l belongs to cube = torch.searchsorted(loop_cube_off, l) - 1
```

This is identical to `torch_scatter.segment_csr` expectations, which means per-segment reductions (`segment_sum`, `segment_max`, `segment_softmax`) work natively.

### 5.5 Interop Bridges

`interop/from_custom.py` converts `custom/`'s list-of-dict format into a `CubeBatch`:

```python
def cube_batch_from_custom(
    face_registers: list[dict],   # output of a custom/ stage
    mesh: MeshTensors,
    include: set[str] = {'all'},  # which fields to populate, others left as empty tensors
) -> CubeBatch:
    ...
```

`interop/to_custom.py` is the inverse, used for:
1. Feeding `corep_fast/` stage outputs into `custom/` for partial end-to-end tests
2. Dumping a debug snapshot from a failing mesh for offline analysis

---

## 6. Stage Rewriting Strategy

Each stage has the same public signature:

```python
# corep_fast/stages/s{N}_*.py

def stage_forward(
    mesh: MeshTensors,
    cube: CubeBatch,
    *,
    config: StageConfig = StageConfig.default(),
) -> CubeBatch:
    """
    Pure function: reads mesh + cube, returns a new CubeBatch with this stage's
    output fields populated. Does not mutate input.
    """
```

The Torch backend and (future) Triton backend both implement this signature; `pipeline.py` dispatches based on `config.backend`.

### 6.1 `s1_voxelize` — Face / Boundary / Non-manifold Registration

**Current** (`custom/voxelize.py:210-242`): per-triangle Python for-loop; SAT is vectorized only within a single triangle's candidate cubes.

**Torch strategy:**

1. Move `triangles: (F, 3, 3)` and `boundaries: (B, 2, 3)` to GPU (done in `MeshTensors`)
2. For each triangle, compute its AABB integer voxel range → `cands_per_tri: (F,) int32`
3. `torch.repeat_interleave` expands into a flat `(sum_cands,)` tensor of triangle indices; candidate cube indices are computed by modular arithmetic on the flat index
4. Run vectorized SAT test over the flat `(sum_cands,)` pair array → boolean hit mask
5. Compact using `torch.nonzero` → `(N_pairs, 2) int32` pairs `(tri_idx, cube_flat_idx)`
6. Sort by `cube_flat_idx` → `torch.unique_consecutive` produces unique cubes and their CSR boundaries in one pass
7. Populate `cube.cube_indices`, `cube.tri_offsets`, `cube.tri_values`

Same pattern for boundaries and non-manifold edges.

**Chunking to bound peak memory:** If `sum_cands` exceeds a memory budget (configurable, default 512 MB), the triangle set is processed in chunks. Each chunk produces its own `(triangle, cube)` hit list, and the final sort/unique pass consolidates them.

**Invariants to check:**
- `tri_offsets[0] == 0 and tri_offsets[-1] == tri_values.shape[0]`
- `cube_indices[:, 0].max() < resolution` (likewise y, z)
- No triangle registered to zero cubes (caught at step 2)

**Expected speedup:** 50–200× over `custom/` single-threaded baseline.

### 6.2 `s2_feature_volume` — Connected Component Count per Cube

**Current** (`custom/feature_volume.py`): per-cube DFS on a local subgraph of the mesh face adjacency; `Pool`-parallelized.

**Torch strategy:**

1. Build a flat `(sum_tri_values,)` "node id within cube" array using CSR arithmetic
2. For each pair of triangles within the same cube that are adjacent in the global mesh (looked up via `mesh.face_adj`), emit an edge `(node_i, node_j)` — this is done with a single `gather` + mask operation
3. Run **iterative label propagation** using `torch_scatter.segment_min`: each node takes the minimum label of its neighbors, iterating until convergence (typically 5–10 iterations for per-cube subgraphs with ≤ 20 nodes)
4. `num_components[cube]` = count of distinct labels within the cube, computed via `segment_unique`

**Fallback for deep subgraphs:** If any cube fails to converge in 20 iterations, mark and run Python networkx DFS on just those cubes (expected to be <1% of cubes).

**Expected speedup:** 5–30×.

### 6.3 `s3_feature_edge` — 18 Edge Crossings per Cube via Möller-Trumbore

**Current** (`custom/feature_edge.py:42-92`): outer Python loop over 18 edges, inner vectorization over triangles within a cube. `Pool`-parallelized.

**Torch strategy:**

1. Constant: `edge_starts: (18, 3), edge_ends: (18, 3)` (unit-cube local coordinates)
2. Per-cube, compute world-space edge origins: `(N, 18, 3)`
3. Gather triangles via CSR: `(sum_tri, 3, 3)`
4. Flatten `(N_cubes × 18, 1)` edges against `(sum_tri_per_cube,)` triangles using CSR segment broadcasting
5. Vectorized Möller-Trumbore: compute det, u, v, t → boolean hit mask
6. `segment_sum` to accumulate hits per `(cube, edge)` pair → `edge_weights[N, 18]`

**Key insight:** Both the edge dimension and the triangle dimension are fused into a single kernel, eliminating the Python loop entirely.

**Expected speedup:** 10–30×.

### 6.4 `s4_feature_face` + `s4_feature_point` — Facet Weights and Component Points

**Current**: Sutherland-Hodgman polygon clipping per (cube, triangle, facet) in Python; O(n²) point dedup in `feature_face.py:403-407` (pairwise `np.linalg.norm` comparison in a list comprehension).

**Torch strategy:**

1. **Polygon clipping as 6 vectorized plane clips:** For each (cube, triangle) pair, start with the triangle as a padded `(M, max_verts=12, 3)` polygon. For each of the 6 AABB planes, compute signed distances, identify kept / crossed edges, interpolate intersections, update polygon in-place. 6 iterations total, fully vectorized over M.
2. **Classify clipped polygon fragments** against the 12 triangular facets using barycentric coordinates, vectorized
3. **Facet weight accumulation:** `scatter_add` into `face_weights[N, 12]`
4. **Component point extraction:** For each connected component (identified by re-running s2-style label propagation on the clipped polygon fragments), compute area-weighted centroid using `scatter_add` — this replaces the per-polygon Python loop
5. **Node dedup** is replaced by sort-based unique: `torch.unique(torch.round(coords * 1e8), return_inverse=True)`, which is O(n log n) on GPU

**Max polygon vertex bound:** `max_verts=12` is sufficient because a triangle clipped against 6 AABB planes has at most 3 + 6 = 9 vertices. Budget 12 for safety.

**Expected speedup:** 30–100× (driven largely by eliminating the O(n²) dedup).

### 6.5 `s5_collapse_edge` — Normal-Curve Loop Extraction for Inner Cubes

**Current** (`custom/collapse_edge.py`): per-cube fixed-topology DFS, 18 edges, 12 triangles. `Pool`-parallelized. Already O(1) per cube.

**Torch strategy:**

1. **Pre-computed arc assignment lookup table**: Given `(w1, w2, w3)` per triangle, the arc counts `(k12, k23, k31)` are closed-form. If `max_edge_weight ≤ 8`, the LUT fits in `(9³ × 12 triangles × 3 outputs) × 4 bytes ≈ 34 KB` — resident in constant memory.
2. **Loop tracing:** For each cube, traverse the 2-regular graph of `(edge, point)` nodes. Use `torch.vmap` over cubes to batch per-cube loop tracing.
3. Convergence is bounded by the loop length, at most `sum(edge_weights) ≤ 18 × max_w`

**Fallback if `max_w > 8`:** Online computation instead of LUT; negligible slowdown since most cubes have `max_w ≤ 4`.

**Expected speedup:** 3–10×. The main gain is removing multiprocessing IPC overhead, not raw compute.

### 6.6 `s6_collapse_face` — U-Turn / Boundary Resolution (**Hardest Stage**)

**Current** (`custom/collapse_face.py:220-296`): fully single-threaded combinatorial enumeration.
- Inner variant: O(W³)^12 (capped at 100K configs)
- Boundary variant: O(W⁶)^12 (weakly capped with `islice(..., 1000)`)

**Three-layer Torch strategy:**

#### 6.6.a Algebraic Pruning (pure algorithmic win)

Currently the code enumerates `(u1, u2, u3)` combinations and then validates triangle inequality + parity. Invert this order: solve the constraints first.

For each facet `f` with face_weight `W`:
- Valid assignments satisfy `u1 + u2 + u3 = W ∧ u_i ≥ 0 ∧ u_i ≤ w_i // 2 ∧ (w_i - 2u_i) satisfies triangle ineq. & parity with the other two adjusted weights`
- The valid set is typically 1–5 elements (not `W³`)

For the boundary variant, add a global constraint: the sum of `(u + b)` over all facets must be even (each open path has two endpoints).

This layer alone compresses worst-case enumeration from 100K to O(1–10K) even before GPU vectorization.

#### 6.6.b Per-cube Vectorized Enumeration

For each cube with a list of per-facet valid assignments `valid_i: (K_i, 3)`:

1. Compute `prod_K = prod(K_0, K_1, ..., K_11)` — total enumeration budget
2. If `prod_K > PROD_THRESHOLD` (default 100000, matches `custom/`), mark as `BUDGET_EXCEEDED` and skip
3. Compute Cartesian product via `torch.cartesian_prod` or index broadcasting → `(prod_K, 12, 3)` global configs
4. For each global config, compute adjusted edge weights per facet, then adjacency edges for the 2-regular graph
5. Batched DFS/BFS loop tracing using label propagation
6. Canonicalize each traced loop set (rotation + reflection normalization) via `torch.sort` + tuple encoding
7. Deduplicate via `torch.unique_dim` on canonical representations
8. Classify: 0 solutions → `UNSOLVABLE`, 1 solution → `OK`, ≥2 solutions → `AMBIGUOUS`

#### 6.6.c Cross-cube Bucketed Dispatch (**critical for GPU utilization**)

Cubes have widely varying `prod_K`:

```
prod_K distribution (typical):
  ≤ 64       : ~70% of cubes (simple meshes, few U-Turns)
  64 – 1K    : ~20%          (moderate folding)
  1K – 100K  : ~10%          (complex multi-layer geometry)
  > 100K     : ~<1%          (pathological — BUDGET_EXCEEDED)
```

A single kernel would be bottlenecked by the worst cube in the warp. Dispatch by bucket:

```python
def dispatch_s6(cubes_with_valid_sets):
    by_bucket = partition_by_prod_k(cubes_with_valid_sets)

    # Bucket 1: tiny (prod_K ≤ 64) — warp-level vmap
    run_tiny_bucket(by_bucket['tiny'])           # 32 cubes per warp

    # Bucket 2: medium (64 < prod_K ≤ 1024) — thread-block level
    run_medium_bucket(by_bucket['medium'])       # 1 cube per block

    # Bucket 3: large (1K < prod_K ≤ 100K) — SM-level parallel
    run_large_bucket(by_bucket['large'])         # 1 cube per SM

    # Bucket 4: huge — marked BUDGET_EXCEEDED, handled by s8 exception path
    mark_budget_exceeded(by_bucket['huge'])
```

This is the standard pattern for GPU irregular parallelism (cf. graph analytics edge-parallel vs vertex-parallel dispatch). In Stage 1 we implement it at the Torch level — each bucket gets its own `torch.vmap` / batched kernel call with different shape configurations. In Stage 2 it becomes Triton kernels with tuned `BLOCK_SIZE` per bucket.

**Hard protection:** Cubes with `prod_K > 100000` are marked `BUDGET_EXCEEDED` and propagated to Stage 8's exception handling (the first component_point is used via lexicographic ownership). This matches `custom/`'s existing behavior and preserves topology equivalence.

**Implementation order within s6:**
1. Write §6.6.a algebraic pruning as pure Python on a CPU validation rig — prove correctness
2. Port §6.6.b to Torch tensors, single-cube at a time (no batching yet) — verify numerical match with custom/
3. Add §6.6.c bucket dispatch — benchmark GPU utilization

**Expected speedup:** 20–500× depending on worst-case cube distribution in the baseline eval set.

### 6.7 `s7_collapse_point` — Loop Rank Sorting + Hungarian Point Assignment

**Current**: DFS-based rank reconstruction + O(k²) loop matching + `scipy.optimize.linear_sum_assignment`.

**Torch strategy:**

1. Rank-aware DFS: `vmap` over cubes. Each cube's state is small (≤ 18 × max_w nodes).
2. Loop sequence matching: replace O(k²) string matching with `torch.roll`-based vectorized comparison (for each candidate shift + direction, compare two int tensors)
3. **Hungarian stays on CPU:** Cost matrices per cube are tiny (≤ 10×10). Build them on GPU, `.cpu()`, call `scipy.optimize.linear_sum_assignment`, results back to GPU. The CPU-GPU roundtrip per cube is fast because it's a single small tensor transfer, and scipy's C implementation beats any GPU Hungarian implementation at these sizes.

**Deferred optimization:** If profiling shows that CPU-GPU roundtrip dominates (unlikely given Hungarian is rarely > 1% of total time), Stage 2 may replace this with a batched GPU Hungarian via `torch_linear_assignment`.

**Expected speedup:** 5–20×.

### 6.8 `s8_collapse` — Global Shared-Edge Stitching + PLY Streaming

**Current** (`custom/collapse.py`): single-threaded iteration over cubes; each cube emits up to 12 global shared edges; lexicographic ownership determines which cube processes each edge; the handler reads 4 neighboring cubes' loop data and writes vertices/faces to streaming PLY temp files.

**Torch strategy:**

1. **Parallel global edge enumeration:**
   - Each cube generates 12 global edge identifiers `(axis, nx, ny, nz)` → encode to `int64` for sorting
   - `torch.sort` + `torch.unique_consecutive` gives unique shared edges and their 4-cube neighborhoods
2. **Lexicographic ownership = GPU argmin** per shared edge over its 4 neighboring cube indices
3. **Shared-edge geometry processing** (`process_shared_edge_geometry` equivalent):
   - For each owned shared edge, CSR-gather the loop data of the 4 neighboring cubes
   - Extract loops crossing this edge at each rank
   - Compute the projection point as the mean of up to 4 component points
   - Emit up to 4 triangles in a fan arrangement around the projection point
   - Vectorized over all owned shared edges simultaneously
4. **Vertex welding:** `torch.unique(torch.round(vertex_coords * 10**merge_decimals), return_inverse=True)` → global vertex table
5. **PLY writing:** The GPU produces dense `(V, 3) float32` and `(F, 3) int32` tensors, which are copied back to CPU and written by a simple Python streaming writer (matches `custom/collapse.py`'s approach for I/O, no need to optimize further)

**Exception cube handling:** Cubes with `status ∈ {AMBIGUOUS, UNSOLVABLE, BUDGET_EXCEEDED}` are folded into the wildcard slot (first component_point injected into missing rank positions), matching `custom/`'s behavior.

**Rank normalization:** Handled vectorized — the sign flip for edges 2, 6, 3, 7 is a single `torch.where` operation on the edge index.

**Expected speedup:** 10–30×.

### 6.9 Expected Speedup Summary

| Stage | Expected Stage 1 speedup |
|-------|-------------------------:|
| s1 voxelize | 50–200× |
| s2 feature_volume | 5–30× |
| s3 feature_edge | 10–30× |
| s4 feature_face + point | 30–100× |
| s5 collapse_edge | 3–10× |
| s6 collapse_face | 20–500× |
| s7 collapse_point | 5–20× |
| s8 collapse | 10–30× |

**End-to-end estimate**: 10–50× total, heavily dependent on how much of the current runtime is spent in s6. A baseline profiling run is required before committing to a numerical target.

---

## 7. Profiling + A/B Validation Harness

### 7.1 Profiling Harness (`corep_fast/profiling/harness.py`)

```python
@contextmanager
def stage_timer(name: str, collector: ProfilingCollector):
    """GPU-fenced timing context manager. Records wall time and peak GPU memory."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mem_baseline = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        collector.record(
            stage=name,
            wall_time_s=time.perf_counter() - t0,
            peak_gpu_mem_bytes=torch.cuda.max_memory_allocated() - mem_baseline,
        )
```

**Collection granularity:**
- **L0 full pipeline**: wall time from mesh load to PLY write
- **L1 per stage** (s1–s8): each stage's wall time + peak memory
- **L2 substage for s6**: algebraic pruning / enumeration / loop tracing / deduplication — separately timed because s6 is the most complex
- **L3 kernel-level**: **not in Stage 1**, deferred to Stage 2 with `torch.profiler` / Nsight Systems

**Output:** `profiling/runs/{timestamp}_{mesh_name}.json` with fields:
```json
{
  "mesh": "sphere.ply",
  "resolution": 1024,
  "impl": "corep_fast",
  "stages": {
    "s1_voxelize": {"wall_time_s": 0.34, "peak_gpu_mem_bytes": 1.2e8},
    "s6_collapse_face": {
      "wall_time_s": 2.15,
      "peak_gpu_mem_bytes": 8.9e8,
      "substages": {
        "algebraic_pruning": 0.12,
        "enumeration":       1.48,
        "loop_tracing":      0.39,
        "deduplication":     0.16
      }
    },
    ...
  },
  "total_wall_time_s": 5.82,
  "total_peak_gpu_mem_bytes": 1.3e9
}
```

### 7.2 Topology Equivalence Checker (`corep_fast/profiling/topology_equivalence.py`)

Five-layer check, failing fast at the earliest divergence:

| Layer | Fields | Tolerance | Semantics |
|-------|--------|-----------|-----------|
| **L1** | `num_components`, `num_boundary`, `edge_weights[18]`, `face_weights[12]`, `status` | None (exact) | Per-cube integer equality |
| **L2** | Loop set per cube | None | Two loop sets are equivalent iff after canonical rotation + reflection, each loop in set A has an exact edge-sequence match in set B |
| **L3** | Rank assignment on loop edges | None | Per loop, after canonicalization, rank sequences must match exactly |
| **L4** | Loop ↔ component_point matching | None | Per loop, matched point index must be identical |
| **L5** | Component point 3D coordinates | `rtol=1e-4, atol=1e-6` | `torch.allclose` tolerance, looser than default for float32 GPU noise |

**Mismatch report format:**

```
Mesh: sphere_triple_layer.ply | Resolution: 1024
Total cubes: 12345

Layer 1 (integer fields):   PASS (12345/12345)
Layer 2 (loop structure):   FAIL (12340/12345)
  First mismatch: cube (ix=512, iy=256, iz=128)
    custom/:
      loop 0: edges=[3,7,11,5], canonical=[3,5,7,11]
      loop 1: edges=[2,9,14,6], canonical=[2,6,9,14]
    corep_fast/:
      loop 0: edges=[3,7,11,5], canonical=[3,5,7,11]  ← match
      loop 1: edges=[2,6,9,14], canonical=[2,6,9,14]
    Issue: loop 1 edge sequence differs; unclear if canonical bug or real divergence
  Suggested debug: run s6 isolation test on cube (512,256,128)

Layer 3 (ranks):            SKIPPED (upstream failed)
Layer 4 (point matching):   SKIPPED
Layer 5 (point coords):     SKIPPED
```

**Stage isolation debug mechanism:** When layer 2 fails at cube `c`, the harness optionally runs:
1. Feed `custom/`'s upstream output (stages 1–5) into `corep_fast/stages/s6_collapse_face.py` via `interop/from_custom.py`
2. Run only s6 on that single cube
3. If s6 isolation passes: bug is in s1–s5
4. If s6 isolation fails: bug is in s6

This "stage bisection" is the single most valuable debugging tool during Stage 1 development.

### 7.3 A/B Rig (`corep_fast/profiling/ab_rig.py`)

```python
@dataclass
class ABReport:
    equivalence: EquivalenceReport
    custom_timings: ProfilingCollector
    fast_timings: ProfilingCollector
    speedups_per_stage: dict[str, float]
    speedup_total: float

def ab_run(
    mesh_path: str,
    resolution: int,
    *,
    stages_torch: set[str] = None,  # None = all stages use corep_fast; subset = partial A/B
    output_dir: str | None = None,
) -> ABReport:
    ...
```

**Partial A/B mode** is crucial during Stage 1 development. When only s1 has been rewritten, the rig runs:
1. `custom/` 8-stage pipeline → output_A
2. `corep_fast/s1_voxelize` + `custom/` stages 2–8 → output_B (using interop bridges)
3. Compares output_A vs output_B to validate s1 in isolation

### 7.4 Baseline Runner (`corep_fast/profiling/baseline_runner.py`)

Uses the existing baseline evaluation set from `memory/project_baseline_experiments.md` (7 models × 2 resolutions in `results/baseline_experiments/`).

**Step 0 (runs before any Torch rewriting begins):**

```bash
.venv/bin/python corep_fast/profiling/baseline_runner.py \
    --impl custom \
    --evalset results/baseline_experiments/data \
    --resolutions 512,1024 \
    --output profiling/runs/baseline_custom_v0.json
```

This produces `baseline_custom_v0.json` — the **single source of truth** for all performance claims in Stage 1. Every later `ab_after_s{N}.json` is compared against this file.

**Incremental A/B runs** (after each stage rewrite):

```bash
.venv/bin/python corep_fast/profiling/baseline_runner.py \
    --impl ab \
    --evalset results/baseline_experiments/data \
    --stages-torch s1,s3,s4 \
    --output profiling/runs/ab_after_s4.json
```

**Final Stage 1 run:**

```bash
.venv/bin/python corep_fast/profiling/baseline_runner.py \
    --impl corep_fast \
    --evalset results/baseline_experiments/data \
    --output profiling/runs/stage1_final.json
```

The final run triggers `stage2_candidate_generator.py` to produce `profiling/stage2_triton_candidates.md`.

### 7.5 Regression Test Suite (`corep_fast/tests/regression/`)

A minimal pytest-organized subset of the baseline eval set, representative of all topology categories:

| Fixture | Category | Validates |
|---------|----------|-----------|
| `sphere.ply` (icosphere subdivisions=3) | Closed smooth | Baseline sanity |
| `open_plane.ply` | Open boundary | s6 boundary variant |
| `triple_layer_sphere.ply` | Multi-shell | `num_components > 1`, U-Turn |
| `turbine.ply` (from `tmp/test_mesh/`) | Industrial complex | Non-manifold handling |
| `cloth_folded.ply` | Folded thin surface | s6 worst case (high prod_K) |

```python
# tests/regression/test_full_pipeline_equivalence.py

@pytest.mark.parametrize("mesh_name,resolution", [
    ("sphere.ply",              256),
    ("open_plane.ply",          256),
    ("triple_layer_sphere.ply", 256),
    ("turbine.ply",             512),
    ("cloth_folded.ply",        256),
])
def test_full_pipeline_equivalence(mesh_name, resolution):
    report = ab_run(f"tests/regression/fixtures/{mesh_name}", resolution)
    assert report.equivalence.all_layers_pass(), report.equivalence.pretty_print()
    assert report.speedup_total >= 2.0
```

### 7.6 CI Integration

**Stage 1 does not wire into CI** — baseline eval set files are too large and require GPU resources. Validation is manual, triggered by:

```bash
bash corep_fast/scripts/run_baseline_ab.sh
```

**Milestone gating** (per `memory/feedback_stay_on_plan`):
- Each stage rewrite is considered complete only after `ab_after_s{N}.json` shows 100% equivalence on the regression fixtures AND no performance regression on any previously-rewritten stage
- Only then may the next stage rewrite begin

### 7.7 Implementation Sequencing

Stage 1 begins with **Step 0: infrastructure only (no Torch rewrites)**:

1. `containers.py` + `constants.py` — frozen data structures
2. `interop/from_custom.py` + `interop/to_custom.py` — bidirectional bridges
3. `profiling/harness.py` + `profiling/topology_equivalence.py`
4. `profiling/baseline_runner.py`
5. Run `baseline_runner --impl custom` to produce `baseline_custom_v0.json`

**After Step 0, re-read `baseline_custom_v0.json` and reconfirm the stage rewriting priority.** The tentative priority is `s1 → s6 → s3 → s4 → s2 → s5 → s7 → s8` (high-impact + high-risk first), but actual priority is determined by where the baseline profile shows the most time spent. This is true Profiling-First ordering — the profiler output directly drives subsequent work.

---

## 8. Multi-GPU Task Distribution (`corep_fast/distributed/`)

### 8.1 Scheduling Model

**Mesh-level static-allocation with dynamic pull.** Each of the 8 GPUs runs one worker process, pinned via `CUDA_VISIBLE_DEVICES`. A shared `multiprocessing.Queue` feeds mesh tasks; workers pull when ready.

**Why not `torch.distributed` / DDP:** No shared model, no gradients, no all-reduce. DDP's NCCL init and synchronization barriers are pure overhead for this workload.

**Why not `Pool.imap`:** Pool's result pickling would ship full `CubeBatch` tensors (hundreds of MB per mesh) back to the main process through IPC, which saturates the pipe. Workers write outputs directly to disk and return only a small summary dict to the main process.

### 8.2 Worker Process (`mesh_worker.py`)

```python
def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    config: WorkerConfig,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)   # device 0 in the pinned view

    # One-time GPU constant initialization
    constants = load_gpu_constants()  # EDGE_VERTS, TRIANGLES, arc LUT, etc.

    while True:
        task = task_queue.get()
        if task is None:
            break  # sentinel, clean shutdown

        try:
            cube = run_corep_fast(
                mesh_path=task.mesh_path,
                resolution=task.resolution,
                out_dir=task.out_dir,
                constants=constants,
                config=config,
            )
            result_queue.put(WorkerResult(
                status='ok',
                mesh_path=task.mesh_path,
                wall_time_s=cube.total_time_s,
                output_path=cube.ply_path,
                stats=cube.stats,
            ))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result_queue.put(WorkerResult(
                status='oom',
                mesh_path=task.mesh_path,
                retry_with='hard_queue',
            ))
        except Exception as e:
            result_queue.put(WorkerResult(
                status='error',
                mesh_path=task.mesh_path,
                error=traceback.format_exc(),
            ))
```

### 8.3 Task Queue with Resume (`task_queue.py`)

**Resume mechanism:**

```bash
.venv/bin/python corep_fast/scripts/run_parallel.py \
    --meshes results/baseline_experiments/data/*.ply \
    --num-gpus 8 \
    --output-dir results/corep_fast_run/ \
    --resolution 1024 \
    --resume
```

- Each completed mesh writes `{out_dir}/done/{mesh_hash}.json` with status + timings
- On `--resume`, scan `done/` at startup, skip meshes whose hash already has a completion marker
- Matches the retry style of `scripts/eval/run_baseline_116.sh` and `phase_a_retry_*` logs

**OOM "hard queue" fallback:**

- First OOM on a mesh → retry with `oom_safe_config` (smaller chunk size, lower max_verts, more aggressive budget limits)
- Second OOM → retry on CPU (extremely slow, but guarantees progress)
- Third failure → mark as permanently failed, record in summary

**Worker crash watchdog:** A watchdog thread in the main process checks worker heartbeat every 30s. Dead workers are `terminate()`-ed and replaced; their in-flight task is re-enqueued.

### 8.4 Top-Level Runner (`scripts/run_parallel.py`)

Call pattern aligned with `scripts/eval/run_baseline_116.sh`:
- Slurm submission first, with `#SBATCH --gres=gpu:8` and `--nodelist=host-10-240-99-116`
- SSH fallback if slurm is unavailable
- `--ssh` flag forces SSH mode

Output:
- `{out_dir}/done/*.json` — per-mesh completion markers
- `{out_dir}/summary.md` — human-readable summary with per-mesh timings and topology equivalence results
- `{out_dir}/summary.csv` — machine-readable for comparison with `results/baseline_experiments/`
- `{out_dir}/profiling_runs/*.json` — raw profiling data

### 8.5 Observability

Main process prints a progress line every 30s:

```
[worker 0: done 12, hard 1, avg 34.2s | worker 1: done 11, hard 0, avg 31.8s | ... | total: 92/112]
```

---

## 9. Error Handling and Debug Modes

### 9.1 Error Layers

| Layer | Error | Handler |
|-------|-------|---------|
| **L0 Input** | Malformed mesh, zero faces, non-triangular | `MeshTensors.validate()` raises `CorepFastInputError`; worker marks mesh as failed, continues |
| **L1 Invariant** | `CubeBatch` field violates stage invariants (debug mode only) | `invariants_check(stage)` raises `CorepFastInvariantError(stage, cube_idx)` |
| **L2 Algorithm** | Per-cube algorithm failure (s6 no valid config, s7 imperfect matching) | Mark cube `status ∈ {AMBIGUOUS, UNSOLVABLE}`, delegate to s8 exception path; do **not** fail the mesh |
| **L3 Resource** | OOM, CUDA illegal memory | `empty_cache()` → retry with smaller chunk → hard_queue → CPU fallback → permanent failure |
| **L4 Scheduling** | Worker crash, hang | Watchdog detects, terminates, restarts; task re-enqueued |

**Invariant**: The corep_fast/ behavior for exception cubes must exactly match `custom/`. The topology equivalence checker (§7.2) will verify that the exception cube set is identical between the two implementations.

### 9.2 Debug Modes

```python
class Mode(enum.Enum):
    PRODUCTION = 'production'  # default: skip invariant checks, fastest
    DEBUG      = 'debug'       # enable invariants_check, dump per-stage pickle
    STRICT     = 'strict'      # DEBUG + per-stage A/B check against custom/ (very slow)

corep_fast.set_mode(Mode.PRODUCTION)
```

**STRICT mode** is the primary debugging tool: when a bug is reported, rerun the failing mesh in STRICT mode; the harness stops at the first stage that diverges from `custom/` and dumps a side-by-side report.

---

## 10. Stage 2 Transition Interface

### 10.1 Frozen Public API

```python
# corep_fast/__init__.py

from .pipeline import run_corep_fast
from .containers import MeshTensors, CubeBatch
from .config import StageConfig, BackendConfig, Mode, set_mode
from .stages import (
    s1_voxelize,
    s2_feature_volume,
    s3_feature_edge,
    s4_feature_face,
    s4_feature_point,
    s5_collapse_edge,
    s6_collapse_face,
    s7_collapse_point,
    s8_collapse,
)
```

All stage modules export a single `stage_forward(mesh, cube, *, config)` function. **The signature is frozen in Stage 1 and must not change in Stage 2.** Triton backends in Stage 2 implement the same signature.

### 10.2 Backend Dispatch

```python
# corep_fast/config.py

@dataclass
class BackendConfig:
    s1: Literal['torch', 'triton']
    s2: Literal['torch', 'triton']
    s3: Literal['torch', 'triton']
    s4_face: Literal['torch', 'triton']
    s4_point: Literal['torch', 'triton']
    s5: Literal['torch', 'triton']
    s6: Literal['torch', 'triton']
    s7: Literal['torch', 'triton']
    s8: Literal['torch', 'triton']

    @classmethod
    def all_torch(cls) -> 'BackendConfig':
        return cls(**{field: 'torch' for field in cls.__dataclass_fields__})
```

Stage 2 can switch stages one at a time:

```bash
.venv/bin/python corep_fast/scripts/run_parallel.py \
    --backend s6=triton \
    --resume
```

### 10.3 Stage 2 Triton Candidate List

At Stage 1 completion, `profiling/stage2_candidate_generator.py` auto-generates `profiling/stage2_triton_candidates.md`:

```
# Stage 2 Triton Kernel Candidates
# Auto-generated from stage1_final.json on {date}

Rank | Stage     | Subpath                  | Time Share | Note
-----|-----------|--------------------------|------------|------------------------------
  1  | s6        | bucket_large_enumeration | 34%        | Warp divergence dominant
  2  | s1        | sat_candidate_filter     | 22%        | Memory-bound
  3  | s6        | adjacency_construction   | 15%        | scatter_add contention
  ...
```

This list is the single input to the Stage 2 design document. Stage 2 does not introspect `custom/` or the full Stage 1 codebase — it only picks kernels from this list.

### 10.4 `CubeBatch` Field Freeze

`CubeBatch` dataclass field list is finalized at end of Stage 1. Stage 2 may reorder memory layout or change internal dtypes (e.g., `int32 → int64` for indices exceeding `2^31`), but **no field additions, removals, or renames**. This guarantees Stage 2 Triton kernels can plug into the existing pipeline without upstream/downstream disruption.

---

## 11. Dependencies

### 11.1 New Dependencies

- `torch_scatter` — segmented reductions (`segment_sum`, `segment_min`, `segment_max`, `segment_softmax`)
- No other new dependencies

### 11.2 Existing Dependencies Used

- `torch` (already present)
- `numpy` (already present)
- `scipy.optimize.linear_sum_assignment` (already present via `custom/collapse_point.py`)
- `trimesh` (already present; for mesh I/O only — not for math)
- `orjson` (already present; for JSON I/O)
- `tqdm` (already present)
- `pytest` (already present)

### 11.3 Explicitly Not Added

- `triton` — deferred to Stage 2
- `cupy` — not needed
- `pybind11` — not needed
- `torch_geometric` — not needed (we only need `torch_scatter`, not the full PyG stack)
- `torch_linear_assignment` — deferred to Stage 2

---

## 12. Validation Plan

### 12.1 Unit Tests

Each geometric kernel and stage gets its own unit test:

- `tests/unit/test_geometry_sat.py`: triangle-AABB SAT against 100+ hand-crafted cases (triangles fully inside, fully outside, edge-crossing, face-crossing)
- `tests/unit/test_geometry_moller_trumbore.py`: ray-triangle with known hit/miss expected outcomes
- `tests/unit/test_geometry_sutherland_hodgman.py`: polygon clipping against planes, verifying vertex count and area preservation
- `tests/unit/test_s{N}.py` (for each stage): tests using small synthetic inputs where the expected output can be computed by hand

### 12.2 Integration Tests

- `tests/integration/test_full_pipeline_small.py`: run the full 8-stage pipeline on a 64³ icosphere, verify output is a valid PLY file with reasonable vertex/face counts

### 12.3 Regression Tests

See §7.5 — 5 representative meshes, 100% topology equivalence required, 2× minimum speedup required.

### 12.4 Baseline Evaluation Set A/B

See §7.4 — final exit criterion. 100% equivalence on 14 meshes, order-of-magnitude speedup.

---

## 13. Open Questions / Risks

### 13.1 Uncertainties to Resolve Before Implementation

1. **Baseline eval set composition**: The memory record lists 7 models × 2 resolutions, but §7.5 needs the actual file list to finalize the regression fixtures. Verify from `results/baseline_experiments/` at implementation time.
2. **`torch_scatter` version compatibility**: Must be compatible with the project's existing `torch` version (likely 2.4+ based on `flex_gemm` usage). Pip install must not break existing env.
3. **s6 BUDGET_EXCEEDED rate on real meshes**: Unknown until Step 0 baseline profiling runs. If >5% of cubes hit the budget, the s6 algebraic pruning in §6.6.a needs to be stronger (possibly with additional boundary-count constraints).
4. **Peak GPU memory for s1 chunk sizing**: Need empirical tuning. Default 512 MB chunk budget is a starting guess.

### 13.2 Known Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| s6 cross-cube bucket dispatch doesn't scale to very large clusters of budget-exceeded cubes | Medium | Medium | Deferred to s8 exception handling — matches `custom/` behavior |
| Topology equivalence checker has false positives on canonical-form bugs (edge sequence difference vs real divergence) | Medium | Low | Stage isolation debug (§7.2) disambiguates; iteratively refine canonicalization |
| Torch GPU float32 noise breaks L5 point coordinate check in rare cases | Low | Low | Tolerance already loose (`rtol=1e-4`); tighten later if real bugs surface |
| `torch_scatter` has a breaking ABI change | Low | High | Pin version in requirements-corep_fast.txt |
| s7 CPU-GPU Hungarian roundtrip is slow for cubes with many loops | Low | Medium | Profile first; if >5% of total time, defer to Stage 2 batched GPU Hungarian |
| Memory budget on s6 worst-case cube still exceeds GPU capacity even with bucketing | Low | Medium | Hard protection marks as BUDGET_EXCEEDED, propagates to s8 exception path |
| Profiling data surprise: s5 or s7 turn out to be the biggest hotspot | Medium | Low | Profiling-First sequencing (§7.7) accommodates this — implementation order is chosen after baseline data |

---

## 14. Appendix

### A. CoReP Constants Reference (Excerpted from `custom/ARCHITECTURE.md`)

```
CUBE_VERTICES (8):
  0: (0, 0, 0)     1: (1, 0, 0)     2: (1, 1, 0)     3: (0, 1, 0)
  4: (0, 0, 1)     5: (1, 0, 1)     6: (1, 1, 1)     7: (0, 1, 1)

CUBE_EDGES (18 total = 12 axis-aligned + 6 face diagonals):
  0:(0,1) 1:(1,2) 2:(2,3) 3:(3,0)          ← bottom face axis edges, share factor 4
  4:(4,5) 5:(5,6) 6:(6,7) 7:(7,4)          ← top face axis edges, share factor 4
  8:(0,4) 9:(1,5) 10:(2,6) 11:(3,7)        ← vertical axis edges, share factor 4
  12:(0,2) 13:(4,6)                         ← bottom/top diagonals, share factor 2
  14:(1,4) 15:(1,6) 16:(2,7) 17:(0,7)      ← face diagonals, share factor 2

CUBE_TRIANGULATED_FACETS (12):
  T0:  (0, 1, 12)    T1:  (2, 3, 12)   ← bottom face halves
  T2:  (4, 5, 13)    T3:  (6, 7, 13)   ← top face halves
  T4:  (0, 8, 14)    T5:  (4, 9, 14)   ← front face halves
  T6:  (1, 10, 15)   T7:  (5, 9, 15)   ← right face halves
  T8:  (2, 11, 16)   T9:  (6, 10, 16)  ← back face halves
  T10: (3, 11, 17)   T11: (7, 8, 17)   ← left face halves

PER_CUBE_UNIQUE_DIMENSIONS (after cross-cube deduplication):
  Edge data:       18 local → 6 unique (3 axis + 3 diagonal)
  Face weights:    12 local → 6 unique (each triangular facet shared by 2 cubes)
  num_components:  1 (per-cube)
  num_boundary:    1 (per-cube)
  component_points: variable (per-cube, typically 1)
  Total minimum:   ~17 int + n×3 float per cube
```

### B. Sanity Check Benchmark Values

(Placeholder — to be filled after Step 0 baseline profiling)

| Mesh | Resolution | Active Cubes | custom/ total wall (s) | Stage breakdown |
|------|-----------|-------------:|-----------------------:|-----------------|
| sphere | 256 | TBD | TBD | TBD |
| open_plane | 256 | TBD | TBD | TBD |
| triple_layer_sphere | 256 | TBD | TBD | TBD |
| turbine | 512 | TBD | TBD | TBD |
| cloth_folded | 256 | TBD | TBD | TBD |

### C. Topology Equivalence Canonical Forms

**Loop canonical form:**
1. Rotate so that the minimum edge index is at position 0
2. If the reversed sequence is lexicographically smaller than the forward sequence, use the reversed sequence
3. The resulting tuple is the canonical form; two loops are equivalent iff their canonical tuples are equal

**Loop set canonical form:**
1. Compute canonical form of each loop
2. Sort the canonical loops lexicographically
3. The resulting tuple of tuples is the loop set canonical form

**Loop-to-point matching canonical form:**
1. Use the loop's canonical form index within the sorted loop set
2. Pair with the matched point's index within the cube's point list (custom/ uses insertion order; corep_fast must preserve this)

### D. Interop Bridge Contract

```python
# corep_fast/interop/from_custom.py

def cube_batch_from_custom(
    face_registers: list[dict],   # output of any custom/ stage
    mesh: MeshTensors,
    include: set[str] = None,     # None = all; else subset of field names to populate
    device: torch.device = None,
) -> CubeBatch:
    """
    Convert custom/'s list-of-dict representation into a CubeBatch.

    Required input dict fields: 'cube_indices', 'face_indices' (or 'edge_indices' for boundary)
    Optional input dict fields (populated if present):
        'num_components', 'num_boundary', 'edge_weights', 'face_weights',
        'component_points', 'loops', 'sorted_loops', 'status'

    If `include` is provided, only those fields are populated; others are left as empty tensors.
    Missing fields are filled with sentinels (-1 for int, NaN for float) so downstream
    stages can detect and fail fast if they try to read unpopulated data.
    """
    ...


# corep_fast/interop/to_custom.py

def custom_from_cube_batch(
    cube: CubeBatch,
    mesh: MeshTensors,
) -> list[dict]:
    """
    Convert a CubeBatch back into custom/'s list-of-dict format.

    Used for:
    1. Partial A/B runs where custom/ consumes a corep_fast stage output
    2. Dumping CubeBatch snapshots for offline analysis in existing custom/ tooling
    """
    ...
```

---

## 15. Document Status

- **Version**: 1.0
- **Status**: Draft, awaiting user review
- **Author**: Designed in brainstorm session with user, 2026-04-15
- **Next step**: User reviews this document, then `writing-plans` skill is invoked to produce the implementation plan
