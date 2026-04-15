# CoReP-Fast Stage 2: Triton Kernel Acceleration

> Date: 2026-04-15
> Prerequisite: Stage 1 (Torch vectorization) complete for s8_collapse
> Goal: 5x+ further speedup for the s8_collapse stage via Triton GPU kernels, eliminating the Python geometry processing bottleneck

---

## 0. Summary

Stage 1 delivered a 2.1x speedup on s8_collapse by vectorizing vertex welding (`_weld_and_dedup`) with `torch.unique` and `scatter_reduce_`. However, **70% of the remaining s8 time** is spent in `_process_shared_edge_geometry` — a pure Python function called 263K times (at resolution 256), each processing a 2×2 grid of cubes around a shared edge. This function cannot be further optimized in Python.

Stage 2 replaces the Python geometry processing loop with a **Triton kernel** that processes all shared edges in a single GPU launch. The kernel fuses:
1. Edge-cube neighborhood lookup (CSR gather)
2. Loop/rank extraction per neighbor cube
3. Rank normalization + grouping
4. Projection point computation (mean)
5. Fan triangle emission

The vertex welding step (`torch.unique`) remains unchanged — it's already near-optimal.

### 0.1 Scope Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What to accelerate | s8 geometry processing loop only | 70% of s8 time; other stages still use custom/ |
| Kernel scope | Single fused Triton kernel for shared-edge geometry | Avoids kernel launch overhead; all data stays on GPU |
| Input format | CubeBatch CSR tensors (not list-of-dict) | Triton can't operate on Python dicts |
| Fallback | Python path remains for correctness validation | A/B testing against Stage 1 path |
| Branch | `triton-s8` off `siyuan-dev` | Isolated testing, no impact on main branch |

### 0.2 Non-goals

1. Triton kernels for stages s1-s7 (future work, separate spec)
2. Cross-mesh batching
3. Replacing `torch.unique` for vertex welding (already fast)
4. Multi-node distribution

---

## 1. Current Bottleneck Analysis

### 1.1 s8_collapse Time Breakdown (resolution=256, icosphere subdiv=3)

| Component | Time (s) | % of s8 |
|-----------|----------|---------|
| Build cube_map (Python dict) | 0.95 | 2.8% |
| Edge iteration + _process_shared_edge_geometry | 27.5 | 82.2% |
| np.concatenate | 0.02 | 0.1% |
| numpy→torch | 0.00 | 0.0% |
| torch.unique (vertex welding) | 3.5 | 10.5% |
| scatter_reduce_ (first-occur) | 0.1 | 0.3% |
| canonical + face dedup | 1.4 | 4.2% |
| **Total** | **33.5** | **100%** |

### 1.2 Why Python is the Bottleneck

`_process_shared_edge_geometry` is called 263K times at resolution 256. Each call:
- Takes a 2×2 grid of cube dicts
- Iterates over loops within each cube
- Extracts ranks, normalizes them
- Groups points by rank
- Detects disconnected components and promotes to exceptions
- Emits 4 fan triangles per complete rank group

The per-call overhead is ~17μs — dominated by Python dict operations, list comprehensions, and `min()`/`all()`/`max()` calls. This cannot be vectorized in Python.

### 1.3 What the Triton Kernel Must Do

For each **owned shared edge** (there are ~264K at res=256):
1. Look up the 4 neighboring cubes (or fewer if at boundary)
2. For each neighbor, find the local edge index for this shared edge
3. For each neighbor's loops, check if any crosses the local edge
4. If a crossing exists, extract the rank and component_point
5. Normalize ranks (flip for edges 2,6,3,7)
6. Group crossing points by normalized rank
7. Compute projection point = mean of up to 4 points
8. If all 4 neighbors contribute to a rank group, emit 4 fan triangles
9. Handle exception cubes (wildcard point injection)

---

## 2. Data Preparation: CubeBatch → Edge-Centric Tensors

The Triton kernel cannot operate on Python list-of-dict. We need a **pre-processing step** that converts the s7 output into edge-centric GPU tensors.

### 2.1 Required Tensors

```python
# Input: prepared by a Torch pre-processing function
class EdgeBatchData:
    # Edge identity
    num_owned_edges: int                      # E
    edge_axis: torch.Tensor                   # (E,) int32 — 0=X, 1=Y, 2=Z
    
    # Neighbor mapping: for each owned edge, 4 slots for neighbor cubes
    neighbor_cube_idx: torch.Tensor           # (E, 4) int32 — -1 if absent
    neighbor_local_edge: torch.Tensor         # (E, 4) int32 — 0..11, -1 if absent
    neighbor_count: torch.Tensor              # (E,) int32 — 1..4
    
    # Per-cube data (flattened across all cubes)
    cube_edge_weights: torch.Tensor           # (N, 18) int32
    cube_exception: torch.Tensor              # (N,) bool
    cube_num_components: torch.Tensor         # (N,) int32
    
    # Per-cube loop data (CSR)
    loop_cube_offsets: torch.Tensor           # (N+1,) int32
    loop_edges: torch.Tensor                  # (L_total, max_edges_per_loop) int32, padded
    loop_ranks: torch.Tensor                  # (L_total, max_edges_per_loop) int32, padded
    loop_component_points: torch.Tensor       # (L_total, 3) float32
    
    # Output buffers (pre-allocated)
    out_tri_verts: torch.Tensor               # (E * 4 * 3, 3) float32 — worst case
    out_tri_count: torch.Tensor               # (E,) int32 — actual tris emitted
```

### 2.2 Pre-processing Function (Torch, not Triton)

```python
def prepare_edge_batch(
    cube_data_list: list[dict],
    resolution: int,
) -> EdgeBatchData:
    """Convert list-of-dict to edge-centric GPU tensors.
    
    This runs once per mesh, on CPU+GPU. It replaces:
    - cube_map construction
    - edge iteration loop
    - active_neighbors / ownership check
    
    The result is a compact set of tensors ready for the Triton kernel.
    """
```

This pre-processing step itself can be partially vectorized using the existing `compute_global_edge_keys`, `enumerate_unique_edges`, and `compute_edge_ownership` functions from Phase 1a.

---

## 3. Triton Kernel Design

### 3.1 Kernel Signature

```python
@triton.jit
def process_shared_edges_kernel(
    # Edge data
    edge_axis_ptr, neighbor_cube_idx_ptr, neighbor_local_edge_ptr,
    neighbor_count_ptr,
    # Cube data 
    cube_edge_weights_ptr, cube_exception_ptr, cube_num_components_ptr,
    # Loop CSR data
    loop_cube_offsets_ptr, loop_edges_ptr, loop_ranks_ptr,
    loop_component_points_ptr,
    # Output
    out_tri_verts_ptr, out_tri_count_ptr,
    # Constants
    num_edges: tl.constexpr,
    max_edges_per_loop: tl.constexpr,
    MAX_RANK: tl.constexpr,  # max possible edge weight (typically 8)
):
    """Process one shared edge per Triton program instance."""
    edge_id = tl.program_id(0)
    if edge_id >= num_edges:
        return
    
    # 1. Load edge metadata
    axis = tl.load(edge_axis_ptr + edge_id)
    n_count = tl.load(neighbor_count_ptr + edge_id)
    
    # 2. For each neighbor slot (0..3), load cube data
    # ... load neighbor_cube_idx, neighbor_local_edge
    
    # 3. For each neighbor cube, iterate over its loops
    #    and find crossings at the local edge
    # ... rank extraction, normalization
    
    # 4. Group points by normalized rank
    # ... fixed-size array (MAX_RANK, 4, 3)
    
    # 5. For each rank group with 4 points, emit 4 fan triangles
    # ... atomic write to out_tri_verts
```

### 3.2 Thread Mapping

- **Grid**: `(num_owned_edges,)` — one Triton program per owned edge
- **Block size**: Not applicable (no inner parallel dimension within a single edge)
- The kernel is embarrassingly parallel across edges

### 3.3 Memory Access Pattern

- **Reads**: edge metadata (coalesced), cube data (scattered by neighbor_cube_idx), loop CSR (scattered)
- **Writes**: triangle vertices (scatter to pre-allocated output buffer)
- **Shared memory**: Not needed — each program instance is independent

### 3.4 Key Implementation Challenges

1. **Variable loop count per cube**: Each cube has 0..N loops, each with 0..K edges. The kernel must handle this with dynamic bounds.

2. **Rank grouping**: For each edge, we need to collect points by normalized rank across up to 4 cubes. With MAX_RANK=8, this requires a (8, 4, 3) working array per program.

3. **Exception handling**: Exception cubes inject their component_point as a wildcard. The kernel must detect exceptions and apply the wildcard injection logic.

4. **Conditional promotion**: When all 4 cubes have components but none connect properly, the kernel must detect this and promote cubes to exception status. This is complex logic that may need a two-pass approach.

5. **Output compaction**: Each edge emits 0..4*MAX_RANK triangles. We need an efficient compaction strategy (atomic counter or prefix sum scan).

### 3.5 Two-Pass Strategy

Given the complexity of conditional promotion and variable output sizes:

**Pass 1: Count + classify** (Triton kernel 1)
- For each edge, determine how many triangles it will emit
- Classify edges: trivial (0 tris), simple (4 tris at rank 0), complex (multiple ranks)
- Output: `tri_count_per_edge: (E,)` and `edge_classification: (E,)`

**Pass 2: Emit triangles** (Triton kernel 2)
- Prefix sum on `tri_count_per_edge` gives output offsets
- Each edge writes its triangles to the correct output location
- No atomics needed (deterministic offset)

### 3.6 Fallback for Complex Cases

Edges requiring conditional promotion (§3.4.4) are rare (<5% of edges). These can be:
- Handled by a separate simpler Python/Torch path
- Or handled within the Triton kernel with full conditional logic

Recommendation: Handle all cases in Triton, with the conditional promotion logic as a series of if/else branches within the kernel.

---

## 4. Expected Performance

### 4.1 Theoretical Analysis

At resolution 256 with 264K owned edges:
- Current Python: 27.5s (104μs per edge on average)
- Triton kernel: Each edge processes ~4 cubes × ~5 loops × ~10 ops → ~200 FLOPs per edge
- At 264K edges × 200 FLOPs = 53M FLOPs — trivial for modern GPU
- Memory-bound: ~264K × (4 × 128 bytes cube data + 4 × 64 bytes loop data) ≈ 200 MB reads
- A100 memory BW: 2 TB/s → theoretical: 0.1ms

Real-world overhead (kernel launch, scattered memory access, branch divergence) will make this 10-100× slower than theoretical. **Conservative target: 0.5-2s at resolution 256**, representing a **15-50× speedup** over current Python path.

### 4.2 Combined with Existing Optimizations

| Component | Stage 1 (current) | Stage 2 (target) |
|-----------|-------------------|-------------------|
| Geometry processing | 27.5s (Python) | 0.5-2s (Triton) |
| Vertex welding (torch.unique) | 3.5s | 3.5s (unchanged) |
| Face dedup | 1.4s | 1.4s (unchanged) |
| Other overhead | 1.1s | ~0.5s (Torch prep) |
| **Total s8** | **33.5s** | **5.9-7.4s** |
| **Speedup vs custom/** | 2.1x | 10-12x |

### 4.3 Full Pipeline Impact

At resolution 256, the full pipeline takes 112.5s:
- s8 drops from 33.5s to ~6.5s → saves 27s
- Pipeline total: 112.5 - 27 = **~86s** (1.3x faster overall)
- s8 drops from 27.8% to ~7.5% of pipeline

To achieve the user's target of 5x+ total speedup, Triton kernels for s3-s7 would also be needed (future work beyond this spec).

---

## 5. Implementation Plan

### Phase 2a: Pre-processing + Simple Triton Kernel

1. **Implement `prepare_edge_batch`**: Convert list-of-dict to edge-centric tensors using existing Torch edge enumeration functions
2. **Write Pass 1 Triton kernel**: Count triangles per edge (no actual geometry, just classification)
3. **Write Pass 2 Triton kernel**: Emit fan triangles (simple case: no conditional promotion)
4. **A/B validation**: Compare output against Stage 1 Python path
5. **Benchmark**: Measure speedup at resolution 128, 256, 512

### Phase 2b: Full Triton Kernel with Exception Handling

6. **Add conditional promotion logic** to the Triton kernel
7. **Add exception cube wildcard injection**
8. **Full A/B validation** including exception-heavy meshes
9. **Performance tuning**: Memory coalescing, block size tuning, profiling with `triton.testing.perf_report`

### Phase 2c: Integration + Cleanup

10. **Wire into pipeline.py**: New `BackendConfig` option for Triton s8
11. **Regression testing**: Full test suite with Triton path
12. **Performance report**: Final speedup numbers across baseline eval set

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Conditional promotion too complex for Triton | Medium | High | Two-pass: simple edges in Triton, complex edges fallback to Python |
| Branch divergence kills GPU utilization | Low | Medium | Edges are fairly uniform — same algorithm for all |
| Memory access pattern too scattered | Medium | Medium | Sort edges by cube locality before kernel launch |
| Correctness bugs in rank normalization | Medium | High | Exhaustive A/B testing at multiple resolutions |
| Triton compilation time | Low | Low | Cache compiled kernels |

---

## 7. Dependencies

- Triton (already available in the environment via PyTorch)
- All Phase 1a infrastructure (CubeBatch, profiling harness, A/B rig, topology equivalence)
- Stage 1 s8_collapse.py as the fallback/reference implementation
