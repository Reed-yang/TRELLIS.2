# CoReP-Fast Stage 2 v2 Implementation Plan: Full Torch Vectorization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all Python loop bottlenecks in s8_collapse. Target: 25x+ total speedup vs custom/.

**Architecture:** Convert list-of-dict cube data to dense GPU tensors once (upfront), then vectorize edge enumeration and geometry processing entirely on GPU. Keep Python path as correctness validator.

**Tech Stack:** Python 3.10+, Torch 2.x + CUDA, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-04-15-corep-fast-stage2-v2-torch-vectorization.md`

**Prerequisite:** Stage 1 complete (s8 = 13.9s at res=256, 5.0x vs custom). Current branch: `triton-s8`.

---

## File Structure

```
corep_fast/
├── stages/
│   └── s8_collapse.py               # Modify: add vectorized path
└── tests/unit/
    └── test_s8_collapse.py          # Modify: add tests for new functions
```

New functions added to `s8_collapse.py`:
- `_cube_data_to_tensors(cube_data_list, device) -> CubeDataTensors` (Task 2)
- `_fix_build_edge_neighbor_table` (Task 1 — replaces existing function)
- `_filter_table_by_mask(table, mask) -> EdgeNeighborTable` (Task 3)
- `process_geometry_vectorized(table, tensors, device) -> torch.Tensor` (Task 5-7)
- `process_shared_edges_batch_torch(...)` (Task 8 — new vectorized entry)
- Modified `process_shared_edges_batch(...)` dispatches to torch or python path (Task 9)

---

## Task 1: Vectorize `build_edge_neighbor_table`

**Purpose:** Eliminate the Python for-loop (line 301-327 in s8_collapse.py) that iterates over E unique edges.

**Files:**
- Modify: `corep_fast/stages/s8_collapse.py:237-335` (build_edge_neighbor_table function)
- Modify: `corep_fast/tests/unit/test_s8_collapse.py` (existing tests for this function)

- [ ] **Step 1: Run existing tests to establish baseline**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestBuildEdgeNeighborTable -xvs
```

Expected: PASS (existing tests).

- [ ] **Step 2: Add a new test for larger input to verify no Python scaling issue**

Append to `corep_fast/tests/unit/test_s8_collapse.py` (in TestBuildEdgeNeighborTable class):
```python
    def test_large_input_performance(self):
        """Verify vectorized path handles 10K cubes quickly (< 1s)."""
        import time
        N = 10000
        # Random cube indices in [0, 100)^3, deduplicated
        torch.manual_seed(42)
        candidates = torch.randint(0, 100, (N*2, 3), dtype=torch.int32)
        unique, _ = torch.unique(candidates, dim=0, return_inverse=True)
        cube_indices = unique[:N]
        
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, 128)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        
        t0 = time.time()
        table = build_edge_neighbor_table(
            edge_id_per_entry, cube_ids, local_ids, unique_keys.shape[0])
        elapsed = time.time() - t0
        # After vectorization this should be <100ms; before was ~2s
        assert elapsed < 1.0, f"build_edge_neighbor_table too slow: {elapsed:.2f}s"
        # Sanity: at least some edges have 2+ neighbors
        assert (table.neighbor_counts >= 2).sum() > 0
```

- [ ] **Step 3: Run the new test (should fail with old implementation on large inputs)**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestBuildEdgeNeighborTable::test_large_input_performance -xvs
```

Expected: passes slowly (~2s for 10K cubes) with old code, or fails the timing bound.

- [ ] **Step 4: Replace `build_edge_neighbor_table` with vectorized implementation**

In `corep_fast/stages/s8_collapse.py`, replace the function body (keeping signature and docstring):

```python
def build_edge_neighbor_table(
    edge_id_per_entry: torch.Tensor,
    cube_ids: torch.Tensor,
    local_ids: torch.Tensor,
    num_unique_edges: int,
) -> EdgeNeighborTable:
    """(docstring unchanged)"""
    device = edge_id_per_entry.device
    E = num_unique_edges
    M = edge_id_per_entry.shape[0]

    if M == 0 or E == 0:
        # Edge case: empty input
        return EdgeNeighborTable(
            neighbor_counts=torch.zeros(E, dtype=torch.int32, device=device),
            neighbor_cube_ids=torch.full((E, 4), -1, dtype=torch.int32, device=device),
            neighbor_positions=torch.full((E, 4), -1, dtype=torch.int32, device=device),
            neighbor_local_edges=torch.full((E, 4), -1, dtype=torch.int32, device=device),
            edge_axes=torch.zeros(E, dtype=torch.int32, device=device),
        )

    # Sort entries by edge_id to group them (stable sort preserves insertion order)
    sort_idx = torch.argsort(edge_id_per_entry, stable=True)
    sorted_edge_ids = edge_id_per_entry[sort_idx]
    sorted_cube_ids = cube_ids[sort_idx]
    sorted_local_ids = local_ids[sort_idx]

    # Count entries per edge
    neighbor_counts = torch.zeros(E, dtype=torch.int32, device=device)
    ones = torch.ones(M, dtype=torch.int32, device=device)
    neighbor_counts.scatter_add_(0, sorted_edge_ids.to(torch.int64), ones)

    # CSR offsets: offsets[e] = index of first entry with edge_id == e in sorted order
    offsets = torch.zeros(E + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(neighbor_counts.to(torch.int64), dim=0)

    # For each entry in sorted order, compute its slot within the edge group
    entry_pos = torch.arange(M, dtype=torch.int64, device=device)
    slot_per_entry = entry_pos - offsets[sorted_edge_ids.to(torch.int64)]
    # Keep only entries whose slot < 4 (valid within the padded table)
    valid = slot_per_entry < 4

    # Scatter into (E, 4) tables
    neighbor_cube_ids = torch.full((E, 4), -1, dtype=torch.int32, device=device)
    neighbor_local_edges = torch.full((E, 4), -1, dtype=torch.int32, device=device)

    v_edge = sorted_edge_ids[valid].to(torch.int64)
    v_slot = slot_per_entry[valid]
    flat_idx = v_edge * 4 + v_slot
    # scatter_ writes values at flat_idx positions
    neighbor_cube_ids.view(-1).scatter_(0, flat_idx, sorted_cube_ids[valid])
    neighbor_local_edges.view(-1).scatter_(0, flat_idx, sorted_local_ids[valid])

    # Derive edge axes from the first entry of each group (via _EDGE_OFFSET_TABLE)
    first_entry_pos = offsets[:-1].clamp(max=M - 1)
    first_local_edges = sorted_local_ids[first_entry_pos]
    edge_offset = _EDGE_OFFSET_TABLE.to(device)
    edge_axes = edge_offset[first_local_edges.to(torch.int64), 0].to(torch.int32)

    # Compute neighbor_positions using _REVERSE_LOCAL_EDGE lookup
    rev = _REVERSE_LOCAL_EDGE.to(device)  # (3, 4)
    axis_per_slot = edge_axes.unsqueeze(1).expand(E, 4).to(torch.int64)
    local_per_slot = neighbor_local_edges.to(torch.int64)
    # (E, 4, 4) candidates: for each (edge, slot), the 4 local edge values at that axis
    candidates = rev[axis_per_slot]
    matches = candidates == local_per_slot.unsqueeze(2)
    positions = matches.to(torch.int32).argmax(dim=2)
    positions = torch.where(
        neighbor_local_edges >= 0,
        positions,
        torch.full_like(positions, -1),
    )

    return EdgeNeighborTable(
        neighbor_counts=neighbor_counts,
        neighbor_cube_ids=neighbor_cube_ids,
        neighbor_positions=positions,
        neighbor_local_edges=neighbor_local_edges,
        edge_axes=edge_axes,
    )
```

- [ ] **Step 5: Run all build_edge_neighbor_table tests**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  .venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestBuildEdgeNeighborTable corep_fast/tests/unit/test_s8_collapse.py::TestComputeEdgeOwnership -xvs
```

Expected: ALL pass. The performance test should now complete in <100ms.

- [ ] **Step 6: Commit**

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "perf(corep_fast): vectorize build_edge_neighbor_table — eliminate E-loop

Replace Python for-loop over unique edges with scatter-based tensor ops.
Expected speedup at res=256: ~7s → <100ms.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Implement `_cube_data_to_tensors`

**Purpose:** Convert list-of-dict cube data to dense GPU tensors for vectorized processing.

**Files:**
- Modify: `corep_fast/stages/s8_collapse.py` (add new function + dataclass)
- Modify: `corep_fast/tests/unit/test_s8_collapse.py` (add tests)

- [ ] **Step 1: Write tests first**

Append to `corep_fast/tests/unit/test_s8_collapse.py`:
```python
from corep_fast.stages.s8_collapse import _cube_data_to_tensors, CubeDataTensors


class TestCubeDataToTensors:
    def test_empty_input(self):
        tensors = _cube_data_to_tensors([], device=torch.device('cpu'))
        assert tensors.cube_indices.shape == (0, 3)
        assert tensors.loop_cube_offsets.shape == (1,)
        assert tensors.loop_cube_offsets[0].item() == 0

    def test_single_cube_no_loops(self):
        data = [{
            'cube_indices': (1, 2, 3),
            'sorted_loops': [],
            'edge_weights': [0]*18,
            'exception': False,
            'num_components': 0,
        }]
        t = _cube_data_to_tensors(data, device=torch.device('cpu'))
        assert t.cube_indices.tolist() == [[1, 2, 3]]
        assert t.cube_exception.tolist() == [False]
        assert t.cube_num_components.tolist() == [0]
        assert t.loop_cube_offsets.tolist() == [0, 0]
        # No loops -> no loop entries
        assert t.loop_component_point.shape[0] == 0

    def test_cube_with_loops(self):
        data = [{
            'cube_indices': (5, 5, 5),
            'sorted_loops': [
                {'loop': [3, 12, 1], 'rank': [0, -1, -1],
                 'component_point': [0.5, 0.5, 0.5]},
                {'loop': [4, 6], 'rank': [1, 2],
                 'component_point': [0.6, 0.6, 0.6]},
            ],
            'edge_weights': [0]*3 + [1] + [0]*14,
            'exception': False,
            'num_components': 2,
        }]
        t = _cube_data_to_tensors(data, device=torch.device('cpu'))
        assert t.loop_cube_offsets.tolist() == [0, 2]
        assert t.loop_component_point.shape == (2, 3)
        # max loop length across all loops is 3
        assert t.max_loop_len >= 3
        # Check loop 0: edges [3, 12, 1], ranks [0, -1, -1]
        K = t.max_loop_len
        loop0_edges = t.loop_edges_flat[:K].tolist()
        loop0_ranks = t.loop_ranks_flat[:K].tolist()
        assert loop0_edges[:3] == [3, 12, 1]
        assert loop0_ranks[:3] == [0, -1, -1]
        # Padding beyond length should be -1
        if K > 3:
            assert loop0_edges[3] == -1

    def test_exception_cube(self):
        data = [{
            'cube_indices': (0, 0, 0),
            'sorted_loops': [{'component_point': [0.1, 0.2, 0.3]}],
            'edge_weights': [0]*18,
            'exception': True,
            'num_components': 1,
        }]
        t = _cube_data_to_tensors(data, device=torch.device('cpu'))
        assert t.cube_exception.tolist() == [True]
        assert t.loop_component_point.shape == (1, 3)
        assert torch.allclose(t.loop_component_point[0], torch.tensor([0.1, 0.2, 0.3]))
```

Run to confirm failure:
```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestCubeDataToTensors -xvs
```

Expected: FAIL with `ImportError: cannot import name '_cube_data_to_tensors'`.

- [ ] **Step 2: Implement `_cube_data_to_tensors` and `CubeDataTensors`**

Add to `corep_fast/stages/s8_collapse.py` (before `process_shared_edges_batch`):

```python
# ---------------------------------------------------------------------------
# Tensorization: convert list-of-dict cube data to GPU tensors
# ---------------------------------------------------------------------------

@dataclass
class CubeDataTensors:
    """Dense tensor representation of cube_data_list for vectorized processing.
    
    All per-cube and per-loop data is flattened into tensors with CSR offsets
    for ragged loop arrays.
    """
    # Per-cube fields
    cube_indices: torch.Tensor        # (N, 3) int32
    cube_edge_weights: torch.Tensor   # (N, 18) int32
    cube_exception: torch.Tensor      # (N,) bool
    cube_num_components: torch.Tensor # (N,) int32

    # CSR offsets for loops (one range per cube)
    loop_cube_offsets: torch.Tensor   # (N+1,) int64, cumulative loop count

    # Per-loop fields (total L loops)
    loop_component_point: torch.Tensor # (L, 3) float32

    # Padded loop edges and ranks: (L, max_loop_len) with -1 padding
    max_loop_len: int
    loop_edges_flat: torch.Tensor     # (L * max_loop_len,) int32, -1 padding
    loop_ranks_flat: torch.Tensor     # (L * max_loop_len,) int32, -1 padding


def _cube_data_to_tensors(
    cube_data_list: list[dict],
    device: torch.device,
) -> CubeDataTensors:
    """Convert list-of-dict cube data to dense GPU tensors.
    
    This is a one-time conversion at the start of s8. After this, all processing
    operates on tensors.
    """
    N = len(cube_data_list)
    if N == 0:
        return CubeDataTensors(
            cube_indices=torch.zeros((0, 3), dtype=torch.int32, device=device),
            cube_edge_weights=torch.zeros((0, 18), dtype=torch.int32, device=device),
            cube_exception=torch.zeros((0,), dtype=torch.bool, device=device),
            cube_num_components=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_cube_offsets=torch.zeros((1,), dtype=torch.int64, device=device),
            loop_component_point=torch.zeros((0, 3), dtype=torch.float32, device=device),
            max_loop_len=1,
            loop_edges_flat=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_ranks_flat=torch.zeros((0,), dtype=torch.int32, device=device),
        )
    
    # Collect per-cube data
    cube_indices_list = []
    cube_edge_weights_list = []
    cube_exception_list = []
    cube_num_components_list = []
    
    # Collect per-loop data
    loop_points_list = []
    loop_edges_nested: list[list[int]] = []
    loop_ranks_nested: list[list[int]] = []
    loops_per_cube = []  # list of ints
    
    max_loop_len = 1
    
    for data in cube_data_list:
        idx = data.get('cube_indices', (0, 0, 0))
        cube_indices_list.append(list(idx) if not isinstance(idx, (list, tuple))
                                 else [idx[0], idx[1], idx[2]])
        cube_edge_weights_list.append(list(data.get('edge_weights', [0] * 18))[:18] +
                                      [0] * (18 - len(data.get('edge_weights', []))))[:18]
        cube_exception_list.append(bool(data.get('exception', False)))
        cube_num_components_list.append(int(data.get('num_components', 0)))
        
        sorted_loops = data.get('sorted_loops', []) or []
        loops_per_cube.append(len(sorted_loops))
        for loop in sorted_loops:
            pt = loop.get('component_point', [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            loop_points_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            edges = list(loop.get('loop', []) or [])
            ranks = list(loop.get('rank', []) or [])
            # Align ranks length to edges length
            if len(ranks) < len(edges):
                ranks = ranks + [-1] * (len(edges) - len(ranks))
            elif len(ranks) > len(edges):
                ranks = ranks[:len(edges)]
            loop_edges_nested.append(edges)
            loop_ranks_nested.append(ranks)
            if len(edges) > max_loop_len:
                max_loop_len = len(edges)
    
    # Fix the slicing syntax error above: list comprehension bracket
    cube_edge_weights_list = []
    for data in cube_data_list:
        ew = list(data.get('edge_weights', [0] * 18) or [0] * 18)
        if len(ew) < 18:
            ew = ew + [0] * (18 - len(ew))
        else:
            ew = ew[:18]
        cube_edge_weights_list.append(ew)
    
    # Pad loop edges and ranks to max_loop_len
    L = len(loop_points_list)
    loop_edges_flat = np.full((L, max_loop_len), -1, dtype=np.int32)
    loop_ranks_flat = np.full((L, max_loop_len), -1, dtype=np.int32)
    for i, (edges, ranks) in enumerate(zip(loop_edges_nested, loop_ranks_nested)):
        k = min(len(edges), max_loop_len)
        loop_edges_flat[i, :k] = edges[:k]
        loop_ranks_flat[i, :k] = ranks[:k]
    
    # Build CSR offsets
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(loops_per_cube)
    
    return CubeDataTensors(
        cube_indices=torch.tensor(cube_indices_list, dtype=torch.int32, device=device),
        cube_edge_weights=torch.tensor(cube_edge_weights_list, dtype=torch.int32, device=device),
        cube_exception=torch.tensor(cube_exception_list, dtype=torch.bool, device=device),
        cube_num_components=torch.tensor(cube_num_components_list, dtype=torch.int32, device=device),
        loop_cube_offsets=torch.from_numpy(offsets).to(device),
        loop_component_point=torch.tensor(loop_points_list, dtype=torch.float32, device=device)
            if loop_points_list else torch.zeros((0, 3), dtype=torch.float32, device=device),
        max_loop_len=max_loop_len,
        loop_edges_flat=torch.from_numpy(loop_edges_flat.reshape(-1)).to(device)
            if L > 0 else torch.zeros((0,), dtype=torch.int32, device=device),
        loop_ranks_flat=torch.from_numpy(loop_ranks_flat.reshape(-1)).to(device)
            if L > 0 else torch.zeros((0,), dtype=torch.int32, device=device),
    )
```

**IMPORTANT:** The code above has a duplication bug — remove the first `cube_edge_weights_list.append(...)` in the loop (the one with the broken bracket). Keep only the cleaner second loop that builds `cube_edge_weights_list`. Also consolidate to a single pass. The final clean implementation should be:

```python
def _cube_data_to_tensors(
    cube_data_list: list[dict],
    device: torch.device,
) -> CubeDataTensors:
    """Convert list-of-dict cube data to dense GPU tensors."""
    N = len(cube_data_list)
    if N == 0:
        return CubeDataTensors(
            cube_indices=torch.zeros((0, 3), dtype=torch.int32, device=device),
            cube_edge_weights=torch.zeros((0, 18), dtype=torch.int32, device=device),
            cube_exception=torch.zeros((0,), dtype=torch.bool, device=device),
            cube_num_components=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_cube_offsets=torch.zeros((1,), dtype=torch.int64, device=device),
            loop_component_point=torch.zeros((0, 3), dtype=torch.float32, device=device),
            max_loop_len=1,
            loop_edges_flat=torch.zeros((0,), dtype=torch.int32, device=device),
            loop_ranks_flat=torch.zeros((0,), dtype=torch.int32, device=device),
        )
    
    cube_indices_list: list[list[int]] = []
    cube_edge_weights_list: list[list[int]] = []
    cube_exception_list: list[bool] = []
    cube_num_components_list: list[int] = []
    loop_points_list: list[list[float]] = []
    loop_edges_nested: list[list[int]] = []
    loop_ranks_nested: list[list[int]] = []
    loops_per_cube: list[int] = []
    max_loop_len = 1
    
    for data in cube_data_list:
        idx = data.get('cube_indices', (0, 0, 0))
        if isinstance(idx, (list, tuple)):
            cube_indices_list.append([int(idx[0]), int(idx[1]), int(idx[2])])
        else:
            cube_indices_list.append([0, 0, 0])
        
        ew = list(data.get('edge_weights', [0] * 18) or [0] * 18)
        if len(ew) < 18:
            ew = ew + [0] * (18 - len(ew))
        else:
            ew = ew[:18]
        cube_edge_weights_list.append([int(x) for x in ew])
        
        cube_exception_list.append(bool(data.get('exception', False)))
        cube_num_components_list.append(int(data.get('num_components', 0)))
        
        sorted_loops = data.get('sorted_loops', []) or []
        loops_per_cube.append(len(sorted_loops))
        for loop in sorted_loops:
            pt = loop.get('component_point', [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            loop_points_list.append([float(pt[0]), float(pt[1]), float(pt[2])])
            edges = list(loop.get('loop', []) or [])
            ranks = list(loop.get('rank', []) or [])
            if len(ranks) < len(edges):
                ranks = ranks + [-1] * (len(edges) - len(ranks))
            elif len(ranks) > len(edges):
                ranks = ranks[:len(edges)]
            loop_edges_nested.append([int(x) for x in edges])
            loop_ranks_nested.append([int(x) for x in ranks])
            if len(edges) > max_loop_len:
                max_loop_len = len(edges)
    
    L = len(loop_points_list)
    loop_edges_padded = np.full((L, max_loop_len), -1, dtype=np.int32)
    loop_ranks_padded = np.full((L, max_loop_len), -1, dtype=np.int32)
    for i, (edges, ranks) in enumerate(zip(loop_edges_nested, loop_ranks_nested)):
        k = min(len(edges), max_loop_len)
        loop_edges_padded[i, :k] = edges[:k]
        loop_ranks_padded[i, :k] = ranks[:k]
    
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(loops_per_cube)
    
    return CubeDataTensors(
        cube_indices=torch.tensor(cube_indices_list, dtype=torch.int32, device=device),
        cube_edge_weights=torch.tensor(cube_edge_weights_list, dtype=torch.int32, device=device),
        cube_exception=torch.tensor(cube_exception_list, dtype=torch.bool, device=device),
        cube_num_components=torch.tensor(cube_num_components_list, dtype=torch.int32, device=device),
        loop_cube_offsets=torch.from_numpy(offsets).to(device),
        loop_component_point=(
            torch.tensor(loop_points_list, dtype=torch.float32, device=device)
            if L > 0 else torch.zeros((0, 3), dtype=torch.float32, device=device)
        ),
        max_loop_len=max_loop_len,
        loop_edges_flat=(
            torch.from_numpy(loop_edges_padded.reshape(-1)).to(device)
            if L > 0 else torch.zeros((0,), dtype=torch.int32, device=device)
        ),
        loop_ranks_flat=(
            torch.from_numpy(loop_ranks_padded.reshape(-1)).to(device)
            if L > 0 else torch.zeros((0,), dtype=torch.int32, device=device)
        ),
    )
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestCubeDataToTensors -xvs
```

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): add _cube_data_to_tensors for vectorized s8 path

Converts list-of-dict cube data to dense GPU tensors with CSR offsets.
One-time upfront conversion enables fully vectorized processing.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Vectorized Geometry Processing — Simple Case

**Purpose:** Implement `process_geometry_vectorized` that handles the common case (no conditional promotion needed). Exception wildcards are supported. Complex edges fall back to Python.

**Files:**
- Modify: `corep_fast/stages/s8_collapse.py` (add new function)
- Modify: `corep_fast/tests/unit/test_s8_collapse.py` (add tests)

- [ ] **Step 1: Write test first — simple case with 4-cube edge**

Append to test file:
```python
from corep_fast.stages.s8_collapse import process_geometry_vectorized


class TestProcessGeometryVectorized:
    def test_four_cubes_produce_fan_triangles(self):
        """Same test case as TestProcessSharedEdgesBatch.test_four_cubes_produce_fan_triangles,
        but going through the vectorized path."""
        from corep_fast.stages.s8_collapse import (
            _cube_data_to_tensors,
            compute_global_edge_keys,
            enumerate_unique_edges,
            build_edge_neighbor_table,
            compute_edge_ownership,
        )
        device = torch.device('cpu')
        cube_data_list = [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False, 'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False, 'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False, 'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'loop': [3, 12, 1], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': False, 'num_components': 1,
            },
        ]
        tensors = _cube_data_to_tensors(cube_data_list, device)
        keys, cube_ids, local_ids = compute_global_edge_keys(tensors.cube_indices, 64)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        # Filter to edges with >=2 neighbors
        keep = table.neighbor_counts >= 2
        kept_table = EdgeNeighborTable(
            neighbor_counts=table.neighbor_counts[keep],
            neighbor_cube_ids=table.neighbor_cube_ids[keep],
            neighbor_positions=table.neighbor_positions[keep],
            neighbor_local_edges=table.neighbor_local_edges[keep],
            edge_axes=table.edge_axes[keep],
        )
        
        tri_verts = process_geometry_vectorized(kept_table, tensors, device)
        # Should produce fan triangles for the shared Y-axis edge
        # 4 fan triangles × 3 vertices = 12 rows
        assert tri_verts.shape[0] >= 12
        assert tri_verts.shape[1] == 3
```

- [ ] **Step 2: Implement `process_geometry_vectorized`**

Add to `corep_fast/stages/s8_collapse.py`:

```python
def process_geometry_vectorized(
    table: EdgeNeighborTable,
    tensors: CubeDataTensors,
    device: torch.device,
    max_rank: int = 16,
) -> torch.Tensor:
    """Vectorized geometry processing: emit fan triangles for all owned edges.
    
    Handles exception cube wildcards. Does NOT handle conditional promotion
    (complex edges are handled by the Python fallback path).
    
    Args:
        table: EdgeNeighborTable with edges filtered to neighbor_counts >= 2
        tensors: CubeDataTensors with cube and loop data
        device: torch device
        max_rank: max normalized rank to consider (bounded by edge weight)
    
    Returns:
        (T*3, 3) float64 tensor of triangle vertex coordinates.
    """
    E = table.neighbor_counts.shape[0]
    if E == 0 or tensors.cube_indices.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float64, device=device)
    
    # === Step 1: Gather neighbor info per edge ===
    n_cube = table.neighbor_cube_ids               # (E, 4) int32, -1 for missing
    n_loc = table.neighbor_local_edges             # (E, 4) int32
    valid_slot = n_cube >= 0                       # (E, 4)
    n_cube_safe = n_cube.clamp(min=0).to(torch.int64)  # for indexing
    
    # === Step 2: Gather exception + edge weight info ===
    cube_exc_flat = tensors.cube_exception         # (N,)
    cube_ew_flat = tensors.cube_edge_weights       # (N, 18)
    cube_exc = cube_exc_flat[n_cube_safe]          # (E, 4)
    cube_exc = cube_exc & valid_slot               # mask missing slots
    
    n_loc_safe = n_loc.clamp(min=0).to(torch.int64)  # (E, 4)
    cube_ew = cube_ew_flat[n_cube_safe]            # (E, 4, 18)
    W_at_edge = torch.gather(cube_ew, 2, n_loc_safe.unsqueeze(2)).squeeze(2)
    # W_at_edge: (E, 4)
    
    # === Step 3: Gather loop ranges per neighbor cube ===
    loop_start = tensors.loop_cube_offsets[n_cube_safe]     # (E, 4)
    loop_end = tensors.loop_cube_offsets[n_cube_safe + 1]   # (E, 4)
    num_loops = (loop_end - loop_start).clamp(min=0)        # (E, 4)
    max_loops = int(num_loops.max().item()) if E > 0 else 0
    
    if max_loops == 0:
        # No loops anywhere - handle exceptions only
        max_loops = 1  # so we can at least reference the first "loop" for exceptions
    
    # Expand to (E, 4, max_loops) flat loop indices
    loop_offset_range = torch.arange(max_loops, device=device).view(1, 1, max_loops)
    flat_loop_idx = loop_start.unsqueeze(2) + loop_offset_range  # (E, 4, max_loops)
    in_range = loop_offset_range < num_loops.unsqueeze(2)  # (E, 4, max_loops)
    flat_loop_idx_safe = flat_loop_idx.clamp(min=0, max=max(tensors.loop_component_point.shape[0] - 1, 0))
    
    # === Step 4: For each loop, gather edges and ranks ===
    K = tensors.max_loop_len
    # loop_edges_flat is (L*K,), reshape to (L, K)
    if tensors.loop_edges_flat.numel() > 0:
        loop_edges_2d = tensors.loop_edges_flat.view(-1, K)  # (L, K)
        loop_ranks_2d = tensors.loop_ranks_flat.view(-1, K)  # (L, K)
    else:
        loop_edges_2d = torch.zeros((1, K), dtype=torch.int32, device=device)
        loop_ranks_2d = torch.zeros((1, K), dtype=torch.int32, device=device)
    
    # Gather: (E, 4, max_loops, K)
    gathered_edges = loop_edges_2d[flat_loop_idx_safe]
    gathered_ranks = loop_ranks_2d[flat_loop_idx_safe]
    
    # Mask invalid loops
    gathered_edges = torch.where(
        in_range.unsqueeze(3),
        gathered_edges,
        torch.full_like(gathered_edges, -1),
    )
    gathered_ranks = torch.where(
        in_range.unsqueeze(3),
        gathered_ranks,
        torch.full_like(gathered_ranks, -1),
    )
    
    # === Step 5: Find crossings at local_edge ===
    # For each (E, slot, loop, k), check if gathered_edges == n_loc[slot]
    target_local = n_loc.unsqueeze(2).unsqueeze(3)  # (E, 4, 1, 1)
    match = (gathered_edges == target_local) & in_range.unsqueeze(3) & valid_slot.unsqueeze(2).unsqueeze(3)
    # match: (E, 4, max_loops, K) — True where this edge-in-loop matches the shared local_edge
    
    # === Step 6: Rank normalization ===
    # For edges 2, 6, 3, 7: normalized_rank = W - 1 - rank
    is_neg = (n_loc == 2) | (n_loc == 6) | (n_loc == 3) | (n_loc == 7)  # (E, 4)
    W_broadcast = W_at_edge.unsqueeze(2).unsqueeze(3)  # (E, 4, 1, 1)
    norm_ranks = torch.where(
        is_neg.unsqueeze(2).unsqueeze(3),
        W_broadcast - 1 - gathered_ranks,
        gathered_ranks,
    )
    # Mask: only keep where match is True and norm_rank >= 0 and norm_rank < max_rank
    rank_valid = match & (norm_ranks >= 0) & (norm_ranks < max_rank)
    
    # === Step 7: Gather component points (per loop, same across K) ===
    # loop_component_point: (L, 3)
    if tensors.loop_component_point.shape[0] > 0:
        loop_points_gathered = tensors.loop_component_point[flat_loop_idx_safe]
    else:
        loop_points_gathered = torch.zeros((E, 4, max_loops, 3), dtype=torch.float32, device=device)
    # Shape: (E, 4, max_loops, 3)
    # Expand to (E, 4, max_loops, K, 3) for scatter
    points_per_match = loop_points_gathered.unsqueeze(3).expand(E, 4, max_loops, K, 3)
    
    # === Step 8: Scatter points into (E, max_rank, 4, 3) by normalized rank ===
    points_by_rank = torch.zeros((E, max_rank, 4, 3), dtype=torch.float32, device=device)
    has_point = torch.zeros((E, max_rank, 4), dtype=torch.bool, device=device)
    
    # Flat indices for scattering
    # Each match event: (edge_id, slot, rank) → writes point
    if rank_valid.any():
        # Get flat indices of valid matches
        valid_flat = rank_valid.view(-1)  # (E*4*max_loops*K,)
        idx_flat = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # (M,)
        
        # Recover (e, slot, loop_idx, k) indices
        stride_k = 1
        stride_l = K
        stride_s = max_loops * K
        stride_e = 4 * max_loops * K
        e_idx = idx_flat // stride_e
        rem = idx_flat % stride_e
        s_idx = rem // stride_s
        rem = rem % stride_s
        l_idx = rem // stride_l
        k_idx = rem % stride_l
        
        # Get normalized rank per match
        r_idx = norm_ranks.view(-1)[idx_flat]  # (M,)
        r_idx_long = r_idx.to(torch.int64)
        e_long = e_idx.to(torch.int64)
        s_long = s_idx.to(torch.int64)
        
        # Get points per match
        pts = loop_points_gathered[e_long, s_long, l_idx]  # (M, 3)
        
        # Scatter: points_by_rank[e, r, s] = pt
        # Use index_put_ for vectorized scatter
        points_by_rank[e_long, r_idx_long, s_long] = pts
        has_point[e_long, r_idx_long, s_long] = True
    
    # === Step 9: Exception wildcards ===
    # For each (e, slot) with cube_exc True, get first loop's component_point
    # Only relevant if that slot would otherwise be missing
    any_has_point = has_point.any(dim=2)  # (E, max_rank)
    # For each rank where any slot has a point, fill missing slots with exception
    # exc_first_point: (E, 4, 3) — first component_point of the neighbor cube
    first_loop_idx = loop_start  # (E, 4)
    loop_count = tensors.loop_component_point.shape[0]
    if loop_count > 0:
        first_loop_safe = first_loop_idx.clamp(min=0, max=loop_count - 1)
        exc_pt = tensors.loop_component_point[first_loop_safe.to(torch.int64)]  # (E, 4, 3)
    else:
        exc_pt = torch.zeros((E, 4, 3), dtype=torch.float32, device=device)
    
    # Wildcard fill: (E, max_rank, 4) — slot is exception AND rank has any presence AND not already has_point
    fill_mask = (
        (~has_point) &
        cube_exc.unsqueeze(1) &
        any_has_point.unsqueeze(2) &
        (num_loops.unsqueeze(1) > 0)
    )
    points_by_rank = torch.where(
        fill_mask.unsqueeze(3),
        exc_pt.unsqueeze(1).expand(E, max_rank, 4, 3),
        points_by_rank,
    )
    has_point = has_point | fill_mask
    
    # Rank-0 fallback: if NO rank has any points but exceptions exist → create rank 0 with exceptions
    no_rank = ~any_has_point.any(dim=1)  # (E,)
    has_exc = (cube_exc & (num_loops > 0)).any(dim=1)  # (E,)
    synth_rank0 = no_rank & has_exc  # (E,)
    if synth_rank0.any():
        # For these edges, set rank 0 with exception points
        idx = torch.nonzero(synth_rank0, as_tuple=False).squeeze(1)
        valid_exc = cube_exc[idx] & (num_loops[idx] > 0)  # (M, 4)
        # Set has_point[idx, 0, :] = valid_exc; points = exc_pt[idx, :]
        has_point[idx, 0, :] = valid_exc
        pts_for_rank0 = torch.where(
            valid_exc.unsqueeze(2),
            exc_pt[idx],
            torch.zeros_like(exc_pt[idx]),
        )
        points_by_rank[idx, 0, :, :] = pts_for_rank0
    
    # === Step 10: Emit fan triangles for full rank groups (4 slots present) ===
    full_mask = has_point.all(dim=2)  # (E, max_rank)
    if not full_mask.any():
        return torch.zeros((0, 3), dtype=torch.float64, device=device)
    
    full_edges, full_ranks = torch.nonzero(full_mask, as_tuple=True)
    F = full_edges.shape[0]
    
    # Compute projection point (mean of 4 slots)
    proj = points_by_rank[full_edges, full_ranks].mean(dim=1)  # (F, 3)
    p0 = points_by_rank[full_edges, full_ranks, 0]  # (F, 3)
    p1 = points_by_rank[full_edges, full_ranks, 1]
    p2 = points_by_rank[full_edges, full_ranks, 2]
    p3 = points_by_rank[full_edges, full_ranks, 3]
    
    # Fan triangles: (proj, p0, p1), (proj, p1, p2), (proj, p2, p3), (proj, p3, p0)
    tris = torch.stack([
        torch.stack([proj, p0, p1], dim=1),
        torch.stack([proj, p1, p2], dim=1),
        torch.stack([proj, p2, p3], dim=1),
        torch.stack([proj, p3, p0], dim=1),
    ], dim=1)  # (F, 4, 3, 3)
    
    return tris.reshape(-1, 3).to(torch.float64)
```

- [ ] **Step 3: Run test**

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestProcessGeometryVectorized -xvs
```

Expected: Pass.

- [ ] **Step 4: Commit**

```bash
git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): add process_geometry_vectorized — simple case + exceptions

Vectorized tensor ops replace Python per-edge iteration. Handles exception
cube wildcards and rank-0 synthesis. Conditional promotion deferred to
Python fallback on residual edges.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Integrate Torch path into `process_shared_edges_batch`

**Purpose:** Wire the vectorized path as the default; keep Python path as fallback.

- [ ] **Step 1: Add A/B comparison test for Torch path**

Append to test file:
```python
class TestProcessSharedEdgesTorch:
    """A/B test: compare Torch vectorized path against existing Python path."""
    
    def test_matches_python_path_four_cubes(self):
        """Torch path should produce same vertex/face count as Python path."""
        data = [
            {'cube_indices': (4, 5, 4),
             'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                               'component_point': [0.04, 0.05, 0.04]}],
             'edge_weights': [0]*5 + [1] + [0]*12,
             'exception': False, 'num_components': 1},
            {'cube_indices': (5, 5, 4),
             'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                               'component_point': [0.05, 0.05, 0.04]}],
             'edge_weights': [0]*7 + [1] + [0]*10,
             'exception': False, 'num_components': 1},
            {'cube_indices': (4, 5, 5),
             'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                               'component_point': [0.04, 0.05, 0.05]}],
             'edge_weights': [0, 1] + [0]*16,
             'exception': False, 'num_components': 1},
            {'cube_indices': (5, 5, 5),
             'sorted_loops': [{'loop': [3, 12, 1], 'rank': [0, -1, -1],
                               'component_point': [0.05, 0.05, 0.05]}],
             'edge_weights': [0]*3 + [1] + [0]*14,
             'exception': False, 'num_components': 1},
        ]
        # Python path
        v_py, f_py = process_shared_edges_batch(
            resolution=64, cube_data_list=data, merge_decimals=5, num_workers=1)
        # Torch path
        v_torch, f_torch = process_shared_edges_batch(
            resolution=64, cube_data_list=data, merge_decimals=5,
            num_workers=1, use_torch_path=True)
        assert v_py.shape[0] == v_torch.shape[0], \
            f"vertex count: py={v_py.shape[0]}, torch={v_torch.shape[0]}"
        assert f_py.shape[0] == f_torch.shape[0], \
            f"face count: py={f_py.shape[0]}, torch={f_torch.shape[0]}"
```

- [ ] **Step 2: Modify `process_shared_edges_batch` signature to add `use_torch_path` flag**

```python
def process_shared_edges_batch(
    resolution: int,
    cube_data_list: list[dict],
    merge_decimals: int = 5,
    num_workers: int = 0,
    use_torch_path: bool = False,  # NEW: enable vectorized path
) -> tuple[torch.Tensor, torch.Tensor]:
    """..."""
    if use_torch_path:
        return _process_shared_edges_torch(
            resolution, cube_data_list, merge_decimals)
    # ... existing Python path
```

- [ ] **Step 3: Implement `_process_shared_edges_torch`**

Add to `corep_fast/stages/s8_collapse.py`:
```python
def _process_shared_edges_torch(
    resolution: int,
    cube_data_list: list[dict],
    merge_decimals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Torch path: replaces edge enumeration + geometry with tensor ops."""
    if not cube_data_list:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step A: Convert to tensors
    tensors = _cube_data_to_tensors(cube_data_list, device)
    
    # Step B: Edge enumeration
    keys, cube_ids, local_ids = compute_global_edge_keys(tensors.cube_indices, resolution)
    unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
    num_unique = unique_keys.shape[0]
    table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids, num_unique)
    
    # Filter: only edges with >=2 neighbors (need at least 2 cubes)
    keep = table.neighbor_counts >= 2
    if not keep.any():
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )
    
    kept_table = EdgeNeighborTable(
        neighbor_counts=table.neighbor_counts[keep],
        neighbor_cube_ids=table.neighbor_cube_ids[keep],
        neighbor_positions=table.neighbor_positions[keep],
        neighbor_local_edges=table.neighbor_local_edges[keep],
        edge_axes=table.edge_axes[keep],
    )
    
    # Step C: Vectorized geometry
    tri_verts = process_geometry_vectorized(kept_table, tensors, device)
    
    if tri_verts.shape[0] == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
        )
    
    # Step D: Welding (reuse existing _weld_and_dedup)
    tri_verts_np = tri_verts.cpu().numpy()
    return _weld_and_dedup(tri_verts_np, merge_decimals, device=device)
```

- [ ] **Step 4: Run A/B test**

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_s8_collapse.py::TestProcessSharedEdgesTorch -xvs
```

Expected: PASS — vertex and face counts match exactly.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
.venv/bin/python -m pytest corep_fast/tests/ -q --tb=short
```

Expected: All 121+ tests pass.

- [ ] **Step 6: Commit**

```bash
git add corep_fast/stages/s8_collapse.py corep_fast/tests/unit/test_s8_collapse.py && \
  git commit -m "feat(corep_fast): wire Torch vectorized path into process_shared_edges_batch

New use_torch_path flag enables the fully vectorized path. Python path
remains as fallback and for A/B validation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: End-to-end A/B with real meshes

**Purpose:** Verify the Torch path produces identical output to Python path on real meshes.

- [ ] **Step 1: Add pipeline-level test**

Append to `corep_fast/tests/unit/test_pipeline.py`:
```python
class TestTorchPathABComparison:
    """End-to-end: custom s1-s7 + Torch s8 vs custom s1-s7 + Python s8."""
    
    @pytest.fixture
    def simple_mesh_path(self, tmp_path):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        path = tmp_path / "mesh.ply"
        mesh.export(str(path))
        return str(path)
    
    def test_torch_path_matches_python(self, simple_mesh_path, tmp_path):
        """Run pipeline twice and verify Torch path produces same output counts."""
        import tempfile as _tf
        from corep_fast.pipeline import _run_custom_s1_to_s7
        from corep_fast.stages.s8_collapse import (
            process_shared_edges_batch, _write_ply_ascii
        )
        
        out_dir = _tf.mkdtemp()
        all_regs = _run_custom_s1_to_s7(simple_mesh_path, 64, out_dir)
        
        # Python path
        v_py, f_py = process_shared_edges_batch(
            resolution=64, cube_data_list=all_regs,
            merge_decimals=5, num_workers=1)
        
        # Torch path (need fresh list to avoid mutation)
        all_regs_copy = [dict(d) for d in all_regs]
        v_t, f_t = process_shared_edges_batch(
            resolution=64, cube_data_list=all_regs_copy,
            merge_decimals=5, use_torch_path=True)
        
        assert v_py.shape[0] == v_t.shape[0], \
            f"vertex: py={v_py.shape[0]}, torch={v_t.shape[0]}"
        assert f_py.shape[0] == f_t.shape[0], \
            f"face: py={f_py.shape[0]}, torch={f_t.shape[0]}"
```

- [ ] **Step 2: Run the A/B test**

```bash
.venv/bin/python -m pytest corep_fast/tests/unit/test_pipeline.py::TestTorchPathABComparison -xvs
```

**If test fails:** Debug by:
1. Running both paths, capture intermediate tensors
2. Compare `process_geometry_vectorized` output vs `_process_shared_edge_geometry` on identical inputs
3. Likely sources of discrepancy:
   - Conditional promotion cases not handled (expected gap; identify via difference in triangle count)
   - Rank normalization bug
   - Exception wildcard missing some slots

If discrepancy is small (<1%), proceed and document the gap. If large, fix before next task.

- [ ] **Step 3: Benchmark performance**

Create `/tmp/bench_torch_path.py`:
```python
import sys, time
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2')
sys.path.insert(0, '/mnt/novita2/siyuan/workspace/TRELLIS.2/custom')
import trimesh, tempfile, torch
from corep_fast.pipeline import _run_custom_s1_to_s7
from corep_fast.stages.s8_collapse import process_shared_edges_batch

mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
path = '/tmp/bench.ply'; mesh.export(path)

for res in [128, 256]:
    tmp = tempfile.mkdtemp()
    all_regs = _run_custom_s1_to_s7(path, res, tmp)
    
    # Warmup GPU
    _ = torch.zeros(100, device='cuda')
    torch.cuda.synchronize()
    
    # Python path
    t0 = time.time()
    v_py, f_py = process_shared_edges_batch(res, [dict(d) for d in all_regs], 5)
    t_py = time.time() - t0
    
    # Torch path (3 runs, take best)
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        v_t, f_t = process_shared_edges_batch(res, [dict(d) for d in all_regs], 5,
                                              use_torch_path=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    t_t = min(times)
    
    print(f"res={res}: py={t_py:.2f}s torch={t_t:.2f}s speedup={t_py/t_t:.1f}x "
          f"V_match={v_py.shape[0]==v_t.shape[0]} F_match={f_py.shape[0]==f_t.shape[0]}")
    
    import shutil; shutil.rmtree(tmp, True)
```

Run: `.venv/bin/python /tmp/bench_torch_path.py`

Expected: Torch path should be 2-5x faster than Python path on s8 alone.

- [ ] **Step 4: Commit**

```bash
git add corep_fast/tests/unit/test_pipeline.py && \
  git commit -m "test(corep_fast): add end-to-end A/B test for Torch s8 path

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Conditional Promotion Fallback (if Task 5 reveals gap)

**Purpose:** If the Torch path's output differs from Python path due to conditional promotion logic, add a Python fallback for those rare cases.

- [ ] **Step 1: Detect conditional promotion candidates**

Conditional promotion fires when:
- All 4 neighbors present (`table.neighbor_counts == 4`)
- All cubes have `num_components > 0`
- Some shared edge weight exists (`W_at_edge.max() > 0`)
- No full rank group exists (`~has_point.all(dim=2).any(dim=1)`)

- [ ] **Step 2: Run Python fallback on residual edges only**

Only call `_process_shared_edge_geometry` for edges flagged as needing promotion. This is typically <1% of edges, so the fallback overhead is minimal.

- [ ] **Step 3: Merge results from both paths**

Concatenate vectorized triangles + Python fallback triangles before welding.

*(Detailed implementation deferred — only needed if Task 5 reveals a mismatch.)*

---

## Task 7: Final Integration + Benchmark

**Purpose:** Make Torch path the default, remove the flag once stable, and benchmark end-to-end.

- [ ] **Step 1: Make `use_torch_path=True` the default in `process_shared_edges_batch`**

- [ ] **Step 2: Run full pipeline benchmark at res=128, 256**

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest corep_fast/tests/ -q --tb=short
```

Expected: All tests pass.

- [ ] **Step 4: Update logs/progress.md with final numbers**

- [ ] **Step 5: Commit**

```bash
git add corep_fast/stages/s8_collapse.py logs/progress.md && \
  git commit -m "perf(corep_fast): enable Torch vectorized s8 path as default

Final Stage 2 v2 benchmark (res=256, H100):
- Before Stage 2: 13.9s (5.0x vs custom)
- After Stage 2:  [TBD]s ([TBD]x vs custom)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist

Before exit:
1. ✅ All 121+ existing tests pass
2. ✅ New tests added for each new function
3. ✅ Vertex/face counts match Python path on synthetic + real meshes
4. ✅ Benchmark shows ≥25x speedup vs custom/ at res=256
5. ✅ Code committed incrementally with clear messages
