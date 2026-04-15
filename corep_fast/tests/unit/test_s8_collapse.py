"""Unit tests for corep_fast/stages/s8_collapse.py — Torch s8 rewrite."""
import os
import tempfile

import numpy as np
import pytest
import torch
import trimesh

from corep_fast.stages.s8_collapse import (
    compute_global_edge_keys,
    enumerate_unique_edges,
    build_edge_neighbor_table,
    compute_edge_ownership,
    process_shared_edges_batch,
    _weld_and_dedup,
    s8_collapse_to_ply,
)


# ---- Fixtures ----

@pytest.fixture
def single_cube():
    """One cube at (5, 5, 5), resolution 64."""
    cube_indices = torch.tensor([[5, 5, 5]], dtype=torch.int32)
    resolution = 64
    return cube_indices, resolution


@pytest.fixture
def two_adjacent_cubes():
    """Two cubes sharing an edge: (5,5,5) and (5,5,6), resolution 64.
    These share the X-axis edge at global coords (5, 5, 6) and others."""
    cube_indices = torch.tensor([[5, 5, 5], [5, 5, 6]], dtype=torch.int32)
    resolution = 64
    return cube_indices, resolution


@pytest.fixture
def four_cubes_around_edge():
    """Four cubes sharing a single Y-axis edge at (5,5,5):
    (4,5,4), (5,5,4), (4,5,5), (5,5,5)
    These are the 4 neighbors returned by get_surrounding_cubes('y', 5, 5, 5)."""
    cube_indices = torch.tensor([
        [4, 5, 4], [5, 5, 4], [4, 5, 5], [5, 5, 5]
    ], dtype=torch.int32)
    resolution = 64
    return cube_indices, resolution


# ---- Tests for compute_global_edge_keys ----

class TestComputeGlobalEdgeKeys:
    def test_single_cube_produces_12_edges(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        # Each cube generates 12 global edge keys
        assert keys.shape == (12,)
        assert cube_ids.shape == (12,)
        assert local_ids.shape == (12,)
        # All cube_ids should be 0 (single cube)
        assert (cube_ids == 0).all()
        # local_ids should be 0..11
        assert local_ids.tolist() == list(range(12))

    def test_two_cubes_produce_24_edges(self, two_adjacent_cubes):
        cube_indices, resolution = two_adjacent_cubes
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        assert keys.shape == (24,)

    def test_global_keys_are_int64(self, single_cube):
        cube_indices, resolution = single_cube
        keys, _, _ = compute_global_edge_keys(cube_indices, resolution)
        assert keys.dtype == torch.int64

    def test_adjacent_cubes_share_edges(self, two_adjacent_cubes):
        """Two adjacent cubes must share some global edge keys."""
        cube_indices, resolution = two_adjacent_cubes
        keys, cube_ids, _ = compute_global_edge_keys(cube_indices, resolution)
        keys_0 = set(keys[cube_ids == 0].tolist())
        keys_1 = set(keys[cube_ids == 1].tolist())
        shared = keys_0 & keys_1
        # Cubes (5,5,5) and (5,5,6) are adjacent in Z — they share 4 edges
        assert len(shared) >= 4


# ---- Tests for enumerate_unique_edges ----

class TestEnumerateUniqueEdges:
    def test_single_cube_has_12_unique_edges(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        assert unique_keys.shape[0] == 12  # isolated cube, all edges unique

    def test_shared_edges_reduce_count(self, two_adjacent_cubes):
        cube_indices, resolution = two_adjacent_cubes
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        # 24 total entries, some deduplicated
        assert unique_keys.shape[0] < 24
        assert edge_id_per_entry.shape == keys.shape

    def test_four_cubes_reduce_more(self, four_cubes_around_edge):
        cube_indices, resolution = four_cubes_around_edge
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        # 4 cubes * 12 = 48 entries, shared edges reduce the unique count
        assert unique_keys.shape[0] < 48


# ---- Tests for build_edge_neighbor_table ----

class TestBuildEdgeNeighborTable:
    def test_single_cube_all_edges_have_one_neighbor(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        # table.neighbor_counts: (num_unique_edges,) — each should be 1
        assert (table.neighbor_counts == 1).all()

    def test_four_cubes_shared_edge_has_four_neighbors(self, four_cubes_around_edge):
        """The Y-axis edge at (5,5,5) is shared by all 4 cubes."""
        cube_indices, resolution = four_cubes_around_edge
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        # At least one edge should have 4 neighbors
        assert (table.neighbor_counts == 4).any()


# ---- Tests for compute_edge_ownership ----

class TestComputeEdgeOwnership:
    def test_single_cube_owns_all_edges(self, single_cube):
        cube_indices, resolution = single_cube
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        owner_cube = compute_edge_ownership(table, cube_indices)
        # Single cube, it owns all 12 edges
        assert (owner_cube == 0).all()

    def test_shared_edge_owned_by_lex_smallest(self, four_cubes_around_edge):
        """Edge ownership should go to the cube with smallest (ix,iy,iz)."""
        cube_indices, resolution = four_cubes_around_edge
        keys, cube_ids, local_ids = compute_global_edge_keys(cube_indices, resolution)
        unique_keys, edge_id_per_entry = enumerate_unique_edges(keys)
        table = build_edge_neighbor_table(edge_id_per_entry, cube_ids, local_ids,
                                          unique_keys.shape[0])
        owner_cube = compute_edge_ownership(table, cube_indices)
        # For the fully-shared edge, owner should be cube 0 = (4,5,4) — lex smallest
        max_neighbors_edge = table.neighbor_counts.argmax()
        assert owner_cube[max_neighbors_edge].item() == 0


# ---- Tests for process_shared_edges_batch ----

class TestProcessSharedEdgesBatch:
    """Test the shared-edge geometry processing."""

    def _make_simple_cube_data(self):
        """Create 4 cubes around a Y-axis edge, each with 1 loop crossing the shared edge.

        Cubes: (4,5,4), (5,5,4), (4,5,5), (5,5,5)
        Shared Y-axis edge at global (5,5,5).

        For axis Y neighbors:
          pos 0: (5,5,5) → local_edge 3
          pos 1: (4,5,5) → local_edge 1
          pos 2: (4,5,4) → local_edge 5
          pos 3: (5,5,4) → local_edge 7

        Each cube has 1 loop crossing its local edge at rank 0, with a component_point.
        """
        cube_data_list = [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'loop': [3, 12, 1], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': False,
                'num_components': 1,
            },
        ]
        return cube_data_list

    def test_four_cubes_produce_fan_triangles(self):
        """With all 4 neighbors present at rank 0, we should get 4 fan triangles."""
        data = self._make_simple_cube_data()
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # 4 component_points + 1 projection_point = 5 vertices (before welding)
        assert verts.shape[0] >= 5
        # 4 fan triangles
        assert faces.shape[0] >= 4
        assert faces.shape[1] == 3

    def test_two_cubes_produce_no_fan(self):
        """With only 2 neighbors, rank group has 2 points — no fan triangles (need 4)."""
        data = self._make_simple_cube_data()[:2]
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # No faces because we need 4 neighbors for fan generation
        assert faces.shape[0] == 0

    def test_empty_input_returns_empty(self):
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=[],
            merge_decimals=5,
        )
        assert verts.shape == (0, 3)
        assert faces.shape == (0, 3)

    def test_projection_point_is_mean(self):
        """The projection point should be the mean of the 4 component points."""
        data = self._make_simple_cube_data()
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # Expected mean: (0.04+0.05+0.04+0.05)/4=0.045, 0.05, (0.04+0.04+0.05+0.05)/4=0.045
        expected_proj = torch.tensor([0.045, 0.05, 0.045])
        # Find the projection point — it should be one of the vertices
        found = False
        for i in range(verts.shape[0]):
            if torch.allclose(verts[i], expected_proj, atol=1e-5):
                found = True
                break
        assert found, f"Projection point {expected_proj} not found in vertices:\n{verts}"


# ---- Tests for _weld_and_dedup ----

class TestWeldAndDedup:
    @staticmethod
    def _tris_to_np(*tris):
        """Convert triangle vertex tuples to (T*3, 3) numpy array."""
        rows = []
        for tri in tris:
            for pt in tri:
                rows.append(list(pt))
        if not rows:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    def test_basic_welding(self):
        """Vertices that round to the same value should be merged."""
        v0 = (0.100001, 0.200001, 0.300001)
        v1 = (0.100002, 0.200002, 0.300002)  # same after rounding to 4 decimals
        v2 = (0.5, 0.5, 0.5)
        arr = self._tris_to_np((v0, v1, v2))
        v, f = _weld_and_dedup(arr, merge_decimals=4)
        # After welding, v0 and v1 merge → 2 unique vertices
        assert v.shape[0] == 2
        # But the face becomes degenerate (v0==v1), so it's removed
        assert f.shape[0] == 0

    def test_non_degenerate_preserved(self):
        """Three distinct vertices form a valid triangle."""
        arr = self._tris_to_np(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )
        v, f = _weld_and_dedup(arr, merge_decimals=5)
        assert v.shape[0] == 3
        assert f.shape[0] == 1

    def test_duplicate_faces_removed(self):
        """Same triangle appearing twice should be deduplicated."""
        arr = self._tris_to_np(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),  # same, rotated
        )
        v, f = _weld_and_dedup(arr, merge_decimals=5)
        assert v.shape[0] == 3
        assert f.shape[0] == 1

    def test_different_faces_preserved(self):
        """Two distinct triangles should both be kept."""
        arr = self._tris_to_np(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        )
        v, f = _weld_and_dedup(arr, merge_decimals=5)
        assert v.shape[0] == 4  # (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        assert f.shape[0] == 2

    def test_empty_input(self):
        v, f = _weld_and_dedup(np.zeros((0, 3), dtype=np.float64), merge_decimals=5)
        assert v.shape == (0, 3)
        assert f.shape == (0, 3)

    def test_output_dtypes(self):
        arr = self._tris_to_np(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )
        v, f = _weld_and_dedup(arr, merge_decimals=5)
        assert v.dtype == torch.float32
        assert f.dtype == torch.int32


# ---- Tests for s8_collapse_to_ply ----

class TestS8CollapseToPly:
    def _make_cube_data_four_around_edge(self):
        """4 cubes around a Y-axis edge with complete rank 0 data."""
        return [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'loop': [3, 12, 1], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': False,
                'num_components': 1,
            },
        ]

    def test_produces_valid_ply(self, tmp_path):
        data = self._make_cube_data_four_around_edge()
        out_path = str(tmp_path / "test_output.ply")
        result = s8_collapse_to_ply(
            resolution=64,
            cube_data_list=data,
            output_filepath=out_path,
        )
        assert os.path.exists(result)
        mesh = trimesh.load(result)
        assert mesh.vertices.shape[0] > 0
        assert mesh.faces.shape[0] > 0

    def test_empty_input_produces_empty_ply(self, tmp_path):
        out_path = str(tmp_path / "empty.ply")
        result = s8_collapse_to_ply(
            resolution=64,
            cube_data_list=[],
            output_filepath=out_path,
        )
        assert os.path.exists(result)

    def test_ply_header_format(self, tmp_path):
        data = self._make_cube_data_four_around_edge()
        out_path = str(tmp_path / "header_check.ply")
        s8_collapse_to_ply(resolution=64, cube_data_list=data, output_filepath=out_path)
        with open(out_path, 'r') as f:
            lines = f.readlines()
        assert lines[0].strip() == 'ply'
        assert lines[1].strip() == 'format ascii 1.0'
        assert 'element vertex' in lines[2]
        assert 'end_header' in ''.join(lines[:10])


# ---- Tests for exception cube handling ----

class TestExceptionCubeHandling:
    """Test that exception cubes inject their component_point as wildcards."""

    def _make_data_with_exception(self):
        """3 normal cubes + 1 exception cube around a shared edge.

        Normal cubes have loops crossing the shared edge at rank 0.
        Exception cube has exception=True and a component_point.
        """
        return [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'loop': [5, 13, 7], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'loop': [7, 13, 5], 'rank': [0, -1, -1],
                                  'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': False,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'loop': [1, 12, 3], 'rank': [0, -1, -1],
                                  'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': False,
                'num_components': 1,
            },
            {
                # Exception cube — its component_point should fill in as wildcard
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': True,
                'num_components': 1,
            },
        ]

    def test_exception_fills_missing_slot(self):
        """With 3 normal + 1 exception, we should still get fan triangles
        because the exception point fills the 4th slot."""
        data = self._make_data_with_exception()
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # Should produce fan triangles (exception fills the 4th position)
        assert faces.shape[0] >= 4

    def test_all_exceptions_produce_fan(self):
        """4 exception cubes — all have component_points, should group at rank 0
        and form fan if 4 points available."""
        data = [
            {
                'cube_indices': (4, 5, 4),
                'sorted_loops': [{'component_point': [0.04, 0.05, 0.04]}],
                'edge_weights': [0]*5 + [1] + [0]*12,
                'exception': True,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 4),
                'sorted_loops': [{'component_point': [0.05, 0.05, 0.04]}],
                'edge_weights': [0]*7 + [1] + [0]*10,
                'exception': True,
                'num_components': 1,
            },
            {
                'cube_indices': (4, 5, 5),
                'sorted_loops': [{'component_point': [0.04, 0.05, 0.05]}],
                'edge_weights': [0, 1] + [0]*16,
                'exception': True,
                'num_components': 1,
            },
            {
                'cube_indices': (5, 5, 5),
                'sorted_loops': [{'component_point': [0.05, 0.05, 0.05]}],
                'edge_weights': [0]*3 + [1] + [0]*14,
                'exception': True,
                'num_components': 1,
            },
        ]
        verts, faces = process_shared_edges_batch(
            resolution=64,
            cube_data_list=data,
            merge_decimals=5,
        )
        # All exceptions → grouped at rank 0 with 4 points → 4 fan triangles
        assert faces.shape[0] >= 4
