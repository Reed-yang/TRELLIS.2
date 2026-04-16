"""Unit tests for corep_fast/constants.py cube topology tables."""
import torch

from corep_fast import constants as C


def test_num_edges_is_18():
    assert C.NUM_EDGES == 18


def test_num_facets_is_12():
    assert C.NUM_FACETS == 12


def test_num_vertices_is_8():
    assert C.NUM_VERTICES == 8


def test_cube_vertices_shape_and_dtype():
    assert isinstance(C.CUBE_VERTICES, torch.Tensor)
    assert C.CUBE_VERTICES.shape == (8, 3)
    assert C.CUBE_VERTICES.dtype == torch.float32
    # Corner (0,0,0) exists
    assert torch.equal(C.CUBE_VERTICES[0], torch.tensor([0., 0., 0.]))
    # Corner (1,1,1) exists
    assert torch.equal(C.CUBE_VERTICES[6], torch.tensor([1., 1., 1.]))


def test_cube_edges_shape_and_values():
    assert isinstance(C.CUBE_EDGES, torch.Tensor)
    assert C.CUBE_EDGES.shape == (18, 2)
    assert C.CUBE_EDGES.dtype == torch.int32
    # Edge 0 connects vertex 0 and 1
    assert C.CUBE_EDGES[0].tolist() == [0, 1]
    # Edge 12 is the bottom diagonal (0,2)
    assert C.CUBE_EDGES[12].tolist() == [0, 2]
    # Edge 17 is the left face diagonal (0,7)
    assert C.CUBE_EDGES[17].tolist() == [0, 7]


def test_cube_facets_shape_and_values():
    """12 triangular facets, each defined by 3 edge indices."""
    assert isinstance(C.CUBE_FACETS, torch.Tensor)
    assert C.CUBE_FACETS.shape == (12, 3)
    assert C.CUBE_FACETS.dtype == torch.int32
    # T0: bottom half 1 — edges (0, 1, 12)
    assert C.CUBE_FACETS[0].tolist() == [0, 1, 12]
    # T11: left half 2 — edges (7, 8, 17)
    assert C.CUBE_FACETS[11].tolist() == [7, 8, 17]


def test_cube_edges_all_reference_valid_vertices():
    assert C.CUBE_EDGES.min() >= 0
    assert C.CUBE_EDGES.max() < 8


def test_cube_facets_all_reference_valid_edges():
    assert C.CUBE_FACETS.min() >= 0
    assert C.CUBE_FACETS.max() < 18


def test_edge_share_factors_shape():
    """Each edge has a cross-cube sharing factor (4 for axis edges, 2 for diagonals)."""
    assert C.EDGE_SHARE_FACTORS.shape == (18,)
    assert C.EDGE_SHARE_FACTORS.dtype == torch.int32
    # Axis edges 0-11: shared by 4 cubes
    assert torch.all(C.EDGE_SHARE_FACTORS[:12] == 4)
    # Diagonal edges 12-17: shared by 2 cubes
    assert torch.all(C.EDGE_SHARE_FACTORS[12:] == 2)
