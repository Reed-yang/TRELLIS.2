"""Unit tests for corep_fast/profiling/topology_equivalence.py."""
import pytest
import torch

from corep_fast.containers import CubeBatch
from corep_fast.profiling.topology_equivalence import (
    EquivalenceReport,
    check_layer1_integer_fields,
    check_topology_equivalence,
)


def _make_cube_batch_with_fields(num_cubes=3, device='cpu') -> CubeBatch:
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device(device))
    cb.num_components[:] = torch.arange(num_cubes, dtype=torch.int32) % 3 + 1
    cb.num_boundary[:] = torch.arange(num_cubes, dtype=torch.int32) % 2
    cb.edge_weights[:] = torch.arange(num_cubes * 18, dtype=torch.int32).reshape(num_cubes, 18) % 4
    cb.face_weights[:] = torch.arange(num_cubes * 12, dtype=torch.int32).reshape(num_cubes, 12) % 3
    cb.status[:] = torch.zeros(num_cubes, dtype=torch.int32)
    return cb


def test_l1_equivalence_passes_on_identical_batches():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is True
    assert report.num_cubes == 3
    assert len(report.mismatches) == 0


def test_l1_equivalence_fails_on_num_components_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.num_components[1] = 3  # diverge
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'num_components' for m in report.mismatches)
    assert report.mismatches[0].cube_idx == 1
    assert report.mismatches[0].value_a == 2
    assert report.mismatches[0].value_b == 3


def test_l1_equivalence_fails_on_edge_weights_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.edge_weights[0, 5] = 99  # diverge at cube 0, edge 5
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'edge_weights' for m in report.mismatches)


def test_l1_equivalence_fails_on_face_weights_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.face_weights[2, 7] = 99
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'face_weights' for m in report.mismatches)


def test_l1_equivalence_fails_on_status_diff():
    cb_a = _make_cube_batch_with_fields()
    cb_b = _make_cube_batch_with_fields()
    cb_b.status[0] = 2  # UNSOLVABLE
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'status' for m in report.mismatches)


def test_l1_equivalence_fails_on_shape_mismatch():
    cb_a = _make_cube_batch_with_fields(num_cubes=3)
    cb_b = _make_cube_batch_with_fields(num_cubes=4)
    report = check_layer1_integer_fields(cb_a, cb_b)
    assert report.passed is False
    assert any(m.field == 'shape' for m in report.mismatches)


def test_top_level_check_layer1_only_runs_when_shapes_match():
    """The top-level check_topology_equivalence should short-circuit on shape mismatch."""
    cb_a = _make_cube_batch_with_fields(num_cubes=3)
    cb_b = _make_cube_batch_with_fields(num_cubes=4)
    report = check_topology_equivalence(cb_a, cb_b, layers=['l1'])
    assert not report.all_passed()
    assert report.layer1.passed is False


# ---------------------------------------------------------------------------
# Layer 2: Loop canonical form + loop set equivalence
# ---------------------------------------------------------------------------
from corep_fast.profiling.topology_equivalence import (
    canonicalize_loop,
    canonicalize_loop_set,
    check_layer2_loop_structure,
)


def test_canonicalize_loop_rotates_to_min_start():
    loop = [5, 7, 11, 3]
    canon = canonicalize_loop(loop)
    assert canon == (3, 5, 7, 11)


def test_canonicalize_loop_prefers_forward_direction():
    loop = [3, 5, 7, 11]
    canon = canonicalize_loop(loop)
    assert canon == (3, 5, 7, 11)


def test_canonicalize_loop_picks_reversed_when_smaller():
    loop = [3, 11, 7, 5]
    canon = canonicalize_loop(loop)
    assert canon == (3, 5, 7, 11)


def test_canonicalize_loop_single_edge():
    assert canonicalize_loop([7]) == (7,)


def test_canonicalize_loop_empty_is_error():
    with pytest.raises(ValueError):
        canonicalize_loop([])


def test_canonicalize_loop_set_sorts_loops():
    loops = [[5, 7, 11], [2, 4, 6]]
    canon = canonicalize_loop_set(loops)
    assert canon == ((2, 4, 6), (5, 7, 11))


def test_canonicalize_loop_set_handles_rotation_and_reflection_per_loop():
    loops = [[7, 11, 5], [6, 4, 2]]
    canon = canonicalize_loop_set(loops)
    assert canon == ((2, 4, 6), (5, 7, 11))


def _cube_batch_with_loops(
    loops_per_cube: list[list[list[int]]],
    num_cubes: int = None,
    device: str = 'cpu',
) -> CubeBatch:
    """Helper: build a CubeBatch whose loop CSR contains the given nested list."""
    if num_cubes is None:
        num_cubes = len(loops_per_cube)
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device(device))

    loop_cube_off = [0]
    loop_edge_off = [0]
    loop_edge_val = []
    for cube_loops in loops_per_cube:
        loop_cube_off.append(loop_cube_off[-1] + len(cube_loops))
        for loop in cube_loops:
            loop_edge_off.append(loop_edge_off[-1] + len(loop))
            loop_edge_val.extend(loop)

    cb.loop_cube_off = torch.tensor(loop_cube_off, dtype=torch.int64)
    cb.loop_edge_off = torch.tensor(loop_edge_off, dtype=torch.int64)
    cb.loop_edge_val = torch.tensor(loop_edge_val, dtype=torch.int32)
    cb.loop_edge_rank = torch.full((len(loop_edge_val),), -1, dtype=torch.int32)
    return cb


def test_l2_equivalence_passes_on_identical_loops():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]], [[2, 4, 6]]])
    cb_b = _cube_batch_with_loops([[[5, 7, 11, 3]], [[2, 4, 6]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert report.passed


def test_l2_equivalence_passes_on_rotated_loops():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]]])
    cb_b = _cube_batch_with_loops([[[3, 5, 7, 11]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert report.passed


def test_l2_equivalence_passes_on_reversed_loops():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]]])
    cb_b = _cube_batch_with_loops([[[3, 11, 7, 5]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert report.passed


def test_l2_equivalence_fails_on_different_loop_sets():
    cb_a = _cube_batch_with_loops([[[5, 7, 11, 3]]])
    cb_b = _cube_batch_with_loops([[[5, 7, 11, 4]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].cube_idx == 0
    assert report.mismatches[0].field == 'loop_set'


def test_l2_equivalence_fails_on_different_loop_count():
    cb_a = _cube_batch_with_loops([[[1, 2, 3], [4, 5, 6]]])
    cb_b = _cube_batch_with_loops([[[1, 2, 3]]])
    report = check_layer2_loop_structure(cb_a, cb_b)
    assert not report.passed


from corep_fast.profiling.topology_equivalence import (
    check_layer3_rank_assignment,
    check_layer4_point_matching,
    check_layer5_point_coordinates,
)


def _cube_batch_with_ranks_and_points(
    loops_per_cube: list[list[list[int]]],
    ranks_per_cube: list[list[list[int]]],
    match_per_cube: list[list[int]],
    points_per_cube: list[list[list[float]]],
) -> CubeBatch:
    """Build CubeBatch with loops + ranks + point matching + points."""
    num_cubes = len(loops_per_cube)
    cb = CubeBatch.empty(num_cubes=num_cubes, resolution=64, device=torch.device('cpu'))

    loop_cube_off = [0]
    loop_edge_off = [0]
    loop_edge_val = []
    loop_edge_rank = []
    loop_point_match = []
    point_offsets = [0]
    point_values = []

    for ci in range(num_cubes):
        cube_loops = loops_per_cube[ci]
        cube_ranks = ranks_per_cube[ci]
        cube_match = match_per_cube[ci]
        cube_points = points_per_cube[ci]

        loop_cube_off.append(loop_cube_off[-1] + len(cube_loops))
        for li, loop in enumerate(cube_loops):
            loop_edge_off.append(loop_edge_off[-1] + len(loop))
            loop_edge_val.extend(loop)
            loop_edge_rank.extend(cube_ranks[li])
        loop_point_match.extend(cube_match)
        point_offsets.append(point_offsets[-1] + len(cube_points))
        point_values.extend(cube_points)

    cb.loop_cube_off = torch.tensor(loop_cube_off, dtype=torch.int64)
    cb.loop_edge_off = torch.tensor(loop_edge_off, dtype=torch.int64)
    cb.loop_edge_val = torch.tensor(loop_edge_val, dtype=torch.int32)
    cb.loop_edge_rank = torch.tensor(loop_edge_rank, dtype=torch.int32)
    cb.loop_point_match = torch.tensor(loop_point_match, dtype=torch.int32)
    cb.point_offsets = torch.tensor(point_offsets, dtype=torch.int64)
    if point_values:
        cb.point_values = torch.tensor(point_values, dtype=torch.float32)
    else:
        cb.point_values = torch.zeros((0, 3), dtype=torch.float32)
    return cb


# ── L3 tests ──

def test_l3_passes_identical_ranks():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    report = check_layer3_rank_assignment(cb_a, cb_b)
    assert report.passed


def test_l3_fails_on_rank_diff():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 1, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    report = check_layer3_rank_assignment(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].field == 'loop_edge_rank'


# ── L4 tests ──

def test_l4_passes_identical_matching():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[0, 1]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[0, 1]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    report = check_layer4_point_matching(cb_a, cb_b)
    assert report.passed


def test_l4_fails_on_match_swap():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[0, 1]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7], [1, 2]]],
        ranks_per_cube=[[[0, 0, 0], [0, 0]]],
        match_per_cube=[[1, 0]],
        points_per_cube=[[[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]]],
    )
    report = check_layer4_point_matching(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].field == 'loop_point_match'


# ── L5 tests ──

def test_l5_passes_within_tolerance():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.50005, 0.49998, 0.50001]]],
    )
    report = check_layer5_point_coordinates(cb_a, cb_b)
    assert report.passed


def test_l5_fails_outside_tolerance():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5]]],
        ranks_per_cube=[[[0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.6, 0.5, 0.5]]],
    )
    report = check_layer5_point_coordinates(cb_a, cb_b)
    assert not report.passed
    assert report.mismatches[0].field == 'point_values'


def test_full_5layer_check_passes():
    cb_a = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    cb_b = _cube_batch_with_ranks_and_points(
        loops_per_cube=[[[3, 5, 7]]],
        ranks_per_cube=[[[0, 0, 0]]],
        match_per_cube=[[0]],
        points_per_cube=[[[0.5, 0.5, 0.5]]],
    )
    report = check_topology_equivalence(cb_a, cb_b, layers=['l1', 'l2', 'l3', 'l4', 'l5'])
    assert report.all_passed()
