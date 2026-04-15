"""Unit tests for corep_fast/interop/from_custom.py and to_custom.py."""
import numpy as np
import pytest
import torch
import trimesh

from corep_fast.containers import MeshTensors, CubeBatch
from corep_fast.interop.from_custom import cube_batch_from_custom


def _make_fake_face_registers() -> list[dict]:
    """Simulate custom/'s output: a list of dicts, one per cube."""
    return [
        {
            'cube_indices': (10, 20, 30),
            'face_indices': [0, 1, 2],
            'num_components': 1,
            'num_boundary': 0,
            'edge_weights': list(range(18)),
            'face_weights': list(range(12)),
            'component_points': [[0.5, 0.5, 0.5]],
        },
        {
            'cube_indices': (11, 20, 30),
            'face_indices': [3, 4],
            'num_components': 2,
            'num_boundary': 1,
            'edge_weights': [1] * 18,
            'face_weights': [0] * 12,
            'component_points': [[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]],
        },
    ]


def _make_mesh_tensors() -> MeshTensors:
    mesh = trimesh.creation.box(extents=[1., 1., 1.])
    return MeshTensors.from_trimesh(mesh, resolution=64, device='cpu')


def test_from_custom_basic():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.num_cubes == 2
    assert torch.equal(cb.cube_indices[0], torch.tensor([10, 20, 30], dtype=torch.int32))
    assert torch.equal(cb.cube_indices[1], torch.tensor([11, 20, 30], dtype=torch.int32))


def test_from_custom_tri_csr():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    tris_0 = cb.get_tris(0)
    assert torch.equal(tris_0, torch.tensor([0, 1, 2], dtype=torch.int32))
    tris_1 = cb.get_tris(1)
    assert torch.equal(tris_1, torch.tensor([3, 4], dtype=torch.int32))


def test_from_custom_scalar_fields():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.num_components[0].item() == 1
    assert cb.num_components[1].item() == 2
    assert cb.num_boundary[0].item() == 0
    assert cb.num_boundary[1].item() == 1


def test_from_custom_edge_weights():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.edge_weights[0].tolist() == list(range(18))
    assert cb.edge_weights[1].tolist() == [1] * 18


def test_from_custom_face_weights():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    assert cb.face_weights[0].tolist() == list(range(12))
    assert cb.face_weights[1].tolist() == [0] * 12


def test_from_custom_component_points():
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')

    # Cube 0 has 1 point, cube 1 has 2 points
    assert cb.point_offsets.tolist() == [0, 1, 3]
    assert cb.point_values.shape == (3, 3)
    assert torch.allclose(cb.point_values[0], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(cb.point_values[2], torch.tensor([0.7, 0.7, 0.7]))


def test_from_custom_missing_optional_fields():
    """If fields like loops or edge_weights are absent, they default to zeros."""
    regs = [{'cube_indices': (5, 5, 5), 'face_indices': [0]}]
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, device='cpu')
    assert cb.num_cubes == 1
    assert cb.edge_weights[0].sum().item() == 0  # default zeros
    assert cb.num_components[0].item() == 0


def test_from_custom_include_subset():
    """Only populate specified fields; others get sentinels/zeros."""
    regs = _make_fake_face_registers()
    mt = _make_mesh_tensors()
    cb = cube_batch_from_custom(regs, mt, include={'cube_indices', 'face_indices'}, device='cpu')
    assert cb.num_cubes == 2
    # tri CSR should be populated
    assert cb.tri_values.shape[0] > 0
    # edge_weights should be zero (not populated)
    assert cb.edge_weights.sum().item() == 0
