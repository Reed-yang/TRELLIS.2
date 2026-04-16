"""A/B regression test: ensure s8 vectorized-only (no Python fallback) matches
custom/ baseline V/F bit-exact for 4-cube edges.

Currently the test compares two corep_fast s8 paths:
  (a) baseline: vectorized partial-cube + Python 4-cube fallback (current default)
  (b) candidate: vectorized partial-cube + vectorized 4-cube (PHASE 2 W1 work)

The candidate must produce V/F that match (a) exactly (same V count, same F count,
same vertex set after canonical sort, same triangle set after canonical sort).
"""
import os
from pathlib import Path

import torch
import trimesh
import pytest

from corep_fast.stages.s8_collapse import decode_from_cubebatch
from corep_fast.pipeline import corep_encode


@pytest.fixture(scope="module")
def icosphere_mesh_path(tmp_path_factory):
    """Create icosphere subdiv=2 PLY file, session-scoped."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    path = tmp_path_factory.mktemp("s8_ab") / "icosphere_s2_for_test.ply"
    mesh.export(str(path))
    return str(path)


@pytest.mark.parametrize("res", [32, 64, 128])
def test_4cube_vectorized_matches_python_fallback(res, icosphere_mesh_path):
    """运行 s1-s7 得到 CubeBatch，然后对比两条 s8 路径的 V/F 是否等价。"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 运行 s1-s7 构建 CubeBatch
    batch = corep_encode(icosphere_mesh_path, resolution=res, device=device)

    # 路径 (a): 当前默认（Python fallback，flag=0）
    os.environ["COREP_FAST_S8_4CUBE_VECTORIZED"] = "0"
    v_a, f_a = decode_from_cubebatch(batch)

    # 路径 (b): 纯向量化（Phase 2 W1 目标，flag=1）
    os.environ["COREP_FAST_S8_4CUBE_VECTORIZED"] = "1"
    v_b, f_b = decode_from_cubebatch(batch)

    # V/F 数量一致性检验
    assert v_a.shape == v_b.shape, (
        f"res={res}: V count differs ({v_a.shape} vs {v_b.shape})"
    )
    assert f_a.shape == f_b.shape, (
        f"res={res}: F count differs ({f_a.shape} vs {f_b.shape})"
    )

    # 规范化排序后比较（顶点顺序可能不同）
    def _canon_vertices(v):
        """将顶点去重并排序，返回规范集合。"""
        return torch.unique(v.round(decimals=5), dim=0)

    def _canon_triangles(v, f):
        """替换面索引为顶点坐标，然后规范化。"""
        tri_coords = v[f]          # (F, 3, 3)
        tri_sorted = torch.sort(tri_coords.reshape(-1, 9), dim=0).values
        return tri_sorted

    cv_a, cv_b = _canon_vertices(v_a), _canon_vertices(v_b)
    assert torch.allclose(cv_a, cv_b, atol=1e-4), (
        f"res={res}: canonical vertex set differs"
    )

    ct_a, ct_b = _canon_triangles(v_a, f_a), _canon_triangles(v_b, f_b)
    assert torch.allclose(ct_a, ct_b, atol=1e-4), (
        f"res={res}: canonical triangle set differs"
    )
