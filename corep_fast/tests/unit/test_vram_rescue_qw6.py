"""QW6 parity test — on a small single-group input forced through the CPU
fallback, verify output is bit-identical to the GPU path."""
import importlib, os
import pytest
import numpy as np
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VRAM rescue tests require CUDA",
)


def _set_flag(monkeypatch, on: bool):
    monkeypatch.setenv("COREP_FAST_VRAM_RESCUE", "1" if on else "0")
    import corep_fast.config as cfg
    importlib.reload(cfg)


def test_qw6_single_group_cpu_fallback_matches_gpu_on_small_input(monkeypatch):
    """On a small single-group input, baseline the GPU path; then flag on with
    tiny budget forces the CPU fallback and must match bit-exact."""
    from corep_fast.stages.s4_face_point import _count_uturns_gpu_batched_csr_chunk

    # Build 1 group with P=32
    G, P = 1, 32
    S = P // 2
    rng = np.random.default_rng(0)
    A = rng.random((S, 3)).astype(np.float64)
    B = rng.random((S, 3)).astype(np.float64)
    cf = np.array([5 * 12 + 3], dtype=np.int64)
    off = np.array([0, S], dtype=np.int64)
    cube_idx = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0]], dtype=np.int32)

    _set_flag(monkeypatch, on=False)
    out_gpu = _count_uturns_gpu_batched_csr_chunk(
        cf, off, A, B, cube_idx, step=1.0 / 32).cpu().numpy()

    _set_flag(monkeypatch, on=True)
    monkeypatch.setenv("COREP_FAST_S4_UTURN_CHUNK_ELEMS", "1")
    out_cpu = _count_uturns_gpu_batched_csr_chunk(
        cf, off, A, B, cube_idx, step=1.0 / 32).cpu().numpy()

    assert np.array_equal(out_gpu, out_cpu), (
        f"QW6 CPU fallback output {out_cpu} diverges from GPU {out_gpu}")
