"""Smoke test: all corep_fast subpackages import without error."""


def test_corep_fast_root_imports():
    import corep_fast  # noqa: F401


def test_corep_fast_geometry_imports():
    import corep_fast.geometry  # noqa: F401


def test_corep_fast_stages_imports():
    import corep_fast.stages  # noqa: F401


def test_corep_fast_interop_imports():
    import corep_fast.interop  # noqa: F401


def test_corep_fast_profiling_imports():
    import corep_fast.profiling  # noqa: F401


def test_corep_fast_distributed_imports():
    import corep_fast.distributed  # noqa: F401


def test_torch_scatter_available():
    import torch_scatter
    assert hasattr(torch_scatter, 'segment_csr')
