"""Unit tests for corep_fast/config.py."""
import pytest

from corep_fast import config as cf


def test_mode_enum_values():
    assert cf.Mode.PRODUCTION.value == 'production'
    assert cf.Mode.DEBUG.value == 'debug'
    assert cf.Mode.STRICT.value == 'strict'


def test_set_and_get_mode_default_is_production():
    cf.set_mode(cf.Mode.PRODUCTION)
    assert cf.get_mode() == cf.Mode.PRODUCTION


def test_set_and_get_mode_roundtrip():
    cf.set_mode(cf.Mode.DEBUG)
    assert cf.get_mode() == cf.Mode.DEBUG
    cf.set_mode(cf.Mode.PRODUCTION)
    assert cf.get_mode() == cf.Mode.PRODUCTION


def test_stage_config_defaults():
    sc = cf.StageConfig.default()
    assert sc.chunk_size_bytes == 512 * 1024 * 1024  # 512 MiB
    assert sc.prod_k_threshold == 100_000
    assert sc.max_poly_verts == 12


def test_stage_config_override():
    sc = cf.StageConfig(chunk_size_bytes=1 << 30, prod_k_threshold=50_000, max_poly_verts=16)
    assert sc.chunk_size_bytes == 1 << 30
    assert sc.prod_k_threshold == 50_000
    assert sc.max_poly_verts == 16


def test_backend_config_all_torch_default():
    bc = cf.BackendConfig.all_torch()
    assert bc.s1 == 'torch'
    assert bc.s2 == 'torch'
    assert bc.s3 == 'torch'
    assert bc.s4_face == 'torch'
    assert bc.s4_point == 'torch'
    assert bc.s5 == 'torch'
    assert bc.s6 == 'torch'
    assert bc.s7 == 'torch'
    assert bc.s8 == 'torch'


def test_backend_config_rejects_invalid_value():
    with pytest.raises((ValueError, AssertionError)):
        cf.BackendConfig(
            s1='fortran', s2='torch', s3='torch', s4_face='torch',
            s4_point='torch', s5='torch', s6='torch', s7='torch', s8='torch',
        )


def test_backend_config_partial_override():
    bc = cf.BackendConfig.all_torch()
    bc2 = bc.with_override(s6='triton')
    assert bc2.s6 == 'triton'
    assert bc2.s1 == 'torch'
    # Original is not mutated
    assert bc.s6 == 'torch'
