"""Unit tests for corep_fast/pipeline.py — hybrid pipeline orchestrator."""
import os
import tempfile

import pytest
import trimesh

from corep_fast.pipeline import run_hybrid_pipeline, PipelineConfig


class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig()
        # By default, s1-s7 use custom/, s8 uses corep_fast
        assert cfg.s8_impl == 'corep_fast'
        assert cfg.s1_to_s7_impl == 'custom'

    def test_all_custom(self):
        cfg = PipelineConfig(s8_impl='custom')
        assert cfg.s8_impl == 'custom'

    def test_invalid_impl_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(s8_impl='invalid')
