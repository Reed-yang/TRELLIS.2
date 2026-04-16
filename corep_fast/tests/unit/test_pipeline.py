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


class TestABComparison:
    """A/B comparison: custom/ s8 vs corep_fast/ s8 on the same s1-s7 output."""

    @pytest.fixture
    def simple_mesh_path(self, tmp_path):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        path = tmp_path / "test_sphere.ply"
        mesh.export(str(path))
        return str(path)

    def test_mesh_vertex_count_within_tolerance(self, simple_mesh_path, tmp_path):
        """Both implementations should produce similar vertex counts."""
        # Run custom/ for everything
        cfg_custom = PipelineConfig(s8_impl='custom')
        out_custom = str(tmp_path / "out_custom.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_custom,
            config=cfg_custom,
        )

        # Run custom/ s1-s7 + corep_fast/ s8
        cfg_fast = PipelineConfig(s8_impl='corep_fast')
        out_fast = str(tmp_path / "out_fast.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_fast,
            config=cfg_fast,
        )

        mesh_custom = trimesh.load(out_custom)
        mesh_fast = trimesh.load(out_fast)

        # Vertex count should match exactly (same algorithm, same data)
        assert mesh_custom.vertices.shape[0] == mesh_fast.vertices.shape[0], \
            f"Vertex count mismatch: custom={mesh_custom.vertices.shape[0]}, fast={mesh_fast.vertices.shape[0]}"

    def test_mesh_face_count_matches(self, simple_mesh_path, tmp_path):
        """Both implementations should produce the same face count."""
        cfg_custom = PipelineConfig(s8_impl='custom')
        out_custom = str(tmp_path / "out_custom.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_custom,
            config=cfg_custom,
        )

        cfg_fast = PipelineConfig(s8_impl='corep_fast')
        out_fast = str(tmp_path / "out_fast.ply")
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out_fast,
            config=cfg_fast,
        )

        mesh_custom = trimesh.load(out_custom)
        mesh_fast = trimesh.load(out_fast)

        assert mesh_custom.faces.shape[0] == mesh_fast.faces.shape[0], \
            f"Face count mismatch: custom={mesh_custom.faces.shape[0]}, fast={mesh_fast.faces.shape[0]}"

    def test_profiled_pipeline_records_s8(self, simple_mesh_path, tmp_path):
        """Profiling collector should record s8_collapse timing."""
        from corep_fast.profiling.harness import ProfilingCollector

        cfg = PipelineConfig(s8_impl='corep_fast')
        out = str(tmp_path / "out_profiled.ply")
        pc = ProfilingCollector(mesh_name='test', resolution=64, impl='hybrid')
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=64,
            output_path=out,
            config=cfg,
            collector=pc,
        )

        assert 's8_collapse' in pc
        assert pc['s8_collapse'].wall_time_s > 0


class TestPerformanceBenchmark:
    """Performance comparison — not strict assertions, but reports speedup."""

    @pytest.fixture
    def simple_mesh_path(self, tmp_path):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
        path = tmp_path / "bench_sphere.ply"
        mesh.export(str(path))
        return str(path)

    def test_report_speedup(self, simple_mesh_path, tmp_path, capsys):
        """Run both implementations and print timing comparison."""
        from corep_fast.profiling.harness import ProfilingCollector

        resolution = 128

        # Custom
        cfg_custom = PipelineConfig(s8_impl='custom')
        out_custom = str(tmp_path / "bench_custom.ply")
        pc_custom = ProfilingCollector(mesh_name='bench', resolution=resolution, impl='custom')
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=resolution,
            output_path=out_custom,
            config=cfg_custom,
            collector=pc_custom,
        )

        # corep_fast
        cfg_fast = PipelineConfig(s8_impl='corep_fast')
        out_fast = str(tmp_path / "bench_fast.ply")
        pc_fast = ProfilingCollector(mesh_name='bench', resolution=resolution, impl='corep_fast')
        run_hybrid_pipeline(
            mesh_path=simple_mesh_path,
            resolution=resolution,
            output_path=out_fast,
            config=cfg_fast,
            collector=pc_fast,
        )

        t_custom = pc_custom['s8_collapse'].wall_time_s
        t_fast = pc_fast['s8_collapse'].wall_time_s
        speedup = t_custom / t_fast if t_fast > 0 else float('inf')

        print(f"\n--- s8_collapse Benchmark ---")
        print(f"  Custom:     {t_custom:.3f}s")
        print(f"  corep_fast: {t_fast:.3f}s")
        print(f"  Speedup:    {speedup:.1f}x")

        mesh_c = trimesh.load(out_custom)
        mesh_f = trimesh.load(out_fast)
        print(f"  Custom mesh:     V={mesh_c.vertices.shape[0]}, F={mesh_c.faces.shape[0]}")
        print(f"  corep_fast mesh: V={mesh_f.vertices.shape[0]}, F={mesh_f.faces.shape[0]}")

        # No strict speedup assertion in Phase 1a — the torch.unique optimization
        # is the critical path and will show benefit at higher resolution.
        # At low resolution (128), overhead may dominate.
        assert t_custom > 0
        assert t_fast > 0
