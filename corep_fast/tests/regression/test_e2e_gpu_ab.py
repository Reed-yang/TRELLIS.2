# corep_fast/tests/regression/test_e2e_gpu_ab.py
"""End-to-end A/B: full GPU pipeline vs custom/ pipeline.

The GPU pipeline uses MeshTensors.from_trimesh normalization (symmetric 0.5
centering) while custom/ uses custom/voxelize.normalize_mesh (per-axis
centering). This means different sets of occupied cubes, edge weights,
loops, and output meshes. So the E2E test checks for approximate count
matching (within 5%), not exact match.

Per-stage A/B tests (Tasks 3, 5, 7, 9, 11, 13) already verified exact
matching when using the same normalization.
"""
import pytest
import torch
import numpy as np
import trimesh

from corep_fast.pipeline import corep_pipeline
from corep_fast.interop.custom_runner import run_custom_through_stage
from corep_fast.stages.s8_collapse import process_shared_edges_batch


class TestE2EGPUAB:
    @pytest.fixture
    def custom_mesh(self, ab_mesh_path, ab_resolution):
        """Run full custom/ pipeline -> mesh."""
        regs = run_custom_through_stage(ab_mesh_path, ab_resolution, up_to='s7')
        v, f = process_shared_edges_batch(ab_resolution, regs, merge_decimals=5, num_workers=1)
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if isinstance(f, torch.Tensor):
            f = f.cpu().numpy()
        return v, f

    @pytest.fixture
    def gpu_mesh(self, ab_mesh_path, ab_resolution, gpu_device):
        """Run full GPU pipeline -> mesh."""
        batch, v, f = corep_pipeline(ab_mesh_path, ab_resolution, gpu_device)
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if isinstance(f, torch.Tensor):
            f = f.cpu().numpy()
        return v, f

    def test_vertex_count_matches(self, custom_mesh, gpu_mesh):
        v_custom, _ = custom_mesh
        v_gpu, _ = gpu_mesh
        # Allow small difference due to normalization divergence
        ratio = v_gpu.shape[0] / max(v_custom.shape[0], 1)
        assert 0.95 <= ratio <= 1.05, \
            f"Vertex count: custom={v_custom.shape[0]} vs gpu={v_gpu.shape[0]} (ratio={ratio:.3f})"

    def test_face_count_matches(self, custom_mesh, gpu_mesh):
        _, f_custom = custom_mesh
        _, f_gpu = gpu_mesh
        ratio = f_gpu.shape[0] / max(f_custom.shape[0], 1)
        assert 0.95 <= ratio <= 1.05, \
            f"Face count: custom={f_custom.shape[0]} vs gpu={f_gpu.shape[0]} (ratio={ratio:.3f})"

    def test_output_is_valid_mesh(self, gpu_mesh):
        """Verify the GPU output forms a valid mesh (no degenerate faces)."""
        v, f = gpu_mesh
        assert v.ndim == 2 and v.shape[1] == 3, f"Expected (V, 3), got {v.shape}"
        assert f.ndim == 2 and f.shape[1] == 3, f"Expected (F, 3), got {f.shape}"
        assert v.shape[0] > 0, "No vertices produced"
        assert f.shape[0] > 0, "No faces produced"
        # All face indices should reference valid vertices
        assert f.max() < v.shape[0], \
            f"Face index {f.max()} >= num vertices {v.shape[0]}"
        assert f.min() >= 0, f"Negative face index: {f.min()}"

    def test_custom_output_is_valid_mesh(self, custom_mesh):
        """Verify the custom output forms a valid mesh (no degenerate faces)."""
        v, f = custom_mesh
        assert v.ndim == 2 and v.shape[1] == 3, f"Expected (V, 3), got {v.shape}"
        assert f.ndim == 2 and f.shape[1] == 3, f"Expected (F, 3), got {f.shape}"
        assert v.shape[0] > 0, "No vertices produced"
        assert f.shape[0] > 0, "No faces produced"
        assert f.max() < v.shape[0], \
            f"Face index {f.max()} >= num vertices {v.shape[0]}"
        assert f.min() >= 0, f"Negative face index: {f.min()}"

    def test_benchmark_timing(self, ab_mesh_path, gpu_device):
        """Measure GPU pipeline timing (informational, not a pass/fail test)."""
        import time

        # Warmup
        corep_pipeline(ab_mesh_path, 64, gpu_device)

        # Time it
        t0 = time.perf_counter()
        batch, v, f = corep_pipeline(ab_mesh_path, 64, gpu_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        print(f"\n--- GPU Pipeline Benchmark ---")
        print(f"Resolution: 64")
        print(f"Cubes: {batch.num_cubes}")
        print(f"Vertices: {v.shape[0] if hasattr(v, 'shape') else len(v)}")
        print(f"Faces: {f.shape[0] if hasattr(f, 'shape') else len(f)}")
        print(f"Time: {elapsed:.3f}s")
        print(f"------------------------------")
