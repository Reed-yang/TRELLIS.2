"""OOM-resilient chunked dense label-propagation for s2_components.

The original dense path in s2_components.py allocates several (N, M, M)
tensors inside the label-propagation loop:

  match         (N, M, 3, M) bool   — 3*N*M^2 bytes (transient)
  adj           (N, M, M)    bool   — N*M^2 bytes
  adj_labels    (N, M, M)    int64  — 8*N*M^2 bytes (dominant, lives across
                                      the per-iteration clone)

At M=64 and N=3M (res=512 dense occupancy), the adj_labels clone alone is
~98 GiB, which OOMs on an 80 GiB HBM GPU. Real-mesh profiling on 2026-04-21
confirmed this is the primary OOM source for res=512 meshes (easy-40GiB +
median-90GiB both OOMed in s2, not s4).

Per-cube label propagation is independent — each cube's final labelling
depends only on its own (padded_faces, neighbors_of, mask) rows. Chunking
by N (batch axis) is therefore semantically safe and bit-exact against the
non-chunked reference implementation.

This module provides three tiers analogous to s4_phase_a_chunked:
  Tier 0 — full-N GPU (no chunking) if nominal peak ≤ 70% of free VRAM.
  Tier 1 — adaptive chunked GPU with reactive OOM retry.
  Tier 2 — chunked CPU fallback when all GPU attempts OOM.
"""
from __future__ import annotations

import os
import torch


# Per-cube nominal footprint for the chunk-sizing heuristic.
# Coexistent allocations per cube at peak:
#   match:       3 * M^2 bytes (bool)          = 3 * M^2
#   adj:             M^2 bytes (bool)          =     M^2
#   adj_labels:  8 * M^2 bytes (int64)         = 8 * M^2
#   labels row:  8 * M     bytes (int64)       — small, absorbed in margin
# Total dominant: (3 + 1 + 8) * M^2 = 12 * M^2 bytes per cube.
_BYTES_PER_CUBE_MULTIPLIER = 12

# Adaptive dispatch fractions: start at 75% of free VRAM, halve each retry.
_ADAPTIVE_GPU_FRACTIONS = (0.75, 0.40, 0.20, 0.10)

# CPU tier budget (bytes) — conservative default for host RAM.
_DEFAULT_CPU_BUDGET_BYTES = 40 * (1024 ** 3)
_DEFAULT_CHUNK_BYTES = 20 * (1024 ** 3)  # fallback when mem_get_info fails


def _label_prop_chunk_impl(
    padded_faces: torch.Tensor,  # (N, M) int32
    neighbors_of: torch.Tensor,  # (N, M, 3) int32
    mask: torch.Tensor,          # (N, M) bool
    N: int,
    max_faces: int,
    n_chunk: int,
    work_device: torch.device,
) -> torch.Tensor:
    """Shared chunked label-propagation. Runs on `work_device`, returns
    `labels (N, M) int64` on the ORIGINAL device of `padded_faces`.

    Semantics match the non-chunked dense path exactly: each chunk runs
    the full label-prop convergence loop independently, then results are
    assembled back into the full (N, M) output.
    """
    orig_dev = padded_faces.device

    if work_device is None or str(work_device) == str(orig_dev):
        pf_w = padded_faces
        nbr_w = neighbors_of
        mask_w = mask
    else:
        pf_w = padded_faces.detach().to(work_device)
        nbr_w = neighbors_of.detach().to(work_device)
        mask_w = mask.detach().to(work_device)

    M = max_faces
    labels_full = torch.arange(M, device=work_device).unsqueeze(0).expand(N, -1).clone()
    labels_full[~mask_w] = M  # sentinel for padding

    if n_chunk < 1:
        n_chunk = 1

    max_iters = min(M, 32)

    for n_start in range(0, N, n_chunk):
        n_end = min(n_start + n_chunk, N)
        nc = n_end - n_start

        pf = pf_w[n_start:n_end]                        # (nc, M)
        nbr = nbr_w[n_start:n_end]                      # (nc, M, 3)
        cm = mask_w[n_start:n_end]                      # (nc, M)

        # Build (nc, M, M) adjacency matrix for this chunk.
        neighbors_exp = nbr.unsqueeze(-1)               # (nc, M, 3, 1)
        faces_exp = pf.unsqueeze(1).unsqueeze(2)        # (nc, 1, 1, M)
        match = (neighbors_exp == faces_exp) & cm.unsqueeze(1).unsqueeze(2)  # (nc, M, 3, M)
        adj = match.any(dim=2)                          # (nc, M, M)
        adj = adj | adj.transpose(1, 2)
        adj = adj & cm.unsqueeze(2) & cm.unsqueeze(1)

        # Iterative label prop on just this chunk.
        chunk_labels = labels_full[n_start:n_end].clone()  # (nc, M) int64
        for _ in range(max_iters):
            old_labels = chunk_labels.clone()
            adj_labels = chunk_labels.unsqueeze(1).expand(nc, M, M).clone()
            adj_labels[~adj] = M
            min_neighbor = adj_labels.min(dim=2).values  # (nc, M)
            chunk_labels = torch.minimum(chunk_labels, min_neighbor)
            chunk_labels[~cm] = M
            if torch.equal(chunk_labels, old_labels):
                break

        labels_full[n_start:n_end] = chunk_labels

        del match, adj, chunk_labels, old_labels, adj_labels, min_neighbor
        if work_device.type == "cuda":
            # Free the transient chunk tensors before the next iteration so
            # the allocator doesn't accumulate peak across chunks.
            torch.cuda.empty_cache()

    if labels_full.device != orig_dev:
        labels_full = labels_full.to(orig_dev)
    return labels_full


def _label_prop_dense_gpu(
    padded_faces: torch.Tensor,
    neighbors_of: torch.Tensor,
    mask: torch.Tensor,
    N: int,
    max_faces: int,
) -> torch.Tensor:
    """Tier 0: full-N GPU (single chunk covering all N cubes).

    Bit-exact with the original non-chunked dense path because a single chunk
    of size N reproduces the same tensor shapes and ops.
    """
    return _label_prop_chunk_impl(
        padded_faces, neighbors_of, mask, N, max_faces,
        n_chunk=N, work_device=padded_faces.device)


def _label_prop_chunked_gpu(
    padded_faces: torch.Tensor,
    neighbors_of: torch.Tensor,
    mask: torch.Tensor,
    N: int,
    max_faces: int,
    n_chunk: int,
) -> torch.Tensor:
    """Tier 1: chunked GPU. Peak per iter: n_chunk * M^2 * 12 bytes."""
    return _label_prop_chunk_impl(
        padded_faces, neighbors_of, mask, N, max_faces,
        n_chunk=n_chunk, work_device=padded_faces.device)


def _label_prop_chunked_cpu(
    padded_faces: torch.Tensor,
    neighbors_of: torch.Tensor,
    mask: torch.Tensor,
    N: int,
    max_faces: int,
    n_chunk: int,
) -> torch.Tensor:
    """Tier 2: chunked CPU. Ships tensors to host, processes, ships back."""
    cpu = torch.device("cpu")
    return _label_prop_chunk_impl(
        padded_faces, neighbors_of, mask, N, max_faces,
        n_chunk=n_chunk, work_device=cpu)


def _label_prop_chunked_gpu_adaptive(
    padded_faces: torch.Tensor,
    neighbors_of: torch.Tensor,
    mask: torch.Tensor,
    N: int,
    max_faces: int,
    max_chunk_bytes_override: int | None = None,
) -> torch.Tensor:
    """Adaptive chunked GPU with reactive OOM retry.

    Sizes n_chunk at 75% of free VRAM on first attempt, halves per retry
    (40%, 20%, 10%). After 4 OOMs, raises so the caller falls through to CPU.
    """
    dev = padded_faces.device
    M = max_faces
    per_cube_bytes = max(1, M * M * _BYTES_PER_CUBE_MULTIPLIER)

    if max_chunk_bytes_override is not None:
        n_chunk = max(1, max_chunk_bytes_override // per_cube_bytes)
        return _label_prop_chunked_gpu(
            padded_faces, neighbors_of, mask, N, max_faces, n_chunk)

    last_err: BaseException | None = None
    for frac in _ADAPTIVE_GPU_FRACTIONS:
        torch.cuda.empty_cache()
        try:
            free_vram, _total = torch.cuda.mem_get_info(dev)
        except Exception:
            free_vram = _DEFAULT_CHUNK_BYTES
        budget = max(per_cube_bytes, int(frac * free_vram))
        n_chunk = max(1, budget // per_cube_bytes)
        try:
            return _label_prop_chunked_gpu(
                padded_faces, neighbors_of, mask, N, max_faces, n_chunk)
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
            continue
    assert last_err is not None
    raise last_err


def _s2_dense_label_prop_chunked(
    padded_faces: torch.Tensor,     # (N, M) int32
    neighbors_of: torch.Tensor,     # (N, M, 3) int32
    mask: torch.Tensor,             # (N, M) bool
    N: int,
    max_faces: int,
    device: torch.device,
    *,
    max_chunk_bytes: int | None = None,
) -> torch.Tensor:
    """Three-tier OOM-resilient dense label-propagation for s2_components.

    Returns `labels (N, M) int64` with the same canonical labelling as the
    original non-chunked dense path. Output is bit-exact across Tier 0/1/2.

    Tier selection:
      - Tier 0 (full N on GPU) if nominal peak (N * M^2 * 12 B) ≤ 70% of
        free VRAM.
      - Tier 1 (adaptive chunked GPU) uses 75% of free VRAM on first attempt,
        halves on each OOM retry (40% / 20% / 10%); after 4 OOMs falls through.
      - Tier 2 (chunked CPU) final fallback.

    Env overrides:
      COREP_FAST_S2_DENSE_FORCE_CPU=1        force Tier 2 (CPU).
      COREP_FAST_S2_DENSE_FORCE_CHUNKED=1    force Tier 1 (chunked GPU).
      COREP_FAST_S2_DENSE_CHUNK_BYTES=<int>  fix chunk budget in bytes
                                             (disables adaptive sizing and
                                             reactive retry).
    """
    M = max_faces
    per_cube_bytes = max(1, M * M * _BYTES_PER_CUBE_MULTIPLIER)
    dense_bytes = N * per_cube_bytes

    force_cpu = os.environ.get("COREP_FAST_S2_DENSE_FORCE_CPU", "0") == "1"
    force_chunked = os.environ.get("COREP_FAST_S2_DENSE_FORCE_CHUNKED", "0") == "1"

    env_chunk_bytes: int | None = None
    env_chunk = os.environ.get("COREP_FAST_S2_DENSE_CHUNK_BYTES", "").strip()
    if env_chunk:
        try:
            env_chunk_bytes = int(env_chunk)
        except ValueError:
            env_chunk_bytes = None
    if max_chunk_bytes is None:
        max_chunk_bytes = env_chunk_bytes

    if force_cpu:
        n_chunk = max(1, _DEFAULT_CPU_BUDGET_BYTES // per_cube_bytes)
        return _label_prop_chunked_cpu(
            padded_faces, neighbors_of, mask, N, max_faces, n_chunk)

    if force_chunked:
        if max_chunk_bytes is not None:
            n_chunk = max(1, max_chunk_bytes // per_cube_bytes)
            return _label_prop_chunked_gpu(
                padded_faces, neighbors_of, mask, N, max_faces, n_chunk)
        return _label_prop_chunked_gpu_adaptive(
            padded_faces, neighbors_of, mask, N, max_faces)

    free_vram = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            free_vram, _total = torch.cuda.mem_get_info(device)
        except Exception:
            free_vram = None

    if free_vram is None:
        # No CUDA or mem_get_info failed — try dense, fall back to CPU on OOM.
        try:
            return _label_prop_dense_gpu(
                padded_faces, neighbors_of, mask, N, max_faces)
        except torch.cuda.OutOfMemoryError:
            n_chunk = max(1, _DEFAULT_CPU_BUDGET_BYTES // per_cube_bytes)
            return _label_prop_chunked_cpu(
                padded_faces, neighbors_of, mask, N, max_faces, n_chunk)

    # Tier 0: full-N GPU if nominal peak fits 70% of free VRAM.
    if dense_bytes <= int(0.7 * free_vram):
        try:
            return _label_prop_dense_gpu(
                padded_faces, neighbors_of, mask, N, max_faces)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    # Tier 1: adaptive chunked GPU.
    try:
        return _label_prop_chunked_gpu_adaptive(
            padded_faces, neighbors_of, mask, N, max_faces,
            max_chunk_bytes_override=max_chunk_bytes)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # Tier 2: chunked CPU fallback.
    n_chunk_cpu = max(1, _DEFAULT_CPU_BUDGET_BYTES // per_cube_bytes)
    return _label_prop_chunked_cpu(
        padded_faces, neighbors_of, mask, N, max_faces, n_chunk_cpu)
