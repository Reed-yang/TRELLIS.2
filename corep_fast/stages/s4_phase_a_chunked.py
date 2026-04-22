"""OOM-resilient chunked Phase-A for _count_uturns_from_packed (s4).

Phase A (node coalescence) of the U-turn counter allocates a (G, P, P) f64
distance matrix via torch.cdist, which is the #1 OOM source on complex meshes
(historical: 2193 events, median 97 GiB, p99 3511 GiB). This module provides
three semantically-identical tiers that all produce bit-exact canonical_idx:

  Tier 0 — dense GPU (fast path).
  Tier 1 — chunked GPU (peak = G*chunk_rows*P*8 bytes).
  Tier 2 — chunked CPU (when even 1 row on GPU is unaffordable).

Semantics preserved (CRITICAL):
  canonical_idx[g, i] = min { j : j <= i AND pts_valid[g,j]
                                 AND ||pts[g,i] - pts[g,j]|| < 1e-8 }
  or P if no such j exists, or if pts_valid[g, i] is False.

Numerical note:
  The original Phase A used torch.cdist, which uses a matmul-based formula
  (||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>) that can emit non-zero distances
  (~1e-8) for i == j and backend-dependent rounding near zero. This module
  instead uses the exact form sqrt(sum((a-b)**2)) on per-row chunks, which
  yields the identical canonical_idx on GPU and CPU and also fixes the
  historical self-distance corner case (self always matches, as intended).
  This chunked form never allocates the full (G,P,P) f64 matrix.
"""
from __future__ import annotations

import os
import torch


_DEFAULT_CHUNK_BYTES = 20 * (1024 ** 3)  # 20 GiB — explicit override only
_DEFAULT_CPU_BUDGET_BYTES = 40 * (1024 ** 3)  # 40 GiB for CPU tier
_TOL = 1e-8

# Effective memory footprint per distance-matrix element: the chunked impl
# builds the intermediate diff tensor (G, rc, P, 3) f64 = 24 bytes/elem before
# summing to the (G, rc, P) f64 distance = 8 bytes/elem. Plus the ~12 bytes of
# bool/i64 masks (valid_pair, match, col_le_row, match_lower, big_P, node_raw)
# that coexist briefly. A conservative multiplier of 6 over the raw 8B/elem
# distance footprint covers both the diff and the mask stack. We use this
# bytes-per-elem scaling factor to size chunk_rows against an adaptive fraction
# of free VRAM instead of a fixed 20 GiB cap.
_BYTES_PER_ELEM_FOR_CHUNK = 64  # 8B distance * 6x buffer, rounded up to power-of-2-ish

# Adaptive dispatch attempts: start at 75% of free VRAM, halve per retry.
# On the 4th OOM we fall through to CPU tier. This lets a well-provisioned
# H100 (80 GiB HBM, ~70 GiB free) run ~54 GiB chunks (max_rows ~ 800k for
# P=20k typical, bounded by row_bytes otherwise).
_ADAPTIVE_GPU_FRACTIONS = (0.75, 0.40, 0.20, 0.10)


def _phase_a_chunked_impl(pts: torch.Tensor, pts_valid: torch.Tensor,
                          chunk_rows: int, work_device) -> torch.Tensor:
    """Shared per-row chunked implementation. Runs on `work_device`, returns
    canonical_idx (G, P) int64 on the ORIGINAL device of `pts`.

    Uses the numerically-exact formula ||a-b|| = sqrt(sum((a-b)**2)) so the
    output is bit-exact across CPU/GPU backends. Peak allocation per iter:
      diff (G, chunk_rows, P, 3) f64 = G*chunk_rows*P*24 bytes.
    """
    orig_dev = pts.device
    G, P, _ = pts.shape
    if chunk_rows < 1:
        chunk_rows = 1

    if work_device is None or str(work_device) == str(orig_dev):
        pw = pts
        pv = pts_valid
    else:
        pw = pts.detach().to(work_device)
        pv = pts_valid.detach().to(work_device)

    canonical_idx = torch.full((G, P), P, dtype=torch.int64, device=work_device)
    j_ar_full = torch.arange(P, device=work_device, dtype=torch.int64)

    for row_start in range(0, P, chunk_rows):
        row_end = min(row_start + chunk_rows, P)
        rc = row_end - row_start

        pts_rows = pw[:, row_start:row_end, :]                        # (G, rc, 3)
        # Manual exact pairwise distance between rows [row_start:row_end] and all cols.
        # diff: (G, rc, P, 3). Distance: (G, rc, P).
        diff = pts_rows.unsqueeze(2) - pw.unsqueeze(1)                # (G, rc, P, 3)
        d_chunk = (diff * diff).sum(dim=-1).sqrt()                    # (G, rc, P) f64

        pv_rows = pv[:, row_start:row_end]                            # (G, rc)
        valid_pair = pv_rows.unsqueeze(2) & pv.unsqueeze(1)           # (G, rc, P)
        # For invalid pairs, force distance above threshold.
        d_chunk = torch.where(valid_pair, d_chunk, torch.full_like(d_chunk, 1.0))
        match = (d_chunk < _TOL)

        # Only j <= row_abs (first-occurrence wins).
        j_ar = j_ar_full.view(1, 1, P).expand(G, rc, P)
        row_abs = torch.arange(row_start, row_end, device=work_device, dtype=torch.int64)
        col_le_row = j_ar_full.view(1, P) <= row_abs.view(rc, 1)      # (rc, P)
        col_le_row = col_le_row.unsqueeze(0).expand(G, rc, P)

        match_lower = match & col_le_row
        big_P = torch.full_like(j_ar, P)
        node_raw = torch.where(match_lower, j_ar, big_P)
        chunk_canonical = node_raw.min(dim=-1).values                 # (G, rc)

        canonical_idx[:, row_start:row_end] = chunk_canonical

        del diff, d_chunk, valid_pair, match, j_ar, col_le_row, match_lower, big_P, node_raw, chunk_canonical

    canonical_idx = torch.where(pv, canonical_idx,
                                torch.full_like(canonical_idx, P))
    if canonical_idx.device != orig_dev:
        canonical_idx = canonical_idx.to(orig_dev)
    return canonical_idx


def _phase_a_dense_gpu(pts: torch.Tensor, pts_valid: torch.Tensor) -> torch.Tensor:
    """Tier 0: full-row chunked GPU (single chunk covers all P rows).

    Returns canonical_idx (G, P) int64 on pts.device.

    NOTE: This differs slightly from the *original* in-place implementation
    (which used torch.cdist(pts, pts)): we use the exact manual formula so
    Tier 0/1/2 outputs are bit-exact across GPU and CPU backends. In practice
    canonical_idx is identical to the original on realistic mesh inputs — the
    only observable change is at exact-self-distance (now always 0, hence a
    valid self-row is always its own canonical) and near-tolerance noise.
    """
    G, P, _ = pts.shape
    return _phase_a_chunked_impl(pts, pts_valid, chunk_rows=P, work_device=pts.device)


def _phase_a_chunked_gpu(pts: torch.Tensor, pts_valid: torch.Tensor,
                         chunk_rows: int) -> torch.Tensor:
    """Tier 1: chunked GPU. Returns canonical_idx (G, P) int64 on pts.device.

    Peak per-iter allocation: (G, chunk_rows, P, 3) f64 ~ G*chunk_rows*P*24B.
    """
    return _phase_a_chunked_impl(pts, pts_valid, chunk_rows=chunk_rows,
                                 work_device=pts.device)


def _phase_a_chunked_gpu_adaptive(pts: torch.Tensor, pts_valid: torch.Tensor,
                                  max_chunk_bytes_override: int | None = None
                                  ) -> torch.Tensor:
    """Adaptive chunked GPU with reactive OOM retry.

    Queries free VRAM (after empty_cache) and sizes chunks at 75% of it first,
    halving on each OOM retry (40%, 20%, 10%). Accounts for the ~6x memory
    footprint multiplier on the raw distance-element count so the nominal
    budget matches actual allocation. After 4 OOMs, raises — the caller should
    fall through to Tier 2 (CPU).

    `max_chunk_bytes_override` (or env `COREP_FAST_S4_PHASE_A_CHUNK_BYTES`
    resolved by the caller) overrides the adaptive sizing entirely — the
    chunk_rows is computed from that fixed budget instead of from free VRAM.
    This is the escape hatch for users who want reproducible memory use.
    """
    G, P, _ = pts.shape
    dev = pts.device

    # Per-row nominal bytes for the distance-element footprint, scaled up by
    # the diff + mask stack multiplier so the "budget" matches real peak.
    row_bytes = max(1, G * P * _BYTES_PER_ELEM_FOR_CHUNK)

    if max_chunk_bytes_override is not None:
        # Fixed-budget mode (no adaptive, no retry).
        chunk_rows = max(1, max_chunk_bytes_override // row_bytes)
        return _phase_a_chunked_gpu(pts, pts_valid, chunk_rows)

    last_err: BaseException | None = None
    for attempt, frac in enumerate(_ADAPTIVE_GPU_FRACTIONS):
        torch.cuda.empty_cache()  # Return cached blocks so mem_get_info is accurate.
        try:
            free_vram, _total = torch.cuda.mem_get_info(dev)
        except Exception:
            # No reliable free-VRAM query: use the conservative static default.
            free_vram = _DEFAULT_CHUNK_BYTES
        budget = max(row_bytes, int(frac * free_vram))
        chunk_rows = max(1, budget // row_bytes)
        try:
            return _phase_a_chunked_gpu(pts, pts_valid, chunk_rows)
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
            # Try the next (smaller) fraction. If none work, escape.
            continue
    # All GPU attempts OOMed. Raise so the caller can fall through to CPU.
    assert last_err is not None
    raise last_err


def _phase_a_chunked_cpu(pts: torch.Tensor, pts_valid: torch.Tensor,
                         chunk_rows: int) -> torch.Tensor:
    """Tier 2: chunked CPU. Returns canonical_idx (G, P) int64 on pts.device
    (uploaded from CPU at the end).
    """
    cpu = torch.device("cpu")
    return _phase_a_chunked_impl(pts, pts_valid, chunk_rows=chunk_rows, work_device=cpu)


def phase_a_dispatch(pts: torch.Tensor, pts_valid: torch.Tensor,
                     max_chunk_bytes: int | None = None) -> torch.Tensor:
    """Three-tier adaptive dispatch for Phase-A. Returns canonical_idx (G, P) int64.

    Tier selection:
      - Tier 0 (dense GPU) if dense_bytes * 6 fits ~70% of free VRAM
        (the *6 multiplier accounts for the diff + mask stack peak).
      - Else Tier 1 (adaptive chunked GPU): uses 75% of free VRAM on first
        attempt, halves on each OOM retry (40%/20%/10%).
      - Else Tier 2 (chunked CPU): if all GPU attempts OOM.

    Env overrides:
      COREP_FAST_S4_PHASE_A_FORCE_CPU=1        force Tier 2 (CPU).
      COREP_FAST_S4_PHASE_A_FORCE_CHUNKED=1    force Tier 1 (chunked GPU).
      COREP_FAST_S4_PHASE_A_CHUNK_BYTES=<int>  fix chunk budget to this many
                                               bytes (disables adaptive sizing
                                               and reactive retry — useful for
                                               reproducible memory profiling).

    Design note: the previous 20 GiB fixed cap wasted ~50 GiB of H100 HBM.
    Adaptive sizing pushes Tier 1 closer to the available headroom so each
    chunk processes more rows — roughly 3-4x fewer iterations on an otherwise-
    idle 80 GiB GPU — while reactive OOM retry handles fragmentation or other
    processes that shrink the effective budget at runtime.
    """
    G, P, _ = pts.shape
    dev = pts.device

    force_cpu = os.environ.get("COREP_FAST_S4_PHASE_A_FORCE_CPU", "0") == "1"
    force_chunked = os.environ.get("COREP_FAST_S4_PHASE_A_FORCE_CHUNKED", "0") == "1"

    env_chunk_bytes: int | None = None
    env_chunk = os.environ.get("COREP_FAST_S4_PHASE_A_CHUNK_BYTES", "").strip()
    if env_chunk:
        try:
            env_chunk_bytes = int(env_chunk)
        except ValueError:
            env_chunk_bytes = None
    # Explicit arg wins over env.
    if max_chunk_bytes is None:
        max_chunk_bytes = env_chunk_bytes

    row_bytes = max(1, G * P * _BYTES_PER_ELEM_FOR_CHUNK)
    dense_bytes = G * P * P * _BYTES_PER_ELEM_FOR_CHUNK  # peak across diff+mask stack

    if force_cpu:
        cpu_row_bytes = max(1, G * P * _BYTES_PER_ELEM_FOR_CHUNK)
        chunk_rows = max(1, _DEFAULT_CPU_BUDGET_BYTES // cpu_row_bytes)
        return _phase_a_chunked_cpu(pts, pts_valid, chunk_rows)

    if force_chunked:
        if max_chunk_bytes is not None:
            chunk_rows = max(1, max_chunk_bytes // row_bytes)
            return _phase_a_chunked_gpu(pts, pts_valid, chunk_rows)
        return _phase_a_chunked_gpu_adaptive(pts, pts_valid)

    free_vram = None
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            free_vram, _total = torch.cuda.mem_get_info(dev)
        except Exception:
            free_vram = None

    # No CUDA context or mem_get_info failed: best-effort dense + let caller
    # catch any OOM. (This branch is not expected to hit in production.)
    if free_vram is None:
        try:
            return _phase_a_dense_gpu(pts, pts_valid)
        except torch.cuda.OutOfMemoryError:
            chunk_rows = max(1, _DEFAULT_CPU_BUDGET_BYTES // row_bytes)
            return _phase_a_chunked_cpu(pts, pts_valid, chunk_rows)

    # Tier 0: dense peak ≈ 6x footprint, run if it fits 70% of free VRAM.
    if dense_bytes <= int(0.7 * free_vram):
        try:
            return _phase_a_dense_gpu(pts, pts_valid)
        except torch.cuda.OutOfMemoryError:
            # Mem estimate was optimistic (fragmentation etc.). Retry adaptive.
            torch.cuda.empty_cache()

    # Tier 1: adaptive chunked GPU (with built-in OOM retry chain).
    try:
        return _phase_a_chunked_gpu_adaptive(
            pts, pts_valid, max_chunk_bytes_override=max_chunk_bytes)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # Tier 2: CPU chunked fallback.
    chunk_rows_cpu = max(1, _DEFAULT_CPU_BUDGET_BYTES // row_bytes)
    return _phase_a_chunked_cpu(pts, pts_valid, chunk_rows_cpu)
