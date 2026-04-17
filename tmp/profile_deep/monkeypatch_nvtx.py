"""Monkey-patch instrumentation for corep_fast pipeline profiling.

Two layers of instrumentation (must be applied before any corep_fast.* import):
  1. apply_stage_nvtx()      — NVTX ranges around 7 stage entry functions.
  2. apply_substage_events() — CUDA event timing around 11 sub-stage helpers.

Both apply_* are idempotent: re-calling re-wraps the true original function.
Call dump_substage_timings() after pipeline completion to collect sub-stage ms.

Patched functions carry `_nvtx_wrapped` (stage level) or `_event_wrapped`
(sub-stage level) attributes for introspection; the original is stored on
`_original`.
"""
import functools
import torch
import torch.cuda.nvtx as nvtx


# (module_path, function_name, nvtx_label)
_STAGE_ENTRIES = [
    ("corep_fast.stages.s1_voxelize", "s1_voxelize", "s1_voxelize"),
    ("corep_fast.stages.s2_components", "s2_components", "s2_components"),
    ("corep_fast.stages.s3_edge_weights", "s3_edge_weights", "s3_edge_weights"),
    ("corep_fast.stages.s4_face_point", "s4_face_point", "s4_face_point"),
    ("corep_fast.stages.s6_collapse", "s6_collapse", "s6_collapse"),
    ("corep_fast.stages.s7_rank_assign", "s7_rank_assign", "s7_rank_assign"),
    ("corep_fast.stages.s8_collapse", "decode_from_cubebatch", "s8_decode"),
]


def _make_nvtx_wrapper(fn, label):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nvtx.range_push(label)
        try:
            return fn(*args, **kwargs)
        finally:
            nvtx.range_pop()
    wrapper._nvtx_wrapped = label
    wrapper._original = fn
    return wrapper


def apply_stage_nvtx():
    """Patch each stage entry function. Idempotent: re-patches original."""
    import importlib
    for mod_path, fn_name, label in _STAGE_ENTRIES:
        mod = importlib.import_module(mod_path)
        original = getattr(mod, fn_name)
        # If already wrapped, rewrap the true original to keep one level.
        if hasattr(original, "_original"):
            original = original._original
        setattr(mod, fn_name, _make_nvtx_wrapper(original, label))


# Convenience for "import activates all patches"
if __name__ != "__main__":
    # Defer: caller must explicitly call apply_stage_nvtx() (makes test easier).
    pass


# Sub-stage CUDA event targets: (module_path, fn_name, label)
_SUBSTAGE_ENTRIES = [
    ("corep_fast.stages.s4_face_point", "_compute_face_weights_gpu", "s4/face_weights_gpu_entry"),
    ("corep_fast.stages.s4_face_point", "_expand_pairs_gpu",         "s4/expand_pairs"),
    ("corep_fast.stages.s4_face_point", "_batch_plane_tri_with_clip","s4/plane_tri_clip"),
    # NOTE: CPU-only (multiprocessing.Pool); CUDA event here only captures final H2D
    # memcpy, NOT the CPU wall-time. Also normally bypassed when USE_GPU_FW_S4=1
    # (default). Interpret total_ms ≈ 0 as "path not taken", not "fast path".
    ("corep_fast.stages.s4_face_point", "_compute_face_weights_mp",  "s4/face_weights_mp"),
    ("corep_fast.stages.s4_face_point", "_compute_component_points_gpu", "s4/component_points"),
    ("corep_fast.stages.s6_collapse",   "_fastpath_gpu_build_adjacency", "s6/fastpath_adj_build"),
    ("corep_fast.stages.s6_collapse",   "_fastpath_trace_loops_numpy",   "s6/fastpath_trace"),
    ("corep_fast.stages.s6_collapse",   "_collapse_with_uturns",         "s6/slowpath_uturns"),
    ("corep_fast.stages.s6_collapse",   "_collapse_with_uturns_tracked", "s6/slowpath_tracked"),
    ("corep_fast.stages.s7_rank_assign", "_build_adjacency_gpu",     "s7/adj_build_gpu"),
    ("corep_fast.stages.s7_rank_assign", "_phase1_gpu_rank_assign",  "s7/phase1_gpu_bfs"),
]


# Accumulator: label -> list of (start, end) Event pairs (one per call).
_SUBSTAGE_TIMINGS: dict[str, list] = {}


def _make_event_wrapper(fn, label):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            end.record()
            # Defer sync — accumulate events; sync + elapsed_time happens at dump.
            _SUBSTAGE_TIMINGS.setdefault(label, []).append((start, end))
    wrapper._event_wrapped = label
    wrapper._original = fn
    return wrapper


def apply_substage_events():
    """Patch sub-stage functions with CUDA event timing."""
    import importlib
    for mod_path, fn_name, label in _SUBSTAGE_ENTRIES:
        mod = importlib.import_module(mod_path)
        original = getattr(mod, fn_name)
        if hasattr(original, "_original"):
            original = original._original
        setattr(mod, fn_name, _make_event_wrapper(original, label))


def dump_substage_timings() -> dict[str, dict]:
    """Sync CUDA, compute elapsed ms per (start, end) pair; return summary dict.

    Returns: {label: {count, total_ms, per_call_ms: [...]}}.
    """
    torch.cuda.synchronize()
    out: dict = {}
    for label, pairs in _SUBSTAGE_TIMINGS.items():
        per_call = [s.elapsed_time(e) for s, e in pairs]
        out[label] = {
            "count": len(per_call),
            "total_ms": sum(per_call),
            "per_call_ms": per_call,
        }
    return out
