"""Monkey-patch NVTX ranges around corep_fast stage entry functions.

Must be imported BEFORE any corep_fast.* import. The `apply_*` functions are
idempotent — calling twice re-wraps the original (not the already-wrapped).

Design: wrap each stage entry, record attribute `_nvtx_wrapped = <name>` on the
patched function so the smoke test can detect successful patching.
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
