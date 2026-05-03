"""coart — corep-related finetune training sub-package.

Sub-packages:
    common/   shared training infra (EMA, checkpoint, dist, logging, patches)
    data/     shared data pipeline (Feat18Dataset, BucketedSampler, stats)
    vae/      Shape-VAE feat18 finetune task (`python -m coart.vae`)
    dit/      placeholder for future DiT / flow-matching finetune

See docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md for design.

Side effect: sets TRITON_CACHE_DIR default to a per-rank subdir under
`<repo_root>/.cache/triton/rank-<N>/`. NFS atomic-rename races otherwise
(Errno 39 ENOTEMPTY) when 8 mp.spawn workers compute the same kernel hash
concurrently. Per-rank dir gives persistence + cross-run cache hit (rank
assignment stable per node) without contention. Override by exporting
TRITON_CACHE_DIR before importing coart.
"""
import os as _os
import multiprocessing as _mp

_COART_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def _detect_local_rank():
    # mp.spawn children have non-empty _identity tuple even at site.py time.
    ident = getattr(_mp.current_process(), "_identity", ())
    if ident:
        return str(ident[0] - 1)  # mp uses 1-indexed; map to 0-indexed rank
    # torchrun / SLURM srun set explicit env.
    for k in ("LOCAL_RANK", "SLURM_LOCALID"):
        v = _os.environ.get(k)
        if v is not None and v != "":
            return v
    return None  # main process — don't set, let spawn children own their rank


_rank = _detect_local_rank()
_base = _os.path.join(_COART_ROOT, ".cache", "triton")
if _rank is None:
    # Main process: use a separate "main" subdir so it doesn't collide with
    # rank-0 (which spawn child 0 will own). setdefault so user env wins.
    _os.environ.setdefault("TRITON_CACHE_DIR", _os.path.join(_base, "main"))
else:
    # Spawn / torchrun / SLURM child: ALWAYS override. Parent may have
    # leaked its TRITON_CACHE_DIR via env inheritance, which would put all
    # children in the same dir and re-create the NFS atomic-rename race.
    _os.environ["TRITON_CACHE_DIR"] = _os.path.join(_base, f"rank-{_rank}")
del _os, _mp, _COART_ROOT, _detect_local_rank, _rank, _base
