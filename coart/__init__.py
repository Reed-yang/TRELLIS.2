"""coart — corep-related finetune training sub-package.

Sub-packages:
    common/   shared training infra (EMA, checkpoint, dist, logging, patches)
    data/     shared data pipeline (Feat18Dataset, BucketedSampler, stats)
    vae/      Shape-VAE feat18 finetune task (`python -m coart.vae`)
    dit/      placeholder for future DiT / flow-matching finetune

See docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md for design.

Side effect: sets TRITON_CACHE_DIR default to `<repo_root>/.cache/triton` so
Triton JIT caches are NFS-shared across nodes / persist across runs. Covered
by the existing `.cache` gitignore rule. Users can override by exporting
TRITON_CACHE_DIR before importing coart.
"""
import os as _os

_COART_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.environ.setdefault(
    "TRITON_CACHE_DIR", _os.path.join(_COART_ROOT, ".cache", "triton")
)
del _os, _COART_ROOT
