"""coart — corep-related finetune training sub-package.

Sub-packages:
    common/   shared training infra (EMA, checkpoint, dist, logging, patches)
    data/     shared data pipeline (Feat18Dataset, BucketedSampler, stats)
    vae/      Shape-VAE feat18 finetune task (`python -m coart.vae`)
    dit/      placeholder for future DiT / flow-matching finetune

See docs/superpowers/specs/2026-04-22-coart-vae-feat18-design.md for design.
"""
