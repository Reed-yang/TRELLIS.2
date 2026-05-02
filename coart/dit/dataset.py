"""Cached dataset for shape DiT finetune.

Reads pre-cached fp16 DINOv3 features + shape-VAE latent npz from
``coart_dit_data_v0/`` instead of running DINO + the encoder online.

Layout expected under ``roots`` (passed positionally to ``__init__``):

    roots/
      instances_10k.csv          # sha256, aesthetic_score, ...
      manifest.csv               # sha256, render_done, dino_done, slat_done, ...
      dino_l16_s512/<sha>.npz    # features (16, T, 1024) fp16, view_idx (16,)
      slat/<vae_tag>/<sha>.npz   # coords (N, 3) int16 @ res=512, feats (M, 32) fp16

Notes:
  * The cached SLat npz stores the *input* surface-voxel coords at the input
    resolution (512); the latent feats live on a 16x-downsampled grid. We
    reconstruct latent coords as ``unique(coords // 16)`` which matches the
    sort order produced by ``trellis2.modules.sparse.spatial.basic.SparseDownsample``
    (verified: torch ``code.unique`` returns sorted values, identical to
    ``np.unique(..., axis=0)`` over the downsampled coord tuples).
  * Conditioning is the cached DINO features for one *random* view per sample
    (mirrors the original ``ImageConditionedMixin`` random-view selection).

The class is registered into ``trellis2.datasets`` namespace at import time so
``train.py`` can resolve it generically via ``getattr(datasets, cfg.dataset.name)``.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from trellis2.modules.sparse import SparseTensor
from trellis2.utils.data_utils import load_balanced_group_indices


__all__ = ["CachedImageConditionedSLatShape"]


class CachedImageConditionedSLatShape(Dataset):
    """Cached image-conditioned SLat-shape dataset.

    Args:
        roots: Path to ``coart_dit_data_v0/`` root (single string; comma-list also
            tolerated for parity with ``StandardDatasetBase`` callers).
        resolution: Shape resolution (informational, used by SLatShape vis only;
            kept for config parity).
        image_size: Stored only for parity with the official
            ``ImageConditionedMixin`` config; the cached DINO features were
            extracted at this size (default 512).
        min_aesthetic_score: Filter ``instances_10k.csv`` by aesthetic.
        max_tokens: Drop assets whose latent-token count exceeds this (matches
            the official trainer's GPU-memory ceiling).
        normalization: Dict with ``mean`` (32,) and ``std`` (32,) used to
            (re-)normalise the cached fp16 latents at load time. Pass ``None``
            to skip (the cached latents already had the encoder-time stats
            applied; pass the official DiT-config mean/std to *re-normalise*
            from encoder-stats space to DiT-stats space — consult callers).
        dino_dir: Sub-path under ``roots`` for DINO npz dir (default
            ``dino_l16_s512``).
        slat_dir: Sub-path under ``roots`` for SLat npz dir, including the
            ``vae_tag`` segment (default
            ``slat/vae_three_branch_ws_v0_ema_s0155000``).
        instances_csv: Sub-path under ``roots`` for the eligibility CSV
            (default ``instances_10k.csv``).
        latent_factor: Downsample factor from input grid (res=512) to latent
            grid (res=32). Default 16.
        latent_resolution: Spatial extent of the latent grid (default 32).
    """

    def __init__(
        self,
        roots: str,
        *,
        resolution: int = 512,
        image_size: int = 512,
        min_aesthetic_score: float = 4.5,
        max_tokens: int = 8192,
        normalization: Optional[Dict[str, List[float]]] = None,
        dino_dir: str = "dino_l16_s512",
        slat_dir: str = "slat/vae_three_branch_ws_v0_ema_s0155000",
        instances_csv: str = "instances_10k.csv",
        latent_factor: int = 16,
        latent_resolution: int = 32,
        # Accept (and ignore) the official SLatShape pretrained_slat_dec
        # so JSON configs cloned from the upstream config don't break.
        pretrained_slat_dec: Optional[str] = None,
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        super().__init__()
        # Tolerate comma-separated roots like StandardDatasetBase does.
        self.roots = roots.split(",") if "," in roots else [roots]
        self.resolution = resolution
        self.image_size = image_size
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.dino_dir = dino_dir
        self.slat_dir = slat_dir
        self.instances_csv = instances_csv
        self.latent_factor = latent_factor
        self.latent_resolution = latent_resolution
        # Stored for parity (vis path is not used by the trainer's training_losses).
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt

        if normalization is not None:
            self.mean = torch.tensor(normalization["mean"], dtype=torch.float32).reshape(1, -1)
            self.std = torch.tensor(normalization["std"], dtype=torch.float32).reshape(1, -1)
        else:
            self.mean = None
            self.std = None

        self.instances: List[Tuple[str, str]] = []  # (root, sha)
        self._stats: Dict[str, Dict[str, int]] = {}
        for root in self.roots:
            stats: Dict[str, int] = {}
            shas = self._enumerate_root(root, stats)
            self._stats[os.path.basename(root.rstrip("/"))] = stats
            self.instances.extend((root, sha) for sha in shas)

    # ------------------------------------------------------------------ enumeration

    def _enumerate_root(self, root: str, stats: Dict[str, int]) -> List[str]:
        """Filesystem-driven eligibility: keep sha with both DINO and SLat caches.

        Aesthetic filtering uses ``instances_csv`` if present; otherwise we just
        include every sha that survives the on-disk existence check.
        """
        import pandas as pd

        instances_path = os.path.join(root, self.instances_csv)
        if os.path.isfile(instances_path):
            df = pd.read_csv(instances_path, dtype={"sha256": str})
            stats["Total"] = len(df)
            df = df[df["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(df)
            candidate_shas = list(df["sha256"].astype(str).values)
        else:
            slat_root = os.path.join(root, self.slat_dir)
            candidate_shas = [
                fn[:-4] for fn in os.listdir(slat_root) if fn.endswith(".npz")
            ] if os.path.isdir(slat_root) else []
            stats["Total (no instances csv)"] = len(candidate_shas)

        dino_root = os.path.join(root, self.dino_dir)
        slat_root = os.path.join(root, self.slat_dir)

        kept: List[str] = []
        n_token_drop = 0
        for sha in candidate_shas:
            dino_p = os.path.join(dino_root, f"{sha}.npz")
            slat_p = os.path.join(slat_root, f"{sha}.npz")
            if not (os.path.isfile(dino_p) and os.path.isfile(slat_p)):
                continue
            # Cheap token-count check — read only the slat npz header. Use
            # mmap_mode='r' to avoid full decode for the gating step.
            try:
                with np.load(slat_p) as z:
                    n_tokens = int(z["feats"].shape[0])
            except (KeyError, ValueError, OSError):
                continue
            if n_tokens > self.max_tokens:
                n_token_drop += 1
                continue
            kept.append(sha)

        stats["Has dino + slat caches"] = len(kept) + n_token_drop
        stats[f"latent tokens <= {self.max_tokens}"] = len(kept)
        return kept

    # ------------------------------------------------------------------ accessors

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            root, sha = self.instances[index]
            return self._get_instance(root, sha)
        except Exception as e:  # noqa: BLE001 — match SLat behaviour
            print(f"[CachedImageConditionedSLatShape] error loading {sha!r}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

    def _get_instance(self, root: str, sha: str) -> Dict[str, Any]:
        # 1) latent + reconstructed latent coords
        slat_p = os.path.join(root, self.slat_dir, f"{sha}.npz")
        with np.load(slat_p) as z:
            input_coords = z["coords"].astype(np.int32)  # (N, 3) at input grid
            feats = z["feats"].astype(np.float32)  # (M, 32)

        # Downsample input coords -> latent coords (sort order matches
        # SparseDownsample: torch.unique on encoded codes returns sorted).
        latent_coords_np = np.unique(input_coords // self.latent_factor, axis=0)
        if latent_coords_np.shape[0] != feats.shape[0]:
            raise RuntimeError(
                f"latent coord count mismatch for {sha}: "
                f"got {latent_coords_np.shape[0]} unique downsampled coords "
                f"vs {feats.shape[0]} cached feats"
            )
        coords = torch.from_numpy(latent_coords_np).int()
        feats_t = torch.from_numpy(feats)
        if self.mean is not None and self.std is not None:
            feats_t = (feats_t - self.mean) / self.std

        # 2) cached DINO features — pick a random view of the 16
        dino_p = os.path.join(root, self.dino_dir, f"{sha}.npz")
        with np.load(dino_p) as z:
            features = z["features"]  # (16, T, 1024) fp16
            n_views = features.shape[0]
            view_idx = int(np.random.randint(n_views))
            cond = torch.from_numpy(features[view_idx].astype(np.float32))

        return {
            "coords": coords,
            "feats": feats_t,
            "cond": cond,
        }

    # ------------------------------------------------------------------ collate

    @staticmethod
    def collate_fn(batch, split_size=None):
        """Mirror ``trellis2.datasets.structured_latent.SLat.collate_fn``.

        Produces packs of ``{x_0: SparseTensor, cond: (B, T, 1024)}``.
        """
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b["coords"].shape[0] for b in batch], split_size
            )
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack: Dict[str, Any] = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(
                    torch.cat(
                        [
                            torch.full(
                                (b["coords"].shape[0], 1), i, dtype=torch.int32
                            ),
                            b["coords"],
                        ],
                        dim=-1,
                    )
                )
                feats.append(b["feats"])
                layout.append(slice(start, start + b["coords"].shape[0]))
                start += b["coords"].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack["x_0"] = SparseTensor(coords=coords, feats=feats)
            pack["x_0"]._shape = torch.Size(
                [len(group), *sub_batch[0]["feats"].shape[1:]]
            )
            pack["x_0"].register_spatial_cache("layout", layout)

            # Stack auxiliary keys
            keys = [k for k in sub_batch[0].keys() if k not in ("coords", "feats")]
            for k in keys:
                vals = [b[k] for b in sub_batch]
                if isinstance(vals[0], torch.Tensor):
                    pack[k] = torch.stack(vals)
                elif isinstance(vals[0], list):
                    pack[k] = sum(vals, [])
                else:
                    pack[k] = vals
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs

    # ------------------------------------------------------------------ misc

    def __str__(self) -> str:
        lines = [self.__class__.__name__, f"  - Total instances: {len(self)}"]
        lines.append("  - Sources:")
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for k, v in stats.items():
                lines.append(f"      - {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- registration
# Register into trellis2.datasets so train.py's getattr(datasets, cfg.dataset.name)
# resolves us. The package uses a __getattr__ lazy loader, so direct
# globals() insertion is sufficient (and idempotent).
def _register() -> None:
    import trellis2.datasets as td

    setattr(td, "CachedImageConditionedSLatShape", CachedImageConditionedSLatShape)


_register()
