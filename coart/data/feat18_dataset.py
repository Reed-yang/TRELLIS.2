"""Feat18Dataset — precomputed corep .npz shards with deterministic val split.

Each .npz file (produced by precompute_feat18.py) has:
    cube_indices : (N, 3) int16
    feats        : (N, 18) float16
    num_boundary : (N,)   int8
    resolution   : scalar int32

Train / val split: `int(sha[:8], 16) % val_split_mod == 0` -> val, else train.
The split is deterministic and reproducible across machines.
"""
from __future__ import annotations

import glob
import os
import zipfile
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_npz_array_shape(path: str, array_name: str) -> tuple:
    """Read .npy header for one array inside a .npz without decompressing data.

    Much faster than np.load() when we only need the row count (for bucket sampling).
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open(f"{array_name}.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


def _is_val_sha(sha: str, mod: int) -> bool:
    """Hash-based deterministic val assignment: sha[:8] hex % mod == 0."""
    if mod <= 0:
        return False
    try:
        h = int(sha[:8], 16)
    except ValueError:
        h = abs(hash(sha))
    return (h % mod) == 0


class Feat18Dataset(Dataset):
    """Loads precomputed .npz shards and applies random integer translation augmentation.

    Args:
        data_dir: directory containing *.npz shards.
        resolution: grid resolution (must match precompute_feat18.py setting).
        max_translate: +/- shift range in voxel units (0 = disable augmentation).
        augment: if True, apply random translation; else deterministic identity.
        precompute_voxel_counts: if True, read .npy headers to cache per-sample N.
        max_voxels: if > 0, subsample each sample down to this cap (sha-seeded).
        val_split_mod: if > 0, determines train/val partition (see below).
        split: "train" or "val" or "all". If "train"/"val", filter by hash mod.

    val_split_mod examples:
        val_split_mod=200 -> ~0.5% held-out (200 samples from 41k)
        val_split_mod=0   -> split=ignored; all samples included regardless of `split`.
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        max_translate: int = 16,
        augment: bool = True,
        precompute_voxel_counts: bool = False,
        max_voxels: int = 0,
        val_split_mod: int = 0,
        split: str = "train",
    ):
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train/val/all, got {split!r}")

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"no .npz files in {data_dir}")

        if val_split_mod > 0 and split != "all":
            want_val = split == "val"
            filtered = [
                f for f in all_files
                if _is_val_sha(os.path.splitext(os.path.basename(f))[0], val_split_mod)
                is want_val
            ]
            if not filtered:
                raise RuntimeError(
                    f"split={split} val_split_mod={val_split_mod} produced empty set "
                    f"from {len(all_files)} total files"
                )
            self.files = filtered
        else:
            self.files = all_files

        self.resolution = resolution
        self.max_translate = max_translate
        self.augment = augment
        self.max_voxels = int(max_voxels)
        self.num_voxels: np.ndarray | None = None
        if precompute_voxel_counts:
            self.num_voxels = np.array(
                [int(_read_npz_array_shape(f, "cube_indices")[0]) for f in self.files],
                dtype=np.int64,
            )

    def effective_voxels(self) -> np.ndarray | None:
        """Per-sample voxel count after --max_voxels cap (for bucket sampler)."""
        if self.num_voxels is None:
            return None
        if self.max_voxels <= 0:
            return self.num_voxels
        return np.minimum(self.num_voxels, self.max_voxels)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        cube_indices = d["cube_indices"].astype(np.int32)
        feats = d["feats"].astype(np.float32)
        num_boundary = d["num_boundary"].astype(np.int32)
        sha = os.path.splitext(os.path.basename(self.files[idx]))[0]

        # Deterministic per-sample voxel subsample (sha-seeded).
        if self.max_voxels > 0 and cube_indices.shape[0] > self.max_voxels:
            try:
                seed = int(sha[:8], 16) & 0xFFFFFFFF
            except ValueError:
                seed = abs(hash(sha)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            keep = rng.choice(cube_indices.shape[0], size=self.max_voxels, replace=False)
            keep.sort()
            cube_indices = cube_indices[keep]
            feats = feats[keep]

        if self.augment and self.max_translate > 0:
            R = self.resolution
            min_idx = cube_indices.min(axis=0)
            max_idx = cube_indices.max(axis=0)
            shifts = np.empty(3, dtype=np.int32)
            for a in range(3):
                lo = max(-int(min_idx[a]), -self.max_translate)
                hi = min(R - 1 - int(max_idx[a]), self.max_translate)
                shifts[a] = np.random.randint(lo, hi + 1) if hi >= lo else 0
            cube_indices = cube_indices + shifts

        return {
            "cube_indices": cube_indices,
            "feats": feats,
            "num_boundary": num_boundary,
            "sha": sha,
        }


def collate_fn(batch):
    """Concatenate variable-length samples into a single SparseTensor payload.

    Produces:
        coords : (sum_N, 4) int32 - [batch_idx, x, y, z]
        feats  : (sum_N, 18) float32
        sizes  : list[int] per-sample voxel counts
        shas   : list[str] per-sample SHAs
        cube_indices_per_sample, num_boundary_per_sample (for eval mesh dump)
    """
    coords_chunks, feats_chunks, sizes = [], [], []
    cube_indices_list, num_boundary_list, shas = [], [], []
    for i, item in enumerate(batch):
        ci = item["cube_indices"]
        N = ci.shape[0]
        bi = np.full((N, 1), i, dtype=np.int32)
        coords_chunks.append(np.concatenate([bi, ci], axis=1))
        feats_chunks.append(item["feats"])
        sizes.append(N)
        cube_indices_list.append(ci)
        num_boundary_list.append(item["num_boundary"])
        shas.append(item["sha"])
    return {
        "coords": torch.from_numpy(np.concatenate(coords_chunks, axis=0)),
        "feats": torch.from_numpy(np.concatenate(feats_chunks, axis=0)),
        "sizes": sizes,
        "cube_indices_per_sample": cube_indices_list,
        "num_boundary_per_sample": num_boundary_list,
        "shas": shas,
    }
