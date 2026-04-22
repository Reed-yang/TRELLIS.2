"""BucketedDistributedSampler — same-voxel-count batches across DDP ranks.

Sort samples by voxel count, reshape into rows of (num_replicas * batch_size),
shuffle row order every epoch. Within any given global batch, all ranks see
similarly-sized samples -> limits straggler-rank tail.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class BucketedDistributedSampler(Sampler):
    """Emit indices so that within each global batch (``num_replicas * batch_size``)
    samples have similar voxel counts, limiting the 'slowest rank' tail.

    Strategy:
      1. Sort all sample indices by ``voxels_per_sample`` (stable ascending).
      2. Reshape into rows of ``num_replicas * batch_size`` (drop last tail).
      3. Shuffle the *row order* every epoch (so during any given global batch
         all 8 ranks see similarly-sized samples, but across the whole epoch
         batches still span small → large → small in a random walk).
      4. Within each row, shuffle the column order so rank assignment is
         randomized.
      5. Slice per-rank: rank ``r`` gets ``indices[r::num_replicas]``.
    """

    def __init__(
        self,
        voxels_per_sample: np.ndarray,
        num_replicas: int,
        rank: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        sort_mode: str = "shuffle",
    ):
        if sort_mode not in ("shuffle", "ascending"):
            raise ValueError(
                f"BucketedDistributedSampler: unknown sort_mode={sort_mode!r}; "
                "expected 'shuffle' or 'ascending'."
            )
        self.voxels = np.asarray(voxels_per_sample, dtype=np.int64)
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.sort_mode = sort_mode
        self.epoch = 0

        self._row = num_replicas * batch_size
        self._kept = (len(self.voxels) // self._row) * self._row
        if self._kept <= 0:
            raise ValueError(
                f"BucketedDistributedSampler: only {len(self.voxels)} samples, "
                f"need at least num_replicas*batch_size={self._row}."
            )
        self._len_per_rank = self._kept // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len_per_rank

    def __iter__(self):
        sorted_idx = np.argsort(self.voxels, kind="stable")[: self._kept]
        rows = sorted_idx.reshape(-1, self._row)
        g = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle and self.sort_mode == "shuffle":
            # Fully random row order, plus per-row column shuffle.
            rows = rows[g.permutation(rows.shape[0])]
            col_perm = np.stack(
                [g.permutation(self._row) for _ in range(rows.shape[0])], axis=0
            )
            rows = np.take_along_axis(rows, col_perm, axis=1)
        elif self.sort_mode == "ascending":
            # Keep rows in ascending voxel-count order across each epoch
            # (this removes the big-bucket "stall bursts" and lets triton
            # autotune / allocator warm up monotonically). Still shuffle the
            # per-row column assignment every epoch so ranks don't always
            # see the same end of a bucket.
            if self.shuffle:
                col_perm = np.stack(
                    [g.permutation(self._row) for _ in range(rows.shape[0])], axis=0
                )
                rows = np.take_along_axis(rows, col_perm, axis=1)
        flat = rows.reshape(-1)
        return iter(flat[self.rank :: self.num_replicas].tolist())
