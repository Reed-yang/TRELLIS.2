#!/usr/bin/env python
"""Cache coart shape SLat latent (encoder mu) per asset, versioned by VAE tag.

Per-rank sharding via stable_shard; resume by skipping existing output.
Atomic write via os.replace. See spec sections 4.2.3 + 5.5.

Usage:
  CUDA_VISIBLE_DEVICES=$rank python scripts/coart_data_v0/coart_cache_slat.py \\
    --instances <coart_data_root>/instances_10k.csv \\
    --feat18_dir <dataset_root>/feat18_512/data \\
    --vae_ckpt   results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \\
    --vae_tag    vae_three_branch_ws_v0_ema_s0155000 \\
    --vae_io_arch three_branch \\
    --stats_npz  /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/stats_global.npz \\
    --out_root <coart_data_root>/slat \\
    --rank R --world_size W
"""
from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
import time
from typing import Optional

# Make coart / trellis2 importable when invoked via `python scripts/.../foo.py`.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def stable_shard(sha: str, world_size: int, rank: int) -> bool:
    """Stable hash via sha[:8] hex; PYTHONHASHSEED-independent."""
    return (int(sha[:8], 16) % world_size) == rank


def atomic_savez(out_path: str, **arrays) -> None:
    # IMPORTANT: tmp filename MUST end in .npz so np.savez_compressed doesn't
    # auto-append a different name and break os.replace (verified via cache_dino fix).
    tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time()*1e6)}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)


def _nullctx():
    return contextlib.nullcontext()


def load_encoder(vae_ckpt: str, vae_io_arch: str, stats_npz: str, device: str = "cuda"):
    """Build encoder via coart.vae.build, load EMA or raw ckpt, return (encoder, mean, std).

    EMA ckpts are wrapped: {'decay': float, 'shadow': list[Tensor in
    model.parameters() order]} and require coart.common.ema.EMAModel to apply
    them (shadow is a list aligned with model.parameters(), NOT a state_dict).
    Raw ckpts are flat state_dicts loaded via encoder.load_state_dict.
    """
    from coart.vae.build import build_models
    from coart.data.stats import load_stats
    from coart.common.ema import EMAModel

    encoder, _decoder = build_models(io_arch=vae_io_arch, device=device)
    state = torch.load(vae_ckpt, map_location=device, weights_only=True)

    is_ema = (
        isinstance(state, dict)
        and set(state.keys()) == {"decay", "shadow"}
        and isinstance(state["shadow"], list)
    )
    if is_ema:
        ema = EMAModel(encoder, decay=float(state["decay"]))
        ema.load_state_dict(state)
        ema.copy_to(encoder)
        print(
            f"[slat] EMA ckpt loaded (decay={state['decay']}, n_params={len(state['shadow'])})",
            file=sys.stderr,
        )
    else:
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(
            f"[slat] raw ckpt loaded (missing={len(missing)} unexpected={len(unexpected)})",
            file=sys.stderr,
        )
        if len(missing) > 10:
            raise RuntimeError(
                f"too many missing keys ({len(missing)}); ckpt format mismatch. "
                f"first missing: {missing[:5]}"
            )
    encoder.eval()
    mean, std = load_stats(stats_npz, torch.device(device), verbose=True)
    return encoder, mean, std


@torch.no_grad()
def process_one(
    sha: str,
    feat18_dir: str,
    out_dir: str,
    encoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    vae_ckpt_rel: str,
    vae_io_arch: str,
) -> tuple[bool, str]:
    out_path = os.path.join(out_dir, f"{sha}.npz")
    if os.path.exists(out_path):
        return True, "skip-exists"

    feat_path = os.path.join(feat18_dir, f"{sha}.npz")
    if not os.path.isfile(feat_path):
        return False, "no-feat18"

    with np.load(feat_path) as z:
        cube_indices = z["cube_indices"].astype(np.int32)  # (N, 3)
        feats_raw = z["feats"].astype(np.float32)          # (N, 18)
    N = cube_indices.shape[0]
    if N == 0:
        return False, "empty-feat18"

    import trellis2.modules.sparse as sp
    from coart.data.stats import normalize

    device = mean.device if hasattr(mean, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    coords_t = torch.from_numpy(cube_indices).int().to(device)
    coords_with_b = torch.cat(
        [torch.zeros((N, 1), dtype=torch.int32, device=coords_t.device), coords_t],
        dim=1,
    )
    feats_t = torch.from_numpy(feats_raw).to(device)
    feats_n = normalize(feats_t, mean, std)
    x = sp.SparseTensor(feats=feats_n, coords=coords_with_b)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else _nullctx()
    )
    with autocast_ctx:
        z_out = encoder(x, sample_posterior=False)

    # SparseUnetVae downsamples 16x: input (N_input, 3) at res 512
    # -> latent (N_latent, 3) at res 32. Save the latent coords + feats so the
    # downstream DiT trainer can reconstruct the SparseTensor without needing
    # to know the encoder's downsampling pattern. z_out.coords is (N_latent, 4)
    # with batch_idx in col 0; we strip it.
    latent_coords = z_out.coords[:, 1:4].detach().cpu().numpy().astype(np.int16)
    mu = z_out.feats.detach().to(torch.float16).cpu().numpy()
    N_latent = mu.shape[0]
    assert latent_coords.shape == (N_latent, 3), (
        f"latent_coords {latent_coords.shape} mismatch feats {mu.shape}"
    )

    atomic_savez(
        out_path,
        coords=latent_coords,
        feats=mu,
        num_voxels=np.int32(N_latent),
        num_input_voxels=np.int32(N),  # provenance: how many feat18 cubes fed in
        vae_ckpt_rel=np.array(vae_ckpt_rel),
        vae_io_arch=np.array(vae_io_arch),
    )
    return True, f"ok N_in={N} N_lat={N_latent}"


def run(
    instances: str,
    feat18_dir: str,
    vae_ckpt: str,
    vae_tag: str,
    vae_io_arch: str,
    stats_npz: str,
    out_root: str,
    rank: int,
    world_size: int,
    limit: Optional[int] = None,
) -> None:
    print(f"[slat] rank={rank}/{world_size} vae_tag={vae_tag}")
    out_dir = os.path.join(out_root, vae_tag)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(instances, dtype={"sha256": str})
    todo = [s for s in df["sha256"] if stable_shard(s, world_size, rank)]
    if limit:
        todo = todo[:limit]
    print(f"[slat] rank={rank}: {len(todo)} sha to process")

    encoder, mean, std = load_encoder(vae_ckpt, vae_io_arch, stats_npz)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vae_ckpt_rel = os.path.relpath(os.path.abspath(vae_ckpt), repo_root)
    n_ok, n_fail, n_skip = 0, 0, 0
    for sha in tqdm(todo, desc=f"rank{rank}"):
        ok, msg = process_one(
            sha, feat18_dir, out_dir, encoder, mean, std, vae_ckpt_rel, vae_io_arch
        )
        if not ok:
            n_fail += 1
            print(f"[slat] FAIL {sha}: {msg}", file=sys.stderr)
        elif msg == "skip-exists":
            n_skip += 1
        else:
            n_ok += 1
    print(f"[slat] rank={rank} done: ok={n_ok} skip={n_skip} fail={n_fail}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True)
    p.add_argument("--feat18_dir", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--vae_tag", required=True)
    p.add_argument("--vae_io_arch", default="three_branch")
    p.add_argument("--stats_npz", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
