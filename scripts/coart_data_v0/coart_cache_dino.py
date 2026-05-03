#!/usr/bin/env python
"""Cache DINOv3 ViT-L/16 features per asset (16 views x T tokens x 1024 dim, fp16).

Per-rank sharding via stable_shard; resume by skipping existing output.
Atomic write via os.replace. See spec sections 4.2.2 + 5.4.

Usage:
  CUDA_VISIBLE_DEVICES=$rank python scripts/coart_data_v0/coart_cache_dino.py \\
    --instances <coart_data_root>/instances_10k.csv \\
    --renders_dir <coart_data_root>/renders_cond \\
    --out_dir <coart_data_root>/dino_l16_s512 \\
    --rank R --world_size W
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Optional

# Make trellis2 importable when invoked via `python scripts/.../foo.py`.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_LOCAL_PATH = "/mnt/novita2/siyuan/workspace/TRELLIS.2/pretrained/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"


def _resolve_dinov3_path(model_id: str) -> str:
    """Prefer local ckpt over HF hub (gated repo workaround). Return canonical
    model_id if local path lacks config.json."""
    if os.path.isdir(DINO_LOCAL_PATH) and os.path.isfile(os.path.join(DINO_LOCAL_PATH, "config.json")):
        return DINO_LOCAL_PATH
    return model_id


def stable_shard(sha: str, world_size: int, rank: int) -> bool:
    """Stable hash via sha[:8] hex; PYTHONHASHSEED-independent."""
    return (int(sha[:8], 16) % world_size) == rank


def atomic_savez(out_path: str, **arrays) -> None:
    # np.savez_compressed auto-appends ".npz" if missing, so include it in tmp
    # to keep the post-save filename predictable for os.replace.
    tmp = f"{out_path}.tmp.{os.getpid()}.{int(time.time()*1e6)}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)


def build_extractor(image_size: int):
    """Build the real DinoV3 extractor (lazy import; called only at production time)."""
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    path = _resolve_dinov3_path(DINO_MODEL_ID)
    extractor = DinoV3FeatureExtractor(model_name=path, image_size=image_size)
    extractor.cuda()
    return extractor


def _load_view(png_path: str, image_size: int) -> torch.Tensor:
    """Match ImageConditionedMixin.get_instance preprocessing
    (trellis2/datasets/components.py:113-129)."""
    img = Image.open(png_path)
    if "A" not in img.mode:
        img = img.convert("RGBA")
    alpha = np.array(img.getchannel(3))
    bb = alpha.nonzero()
    if bb[0].size == 0:
        img = img.resize((image_size, image_size), Image.LANCZOS)
    else:
        bbox = [bb[1].min(), bb[0].min(), bb[1].max(), bb[0].max()]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        h = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        crop = [int(cx - h), int(cy - h), int(cx + h), int(cy + h)]
        img = img.crop(crop).resize((image_size, image_size), Image.LANCZOS)
    a = np.array(img.getchannel(3), dtype=np.float32) / 255.0
    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    rgb = rgb * a[..., None]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


@torch.no_grad()
def process_one(
    sha: str,
    renders_dir: str,
    out_dir: str,
    extractor,
    image_size: int = 512,
) -> tuple[bool, str]:
    out_path = os.path.join(out_dir, f"{sha}.npz")
    if os.path.exists(out_path):
        return True, "skip-exists"
    asset_dir = os.path.join(renders_dir, sha)
    if not os.path.isdir(asset_dir):
        return False, "no-renders-dir"
    pngs = sorted(p for p in os.listdir(asset_dir) if p.endswith(".png"))
    if len(pngs) != 16:
        return False, f"expected 16 views, got {len(pngs)}"

    views = torch.stack([_load_view(os.path.join(asset_dir, p), image_size) for p in pngs])
    if torch.cuda.is_available():
        views = views.cuda()
    feats = extractor(views)
    feats_f16 = feats.to(torch.float16).cpu().numpy()
    T = feats_f16.shape[1]

    atomic_savez(
        out_path,
        features=feats_f16,
        view_idx=np.arange(16, dtype=np.uint8),
        n_tokens=np.int32(T),
        model_id=np.array(DINO_MODEL_ID),
        image_size=np.int32(image_size),
    )
    return True, f"ok T={T}"


def run(
    instances: str,
    renders_dir: str,
    out_dir: str,
    rank: int,
    world_size: int,
    image_size: int = 512,
    limit: Optional[int] = None,
) -> None:
    print(f"[dino] rank={rank}/{world_size} python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(instances, dtype={"sha256": str})
    todo = [s for s in df["sha256"] if stable_shard(s, world_size, rank)]
    if limit:
        todo = todo[:limit]
    print(f"[dino] rank={rank}: {len(todo)} sha to process")

    extractor = build_extractor(image_size)
    n_ok, n_fail, n_skip = 0, 0, 0
    for sha in tqdm(todo, desc=f"rank{rank}"):
        ok, msg = process_one(sha, renders_dir, out_dir, extractor, image_size)
        if not ok:
            n_fail += 1
            print(f"[dino] FAIL {sha}: {msg}", file=sys.stderr)
        elif msg == "skip-exists":
            n_skip += 1
        else:
            n_ok += 1
    print(f"[dino] rank={rank} done: ok={n_ok} skip={n_skip} fail={n_fail}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True)
    p.add_argument("--renders_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    run(**vars(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
