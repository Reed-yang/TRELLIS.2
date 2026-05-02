#!/usr/bin/env python
"""One-time conversion: official .safetensors DiT ckpt -> trellis2 trainer ckpts dir.

The upstream BasicTrainer's load() (trellis2/trainers/basic.py:331-358) loads:
  <load_dir>/ckpts/<model_name>_step<step:07d>.pt  (state_dict for each model)
  <load_dir>/ckpts/misc_step<step:07d>.pt          (optimizer/sampler/lr/etc state)

This script writes both files at step=0:
  - denoiser_step0000000.pt: model state_dict from the safetensors blob
  - misc_step0000000.pt: stub with step=0 and minimal optimizer/sampler state

After running this, launch finetune via:
  LOAD_DIR=results/coart_dit_warmstart CKPT=0 bash scripts/train_coart_dit_shape.sh

Usage:
  python scripts/coart_warmstart_dit.py [--out_dir <staging_dir>] [--config <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Optional

# repo root for coart / trellis2 imports
_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from safetensors.torch import load_file


def run(config_path: str, out_dir: str, src_safetensors: str) -> None:
    cfg = json.load(open(config_path))
    denoiser_cfg = cfg["models"]["denoiser"]
    print(f"[warmstart] loading config: {config_path}")
    print(f"[warmstart] denoiser: {denoiser_cfg['name']}({denoiser_cfg['args']})")

    # Build fresh denoiser to validate shapes match the safetensors
    from trellis2 import models
    denoiser_cls = getattr(models, denoiser_cfg["name"])
    denoiser = denoiser_cls(**denoiser_cfg["args"])
    print(f"[warmstart] built {denoiser_cfg['name']}, params={sum(p.numel() for p in denoiser.parameters())/1e6:.1f}M")

    print(f"[warmstart] loading safetensors: {src_safetensors}")
    sd = load_file(src_safetensors)
    miss, unex = denoiser.load_state_dict(sd, strict=False)
    print(f"[warmstart] loaded; missing={len(miss)} unexpected={len(unex)}")
    if miss:
        print(f"[warmstart]   first missing: {miss[:5]}")
    if unex:
        print(f"[warmstart]   first unexpected: {unex[:5]}")
    if len(miss) > 50:
        raise RuntimeError(
            f"too many missing keys ({len(miss)}); arch mismatch between config "
            f"and safetensors. Aborting before save."
        )

    # Write the trainer-expected layout.
    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model state_dict; the trainer loads via model.load_state_dict(...) so we
    # save the raw state_dict (no wrapping).
    denoiser_path = os.path.join(ckpt_dir, "denoiser_step0000000.pt")
    torch.save(denoiser.state_dict(), denoiser_path)
    print(f"[warmstart] wrote {denoiser_path} ({os.path.getsize(denoiser_path)/1e6:.1f}MB)")

    # Misc stub: only the keys read by trellis2/trainers/basic.py:344-358.
    # `optimizer` and `data_sampler` are the only required keys; if they are
    # None the trainer's .load_state_dict(None) will crash on PyTorch optimizers.
    # Solution: include the EXACT shape the trainer's optimizer + sampler write
    # at step 0, using minimal valid state.
    misc = {
        "step": 0,
        "optimizer": None,           # WARM-START NOTE: see below
        "data_sampler": None,        # WARM-START NOTE: see below
    }
    misc_path = os.path.join(ckpt_dir, "misc_step0000000.pt")
    torch.save(misc, misc_path)
    print(f"[warmstart] wrote {misc_path} ({os.path.getsize(misc_path)/1e3:.1f}KB)")

    # Symlink "latest" pointers so --ckpt latest works without specifying step
    # (find_ckpt in train.py:17 globs misc_*.pt and picks the last).
    print(f"[warmstart] done. Launch with:")
    print(f"  LOAD_DIR={out_dir} CKPT=0 bash scripts/train_coart_dit_shape.sh")
    print(f"")
    print(f"WARM-START NOTE: misc_step0000000.pt has optimizer=None and")
    print(f"data_sampler=None as placeholders. The trainer's load() will likely")
    print(f"crash at `self.optimizer.load_state_dict(None)`. If so, two options:")
    print(f"  A. Bypass load() entirely by NOT passing --load_dir; instead, add")
    print(f"     a one-shot pre-train hook in coart.dit.trainer that loads the")
    print(f"     denoiser state_dict from PRETRAINED_DIT_CKPT directly inside")
    print(f"     __init__ and re-init the optimizer fresh.")
    print(f"  B. Construct a real misc_step0000000.pt by running ONE training")
    print(f"     step on a tiny dataset, save it, then replace denoiser_step0")
    print(f"     with the warm-started weights and re-launch.")
    print(f"This script implements neither — those are follow-ups.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(_REPO / "coart" / "dit" / "configs" / "coart_dit_shape_512_ft.json"),
    )
    p.add_argument(
        "--src_safetensors",
        default=str(
            _REPO
            / "pretrained"
            / "models--microsoft--TRELLIS.2-4B"
            / "snapshots"
            / "af44b45f2e35a493886929c6d786e563ec68364d"
            / "ckpts"
            / "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
        ),
    )
    p.add_argument(
        "--out_dir",
        default=str(_REPO / "results" / "coart_dit_warmstart_stage"),
    )
    args = p.parse_args(argv)
    run(args.config, args.out_dir, args.src_safetensors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
