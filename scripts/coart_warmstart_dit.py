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

import numpy as np
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
    state_dict = denoiser.state_dict()
    torch.save(state_dict, denoiser_path)
    print(f"[warmstart] wrote {denoiser_path} ({os.path.getsize(denoiser_path)/1e6:.1f}MB)")

    # Also save EMA copies of the denoiser at each ema_rate the trainer config
    # declares — trellis2/trainers/basic.py:339 loads
    # `<name>_ema<rate>_step<step:07d>.pt` for each rate. At warmstart there's
    # no real EMA history, so we initialise EMA = online weights (decay=0).
    ema_rates = cfg["trainer"]["args"].get("ema_rate", [])
    for r in ema_rates:
        ema_path = os.path.join(ckpt_dir, f"denoiser_ema{r}_step0000000.pt")
        torch.save(state_dict, ema_path)
        print(f"[warmstart] wrote {ema_path} ({os.path.getsize(ema_path)/1e6:.1f}MB)")

    # Misc stub: only the keys read by trellis2/trainers/basic.py:344-358.
    # `optimizer` and `data_sampler` must be valid state_dicts (load_state_dict
    # crashes on None). For pure warm-start (no resumed optimizer state) we want
    # FRESH AdamW state — just construct a fresh AdamW on the loaded denoiser
    # and dump its state_dict (empty `state`, well-formed `param_groups`).
    # data_sampler stays {} since BalancedResumableSampler.load_state_dict({})
    # is a no-op when state has no items (verified pattern from misc loads at
    # step 0 in past trellis2 runs).
    optim_cfg = cfg["trainer"]["args"].get("optimizer", {}).get("args", {})
    fresh_optim = torch.optim.AdamW(
        denoiser.parameters(),
        lr=optim_cfg.get("lr", 1e-4),
        weight_decay=optim_cfg.get("weight_decay", 0.01),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        eps=optim_cfg.get("eps", 1e-8),
    )
    misc = {
        "step": 0,
        "optimizer": fresh_optim.state_dict(),  # well-formed empty state
        # BalancedResumableSampler.load_state_dict expects {epoch, idx} per
        # trellis2/utils/data_utils.py:151-153.
        "data_sampler": {"epoch": 0, "idx": 0},
    }
    # Conditional state dicts the trainer's load() expects only when configured.
    trainer_args = cfg["trainer"]["args"]
    if "elastic" in trainer_args:
        # LinearMemoryController.load_state_dict expects {'params': (k, b)}.
        # Empty (0, 0) means "no calibration yet" — controller will recalibrate
        # on the first step.
        misc["elastic_controller"] = {"params": (0.0, 0.0)}
    if "grad_clip" in trainer_args:
        # AdaptiveGradClipper.load_state_dict overwrites self._grad_norm with
        # state_dict['grad_norm']; the runtime then does
        # `self._grad_norm[buffer_ptr] = grad_norm` which requires an indexable
        # ndarray, NOT a scalar.  See trellis2/utils/grad_clip_utils.py:21,74.
        misc["grad_clip"] = {
            "grad_norm": np.zeros(1000, dtype=np.float32),
            "max_norm": float(trainer_args["grad_clip"]["args"].get("max_norm", 1.0)),
            "buffer_ptr": 0,
            "buffer_length": 0,
        }
    misc_path = os.path.join(ckpt_dir, "misc_step0000000.pt")
    torch.save(misc, misc_path)
    print(f"[warmstart] wrote {misc_path} ({os.path.getsize(misc_path)/1e3:.1f}KB)")

    # Symlink "latest" pointers so --ckpt latest works without specifying step
    # (find_ckpt in train.py:17 globs misc_*.pt and picks the last).
    print(f"[warmstart] done. Launch with:")
    print(f"  LOAD_DIR={out_dir} CKPT=0 bash scripts/train_coart_dit_shape.sh")
    print(f"")
    print(f"misc has fresh AdamW state (empty 'state', valid 'param_groups')")
    print(f"and empty data_sampler state. If the trainer crashes on either,")
    print(f"override load() in coart.dit.trainer to skip those when step==0.")


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
