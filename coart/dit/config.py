"""Static config + path helpers for the coart shape-DiT finetune.

Mirrors the role of ``coart/vae/config.py`` but the actual training entry
point is the upstream ``train.py`` driven by a JSON config under
``coart/dit/configs/``. We therefore expose:

  * a ``default_config_path()`` helper returning the JSON path,
  * shared constants (data root, vae tag, pretrained ckpt path),
  * a ``WARMSTART_HINT`` string documenting how to load the pretrained DiT
    safetensors into ``ElasticSLatFlowModel`` before launching training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------- paths

# Cache directory built by scripts/coart_data_v0/run.sh.
COART_DIT_DATA_ROOT = (
    "/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0"
)

# Tag identifying which shape-VAE encoder produced the cached latents.
# Sub-directory under ``COART_DIT_DATA_ROOT/slat/``.
DEFAULT_VAE_TAG = "vae_three_branch_ws_v0_ema_s0155000"

# Resolved snapshot dir for the official 1.3B img2shape DiT (bf16 safetensors).
PRETRAINED_DIT_CKPT = (
    "/mnt/novita2/siyuan/workspace/TRELLIS.2/pretrained/"
    "models--microsoft--TRELLIS.2-4B/snapshots/"
    "af44b45f2e35a493886929c6d786e563ec68364d/"
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
)

# Companion JSON metadata next to the safetensors blob (model arch hints,
# normalisation stats, etc.). Same dir as ``PRETRAINED_DIT_CKPT``.
PRETRAINED_DIT_META_JSON = (
    "/mnt/novita2/siyuan/workspace/TRELLIS.2/pretrained/"
    "models--microsoft--TRELLIS.2-4B/snapshots/"
    "af44b45f2e35a493886929c6d786e563ec68364d/"
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json"
)


def default_config_path() -> str:
    """Return absolute path to the default DiT finetune JSON config."""
    return os.path.join(os.path.dirname(__file__), "configs", "coart_dit_shape_512_ft.json")


# ---------------------------------------------------------------------- warmstart

WARMSTART_HINT = """\
Warm-start from the official 1.3B shape DiT (bf16 safetensors) before launching
``train.py`` on this config:

    from safetensors.torch import load_file
    from trellis2 import models
    from coart.dit.config import PRETRAINED_DIT_CKPT, default_config_path
    import json
    cfg = json.load(open(default_config_path()))
    denoiser_cfg = cfg["models"]["denoiser"]
    denoiser = getattr(models, denoiser_cfg["name"])(**denoiser_cfg["args"])
    sd = load_file(PRETRAINED_DIT_CKPT)
    miss, unex = denoiser.load_state_dict(sd, strict=False)
    print("missing", miss[:5], "unexpected", unex[:5])

The upstream ``train.py``'s ``--load_dir/--ckpt`` mechanism only knows the
in-house ``misc_step*.pt`` layout (see ``find_ckpt`` in train.py:17), so a
safetensors -> .pt conversion (or a small monkey-patch in the trainer's
``load_state_dict`` step) is currently a TODO before the launcher script can
warm-start automatically. For the first finetune run, materialise an initial
``ckpts/denoiser_step0.pt`` from the converted state-dict and point
``--load_dir`` at the surrounding directory with ``--ckpt 0``.
"""


# ---------------------------------------------------------------------- dataclass

@dataclass(frozen=True)
class CoartDitFinetunePaths:
    """Resolved path bundle for a single finetune launch."""
    data_root: str = COART_DIT_DATA_ROOT
    vae_tag: str = DEFAULT_VAE_TAG
    pretrained_ckpt: str = PRETRAINED_DIT_CKPT
    config_json: str = ""

    @classmethod
    def default(cls) -> "CoartDitFinetunePaths":
        return cls(config_json=default_config_path())
