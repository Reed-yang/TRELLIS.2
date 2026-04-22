"""Build feat18 encoder/decoder with selectable IO architecture.

IO variants:
    io_arch="three_branch"  - independent Feat18EncIO / Feat18DecIO (default).
                              Warm-start copies pretrained vertex cols into
                              p1_branch AND p2_branch at full strength; p2 bias=0.
    io_arch="monolithic"    - original sp.SparseLinear(18 -> C0) / (C_end -> 18).
                              Warm-start copies pretrained vertex cols into
                              the first 3 slots only (partial warm-start).

Backbone weights are always loaded via strict=False + module-level state_dict;
only I/O layers differ per io_arch.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn as nn

from trellis2 import models
from trellis2.models.sc_vaes.sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)

from .io_stems import Feat18EncIO, Feat18DecIO


IOArch = Literal["three_branch", "monolithic"]


def build_models(
    latent_channels: int = 32,
    device: str | torch.device = "cuda",
    io_arch: IOArch = "three_branch",
    model_channels=(64, 128, 256, 512, 1024),
    num_blocks=(0, 4, 8, 16, 4),
    pred_subdiv=(False, True, True, True, False),
    in_channels: int = 18,
    out_channels: int = 18,
) -> Tuple[SparseUnetVaeEncoder, SparseUnetVaeDecoder]:
    """Construct a (encoder, decoder) pair with the selected IO architecture.

    Model channel / block counts / subdiv default to the shape_vae_next_dc_f16c32
    config. Returns fresh (uninitialised beyond xavier) modules; use
    `load_pretrained_into` to warm-start them.
    """
    encoder = SparseUnetVaeEncoder(
        in_channels=in_channels,
        model_channels=list(model_channels),
        latent_channels=latent_channels,
        num_blocks=list(num_blocks),
        block_type=["SparseConvNeXtBlock3d"] * len(num_blocks),
        down_block_type=["SparseResBlockS2C3d"] * (len(num_blocks) - 1),
        block_args=[{"use_checkpoint": True}] * len(num_blocks),
        use_fp16=False,
    ).to(device)

    decoder = SparseUnetVaeDecoder(
        out_channels=out_channels,
        model_channels=list(reversed(list(model_channels))),
        latent_channels=latent_channels,
        num_blocks=list(reversed(list(num_blocks))),
        block_type=["SparseConvNeXtBlock3d"] * len(num_blocks),
        up_block_type=["SparseResBlockC2S3d"] * (len(num_blocks) - 1),
        block_args=[{"use_checkpoint": True}] * len(num_blocks),
        pred_subdiv=list(reversed(list(pred_subdiv))),
        use_fp16=False,
    ).to(device)

    if io_arch == "three_branch":
        c0 = encoder.input_layer.out_features
        c_end = decoder.output_layer.in_features
        encoder.input_layer = Feat18EncIO(c0).to(device)
        decoder.output_layer = Feat18DecIO(c_end).to(device)
    elif io_arch == "monolithic":
        pass  # keep default sp.SparseLinear(18 -> C0) and (C_end -> 18)
    else:
        raise ValueError(f"unknown io_arch: {io_arch!r}")

    return encoder, decoder


@torch.no_grad()
def _apply_warmstart_three_branch(
    enc_stem: Feat18EncIO,
    dec_stem: Feat18DecIO,
    pre_enc_sd: Dict[str, torch.Tensor],
    pre_dec_sd: Dict[str, torch.Tensor],
) -> None:
    """Copy pretrained vertex cols into both point branches; zero p2 biases."""
    pre_enc_w = pre_enc_sd["input_layer.weight"]   # (C0, 6)
    pre_enc_b = pre_enc_sd["input_layer.bias"]     # (C0,)
    pre_dec_w = pre_dec_sd["output_layer.weight"]  # (7, C_end)
    pre_dec_b = pre_dec_sd["output_layer.bias"]    # (7,)

    # Encoder: both point branches get pretrained vertex prior (full strength).
    # ef_branch stays xavier_uniform (nothing to warm-start from).
    enc_stem.p1_branch.weight.data.copy_(pre_enc_w[:, 0:3])
    enc_stem.p1_branch.bias.data.copy_(pre_enc_b)
    enc_stem.p2_branch.weight.data.copy_(pre_enc_w[:, 0:3])
    enc_stem.p2_branch.bias.data.zero_()

    # Decoder: both point heads get pretrained vertex rows; ef_head stays xavier.
    dec_stem.p1_head.weight.data.copy_(pre_dec_w[0:3, :])
    dec_stem.p1_head.bias.data.copy_(pre_dec_b[0:3])
    dec_stem.p2_head.weight.data.copy_(pre_dec_w[0:3, :])
    dec_stem.p2_head.bias.data.zero_()


@torch.no_grad()
def _apply_warmstart_monolithic(
    encoder: SparseUnetVaeEncoder,
    decoder: SparseUnetVaeDecoder,
    pre_enc_sd: Dict[str, torch.Tensor],
    pre_dec_sd: Dict[str, torch.Tensor],
) -> None:
    """Copy pretrained vertex cols into point1 slot only (partial warm-start).

    For monolithic single-layer IO, only the first 3 input columns (= point1)
    and first 3 output rows are overwritten. The other 15 columns/rows keep
    their xavier_uniform init.
    """
    pre_enc_w = pre_enc_sd["input_layer.weight"]
    pre_enc_b = pre_enc_sd["input_layer.bias"]
    pre_dec_w = pre_dec_sd["output_layer.weight"]
    pre_dec_b = pre_dec_sd["output_layer.bias"]

    encoder.input_layer.weight.data[:, 0:3].copy_(pre_enc_w[:, 0:3])
    encoder.input_layer.bias.data.copy_(pre_enc_b)
    decoder.output_layer.weight.data[0:3, :].copy_(pre_dec_w[0:3, :])
    decoder.output_layer.bias.data[0:3].copy_(pre_dec_b[0:3])


def load_pretrained_into(
    encoder: SparseUnetVaeEncoder,
    decoder: SparseUnetVaeDecoder,
    enc_path: str,
    dec_path: str,
    io_arch: IOArch = "three_branch",
    warmstart_io: bool = True,
    verbose: bool = True,
) -> None:
    """Load pretrained backbone + optional IO warm-start.

    Backbone weights are always loaded via strict=False filtering (skipping
    `input_layer.*` / `output_layer.*` which have different shapes in the new IO).

    If `warmstart_io`, dispatch to the per-arch helper for IO initialisation.
    """
    pre_enc = models.from_pretrained(enc_path)
    pre_dec = models.from_pretrained(dec_path)
    pre_enc_sd = pre_enc.state_dict()
    pre_dec_sd = pre_dec.state_dict()

    enc_filtered = {
        k: v for k, v in pre_enc_sd.items() if not k.startswith("input_layer.")
    }
    dec_filtered = {
        k: v for k, v in pre_dec_sd.items() if not k.startswith("output_layer.")
    }
    miss_e, unex_e = encoder.load_state_dict(enc_filtered, strict=False)
    miss_d, unex_d = decoder.load_state_dict(dec_filtered, strict=False)

    if verbose:
        only_io_missing_e = [k for k in miss_e if not k.startswith("input_layer")]
        only_io_missing_d = [k for k in miss_d if not k.startswith("output_layer")]
        print(f"[init] encoder missing (non-IO): {only_io_missing_e}")
        print(f"[init] encoder unexpected     : {list(unex_e)}")
        print(f"[init] decoder missing (non-IO): {only_io_missing_d}")
        print(f"[init] decoder unexpected     : {list(unex_d)}")

    if warmstart_io:
        if io_arch == "three_branch":
            _apply_warmstart_three_branch(
                encoder.input_layer, decoder.output_layer,
                pre_enc_sd, pre_dec_sd,
            )
            if verbose:
                print("[init] three_branch warm-start: pretrained vertex -> "
                      "p1_branch + p2_branch (full strength, p2 bias=0); "
                      "decoder p1_head + p2_head rows; ef untouched")
        elif io_arch == "monolithic":
            _apply_warmstart_monolithic(
                encoder, decoder, pre_enc_sd, pre_dec_sd,
            )
            if verbose:
                print("[init] monolithic partial warm-start: pretrained vertex "
                      "-> point1 cols only; rest kept at xavier_uniform")
        else:
            raise ValueError(f"unknown io_arch: {io_arch!r}")

    del pre_enc, pre_dec, pre_enc_sd, pre_dec_sd
    torch.cuda.empty_cache()
