"""CLI argparse + VaeTrainConfig dataclass for coart.vae training."""
from __future__ import annotations

import argparse
import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Optional


_RUN_TAG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class VaeTrainConfig:
    """Typed wrapper over argparse Namespace; constructed by parse_args().

    Serialised as JSON to `<output_dir>/config.json` at training start for
    provenance and later resume consistency checks.
    """
    # Paths
    data_root: str
    data_dir: Optional[str]
    stats_path: Optional[str]
    output_dir: str
    run_tag: str

    # Pretrained
    from_pretrained: bool
    enc_pretrained: str
    dec_pretrained: str

    # IO architecture
    io_arch: str            # "three_branch" or "monolithic"
    warmstart_io: bool

    # Optimisation
    lr: float
    lr_unfreeze_warmup_steps: int
    freeze_backbone_steps: int
    grad_clip_max: float
    grad_clip_pct: float
    use_bf16: bool

    # Data / augmentation
    batch_size: int
    num_workers: int
    max_translate: int
    max_voxels: int
    bucket_sampler: bool
    bucket_sort_mode: str
    val_split_mod: int

    # Loss
    lambda_kl: float
    lambda_subdiv: float

    # Training schedule
    resolution: int
    latent_channels: int
    max_steps: int
    i_log: int
    i_save: int
    i_sample: int
    i_val: int
    n_dump: int
    sample_at_step_one: bool

    # EMA + ckpt
    use_ema: bool
    ema_rate: float
    rolling_ckpts: int
    resume_from: str       # "none", "latest", or explicit path

    # Wandb logging
    use_wandb: bool
    wandb_project: str
    wandb_mode: str        # "online" / "offline" / "disabled"
    wandb_resume: str      # "auto" = continue prior run on --resume_from ; "never" = always new

    # EMA ckpt rolling (split from rolling_ckpts so EMA doesn't waste disk)
    rolling_ckpts_ema: int

    # Deep-eval / dump
    log_3d: bool
    n_dump_names: list
    first_deep_eval_step: int


def _build_output_dir(
    output_dir: Optional[str],
    run_tag: str,
    resume_from: str,
    search_root: str = "results",
) -> str:
    """Select or build the output directory.

    Precedence:
      1. Caller-supplied ``--output_dir`` wins unchanged.
      2. Else, scan ``{search_root}/coart_feat18_*_{run_tag}/`` for existing
         dirs that actually contain ``ckpt_step*.pt``:
           - ``resume_from != "none"``: must find at least one; if multiple,
             pick the most recently modified (and warn).
           - ``resume_from == "none"``: proceed to step 3 regardless; the
             caller's downstream check (train.py:91-94) will refuse to
             overwrite any accidental match.
      3. Fresh dir using today's date: ``{search_root}/coart_feat18_{YYYYMMDD}_{run_tag}``.

    Why this exists: the original implementation always stamped today's
    date into a freshly-built path. Resuming across midnight picked up a
    new-day directory, which ``find_latest_ckpt`` then saw as empty → the
    trainer silently fell back to base mode and started a parallel run
    from scratch, created a new wandb run, and diverged the checkpoint
    stream. Matching against existing dirs containing real checkpoints
    eliminates that class of bug without forcing callers to thread
    ``--output_dir`` through every launcher invocation.
    """
    import glob
    import sys

    if output_dir:
        return output_dir

    if resume_from != "none":
        pattern = os.path.join(search_root, f"coart_feat18_*_{run_tag}")
        candidates = sorted(glob.glob(pattern))
        with_ckpt = [
            c for c in candidates
            if glob.glob(os.path.join(c, "ckpt_step*.pt"))
        ]
        if not with_ckpt:
            raise RuntimeError(
                f"--resume_from={resume_from!r} but no {pattern}/ckpt_step*.pt "
                f"exists. Pass --output_dir explicitly, or drop --resume_from "
                f"to start a fresh run."
            )
        if len(with_ckpt) > 1:
            with_ckpt.sort(key=os.path.getmtime, reverse=True)
            print(
                f"[coart] multiple matches for run_tag={run_tag!r}: "
                f"{with_ckpt}; picking most recent: {with_ckpt[0]}",
                file=sys.stderr,
            )
        return with_ckpt[0]

    ts = datetime.datetime.now().strftime("%Y%m%d")
    return os.path.join(search_root, f"coart_feat18_{ts}_{run_tag}")


def parse_args() -> VaeTrainConfig:
    p = argparse.ArgumentParser(description="coart.vae — feat18 Shape-VAE finetune")

    # Paths
    p.add_argument("--data_root",
                   default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/"
                           "ObjaverseXL_sketchfab/feat18_512",
                   help="dir containing data/ subdir with *.npz shards")
    p.add_argument("--data_dir", default=None, help="override <data_root>/data")
    p.add_argument("--stats_path", default=None,
                   help="override <data_root>/stats_global.npz")
    p.add_argument("--output_dir", default=None,
                   help="full output dir; if unset, auto-built as "
                        "results/coart_feat18_{YYYYMMDD}_{run_tag}")
    p.add_argument("--run_tag", required=True,
                   help=r"run identifier; regex ^[a-zA-Z0-9_-]+$")

    # Pretrained
    p.add_argument("--from_pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", action="store_false", dest="from_pretrained")
    p.add_argument("--enc_pretrained",
                   default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    p.add_argument("--dec_pretrained",
                   default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16")

    # IO
    p.add_argument("--io_arch", choices=["three_branch", "monolithic"],
                   default="three_branch")
    p.add_argument("--warmstart_io", action="store_true", default=True)
    p.add_argument("--no_warmstart_io", action="store_false", dest="warmstart_io")

    # Optim
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_unfreeze_warmup_steps", type=int, default=500)
    p.add_argument("--freeze_backbone_steps", type=int, default=2000)
    p.add_argument("--grad_clip_max", type=float, default=1.0)
    p.add_argument("--grad_clip_pct", type=float, default=95.0)
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="use_bf16")

    # Data / augment
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_translate", type=int, default=16)
    p.add_argument("--max_voxels", type=int, default=500000)
    p.add_argument("--bucket_sampler", action="store_true", default=True)
    p.add_argument("--no_bucket_sampler", action="store_false", dest="bucket_sampler")
    p.add_argument("--bucket_sort_mode", choices=["shuffle", "ascending"],
                   default="shuffle")
    p.add_argument("--val_split_mod", type=int, default=200,
                   help="sha hash modulus for val partition (0 = no val)")

    # Loss
    p.add_argument("--lambda_kl", type=float, default=1e-6)
    p.add_argument("--lambda_subdiv", type=float, default=0.1)

    # Schedule
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--latent_channels", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--i_log", type=int, default=100)
    p.add_argument("--i_save", type=int, default=5000)
    p.add_argument("--i_sample", type=int, default=5000)
    p.add_argument("--i_val", type=int, default=5000)
    p.add_argument("--n_dump", type=int, default=2)
    p.add_argument("--sample_at_step_one", action="store_true", default=True)
    p.add_argument("--no_sample_at_step_one",
                   action="store_false", dest="sample_at_step_one")

    # EMA / ckpt
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.add_argument("--ema_rate", type=float, default=0.9999)
    p.add_argument("--rolling_ckpts", type=int, default=5)
    p.add_argument("--resume_from", default="latest",
                   help='"none", "latest", or explicit ckpt path')

    # Wandb
    p.add_argument("--use_wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    p.add_argument("--wandb_project", default="coart-vae")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb_resume", choices=["auto", "never"], default="auto",
                   help="auto: when --resume_from loads a ckpt, continue the same "
                        "wandb run (lookup id from misc ckpt / "
                        "<output_dir>/.wandb_run_id). never: always create a fresh run.")

    # EMA rolling split
    p.add_argument("--rolling_ckpts_ema", type=int, default=1,
                   help="rolling K for EMA ckpts (separate from --rolling_ckpts)")

    # Deep-eval dump
    p.add_argument("--log_3d", action="store_true", default=False,
                   help="log wandb.Object3D of decoded mesh (VRAM-hungry)")
    p.add_argument("--n_dump_names", nargs="+",
                   default=["helmet", "val_p95"],
                   help="asset names for normal-map renders in wandb")
    p.add_argument("--first_deep_eval_step", type=int, default=10000,
                   help="skip deep-eval until step >= this (avoids noise "
                        "from untrained ef channels failing feature_to_mesh)")

    args = p.parse_args()

    if not _RUN_TAG_RE.match(args.run_tag):
        raise ValueError(
            f"--run_tag must match ^[a-zA-Z0-9_-]+$, got {args.run_tag!r}"
        )

    args.output_dir = _build_output_dir(
        args.output_dir, args.run_tag, args.resume_from,
    )

    return VaeTrainConfig(**vars(args))
