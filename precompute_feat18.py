"""
Precompute (cube_indices, feats_18) for the 1k Objaverse-XL Sketchfab subset.

Runs corep_fast s1-s4 (mesh_to_param) + the silent version of param_to_feats
once per mesh and stores everything needed to (a) train the SC-VAE without
re-running the heavy CoReP pipeline online and (b) reconstruct a mesh from
the predicted features at eval time.

Layout produced (under --out_dir, default <dataset_root>/feat18_<resolution>):
    data/<sha256>.npz
        cube_indices  : (N, 3)     int16
        feats         : (N, 18)    float16   # local point coords clipped to [0,1]
        num_boundary  : (N,)       int8      # opaque per-cube flag for s6/s7
        resolution    : ()         int32
    stats_rank<r>.npz
        mean / std    : (18,)      float32
        n_voxels      : ()         int64
    failed_rank<r>.txt
    index.csv         : list of successfully processed sha256 + voxel counts

Usage (single GPU):
    python precompute_feat18.py
Multi-GPU sharding:
    for r in 0..N-1:
        CUDA_VISIBLE_DEVICES=$r python precompute_feat18.py --rank $r --world_size N
"""

from __future__ import annotations

# NOTE: PYTORCH_CUDA_ALLOC_CONF must be exported BEFORE torch is imported,
# otherwise the setting has no effect. expandable_segments dramatically
# reduces fragmentation-related OOMs on long loops over meshes of varying
# size; max_split_size_mb caps the largest contiguous allocation that can
# be split, keeping some headroom for transient tensors.
import os as _os_early
_os_early.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import argparse
import contextlib
import gc
import io
import os
import signal
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from corep_fast.pipeline import CorepParam, mesh_to_param


N_FEAT = 18


def _shutdown_pool_safely() -> None:
    """Terminate the shared multiprocessing.Pool used by corep_fast stages.

    Called after OOM (workers may have inherited a broken CUDA context or
    simply hold onto fork-COW arrays we no longer need) and from signal
    handlers (so orphan workers are not left running after the rank exits).
    """
    try:
        from corep_fast.utils.persistent_pool import shutdown_pool
        shutdown_pool()
    except Exception:
        pass


def _free_cuda_memory() -> None:
    """Aggressively return CUDA memory to the driver.

    `empty_cache` alone leaves memory pinned by Python-side tensor refs;
    a gc.collect cycle first ensures those refs are dropped so the
    subsequent empty_cache actually frees the blocks.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _install_signal_handlers() -> None:
    """Ensure Ctrl-C / SIGTERM tears down forked pool workers.

    Without this, killing the launcher leaves dozens of orphan
    multiprocessing worker processes per rank.
    """
    def _handler(signum, frame):
        _shutdown_pool_safely()
        os._exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


# Substrings that indicate the CUDA context itself is now corrupted and
# cannot be reused. Empirically, once we see one of these, every
# subsequent kernel launch fails the same way even for tiny tensors —
# the only recovery is a fresh process. OOMs are deliberately NOT in
# this list: they normally leave the context usable and just need
# empty_cache to move on.
_CUDA_POISON_MARKERS = (
    "invalid configuration argument",
    "CUDA error",
    "cuBLAS",
    "cuDNN",
    "unspecified launch failure",
    "an illegal memory access",
)


def _is_cuda_context_poisoned(exc: BaseException) -> bool:
    msg = str(exc)
    return any(m.lower() in msg.lower() for m in _CUDA_POISON_MARKERS)


def _write_failure_line(out_dir: str, rank: int, sha: str, msg: str) -> None:
    """Append a single failure record to failed_rank<r>.txt immediately.

    The end-of-run summary also writes this file, but doing it inline
    means a restart / kill does not lose the log of which meshes failed.
    """
    try:
        with open(os.path.join(out_dir, f"failed_rank{rank}.txt"), "a") as fh:
            fh.write(f"{sha}\t{msg}\n")
    except Exception:
        pass


def _mark_failed_sentinel(sentinel_path: str, msg: str) -> None:
    """Write a sibling `<sha>.failed` sentinel so restarts skip this mesh.

    Without this we would either re-trigger the same OOM on every restart
    (infinite loop) or silently reprocess and possibly re-poison the
    CUDA context.
    """
    try:
        with open(sentinel_path, "w") as fh:
            fh.write(msg)
    except Exception:
        pass


def _restart_self() -> None:
    """Replace this process with a fresh copy of itself.

    Called when the CUDA context is irrecoverable. On restart the main
    loop re-accumulates running statistics from the already-written .npz
    files and skips any mesh that has a .failed sentinel, so no progress
    is lost. Uses os.execv so the process keeps the same PID and the
    parent shell's job table remains consistent.
    """
    _shutdown_pool_safely()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ─────────────────────── silent param_to_feats ─────────────────────────────────
# This is a quiet copy of the function in train_overfit_feat18.py.
# The difference: no per-mesh print/warning (we'd flood stdout for 1k meshes),
# and local point coords are CLIPPED to [0, 1] so downstream normalisation
# stays well-behaved even for the boundary cubes that occasionally produce
# slightly out-of-range points.

def param_to_feats(param: CorepParam) -> tuple[np.ndarray, np.ndarray]:
    N = param.cube_indices.shape[0]
    R = float(param.resolution)
    offsets = param.point_offsets
    n_pts = offsets[1:] - offsets[:-1]
    cube_idx_f = param.cube_indices.astype(np.float32)

    first_2 = np.zeros((N, 6), dtype=np.float32)

    mask1 = n_pts == 1
    if mask1.any():
        idx1 = offsets[:-1][mask1]
        first_2[mask1, :3] = param.point_values[idx1] * R - cube_idx_f[mask1]

    mask2 = n_pts == 2
    if mask2.any():
        starts = offsets[:-1][mask2]
        first_2[mask2, :3] = param.point_values[starts] * R - cube_idx_f[mask2]
        first_2[mask2, 3:6] = param.point_values[starts + 1] * R - cube_idx_f[mask2]

    mask_many = n_pts >= 3
    for cube_i in np.where(mask_many)[0]:
        s, e = offsets[cube_i], offsets[cube_i + 1]
        pts_local = param.point_values[s:e] * R - cube_idx_f[cube_i]
        diffs = pts_local[:, None, :] - pts_local[None, :, :]
        dists_sq = np.einsum("ijk,ijk->ij", diffs, diffs)
        i, j = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
        first_2[cube_i, :3] = pts_local[i]
        first_2[cube_i, 3:6] = pts_local[j]

    # Clip silently: corep_fast occasionally emits points slightly outside
    # [0, 1] for boundary cubes; clipping keeps the downstream normalisation
    # in the same range the pretrained encoder was trained on.
    np.clip(first_2, 0.0, 1.0, out=first_2)

    feats = np.concatenate(
        [
            first_2,
            param.edge_weights.astype(np.float32),
            param.face_weights.astype(np.float32),
        ],
        axis=1,
    )
    return param.cube_indices, feats


# ─────────────────────────── encoding driver ──────────────────────────────────

def encode_one(mesh_path: str, resolution: int, device: torch.device,
               num_workers: int | None = None, verbose: bool = False):
    """Run mesh_to_param + param_to_feats for a single mesh.

    By default stdout from corep_fast is suppressed to keep the progress
    bar clean; set ``verbose=True`` to forward it to the terminal so you
    can see which sub-stage is running on a slow / hanging mesh.
    """
    if verbose:
        ctx_out = contextlib.nullcontext()
        ctx_err = contextlib.nullcontext()
    else:
        ctx_out = contextlib.redirect_stdout(io.StringIO())
        ctx_err = contextlib.redirect_stderr(io.StringIO())
    with ctx_out, ctx_err:
        param = mesh_to_param(mesh_path, resolution, device, num_workers=num_workers)
    cube_indices, feats = param_to_feats(param)
    return cube_indices, feats, param.num_boundary


# ─────────────────────────────── main ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset_root",
        default="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab",
        help="Root containing metadata_first1k.csv and raw/.",
    )
    p.add_argument(
        "--metadata_csv",
        default="metadata_first1k.csv",
        help="CSV with sha256 + local_path columns (relative to dataset_root).",
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <dataset_root>/feat18_<resolution>).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Multiprocessing workers per rank for corep_fast s4/s6/s7. "
            "Default shares the machine's CPUs across --world_size ranks "
            "so the total fork fan-out does not explode when running "
            "ranks in parallel."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N (post-shard) entries — useful for smoke tests.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Forward corep_fast stdout/stderr to the terminal so you can see "
            "which sub-stage (s1/s2/s3/s4/s6/s7) is running on a slow mesh."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    _install_signal_handlers()

    out_dir = args.out_dir or os.path.join(
        args.dataset_root, f"feat18_{args.resolution}"
    )
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    metadata = pd.read_csv(os.path.join(args.dataset_root, args.metadata_csv))
    if "local_path" not in metadata.columns or "sha256" not in metadata.columns:
        raise RuntimeError(
            f"metadata_csv must contain 'sha256' and 'local_path' columns, got {list(metadata.columns)}"
        )
    metadata = metadata.iloc[args.rank :: args.world_size].reset_index(drop=True)
    if args.limit is not None:
        metadata = metadata.head(args.limit).reset_index(drop=True)

    # Share the CPU budget across parallel ranks. Without this, each of the
    # N ranks would spawn ~(cpu_count - 4) fork workers, for a total of
    # N * (cpu_count - 4) children — trivially enough to exhaust RAM and
    # destabilise CUDA on a shared node.
    if args.num_workers is None:
        cpu_total = os.cpu_count() or 4
        per_rank_budget = max(1, (cpu_total - 4) // max(1, args.world_size))
        num_workers = per_rank_budget
    else:
        num_workers = max(1, args.num_workers)

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    # Running statistics for per-channel mean/std (computed online over voxels
    # actually present in successfully processed meshes).
    sums = np.zeros(N_FEAT, dtype=np.float64)
    sumsqs = np.zeros(N_FEAT, dtype=np.float64)
    n_vox_total = 0

    failed: list[tuple[str, str]] = []
    succeeded: list[tuple[str, int]] = []

    pbar = tqdm(
        metadata.itertuples(index=False),
        total=len(metadata),
        desc=f"precompute[rank{args.rank}/{args.world_size}]",
        dynamic_ncols=True,
    )
    t0 = time.time()
    for row in pbar:
        sha = row.sha256
        rel = row.local_path
        out_path = os.path.join(data_dir, f"{sha}.npz")
        sentinel_path = out_path + ".failed"

        if os.path.exists(out_path):
            try:
                d = np.load(out_path)
                f = d["feats"].astype(np.float64)
                sums += f.sum(axis=0)
                sumsqs += (f * f).sum(axis=0)
                n_vox_total += f.shape[0]
                succeeded.append((sha, f.shape[0]))
            except Exception as e:  # corrupt file — re-encode
                failed.append((sha, f"reread:{e}"))
            else:
                pbar.set_postfix(skip=sha[:8], n=n_vox_total)
                continue

        # A sibling .failed sentinel means a previous attempt hit a
        # catastrophic CUDA failure (OOM beyond this GPU, or a poisoned
        # context) on this exact mesh. Retrying inside the current
        # process would either OOM again or re-poison the context, so
        # we skip. Delete the sentinel manually to force a retry.
        if os.path.exists(sentinel_path):
            failed.append((sha, "skipped: prior_failure_sentinel"))
            pbar.set_postfix(skip_fail=sha[:8])
            continue

        mesh_path = os.path.join(args.dataset_root, rel)
        if not os.path.exists(mesh_path):
            failed.append((sha, "missing_file"))
            continue

        # Heartbeat so the user can see we are NOT hung when corep_fast is
        # silently grinding (each mesh takes ~10–60 s; without this line the
        # tqdm bar appears frozen for the whole encode).
        try:
            mesh_mb = os.path.getsize(mesh_path) / (1024 * 1024)
        except OSError:
            mesh_mb = float("nan")
        tqdm.write(
            f"[encode] {sha[:16]} {mesh_mb:6.1f} MB  {rel}",
        )
        t_enc = time.time()

        try:
            cube_indices, feats, num_boundary = encode_one(
                mesh_path, args.resolution, device, num_workers=num_workers,
                verbose=args.verbose,
            )
            # IMPORTANT: numpy auto-appends `.npz` if the path does not already
            # end in `.npz`, so a name like `<sha>.npz.tmp` becomes
            # `<sha>.npz.tmp.npz` on disk and the subsequent os.replace silently
            # fails with FileNotFoundError. Keep the `.npz` suffix on the temp
            # name so the written file matches the path we hand to os.replace.
            tmp_path = out_path + ".tmp.npz"
            np.savez_compressed(
                tmp_path,
                cube_indices=cube_indices.astype(np.int16),
                feats=feats.astype(np.float16),
                num_boundary=num_boundary.astype(np.int8),
                resolution=np.int32(args.resolution),
            )
            os.replace(tmp_path, out_path)

            f64 = feats.astype(np.float64)
            sums += f64.sum(axis=0)
            sumsqs += (f64 * f64).sum(axis=0)
            n_vox_total += feats.shape[0]
            succeeded.append((sha, feats.shape[0]))

            enc_dt = time.time() - t_enc
            elapsed = time.time() - t0
            pbar.set_postfix(
                voxels=feats.shape[0],
                tot_v=f"{n_vox_total / 1e6:.1f}M",
                rate=f"{(len(succeeded) + len(failed)) / elapsed:.2f}/s",
            )
            tqdm.write(
                f"[ ok  ] {sha[:16]} voxels={feats.shape[0]:>7d}  enc={enc_dt:6.1f}s"
            )
            # Drop references to the big arrays before the next iteration so
            # that gc.collect below can actually reclaim their memory.
            del cube_indices, feats, num_boundary, f64
        except torch.cuda.OutOfMemoryError as e:
            msg = f"OOM: {str(e)[:120]}"
            failed.append((sha, msg))
            tqdm.write(f"[oom ] {sha} {e}")
            # Persist immediately so the info is not lost if we die or
            # exec ourselves below; and drop a sentinel so on any retry
            # we skip this mesh (it will just OOM again — the mesh is
            # simply too large for this GPU).
            _write_failure_line(out_dir, args.rank, sha, msg)
            _mark_failed_sentinel(sentinel_path, msg)
            # After a CUDA OOM the persistent fork-Pool may be holding
            # references to mesh arrays via fork-COW globals, and a
            # worker may have been killed by the host's OOM killer.
            # Restart the pool so the next mesh starts from a clean slate.
            _shutdown_pool_safely()
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:120]}"
            failed.append((sha, msg))
            tqdm.write(f"[fail] {sha} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            _write_failure_line(out_dir, args.rank, sha, msg)
            _mark_failed_sentinel(sentinel_path, msg)

            if _is_cuda_context_poisoned(e):
                # Context is dead — every subsequent kernel launch will
                # fail with the same "invalid configuration argument"
                # error, so we burn through the rest of the shard doing
                # nothing. Exec a fresh copy of ourselves; already-done
                # .npz files and .failed sentinels are both picked up on
                # restart, so no progress is lost.
                tqdm.write(
                    f"[warn] CUDA context appears poisoned; "
                    f"restarting rank {args.rank} to recover."
                )
                try:
                    pbar.close()
                except Exception:
                    pass
                _free_cuda_memory()
                _restart_self()
        finally:
            # Release cached CUDA memory and Python-level tensor refs after
            # every mesh — success or failure — so peak memory tracks the
            # current mesh rather than the largest one ever seen.
            _free_cuda_memory()

    # ── per-channel statistics ──
    if n_vox_total > 0:
        mean = sums / n_vox_total
        var = sumsqs / n_vox_total - mean * mean
        std = np.sqrt(np.maximum(var, 1e-6))

        # Override the 6 point channels: the pretrained encoder was trained on
        # vertex coords centred at 0.5 → here we want (point - 0.5) only, no
        # /std rescaling, so the warm-started input_layer matches the trained one.
        mean[:6] = 0.5
        std[:6] = 1.0

        stats_path = os.path.join(out_dir, f"stats_rank{args.rank}.npz")
        np.savez(
            stats_path,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            n_voxels=np.int64(n_vox_total),
        )
        print(f"[stats] wrote {stats_path}")
        print(f"[stats] mean = {np.round(mean, 4).tolist()}")
        print(f"[stats] std  = {np.round(std, 4).tolist()}")

    # ── per-rank index of successes ──
    pd.DataFrame(succeeded, columns=["sha256", "num_voxels"]).to_csv(
        os.path.join(out_dir, f"index_rank{args.rank}.csv"), index=False
    )

    # ── failure log ──
    if failed:
        fail_path = os.path.join(out_dir, f"failed_rank{args.rank}.txt")
        with open(fail_path, "w") as fh:
            for sha, msg in failed:
                fh.write(f"{sha}\t{msg}\n")
        print(f"[fail] {len(failed)} sample(s) skipped, see {fail_path}")

    print(
        f"[done] rank={args.rank}/{args.world_size}: "
        f"{len(succeeded)} ok, {len(failed)} failed, "
        f"{n_vox_total / 1e6:.2f}M voxels total"
    )

    # Explicitly tear down the shared worker pool. atexit would normally
    # handle this, but an explicit shutdown guarantees no orphan workers
    # even if the interpreter takes an unusual exit path.
    _shutdown_pool_safely()


if __name__ == "__main__":
    main()
