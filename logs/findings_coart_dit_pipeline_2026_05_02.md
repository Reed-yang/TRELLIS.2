# Coart DiT Data Pipeline + Finetune — AFK Run Findings (2026-05-02)

**Branch**: `vae-finetune` | **Final state**: data pipeline ✅ done; trainer launches but first step stalls; needs interactive debug.

## Outcome summary

- **Data pipeline**: ✅ all 3 caches built. ~9700 render dirs, ~9600 dino npz, 10000/10000 slat npz at `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/`.
- **DiT scaffold**: ✅ all code committed; `import coart.dit` registers `CachedImageConditionedSLatShape` + `CachedImageConditionedSparseFlowMatchingCFGTrainer` into trellis2 namespaces.
- **Warmstart**: ✅ official 1.3B safetensors converted to trellis2-trainer ckpt layout (denoiser_step0 + EMA + misc) at `results/coart_dit_warmstart_stage/`. Trainer constructs cleanly with this load_dir.
- **DiT first training step**: ⚠️ launches, model loaded to GPU (22 GB on 7/8 GPUs), workers spawn, but **first step never completes** — workers sit in `time.sleep` for 40+ min with no log output.

## Commits

```
b0a6b5f  docs(coart_data_v0): add data pipeline spec + implementation plan
1f2ab31  feat(coart_data_v0): add pick_instances pre-flight selector
7017047  feat(coart_data_v0): add build_manifest aggregator
616a354  feat(coart): add SS-flow occupancy IoU validation script
370b015  feat(coart_data_v0): add coart_cache_dino with stable shard + atomic save
92f900f  feat(coart_data_v0): add coart_cache_slat with VAE-tag versioning
6cc9252  feat(coart_data_v0): add run.sh stage orchestrator + README
70fb2c9  fix(coart_data_v0): EMA shadow-list loader + sys.path bootstrap
c687aec  feat(coart.dit): add shape DiT finetune scaffold reusing trellis2 trainer
d3e4ee9  fix(coart_data_v0): save latent (not input) coords from cache_slat + tune render concurrency
2ad7818  feat(coart_dit): add safetensors -> trainer ckpt warmstart converter
62b3ed9  fix(coart_dit): warmstart misc has well-formed empty AdamW state
543cb4e  fix(coart_dit): complete warmstart misc state + add dataset.loads attribute
df8f34e  fix(coart_data_v0): build_manifest._check_dino avoids decompressing 33MB features
dfa0842  fix(coart_dit): drop torchrun, propagate registration to mp.spawn, skip snapshot_dataset
```

## Bugs found and fixed during the AFK run

1. **EMA ckpt format**: trained ckpts have `{decay, shadow=List[Tensor]}`, not flat state_dict. Fix: use `coart.common.ema.EMAModel.load_state_dict + copy_to`.
2. **DINO gated repo**: `facebook/dinov3-vitl16-pretrain-lvd1689m` is gated; transformers raises 401. Fix: prefer local cache at `pretrained/dinov3-vitl16-pretrain-lvd1689m/...`.
3. **`coart` not importable in dispatched scripts**: `python scripts/.../foo.py` doesn't add repo root to sys.path. Fix: `sys.path.insert(0, repo_root)` at top of each script.
4. **`atomic_savez` filename**: `np.savez_compressed` auto-appends `.npz`, breaking `os.replace`. Fix: tmp filename ends in `.npz`.
5. **PYTHON path becomes invalid after cd in bash**: `.venv/bin/python` is relative; after `cd data_toolkit` it broke. Fix: resolve to absolute path in run.sh.
6. **Render path resolution**: `data_toolkit/render_cond.py:foreach_instance` joins `local_path` against `--render_cond_root` not `--download_root`. Fix: symlink `coart_dit_data_v0/raw -> ../raw`.
7. **Blender thread oversubscription**: 24 ranks × 8 workers = 192 concurrent Blender, oversubscribed 128 cores. Fix: `--max_workers 1` per rank → ~22-27/min cluster throughput.
8. **Slat coords/feats N mismatch**: SparseUnetVae downsamples 16x; saved cube_indices (input res 512, ~500K rows) didn't match mu (latent res 32, ~1.5K rows). Fix: save `z_out.coords[:, 1:4]` (latent res 32).
9. **Warmstart misc had `optimizer=None`**: would crash `optimizer.load_state_dict(None)`. Fix: build a fresh AdamW with same hyperparams, dump its state_dict.
10. **Warmstart misc missing keys for elastic + grad_clip**: trainer requires these state_dicts when configured. Fix: synthesise minimal valid state for both.
11. **build_manifest._check_dino slow**: decompressed 33MB features just to read shape. Fix: probe `n_tokens` (0-d int32) instead.
12. **torchrun + train.py port conflict**: train.py runs its own `mp.spawn` with master_port=12345; torchrun added a second TCP store. Fix: drop torchrun, use plain python with `--num_gpus N`.
13. **mp.spawn workers don't inherit imports**: parent's `import coart.dit` doesn't register into trellis2 in spawn workers. Fix: `.pth` file in venv site-packages auto-imports coart.dit when `COART_AUTO_REGISTER_DIT=1`.
14. **snapshot_dataset incompatible with SparseTensor**: BasicTrainer.snapshot_dataset uses `torch.stack` which fails on SparseTensor x_0. Fix: override to no-op in our trainer.

## Open issue: first training step never completes

After all 14 bug fixes, the trainer:
- ✅ Constructs cleanly with all components (AdamW lr=2e-5, EMA 0.9999, elastic, grad_clip, bf16 AMP)
- ✅ Loads warmstart denoiser + EMA ckpts
- ✅ Spawns 8 workers via `mp.spawn`
- ✅ Each worker allocates 22 GB on its GPU (model + optimizer + grad)
- ⚠️ **But all workers then sit in `hrtimer_nanosleep` (i.e., `time.sleep`) for 40+ min with no log output**

GPU 0 oddly only has 532 MB allocated while GPUs 1-7 have 22 GB. Suggests rank 0 is stuck on something pre-model-load while ranks 1-7 finished model load and are waiting for rank 0.

### Hypotheses (in order of likelihood)

1. **Dataset enumeration on rank 0 is slow**: `_enumerate_root` reads every slat npz to compute `.loads`. With 9600 npz on NFS, this can take 5-15 min. Should run on all ranks but rank 0 might hit it first then barrier-wait. **Quick test**: pre-cache `loads` to a JSON file at instances_10k.csv build time so all ranks load from one file.
2. **NCCL init hang**: workers create NCCL communicator at first AllReduce. If a network config is off (e.g., firewall, IB unhealthy), this hangs silently with 600s timeout × N retries. **Quick test**: `NCCL_DEBUG=INFO` env var to surface the handshake.
3. **DataLoader workers blocked on first batch**: BalancedResumableSampler sorts the loads list to bucket batches; with N=9600 this should be sub-second. But if num_workers > 0 (PyTorch DataLoader spawns prefetch workers), they each do their own fork with fresh imports — could be hitting the same registration issue.

### Recommended next steps for interactive debug

```bash
# 1. py-spy to localise the stall:
ssh host-10-240-99-119 "sudo /mnt/novita2/siyuan/workspace/TRELLIS.2/.venv/bin/py-spy dump --pid <rank0_pid>"

# 2. Re-launch with verbose NCCL:
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=INFO LOAD_DIR=results/coart_dit_warmstart_stage CKPT=0 \
  bash scripts/train_coart_dit_shape.sh 2>&1 | tee logs/dit_relaunch_verbose.log

# 3. If hypothesis #1: precompute loads
python -c "
import json, os, numpy as np
slat_dir = '/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/slat/vae_three_branch_ws_v0_ema_s0155000'
loads = {}
for f in os.listdir(slat_dir):
    sha = f.removesuffix('.npz')
    loads[sha] = int(np.load(os.path.join(slat_dir, f))['num_voxels'])
json.dump(loads, open('coart_dit_data_v0/loads.json', 'w'))
"
# Then update CachedImageConditionedSLatShape._enumerate_root to load from loads.json if present.

# 4. If hypothesis #2: reduce world_size to test
NPROC=1 LOAD_DIR=... bash scripts/train_coart_dit_shape.sh
# Single-process avoids DDP/NCCL entirely; if it works, the issue is dist init.
```

## Data pipeline status (handover-ready)

- `instances_10k.csv` (10000 rows, aesthetic ≥ 4.5)
- `renders_cond/{sha}/{000..015}.png + transforms.json` (~9700 dirs, Blender CYCLES @ 1024)
- `dino_l16_s512/{sha}.npz` features (16, 1029, 1024) fp16 (~9600)
- `slat/vae_three_branch_ws_v0_ema_s0155000/{sha}.npz`: coords (N, 3) int16 @ res 32, feats (N, 32) fp16 (10000/10000)
- `manifest.csv` was started but old slow build_manifest version was killed at iter 1; rerun with the fixed `build_manifest.py` (commit `df8f34e`) to get full manifest.

## Out of scope

- 50K scale-up (linear in render time, ~30-45h on 24 GPUs at 22-27/min)
- SS-flow IoU validation: `coart_compare_occupancy.py` has a small `aabb=2D-tensor` bug; not blocking trainer correctness, only affects spec §6.2 affine-alignment recommendation
- Multi-node DDP for DiT (single-node 8× H100 is enough capacity for the 1.3B model)
- PBR DiT: spec defers verification to after shape DiT trains
