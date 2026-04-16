"""
Dynamic Phase B scheduler with load-balanced work distribution.

Features:
1. Sort items by face_count descending, round-robin assign to GPUs (greedy LPT)
2. Skip already-completed UIDs (resume from existing CSVs)
3. Write per-GPU work manifests, launch processes
4. Monitor and backfill: when a GPU finishes, give it remaining work from slow GPUs

Usage:
    python scripts/pipeline/phase_b_scheduler.py \
        --manifest experiments/component_eval/test_set/manifest_pbr.json \
        --output_dir experiments/component_eval \
        --local_gpus 0,1,2,3,4,5,6,7 \
        --remote "host-10-240-99-118:0,1,2,3,4,5,6,7" \
        --remote "host-10-240-99-120:4,5,6"
"""

import os
import sys
import json
import csv
import glob
import time
import argparse
import subprocess
import heapq
from collections import defaultdict

WORKDIR = "/mnt/novita2/siyuan/workspace/TRELLIS.2"
PYTHON = os.path.join(WORKDIR, ".venv/bin/python")
EVAL_SCRIPT = "scripts/eval/component_eval.py"


def get_done_uids(output_dir):
    """Scan all existing phase_b CSVs for completed UIDs."""
    done = set()
    results_dir = os.path.join(output_dir, "phase_b", "results")
    for f in glob.glob(os.path.join(results_dir, "per_sample_rank*.csv")):
        if os.path.getsize(f) > 0:
            with open(f) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    uid = row.get("uid", "")
                    if uid:
                        done.add(uid)
    return done


def load_balanced_assign(items, num_gpus):
    """Assign items to GPUs using LPT (Longest Processing Time first).

    Sort by face_count descending, greedily assign each item to the
    GPU with the smallest current total load.
    """
    # Sort by face_count descending
    sorted_items = sorted(items, key=lambda x: x["face_count"], reverse=True)

    # Min-heap: (total_faces, gpu_id)
    gpu_loads = [(0, i) for i in range(num_gpus)]
    heapq.heapify(gpu_loads)

    assignments = defaultdict(list)  # gpu_id -> [items]

    for item in sorted_items:
        load, gpu_id = heapq.heappop(gpu_loads)
        assignments[gpu_id].append(item)
        heapq.heappush(gpu_loads, (load + item["face_count"], gpu_id))

    return dict(assignments)


def write_gpu_manifest(items, path):
    """Write a per-GPU manifest file."""
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def launch_process(gpu_spec, rank, manifest_path, output_dir, log_path):
    """Launch a Phase B process on a specific GPU.

    gpu_spec: "0" for local, "host:0" for remote
    """
    if ":" in gpu_spec:
        host, gpu_id = gpu_spec.rsplit(":", 1)
        cmd = (
            f'ssh siyuan@{host} "cd {WORKDIR} && '
            f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 "
            f"nohup {PYTHON} {EVAL_SCRIPT} "
            f"--phase b --manifest {manifest_path} "
            f"--output_dir {output_dir} "
            f"--rank {rank} --world_size 1 "
            f'> {log_path} 2>&1 &"'
        )
    else:
        gpu_id = gpu_spec
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 "
            f"nohup {PYTHON} {EVAL_SCRIPT} "
            f"--phase b --manifest {manifest_path} "
            f"--output_dir {output_dir} "
            f"--rank {rank} --world_size 1 "
            f"> {log_path} 2>&1 &"
        )

    subprocess.Popen(cmd, shell=True, cwd=WORKDIR)
    return cmd


def check_process_alive(gpu_spec, log_path):
    """Check if the process for a GPU is still alive by checking log activity."""
    if not os.path.exists(log_path):
        return False
    try:
        mtime = os.path.getmtime(log_path)
        # If CSV for this rank was modified in last 5 minutes, consider alive
        # Also check if process is running
        ago = time.time() - mtime
        return ago < 600  # 10 min tolerance
    except:
        return False


def check_gpu_process_alive_local(gpu_id):
    """Check if a component_eval process is using this local GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id),
             "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            pid = pid.strip()
            if not pid:
                continue
            # Check if this PID is a component_eval process
            try:
                cmdline = open(f"/proc/{pid}/cmdline").read()
                if "component_eval" in cmdline and "phase" in cmdline:
                    return True
            except:
                pass
        return False
    except:
        return False


def check_gpu_process_alive_remote(host, gpu_id):
    """Check if a component_eval process is running on a remote GPU."""
    try:
        result = subprocess.run(
            ["ssh", f"siyuan@{host}",
             f"nvidia-smi -i {gpu_id} --query-compute-apps=pid --format=csv,noheader"],
            capture_output=True, text=True, timeout=15
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            pid = pid.strip()
            if not pid:
                continue
            check = subprocess.run(
                ["ssh", f"siyuan@{host}",
                 f"cat /proc/{pid}/cmdline 2>/dev/null"],
                capture_output=True, text=True, timeout=10
            )
            if "component_eval" in check.stdout:
                return True
        return False
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="experiments/component_eval")
    parser.add_argument("--local_gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--remote", action="append", default=[],
                        help="host:gpu1,gpu2,... (can specify multiple)")
    parser.add_argument("--check_interval", type=int, default=120,
                        help="Seconds between backfill checks")
    parser.add_argument("--pbr_uids", default=None,
                        help="Path to PBR UIDs JSON (for progress reporting)")
    args = parser.parse_args()

    # Parse GPU specs
    gpu_specs = []  # list of "gpu_id" (local) or "host:gpu_id" (remote)
    for g in args.local_gpus.split(","):
        gpu_specs.append(g.strip())
    for remote in args.remote:
        host, gpus = remote.split(":", 1)
        for g in gpus.split(","):
            gpu_specs.append(f"{host}:{g.strip()}")

    num_gpus = len(gpu_specs)
    print(f"[Scheduler] {num_gpus} GPUs: {gpu_specs}")

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Load PBR UIDs for reporting
    pbr_uids = None
    if args.pbr_uids:
        with open(args.pbr_uids) as f:
            pbr_uids = set(json.load(f))

    # Get already-done UIDs
    done_uids = get_done_uids(args.output_dir)
    remaining = [item for item in manifest if item["uid"] not in done_uids]
    print(f"[Scheduler] Manifest: {len(manifest)}, Done: {len(done_uids)}, Remaining: {len(remaining)}")

    if not remaining:
        print("[Scheduler] All items done!")
        return

    # Load-balanced assignment
    assignments = load_balanced_assign(remaining, num_gpus)

    # Print assignment stats
    for gpu_idx in range(num_gpus):
        items = assignments.get(gpu_idx, [])
        total_faces = sum(i["face_count"] for i in items)
        print(f"  GPU {gpu_specs[gpu_idx]}: {len(items)} items, {total_faces:,} total faces")

    # Write per-GPU manifests and launch
    manifest_dir = os.path.join(args.output_dir, "phase_b", "gpu_manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    log_dir = os.path.join(WORKDIR, "logs")

    active_gpus = {}  # gpu_idx -> {manifest_path, log_path, rank, items}

    for gpu_idx, gpu_spec in enumerate(gpu_specs):
        items = assignments.get(gpu_idx, [])
        if not items:
            continue

        rank = 700 + gpu_idx
        manifest_path = os.path.join(manifest_dir, f"manifest_gpu{gpu_idx}.json")
        safe_spec = gpu_spec.replace(":", "_")
        log_path = os.path.join(log_dir, f"phase_b_sched_{safe_spec}.log")

        write_gpu_manifest(items, manifest_path)
        launch_process(gpu_spec, rank, manifest_path, args.output_dir, log_path)
        print(f"  Launched rank {rank} on {gpu_spec} ({len(items)} items)")

        active_gpus[gpu_idx] = {
            "gpu_spec": gpu_spec,
            "manifest_path": manifest_path,
            "log_path": log_path,
            "rank": rank,
            "items": items,
        }
        time.sleep(2)  # stagger slightly

    print(f"\n[Scheduler] All {len(active_gpus)} GPUs launched. Monitoring...")

    # Monitor loop
    next_rank = 700 + num_gpus
    while True:
        time.sleep(args.check_interval)

        # Check progress
        done_uids = get_done_uids(args.output_dir)
        total_remaining = len(manifest) - len(done_uids)

        if pbr_uids:
            pbr_done = len(pbr_uids & done_uids)
            print(f"[{time.strftime('%H:%M:%S')}] PBR: {pbr_done}/{len(pbr_uids)}, "
                  f"total remaining: {total_remaining}")
            if pbr_done >= len(pbr_uids):
                print("[Scheduler] All PBR items done!")
                break
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Remaining: {total_remaining}")
            if total_remaining <= 0:
                print("[Scheduler] All items done!")
                break

        # Check for idle GPUs and backfill
        for gpu_idx, info in list(active_gpus.items()):
            gpu_spec = info["gpu_spec"]

            # Check if process is still alive
            if ":" in gpu_spec:
                host, gid = gpu_spec.rsplit(":", 1)
                alive = check_gpu_process_alive_remote(host, gid)
            else:
                alive = check_gpu_process_alive_local(gpu_spec)

            if not alive:
                # GPU is idle - find remaining work
                still_todo = [item for item in manifest if item["uid"] not in done_uids]
                if not still_todo:
                    break

                # Give this GPU a chunk of remaining work
                chunk_size = max(1, len(still_todo) // max(1, len(active_gpus)))
                chunk = still_todo[:chunk_size]

                manifest_path = os.path.join(manifest_dir, f"manifest_gpu{gpu_idx}_r{next_rank}.json")
                safe_spec = gpu_spec.replace(":", "_")
                log_path = os.path.join(log_dir, f"phase_b_sched_{safe_spec}_r{next_rank}.log")

                write_gpu_manifest(chunk, manifest_path)
                launch_process(gpu_spec, next_rank, manifest_path, args.output_dir, log_path)
                print(f"  [Backfill] {gpu_spec} -> rank {next_rank} ({len(chunk)} items)")

                info["manifest_path"] = manifest_path
                info["log_path"] = log_path
                info["rank"] = next_rank
                info["items"] = chunk
                next_rank += 1

    print("[Scheduler] Done!")


if __name__ == "__main__":
    main()
