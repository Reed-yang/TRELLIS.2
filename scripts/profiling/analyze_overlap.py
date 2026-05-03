"""Analyze a chrome trace to extract compute / comm overlap metrics for W3.

Outputs:
  bucket_table.md            — kernel bucket breakdown like prior baseline
  overlap_summary.json       — comm_hidden_ratio + eltwise critical-path ms
"""
import argparse, json, os
from collections import defaultdict


def parse_trace(path):
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return events


def classify_bucket(name: str) -> str:
    n = name.lower()
    if "nccl" in n or "all_reduce" in n or "reduce_scatter" in n or "all_gather" in n:
        return "nccl"
    if "flash" in n and ("attn" in n or "fwd" in n or "bwd" in n):
        return "flash_attn"
    if "gemm" in n or "cublas" in n or "_gemm_" in n:
        return "gemm"
    if "rmsnorm" in n or "layer_norm" in n or "layernorm" in n:
        return "layernorm"
    if "binaryfunctor" in n or "unaryfunctor" in n or "elementwise" in n \
       or "copy_kernel" in n or "addcmul" in n or "_add" in n or "_mul" in n:
        return "eltwise"
    if "indexing_backward" in n or "_index_" in n:
        return "other"
    if "adam" in n or "foreach" in n:
        return "optimizer"
    if "memcpy" in n:
        return "memcpy"
    if "reduce" in n:
        return "reduce"
    if "gelu" in n:
        return "gelu"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=5,
                    help="Number of active steps captured (used to compute ms/step)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    events = parse_trace(args.trace)
    kernels = [e for e in events
               if e.get("ph") == "X" and "dur" in e and ("stream" in e.get("args", {})
                                                          or e.get("cat") == "kernel")]

    bucket_total = defaultdict(float)
    for e in kernels:
        bucket_total[classify_bucket(e["name"])] += e["dur"]

    streams = defaultdict(list)
    for e in kernels:
        s = e.get("args", {}).get("stream")
        if s is not None:
            streams[s].append((e["ts"], e["ts"] + e["dur"]))

    nccl_streams = defaultdict(float)
    for e in kernels:
        if classify_bucket(e["name"]) == "nccl":
            s = e.get("args", {}).get("stream")
            if s is not None:
                nccl_streams[s] += e["dur"]
    comm_stream = max(nccl_streams.items(), key=lambda kv: kv[1])[0] if nccl_streams else None

    if streams:
        all_intervals = [iv for s in streams.values() for iv in s]
        wall = max(e for _, e in all_intervals) - min(s for s, _ in all_intervals)
    else:
        wall = 0
    compute_busy = sum(e - s for s_ in streams for s, e in streams[s_] if s_ != comm_stream) if streams else 0
    comm_busy = sum(e - s for s, e in streams.get(comm_stream, [])) if comm_stream is not None else 0
    comm_hidden_ratio = 1.0 - (comm_busy / wall) if wall > 0 else 0.0

    n_steps = args.n_steps
    summary = {
        "trace_path": args.trace,
        "n_steps": n_steps,
        "wall_us": wall,
        "compute_stream_busy_us": compute_busy,
        "comm_stream_busy_us": comm_busy,
        "comm_hidden_ratio": comm_hidden_ratio,
        "buckets_us": dict(bucket_total),
        "buckets_ms_per_step": {k: v / 1000 / n_steps for k, v in bucket_total.items()},
    }

    with open(f"{args.out}/overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["| bucket | ms/step | % |", "|---|---|---|"]
    total_ms = sum(summary["buckets_ms_per_step"].values())
    for k, v in sorted(summary["buckets_ms_per_step"].items(), key=lambda kv: -kv[1]):
        pct = v / total_ms * 100 if total_ms > 0 else 0
        lines.append(f"| {k} | {v:.1f} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total_ms:.1f}** | **100%** |")
    lines.append(f"\ncomm_hidden_ratio: {comm_hidden_ratio:.3f}")
    lines.append(f"compute_stream_busy_ms_per_step: {compute_busy / 1000 / n_steps:.1f}")
    lines.append(f"comm_stream_busy_ms_per_step: {comm_busy / 1000 / n_steps:.1f}")
    with open(f"{args.out}/bucket_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[overlap] wrote {args.out}/{{overlap_summary.json,bucket_table.md}}")


if __name__ == "__main__":
    main()
