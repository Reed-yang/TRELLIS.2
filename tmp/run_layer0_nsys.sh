#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
export CUDA_VISIBLE_DEVICES=0
OUT=/mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/profile_deep/results
mkdir -p "$OUT"

# Run nsys; --nvtx captures our stage ranges. --cuda-memory-usage adds sync points
# (small overhead, worth it for d2h/h2d visibility).
nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --cuda-memory-usage=true \
    --sample=cpu \
    --python-backtrace=cuda \
    --force-overwrite=true \
    --output="$OUT/nsys_res256_full" \
    .venv/bin/python tmp/profile_deep/driver.py --layer 0 --res 256

# Post: stats dump (text-friendly)
nsys stats --format csv --output "$OUT/nsys_res256_stats" "$OUT/nsys_res256_full.nsys-rep" \
    > "$OUT/nsys_res256_stats.txt" 2>&1 || \
    nsys stats "$OUT/nsys_res256_full.nsys-rep" > "$OUT/nsys_res256_stats.txt" 2>&1

echo "nsys output:"
ls -lh "$OUT"/nsys_res256*
