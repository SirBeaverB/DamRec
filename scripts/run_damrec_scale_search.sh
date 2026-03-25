#!/usr/bin/env bash
# 两张 GPU 并行扫 DamRec FLA 预条件缩放上下界（示例：max=3 vs 5，min 与 max 配对）。
# 用法: bash scripts/run_damrec_scale_search.sh
# 可按需改 CUDA 编号、-L、--damrec-scale-min/max。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 卡0: [1/3, 3] 与 max=3 配对（min 默认 1/max，可显式写 0.333...）
CUDA_VISIBLE_DEVICES=0 python scripts/run_non_streaming_experiments_1m.py \
  -L 64 --models DamRec \
  --damrec-scale-max 3.0 \
  --damrec-scale-min 0.3 &

# 卡1: [0.2, 5]
CUDA_VISIBLE_DEVICES=1 python scripts/run_non_streaming_experiments_1m.py \
  -L 64 --models DamRec \
  --damrec-scale-max 5.0 \
  --damrec-scale-min 0.2 &

wait
echo "All done"
