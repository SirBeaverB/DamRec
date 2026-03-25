#!/usr/bin/env bash
# DamRec FLA 预条件 scale 双卡分波搜索；每组参数独立日志。
#
# 训练时长：RecBole 默认按 yaml 的 valid 指标 + stopping_step 早停（sequential_DamRec.yaml），
#   epochs 只是上界，多数任务在收敛后提前结束，无需靠改小 EPOCHS 来「控时」。
# 本脚本名「4h」表示双卡连续跑多组 scale 的大致量级；实际墙钟因早停与机器而异。
#
# 用法:
#   bash scripts/run_damrec_scale_search_4h_dual_gpu.sh
#   EPOCHS=200 L=64 bash scripts/run_damrec_scale_search_4h_dual_gpu.sh   # 仅提高上限；早停仍生效
#   LOGDIR=/path/to/logs bash scripts/run_damrec_scale_search_4h_dual_gpu.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 与 run_non_streaming_experiments_1m.py 默认一致；真跑满很少见（有早停）
EPOCHS="${EPOCHS:-150}"
L="${L:-64}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$ROOT/logs/damrec_scale_4h_${STAMP}}"
mkdir -p "$LOGDIR"

{
  echo "=== DamRec scale search (dual GPU, batched) ==="
  echo "start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "ROOT=$ROOT"
  echo "LOGDIR=$LOGDIR"
  echo "EPOCHS=$EPOCHS  L=$L"
  echo "CUDA_VISIBLE_DEVICES (host): ${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "$LOGDIR/00_manifest.txt"

# scale_max scale_min  log_suffix
SMAX=(2.0 3.0 4.0 5.0)
SMIN=(0.5 0.3 0.25 0.2)
TAG=(s2p0_m0p5 s3p0_m0p3 s4p0_m0p25 s5p0_m0p2)

n=${#SMAX[@]}
if (( n != ${#SMIN[@]} || n != ${#TAG[@]} )); then
  echo "Internal error: SMAX/SMIN/TAG length mismatch" >&2
  exit 1
fi

wave=0
for ((i = 0; i < n; i += 2)); do
  wave=$((wave + 1))
  {
    echo ""
    echo "--- wave ${wave} start $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
  } | tee -a "$LOGDIR/00_manifest.txt"

  CUDA_VISIBLE_DEVICES=0 python scripts/run_non_streaming_experiments_1m.py \
    -L "$L" --models DamRec \
    --epochs "$EPOCHS" \
    --damrec-scale-max "${SMAX[i]}" \
    --damrec-scale-min "${SMIN[i]}" \
    >>"$LOGDIR/${TAG[i]}_gpu0.log" 2>&1 &
  pid0=$!

  if (( i + 1 < n )); then
    j=$((i + 1))
    CUDA_VISIBLE_DEVICES=1 python scripts/run_non_streaming_experiments_1m.py \
      -L "$L" --models DamRec \
      --epochs "$EPOCHS" \
      --damrec-scale-max "${SMAX[j]}" \
      --damrec-scale-min "${SMIN[j]}" \
      >>"$LOGDIR/${TAG[j]}_gpu1.log" 2>&1 &
    pid1=$!
    wait "$pid0" "$pid1"
  else
    wait "$pid0"
  fi

  {
    echo "--- wave ${wave} end $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
  } | tee -a "$LOGDIR/00_manifest.txt"
done

{
  echo ""
  echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "All waves finished. Logs: $LOGDIR"
} | tee -a "$LOGDIR/00_manifest.txt"

echo "Done. Logs under: $LOGDIR"
