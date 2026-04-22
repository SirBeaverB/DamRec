#!/usr/bin/env bash
# DamRec FLA 预条件 scale 双卡分波搜索；每组参数独立日志 + RecBole 内部日志归档。
#
# 训练时长：RecBole 按 yaml 的 valid 指标 + stopping_step 早停，epochs 只是上界。
# "4h" 仅表示双卡跑多组的大致量级。
#
# 用法:
#   bash scripts/run_damrec_scale_search_4h_dual_gpu.sh
#   EPOCHS=200 L=64 bash scripts/run_damrec_scale_search_4h_dual_gpu.sh
#   LOGDIR=/path/to/logs bash scripts/run_damrec_scale_search_4h_dual_gpu.sh
#   SMAX_LIST="2.0 3.0" SMIN_LIST="0.5 0.3" TAG_LIST="s2p0_m0p5 s3p0_m0p3" \
#     bash scripts/run_damrec_scale_search_4h_dual_gpu.sh
#
# 输出（$LOGDIR 下）:
#   00_manifest.txt          — 波次时间 + 每组 scale 参数
#   <TAG>_gpu{0,1}.log       — 对应 Python 进程的 stdout+stderr 全部输出（实时刷盘）
#   <TAG>_recbole.log        — 跑完后从 ./log/DamRec/ 归档的 RecBole 结构化日志
#   <TAG>_cmd.txt            — 实际执行的命令行，便于复现
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-150}"
L="${L:-64}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$ROOT/logs/damrec_scale_4h_${STAMP}}"
mkdir -p "$LOGDIR"

# 允许通过环境变量覆盖 scale 列表，否则用默认四组
SMAX_LIST="${SMAX_LIST:-2.0 3.0 4.0 5.0}"
SMIN_LIST="${SMIN_LIST:-0.5 0.3 0.25 0.2}"
TAG_LIST="${TAG_LIST:-s2p0_m0p5 s3p0_m0p3 s4p0_m0p25 s5p0_m0p2}"
read -ra SMAX <<<"$SMAX_LIST"
read -ra SMIN <<<"$SMIN_LIST"
read -ra TAG  <<<"$TAG_LIST"

n=${#SMAX[@]}
if (( n != ${#SMIN[@]} || n != ${#TAG[@]} )); then
  echo "SMAX/SMIN/TAG 数量不一致: ${#SMAX[@]}/${#SMIN[@]}/${#TAG[@]}" >&2
  exit 1
fi

MANIFEST="$LOGDIR/00_manifest.txt"
{
  echo "=== DamRec scale search (dual GPU, batched) ==="
  echo "start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "ROOT=$ROOT"
  echo "LOGDIR=$LOGDIR"
  echo "EPOCHS=$EPOCHS  L=$L"
  echo "CUDA_VISIBLE_DEVICES (host): ${CUDA_VISIBLE_DEVICES:-unset}"
  echo ""
  echo "--- configs ---"
  for ((k = 0; k < n; k++)); do
    printf "  [%d] tag=%-14s scale_max=%s scale_min=%s\n" "$k" "${TAG[k]}" "${SMAX[k]}" "${SMIN[k]}"
  done
  echo ""
} | tee "$MANIFEST"

# 记录跑这组配置前 ./log/DamRec/ 里已存在的文件，方便事后 diff 出新日志
pre_snapshot() {
  if [[ -d "$ROOT/log/DamRec" ]]; then
    (cd "$ROOT/log/DamRec" && ls -1 2>/dev/null | sort) > "$LOGDIR/.pre_$1"
  else
    : > "$LOGDIR/.pre_$1"
  fi
}

# 归档 RecBole 内部日志（跑完后从 ./log/DamRec/ 里找出新生成的那一个）
archive_recbole_log() {
  local tag="$1"
  local dst="$LOGDIR/${tag}_recbole.log"
  if [[ ! -d "$ROOT/log/DamRec" ]]; then
    echo "[archive] ./log/DamRec 不存在，跳过 $tag" | tee -a "$MANIFEST"
    return 0
  fi
  local pre="$LOGDIR/.pre_${tag}"
  local cur
  cur="$(cd "$ROOT/log/DamRec" && ls -1 2>/dev/null | sort)"
  local new_files
  new_files="$(comm -13 "$pre" <(echo "$cur") || true)"
  if [[ -z "$new_files" ]]; then
    echo "[archive] $tag: 未发现新 RecBole 日志" | tee -a "$MANIFEST"
    return 0
  fi
  # 挑 mtime 最新的那个
  local best
  best="$(echo "$new_files" | xargs -I{} find "$ROOT/log/DamRec/{}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')"
  if [[ -n "$best" && -f "$best" ]]; then
    cp -f "$best" "$dst"
    echo "[archive] $tag <- $(basename "$best")" | tee -a "$MANIFEST"
  fi
  rm -f "$pre"
}

# 启动一组训练（后台），通过全局变量 LAST_PID 把 pid 带回调用处。
# 必须不在 $(...) 子 shell 中调用，否则父 shell 无法 wait 该 pid。
launch_one() {
  local gpu="$1" tag="$2" smax="$3" smin="$4"
  local logf="$LOGDIR/${tag}_gpu${gpu}.log"
  local cmdf="$LOGDIR/${tag}_cmd.txt"
  # 先显式建空文件，保证即使 Python 早退也能看到文件存在
  : > "$logf"
  {
    echo "# $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \\"
    echo "  python -u scripts/run_non_streaming_experiments_1m.py \\"
    echo "    -L $L --models DamRec --epochs $EPOCHS \\"
    echo "    --damrec-scale-max $smax --damrec-scale-min $smin"
  } > "$cmdf"
  pre_snapshot "$tag"
  # PYTHONUNBUFFERED + python -u 保证 stdout/stderr 行级刷盘
  # stdbuf -oL -eL 进一步给子进程也套上行缓冲
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    stdbuf -oL -eL python -u scripts/run_non_streaming_experiments_1m.py \
      -L "$L" --models DamRec \
      --epochs "$EPOCHS" \
      --damrec-scale-max "$smax" \
      --damrec-scale-min "$smin" \
      >>"$logf" 2>&1 &
  LAST_PID=$!
}

wave=0
# set +e 包一下后台 wait，避免 Python 非零退出直接 abort 整个 sweep
for ((i = 0; i < n; i += 2)); do
  wave=$((wave + 1))
  {
    echo ""
    echo "--- wave ${wave} start $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
    echo "  gpu0: tag=${TAG[i]} scale=[${SMIN[i]}, ${SMAX[i]}]"
    if (( i + 1 < n )); then
      j=$((i + 1))
      echo "  gpu1: tag=${TAG[j]} scale=[${SMIN[j]}, ${SMAX[j]}]"
    fi
  } | tee -a "$MANIFEST"

  launch_one 0 "${TAG[i]}" "${SMAX[i]}" "${SMIN[i]}"
  pid0=$LAST_PID
  pid1=""
  if (( i + 1 < n )); then
    j=$((i + 1))
    launch_one 1 "${TAG[j]}" "${SMAX[j]}" "${SMIN[j]}"
    pid1=$LAST_PID
  fi

  set +e
  wait "$pid0"; rc0=$?
  rc1=0
  if [[ -n "$pid1" ]]; then
    wait "$pid1"; rc1=$?
  fi
  set -e

  {
    echo "  gpu0 exit_code=$rc0"
    if [[ -n "$pid1" ]]; then echo "  gpu1 exit_code=$rc1"; fi
    echo "--- wave ${wave} end $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
  } | tee -a "$MANIFEST"

  archive_recbole_log "${TAG[i]}"
  if [[ -n "$pid1" ]]; then
    archive_recbole_log "${TAG[$((i + 1))]}"
  fi
done

{
  echo ""
  echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "All waves finished. Logs: $LOGDIR"
  echo ""
  echo "--- per-tag log sizes ---"
  (cd "$LOGDIR" && ls -la *.log 2>/dev/null || echo "(no .log found)")
} | tee -a "$MANIFEST"

echo "Done. Logs under: $LOGDIR"
