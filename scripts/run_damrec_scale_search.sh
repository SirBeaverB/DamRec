#!/usr/bin/env bash
# 两卡并行扫 DamRec FLA 预条件缩放上下界（演示用两组：[0.3, 3] 与 [0.2, 5]）。
# 每组独立 log，内容行级刷盘；RecBole 内部日志跑完后归档到同目录。
# 用法: bash scripts/run_damrec_scale_search.sh
#       LOGDIR=/path bash scripts/run_damrec_scale_search.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

L="${L:-64}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$ROOT/logs/damrec_scale_${STAMP}}"
mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/00_manifest.txt"

{
  echo "=== DamRec scale search (2 configs, dual GPU) ==="
  echo "start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "LOGDIR=$LOGDIR  L=$L"
  echo "cfg0 gpu=0 tag=s3p0_m0p3 scale=[0.3, 3.0]"
  echo "cfg1 gpu=1 tag=s5p0_m0p2 scale=[0.2, 5.0]"
} | tee "$MANIFEST"

pre_snapshot() {
  if [[ -d "$ROOT/log/DamRec" ]]; then
    (cd "$ROOT/log/DamRec" && ls -1 2>/dev/null | sort) > "$LOGDIR/.pre_$1"
  else
    : > "$LOGDIR/.pre_$1"
  fi
}

archive_recbole_log() {
  local tag="$1"
  local dst="$LOGDIR/${tag}_recbole.log"
  [[ -d "$ROOT/log/DamRec" ]] || { echo "[archive] ./log/DamRec 不存在" | tee -a "$MANIFEST"; return 0; }
  local pre="$LOGDIR/.pre_${tag}"
  local cur new best
  cur="$(cd "$ROOT/log/DamRec" && ls -1 2>/dev/null | sort)"
  new="$(comm -13 "$pre" <(echo "$cur") || true)"
  [[ -n "$new" ]] || { echo "[archive] $tag: 无新日志" | tee -a "$MANIFEST"; return 0; }
  best="$(echo "$new" | xargs -I{} find "$ROOT/log/DamRec/{}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')"
  [[ -n "$best" && -f "$best" ]] && cp -f "$best" "$dst" && echo "[archive] $tag <- $(basename "$best")" | tee -a "$MANIFEST"
  rm -f "$pre"
}

launch_one() {
  local gpu="$1" tag="$2" smax="$3" smin="$4"
  local logf="$LOGDIR/${tag}_gpu${gpu}.log"
  : > "$logf"
  pre_snapshot "$tag"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    stdbuf -oL -eL python -u scripts/run_non_streaming_experiments_1m.py \
      -L "$L" --models DamRec \
      --damrec-scale-max "$smax" \
      --damrec-scale-min "$smin" \
      >>"$logf" 2>&1 &
  echo "$!"
}

pid0=$(launch_one 0 s3p0_m0p3 3.0 0.3)
pid1=$(launch_one 1 s5p0_m0p2 5.0 0.2)

set +e
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
set -e

{
  echo ""
  echo "gpu0 exit_code=$rc0"
  echo "gpu1 exit_code=$rc1"
  echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} | tee -a "$MANIFEST"

archive_recbole_log s3p0_m0p3
archive_recbole_log s5p0_m0p2

{
  echo ""
  echo "--- per-tag log sizes ---"
  (cd "$LOGDIR" && ls -la *.log 2>/dev/null || echo "(no .log found)")
} | tee -a "$MANIFEST"

echo "All done. Logs: $LOGDIR"
