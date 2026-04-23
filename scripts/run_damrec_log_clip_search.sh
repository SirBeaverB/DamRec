#!/usr/bin/env bash
# DamRec FLA 路径 g=log(α) 前下界 clamp 阈值 sweep。
# 双卡并行：每张卡顺序跑 N 个值（默认 2 个），两张卡互不阻塞。
#
# 默认 4 个值跨 4 个量级：CLIP_LIST="1e-2 1e-3 1e-4 1e-6"
#   gpu0: 1e-2, 1e-3
#   gpu1: 1e-4, 1e-6
#
# 用法:
#   bash scripts/run_damrec_log_clip_search.sh
#   EPOCHS=200 L=64 bash scripts/run_damrec_log_clip_search.sh
#   LOGDIR=/path bash scripts/run_damrec_log_clip_search.sh
#   CLIP_LIST="1e-2 1e-4 1e-5 1e-6" TAG_LIST="c1em2 c1em4 c1em5 c1em6" \
#     bash scripts/run_damrec_log_clip_search.sh
#
# 输出（$LOGDIR 下）:
#   00_manifest.txt          — 启动信息 + 每组 clip 参数 + 退出码
#   00_summary.txt           — 所有 tag 按 clip 升序的结果对比表（recall/ndcg/time/mem）
#   <TAG>_gpu{0,1}.log       — 对应 Python 进程的 stdout+stderr 全部输出（实时刷盘）
#   <TAG>_recbole.log        — 跑完后从 ./log/DamRec/ 归档的 RecBole 结构化日志
#   <TAG>_cmd.txt            — 实际执行的命令行
#   result_<TAG>/            — 该 tag 的 non_streaming_1m_L{L}_*.txt / .csv（隔离避免并发冲突）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-150}"
L="${L:-64}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGDIR:-$ROOT/logs/damrec_log_clip_${STAMP}}"
mkdir -p "$LOGDIR"

CLIP_LIST="${CLIP_LIST:-1e-2 1e-3 1e-4 1e-6}"
TAG_LIST="${TAG_LIST:-c1em2 c1em3 c1em4 c1em6}"
read -ra CLIPS <<<"$CLIP_LIST"
read -ra TAGS  <<<"$TAG_LIST"

n=${#CLIPS[@]}
if (( n != ${#TAGS[@]} )); then
  echo "CLIP_LIST/TAG_LIST 数量不一致: ${#CLIPS[@]}/${#TAGS[@]}" >&2
  exit 1
fi
if (( n % 2 != 0 )); then
  echo "为简化双卡均分，CLIP_LIST 长度需为偶数（当前 $n）" >&2
  exit 1
fi
half=$((n / 2))

MANIFEST="$LOGDIR/00_manifest.txt"
{
  echo "=== DamRec log_clip_min sweep (dual GPU, each card serial) ==="
  echo "start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "ROOT=$ROOT"
  echo "LOGDIR=$LOGDIR"
  echo "EPOCHS=$EPOCHS  L=$L"
  echo "CUDA_VISIBLE_DEVICES (host): ${CUDA_VISIBLE_DEVICES:-unset}"
  echo ""
  echo "--- gpu0 (serial) ---"
  for ((k = 0; k < half; k++)); do
    printf "  [%d] tag=%-10s log_clip_min=%s\n" "$k" "${TAGS[k]}" "${CLIPS[k]}"
  done
  echo "--- gpu1 (serial) ---"
  for ((k = half; k < n; k++)); do
    printf "  [%d] tag=%-10s log_clip_min=%s\n" "$k" "${TAGS[k]}" "${CLIPS[k]}"
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

archive_recbole_log() {
  local tag="$1"
  local dst="$LOGDIR/${tag}_recbole.log"
  if [[ ! -d "$ROOT/log/DamRec" ]]; then
    echo "[archive] ./log/DamRec 不存在，跳过 $tag" | tee -a "$MANIFEST"
    return 0
  fi
  local pre="$LOGDIR/.pre_${tag}"
  local cur new_files best
  cur="$(cd "$ROOT/log/DamRec" && ls -1 2>/dev/null | sort)"
  new_files="$(comm -13 "$pre" <(echo "$cur") || true)"
  if [[ -z "$new_files" ]]; then
    echo "[archive] $tag: 未发现新 RecBole 日志" | tee -a "$MANIFEST"
    return 0
  fi
  best="$(echo "$new_files" | xargs -I{} find "$ROOT/log/DamRec/{}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')"
  if [[ -n "$best" && -f "$best" ]]; then
    cp -f "$best" "$dst"
    echo "[archive] $tag <- $(basename "$best")" | tee -a "$MANIFEST"
  fi
  rm -f "$pre"
}

# 在指定 GPU 上串行跑该卡分配到的所有 clip 值。
# 单条失败不中断后续；最终汇总到 manifest。
run_gpu_lane() {
  local gpu="$1"
  shift
  local lane_label="gpu${gpu}"
  local idx
  for idx in "$@"; do
    local tag="${TAGS[idx]}"
    local clip="${CLIPS[idx]}"
    local logf="$LOGDIR/${tag}_${lane_label}.log"
    local cmdf="$LOGDIR/${tag}_cmd.txt"
    local resdir="$LOGDIR/result_${tag}"
    : > "$logf"
    mkdir -p "$resdir"
    {
      echo "# $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \\"
      echo "  python -u scripts/run_non_streaming_experiments_1m.py \\"
      echo "    -L $L --models DamRec --epochs $EPOCHS \\"
      echo "    --damrec-log-clip-min $clip \\"
      echo "    --output-dir $resdir"
    } > "$cmdf"
    pre_snapshot "$tag"

    {
      echo ""
      echo "--- ${lane_label} start tag=${tag} clip=${clip} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
    } | tee -a "$MANIFEST"

    set +e
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      stdbuf -oL -eL python -u scripts/run_non_streaming_experiments_1m.py \
        -L "$L" --models DamRec \
        --epochs "$EPOCHS" \
        --damrec-log-clip-min "$clip" \
        --output-dir "$resdir" \
        >>"$logf" 2>&1
    local rc=$?
    set -e

    {
      echo "  ${lane_label} tag=${tag} clip=${clip} exit_code=${rc} end=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } | tee -a "$MANIFEST"

    archive_recbole_log "$tag"
  done
}

# gpu0 lane: 索引 0..half-1
gpu0_idxs=()
for ((k = 0; k < half; k++)); do gpu0_idxs+=("$k"); done
# gpu1 lane: 索引 half..n-1
gpu1_idxs=()
for ((k = half; k < n; k++)); do gpu1_idxs+=("$k"); done

# 后台并行启动两条 lane
run_gpu_lane 0 "${gpu0_idxs[@]}" &
pid0=$!
run_gpu_lane 1 "${gpu1_idxs[@]}" &
pid1=$!

set +e
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
set -e

{
  echo ""
  echo "gpu0 lane exit_code=$rc0"
  echo "gpu1 lane exit_code=$rc1"
  echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo ""
  echo "--- per-tag log sizes ---"
  (cd "$LOGDIR" && ls -la *.log 2>/dev/null || echo "(no .log found)")
} | tee -a "$MANIFEST"

# 解析每个 tag 的 result CSV，按 clip 排序输出 00_summary.txt。
# CSV 列（1-indexed）：1=model 2=v_recall 4=v_ndcg 7=t_recall 9=t_ndcg 12=time 13=mem
SUMMARY="$LOGDIR/00_summary.txt"
{
  echo "=== DamRec log_clip_min sweep summary ==="
  echo "LOGDIR=$LOGDIR"
  echo "L=$L  EPOCHS=$EPOCHS  end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo ""
  printf "%-12s %-10s %-12s %-12s %-12s %-12s %-10s %-8s\n" \
    tag clip v_recall@10 v_ndcg@10 t_recall@10 t_ndcg@10 time_sec mem_GB
  printf "%-12s %-10s %-12s %-12s %-12s %-12s %-10s %-8s\n" \
    ------------ ---------- ------------ ------------ ------------ ------------ ---------- --------
} > "$SUMMARY"

# 收集 (clip, tag, csv) 三元组，按 clip 数值升序后写入 summary
# 先把元组写到临时文件，再 sort 一次（按数值升序）
TMP_TUPLES="$(mktemp)"
trap 'rm -f "$TMP_TUPLES"' EXIT

for ((i = 0; i < n; i++)); do
  tag="${TAGS[i]}"
  clip="${CLIPS[i]}"
  csv="$(ls "$LOGDIR/result_${tag}/non_streaming_1m_L${L}_"*.csv 2>/dev/null | head -1)"
  printf "%s\t%s\t%s\n" "$clip" "$tag" "${csv:-MISSING}" >>"$TMP_TUPLES"
done

# 按 clip 数值升序（-g = 一般数值排序，支持 1e-2 这种科学计数法）
sort -g -k1,1 "$TMP_TUPLES" | while IFS=$'\t' read -r clip tag csv; do
  if [[ "$csv" == "MISSING" || ! -f "$csv" ]]; then
    printf "%-12s %-10s %-12s %-12s %-12s %-12s %-10s %-8s\n" \
      "$tag" "$clip" MISSING MISSING MISSING MISSING MISSING MISSING >>"$SUMMARY"
    continue
  fi
  awk -F, -v tag="$tag" -v clip="$clip" '
    NR==2 {
      printf "%-12s %-10s %-12s %-12s %-12s %-12s %-10s %-8s\n",
        tag, clip, $2, $4, $7, $9, $12, $13
    }
  ' "$csv" >>"$SUMMARY"
done

{
  echo ""
  echo "--- 原始 CSV 路径 ---"
  for ((i = 0; i < n; i++)); do
    tag="${TAGS[i]}"
    csv="$(ls "$LOGDIR/result_${tag}/non_streaming_1m_L${L}_"*.csv 2>/dev/null | head -1)"
    echo "  $tag -> ${csv:-MISSING}"
  done
} >>"$SUMMARY"

echo ""
echo "=== Summary ==="
cat "$SUMMARY"
echo ""
echo "Done. Logs under: $LOGDIR"
echo "Summary: $SUMMARY"
