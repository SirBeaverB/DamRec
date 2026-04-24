#!/usr/bin/env bash
# 烟测：若已存在 per_user_pretrain_perN10_L64（或带 _s2020）与 GDN ckp，则只重评 N=10、GDN、seed=2020。
# 用于验证 retest 脚本与指标表输出；无 ckp 时会失败并提示先跑 run_pretrain_perN_streaming_t2t_1m.py

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

L="${L:-64}"
export L
exec python scripts/retest_perN_streaming_t2t_1m.py \
  --Ns "10" \
  --models GDN \
  --seeds 2020 \
  -L "${L}" \
  --n_gpus 1
