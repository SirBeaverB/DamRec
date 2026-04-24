#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FroRec 二阶矩 V 有效性消融（streaming T2T 场景）

目的：
  隔离 FroRec 中 scalar Frobenius 二阶矩 V 的独立贡献。对比：
    Fro     = M + bc1 + scalar V  (完整 F-Adam)
    FroNoV  = M + bc1, denom=1    (只保留一阶动量 + Adam bias correction)
  两者唯一差异是 denom 是否等于 sqrt(V/bc2)+eps。若 Fro > FroNoV 稳定显著，
  则 scalar V 有独立贡献；若 Fro ≈ FroNoV，则 Fro 相对 GDN 的增益全部来自 M + bc1，
  paper 需改叙事为「一阶动量 + bias correction 是关键」，不再宣称二阶预条件子有效。

实验协议（复用 run_pretrain_perN_streaming_t2t_1m.py）：
  每个 (model, N, seed) 独立 pretrain ml-1m-pretrain-perN → dump user_states
  → streaming T2T on ml-1m-t2t（prequential）。ckp/state 按 seed 隔离。

默认配置：
  Ns     = [100]       (已有单 seed N=100 显示 Fro > GDN +9.6%, 先验最强信号点)
  models = [Fro, FroNoV] (可选加 GDN 做三方对照)
  seeds  = [2020, 2021, 2022]
  总 2 × 1 × 3 = 6 runs，双卡并行约 1 小时 wall

Usage:
  # 默认：Fro vs FroNoV × N=100 × 3 seed
  python scripts/run_fro_ablation_t2t_1m.py

  # 三方对照：加 GDN 做参照基线
  python scripts/run_fro_ablation_t2t_1m.py --include_gdn

  # 扫多个 N（回答「V 贡献是否随 N 变化」）
  python scripts/run_fro_ablation_t2t_1m.py --Ns 20,50,100

  # 自定 seed
  python scripts/run_fro_ablation_t2t_1m.py --seeds 2020,2021,2022,2023,2024
"""

import argparse
import os
import subprocess
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(_script_dir)
PERN_SCRIPT = os.path.join(_script_dir, "run_pretrain_perN_streaming_t2t_1m.py")


def _parse_csv_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="FroRec 二阶矩 V 消融 (Fro vs FroNoV) on ml-1m streaming T2T"
    )
    parser.add_argument("--Ns", type=_parse_csv_int, default=[100],
                        help="每用户保留真实交互数，默认 [100]")
    parser.add_argument("--seeds", type=_parse_csv_int, default=[2020, 2021, 2022],
                        help="随机 seed 列表，默认 [2020,2021,2022]")
    parser.add_argument("--include_gdn", action="store_true",
                        help="三方对照：加 GDN 做非自适应基线")
    parser.add_argument("-L", "--max_seq_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--t2t_lr", type=float, default=None)
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--skip_dump", action="store_true")
    parser.add_argument("--skip_t2t", action="store_true")
    parser.add_argument("--force_redump", action="store_true")
    args = parser.parse_args()

    models = ["Fro", "FroNoV"]
    if args.include_gdn:
        models = ["GDN"] + models

    cmd = [
        sys.executable, PERN_SCRIPT,
        "--Ns", ",".join(str(n) for n in args.Ns),
        "--models", ",".join(models),
        "--seeds", ",".join(str(s) for s in args.seeds),
        "--max_seq_len", str(args.max_seq_len),
        "--n_gpus", str(args.n_gpus),
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.t2t_lr is not None:
        cmd += ["--t2t_lr", str(args.t2t_lr)]
    if args.show_progress:
        cmd += ["--show_progress"]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.skip_pretrain:
        cmd += ["--skip_pretrain"]
    if args.skip_dump:
        cmd += ["--skip_dump"]
    if args.skip_t2t:
        cmd += ["--skip_t2t"]
    if args.force_redump:
        cmd += ["--force_redump"]

    n_jobs = len(models) * len(args.Ns) * len(args.seeds)
    print(f"[fro-ablation] {n_jobs} jobs = {len(models)} models × "
          f"{len(args.Ns)} Ns × {len(args.seeds)} seeds")
    print(f"              models={models} Ns={args.Ns} seeds={args.seeds}")
    print(f"              cmd: {' '.join(cmd)}")

    rc = subprocess.call(cmd, cwd=PROJ)
    sys.exit(rc)


if __name__ == "__main__":
    main()
