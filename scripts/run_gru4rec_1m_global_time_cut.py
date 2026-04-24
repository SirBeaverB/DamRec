#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU4Rec @ ml-1m：全库时间排序后「一刀切」比例划分（与按用户留一的 baseline 不同）

RecBole 设置（见 sequential_GRU4Rec_1m_global_time_cut.yaml）：
  eval_args.order: TO        → 先按 TIME 字段全表升序
  eval_args.group_by: none  → 不按 user 分组
  eval_args.split: RS      → 在排序后的行序列上按 [p,q,r] 切 train/valid/test

与 scripts/run_baseline_experiments_1m.py 中 GRU4Rec 用的 LS 留一+TO 相对照。

用法:
  python scripts/run_gru4rec_1m_global_time_cut.py
  python scripts/run_gru4rec_1m_global_time_cut.py -L 64
  python scripts/run_gru4rec_1m_global_time_cut.py --rs 0.8,0,0.2
  python scripts/run_gru4rec_1m_global_time_cut.py --saved
"""

import argparse
import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

from run_non_streaming_experiments import run_single_model

CONFIG = "recbole/properties/quick_start_config/baselines/sequential_GRU4Rec_1m_global_time_cut.yaml"


def _parse_rs(s: str):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"--rs 需要三个数 train,valid,test，如 0.8,0.1,0.1，得到: {parts}")
    t = sum(parts)
    if t <= 0:
        raise SystemExit(f"--rs 之和必须为正，得到 sum={t}")
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="GRU4Rec on ml-1m with global time order + RS one-cut (group_by=none)"
    )
    parser.add_argument("--max_seq_len", "-L", type=int, default=128, help="MAX_ITEM_LIST_LENGTH，默认 128")
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数，默认 150")
    parser.add_argument("--worker", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--saved", action="store_true", help="保存 best checkpoint")
    parser.add_argument(
        "--rs",
        type=str,
        default=None,
        help="覆盖切分比例 train,valid,test，如 0.8,0.1,0.1 或 0.8,0,0.2（与 prepare_ml1m 的 80/20 可对照，注意 valid 为 0 时 early_stop 仍可用 train loss）",
    )
    args = parser.parse_args()

    proj_root = os.path.dirname(_script_dir)
    checkpoint_dir = os.path.join(proj_root, "saved", f"gru4rec_1m_global_time_cut_L{args.max_seq_len}")

    config_overrides = None
    if args.rs:
        p, q, r = _parse_rs(args.rs)
        config_overrides = {
            "eval_args": {
                "split": {"RS": [p, q, r]},
                "order": "TO",
                "group_by": "none",
                "mode": "full",
            }
        }

    ret = run_single_model(
        "GRU4Rec",
        CONFIG,
        dataset="ml-1m",
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        worker=args.worker,
        saved=args.saved,
        show_progress=True,
        checkpoint_dir=checkpoint_dir,
        config_overrides=config_overrides,
    )
    if ret is None:
        raise SystemExit(1)

    valid_result, test_result, train_time, peak_mem = ret
    out_dir = os.path.join(proj_root, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(out_dir, f"gru4rec_1m_global_time_cut_L{args.max_seq_len}_{ts}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("GRU4Rec ml-1m 全库时间一刀切 (order=TO, group_by=none, RS)\n")
        f.write(f"config={CONFIG}\n")
        f.write(f"rs_override={args.rs!r}\n")
        f.write(f"L={args.max_seq_len} epochs={args.epochs} train_time_s={train_time} peak_mem_gb={peak_mem}\n\n")
        f.write(f"valid: {valid_result}\n\ntest: {test_result}\n")
    print(f"\nWrote {out_txt}")


if __name__ == "__main__":
    main()
