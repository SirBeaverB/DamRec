#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDN / GRU4Rec @ ml-1m：在**单表 dataset=ml-1m** 上做 RecBole 非流式「全库时间序 + RS + group_by=none」一刀切段。
**不是** `prepare_ml1m_80_20_split` + `run_pretrain_t2t_1m` 那条「前 80% 预训、后 20% 流式 T2T」管线。
要对齐 80% pretrain / 20% 流式测试请用:
  python scripts/run_pretrain_t2t_1m.py --mode full --model GRU4Rec
（或 GDN 等，见该脚本 --help）

用法（本脚本的离线全表 RS）:
  python scripts/run_sequential_1m_global_time_cut.py --model GDN
  python scripts/run_sequential_1m_global_time_cut.py --model GRU4Rec
  python scripts/run_sequential_1m_global_time_cut.py --model GDN -L 64 --rs 0.8,0,0.2 --saved
"""

import argparse
import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

from run_non_streaming_experiments import run_single_model

PROJ_ROOT = os.path.dirname(_script_dir)

CONFIGS = {
    "GDN": "recbole/properties/quick_start_config/sequential_GDN_1m_global_time_cut.yaml",
    "GRU4Rec": "recbole/properties/quick_start_config/baselines/sequential_GRU4Rec_1m_global_time_cut.yaml",
}


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
        description="GDN or GRU4Rec on ml-1m: global time order + RS one-cut (group_by=none)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GDN",
        choices=list(CONFIGS.keys()),
        help="模型名，默认 GDN",
    )
    parser.add_argument("--max_seq_len", "-L", type=int, default=128, help="MAX_ITEM_LIST_LENGTH，默认 128")
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数，默认 150")
    parser.add_argument("--worker", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--saved", action="store_true", help="保存 best checkpoint")
    parser.add_argument(
        "--rs",
        type=str,
        default=None,
        help="覆盖切分比例 train,valid,test，如 0.8,0.1,0.1 或 0.8,0,0.2",
    )
    args = parser.parse_args()

    mkey = args.model
    config_relpath = CONFIGS[mkey]
    config_abspath = os.path.join(PROJ_ROOT, config_relpath)
    if not os.path.isfile(config_abspath):
        raise SystemExit(f"Missing config: {config_abspath}")

    tag = f"{mkey.lower()}_1m_global_time_cut"
    checkpoint_dir = os.path.join(PROJ_ROOT, "saved", f"{tag}_L{args.max_seq_len}")

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
        mkey,
        config_relpath,
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
    out_dir = os.path.join(PROJ_ROOT, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(out_dir, f"{tag}_L{args.max_seq_len}_{ts}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"{mkey} ml-1m 全库时间一刀切 (order=TO, group_by=none, RS)\n")
        f.write(f"config={config_relpath}\n")
        f.write(f"rs_override={args.rs!r}\n")
        f.write(f"L={args.max_seq_len} epochs={args.epochs} train_time_s={train_time} peak_mem_gb={peak_mem}\n\n")
        f.write(f"valid: {valid_result}\n\ntest: {test_result}\n")
    print(f"\nWrote {out_txt}")


if __name__ == "__main__":
    main()
