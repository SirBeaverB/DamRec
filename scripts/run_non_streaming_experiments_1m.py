#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非流式实验脚本 - ml-1m
依次运行 GDN、MoRec、NestRec、DamRec、FroRec，记录 valid/test 各项指标和训练时间。
配置: dataset=ml-1m, L=128(默认) 或 L=64, epochs=150, worker=4

输出: recall@10 / mrr@10 / ndcg@10 / hit@10 / precision@10（valid+test）、训练时间(s)、峰值显存(GB)，
      写入 experiment_results/non_streaming_1m_L{L}_*.txt 与同名 .csv

用法:
  python scripts/run_non_streaming_experiments_1m.py                         # L=128，跑全部模型
  python scripts/run_non_streaming_experiments_1m.py --max_seq_len 64        # L=64
  python scripts/run_non_streaming_experiments_1m.py -L 64 --models GDN DamRec   # 只跑 GDN 与 DamRec（Adam 为旧别名）
  python scripts/run_non_streaming_experiments_1m.py -L 64 2>&1 | tee logs/ml1m_l64.txt   # 同时保存终端日志
"""

import argparse
import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

from run_non_streaming_experiments import (
    MODEL_CONFIGS,
    MODEL_KEY_ALIASES,
    resolve_model_key,
    run_single_model,
)

# 要记录的指标（与 RecBole 默认 metrics 一致）
METRIC_KEYS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]


def main():
    parser = argparse.ArgumentParser(description="非流式实验 ml-1m")
    parser.add_argument("--max_seq_len", "-L", type=int, default=128,
                        help="序列长度 L，默认 128；可用 64")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="KEY",
        help="只跑指定模型键，如: GDN DamRec。默认跑全部。可选: GDN Mo Nest DamRec Fro（Adam 为 DamRec 别名）",
    )
    parser.add_argument("--saved", action="store_true", help="保存 checkpoint")
    args = parser.parse_args()

    dataset = "ml-1m"
    max_seq_len = args.max_seq_len
    epochs = 150
    worker = 4
    show_progress = True
    saved = args.saved

    _valid_cli = set(MODEL_CONFIGS.keys()) | set(MODEL_KEY_ALIASES.keys())
    if args.models:
        bad = [k for k in args.models if k not in _valid_cli]
        if bad:
            raise SystemExit(
                f"Unknown --models keys: {bad}. Valid: {sorted(_valid_cli)}"
            )
        model_cols = []
        _seen = set()
        for k in args.models:
            nk = resolve_model_key(k)
            if nk not in _seen:
                _seen.add(nk)
                model_cols.append(nk)
    else:
        model_cols = list(MODEL_CONFIGS.keys())

    # checkpoint 目录包含 L，便于区分
    proj_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(proj_root, "saved", f"non_streaming_1m_L{max_seq_len}")

    results = {}
    for model_key in model_cols:
        config_file = MODEL_CONFIGS[model_key]
        ret = run_single_model(
            model_key, config_file,
            dataset=dataset,
            max_seq_len=max_seq_len,
            epochs=epochs,
            worker=worker,
            saved=saved,
            show_progress=show_progress,
            checkpoint_dir=checkpoint_dir,
        )
        results[model_key] = ret

    # 输出表格（文件名包含 L）
    output_dir = os.path.join(proj_root, "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"non_streaming_1m_L{max_seq_len}_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"non_streaming_1m_L{max_seq_len}_{timestamp}.csv")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    lines.append(f"非流式实验 (streaming_mode=False) - ml-1m")
    lines.append(f"dataset={dataset}, L={max_seq_len}, epochs={epochs}, time={timestamp}")
    lines.append(f"checkpoint_dir={checkpoint_dir}")
    lines.append("")

    sep = "-" * max(90, 12 * len(model_cols) + 20)

    # 按指标分行：valid / test 各一组
    for split, prefix in [("valid", "valid"), ("test", "test")]:
        lines.append(f"--- {split.upper()} ---")
        lines.append("模型\t\t\t" + "\t".join(model_cols))
        lines.append(sep)
        for mk in METRIC_KEYS:
            row = f"{mk}\t\t\t"
            for k in model_cols:
                r = results.get(k)
                if r is None:
                    row += "N/A\t\t"
                else:
                    vres, tres, tt, mem = r[0], r[1], r[2], r[3]
                    d = vres if split == "valid" else tres
                    val = d.get(mk)
                    row += f"{fmt(val)}\t\t" if val is not None else "N/A\t\t"
            lines.append(row)
        lines.append("")

    # time & mem
    lines.append("--- 训练 ---")
    time_row = "time (s)\t\t"
    mem_row = "显存 (GB)\t\t"
    for k in model_cols:
        r = results.get(k)
        if r is None:
            time_row += "N/A\t\t"
            mem_row += "N/A\t\t"
        else:
            _, _, tt, mem = r[0], r[1], r[2], r[3]
            time_row += f"{fmt(tt):>8}\t"
            mem_row += f"{fmt(mem) if mem is not None else 'N/A':>8}\t"
    lines.append(time_row)
    lines.append(mem_row)

    table = "\n".join(lines)
    print("\n" + "=" * 90)
    print(f"RESULTS (ml-1m, L={max_seq_len})")
    print("=" * 90)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

    # CSV：所有指标 + time + mem
    csv_cols = ["model"] + [f"valid_{m}" for m in METRIC_KEYS] + [f"test_{m}" for m in METRIC_KEYS] + ["train_time_sec", "peak_mem_gb"]
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for k in model_cols:
            r = results.get(k)
            if r is None:
                f.write(k + "," + ",".join(["N/A"] * (len(METRIC_KEYS) * 2 + 2)) + "\n")
            else:
                vres, tres, tt, mem = r[0], r[1], r[2], r[3]
                parts = [k]
                for d in [vres, tres]:
                    for mk in METRIC_KEYS:
                        v = d.get(mk)
                        parts.append(f"{v:.4f}" if v is not None else "N/A")
                parts.append(f"{tt:.2f}" if tt is not None else "N/A")
                parts.append(f"{mem:.2f}" if mem is not None else "N/A")
                f.write(",".join(parts) + "\n")
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
