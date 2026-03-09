#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 实验脚本 - ml-1m
任务与配置与 run_non_streaming_experiments_1m.py 完全一致，但测试 RecBole 自带的经典/SOTA 基线模型。
选取 3 个：SASRec (2018 自注意力经典)、GRU4Rec (2015 RNN 经典)、LightSANs (2021 轻量自注意力 SOTA)。

配置: dataset=ml-1m, L=128(默认) 或 L=64, epochs=150, worker=4

用法:
  python scripts/run_baseline_experiments_1m.py              # L=128 默认
  python scripts/run_baseline_experiments_1m.py --max_seq_len 64   # L=64
  python scripts/run_baseline_experiments_1m.py -L 64 --saved      # 保存 checkpoint
"""

import argparse
import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

from run_non_streaming_experiments import run_single_model

# RecBole 经典/SOTA 基线：SASRec(自注意力)、LinRec(线性注意力)、GRU4Rec(RNN)、LightSANs(轻量自注意力)
BASELINE_CONFIGS = {
    "SASRec": "recbole/properties/quick_start_config/baselines/sequential_SASRec_1m.yaml",
    "LinRec": "recbole/properties/quick_start_config/baselines/sequential_LinRec_1m.yaml",
    "GRU4Rec": "recbole/properties/quick_start_config/baselines/sequential_GRU4Rec_1m.yaml",
    "LightSANs": "recbole/properties/quick_start_config/baselines/sequential_LightSANs_1m.yaml",
}

METRIC_KEYS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]


def main():
    parser = argparse.ArgumentParser(description="Baseline 实验 ml-1m (SASRec, GRU4Rec, LightSANs)")
    parser.add_argument("--max_seq_len", "-L", type=int, default=128,
                        help="序列长度 L，默认 128；可用 64")
    parser.add_argument("--saved", action="store_true", help="保存 checkpoint")
    args = parser.parse_args()

    dataset = "ml-1m"
    max_seq_len = args.max_seq_len
    epochs = 150
    worker = 4
    show_progress = True
    saved = args.saved

    proj_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(proj_root, "saved", f"baseline_1m_L{max_seq_len}")

    results = {}
    for model_key, config_file in BASELINE_CONFIGS.items():
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

    output_dir = os.path.join(proj_root, "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"baseline_1m_L{max_seq_len}_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"baseline_1m_L{max_seq_len}_{timestamp}.csv")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    model_cols = list(BASELINE_CONFIGS.keys())
    lines = []
    lines.append("Baseline 实验 (streaming_mode=False) - ml-1m")
    lines.append(f"dataset={dataset}, L={max_seq_len}, epochs={epochs}, time={timestamp}")
    lines.append(f"checkpoint_dir={checkpoint_dir}")
    lines.append("")

    sep = "-" * 90
    for split in ["valid", "test"]:
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
    print(f"BASELINE RESULTS (ml-1m, L={max_seq_len})")
    print("=" * 90)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

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
