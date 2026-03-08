#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非流式实验脚本 - ml-100k
依次运行 GDN、MoRec、NestRec、DamRec、FroRec，记录 valid/test recall@10 和训练时间。
配置: dataset=ml-100k, L=50, epochs=100

用法: python scripts/run_non_streaming_experiments_100k.py
"""

import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

from run_non_streaming_experiments import (
    MODEL_CONFIGS,
    run_single_model,
)


def main():
    dataset = "ml-100k"
    max_seq_len = None   # 默认 L=50
    epochs = 100
    show_progress = True
    saved = False

    results = {}
    for model_key, config_file in MODEL_CONFIGS.items():
        ret = run_single_model(
            model_key, config_file,
            dataset=dataset,
            max_seq_len=max_seq_len,
            epochs=epochs,
            saved=saved,
            show_progress=show_progress,
        )
        results[model_key] = ret

    # 输出表格
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"non_streaming_100k_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"non_streaming_100k_{timestamp}.csv")

    lines = []
    lines.append("非流式实验 (streaming_mode=False) - ml-100k")
    lines.append(f"dataset={dataset}, L=50, epochs={epochs}, time={timestamp}")
    lines.append("")
    lines.append("模型\t\t\tGDN\t\tMo\t\tNest\t\tAdam\t\tFro")
    lines.append("-" * 70)

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    valid_row = "valid recall@10\t"
    test_row = "test recall@10\t"
    time_row = "time (s)\t\t"
    mem_row = "显存 (GB)\t\t"
    for k in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
        r = results.get(k)
        if r is None:
            valid_row += "N/A\t\t"
            test_row += "N/A\t\t"
            time_row += "N/A\t\t"
            mem_row += "N/A\t\t"
        else:
            vr, tr, tt, mem = r[0], r[1], r[2], r[3]
            valid_row += f"{fmt(vr)}\t\t"
            test_row += f"{fmt(tr)}\t\t"
            time_row += f"{fmt(tt):>8}\t"
            mem_row += f"{fmt(mem) if mem is not None else 'N/A':>8}\t"

    lines.append(valid_row)
    lines.append(test_row)
    lines.append(time_row)
    lines.append(mem_row)

    table = "\n".join(lines)
    print("\n" + "=" * 70)
    print("RESULTS (ml-100k)")
    print("=" * 70)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("model,valid_recall@10,test_recall@10,train_time_sec,peak_mem_gb\n")
        for k in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
            r = results.get(k)
            if r is None:
                f.write(f"{k},N/A,N/A,N/A,N/A\n")
            else:
                vr, tr, tt, mem = r[0], r[1], r[2], r[3]
                mem_str = f"{mem:.2f}" if mem is not None else "N/A"
                f.write(f"{k},{vr:.4f},{tr:.4f},{tt:.2f},{mem_str}\n")
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
