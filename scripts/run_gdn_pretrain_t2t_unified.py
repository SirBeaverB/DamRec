#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一脚本：GDN 预训练一次，用同一 checkpoint 为五个模型（含 GDN）各跑一次 T2T，输出 non_streaming 风格表格。

流程：
  1. 准备 80/20 划分
  2. GDN 在 80% 上预训练，保存 checkpoint
  3. 用该 checkpoint 依次跑 T2T：GDN（直接加载）、MoRec/NestRec/DamRec/FroRec（嫁接）
  4. 输出表格：valid/test 各 recall@10、mrr@10、ndcg@10、hit@10、precision@10，time，显存

用法:
  python scripts/run_gdn_pretrain_t2t_unified.py
  python scripts/run_gdn_pretrain_t2t_unified.py --ckp saved/GDN-xxx.pth   # 跳过预训练
  python scripts/run_gdn_pretrain_t2t_unified.py -L 64                      # L=64
"""

import argparse
import glob
import os
import sys
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

from run_pretrain_t2t_1m import (
    MODEL_CONFIGS,
    _ensure_split,
    run_pretrain,
    run_t2t_from_ckp,
)

METRIC_KEYS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]
MODEL_COLS = ["GDN", "Mo", "Nest", "Adam", "Fro"]


def main():
    parser = argparse.ArgumentParser(description="GDN 预训练 + 五模型 T2T 统一表格")
    parser.add_argument("--max_seq_len", "-L", type=int, default=128,
                        help="序列长度 L，默认 128；可用 64")
    parser.add_argument("--ckp", type=str, default=None,
                        help="已有 GDN checkpoint，若提供则跳过预训练")
    parser.add_argument("--ckp_dir", type=str, default=None,
                        help="预训练 checkpoint 保存目录")
    parser.add_argument("--t2t_lr", type=float, default=None,
                        help="T2T 流式学习率，默认 0.0001")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    proj = os.path.dirname(os.path.dirname(__file__))
    max_seq_len = args.max_seq_len
    ckp_dir = args.ckp_dir or os.path.join(proj, "saved", f"gdn_pretrain_t2t_unified_L{max_seq_len}")
    show_progress = not args.no_progress

    _ensure_split()

    # 1. 预训练或加载已有 checkpoint
    if args.ckp:
        if os.path.isdir(args.ckp):
            candidates = sorted(glob.glob(os.path.join(args.ckp, "GDN-*.pth")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(args.ckp, "*.pth")))
            if not candidates:
                print(f"目录 {args.ckp} 下未找到 .pth 文件")
                sys.exit(1)
            ckp_path = candidates[-1]
        else:
            ckp_path = args.ckp
        if not os.path.isfile(ckp_path):
            print(f"checkpoint 不存在: {ckp_path}")
            sys.exit(1)
        print(f"使用已有 checkpoint: {ckp_path}")
    else:
        os.makedirs(ckp_dir, exist_ok=True)
        model_name, config_file = MODEL_CONFIGS["GDN"]
        print(f"\n{'='*60}\n预训练 GDN (L={max_seq_len}) ...\n{'='*60}")
        ckp_path, pretrain_valid, pretrain_test = run_pretrain(
            model_name, config_file, ckp_dir, show_progress, max_seq_len=max_seq_len
        )
        print(f"已保存: {ckp_path}")
        print(
            f"  [pretrain 子集] valid recall@10={pretrain_valid.get('recall@10', 'N/A')}, "
            f"test recall@10={pretrain_test.get('recall@10', 'N/A')}\n"
        )

    # 2. 五个模型依次 T2T
    results = {}
    for model_key in MODEL_COLS:
        print(f"\n{'='*60}\nT2T {model_key} ...\n{'='*60}")
        try:
            result, t_sec, peak_mem_gb = run_t2t_from_ckp(
                ckp_path,
                show_progress=show_progress,
                t2t_model=model_key,
                t2t_lr=args.t2t_lr,
                max_seq_len=max_seq_len,
            )
            # T2T 只有一次评估（在流式 test 点），valid 填 N/A，test 用结果
            vres = {k: None for k in METRIC_KEYS}
            tres = {k: float(v) for k, v in result.items() if k in METRIC_KEYS}
            results[model_key] = (vres, tres, t_sec, peak_mem_gb)
        except Exception as e:
            print(f"[ERROR] {model_key} T2T failed: {e}")
            import traceback
            traceback.print_exc()
            results[model_key] = None

    # 3. 输出表格（与 non_streaming 同格式）
    output_dir = os.path.join(proj, "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"gdn_pretrain_t2t_unified_L{max_seq_len}_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"gdn_pretrain_t2t_unified_L{max_seq_len}_{timestamp}.csv")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    lines.append("GDN 预训练 + 五模型 T2T 统一实验 (ml-1m 80/20)")
    lines.append(f"L={max_seq_len}, pretrain_ckp={ckp_path}, time={timestamp}")
    lines.append("T2T 仅有一次评估（流式 test 点），valid 列填 N/A")
    lines.append("")

    sep = "-" * 90
    for split in ["valid", "test"]:
        lines.append(f"--- {split.upper()} ---")
        lines.append("模型\t\t\t" + "\t".join(MODEL_COLS))
        lines.append(sep)
        for mk in METRIC_KEYS:
            row = f"{mk}\t\t\t"
            for k in MODEL_COLS:
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

    lines.append("--- T2T 耗时 ---")
    time_row = "time (s)\t\t"
    mem_row = "显存 (GB)\t\t"
    for k in MODEL_COLS:
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
    print("RESULTS (GDN pretrain + 5 models T2T)")
    print("=" * 90)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

    csv_cols = ["model"] + [f"valid_{m}" for m in METRIC_KEYS] + [f"test_{m}" for m in METRIC_KEYS] + ["t2t_time_sec", "peak_mem_gb"]
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for k in MODEL_COLS:
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
