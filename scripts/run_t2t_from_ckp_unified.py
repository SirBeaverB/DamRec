#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2T 集成实验：从已有 checkpoint 加载预训练模型，依次测试 5 个模型（GDN/MoRec/NestRec/DamRec/FroRec）的 T2T 效果。

无需从头预训练，直接读入 checkpoint 即可运行。checkpoint 可来自：
  - non_streaming 实验：saved/non_streaming_1m_L128/GDN-xxx.pth
  - 预训练脚本：saved/pretrain_t2t_1m/GDN-xxx.pth
  - 任意 GDN/MoRec/NestRec/DamRec/FroRec 的 .pth 文件

用法:
  # 必须指定 --ckp，不支持预训练（用 run_gdn_pretrain_t2t_unified.py 做预训练）
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/non_streaming_1m_L128
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/non_streaming_1m_L128/GDN-xxx.pth
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/pretrain_t2t_1m
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/non_streaming_1m_L128 -L 64
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/GDN-xxx.pth --t2t_lr 0.00005
  # Zero-Shot 体检：仅评估不训练，排查预训练权重是否加载成功
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/gdn_pretrain_t2t_unified_L64 --zero_shot -L 64
  # 工业标准：先导出 user_states.pt 再 T2T（若已存在则跳过）
  python scripts/run_t2t_from_ckp_unified.py --ckp saved/gdn_pretrain_t2t_unified_L64 --dump_state -L 64
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
    run_t2t_from_ckp,
    run_state_dump,
)

METRIC_KEYS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]
MODEL_COLS = ["GDN", "Mo", "Nest", "Adam", "Fro"]


def _resolve_ckp(ckp_arg):
    """解析 checkpoint 路径：支持文件或目录。目录时优先 GDN/MoRec/NestRec/DamRec/FroRec，否则任意 *.pth"""
    if os.path.isfile(ckp_arg):
        return ckp_arg
    if os.path.isdir(ckp_arg):
        # 优先 GDN（嫁接基座），其次其他四模型
        for prefix in ["GDN", "MoRec", "NestRec", "DamRec", "FroRec"]:
            candidates = sorted(glob.glob(os.path.join(ckp_arg, f"{prefix}-*.pth")))
            if candidates:
                return candidates[-1]
        candidates = sorted(glob.glob(os.path.join(ckp_arg, "*.pth")))
        if not candidates:
            return None
        return candidates[-1]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="从 checkpoint 加载预训练模型，五模型 T2T 集成实验（不预训练）"
    )
    parser.add_argument("--ckp", "-c", type=str, required=True,
                        help="checkpoint 路径或目录，如 saved/non_streaming_1m_L128 或 saved/GDN-xxx.pth")
    parser.add_argument("--max_seq_len", "-L", type=int, default=128,
                        help="序列长度 L，默认 128；需与 checkpoint 训练时一致")
    parser.add_argument("--t2t_lr", type=float, default=None,
                        help="T2T 流式学习率，默认 0.0001")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-Shot 体检：仅评估不训练，用于排查预训练权重是否加载成功")
    parser.add_argument("--dump_state", action="store_true",
                        help="先导出 user_states.pt (工业标准)，再跑 T2T；若已存在则跳过")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    ckp_path = _resolve_ckp(args.ckp)
    if not ckp_path or not os.path.isfile(ckp_path):
        print(f"checkpoint 不存在或无法解析: {args.ckp}")
        print("  支持: 1) 文件路径 saved/xxx/GDN-xxx.pth  2) 目录 saved/non_streaming_1m_L128")
        sys.exit(1)

    proj = os.path.dirname(os.path.dirname(__file__))
    max_seq_len = args.max_seq_len
    show_progress = not args.no_progress

    _ensure_split()

    ckp_dir = os.path.dirname(ckp_path)
    if args.dump_state:
        for model_key in MODEL_COLS:
            model_name = MODEL_CONFIGS[model_key][0]
            states_file = os.path.join(ckp_dir, f"user_states_{model_name}.pt")
            if not os.path.isfile(states_file):
                print(f"\n[State Dump] {model_name} 导出用户状态 -> {states_file}")
                run_state_dump(ckp_path, save_path=states_file, model_name=model_name, max_seq_len=max_seq_len, show_progress=show_progress)

    print(f"\n使用预训练 checkpoint: {ckp_path}")
    mode = "Zero-Shot 体检 (仅评估)" if args.zero_shot else "T2T"
    print(f"L={max_seq_len}，模式={mode}，将依次: {', '.join(MODEL_COLS)}\n")

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
                zero_shot=args.zero_shot,
            )
            vres = {k: None for k in METRIC_KEYS}
            tres = {k: float(v) for k, v in result.items() if k in METRIC_KEYS}
            results[model_key] = (vres, tres, t_sec, peak_mem_gb)
        except Exception as e:
            print(f"[ERROR] {model_key} T2T failed: {e}")
            import traceback
            traceback.print_exc()
            results[model_key] = None

    # 输出表格
    output_dir = os.path.join(proj, "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "zero_shot" if args.zero_shot else "t2t"
    out_file = os.path.join(output_dir, f"t2t_from_ckp_L{max_seq_len}_{suffix}_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"t2t_from_ckp_L{max_seq_len}_{suffix}_{timestamp}.csv")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    title = "Zero-Shot 体检" if args.zero_shot else "T2T 集成实验"
    lines.append(f"{title}：从 checkpoint 加载预训练，五模型 (ml-1m 80/20)")
    lines.append(f"ckp={ckp_path}, L={max_seq_len}, mode={'zero_shot(仅评估)' if args.zero_shot else 't2t'}, time={timestamp}")
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
    print("RESULTS (T2T from checkpoint)")
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
