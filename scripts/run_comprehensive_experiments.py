#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: 综合测试脚本
- 对比 GDN、MoRec、NestRec、DamRec、FroRec、SASRec(baseline)
- 支持: 不同数据集、epoch 消融、长短序列消融

用法:
  python scripts/run_comprehensive_experiments.py
  python scripts/run_comprehensive_experiments.py --dataset ml-1m
  python scripts/run_comprehensive_experiments.py --epochs 20
  python scripts/run_comprehensive_experiments.py --max_seq_len 128
"""

import argparse
import os
import sys
from datetime import datetime

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    init_logger,
    init_seed,
    get_model,
    get_trainer,
    set_color,
    get_flops,
)
from recbole.data.transform import construct_transform

from run_non_streaming_experiments import resolve_model_key

MODEL_CONFIGS = {
    "GDN": "recbole/properties/quick_start_config/sequential_GDN.yaml",
    "Mo": "recbole/properties/quick_start_config/sequential_MoRec.yaml",
    "Nest": "recbole/properties/quick_start_config/sequential_NestRec.yaml",
    "DamRec": "recbole/properties/quick_start_config/sequential_DamRec.yaml",
    "Fro": "recbole/properties/quick_start_config/sequential_FroRec.yaml",
    "SASRec": "recbole/properties/quick_start_config/sequential_SASRec.yaml",
}

# 旧键名 Adam 与 DamRec 同义（见 run_non_streaming_experiments.MODEL_KEY_ALIASES）
CONFIG_TO_MODEL = {
    "GDN": "GDN",
    "Mo": "MoRec",
    "Nest": "NestRec",
    "DamRec": "DamRec",
    "Fro": "FroRec",
    "SASRec": "SASRec",
}


def run_single_model(
    model_key,
    config_file,
    dataset="ml-100k",
    epochs=None,
    max_seq_len=None,
    saved=False,
    show_progress=False,
):
    """运行单个模型，返回 (valid_recall, test_recall, train_time_sec) 或 None"""
    model_name = CONFIG_TO_MODEL[model_key]
    print(f"\n{'='*60}")
    print(f"Running {model_key} ({model_name}) ...")
    print("=" * 60)

    config_dict = {"show_progress": show_progress}
    if epochs is not None:
        config_dict["epochs"] = epochs
    if max_seq_len is not None:
        config_dict["MAX_ITEM_LIST_LENGTH"] = max_seq_len

    try:
        config = Config(
            model=model_name,
            dataset=dataset,
            config_file_list=[config_file],
            config_dict=config_dict,
        )
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)

        from logging import getLogger
        logger = getLogger()

        dataset_obj = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)

        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = get_model(config["model"])(config, train_data._dataset).to(config["device"])

        if model_name != "FroRec":
            try:
                transform = construct_transform(config)
                flops = get_flops(model, dataset_obj, config["device"], logger, transform)
                logger.info(set_color("FLOPs", "blue") + f": {flops}")
            except Exception as e:
                logger.warning(f"get_flops skipped: {e}")

        use_compile = config.final_config_dict.get("use_compile", False)
        single_spec = config.final_config_dict.get("single_spec", True)
        if use_compile and hasattr(torch, "compile") and single_spec:
            model = torch.compile(model, mode="reduce-overhead")

        trainer = get_trainer(config["MODEL_TYPE"], config["model"], config)(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=show_progress
        )
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=show_progress
        )

        valid_recall = float(best_valid_result.get("recall@10", 0.0))
        test_recall = float(test_result.get("recall@10", 0.0))
        train_time = getattr(trainer, "total_train_time", 0.0)

        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
        logger.info(set_color("test result", "yellow") + f": {test_result}")

        return valid_recall, test_recall, train_time

    except Exception as e:
        print(f"[ERROR] {model_key} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="ml-100k")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 epochs，用于消融")
    parser.add_argument("--max_seq_len", type=int, default=None, help="覆盖 MAX_ITEM_LIST_LENGTH，用于长短序列消融")
    parser.add_argument("--saved", action="store_true", help="保存 checkpoint")
    parser.add_argument("--no_progress", action="store_true", help="不显示进度条")
    parser.add_argument("--models", type=str, default=None, help="逗号分隔，如 GDN,Mo,Nest。默认全部")
    args = parser.parse_args()

    show_progress = not args.no_progress
    raw_keys = [k.strip() for k in args.models.split(",")] if args.models else list(MODEL_CONFIGS.keys())
    model_keys = []
    _seen = set()
    for k in raw_keys:
        nk = resolve_model_key(k)
        if nk not in MODEL_CONFIGS:
            print(f"[WARN] 未知模型 {k}（解析后 {nk}），跳过")
            continue
        if nk not in _seen:
            _seen.add(nk)
            model_keys.append(nk)

    results = {}
    for model_key in model_keys:
        config_file = MODEL_CONFIGS[model_key]
        ret = run_single_model(
            model_key,
            config_file,
            dataset=args.dataset,
            epochs=args.epochs,
            max_seq_len=args.max_seq_len,
            saved=args.saved,
            show_progress=show_progress,
        )
        results[model_key] = ret

    # 输出表格
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_ep{args.epochs}" if args.epochs else ""
    suffix += f"_L{args.max_seq_len}" if args.max_seq_len else ""
    out_file = os.path.join(output_dir, f"comprehensive_{args.dataset}{suffix}_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"comprehensive_{args.dataset}{suffix}_{timestamp}.csv")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "模型\t\t\t" + "\t".join(model_keys)
    lines = [
        "综合实验 (Step 3)",
        f"dataset={args.dataset}, epochs={args.epochs or 'default'}, max_seq_len={args.max_seq_len or 'default'}",
        f"time={timestamp}",
        "",
        header,
        "-" * (20 + 12 * len(model_keys)),
    ]

    valid_row = "valid recall@10\t"
    test_row = "test recall@10\t"
    time_row = "time (s)\t\t"
    for k in model_keys:
        r = results.get(k)
        if r is None:
            valid_row += "N/A\t\t"
            test_row += "N/A\t\t"
            time_row += "N/A\t\t"
        else:
            vr, tr, tt = r
            valid_row += f"{fmt(vr)}\t\t"
            test_row += f"{fmt(tr)}\t\t"
            time_row += f"{fmt(tt):>8}\t"

    lines.extend([valid_row, test_row, time_row])
    table = "\n".join(lines)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("model,valid_recall@10,test_recall@10,train_time_sec\n")
        for k in model_keys:
            r = results.get(k)
            if r is None:
                f.write(f"{k},N/A,N/A,N/A\n")
            else:
                vr, tr, tt = r
                f.write(f"{k},{vr:.4f},{tr:.4f},{tt:.2f}\n")
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
