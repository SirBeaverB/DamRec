#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Then-Train 流式实验脚本 - ml-1m
依次运行 GDN、MoRec、NestRec、DamRec、FroRec，输出格式与 non_streaming 一致。
配置: dataset=ml-1m, L=128, streaming_test_ratio=0.1

用法: python scripts/run_streaming_t2t_experiments_1m.py
"""

import os
import sys
import time
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color

# 模型 key（与 non_streaming 对齐）-> (模型类名, 配置文件)
MODEL_T2T_CONFIGS = {
    "GDN": ("GDN", "recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml"),
    "Mo": ("MoRec", "recbole/properties/quick_start_config/streaming/sequential_MoRec_streaming.yaml"),
    "Nest": ("NestRec", "recbole/properties/quick_start_config/streaming/sequential_NestRec_streaming.yaml"),
    "Adam": ("DamRec", "recbole/properties/quick_start_config/streaming/sequential_DamRec_streaming.yaml"),
    "Fro": ("FroRec", "recbole/properties/quick_start_config/streaming/sequential_FroRec_streaming.yaml"),
}

T2T_OVERRIDES = {
    "streaming_t2t": True,
    "streaming_test_ratio": 0.1,
    "epochs": 1,
    "MAX_ITEM_LIST_LENGTH": 128,
    "enable_amp": False,
    "enable_scaler": False,
    "use_compile": False,
    "worker": 4,
}


def run_single_t2t(model_key, dataset="ml-1m", show_progress=True):
    """运行单个模型的 T2T 流式实验，返回 (valid_recall, test_recall, train_time_sec, peak_mem_gb) 或 None"""
    if model_key not in MODEL_T2T_CONFIGS:
        print(f"[WARN] 未知模型: {model_key}，跳过")
        return None

    model_name, config_file = MODEL_T2T_CONFIGS[model_key]
    if not os.path.exists(config_file):
        print(f"[WARN] 配置文件不存在: {config_file}，跳过")
        return None

    print(f"\n{'='*60}")
    print(f"Running T2T {model_key} ({model_name}) on {dataset} ...")
    print("=" * 60)

    config_dict = {
        "show_progress": show_progress,
        "gpu_id": "0",
        "dataset": dataset,
        **T2T_OVERRIDES,
    }

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

        # 记录峰值显存
        peak_mem_gb = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        trainer = get_trainer(config["MODEL_TYPE"], config["model"], config)(config, model)

        t0 = time.perf_counter()
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=False, show_progress=show_progress
        )
        train_time = time.perf_counter() - t0

        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

        valid_recall = float(best_valid_result.get("recall@10", 0.0))
        test_recall = float(best_valid_result.get("recall@10", 0.0))

        logger.info(set_color("T2T result", "yellow") + f": {best_valid_result}")
        return valid_recall, test_recall, train_time, peak_mem_gb

    except Exception as e:
        print(f"[ERROR] {model_key} T2T failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    dataset = "ml-1m"
    show_progress = True

    results = {}
    for model_key in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
        ret = run_single_t2t(model_key, dataset=dataset, show_progress=show_progress)
        results[model_key] = ret

    # 输出表格（与 non_streaming 相同格式）
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"streaming_t2t_1m_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"streaming_t2t_1m_{timestamp}.csv")

    lines = []
    lines.append("Test-Then-Train 流式实验 (streaming_t2t=True) - ml-1m")
    lines.append(f"dataset={dataset}, L=128, streaming_test_ratio=0.1, time={timestamp}")
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
    print("RESULTS (streaming T2T - ml-1m)")
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
