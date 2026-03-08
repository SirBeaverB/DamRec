#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非流式实验脚本：依次运行 GDN、MoRec、NestRec、DamRec、FroRec，记录 valid/test recall@10 和训练时间。
用法: python scripts/run_non_streaming_experiments.py
"""

import os
import sys
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    init_seed,
    get_model,
    get_trainer,
    set_color,
    get_flops,
    get_environment,
)


# 模型名 -> 配置文件
MODEL_CONFIGS = {
    "GDN": "recbole/properties/quick_start_config/sequential_GDN.yaml",
    "Mo": "recbole/properties/quick_start_config/sequential_MoRec.yaml",
    "Nest": "recbole/properties/quick_start_config/sequential_NestRec.yaml",
    "Adam": "recbole/properties/quick_start_config/sequential_DamRec.yaml",
    "Fro": "recbole/properties/quick_start_config/sequential_FroRec.yaml",
}

# 配置文件 -> 实际模型类名
CONFIG_TO_MODEL = {
    "GDN": "GDN",
    "Mo": "MoRec",
    "Nest": "NestRec",
    "Adam": "DamRec",
    "Fro": "FroRec",
}


def run_single_model(model_key, config_file, dataset="ml-100k", saved=False, show_progress=False):
    """运行单个模型，返回 (valid_recall, test_recall, train_time_sec) 或 None（失败时）"""
    model_name = CONFIG_TO_MODEL[model_key]
    print(f"\n{'='*60}")
    print(f"Running {model_key} ({model_name}) ...")
    print("=" * 60)

    try:
        config = Config(
            model=model_name,
            dataset=dataset,
            config_file_list=[config_file],
            config_dict={"show_progress": show_progress},
        )
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)

        from logging import getLogger
        logger = getLogger()

        dataset_obj = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)

        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = get_model(config["model"])(config, train_data._dataset).to(config["device"])

        # 确认 Chunk 级优化是否生效（需 FLA 安装）
        chunk_status = {}
        if model_name == "MoRec":
            chunk_status["MoRec"] = "Chunk" if model.use_chunk_momentum else "Token (FLA 未安装)"
        elif model_name == "NestRec":
            chunk_status["NestRec"] = "Chunk" if model.use_chunk_nesterov else "Token (FLA 未安装)"
        elif model_name == "DamRec":
            chunk_status["DamRec"] = "Chunk" if model.use_chunk_adam else "Token (FLA 未安装)"
        elif model_name == "FroRec":
            chunk_status["FroRec"] = "Chunk (FroRec 仅 Chunk)"
        if chunk_status:
            logger.info(set_color("[Chunk 级]", "cyan") + f" {model_name}: {list(chunk_status.values())[0]}")

        # FroRec 需 FLA+CUDA，get_flops 会触发 forward 可能失败，跳过
        if model_name != "FroRec":
            try:
                transform = construct_transform(config)
                flops = get_flops(model, dataset_obj, config["device"], logger, transform)
                logger.info(set_color("FLOPs", "blue") + f": {flops}")
            except Exception as e:
                logger.warning(f"get_flops skipped: {e}")

        if config["use_compile"] and hasattr(torch, "compile") and config["single_spec"]:
            model = torch.compile(model, mode="reduce-overhead")

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=show_progress
        )
        test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=show_progress)

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
    dataset = "ml-100k"
    show_progress = True
    saved = False  # 实验脚本不保存 checkpoint，节省磁盘

    results = {}
    for model_key, config_file in MODEL_CONFIGS.items():
        ret = run_single_model(model_key, config_file, dataset=dataset, saved=saved, show_progress=show_progress)
        results[model_key] = ret

    # 输出表格
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"non_streaming_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"non_streaming_{timestamp}.csv")

    lines = []
    lines.append("非流式实验 (streaming_mode=False)")
    lines.append(f"dataset={dataset}, time={timestamp}")
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
    for k in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
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

    lines.append(valid_row)
    lines.append(test_row)
    lines.append(time_row)

    table = "\n".join(lines)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

    # CSV for easy import
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("model,valid_recall@10,test_recall@10,train_time_sec\n")
        for k in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
            r = results.get(k)
            if r is None:
                f.write(f"{k},N/A,N/A,N/A\n")
            else:
                vr, tr, tt = r
                f.write(f"{k},{vr:.4f},{tr:.4f},{tt:.2f}\n")
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
