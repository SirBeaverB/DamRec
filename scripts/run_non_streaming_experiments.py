#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非流式实验脚本：依次运行 GDN、MoRec、NestRec、DamRec、FroRec，记录 valid/test recall@10 和训练时间。
建议使用专用脚本：
  python scripts/run_non_streaming_experiments_100k.py   # ml-100k, L=50, epochs=100
  python scripts/run_non_streaming_experiments_1m.py     # ml-1m, L=128, epochs=150
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


def run_single_model(model_key, config_file, dataset="ml-100k", max_seq_len=None, epochs=None, worker=None, saved=False, show_progress=False, checkpoint_dir=None):
    """运行单个模型，返回 (valid_result_dict, test_result_dict, train_time_sec, peak_mem_gb) 或 None（失败时）。
    valid_result_dict / test_result_dict 包含 recall@10, mrr@10, ndcg@10, hit@10, precision@10 等。
    model_key 若不在 CONFIG_TO_MODEL 中（如 RecBole baseline），则直接用 model_key 作为模型类名。"""
    model_name = CONFIG_TO_MODEL.get(model_key, model_key)
    print(f"\n{'='*60}")
    print(f"Running {model_key} ({model_name}) ...")
    print("=" * 60)

    config_dict = {"show_progress": show_progress, "gpu_id": "0", "dataset": dataset}
    if max_seq_len is not None:
        config_dict["MAX_ITEM_LIST_LENGTH"] = max_seq_len
    if epochs is not None:
        config_dict["epochs"] = epochs
    if worker is not None:
        config_dict["worker"] = worker
    if checkpoint_dir is not None:
        config_dict["checkpoint_dir"] = checkpoint_dir

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

        # 记录峰值显存
        peak_mem_gb = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=show_progress
        )
        test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=show_progress)

        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

        train_time = getattr(trainer, "total_train_time", 0.0)
        valid_result = {k: float(v) for k, v in best_valid_result.items()}
        test_result = {k: float(v) for k, v in test_result.items()}

        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
        logger.info(set_color("test result", "yellow") + f": {test_result}")

        return valid_result, test_result, train_time, peak_mem_gb

    except Exception as e:
        print(f"[ERROR] {model_key} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 默认 ml-100k；建议用 run_non_streaming_experiments_100k.py / _1m.py
    dataset = "ml-100k"
    max_seq_len = None
    epochs = 100
    show_progress = True
    saved = False

    results = {}
    for model_key, config_file in MODEL_CONFIGS.items():
        ret = run_single_model(
            model_key, config_file,
            dataset=dataset, max_seq_len=max_seq_len, epochs=epochs,
            saved=saved, show_progress=show_progress
        )
        results[model_key] = ret

    # 输出表格
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"non_streaming_{dataset}_{timestamp}.txt")
    csv_file = os.path.join(output_dir, f"non_streaming_{dataset}_{timestamp}.csv")

    lines = []
    lines.append(f"非流式实验 (streaming_mode=False) - {dataset}")
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
    mem_row = "显存 (GB)\t\t"
    for k in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
        r = results.get(k)
        if r is None:
            valid_row += "N/A\t\t"
            test_row += "N/A\t\t"
            time_row += "N/A\t\t"
            mem_row += "N/A\t\t"
        else:
            vres, tres, tt, mem = r[0], r[1], r[2], r[3]
            vr = vres.get("recall@10") if isinstance(vres, dict) else vres
            tr = tres.get("recall@10") if isinstance(tres, dict) else tres
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
    print("RESULTS")
    print("=" * 70)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")

    # CSV for easy import
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("model,valid_recall@10,test_recall@10,train_time_sec,peak_mem_gb\n")
        for k in ["GDN", "Mo", "Nest", "Adam", "Fro"]:
            r = results.get(k)
            if r is None:
                f.write(f"{k},N/A,N/A,N/A,N/A\n")
            else:
                vres, tres, tt, mem = r[0], r[1], r[2], r[3]
                vr = vres.get("recall@10") if isinstance(vres, dict) else vres
                tr = tres.get("recall@10") if isinstance(tres, dict) else tres
                vr_str = f"{vr:.4f}" if vr is not None else "N/A"
                tr_str = f"{tr:.4f}" if tr is not None else "N/A"
                mem_str = f"{mem:.2f}" if mem is not None else "N/A"
                f.write(f"{k},{vr_str},{tr_str},{tt:.2f},{mem_str}\n")
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
