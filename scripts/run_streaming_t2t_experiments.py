#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Test-Then-Train 流式实验脚本
- 按全局时间轴单次遍历，先预测再训练，S 永不重置
- 支持: GDN、MoRec、NestRec、DamRec、FroRec

用法:
  python scripts/run_streaming_t2t_experiments.py
  python scripts/run_streaming_t2t_experiments.py --dataset ml-100k
  python scripts/run_streaming_t2t_experiments.py --model GDN
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color

MODEL_T2T_CONFIGS = {
    "GDN": "recbole/properties/quick_start_config/streaming/sequential_GDN_streaming_t2t.yaml",
    "MoRec": "recbole/properties/quick_start_config/streaming/sequential_MoRec_streaming.yaml",
    "NestRec": "recbole/properties/quick_start_config/streaming/sequential_NestRec_streaming.yaml",
    "DamRec": "recbole/properties/quick_start_config/streaming/sequential_DamRec_streaming.yaml",
    "FroRec": "recbole/properties/quick_start_config/streaming/sequential_FroRec_streaming.yaml",
}

# T2T 需要 streaming_t2t: True，部分配置需合并
T2T_OVERRIDES = {
    "streaming_t2t": True,
    "streaming_test_ratio": 0.1,
    "epochs": 1,
    "enable_amp": False,
    "enable_scaler": False,
    "use_compile": False,
}


def run_single_t2t(model_name, dataset="ml-10m", show_progress=True):
    """运行单个模型的 T2T 流式实验"""
    config_file = MODEL_T2T_CONFIGS.get(model_name)
    if not config_file or not os.path.exists(config_file):
        print(f"[WARN] 无 T2T 配置: {model_name}，跳过")
        return None

    print(f"\n{'='*60}")
    print(f"Running T2T: {model_name} on {dataset}")
    print("=" * 60)

    config_dict = {"show_progress": show_progress, "dataset": dataset, **T2T_OVERRIDES}

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

        trainer = get_trainer(config["MODEL_TYPE"], config["model"], config)(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=False, show_progress=show_progress
        )

        test_result = best_valid_result  # T2T 评估在 fit 内完成
        valid_recall = float(best_valid_result.get("recall@10", 0.0))
        test_recall = float(test_result.get("recall@10", 0.0))

        logger.info(set_color("T2T result", "yellow") + f": {test_result}")
        return valid_recall, test_recall

    except Exception as e:
        print(f"[ERROR] {model_name} T2T failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="ml-100k",
                        help="ml-100k 快速验证，ml-1m/ml-10m 完整实验")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="单个模型，如 GDN。默认全部")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    show_progress = not args.no_progress
    models = [args.model] if args.model else list(MODEL_T2T_CONFIGS.keys())

    results = {}
    for model_name in models:
        ret = run_single_t2t(model_name, dataset=args.dataset, show_progress=show_progress)
        results[model_name] = ret

    # 输出
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"streaming_t2t_{args.dataset}_{timestamp}.txt")

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, tuple):
            return f"valid={v[0]:.4f} test={v[1]:.4f}"
        return str(v)

    lines = [
        "Step 4: Test-Then-Train 流式实验",
        f"dataset={args.dataset}, time={timestamp}",
        "",
    ]
    for k, v in results.items():
        lines.append(f"{k}\t{fmt(v)}")

    table = "\n".join(lines)
    print("\n" + "=" * 60)
    print("T2T RESULTS")
    print("=" * 60)
    print(table)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
