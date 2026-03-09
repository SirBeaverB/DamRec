#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练 + T2T 流式测试脚本 - ml-1m 80/20 划分

流程：前 80% 数据预训练，后 20% 做 T2T 流式测试。
- 预训练模型可通过 run_non_streaming_experiments_1m.py 等得到
- 支持 --ckp 指定已有 checkpoint 路径，跳过预训练直接做 T2T

用法:
  # 1. 准备 80/20 划分数据（首次运行）
  python scripts/prepare_ml1m_80_20_split.py

  # 2. 预训练（在 80% 数据上）
  python scripts/run_pretrain_t2t_1m.py --mode pretrain
  python scripts/run_pretrain_t2t_1m.py --mode pretrain --model GDN --ckp_dir saved/pretrain_t2t_1m

  # 3. 从 checkpoint 做 T2T 测试（在 20% 数据上）
  python scripts/run_pretrain_t2t_1m.py --mode t2t --ckp saved/pretrain_t2t_1m/GDN-xxx.pth
  python scripts/run_pretrain_t2t_1m.py --mode t2t --ckp saved/non_streaming_1m_L128
  # 嫁接：用 GDN 的离线权重热启动任意变体（MoRec/NestRec/DamRec/FroRec），M/V 从 0 吸收流式
  python scripts/run_pretrain_t2t_1m.py --mode t2t --ckp saved/GDN-xxx.pth --model DamRec

  # 4. 完整流程：预训练后自动 T2T
  python scripts/run_pretrain_t2t_1m.py --mode full
"""

import argparse
import glob
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

# 模型配置（与 run_non_streaming_experiments 对齐）
MODEL_CONFIGS = {
    "GDN": ("GDN", "recbole/properties/quick_start_config/sequential_GDN.yaml"),
    "Mo": ("MoRec", "recbole/properties/quick_start_config/sequential_MoRec.yaml"),
    "Nest": ("NestRec", "recbole/properties/quick_start_config/sequential_NestRec.yaml"),
    "Adam": ("DamRec", "recbole/properties/quick_start_config/sequential_DamRec.yaml"),
    "Fro": ("FroRec", "recbole/properties/quick_start_config/sequential_FroRec.yaml"),
}

PRETRAIN_OVERRIDES = {
    "dataset": "ml-1m-pretrain",
    "MAX_ITEM_LIST_LENGTH": 128,
    "epochs": 150,
    "worker": 4,
    "streaming_mode": False,
}

# 流式阶段 lr 建议比预训练低 5~10 倍：数据逐条来，易被噪声带偏，步子迈小更稳
T2T_OVERRIDES = {
    "dataset": "ml-1m-t2t",
    "streaming_t2t": True,
    "streaming_mode": True,
    "streaming_test_ratio": 0.1,
    "MAX_ITEM_LIST_LENGTH": 128,
    "epochs": 1,
    "worker": 4,
    "shuffle": False,
    "enable_amp": False,
    "enable_scaler": False,
    "use_compile": False,
    "learning_rate": 0.0001,  # 预训练常用 0.001；流式微调降 10 倍，抗噪防崩塌
}


def _ensure_split():
    """确保 80/20 划分数据已准备"""
    proj = os.path.dirname(os.path.dirname(__file__))
    t2t_inter = os.path.join(proj, "dataset", "ml-1m-t2t", "ml-1m-t2t.inter")
    if not os.path.isfile(t2t_inter):
        print("未找到 ml-1m-t2t 数据集，正在运行 prepare_ml1m_80_20_split.py ...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(_script_dir, "prepare_ml1m_80_20_split.py")], check=True)


def run_pretrain(model_name, config_file, ckp_dir, show_progress=True, max_seq_len=None):
    """在 ml-1m-pretrain 上预训练，保存到 ckp_dir"""
    overrides = dict(PRETRAIN_OVERRIDES)
    if max_seq_len is not None:
        overrides["MAX_ITEM_LIST_LENGTH"] = max_seq_len
    config_dict = {
        "show_progress": show_progress,
        "gpu_id": "0",
        "checkpoint_dir": ckp_dir,
        **overrides,
    }
    config = Config(
        model=model_name,
        dataset="ml-1m-pretrain",
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

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(train_data, valid_data, saved=True, show_progress=show_progress)

    return trainer.saved_model_file


def run_t2t_from_ckp(ckp_path, show_progress=True, t2t_model=None, t2t_lr=None, max_seq_len=None):
    """从 checkpoint 加载，在 ml-1m-t2t 上运行 T2T。
    t2t_model: 若指定且与 checkpoint 中的 model 不同，则用 strict=False 做「嫁接」：
       用 GDN 的离线权重热启动 MoRec/NestRec/DamRec/FroRec，M/V 从 0 吸收流式数据。
    """
    _ensure_split()

    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
    ckp_config = checkpoint["config"]
    ckp_model = ckp_config["model"]

    # 解析 t2t_model：Adam->DamRec 等
    key2name = {k: m for k, (m, _) in MODEL_CONFIGS.items()}
    model_name = key2name.get(t2t_model, t2t_model) if t2t_model else ckp_model
    hot_swap = model_name != ckp_model

    config_file = None
    for k, (m, cf) in MODEL_CONFIGS.items():
        if m == model_name:
            config_file = cf
            break
    if config_file is None:
        cfg_path = f"recbole/properties/quick_start_config/sequential_{model_name}.yaml"
        if os.path.exists(cfg_path):
            config_file = cfg_path
        else:
            config_file = f"recbole/properties/quick_start_config/streaming/sequential_{model_name}_streaming.yaml"

    proj = os.path.dirname(os.path.dirname(__file__))
    ckp_dir = os.path.join(proj, "saved", "pretrain_t2t_1m")

    overrides = dict(T2T_OVERRIDES)
    if t2t_lr is not None:
        overrides["learning_rate"] = t2t_lr
    if max_seq_len is not None:
        overrides["MAX_ITEM_LIST_LENGTH"] = max_seq_len
    config_dict = {
        "show_progress": show_progress,
        "gpu_id": "0",
        "checkpoint_dir": ckp_dir,
        **overrides,
    }
    config = Config(
        model=model_name,
        dataset="ml-1m-t2t",
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

    if hot_swap:
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(
            set_color("嫁接: ", "green")
            + f"loaded {ckp_model} weights into {model_name} (strict=False). "
            + f"M/V start at 0, absorbing stream."
        )
        if missing:
            logger.debug(f"  [{model_name}-only params not in ckp]: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.debug(f"  [{ckp_model}-only params ignored]: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    else:
        model.load_state_dict(checkpoint["state_dict"])
        if "other_parameter" in checkpoint and checkpoint["other_parameter"]:
            model.load_other_parameter(checkpoint["other_parameter"])
        logger.info(set_color("Loaded checkpoint from", "green") + f" {ckp_path}")

    peak_mem_gb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = get_trainer(config["MODEL_TYPE"], config["model"], config)(config, model)
    t0 = time.perf_counter()
    best_valid_score, result = trainer.fit(
        train_data, valid_data, saved=False, show_progress=show_progress
    )
    t_sec = time.perf_counter() - t0

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(set_color("T2T result (last 20%)", "yellow") + f": {result}")
    return result, t_sec, peak_mem_gb


def main():
    parser = argparse.ArgumentParser(description="预训练 + T2T 流式测试 (ml-1m 80/20)")
    parser.add_argument("--mode", choices=["pretrain", "t2t", "full"], default="full",
                        help="pretrain=仅预训练, t2t=仅T2T(需--ckp), full=预训练后T2T")
    parser.add_argument("--ckp", type=str, default=None,
                        help="checkpoint 路径，用于 t2t 模式；可为 experiment 保存的 ckp")
    parser.add_argument("--ckp_dir", type=str, default=None,
                        help="预训练 checkpoint 保存目录，默认 saved/pretrain_t2t_1m")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="t2t 模式: 指定运行模型(如 DamRec)；可与 --ckp 中模型不同，实现 GDN→DamRec 嫁接")
    parser.add_argument("--t2t_lr", type=float, default=None,
                        help="t2t 流式阶段学习率，默认 0.0001(比预训练 0.001 低 10 倍抗噪)；可调")
    parser.add_argument("--max_seq_len", "-L", type=int, default=None,
                        help="序列长度 L，默认 128")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    proj = os.path.dirname(os.path.dirname(__file__))
    ckp_dir = args.ckp_dir or os.path.join(proj, "saved", "pretrain_t2t_1m")
    show_progress = not args.no_progress

    if args.mode == "t2t":
        if not args.ckp:
            print("t2t 模式需要 --ckp 指定 checkpoint 路径或目录")
            sys.exit(1)
        ckp_path = args.ckp
        if os.path.isdir(ckp_path):
            # 目录：按 --model 或任意 .pth 查找
            pattern = f"{args.model}-*.pth" if args.model else "*.pth"
            candidates = sorted(glob.glob(os.path.join(ckp_path, pattern)))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(ckp_path, "*.pth")))
            if not candidates:
                print(f"目录 {ckp_path} 下未找到 .pth 文件")
                sys.exit(1)
            ckp_path = candidates[-1]
            print(f"使用 checkpoint: {ckp_path}")
        if not os.path.isfile(ckp_path):
            print(f"checkpoint 不存在: {ckp_path}")
            sys.exit(1)
        result, t_sec, mem = run_t2t_from_ckp(
            ckp_path, show_progress=show_progress, t2t_model=args.model, t2t_lr=args.t2t_lr,
            max_seq_len=args.max_seq_len,
        )
        print(f"\nT2T 完成: {result}, 耗时 {t_sec:.1f}s, 显存 {mem:.2f}GB" if mem else f"\nT2T 完成: {result}, 耗时 {t_sec:.1f}s")
        return

    if args.mode in ["pretrain", "full"]:
        _ensure_split()
        os.makedirs(ckp_dir, exist_ok=True)

        models = [args.model] if args.model else ["GDN", "Mo", "Nest", "Adam", "Fro"]
        saved_ckps = {}

        for model_key in models:
            if model_key not in MODEL_CONFIGS:
                print(f"[WARN] 未知模型 {model_key}，跳过")
                continue
            model_name, config_file = MODEL_CONFIGS[model_key]
            print(f"\n{'='*60}\n预训练 {model_key} ({model_name}) ...\n{'='*60}")
            try:
                ckp_file = run_pretrain(model_name, config_file, ckp_dir, show_progress, max_seq_len=args.max_seq_len)
                saved_ckps[model_key] = ckp_file
                print(f"已保存: {ckp_file}")
            except Exception as e:
                print(f"[ERROR] {model_key} 预训练失败: {e}")
                import traceback
                traceback.print_exc()

        if args.mode == "pretrain":
            print(f"\n预训练完成，checkpoint 在 {ckp_dir}")
            return

        # full: 对每个预训练好的模型做 T2T
        output_dir = os.path.join(proj, "experiment_results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"pretrain_t2t_1m_{timestamp}.txt")

        lines = ["预训练(80%) + T2T(20%) 实验 - ml-1m", f"time={timestamp}", ""]
        for model_key, ckp_file in saved_ckps.items():
            if not os.path.isfile(ckp_file):
                continue
            print(f"\n{'='*60}\nT2T {model_key} ...\n{'='*60}")
            try:
                result, t_sec, mem = run_t2t_from_ckp(
                    ckp_file, show_progress=show_progress, t2t_lr=args.t2t_lr, max_seq_len=args.max_seq_len
                )
                r10 = result.get("recall@10", "N/A")
                lines.append(f"{model_key}\trecall@10={r10}\ttime={t_sec:.1f}s\tmem={mem:.2f}GB" if mem else f"{model_key}\trecall@10={r10}\ttime={t_sec:.1f}s")
            except Exception as e:
                print(f"[ERROR] {model_key} T2T 失败: {e}")
                lines.append(f"{model_key}\tERROR: {e}")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
