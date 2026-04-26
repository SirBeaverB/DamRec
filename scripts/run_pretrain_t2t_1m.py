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

  # 4. 完整流程：预训练后自动 T2T（默认尝试导出 user_states.pt；GDN 系支持，GRU4Rec 无 forward_with_streaming 会自动跳过）
  python scripts/run_pretrain_t2t_1m.py --mode full
  python scripts/run_pretrain_t2t_1m.py --mode full --model GRU4Rec   # 仅跑 GRU4Rec：80% pretrain + 20% 流式 T2T
  python scripts/run_pretrain_t2t_1m.py --mode full --no_dump_state  # 禁用状态导出

  # 5. 状态导出：从已有 checkpoint 导出用户状态库 (Redis 模拟)
  python scripts/run_pretrain_t2t_1m.py --mode t2t --ckp saved/gdn.pth --dump_state
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

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
    "FroNoV": ("FroRecNoV", "recbole/properties/quick_start_config/sequential_FroRecNoV.yaml"),
    "FroEps8": ("FroRecEps8", "recbole/properties/quick_start_config/sequential_FroRecEps8.yaml"),
    "FroB999": ("FroRecB999", "recbole/properties/quick_start_config/sequential_FroRecB999.yaml"),
    "FroEta01": ("FroRecEta01", "recbole/properties/quick_start_config/sequential_FroRecEta01.yaml"),
    # 与 run_baseline_experiments_1m 同 yaml；T2T 用 StreamingTestThenTrainTrainer，无跨 batch GRU 隐状态（与 GDN 的 S 不同）
    "GRU4Rec": ("GRU4Rec", "recbole/properties/quick_start_config/baselines/sequential_GRU4Rec_1m.yaml"),
}

PRETRAIN_OVERRIDES = {
    "dataset": "ml-1m-pretrain",
    "MAX_ITEM_LIST_LENGTH": 128,
    "epochs": 150,
    "worker": 4,
    "streaming_mode": False,
    "topk": [10, 20, 50],  # 报多 K，避免单 K cherry-pick；@10 最严，@20/@50 递宽
}

# 流式阶段 lr 建议比预训练低 5~10 倍：数据逐条来，易被噪声带偏，步子迈小更稳
# streaming_pretrain_dataset: 用预训练集历史初始化 user_history，否则 T2T 仅含 20% 数据，序列过短 recall 异常低
T2T_OVERRIDES = {
    "dataset": "ml-1m-t2t",
    "streaming_t2t": True,
    "streaming_mode": True,  # 已修复滑动窗口死锁：有老状态时只吸收最后 1 token，无状态时吸收全量
    "streaming_test_ratio": 0.1,
    "streaming_pretrain_dataset": "ml-1m-pretrain",
    "MAX_ITEM_LIST_LENGTH": 128,
    "epochs": 1,
    "worker": 4,
    "shuffle": False,
    "enable_amp": False,
    "enable_scaler": False,
    "use_compile": False,
    "learning_rate": 0.0001,  # 预训练常用 0.001；流式微调降 10 倍，抗噪防崩塌
    "topk": [10, 20, 50],  # 流式评估也报多 K
}


def _ensure_split():
    """确保 80/20 划分数据已准备"""
    proj = os.path.dirname(os.path.dirname(__file__))
    t2t_inter = os.path.join(proj, "dataset", "ml-1m-t2t", "ml-1m-t2t.inter")
    if not os.path.isfile(t2t_inter):
        print("未找到 ml-1m-t2t 数据集，正在运行 prepare_ml1m_80_20_split.py ...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(_script_dir, "prepare_ml1m_80_20_split.py")], check=True)


def run_pretrain(model_name, config_file, ckp_dir, show_progress=True, max_seq_len=None, seed=None):
    """在 ml-1m-pretrain 上预训练，保存到 ckp_dir。

    Returns:
        tuple: (saved_model_path, pretrain_valid_dict, pretrain_test_dict)
        后两者为 **pretrain 子集**上 RecBole 常规划分的 best valid 与 test 全排序指标
        （非流式 sequential），与后段 T2T(20%) 的 ``result`` 不是同一测试分布。
    """
    overrides = dict(PRETRAIN_OVERRIDES)
    if max_seq_len is not None:
        overrides["MAX_ITEM_LIST_LENGTH"] = max_seq_len
    if seed is not None:
        overrides["seed"] = int(seed)
    config_dict = {
        "show_progress": show_progress,
        "gpu_id": "0",
        "checkpoint_dir": ckp_dir,
        **overrides,
    }
    config = Config(
        model=model_name,
        dataset=overrides.get("dataset", "ml-1m-pretrain"),
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
    _bs, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=show_progress
    )
    pretrain_valid = (
        {k: float(v) for k, v in best_valid_result.items()} if best_valid_result else {}
    )
    pretrain_test = {}
    if test_data is not None:
        test_raw = trainer.evaluate(
            test_data, load_best_model=True, show_progress=show_progress
        )
        if test_raw is not None:
            pretrain_test = {k: float(v) for k, v in test_raw.items()}

    return trainer.saved_model_file, pretrain_valid, pretrain_test


def run_state_dump(ckp_path, save_path=None, model_name=None, config_file=None, max_seq_len=None, show_progress=True, seed=None):
    """工业标准：导出预训练结束时的用户状态 (S, M, V) 到 user_states_{model_name}.pt。
    直接按 (user_id, timestamp) 排序遍历底层数据，保证全员覆盖、绝对时序，避免 DataLoader 乱序/漏人。
    """
    import pandas as pd

    _ensure_split()
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)
    ckp_model = checkpoint["config"]["model"]
    model_name = model_name or ckp_model

    if config_file is None:
        for k, (m, cf) in MODEL_CONFIGS.items():
            if m == model_name:
                config_file = cf
                break
        if config_file is None:
            config_file = f"recbole/properties/quick_start_config/sequential_{model_name}.yaml"

    proj = os.path.dirname(os.path.dirname(__file__))
    ckp_dir = os.path.dirname(ckp_path) if os.path.isfile(ckp_path) else ckp_path
    save_path = save_path or os.path.join(ckp_dir, f"user_states_{model_name}.pt")

    overrides = {
        "dataset": PRETRAIN_OVERRIDES.get("dataset", "ml-1m-pretrain"),
        "streaming_t2t": False,  # 避免 StreamingSequentialDataset，直接用底层 DataFrame
        "streaming_mode": True,  # 模型用 forward_with_streaming 吸收历史
        "MAX_ITEM_LIST_LENGTH": max_seq_len or 128,
        "show_progress": show_progress,
    }
    if seed is not None:
        overrides["seed"] = int(seed)
    config = Config(
        model=model_name,
        dataset=overrides["dataset"],
        config_file_list=[config_file],
        config_dict={"show_progress": show_progress, **overrides},
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    from logging import getLogger
    logger = getLogger()
    logger.info(set_color("[State Dump]: ", "pink") + f"导出用户状态，ckp={ckp_path} -> {save_path}")

    dataset_obj = create_dataset(config)
    # 不调用 build/data_preparation，直接访问底层 DataFrame，保证时序与全员覆盖
    inter_feat = dataset_obj.inter_feat
    if hasattr(inter_feat, "interaction"):
        df = pd.DataFrame({k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)).flatten()
                           for k, v in inter_feat.interaction.items()})
    else:
        df = inter_feat

    uid_field = dataset_obj.uid_field
    iid_field = dataset_obj.iid_field
    time_field = config.final_config_dict.get("TIME_FIELD", "timestamp")
    if time_field not in df.columns:
        time_field = next((c for c in df.columns if "time" in c.lower()), df.columns[-1])

    df = df.sort_values(by=[uid_field, time_field])
    user_tensor = torch.tensor(df[uid_field].values, dtype=torch.long)
    item_tensor = torch.tensor(df[iid_field].values, dtype=torch.long)
    all_users = torch.unique(user_tensor)
    max_len = max_seq_len or config["MAX_ITEM_LIST_LENGTH"]
    device = config["device"]

    model = get_model(config["model"])(config, dataset_obj).to(config["device"])
    if model_name != ckp_model:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"嫁接加载: {ckp_model} -> {model_name} (strict=False)")
    else:
        model.load_state_dict(checkpoint["state_dict"])
    if "other_parameter" in checkpoint and checkpoint["other_parameter"]:
        model.load_other_parameter(checkpoint["other_parameter"])

    if not callable(getattr(model, "forward_with_streaming", None)):
        logger.warning(
            set_color("[State Dump]: ", "yellow")
            + f"{model_name} 无 forward_with_streaming，跳过 user_states 导出；T2T 仍按时间线跑（无跨 batch 隐式状态库）。"
        )
        return None

    model.eval()
    model._streaming_state = {}

    n_skip = sum(1 for u in all_users if u.item() == 0)
    logger.info(set_color("[State Dump]: ", "yellow") + f"按绝对时序为 {len(all_users)-n_skip} 用户灌注历史状态...")

    with torch.no_grad():
        for uid in tqdm(all_users, desc="State Dump", disable=not show_progress):
            uid_scalar = uid.item()
            if uid_scalar == 0:
                continue
            mask = (user_tensor == uid_scalar)
            user_history = item_tensor[mask]
            hist_len = len(user_history)
            if hist_len == 0:
                continue
            if hist_len > max_len:
                user_history = user_history[-max_len:]
                seq_len = max_len
            else:
                seq_len = hist_len

            batch_item_seq = torch.zeros((1, max_len), dtype=torch.long, device=device)
            batch_item_seq[0, :seq_len] = user_history.to(device)
            batch_item_seq_len = torch.tensor([seq_len], dtype=torch.long, device=device)
            batch_user_id = torch.tensor([uid_scalar], dtype=torch.long, device=device)

            _ = model.forward_with_streaming(batch_item_seq, batch_item_seq_len, batch_user_id, update_state=True)

    torch.save(model._streaming_state, save_path)
    n_stored = len(model._streaming_state)
    logger.info(set_color("[State Dump]: ", "green") + f"已导出 {n_stored} 用户状态 -> {save_path}")
    if n_stored >= 5500:
        logger.info(set_color("[State Dump]: ", "green") + "状态库完整 (>=5500 用户)")
    else:
        logger.warning(set_color("[State Dump]: ", "red") + f"状态库残缺！仅 {n_stored} 用户，预期 ~6040")
    return save_path


def run_t2t_from_ckp(ckp_path, show_progress=True, t2t_model=None, t2t_lr=None, max_seq_len=None, zero_shot=False, user_states_path=None, seed=None, reset_mv=False):
    """从 checkpoint 加载，在 T2T_OVERRIDES['dataset']（默认 ml-1m-t2t）上运行流式 T2T。
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
    if zero_shot:
        overrides["streaming_zero_shot"] = True
        overrides["learning_rate"] = 0.0
    if seed is not None:
        overrides["seed"] = int(seed)
    config_dict = {
        "show_progress": show_progress,
        "gpu_id": "0",
        "checkpoint_dir": ckp_dir,
        "config_file_list": [config_file],  # 供 data_preparation 词表对齐时构建 pretrain config
        **overrides,
    }
    # 必须与 overrides['dataset'] 一致；曾硬编码 ml-1m-t2t 导致 Yelp 等脚本改了 T2T_OVERRIDES 仍加载错数据、词表对齐全 OOV。
    t2t_dataset = overrides.get("dataset", "ml-1m-t2t")
    config = Config(
        model=model_name,
        dataset=t2t_dataset,
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

    # 工业标准：注入用户状态库 (Redis 模拟)，老用户不再从零开始
    # 专属状态库：user_states_{model_name}.pt，嫁接模型也有对应 state 文件时加载
    ckp_dir = os.path.dirname(ckp_path) if os.path.isfile(ckp_path) else ckp_path
    states_file = user_states_path or os.path.join(ckp_dir, f"user_states_{model_name}.pt")
    if os.path.isfile(states_file) and hasattr(model, "_streaming_state"):
        loaded_states = torch.load(states_file, map_location=config["device"], weights_only=False)
        if reset_mv and model_name == "DamRec":
            n_reset = 0
            for uid, state in loaded_states.items():
                # DamRec state = (S_tuple, Vr_tuple, Vk_tuple, cum, ...)
                # GDN/FroRec states differ — only reset DamRec's Adam moments
                if not (isinstance(state[1], (list, tuple)) and len(state[1]) > 0 and isinstance(state[1][0], torch.Tensor)):
                    continue
                zero_vr = tuple(torch.zeros_like(v) for v in state[1])
                zero_vk = tuple(torch.zeros_like(v) for v in state[2])
                loaded_states[uid] = (state[0], zero_vr, zero_vk) + state[3:]
                n_reset += 1
            logger.info(set_color("reset_mv=True:", "yellow") + f" V_r/V_k zeroed for {n_reset} DamRec users, S preserved")
        elif reset_mv:
            logger.info(set_color("reset_mv=True:", "yellow") + f" skipped for {model_name} (not DamRec)")
        model._streaming_state = loaded_states
        logger.info(set_color("Loaded user states from", "green") + f" {states_file} ({len(model._streaming_state)} users)")

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
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-Shot 体检：仅评估不训练，排查预训练权重是否加载成功")
    parser.add_argument("--dump_state", action="store_true",
                        help="full 模式：预训练后导出 user_states.pt，T2T 时注入 (工业标准)")
    parser.add_argument("--no_dump_state", action="store_true",
                        help="禁用状态导出 (与 --dump_state 二选一)")
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
        if args.dump_state:
            model_name = MODEL_CONFIGS.get(args.model, (args.model, None))[0] if args.model else torch.load(ckp_path, map_location="cpu", weights_only=False)["config"]["model"]
            ckp_dir = os.path.dirname(ckp_path)
            states_file = os.path.join(ckp_dir, f"user_states_{model_name}.pt")
            if not os.path.isfile(states_file):
                print(f"\n[State Dump] {model_name} 导出用户状态 -> {states_file}")
                run_state_dump(ckp_path, save_path=None, model_name=model_name, max_seq_len=args.max_seq_len, show_progress=show_progress)
        result, t_sec, mem = run_t2t_from_ckp(
            ckp_path, show_progress=show_progress, t2t_model=args.model, t2t_lr=args.t2t_lr,
            max_seq_len=args.max_seq_len, zero_shot=args.zero_shot,
        )
        mode_str = "Zero-Shot 体检" if args.zero_shot else "T2T"
        print(f"\n{mode_str} 完成: {result}, 耗时 {t_sec:.1f}s, 显存 {mem:.2f}GB" if mem else f"\n{mode_str} 完成: {result}, 耗时 {t_sec:.1f}s")
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
                ckp_file, pv, pt = run_pretrain(
                    model_name, config_file, ckp_dir, show_progress, max_seq_len=args.max_seq_len
                )
                saved_ckps[model_key] = (ckp_file, pv, pt)
                print(f"已保存: {ckp_file}")
                print(
                    f"  [pretrain 子集] valid recall@10={pv.get('recall@10', 'N/A')} | "
                    f"test recall@10={pt.get('recall@10', 'N/A')}"
                )
            except Exception as e:
                print(f"[ERROR] {model_key} 预训练失败: {e}")
                import traceback
                traceback.print_exc()

        if args.mode == "pretrain":
            print(f"\n预训练完成，checkpoint 在 {ckp_dir}")
            return

        # full: 状态导出 + T2T
        output_dir = os.path.join(proj, "experiment_results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"pretrain_t2t_1m_{timestamp}.txt")

        lines = [
            "预训练(80%) + T2T(20%) 实验 - ml-1m",
            f"time={timestamp}",
            "",
            "[说明] pretrain_valid / pretrain_test 为 pretrain 数据上**常规 sequential 离线**指标；",
            "       下列 streaming T2T 行为后 20% 流式指标。",
            "",
        ]
        for model_key, pack in saved_ckps.items():
            if not isinstance(pack, tuple) or len(pack) != 3:
                ckp_file = pack if isinstance(pack, str) else None
                pv, pt = {}, {}
            else:
                ckp_file, pv, pt = pack
            if not ckp_file or not os.path.isfile(ckp_file):
                continue
            # 工业标准：导出用户状态库 (user_states.pt)，T2T 时自动加载 (full 模式默认开启)
            do_dump = (args.dump_state or (not args.no_dump_state)) and args.mode == "full"
            if do_dump:
                print(f"\n{'='*60}\nState Dump {model_key} ...\n{'='*60}")
                try:
                    run_state_dump(ckp_file, model_name=MODEL_CONFIGS[model_key][0],
                                  config_file=MODEL_CONFIGS[model_key][1], max_seq_len=args.max_seq_len,
                                  show_progress=show_progress)
                except Exception as e:
                    print(f"[WARN] {model_key} 状态导出失败: {e}")

            print(f"\n{'='*60}\nT2T {model_key} ...\n{'='*60}")
            try:
                result, t_sec, mem = run_t2t_from_ckp(
                    ckp_file, show_progress=show_progress, t2t_lr=args.t2t_lr, max_seq_len=args.max_seq_len,
                    zero_shot=args.zero_shot,
                )
                r10 = result.get("recall@10", "N/A")
                pv10 = pv.get("recall@10", "N/A") if isinstance(pv, dict) else "N/A"
                pt10 = pt.get("recall@10", "N/A") if isinstance(pt, dict) else "N/A"
                line = (
                    f"{model_key}\tpretrain_valid@10={pv10}\tpretrain_test@10={pt10}\t"
                    f"streaming@10={r10}\ttime={t_sec:.1f}s"
                )
                if mem:
                    line += f"\tmem={mem:.2f}GB"
                lines.append(line)
            except Exception as e:
                print(f"[ERROR] {model_key} T2T 失败: {e}")
                lines.append(f"{model_key}\tERROR: {e}")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
