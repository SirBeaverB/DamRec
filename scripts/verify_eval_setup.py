#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证评估设置：全排序 + 无 Target Leakage
用法: python scripts/verify_eval_setup.py [--config_files xxx.yaml]
"""

import argparse
import sys

sys.path.insert(0, ".")

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GDN")
    parser.add_argument("--dataset", default="ml-100k")
    parser.add_argument("--config_files", default=None)
    args = parser.parse_args()

    config_list = args.config_files.strip().split() if args.config_files else []
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_list or ["recbole/properties/quick_start_config/sequential_GDN.yaml"],
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    # 1. 确认 eval_args mode
    mode = config["eval_args"]["mode"]
    print("\n" + "=" * 60)
    print("1. 评估模式 (eval_args.mode)")
    print("=" * 60)
    print(f"   valid: {mode.get('valid', 'N/A')}")
    print(f"   test:  {mode.get('test', 'N/A')}")
    if mode.get("valid") == "full" and mode.get("test") == "full":
        print("   ✓ 全排序 (Full-sort)，候选为全部物品")
    else:
        print("   ⚠ 非 full，可能是采样评估！")

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 2. 确认 DataLoader 类型
    dl_type = type(test_data).__name__
    print(f"\n   Test DataLoader: {dl_type}")
    if "FullSort" in dl_type:
        print("   ✓ FullSortEvalDataLoader，全排序")
    else:
        print("   ⚠ 非 FullSort，可能是负采样评估")

    # 3. 检查 Target Leakage：item_seq 最后一位 != pos_item
    print("\n" + "=" * 60)
    print("2. Target Leakage 检查 (item_seq[-1] vs pos_item)")
    print("=" * 60)

    item_id_list_field = getattr(dataset, "item_id_list_field", "item_id_list")
    item_length_field = getattr(dataset, "item_list_length_field", "item_length")
    iid_field = dataset.iid_field

    n_check = min(20, len(test_data.dataset))
    leakage_count = 0
    for idx in range(n_check):
        inter = test_data.dataset[idx]
        item_seq = inter[item_id_list_field]
        seq_len = inter[item_length_field].item()
        pos_item = inter[iid_field].item()
        last_in_seq = item_seq[seq_len - 1].item() if seq_len > 0 else -1
        if last_in_seq == pos_item:
            leakage_count += 1
            print(f"   [LEAK!] idx={idx}: item_seq[-1]={last_in_seq} == pos_item={pos_item}")

    if leakage_count == 0:
        print(f"   随机抽查 {n_check} 条: item_seq[-1] ≠ pos_item，无泄漏 ✓")
    else:
        print(f"   ⚠ 发现 {leakage_count} 条泄漏！")

    print(f"\n   物品总数 (候选数): {dataset.item_num}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
