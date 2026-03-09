#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 ml-1m 的 80/20 时间划分数据集。
前 80% 交互用于预训练，后 20% 用于 T2T 流式测试。
为保证 vocab 一致，会为仅出现在另一子集的 user/item 添加占位交互。

用法: python scripts/prepare_ml1m_80_20_split.py
"""

import os
import shutil

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(_script_dir)
DATA_ROOT = os.path.join(PROJ_ROOT, "dataset")
ML1M_DIR = os.path.join(DATA_ROOT, "ml-1m")
PRETRAIN_DIR = os.path.join(DATA_ROOT, "ml-1m-pretrain")
T2T_DIR = os.path.join(DATA_ROOT, "ml-1m-t2t")
PRETRAIN_RATIO = 0.8


def main():
    inter_path = os.path.join(ML1M_DIR, "ml-1m.inter")
    if not os.path.isfile(inter_path):
        raise FileNotFoundError(f"需要 {inter_path}，请先准备 ml-1m 数据集")

    with open(inter_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0]
    rows = [ln.strip().split("\t") for ln in lines[1:] if ln.strip()]
    # 列: user_id, item_id, rating, timestamp
    uid_idx = 0
    iid_idx = 1
    time_idx = 3

    # 按时间排序
    rows.sort(key=lambda r: float(r[time_idx]))
    n = len(rows)
    split_idx = int(n * PRETRAIN_RATIO)
    pretrain_rows = rows[:split_idx]
    t2t_rows = rows[split_idx:]

    users_pretrain = {r[uid_idx] for r in pretrain_rows}
    users_t2t = {r[uid_idx] for r in t2t_rows}
    items_pretrain = {r[iid_idx] for r in pretrain_rows}
    items_t2t = {r[iid_idx] for r in t2t_rows}

    users_only_t2t = users_t2t - users_pretrain
    items_only_t2t = items_t2t - items_pretrain
    users_only_pretrain = users_pretrain - users_t2t
    items_only_pretrain = items_pretrain - items_t2t

    t_min = pretrain_rows[0][time_idx]
    t_max = t2t_rows[-1][time_idx]
    sample_uid_p = pretrain_rows[0][uid_idx]
    sample_iid_p = pretrain_rows[0][iid_idx]
    sample_uid_t = t2t_rows[0][uid_idx]
    sample_iid_t = t2t_rows[0][iid_idx]

    # 占位：使 pretrain 包含仅出现在 t2t 的 user/item
    dummy_pretrain = []
    for u in users_only_t2t:
        dummy_pretrain.append([u, sample_iid_p, "0", t_min])
    for i in items_only_t2t:
        dummy_pretrain.append([sample_uid_p, i, "0", t_min])

    # 占位：使 t2t 包含仅出现在 pretrain 的 user/item
    dummy_t2t = []
    for u in users_only_pretrain:
        dummy_t2t.append([u, sample_iid_t, "0", t_max])
    for i in items_only_pretrain:
        dummy_t2t.append([sample_uid_t, i, "0", t_max])

    pretrain_all = pretrain_rows + dummy_pretrain
    t2t_all = t2t_rows + dummy_t2t

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    os.makedirs(T2T_DIR, exist_ok=True)

    def write_inter(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for r in rows:
                f.write("\t".join(r) + "\n")

    write_inter(os.path.join(PRETRAIN_DIR, "ml-1m-pretrain.inter"), pretrain_all)
    write_inter(os.path.join(T2T_DIR, "ml-1m-t2t.inter"), t2t_all)

    for suffix in [".user", ".item"]:
        src = os.path.join(ML1M_DIR, "ml-1m" + suffix)
        if os.path.isfile(src):
            for dst_dir, name in [(PRETRAIN_DIR, "ml-1m-pretrain"), (T2T_DIR, "ml-1m-t2t")]:
                dst = os.path.join(dst_dir, name + suffix)
                shutil.copy2(src, dst)

    print(f"已创建 ml-1m-pretrain ({len(pretrain_all)} 行) 和 ml-1m-t2t ({len(t2t_all)} 行)")
    print(f"  pretrain: {split_idx} 真实 + {len(dummy_pretrain)} 占位")
    print(f"  t2t: {n - split_idx} 真实 + {len(dummy_t2t)} 占位")


if __name__ == "__main__":
    main()
