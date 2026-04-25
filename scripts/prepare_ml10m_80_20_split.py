#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 ML-10M 的 80/20 时间划分数据集。

前置：
  python scripts/download_ml10m.py
  产出: dataset/ml-10m/ml-10m.inter

规则：
  - 可选 k-core 过滤（--min_user_inter / --min_item_inter，迭代至稳定）
  - 按 timestamp 全局排序后切 80/20
  - 前 80% 作 ml-10m-pretrain-{tag}，后 20% 作 ml-10m-t2t-{tag}
  - vocab 对齐占位行（rating=0）保证两子集 user/item ID 一致

推荐正式实验（稠密子集）：
  --max_users 20000 --min_user_inter 20 --min_pretrain_inter 20 --filter_cold_items
  预期：~20k users，~10k items，avg pretrain ~80-100 inter/user，dump ~1.5h/model

用法：
  # 推荐：20k 稠密用户（avg 100+ inter/user，item 10k）
  python scripts/prepare_ml10m_80_20_split.py \
      --data_tag dense_u20k \
      --max_users 20000 \
      --min_user_inter 20 \
      --min_item_inter 5 \
      --min_pretrain_inter 20 \
      --filter_cold_items \
      --seed 42

  # 小规模 smoke test
  python scripts/prepare_ml10m_80_20_split.py \
      --data_tag u2k_test \
      --max_users 2000 \
      --min_user_inter 10 \
      --seed 42
"""

import argparse
import os
import random
import shutil

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(_script_dir)
DATA_ROOT = os.path.join(PROJ_ROOT, "dataset")
ML10M_DIR = os.path.join(DATA_ROOT, "ml-10m")
PRETRAIN_RATIO = 0.8  # default; overridden by --pretrain_ratio CLI arg


def _sanitize_data_tag(s):
    import re
    if s is None or not str(s).strip():
        return None
    t = str(s).strip()
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$", t):
        raise SystemExit(
            "--data_tag 仅允许 [a-zA-Z0-9] 开头，仅含字母数字、下划线、连字符，长度 1–64"
        )
    return t


def main():
    parser = argparse.ArgumentParser(description="ML-10M 80/20 时间切分，可选稠密用户过滤")
    parser.add_argument("--data_tag", type=str, default=None,
                        help="必须指定，如 dense_u20k。写入独立目录避免覆盖。")
    parser.add_argument("--max_users", type=int, default=None,
                        help="随机保留至多这么多用户（及其全部交互）。")
    parser.add_argument("--min_user_inter", type=int, default=None,
                        help="k-core：保留至少 k 条交互的用户（与 min_item_inter 迭代）。")
    parser.add_argument("--min_item_inter", type=int, default=None,
                        help="k-core：保留至少 k 条交互的 item。")
    parser.add_argument("--min_pretrain_inter", type=int, default=None,
                        help="切分后过滤：只保留 pretrain 段有至少 k 条真实交互的 user。")
    parser.add_argument("--filter_cold_items", action="store_true", default=False,
                        help="切分后过滤：从 t2t 移除 pretrain 无真实交互的 item（cold item）。")
    parser.add_argument("--pretrain_ratio", type=float, default=PRETRAIN_RATIO,
                        help="pretrain 占比，默认 0.8；设 0.4 得到 40/60 切分。范围 (0,1)。")
    parser.add_argument("--seed", type=int, default=42, help="用户抽样随机种子")
    args = parser.parse_args()

    tag = _sanitize_data_tag(args.data_tag)
    if tag is None:
        raise SystemExit(
            "ML-10M 切分必须指定 --data_tag（如 dense_u20k），避免误写默认目录。"
        )

    inter_path = os.path.join(ML10M_DIR, "ml-10m.inter")
    if not os.path.isfile(inter_path):
        raise SystemExit(
            f"未找到 {inter_path}。请先运行：\n  python scripts/download_ml10m.py"
        )

    ptn = f"ml-10m-pretrain-{tag}"
    t2n = f"ml-10m-t2t-{tag}"
    pretrain_dir = os.path.join(DATA_ROOT, ptn)
    t2t_dir = os.path.join(DATA_ROOT, t2n)

    print(f"[1/6] 读取 {inter_path}")
    with open(inter_path, "r", encoding="utf-8") as f:
        header = f.readline()
        cols = header.strip().split("\t")
        rows = []
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != len(cols):
                continue
            rows.append(parts)

    # 列索引
    def _find(key):
        for i, c in enumerate(cols):
            if c.split(":")[0].strip().lower() == key:
                return i
        return None
    uid_idx = _find("user_id")
    iid_idx = _find("item_id")
    ts_idx = _find("timestamp")
    rating_idx = _find("rating")
    if uid_idx is None or iid_idx is None or ts_idx is None:
        raise SystemExit(f"header 列缺失: {cols}")

    print(f"      原始交互: {len(rows):,} 条，{len({r[uid_idx] for r in rows}):,} 用户，"
          f"{len({r[iid_idx] for r in rows}):,} items")

    # k-core
    ku, ki = args.min_user_inter, args.min_item_inter
    if ku or ki:
        print(f"[2/6] k-core 过滤 (min_user={ku}, min_item={ki})...")
        import collections as _col
        prev_len = -1
        iteration = 0
        while len(rows) != prev_len:
            prev_len = len(rows)
            iteration += 1
            if ku:
                u_cnt = _col.Counter(r[uid_idx] for r in rows)
                keep_u = {u for u, c in u_cnt.items() if c >= ku}
                rows = [r for r in rows if r[uid_idx] in keep_u]
            if ki:
                i_cnt = _col.Counter(r[iid_idx] for r in rows)
                keep_i = {i for i, c in i_cnt.items() if c >= ki}
                rows = [r for r in rows if r[iid_idx] in keep_i]
        print(f"      {iteration} 轮后: {len(rows):,} 条, "
              f"{len({r[uid_idx] for r in rows}):,} 用户, "
              f"{len({r[iid_idx] for r in rows}):,} items")
    else:
        print("[2/6] 跳过 k-core（未设 --min_user_inter / --min_item_inter）")

    # 用户采样
    u_list = sorted({r[uid_idx] for r in rows})
    n_u = len(u_list)
    if args.max_users is not None and args.max_users < n_u:
        print(f"[3/6] 采样用户 {args.max_users} / {n_u} (seed={args.seed})...")
        rng = random.Random(args.seed)
        keep = set(rng.sample(u_list, args.max_users))
        rows = [r for r in rows if r[uid_idx] in keep]
        print(f"      保留 {len(keep):,} 用户, {len(rows):,} 条交互")
    else:
        print(f"[3/6] 保留全部 {n_u:,} 用户")

    # 时间排序 + 切分
    pretrain_ratio = args.pretrain_ratio
    if not (0.0 < pretrain_ratio < 1.0):
        raise SystemExit(f"--pretrain_ratio 必须在 (0,1) 范围内，当前值: {pretrain_ratio}")
    pct = int(pretrain_ratio * 100)
    t2t_pct = 100 - pct
    print(f"[4/6] 按 timestamp 排序并切分 {pct}/{t2t_pct}...")
    def _ts(r):
        try:
            return float(r[ts_idx])
        except ValueError:
            return 0.0
    rows.sort(key=_ts)
    n = len(rows)
    split_idx = int(n * pretrain_ratio)
    pretrain_rows = rows[:split_idx]
    t2t_rows = rows[split_idx:]
    print(f"      pretrain: {len(pretrain_rows):,} 条，t2t: {len(t2t_rows):,} 条")

    # 切分后过滤 1：min_pretrain_inter
    if args.min_pretrain_inter:
        import collections as _col
        kp = args.min_pretrain_inter
        pre_u_cnt = _col.Counter(r[uid_idx] for r in pretrain_rows)
        keep_u = {u for u, c in pre_u_cnt.items() if c >= kp}
        n_drop = len(pre_u_cnt) - len(keep_u)
        pretrain_rows = [r for r in pretrain_rows if r[uid_idx] in keep_u]
        t2t_rows = [r for r in t2t_rows if r[uid_idx] in keep_u]
        print(f"[min_pretrain_inter={kp}] 移除 {n_drop} 个 pretrain 段不足 user "
              f"-> pretrain {len(pretrain_rows):,}, t2t {len(t2t_rows):,}")

    # 切分后过滤 2：filter_cold_items
    if args.filter_cold_items:
        warm_items = {r[iid_idx] for r in pretrain_rows}
        n_before = len(t2t_rows)
        t2t_rows = [r for r in t2t_rows if r[iid_idx] in warm_items]
        n_dropped = n_before - len(t2t_rows)
        print(f"[filter_cold_items] 移除 {n_dropped:,} 条 cold item 交互 "
              f"({n_dropped/max(1,n_before)*100:.1f}%) -> t2t {len(t2t_rows):,}")

    if not pretrain_rows:
        raise SystemExit("过滤后 pretrain 为空，检查参数。")
    if not t2t_rows:
        raise SystemExit("过滤后 t2t 为空，检查参数。")

    # 统计
    users_pretrain = {r[uid_idx] for r in pretrain_rows}
    users_t2t = {r[uid_idx] for r in t2t_rows}
    items_pretrain = {r[iid_idx] for r in pretrain_rows}
    items_t2t = {r[iid_idx] for r in t2t_rows}
    users_only_t2t = users_t2t - users_pretrain
    users_only_pretrain = users_pretrain - users_t2t
    items_only_t2t = items_t2t - items_pretrain
    items_only_pretrain = items_pretrain - items_t2t

    import statistics
    pre_u_counts = {}
    for r in pretrain_rows:
        pre_u_counts[r[uid_idx]] = pre_u_counts.get(r[uid_idx], 0) + 1
    vals = list(pre_u_counts.values())
    print(f"      pretrain avg inter/user: {statistics.mean(vals):.1f}, "
          f"median: {statistics.median(vals):.0f}")

    # vocab 对齐占位行
    t_min = pretrain_rows[0][ts_idx]
    t_max = t2t_rows[-1][ts_idx]
    sample_uid_p = pretrain_rows[0][uid_idx]
    sample_iid_p = pretrain_rows[0][iid_idx]
    sample_uid_t = t2t_rows[0][uid_idx]
    sample_iid_t = t2t_rows[0][iid_idx]

    def _make_row(uid, iid, rating_val, ts_val):
        r = ["0"] * len(cols)
        r[uid_idx] = uid
        r[iid_idx] = iid
        r[ts_idx] = ts_val
        if rating_idx is not None:
            r[rating_idx] = rating_val
        return r

    dummy_pretrain = []
    for u in users_only_t2t:
        dummy_pretrain.append(_make_row(u, sample_iid_p, "0", t_min))
    for i in items_only_t2t:
        dummy_pretrain.append(_make_row(sample_uid_p, i, "0", t_min))

    dummy_t2t = []
    for u in users_only_pretrain:
        dummy_t2t.append(_make_row(u, sample_iid_t, "0", t_max))
    for i in items_only_pretrain:
        dummy_t2t.append(_make_row(sample_uid_t, i, "0", t_max))

    pretrain_all = pretrain_rows + dummy_pretrain
    t2t_all = t2t_rows + dummy_t2t

    # 写出
    print(f"[5/6] 写出文件...")
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(t2t_dir, exist_ok=True)

    def _write(path, rows_to_write):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for r in rows_to_write:
                f.write("\t".join(r) + "\n")

    pretrain_file = f"{ptn}.inter"
    t2t_file = f"{t2n}.inter"
    _write(os.path.join(pretrain_dir, pretrain_file), pretrain_all)
    _write(os.path.join(t2t_dir, t2t_file), t2t_all)

    print(f"[6/6] 完成")
    print(f"  {pretrain_dir}/{pretrain_file} ({len(pretrain_all):,} 行)")
    print(f"  {t2t_dir}/{t2t_file} ({len(t2t_all):,} 行)")
    print(f"  pretrain: {len(pretrain_rows):,} 真实 + {len(dummy_pretrain)} 占位")
    print(f"  t2t:      {len(t2t_rows):,} 真实 + {len(dummy_t2t)} 占位")
    print(f"  users: pretrain {len(users_pretrain)}, t2t {len(users_t2t)}, "
          f"only-pretrain {len(users_only_pretrain)}, only-t2t {len(users_only_t2t)}")
    print(f"  items: pretrain {len(items_pretrain)}, t2t {len(items_t2t)}, "
          f"only-pretrain {len(items_only_pretrain)}, only-t2t {len(items_only_t2t)}")
    print(f"\n下一步：")
    print(f"  python scripts/run_per_model_pretrain_t2t_ml10m.py \\")
    print(f"      --data_tag {tag} \\")
    print(f"      --models GDN,Adam,Fro,GRU4Rec \\")
    print(f"      --seeds 2020 \\")
    print(f"      --max_seq_len 64 \\")
    print(f"      --n_gpus 2")


if __name__ == "__main__":
    main()
