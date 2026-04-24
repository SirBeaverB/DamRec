#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 yelp2018 的 80/20 时间划分数据集。

前置：
  需要 RecBole atomic 格式的 yelp2018：
    dataset/yelp2018/yelp2018.inter
  header 预期包含 user_id / item_id / rating / timestamp 四列（列名 RecBole 风格，
  带 :token / :float 后缀或无后缀均可自动识别）。

规则：
  - 按 timestamp 全局排序后切 80/20
  - 前 80% 作 yelp2018-pretrain，后 20% 作 yelp2018-t2t
  - 为保证两子集 vocab 一致，为仅出现在对方的 user/item 补 rating=0 的占位行
  - .user / .item 原子文件若存在则复制两份（与 ml-1m split 行为对齐）

按用户下采样（加速实验，指标勿与全量直接对比）后再 80/20 时，**必须**用 --data_tag 指定标签，
产出写到独立目录 dataset/yelp2018-pretrain-{tag}/ 与 yelp2018-t2t-{tag}/，**不会**覆盖全量
yelp2018-pretrain / yelp2018-t2t。全量 80/20 仍可不写 --data_tag，行为与过去一致（写默认目录）。

用法:
  python scripts/prepare_yelp2018_80_20_split.py
  python scripts/prepare_yelp2018_80_20_split.py --data_tag u5000 --max_users 5000 --seed 42
  python scripts/prepare_yelp2018_80_20_split.py --data_tag s10 --user_sample_ratio 0.1 --seed 42
"""

import argparse
import os
import random
import shutil

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(_script_dir)
DATA_ROOT = os.path.join(PROJ_ROOT, "dataset")
YELP_DIR = os.path.join(DATA_ROOT, "yelp2018")
DEFAULT_PRETRAIN_DIR = os.path.join(DATA_ROOT, "yelp2018-pretrain")
DEFAULT_T2T_DIR = os.path.join(DATA_ROOT, "yelp2018-t2t")
PRETRAIN_RATIO = 0.8


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


def _detect_columns(header):
    """返回 header 各列中 user_id / item_id / rating / timestamp 的下标（容忍 RecBole : 后缀）。"""
    cols = header.strip().split("\t")
    def _find(key):
        for i, c in enumerate(cols):
            name = c.split(":")[0].strip().lower()
            if name == key:
                return i
        return None
    uid_idx = _find("user_id")
    iid_idx = _find("item_id")
    ts_idx = _find("timestamp")
    rating_idx = _find("rating")
    if uid_idx is None or iid_idx is None or ts_idx is None:
        raise RuntimeError(
            f"yelp2018.inter header 缺 user_id/item_id/timestamp 列。实际 header: {cols}"
        )
    return uid_idx, iid_idx, rating_idx, ts_idx, cols


def main():
    parser = argparse.ArgumentParser(
        description="Yelp 80/20 时间切分，可选先按 user 子采样"
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="若设置：随机保留恰好这么多用户（及其全部交互）。与全量二选一。",
    )
    parser.add_argument(
        "--user_sample_ratio",
        type=float,
        default=None,
        help="若设置且未用 --max_users：保留该比例用户，(0,1]；如 0.1 约 10%% 用户。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="用户抽样的随机种子，便于复现",
    )
    parser.add_argument(
        "--data_tag",
        type=str,
        default=None,
        help="非空时：写入 yelp2018-pretrain-{tag}/ 与 yelp2018-t2t-{tag}/。"
        "使用 --max_users / --user_sample_ratio 时必须带本参数，避免覆盖全量目录。",
    )
    args = parser.parse_args()
    if args.max_users is not None and args.max_users < 1:
        raise SystemExit("--max_users 须 >= 1")
    if args.user_sample_ratio is not None and not (0.0 < args.user_sample_ratio <= 1.0):
        raise SystemExit("--user_sample_ratio 须在 (0,1]")

    tag = _sanitize_data_tag(args.data_tag)
    use_sub = args.max_users is not None or args.user_sample_ratio is not None
    if use_sub and not tag:
        raise SystemExit(
            "使用 --max_users 或 --user_sample_ratio 时必须同时指定 --data_tag（如 u5000），"
            "子集会写入独立 dataset 目录，不会覆盖 yelp2018-pretrain / yelp2018-t2t。"
        )

    if tag:
        ptn = f"yelp2018-pretrain-{tag}"
        t2n = f"yelp2018-t2t-{tag}"
        pretrain_dir = os.path.join(DATA_ROOT, ptn)
        t2t_dir = os.path.join(DATA_ROOT, t2n)
        pretrain_file = f"{ptn}.inter"
        t2t_file = f"{t2n}.inter"
    else:
        pretrain_dir = DEFAULT_PRETRAIN_DIR
        t2t_dir = DEFAULT_T2T_DIR
        pretrain_file = "yelp2018-pretrain.inter"
        t2t_file = "yelp2018-t2t.inter"

    inter_path = os.path.join(YELP_DIR, "yelp2018.inter")
    if not os.path.isfile(inter_path):
        raise FileNotFoundError(
            f"需要 {inter_path}。请先把 RecBole atomic 格式的 yelp2018.inter 放到该路径下"
            f"（至少包含 user_id, item_id, rating, timestamp 四列，tab 分隔）"
        )

    with open(inter_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0]
    uid_idx, iid_idx, rating_idx, ts_idx, cols = _detect_columns(header)

    rows = [ln.rstrip("\n").split("\t") for ln in lines[1:] if ln.strip()]
    # 过滤列数不对的脏行
    rows = [r for r in rows if len(r) == len(cols)]

    u_list = list({r[uid_idx] for r in rows})
    n_u = len(u_list)
    if args.max_users is not None:
        k = min(args.max_users, n_u)
        if k < n_u:
            rng = random.Random(args.seed)
            keep = set(rng.sample(u_list, k))
            rows = [r for r in rows if r[uid_idx] in keep]
        print(
            f"[user subset] 目标 {args.max_users} 用户 -> 实际保留 {k} 用户, "
            f"{len(rows)} 条交互 (seed={args.seed})"
        )
    elif args.user_sample_ratio is not None:
        r = float(args.user_sample_ratio)
        k = max(1, int(n_u * r))
        if k < n_u:
            rng = random.Random(args.seed)
            keep = set(rng.sample(u_list, k))
            rows = [r for r in rows if r[uid_idx] in keep]
        else:
            k = n_u
        print(
            f"[user subset] 比例 {r:g} -> 保留 {k} / {n_u} 用户, "
            f"{len(rows)} 条交互 (seed={args.seed})"
        )
    else:
        print(f"[user subset] 全量: {n_u} 用户, {len(rows)} 条交互")

    # 按 timestamp 排序
    def _ts(r):
        try:
            return float(r[ts_idx])
        except ValueError:
            return 0.0

    rows.sort(key=_ts)
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

    t_min = pretrain_rows[0][ts_idx]
    t_max = t2t_rows[-1][ts_idx]
    sample_uid_p = pretrain_rows[0][uid_idx]
    sample_iid_p = pretrain_rows[0][iid_idx]
    sample_uid_t = t2t_rows[0][uid_idx]
    sample_iid_t = t2t_rows[0][iid_idx]

    def _make_row(uid, iid, rating_val, ts_val):
        r = [""] * len(cols)
        r[uid_idx] = uid
        r[iid_idx] = iid
        r[ts_idx] = ts_val
        if rating_idx is not None:
            r[rating_idx] = rating_val
        # 其它列填 0 / 空串；rating=0 标记占位
        for i, v in enumerate(r):
            if v == "":
                r[i] = "0"
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

    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(t2t_dir, exist_ok=True)

    def _write_inter(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            for r in rows:
                f.write("\t".join(r) + "\n")

    ptn_base = pretrain_file.replace(".inter", "")
    t2n_base = t2t_file.replace(".inter", "")

    _write_inter(os.path.join(pretrain_dir, pretrain_file), pretrain_all)
    _write_inter(os.path.join(t2t_dir, t2t_file), t2t_all)

    for suffix in [".user", ".item"]:
        src = os.path.join(YELP_DIR, "yelp2018" + suffix)
        if os.path.isfile(src):
            for dst_dir, name_base in [(pretrain_dir, ptn_base), (t2t_dir, t2n_base)]:
                dst = os.path.join(dst_dir, name_base + suffix)
                shutil.copy2(src, dst)

    tag_info = f" [data_tag={tag}]" if tag else " [默认全量目录，无 tag]"
    print(
        f"已创建 {pretrain_dir}/{pretrain_file} ({len(pretrain_all)} 行) 和 "
        f"{t2t_dir}/{t2t_file} ({len(t2t_all)} 行){tag_info}"
    )
    print(f"  pretrain: {split_idx} 真实 + {len(dummy_pretrain)} 占位")
    print(f"  t2t:      {n - split_idx} 真实 + {len(dummy_t2t)} 占位")
    print(f"  users: pretrain {len(users_pretrain)}, t2t {len(users_t2t)}, "
          f"only-in-pretrain {len(users_only_pretrain)}, only-in-t2t {len(users_only_t2t)}")
    print(f"  items: pretrain {len(items_pretrain)}, t2t {len(items_t2t)}, "
          f"only-in-pretrain {len(items_only_pretrain)}, only-in-t2t {len(items_only_t2t)}")


if __name__ == "__main__":
    main()
