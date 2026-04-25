#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Popularity baselines for streaming T2T evaluation on ml-1m (80/20).

用途：作为 scripts/run_per_model_pretrain_t2t_1m.py 五模型流式结果的"地板线"。
判断五模型 0.01 量级的 Recall@10 是真在做有效推荐，还是没超过简单 popularity。

测试点定义与流式 pipeline 完全一致：
  ml-1m-t2t 内每用户按 timestamp 排序后，尾部 max(1, int(n*test_ratio)) 为 test。

两个基线：
  POP_global  永远推荐 ml-1m-pretrain 全局频次 top-10 item（最弱基线，不看 user）
  POP_user    推荐全局热门中该用户在 pretrain 里未见过的 top-10 item（popular unseen）

数据筛选：
  剔除 rating=0 的占位行（prepare_ml1m_80_20_split.py 注入的词表对齐 dummy）。
  与流式 pipeline 的差异：流式 pipeline 不过滤占位，但占位为单一 item ID + t_max
  时间戳，对评估的影响很小且对所有模型一致；这里过滤是为了得到"干净"基线。

用法:
  python scripts/run_popularity_baseline_streaming.py
  python scripts/run_popularity_baseline_streaming.py --test_ratio 0.1
  python scripts/run_popularity_baseline_streaming.py -L 64    # 仅命名用，不影响计算
"""

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_paths(dataset, data_tag=None):
    """
    data_tag=None 或 ""  -> dataset/{dataset}-pretrain/{dataset}-pretrain.inter
    data_tag="u5000"      -> dataset/{dataset}-pretrain-u5000/{dataset}-pretrain-u5000.inter
    """
    suffix = f"-{data_tag}" if data_tag else ""
    pre_name = f"{dataset}-pretrain{suffix}"
    t2t_name = f"{dataset}-t2t{suffix}"
    pretrain = os.path.join(PROJ, "dataset", pre_name, f"{pre_name}.inter")
    t2t = os.path.join(PROJ, "dataset", t2t_name, f"{t2t_name}.inter")
    return pretrain, t2t


def _read_inter(path):
    """Return list of (uid, iid, rating, timestamp). Skip rating==0 placeholders."""
    real_rows, placeholder_rows = [], 0
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                uid, iid = parts[0], parts[1]
                rating = float(parts[2])
                ts = float(parts[3])
            except ValueError:
                continue
            if rating == 0:
                placeholder_rows += 1
                continue
            real_rows.append((uid, iid, rating, ts))
    return real_rows, placeholder_rows


def _build_test_points(t2t_rows, test_ratio):
    """对每个 user，按时间序，尾部 max(1, int(n*ratio)) 为 test。"""
    by_user = defaultdict(list)
    for uid, iid, _r, t in t2t_rows:
        by_user[uid].append((iid, t))
    test_points = []
    for uid, lst in by_user.items():
        lst.sort(key=lambda x: x[1])
        n = len(lst)
        n_test = max(1, int(n * test_ratio))
        for iid, _ in lst[-n_test:]:
            test_points.append((uid, iid))
    return test_points, by_user


def _build_user_history(rows):
    """{uid: Counter(iid → freq)}，从 pretrain 真实交互。"""
    by_user = defaultdict(Counter)
    for uid, iid, _r, _t in rows:
        by_user[uid][iid] += 1
    return by_user


def _topk_metrics(rank, k):
    """rank: 1-based position, or None if not in top-k."""
    if rank is None or rank > k:
        return 0.0, 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1), 1.0 / rank


def _eval_pop_global(test_points, pop_counter, ks=(10, 20, 50)):
    max_k = max(ks)
    top_items = [iid for iid, _ in pop_counter.most_common(max_k)]
    item_to_rank = {iid: r + 1 for r, iid in enumerate(top_items)}
    acc = {k: [0.0, 0.0, 0.0] for k in ks}
    for _uid, target in test_points:
        rank = item_to_rank.get(target)
        for k in ks:
            a, b, c = _topk_metrics(rank, k)
            acc[k][0] += a; acc[k][1] += b; acc[k][2] += c
    n = len(test_points)
    return {k: tuple(v / n for v in acc[k]) for k in ks}


def _eval_pop_user(test_points, user_history, pop_counter, ks=(10, 20, 50)):
    """每 user：全局热门中排除 pretrain 已见 item，按 ks 最大值截取（popular unseen）。"""
    max_k = max(ks)
    global_top = [iid for iid, _ in pop_counter.most_common()]
    acc = {k: [0.0, 0.0, 0.0] for k in ks}
    for uid, target in test_points:
        seen = set(user_history.get(uid, {}).keys())
        ranking = []
        for iid in global_top:
            if iid not in seen:
                ranking.append(iid)
            if len(ranking) >= max_k:
                break
        rank = (ranking.index(target) + 1) if target in ranking else None
        for k in ks:
            a, b, c = _topk_metrics(rank, k)
            acc[k][0] += a; acc[k][1] += b; acc[k][2] += c
    n = len(test_points)
    return {k: tuple(v / n for v in acc[k]) for k in ks}


def main():
    parser = argparse.ArgumentParser(
        description="Popularity baselines for streaming T2T eval (80/20 切分)"
    )
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        help="数据集名，默认 ml-1m。支持 yelp2018 等 (需对应 {name}-pretrain/{name}-t2t 目录存在)")
    parser.add_argument("--data-tag", dest="data_tag", type=str, default=None,
                        help="采样子集 tag，如 u5000 / u200000。默认 None 算全体。"
                             "带 tag 时读 {dataset}-pretrain-{tag} / {dataset}-t2t-{tag} 目录")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="每用户尾部多少比例作 test，默认 0.1，与 streaming pipeline 对齐")
    parser.add_argument("-L", "--max_seq_len", type=int, default=64,
                        help="仅用于输出文件命名与其他脚本对齐；不影响 popularity 计算")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="结果输出目录，默认 experiment_results/")
    args = parser.parse_args()

    pretrain_inter, t2t_inter = _resolve_paths(args.dataset, args.data_tag)
    if not os.path.isfile(pretrain_inter) or not os.path.isfile(t2t_inter):
        tag_part = f"-{args.data_tag}" if args.data_tag else ""
        raise SystemExit(
            f"未找到 {args.dataset}-pretrain{tag_part} / {args.dataset}-t2t{tag_part}.inter。\n"
            f"  pretrain: {pretrain_inter}\n"
            f"  t2t:      {t2t_inter}\n"
            f"请先跑 split 脚本（带对应采样 tag 参数）。"
        )

    print(f"[1/3] 读取 {pretrain_inter}")
    pretrain_rows, n_ph_pre = _read_inter(pretrain_inter)
    print(f"      真实交互 {len(pretrain_rows)} 条，跳过占位 {n_ph_pre} 条")

    print(f"[2/3] 读取 {t2t_inter}")
    t2t_rows, n_ph_t2t = _read_inter(t2t_inter)
    print(f"      真实交互 {len(t2t_rows)} 条，跳过占位 {n_ph_t2t} 条")

    print(f"[3/3] 计算基线")
    pop_counter = Counter(iid for _u, iid, _r, _t in pretrain_rows)
    user_hist = _build_user_history(pretrain_rows)
    test_points, t2t_users = _build_test_points(t2t_rows, args.test_ratio)
    n_users_t2t = len(t2t_users)
    n_test = len(test_points)

    KS = (10, 20, 50)
    res_g = _eval_pop_global(test_points, pop_counter, ks=KS)
    res_u = _eval_pop_user(test_points, user_hist, pop_counter, ks=KS)

    n_items = len(pop_counter)
    n_users_pretrain = len(user_hist)
    n_users_both = len(set(user_hist.keys()) & set(t2t_users.keys()))
    n_users_only_pretrain = n_users_pretrain - n_users_both
    n_users_coldstart = n_users_t2t - n_users_both

    out_dir = args.output_dir or os.path.join(PROJ, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_tag_part = f"_{args.data_tag}" if args.data_tag else ""
    tag = f"{args.dataset}{data_tag_part}_L{args.max_seq_len}"
    txt_path = os.path.join(out_dir, f"popularity_baseline_streaming_{tag}_{ts}.txt")
    csv_path = os.path.join(out_dir, f"popularity_baseline_streaming_{tag}_{ts}.csv")

    label_w, col_w = 16, 14
    sep = "-" * (label_w + col_w * 2)

    def _metric_rows(ks, res_g, res_u):
        rows = []
        for k in ks:
            rg, ng, mg = res_g[k]
            ru, nu, mu = res_u[k]
            rows += [
                f"{'recall@'+str(k):<{label_w}}{rg:<{col_w}.4f}{ru:<{col_w}.4f}",
                f"{'ndcg@'+str(k):<{label_w}}{ng:<{col_w}.4f}{nu:<{col_w}.4f}",
                f"{'mrr@'+str(k):<{label_w}}{mg:<{col_w}.4f}{mu:<{col_w}.4f}",
                "",
            ]
        return rows

    lines = [
        "=" * 90,
        f"Popularity Baselines for Streaming T2T Evaluation ({args.dataset}{data_tag_part} 80/20)",
        "=" * 90,
        "",
        "[实验情景]",
        "  与 scripts/run_per_model_pretrain_t2t_1m.py 共享 test 点定义：",
        f"  {args.dataset}-t2t 内每用户按 timestamp 排序，尾部 max(1, int(n*test_ratio)) 为 test 点。",
        "  此脚本不训练任何模型，只用 pretrain 的 item 频次做 popularity 推荐。",
        "",
        "[基线说明]",
        "  POP_global  永远推荐 pretrain 全局频次 top-K item（不看 user，最弱基线）",
        "  POP_user    推荐全局热门中该用户在 pretrain 里未见过的 item（popular unseen）",
        "",
        "[与 streaming pipeline 的差异]",
        "  本脚本剔除 rating=0 的占位行，得到 clean 基线；流式 pipeline 不过滤。",
        "  对评估影响极小（占位是单一 item + t_max 时间戳），但对所有方法一致。",
        "",
        "[数据规模]",
        f"  pretrain 真实交互 = {len(pretrain_rows):>10}",
        f"  pretrain 活跃 item = {n_items:>10}",
        f"  pretrain 活跃 user = {n_users_pretrain:>10}  (在 pretrain 段有真实交互)",
        f"  t2t 真实交互       = {len(t2t_rows):>10}",
        f"  t2t 活跃 user      = {n_users_t2t:>10}  (在 t2t 段有真实交互)",
        f"    其中 pretrain+t2t 均有 = {n_users_both:>8}  (暖用户，有 pretrain 状态)",
        f"    其中仅出现在 t2t      = {n_users_coldstart:>8}  (冷启动，无 pretrain 状态)",
        f"    仅出现在 pretrain     = {n_users_only_pretrain:>8}  (不参与 t2t 评估)",
        f"  test 点总数       = {n_test:>10}  (test_ratio={args.test_ratio})",
        f"  平均 test 点/user  = {n_test/max(1,n_users_t2t):>10.2f}",
        "",
        sep,
        "TEST  —  Recall / NDCG / MRR @K  (越大越好)",
        sep,
        f"{'Metric':<{label_w}}{'POP_global':<{col_w}}{'POP_user':<{col_w}}",
        sep,
        *_metric_rows(KS, res_g, res_u),
        f"[理论参考] 纯随机 Recall@10 ≈ {10.0/n_items:.5f}  "
        f"@20 ≈ {20.0/n_items:.5f}  @50 ≈ {50.0/n_items:.5f}  (10/N, 20/N, 50/N where N={n_items})",
        "",
        "[判读指南]",
        "  对照 per_model_streaming_L*.txt 里各模型的 Recall@K：",
        "    模型 < POP_global             → 模型完全没学到，严重病理",
        "    POP_global ≤ 模型 < POP_user  → 只学到流行偏好，未学到用户个性化",
        "    模型 ≥ POP_user               → 真正学到了用户级信号，方法有效",
        "  通常论文里强 baseline 是 POP_user，弱 baseline 是 POP_global。",
        "=" * 90,
    ]
    table = "\n".join(lines)
    print("\n" + table)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nTXT: {txt_path}")

    csv_cols = []
    for k in KS:
        csv_cols += [f"recall@{k}", f"ndcg@{k}", f"mrr@{k}"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("baseline," + ",".join(csv_cols) + ",n_test_points,n_t2t_users,test_ratio\n")
        for name, res in [("POP_global", res_g), ("POP_user", res_u)]:
            vals = ",".join(f"{res[k][i]:.4f}" for k in KS for i in range(3))
            f.write(f"{name},{vals},{n_test},{n_users_t2t},{args.test_ratio}\n")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
