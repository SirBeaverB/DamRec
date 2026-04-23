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
  POP_user    推荐该用户在 ml-1m-pretrain 历史中频次 top-10 item，
              不足 10 个补 POP_global

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
PRETRAIN_INTER = os.path.join(PROJ, "dataset", "ml-1m-pretrain", "ml-1m-pretrain.inter")
T2T_INTER = os.path.join(PROJ, "dataset", "ml-1m-t2t", "ml-1m-t2t.inter")


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


def _topk_metrics(rank, k=10):
    """rank: 1-based position, or None if not in top-k."""
    if rank is None or rank > k:
        return 0.0, 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1), 1.0 / rank


def _eval_pop_global(test_points, pop_counter, k=10):
    top_k = [iid for iid, _ in pop_counter.most_common(k)]
    item_to_rank = {iid: r + 1 for r, iid in enumerate(top_k)}
    rec, ndcg, mrr = 0.0, 0.0, 0.0
    for _uid, target in test_points:
        r = item_to_rank.get(target)
        a, b, c = _topk_metrics(r, k=k)
        rec += a; ndcg += b; mrr += c
    n = len(test_points)
    return rec / n, ndcg / n, mrr / n


def _eval_pop_user(test_points, user_history, pop_counter, k=10):
    """每 user：自己历史中 top-k；不足补全局热门（去重）。"""
    global_top = [iid for iid, _ in pop_counter.most_common(k * 3)]  # 备用够长
    rec, ndcg, mrr = 0.0, 0.0, 0.0
    for uid, target in test_points:
        seen = user_history.get(uid, Counter())
        ranking = [iid for iid, _ in seen.most_common(k)]
        if len(ranking) < k:
            for iid in global_top:
                if iid not in ranking:
                    ranking.append(iid)
                if len(ranking) >= k:
                    break
        ranking = ranking[:k]
        rank = (ranking.index(target) + 1) if target in ranking else None
        a, b, c = _topk_metrics(rank, k=k)
        rec += a; ndcg += b; mrr += c
    n = len(test_points)
    return rec / n, ndcg / n, mrr / n


def main():
    parser = argparse.ArgumentParser(
        description="Popularity baselines for streaming T2T eval (ml-1m 80/20)"
    )
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="每用户尾部多少比例作 test，默认 0.1，与 streaming pipeline 对齐")
    parser.add_argument("-L", "--max_seq_len", type=int, default=64,
                        help="仅用于输出文件命名与其他脚本对齐；不影响 popularity 计算")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="结果输出目录，默认 experiment_results/")
    args = parser.parse_args()

    if not os.path.isfile(PRETRAIN_INTER) or not os.path.isfile(T2T_INTER):
        raise SystemExit(
            f"未找到 ml-1m-pretrain / ml-1m-t2t.inter，请先运行：\n"
            f"  python scripts/prepare_ml1m_80_20_split.py"
        )

    print(f"[1/3] 读取 {PRETRAIN_INTER}")
    pretrain_rows, n_ph_pre = _read_inter(PRETRAIN_INTER)
    print(f"      真实交互 {len(pretrain_rows)} 条，跳过占位 {n_ph_pre} 条")

    print(f"[2/3] 读取 {T2T_INTER}")
    t2t_rows, n_ph_t2t = _read_inter(T2T_INTER)
    print(f"      真实交互 {len(t2t_rows)} 条，跳过占位 {n_ph_t2t} 条")

    print(f"[3/3] 计算基线")
    pop_counter = Counter(iid for _u, iid, _r, _t in pretrain_rows)
    user_hist = _build_user_history(pretrain_rows)
    test_points, t2t_users = _build_test_points(t2t_rows, args.test_ratio)
    n_users_t2t = len(t2t_users)
    n_test = len(test_points)

    rec_g, ndcg_g, mrr_g = _eval_pop_global(test_points, pop_counter, k=10)
    rec_u, ndcg_u, mrr_u = _eval_pop_user(test_points, user_hist, pop_counter, k=10)

    # 简单理论参考：纯随机 = 10 / num_unique_items
    n_items = len(pop_counter)
    random_recall = 10.0 / n_items if n_items else 0.0

    out_dir = args.output_dir or os.path.join(PROJ, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(out_dir, f"popularity_baseline_streaming_L{args.max_seq_len}_{ts}.txt")
    csv_path = os.path.join(out_dir, f"popularity_baseline_streaming_L{args.max_seq_len}_{ts}.csv")

    label_w, col_w = 14, 14
    lines = [
        "=" * 90,
        "Popularity Baselines for Streaming T2T Evaluation (ml-1m 80/20)",
        "=" * 90,
        "",
        "[实验情景]",
        "  与 scripts/run_per_model_pretrain_t2t_1m.py 共享 test 点定义：",
        "  ml-1m-t2t 内每用户按 timestamp 排序，尾部 max(1, int(n*test_ratio)) 为 test 点。",
        "  此脚本不训练任何模型，只用 pretrain 的 item 频次做 popularity 推荐。",
        "",
        "[基线说明]",
        "  POP_global  永远推荐 pretrain 全局频次 top-10 item（不看 user，最弱基线）",
        "  POP_user    推荐用户在 pretrain 历史中频次最高的 item，不足 10 个补全局热门",
        "",
        "[与 streaming pipeline 的差异]",
        "  本脚本剔除 rating=0 的占位行，得到 clean 基线；流式 pipeline 不过滤。",
        "  对评估影响极小（占位是单一 item + t_max 时间戳），但对所有方法一致。",
        "",
        "[数据规模]",
        f"  pretrain 真实交互 = {len(pretrain_rows):>10}",
        f"  pretrain 唯一 item = {n_items:>10}",
        f"  pretrain 唯一 user = {len(user_hist):>10}",
        f"  t2t 真实交互       = {len(t2t_rows):>10}",
        f"  t2t 唯一 user      = {n_users_t2t:>10}",
        f"  test 点总数       = {n_test:>10}  (test_ratio={args.test_ratio})",
        f"  平均 test 点/user  = {n_test/max(1,n_users_t2t):>10.2f}",
        "",
        "-" * 90,
        "TEST  —  Recall / NDCG / MRR @10  (越大越好)",
        "-" * 90,
        f"{'Metric':<{label_w}}{'POP_global':<{col_w}}{'POP_user':<{col_w}}",
        "-" * (label_w + col_w * 2),
        f"{'recall@10':<{label_w}}{rec_g:<{col_w}.4f}{rec_u:<{col_w}.4f}",
        f"{'ndcg@10':<{label_w}}{ndcg_g:<{col_w}.4f}{ndcg_u:<{col_w}.4f}",
        f"{'mrr@10':<{label_w}}{mrr_g:<{col_w}.4f}{mrr_u:<{col_w}.4f}",
        "",
        f"[理论参考] 纯随机 Recall@10 ≈ 10/{n_items} = {random_recall:.5f}",
        "",
        "[判读指南]",
        "  对照 per_model_streaming_L*.txt 里 5 模型的 Recall@10：",
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

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("baseline,recall@10,ndcg@10,mrr@10,n_test_points,n_t2t_users,test_ratio\n")
        f.write(f"POP_global,{rec_g:.4f},{ndcg_g:.4f},{mrr_g:.4f},{n_test},{n_users_t2t},{args.test_ratio}\n")
        f.write(f"POP_user,{rec_u:.4f},{ndcg_u:.4f},{mrr_u:.4f},{n_test},{n_users_t2t},{args.test_ratio}\n")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
