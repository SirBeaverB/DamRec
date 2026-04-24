#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yelp2018 流水线 smoke test - 验证 download / split / 3-epoch GDN 训练全链路跑通。

步骤：
  1. download_yelp2018.py  (若 yelp2018.inter 已存在则跳过)
  2. prepare_yelp2018_80_20_split.py  (若 yelp2018-t2t.inter 已存在则跳过)
  3. run_per_model_pretrain_t2t_yelp2018.py --models GDN --seeds 2020 --epochs 3 --n_gpus 1
  4. 校验 checkpoint 和结果 JSON 存在，打印 recall@10

Yelp 上 **user 数可达百万级**，Step B「按用户灌状态」可能占 **数小时**，看起来像卡住但其实在跑。
快速验管线可用 **--skip-dump**（跳过导出，T2T 仍跑，GDN 从空状态起）。

每步失败立即退出并给出对应修复提示。

Usage:
  python scripts/smoke_test_yelp2018.py
  python scripts/smoke_test_yelp2018.py --skip-dump      # 推荐 smoke：省掉百万用户 state dump
  python scripts/smoke_test_yelp2018.py --skip-download   # 已有 zip / 已 extract
  python scripts/smoke_test_yelp2018.py --skip-split      # 已切过
  python scripts/smoke_test_yelp2018.py --epochs 1        # 更快
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(_script_dir)

YELP_INTER = os.path.join(PROJ, "dataset", "yelp2018", "yelp2018.inter")
T2T_INTER = os.path.join(PROJ, "dataset", "yelp2018-t2t", "yelp2018-t2t.inter")
PRETRAIN_INTER = os.path.join(PROJ, "dataset", "yelp2018-pretrain", "yelp2018-pretrain.inter")


def _run(label, cmd, cwd=PROJ):
    print(f"\n========== [{label}] ==========")
    print(f" cmd: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=cwd)
    elapsed = time.time() - t0
    print(f"--- [{label}] rc={rc}  elapsed={elapsed:.0f}s ---")
    if rc != 0:
        raise SystemExit(f"[{label}] FAIL rc={rc}. 终止 smoke test。")
    return elapsed


def _expect_file(path, label):
    if not os.path.isfile(path):
        raise SystemExit(f"[check] {label} 缺失: {path}")
    size = os.path.getsize(path)
    print(f"[check] {label} OK: {path}  ({size/1024/1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="yelp2018 全链路 smoke test")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=3, help="训练 epoch，默认 3")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument(
        "--skip-dump",
        dest="skip_dump",
        action="store_true",
        help="不导出 user_states（Yelp 唯一 user 可超百万，dump 极慢）；T2T 仍跑",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("yelp2018 smoke test - download + split + 3-epoch GDN")
    print("=" * 70)

    total_t = {}

    # Step 1: download
    if args.skip_download or os.path.isfile(YELP_INTER):
        print(f"\n[1/4] skip download (已有 {YELP_INTER})")
    else:
        total_t["download"] = _run(
            "1/4 download",
            [sys.executable, os.path.join(_script_dir, "download_yelp2018.py")],
        )
    _expect_file(YELP_INTER, "yelp2018.inter")

    # Step 2: split
    if args.skip_split or (os.path.isfile(PRETRAIN_INTER) and os.path.isfile(T2T_INTER)):
        print(f"\n[2/4] skip split (已有 pretrain & t2t)")
    else:
        total_t["split"] = _run(
            "2/4 split",
            [sys.executable, os.path.join(_script_dir, "prepare_yelp2018_80_20_split.py")],
        )
    _expect_file(PRETRAIN_INTER, "yelp2018-pretrain.inter")
    _expect_file(T2T_INTER, "yelp2018-t2t.inter")

    # Step 3: 3-epoch GDN pretrain + dump + streaming T2T
    train_cmd = [
        sys.executable,
        os.path.join(_script_dir, "run_per_model_pretrain_t2t_yelp2018.py"),
        "--models", "GDN",
        "--seeds", "2020",
        "--epochs", str(args.epochs),
        "--max_seq_len", str(args.max_seq_len),
        "--n_gpus", "1",
    ]
    if args.skip_dump:
        train_cmd.append("--skip_dump")
    total_t["train"] = _run(
        "3/4 train (GDN, "
        f"{args.epochs} epochs"
        + (", skip state dump" if args.skip_dump else "")
        + ")",
        train_cmd,
    )

    # Step 4: verify artifacts
    print("\n========== [4/4 verify artifacts] ==========")
    ckp_dir = os.path.join(
        PROJ, "saved", f"per_model_pretrain_yelp2018_L{args.max_seq_len}_s2020"
    )
    pths = sorted(glob.glob(os.path.join(ckp_dir, "GDN-*.pth")))
    if not pths:
        raise SystemExit(f"[check] 未找到 GDN checkpoint: {ckp_dir}/GDN-*.pth")
    print(f"[check] GDN ckp OK: {pths[-1]}")

    if not args.skip_dump:
        states_path = os.path.join(ckp_dir, "user_states_GDN.pt")
        _expect_file(states_path, "user_states_GDN.pt")
    else:
        print("[check] 跳过 user_states 校验 (smoke 使用了 --skip-dump)")

    # 找最新的 per_model_streaming_yelp2018_* 目录，校验 json
    out_dirs = sorted(
        glob.glob(os.path.join(PROJ, "experiment_results",
                                 f"per_model_streaming_yelp2018_L{args.max_seq_len}_*"))
    )
    if not out_dirs:
        raise SystemExit("[check] 未找到 experiment_results/per_model_streaming_yelp2018_*/ 目录")
    latest_dir = out_dirs[-1]
    gdn_json = os.path.join(latest_dir, "GDN_s2020.json")
    _expect_file(gdn_json, "GDN_s2020.json")
    with open(gdn_json, encoding="utf-8") as f:
        rec = json.load(f)
    result = rec.get("result", {})
    if not result:
        raise SystemExit(f"[check] {gdn_json} 无 result 字段（T2T 阶段未跑出来）")

    print("\n========== summary ==========")
    for k, v in total_t.items():
        print(f"  {k}: {v:.0f}s")
    print(f"  GDN recall@10  = {result.get('recall@10')}")
    print(f"  GDN ndcg@10    = {result.get('ndcg@10')}")
    print(f"  GDN mrr@10     = {result.get('mrr@10')}")
    print(f"  GDN recall@20  = {result.get('recall@20')}")
    print(f"  GDN recall@50  = {result.get('recall@50')}")
    print(f"  pretrain_sec   = {rec.get('pretrain_sec'):.0f}s" if rec.get("pretrain_sec") else "  pretrain skip")
    print(f"  t2t_sec        = {rec.get('t_sec'):.0f}s" if rec.get("t_sec") else "  t2t skip")
    print(f"  ckp            = {rec.get('ckp_path')}")
    print(f"  log_dir        = {latest_dir}")
    print("\n[smoke OK] 全链路跑通 ✓  现在可以发正式跑：")
    print(f"  python scripts/run_per_model_pretrain_t2t_yelp2018.py \\")
    print(f"    --models GDN,Adam,Fro --seeds 2020,2021,2022 --n_gpus 2")


if __name__ == "__main__":
    main()
