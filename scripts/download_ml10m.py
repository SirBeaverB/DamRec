#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 GroupLens ML-10M 数据集并转换为 RecBole atomic 格式。

数据源：
  https://files.grouplens.org/datasets/movielens/ml-10m.zip
  原始格式：ml-10M100K/ratings.dat  UserID::MovieID::Rating::Timestamp

流程：
  1. 下载 zip -> dataset/ml-10m.zip
  2. 解压 ratings.dat
  3. 转换为 RecBole .inter 格式（tab 分隔，header 含 :token/:float 后缀）
  4. 写出 dataset/ml-10m/ml-10m.inter
  5. 删除 zip（可选）

下载后请接着跑：
  python scripts/prepare_ml10m_80_20_split.py --data_tag dense_u20k \
      --max_users 20000 --min_user_inter 20 --min_item_inter 5 \
      --min_pretrain_inter 20 --filter_cold_items --seed 42

用法：
  python scripts/download_ml10m.py
  python scripts/download_ml10m.py --force
  python scripts/download_ml10m.py --keep-zip
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(_script_dir)
DATA_ROOT = os.path.join(PROJ_ROOT, "dataset")
TARGET_DIR = os.path.join(DATA_ROOT, "ml-10m")
TARGET_INTER = os.path.join(TARGET_DIR, "ml-10m.inter")
ZIP_PATH = os.path.join(DATA_ROOT, "ml-10m.zip")

URL = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
RATINGS_IN_ZIP = "ml-10M100K/ratings.dat"

INTER_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def _human_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def _report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100.0)
        bar = int(pct / 2)
        sys.stdout.write(
            f"\r  [{'=' * bar}{' ' * (50 - bar)}] "
            f"{pct:5.1f}%  {_human_bytes(downloaded)}/{_human_bytes(total_size)}"
        )
    else:
        sys.stdout.write(f"\r  下载中... {_human_bytes(downloaded)}")
    sys.stdout.flush()


def _download(force=False):
    os.makedirs(DATA_ROOT, exist_ok=True)
    if os.path.isfile(ZIP_PATH) and not force:
        print(f"[skip] zip 已存在: {ZIP_PATH}  (传 --force 可重下)")
        return
    print(f"[download] {URL}")
    print(f"           -> {ZIP_PATH}")
    try:
        urllib.request.urlretrieve(URL, ZIP_PATH, _report_progress)
        sys.stdout.write("\n")
    except Exception as e:
        if os.path.isfile(ZIP_PATH):
            os.remove(ZIP_PATH)
        raise SystemExit(
            f"\n[download FAIL] {e}\n"
            f"请手动下载 {URL} 到 {ZIP_PATH}，再重跑本脚本。"
        )


def _convert(force=False):
    if os.path.isfile(TARGET_INTER) and not force:
        print(f"[skip] 已存在: {TARGET_INTER}  (传 --force 可重转)")
        return

    if not os.path.isfile(ZIP_PATH):
        raise SystemExit(f"[convert] zip 不存在: {ZIP_PATH}")

    print(f"[extract+convert] {ZIP_PATH}:{RATINGS_IN_ZIP} -> {TARGET_INTER}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # 容错：zip 内路径可能大小写不同
        names = {n.lower(): n for n in zf.namelist()}
        key = RATINGS_IN_ZIP.lower()
        if key not in names:
            raise SystemExit(
                f"[convert FAIL] zip 内未找到 {RATINGS_IN_ZIP}。\n"
                f"实际文件列表（前20）: {list(zf.namelist())[:20]}"
            )
        actual_name = names[key]
        with zf.open(actual_name) as src, open(TARGET_INTER, "w", encoding="utf-8") as dst:
            dst.write(INTER_HEADER)
            n_written = 0
            for raw in src:
                line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
                if not line.strip():
                    continue
                parts = line.split("::")
                if len(parts) < 4:
                    continue
                uid, iid, rating, ts = parts[0], parts[1], parts[2], parts[3]
                dst.write(f"{uid}\t{iid}\t{rating}\t{ts}\n")
                n_written += 1
                if n_written % 1_000_000 == 0:
                    sys.stdout.write(f"\r  已转换 {n_written:,} 行...")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    print(f"[convert] 完成: {n_written:,} 条交互写入 {TARGET_INTER}")


def _verify():
    if not os.path.isfile(TARGET_INTER):
        raise SystemExit(f"[verify FAIL] 未找到 {TARGET_INTER}")
    size_mb = os.path.getsize(TARGET_INTER) / 1024 / 1024
    with open(TARGET_INTER, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        n_lines = sum(1 for _ in f) + 1
    print(f"[verify] {TARGET_INTER} ({size_mb:.1f} MB, {n_lines:,} 行)")
    print(f"         header: {header}")


def main():
    parser = argparse.ArgumentParser(description="下载并转换 ML-10M 为 RecBole atomic 格式")
    parser.add_argument("--force", action="store_true", help="已存在也重下/重转")
    parser.add_argument("--keep-zip", action="store_true", help="保留 zip 文件")
    args = parser.parse_args()

    if os.path.isfile(TARGET_INTER) and not args.force:
        print(f"[skip] 数据已就绪: {TARGET_INTER}")
        _verify()
        return

    _download(force=args.force)
    _convert(force=args.force)
    _verify()

    if not args.keep_zip and os.path.isfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        print(f"[cleanup] 删除 zip: {ZIP_PATH}")

    print("\n下一步（推荐正式实验配置）：")
    print("  python scripts/prepare_ml10m_80_20_split.py \\")
    print("      --data_tag dense_u20k \\")
    print("      --max_users 20000 \\")
    print("      --min_user_inter 20 \\")
    print("      --min_item_inter 5 \\")
    print("      --min_pretrain_inter 20 \\")
    print("      --filter_cold_items \\")
    print("      --seed 42")


if __name__ == "__main__":
    main()
