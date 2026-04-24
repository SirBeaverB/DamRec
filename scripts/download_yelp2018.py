#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 RecBole 官方 yelp2018 atomic 数据集（含 timestamp）。

数据源：
  https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp2018.zip
  (出自 recbole/properties/dataset/url.yaml 的 yelp-2018 条目)

流程：
  1. 下载 zip → dataset/yelp2018.zip
  2. 解压 → dataset/yelp2018/
  3. 校验 dataset/yelp2018/yelp2018.inter 存在

下载后请接着跑：
  python scripts/prepare_yelp2018_80_20_split.py

用法：
  python scripts/download_yelp2018.py                # 默认路径
  python scripts/download_yelp2018.py --force        # 已存在也重下
  python scripts/download_yelp2018.py --keep-zip     # 保留 zip 不删
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
TARGET_DIR = os.path.join(DATA_ROOT, "yelp2018")
TARGET_INTER = os.path.join(TARGET_DIR, "yelp2018.inter")
ZIP_PATH = os.path.join(DATA_ROOT, "yelp2018.zip")

URL = "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yelp/yelp2018.zip"


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
            f"网络问题请手动下载 {URL} 后放到 {ZIP_PATH}，再重跑本脚本。"
        )


def _extract():
    if not os.path.isfile(ZIP_PATH):
        raise SystemExit(f"[extract] zip 不存在: {ZIP_PATH}")
    print(f"[extract] {ZIP_PATH} -> {DATA_ROOT}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = zf.namelist()
        zf.extractall(DATA_ROOT)

    # RecBole 官方 zip 里的顶层目录通常是 yelp2018/；若大小写或命名不同，做容错
    candidates = []
    for m in members:
        parts = m.split("/")
        if not parts or parts[0] == "":
            continue
        candidates.append(parts[0])
    top_dirs = set(candidates)
    if "yelp2018" not in top_dirs:
        # 尝试自动 rename 成 yelp2018
        for d in top_dirs:
            src = os.path.join(DATA_ROOT, d)
            if os.path.isdir(src) and any(
                f.endswith(".inter") for f in os.listdir(src)
            ):
                dst = TARGET_DIR
                if os.path.isdir(dst):
                    print(f"[extract] 目标目录已存在，先清空: {dst}")
                    shutil.rmtree(dst)
                print(f"[extract] 重命名 {src} -> {dst}")
                shutil.move(src, dst)
                break


def _verify():
    if not os.path.isfile(TARGET_INTER):
        # 容错：也许文件名首字母大小写不同
        found = None
        if os.path.isdir(TARGET_DIR):
            for f in os.listdir(TARGET_DIR):
                if f.lower() == "yelp2018.inter":
                    found = os.path.join(TARGET_DIR, f)
                    break
        if found and found != TARGET_INTER:
            print(f"[verify] 规范化文件名: {found} -> {TARGET_INTER}")
            shutil.move(found, TARGET_INTER)
        else:
            raise SystemExit(
                f"[verify FAIL] 未找到 {TARGET_INTER}。zip 解压内容可能和预期不同，"
                f"请检查 {TARGET_DIR} 目录。"
            )

    size_mb = os.path.getsize(TARGET_INTER) / 1024 / 1024
    with open(TARGET_INTER, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        n_lines = sum(1 for _ in f) + 1  # header + body
    print(f"[verify] {TARGET_INTER} ({size_mb:.1f} MB, {n_lines} 行)")
    print(f"         header: {header[:120]}{'...' if len(header) > 120 else ''}")

    # 列出目录下的其它 atomic 文件
    others = [
        f for f in os.listdir(TARGET_DIR)
        if f != "yelp2018.inter" and not f.startswith(".")
    ]
    if others:
        print(f"         其它文件: {', '.join(sorted(others))}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 RecBole 官方 yelp2018 atomic 数据集"
    )
    parser.add_argument("--force", action="store_true", help="已存在也重下")
    parser.add_argument("--keep-zip", action="store_true", help="保留 zip 文件")
    args = parser.parse_args()

    # 已经完整就跳过
    if os.path.isfile(TARGET_INTER) and not args.force:
        print(f"[skip] 数据已就绪: {TARGET_INTER}")
        _verify()
        return

    _download(force=args.force)
    _extract()
    _verify()

    if not args.keep_zip and os.path.isfile(ZIP_PATH):
        os.remove(ZIP_PATH)
        print(f"[cleanup] 删除 zip: {ZIP_PATH}")

    print("\n下一步：")
    print("  python scripts/prepare_yelp2018_80_20_split.py")


if __name__ == "__main__":
    main()
