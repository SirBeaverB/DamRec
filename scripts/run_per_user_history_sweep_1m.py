#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-user history length sweep on ml-1m (non-streaming, leave-one-out).

研究问题：当每个 user 的可用历史变短时，Adam-style 优化器（DamRec/FroRec）相对
SGD 基线（GDN）的优势是否单调放大？验证 'sample-efficient online adaptation'
假设——若成立，可作为论文主图（横轴 N，纵轴 Δ recall vs GDN）。

实验设计：
  对每个 N ∈ {10, 50, 100}：
    - 预处理 ml-1m → ml-1m-perN.inter，每 user 按 timestamp 保留**最近 N 条**交互
    - 用 RecBole 标准 leave-one-out 训练（每 user 最后 1 个为 test）
    - 跑 GDN / DamRec / FroRec 三个模型
  共 9 (model, N) 任务，双卡并行。

数据集：
  ml-1m 每 user 平均 165 交互，最少 20 交互，所以 N=10 仍保留所有 user。
  对每 user：按 timestamp 排序，取 last N（模拟"用户只有最近 N 条历史"的冷启动）。

序列长度 L 与 N 的关系：
  L 固定为 64（项目默认，CHUNK_SIZE=16 兼容）。
  N=10 → 模型最长输入 9 个 item（L 不绑定）
  N=50 → 模型最长输入 49 个 item（L 不绑定）
  N=100 → 模型最长输入 64 个 item（L 绑定，取 last 64）

Usage:
  # 默认：N=[10,50,100]，model=[GDN,Adam,Fro]，双卡，L=64，150 ep
  python scripts/run_per_user_history_sweep_1m.py

  # 自定义 N
  python scripts/run_per_user_history_sweep_1m.py --Ns 5,20,50

  # 单卡顺序
  python scripts/run_per_user_history_sweep_1m.py --n_gpus 1

  # smoke test
  python scripts/run_per_user_history_sweep_1m.py --Ns 10 --models GDN --epochs 3

  # 单 (model, N) 调试
  CUDA_VISIBLE_DEVICES=0 python scripts/run_per_user_history_sweep_1m.py \\
      --worker --model Adam --N 50 --out_json /tmp/adam_n50.json --show_progress

  # 只可见两张卡时（与 run_per_model_pretrain_t2t_1m 一致，子进程会各绑一张）
  CUDA_VISIBLE_DEVICES=0,1 python scripts/run_per_user_history_sweep_1m.py --n_gpus 2

日志与产物（每次 run 在 experiment_results/per_user_hist_sweep_L{timestamp}/）：
  00_manifest.txt   — 参数、host CUDA、slot 映射
  orchestrate.log   — 调度器 [launch]/[done]/[FAIL] 全量落盘
  {GDN,Adam,Fro}_N{N}.log — 单任务 RecBole stdout/stderr
  {Model}_N{N}_cmd.txt — 复现用命令行与 CUDA_VISIBLE_DEVICES
  汇总表仍写出到 experiment_results/ 下 .txt 与 .csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(_script_dir)
sys.path.insert(0, PROJ)
sys.path.insert(0, _script_dir)

# 用户面 CLI key（与 README 习惯保持一致：Adam = DamRec）
DEFAULT_MODELS = ["GDN", "Adam", "Fro"]
DEFAULT_NS = [10, 50, 100]
METRIC_KEYS = ["recall@10", "ndcg@10", "mrr@10"]
METRIC_KEYS_FULL = ["recall@10", "ndcg@10", "mrr@10", "hit@10", "precision@10"]

# CLI key -> (RecBole 模型类名, yaml 路径)
MODEL_TABLE = {
    "GDN":  ("GDN",     "recbole/properties/quick_start_config/sequential_GDN.yaml"),
    "Mo":   ("MoRec",   "recbole/properties/quick_start_config/sequential_MoRec.yaml"),
    "Nest": ("NestRec", "recbole/properties/quick_start_config/sequential_NestRec.yaml"),
    "Adam": ("DamRec",  "recbole/properties/quick_start_config/sequential_DamRec.yaml"),
    "Dam":  ("DamRec",  "recbole/properties/quick_start_config/sequential_DamRec.yaml"),
    "Fro":  ("FroRec",  "recbole/properties/quick_start_config/sequential_FroRec.yaml"),
}


# ---------------------------------------------------------------- GPU / 工作目录 辅助

def _visible_gpu_ids_for_children(n_wanted: int):
    """
    与 run_per_model_pretrain_t2t_1m 等双卡脚本一致：若父进程已设置
    CUDA_VISIBLE_DEVICES=2,3，则子进程应各自独占其中一张，而不是再写 0,1
    映到宿主机物理 GPU 0,1。

    返回长度 <= n_wanted 的字符串列表，与 subprocess 的 CUDA_VISIBLE_DEVICES 一一对应。
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [str(i) for i in range(n_wanted)]
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        return [str(i) for i in range(n_wanted)]
    if len(parts) < n_wanted:
        print(
            f"[orchestrator] 警告: CUDA_VISIBLE_DEVICES 仅有 {len(parts)} 张"
            f"（{raw}），请求 n_gpus={n_wanted}，将只用前 {len(parts)} 个 slot"
        )
    return parts[:n_wanted]


# ---------------------------------------------------------------- 数据预处理

def _ensure_truncated_dataset(N):
    """创建 ml-1m-perN 数据集：每 user 按时间保留 last N 条交互。"""
    src_dir = os.path.join(PROJ, "dataset", "ml-1m")
    src = os.path.join(src_dir, "ml-1m.inter")
    dst_dir = os.path.join(PROJ, "dataset", f"ml-1m-per{N}")
    dst = os.path.join(dst_dir, f"ml-1m-per{N}.inter")

    if os.path.isfile(dst):
        return dst

    if not os.path.isfile(src):
        raise FileNotFoundError(f"源数据 {src} 不存在")

    print(f"[preprocess] N={N}: 创建 ml-1m-per{N} 数据集 ...")
    os.makedirs(dst_dir, exist_ok=True)

    with open(src, "r", encoding="utf-8") as f:
        header = f.readline()
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            uid, iid, rating, ts = parts[0], parts[1], parts[2], parts[3]
            rows.append((uid, iid, rating, ts))

    by_user = defaultdict(list)
    for uid, iid, rating, ts in rows:
        by_user[uid].append((float(ts), iid, rating))

    out_rows = []
    n_users_kept = 0
    n_users_dropped = 0
    for uid, lst in by_user.items():
        lst.sort()
        keep = lst[-N:]
        # leave-one-out 至少需要 3 条交互（train + valid + test）
        if len(keep) < 3:
            n_users_dropped += 1
            continue
        n_users_kept += 1
        for ts, iid, rating in keep:
            out_rows.append((uid, iid, rating, ts))

    out_rows.sort(key=lambda r: float(r[3]))

    with open(dst, "w", encoding="utf-8") as f:
        f.write(header)
        for uid, iid, rating, ts in out_rows:
            f.write(f"{uid}\t{iid}\t{rating}\t{ts}\n")

    # 复制 .item 和 .user（RecBole 标准目录结构）
    for ext in [".item", ".user"]:
        src_f = os.path.join(src_dir, f"ml-1m{ext}")
        dst_f = os.path.join(dst_dir, f"ml-1m-per{N}{ext}")
        if os.path.isfile(src_f) and not os.path.isfile(dst_f):
            shutil.copy2(src_f, dst_f)

    avg = len(out_rows) / max(1, n_users_kept)
    print(f"[preprocess] N={N}: kept {n_users_kept} users / dropped {n_users_dropped} (interactions<3), "
          f"total {len(out_rows)} rows, avg={avg:.1f} per user")
    return dst


# ---------------------------------------------------------------- worker 模式

def _worker_main(args):
    """跑单个 (model, N) 任务，写结果 JSON。"""
    from run_non_streaming_experiments import run_single_model

    if args.model not in MODEL_TABLE:
        raise SystemExit(f"Unknown --model {args.model}, valid: {list(MODEL_TABLE)}")
    model_name, config_file = MODEL_TABLE[args.model]

    _ensure_truncated_dataset(args.N)
    dataset_name = f"ml-1m-per{args.N}"

    ckp_dir = os.path.join(PROJ, "saved", f"per_user_hist_sweep_N{args.N}_L{args.max_seq_len}")
    os.makedirs(ckp_dir, exist_ok=True)

    record = {
        "model": args.model,
        "model_name": model_name,
        "N": args.N,
        "dataset": dataset_name,
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "ckp_dir": ckp_dir,
        "valid_result": {},
        "test_result": {},
        "train_time_sec": None,
        "peak_mem_gb": None,
        "seed": 2020,
    }

    def _save():
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    _save()  # 落盘一次空骨架，方便外层失败时仍可读

    try:
        ret = run_single_model(
            model_key=model_name,
            config_file=config_file,
            dataset=dataset_name,
            max_seq_len=args.max_seq_len,
            epochs=args.epochs,
            worker=4,
            saved=False,
            show_progress=args.show_progress,
            checkpoint_dir=ckp_dir,
        )
        if ret is None:
            raise RuntimeError("run_single_model returned None (内部已 traceback)")
        valid_res, test_res, train_time, peak_mem = ret
        record["valid_result"] = {k: float(v) for k, v in (valid_res or {}).items() if v is not None}
        record["test_result"] = {k: float(v) for k, v in (test_res or {}).items() if v is not None}
        record["train_time_sec"] = float(train_time) if train_time is not None else None
        record["peak_mem_gb"] = float(peak_mem) if peak_mem is not None else None
    finally:
        _save()


# ------------------------------------------------------------ orchestrator 模式

def _orchestrate(args):
    if args.n_gpus < 1:
        raise SystemExit("--n_gpus 必须 >= 1")

    # 预先创建所有数据集（避免 worker 并发触发同一份预处理 race condition）
    for N in args.Ns:
        _ensure_truncated_dataset(N)

    out_dir = args.output_dir or os.path.join(PROJ, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(out_dir, f"per_user_hist_sweep_L{args.max_seq_len}_{run_id}")
    os.makedirs(work_dir, exist_ok=True)

    olog_path = os.path.join(work_dir, "orchestrate.log")
    olog = open(olog_path, "w", buffering=1)

    def oprint(*a, **k):
        print(*a, **k, flush=True)
        print(*a, file=olog, **k, flush=True)

    try:
        slot_cuda_ids = _visible_gpu_ids_for_children(args.n_gpus)
        n_slots = len(slot_cuda_ids)
        args._slot_cuda_ids = slot_cuda_ids
        args._n_slots = n_slots
        args._orchestrate_log = olog_path
        args._manifest_path = None

        if n_slots < args.n_gpus:
            oprint(
                f"[orchestrator] 将 n_gpus 从 {args.n_gpus} 减为 {n_slots}"
                f"（与可见 GPU 列表一致: {slot_cuda_ids}）"
            )

        manifest = os.path.join(work_dir, "00_manifest.txt")
        args._manifest_path = manifest
        with open(manifest, "w", encoding="utf-8") as mf:
            mf.write("=== Per-user history sweep (ml-1m) ===\n")
            mf.write(f"start (local)  = {datetime.now().isoformat()}\n")
            mf.write(f"work_dir      = {work_dir}\n")
            mf.write(f"run_id        = {run_id}\n")
            mf.write(f"argv          = {sys.argv}\n")
            mf.write(
                f"host CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n"
            )
            mf.write(f"slot GPUs (per worker)  = {slot_cuda_ids}\n")
            mf.write(f"Ns            = {args.Ns}\n")
            mf.write(f"models        = {args.models}\n")
            mf.write(
                f"L, epochs, n_slots = {args.max_seq_len}, {args.epochs}, {n_slots}\n"
            )
            mf.write(
                f"per-task logs: {os.path.join(work_dir, '<MODEL>_N<N>.log')}\n"
            )
            mf.write(f"orchestrate log: {olog_path}\n")
            mf.write(
                f"summary txt/csv: per_user_hist_sweep_L{args.max_seq_len}_{run_id}.(txt|csv) under {out_dir}\n"
            )

        # 任务列表：先按 N 分组，每组按模型顺序——便于按 N 完成时早做局部分析
        jobs = [(m, N) for N in args.Ns for m in args.models]
        oprint(
            f"[orchestrator] {len(jobs)} jobs ({len(args.Ns)} Ns × {len(args.models)} models)"
            f" on {n_slots} parallel slot(s)"
        )
        oprint(f"[orchestrator] work_dir={work_dir}")
        oprint(f"[orchestrator] manifest -> {manifest}")

        pending = list(jobs)
        running = {}   # slot_idx -> (job, Popen, out_json, log_fh, t0, cuda_id_str)
        results = {}
        gpus = list(range(n_slots))  # slot 索引 0..n-1，映到 slot_cuda_ids[slot]

        def _launch(model_key, N, slot_idx):
            cuda_id = slot_cuda_ids[slot_idx]
            out_json = os.path.join(work_dir, f"{model_key}_N{N}.json")
            log_path = os.path.join(work_dir, f"{model_key}_N{N}.log")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker",
                "--model", model_key,
                "--N", str(N),
                "--out_json", out_json,
                "--max_seq_len", str(args.max_seq_len),
                "--epochs", str(args.epochs),
            ]
            if args.show_progress:
                cmd += ["--show_progress"]
            log_fh = open(log_path, "w", buffering=1)
            cmd_txt = os.path.join(work_dir, f"{model_key}_N{N}_cmd.txt")
            with open(cmd_txt, "w", encoding="utf-8") as cf:
                cf.write(" ".join(
                    f'"{c}"' if " " in c else c for c in cmd
                ) + "\n")
                cf.write(
                    f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
                )
            oprint(
                f"[launch] {model_key}@N={N} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"
                f" (slot {slot_idx}) -> {log_path}"
            )
            p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            running[slot_idx] = ((model_key, N), p, out_json, log_fh, time.time(), str(cuda_id))

        while pending or running:
            for slot_idx in gpus:
                if slot_idx not in running and pending:
                    model_key, N = pending.pop(0)
                    _launch(model_key, N, slot_idx)
            time.sleep(10)
            done_slots = []
            for slot_idx, (job, p, out_json, log_fh, t0, cuda_s) in list(running.items()):
                rc = p.poll()
                if rc is not None:
                    log_fh.close()
                    elapsed = time.time() - t0
                    model_key, N = job
                    if rc == 0 and os.path.isfile(out_json):
                        try:
                            with open(out_json, encoding="utf-8") as f:
                                results[job] = json.load(f)
                            oprint(
                                f"[done] {model_key}@N={N} (CUDA {cuda_s}, slot {slot_idx})"
                                f" elapsed={elapsed:.0f}s"
                            )
                        except Exception as e:
                            oprint(
                                f"[FAIL-PARSE] {model_key}@N={N} (slot {slot_idx}): {e}"
                            )
                            results[job] = None
                    else:
                        oprint(
                            f"[FAIL] {model_key}@N={N} (slot {slot_idx}) rc={rc}"
                            f" elapsed={elapsed:.0f}s, see per-task .log"
                        )
                        if os.path.isfile(out_json):
                            try:
                                with open(out_json, encoding="utf-8") as f:
                                    results[job] = json.load(f)
                            except Exception:
                                results[job] = None
                        else:
                            results[job] = None
                    done_slots.append(slot_idx)
            for s in done_slots:
                del running[s]

        _aggregate(args, results, work_dir, run_id, out_dir)
    finally:
        olog.close()


# ---------------------------------------------------------------- 汇总输出

def _aggregate(args, results, work_dir, run_id, out_dir):
    txt_path = os.path.join(out_dir, f"per_user_hist_sweep_L{args.max_seq_len}_{run_id}.txt")
    csv_path = os.path.join(out_dir, f"per_user_hist_sweep_L{args.max_seq_len}_{run_id}.csv")

    Ns = args.Ns
    models = args.models

    label_w, col_w = 12, 11

    def fmt(v, w=col_w, prec=4, suffix=""):
        if v is None:
            return "N/A".ljust(w)
        if isinstance(v, float):
            return (f"{v:.{prec}f}{suffix}").ljust(w)
        return str(v).ljust(w)

    lines = []
    lines.append("=" * 100)
    lines.append("Per-User History Length Sweep on ml-1m  (non-streaming, leave-one-out)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("[实验情景]")
    lines.append("  对每个 N ∈ {10, 50, 100} 创建 ml-1m-perN 子集：每 user 按 timestamp 保留 last N 条交互。")
    lines.append("  RecBole 标准 leave-one-out 划分（每 user 最后 1 个 = test，倒数第 2 = valid，其余 = train）。")
    lines.append("  五模型简化为 3 个对照：GDN（SGD 基线）/ Adam=DamRec（逐维度 Adam）/ Fro=FroRec（F-Adam）。")
    lines.append("")
    lines.append("[研究问题]")
    lines.append("  per-user 历史长度变短时，Adam-family 相对 SGD 的优势 Δ 是否随 N 减小而单调放大？")
    lines.append("  若成立 → 主图：横轴 N，纵轴 Δ%(Adam-GDN) / Δ%(Fro-GDN)，证实 sample-efficient claim。")
    lines.append("")
    lines.append("[超参数]")
    lines.append(f"  N levels       = {Ns}")
    lines.append(f"  models         = {models}")
    lines.append(f"  max_seq_len L  = {args.max_seq_len}  (固定，CHUNK_SIZE=16 兼容；N=100 时 L 绑定取 last 64)")
    lines.append(f"  epochs         = {args.epochs}  (受 stopping_step=10 早停控制)")
    lines.append(f"  random seed    = 2020  (RecBole default; 单 seed 单次运行)")
    lines.append("")
    lines.append("[运行信息]")
    lines.append(f"  run_id     = {run_id}")
    lines.append(f"  work_dir   = {work_dir}")
    n_sl = getattr(args, "_n_slots", None)
    sids = getattr(args, "_slot_cuda_ids", None)
    if n_sl is not None and sids is not None:
        lines.append(f"  parallel   = {n_sl} slot(s), CUDA_VISIBLE_DEVICES per worker = {sids}")
    else:
        lines.append(f"  n_gpus     = {args.n_gpus}")
    man = getattr(args, "_manifest_path", None)
    olp = getattr(args, "_orchestrate_log", None)
    if man:
        lines.append(f"  manifest   = {man}")
    if olp:
        lines.append(f"  orchestrate.log = {olp}")
    lines.append("")

    # 主表 + Δ 列
    for split_name, split_key in [("VALID", "valid_result"), ("TEST", "test_result")]:
        lines.append("-" * 100)
        lines.append(f"{split_name}  —  Recall / NDCG / MRR @10  (越大越好)，附 Δ% 相对 GDN")
        lines.append("-" * 100)
        for mk in METRIC_KEYS:
            lines.append(f"  Metric: {mk}")
            header = "N".ljust(label_w) + "".join(m.ljust(col_w) for m in models)
            for tgt in ["Adam", "Fro"]:
                if tgt in models and "GDN" in models:
                    header += f"Δ({tgt}-GDN)".ljust(col_w)
            lines.append(header)
            lines.append("-" * len(header))
            for N in Ns:
                row = f"N={N}".ljust(label_w)
                vals = {}
                for m in models:
                    r = results.get((m, N))
                    v = r.get(split_key, {}).get(mk) if r else None
                    vals[m] = v
                    row += fmt(v)
                gdn = vals.get("GDN")
                for tgt in ["Adam", "Fro"]:
                    if tgt in models and "GDN" in models:
                        v = vals.get(tgt)
                        if v is not None and gdn not in (None, 0):
                            row += f"{(v - gdn) / gdn * 100:+.1f}%".ljust(col_w)
                        else:
                            row += "N/A".ljust(col_w)
                lines.append(row)
            lines.append("")

    # 训练耗时 / 显存
    lines.append("-" * 100)
    lines.append("训练耗时 (s)  /  峰值显存 (GB)")
    lines.append("-" * 100)
    header = "N".ljust(label_w) + "".join(m.ljust(col_w) for m in models)
    lines.append(header)
    lines.append("-" * len(header))
    for N in Ns:
        row = f"N={N}".ljust(label_w)
        for m in models:
            r = results.get((m, N))
            v = r.get("train_time_sec") if r else None
            row += fmt(v, prec=0)
        lines.append(row)
    lines.append("")
    for N in Ns:
        row = f"N={N}".ljust(label_w)
        for m in models:
            r = results.get((m, N))
            v = r.get("peak_mem_gb") if r else None
            row += fmt(v, prec=2)
        lines.append(row)
    lines.append("")

    lines.append("[判读]")
    lines.append("  TEST 表格里 Δ(Adam-GDN) 和 Δ(Fro-GDN) 是关键。预期：")
    lines.append("    N=100: Δ ≈ +3% ~ +5%   (per-user 数据较多，SGD 已够用)")
    lines.append("    N=50 : Δ ≈ +10% ~ +15%")
    lines.append("    N=10 : Δ ≈ +20% ~ +40% (per-user 极稀疏，Adam 优势放大)")
    lines.append("  若三档单调上升 → sample-efficient 假设成立，可作论文主图。")
    lines.append("  若三档随机起伏 → 单 seed 噪声主导，需补 multi-seed。")
    lines.append("=" * 100)

    table = "\n".join(lines)
    print("\n" + table)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nTXT: {txt_path}")

    # CSV：每行一个 (N, model)，含 valid + test 全 5 指标
    csv_cols = (["N", "model", "model_name", "seed"]
                + [f"valid_{mk}" for mk in METRIC_KEYS_FULL]
                + [f"test_{mk}" for mk in METRIC_KEYS_FULL]
                + ["train_time_sec", "peak_mem_gb"])
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for N in Ns:
            for m in models:
                r = results.get((m, N))
                if r is None:
                    f.write(f"{N},{m},N/A,N/A," + ",".join(["N/A"] * (len(METRIC_KEYS_FULL) * 2 + 2)) + "\n")
                    continue
                parts = [str(N), m, r.get("model_name", "N/A"), str(r.get("seed", 2020))]
                for mk in METRIC_KEYS_FULL:
                    v = r.get("valid_result", {}).get(mk)
                    parts.append(f"{v:.4f}" if v is not None else "N/A")
                for mk in METRIC_KEYS_FULL:
                    v = r.get("test_result", {}).get(mk)
                    parts.append(f"{v:.4f}" if v is not None else "N/A")
                parts.append(f"{r.get('train_time_sec'):.2f}" if r.get("train_time_sec") is not None else "N/A")
                parts.append(f"{r.get('peak_mem_gb'):.2f}" if r.get("peak_mem_gb") is not None else "N/A")
                f.write(",".join(parts) + "\n")
    print(f"CSV: {csv_path}")


# --------------------------------------------------------------- CLI 入口

def main():
    parser = argparse.ArgumentParser(
        description="Per-user history length sweep on ml-1m，双卡并行")
    parser.add_argument("--Ns", type=str, default="10,50,100",
                        help="Comma-separated N values, default 10,50,100")
    parser.add_argument("--models", type=str, default="GDN,Adam,Fro",
                        help="Comma-separated model keys, default GDN,Adam,Fro (Adam = DamRec)")
    parser.add_argument("-L", "--max_seq_len", type=int, default=64,
                        help="序列长度 L，默认 64（CHUNK_SIZE=16 兼容）")
    parser.add_argument("--epochs", type=int, default=150,
                        help="预训练最大 epoch，默认 150（受 stopping_step=10 早停）")
    parser.add_argument("--n_gpus", type=int, default=2,
                        help="并行 GPU 数，默认 2；每任务独占一卡")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", default=None)

    parser.add_argument("--worker", action="store_true", help="内部 worker 模式")
    parser.add_argument("--model", type=str, default=None, help="worker：模型 key")
    parser.add_argument("--N", type=int, default=None, help="worker：每用户保留交互数")
    parser.add_argument("--out_json", type=str, default=None, help="worker：结果 JSON 路径")

    args = parser.parse_args()

    args.Ns = [int(n) for n in args.Ns.split(",") if n.strip()]
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in args.models if m not in MODEL_TABLE]
    if bad:
        raise SystemExit(f"Unknown models: {bad}. Valid: {list(MODEL_TABLE)}")

    if args.worker:
        if not args.model or args.N is None or not args.out_json:
            raise SystemExit("--worker 模式需要 --model, --N, --out_json")
        _worker_main(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
