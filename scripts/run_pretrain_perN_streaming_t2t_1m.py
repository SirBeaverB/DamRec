#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrain-lite + Streaming T2T on ml-1m (per-user 历史稀疏 regime)

目的：
  验证 paper 的 streaming Adam 主张 —— 当预训练期 per-user 历史极稀疏时，
  online streaming 阶段 Adam 式自适应更新应比 SGD（GDN）更 sample-efficient。

协议：
  1. 从 ml-1m-pretrain 派生 ml-1m-pretrain-perN：每 user 按 timestamp 保留最后 N 条真实交互，
     所有占位行（rating=0，用于 vocab 对齐）保留以免破坏 id 空间。
  2. 每个模型在 ml-1m-pretrain-perN 上独立完整预训练 150 epoch（非嫁接）。
  3. 预训练末 dump per-user (S, M, V) 状态 → user_states_{MODEL}.pt。
  4. ml-1m-t2t（原始后 20%）作 streaming T2T，按时间顺序 prequential 评估：
     到达 test 点先 predict 再含 test 的 batch 反向传播，user 的 S/M/V 持续演化。

双卡并行：{N} × {models} 任务矩阵排到 2 GPU，subprocess + CUDA_VISIBLE_DEVICES 隔离。

Usage:
  # 默认：N ∈ {5,10,20,50} × {GDN, Adam, Fro} = 12 jobs on 2 GPUs
  python scripts/run_pretrain_perN_streaming_t2t_1m.py

  # 只跑部分 N
  python scripts/run_pretrain_perN_streaming_t2t_1m.py --Ns 10,50

  # 只跑部分模型
  python scripts/run_pretrain_perN_streaming_t2t_1m.py --models Adam,GDN

  # 调 streaming lr
  python scripts/run_pretrain_perN_streaming_t2t_1m.py --t2t_lr 5e-5

  # 单卡
  CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain_perN_streaming_t2t_1m.py --n_gpus 1

  # 单任务调试（worker）
  CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain_perN_streaming_t2t_1m.py \\
      --worker --model Adam --N 10 --out_json /tmp/adam_n10.json
"""

import argparse
import glob
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

DATASET_ROOT = os.path.join(PROJ, "dataset")
PRETRAIN_SRC_DIR = os.path.join(DATASET_ROOT, "ml-1m-pretrain")
PRETRAIN_SRC_INTER = os.path.join(PRETRAIN_SRC_DIR, "ml-1m-pretrain.inter")

# 主表指标
METRIC_KEYS = ["recall@10", "ndcg@10", "mrr@10"]
METRIC_KEYS_FULL = ["recall@10", "ndcg@10", "mrr@10", "hit@10", "precision@10"]

DEFAULT_NS = [5, 10, 20, 50]
DEFAULT_MODELS = ["GDN", "Adam", "Fro"]


# ============================================================
# 数据预处理：派生 ml-1m-pretrain-perN
# ============================================================
def _build_perN_dataset(N):
    """生成 dataset/ml-1m-pretrain-perN/ml-1m-pretrain-perN.inter
    规则：
      - 读取 ml-1m-pretrain.inter
      - 真实交互（rating > 0）按 user 分组，按 timestamp 排序，保留最后 N 条
      - 占位行（rating == 0）全量保留，以维持 vocab 与原 pretrain 一致
      - .user / .item atomic files 复制
    返回 dataset_name。
    """
    if not os.path.isfile(PRETRAIN_SRC_INTER):
        raise SystemExit(f"未找到 {PRETRAIN_SRC_INTER}，请先 prepare_ml1m_80_20_split.py")

    dataset_name = f"ml-1m-pretrain-per{N}"
    dst_dir = os.path.join(DATASET_ROOT, dataset_name)
    dst_inter = os.path.join(dst_dir, f"{dataset_name}.inter")

    if os.path.isfile(dst_inter):
        return dataset_name

    os.makedirs(dst_dir, exist_ok=True)

    with open(PRETRAIN_SRC_INTER, "r", encoding="utf-8") as f:
        header = f.readline()
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    real_by_user = defaultdict(list)
    placeholders = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        uid, iid, rating, ts = parts[0], parts[1], parts[2], parts[3]
        try:
            rf = float(rating); tf = float(ts)
        except ValueError:
            continue
        if rf == 0.0:
            placeholders.append(ln)
        else:
            real_by_user[uid].append((tf, ln))

    truncated = []
    for uid, recs in real_by_user.items():
        recs.sort(key=lambda x: x[0])
        for _, ln in recs[-N:]:
            truncated.append(ln)

    with open(dst_inter, "w", encoding="utf-8") as f:
        f.write(header)
        for ln in truncated:
            f.write(ln + "\n")
        for ln in placeholders:
            f.write(ln + "\n")

    # 复制 .user / .item（RecBole atomic files）
    for suffix in [".user", ".item"]:
        src = os.path.join(PRETRAIN_SRC_DIR, "ml-1m-pretrain" + suffix)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, dataset_name + suffix))

    print(f"[prep] {dataset_name}: {len(truncated)} 真实 (per-user last {N}) + {len(placeholders)} 占位")
    return dataset_name


# ============================================================
# worker 模式：单 (model, N) 流水线
# ============================================================
def _worker_main(args):
    import run_pretrain_t2t_1m as pt1m
    from run_pretrain_t2t_1m import (
        MODEL_CONFIGS,
        _ensure_split,
        run_pretrain,
        run_state_dump,
        run_t2t_from_ckp,
    )

    model_key = args.model
    if model_key not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown --model {model_key}. Valid: {list(MODEL_CONFIGS.keys())}")
    model_name, config_file = MODEL_CONFIGS[model_key]

    N = int(args.N)
    dataset_name = _build_perN_dataset(N)

    # 关键：覆盖两处全局 OVERRIDES，让 run_pretrain/dump/t2t 都用 perN 子集
    pt1m.PRETRAIN_OVERRIDES["dataset"] = dataset_name
    pt1m.T2T_OVERRIDES["streaming_pretrain_dataset"] = dataset_name
    if args.epochs is not None:
        pt1m.PRETRAIN_OVERRIDES["epochs"] = int(args.epochs)
    if args.t2t_lr is not None:
        pt1m.T2T_OVERRIDES["learning_rate"] = float(args.t2t_lr)

    _ensure_split()

    ckp_dir = os.path.join(PROJ, "saved", f"per_user_pretrain_perN{N}_L{args.max_seq_len}")
    os.makedirs(ckp_dir, exist_ok=True)
    states_path = os.path.join(ckp_dir, f"user_states_{model_name}.pt")

    record = {
        "model": model_key,
        "model_name": model_name,
        "N": N,
        "dataset": dataset_name,
        "ckp_dir": ckp_dir,
        "ckp_path": None,
        "states_path": states_path,
        "result": {},
        "t_sec": None,
        "mem_gb": None,
        "pretrain_sec": None,
        "dump_sec": None,
        "pretrain_epochs": pt1m.PRETRAIN_OVERRIDES.get("epochs"),
        "t2t_lr": pt1m.T2T_OVERRIDES.get("learning_rate"),
        "seed": 2020,
        "stages_run": {"pretrain": False, "dump": False, "t2t": False},
    }

    def _save():
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    try:
        # Step A: pretrain on ml-1m-pretrain-perN
        if args.skip_pretrain:
            cands = sorted(glob.glob(os.path.join(ckp_dir, f"{model_name}-*.pth")))
            if not cands:
                raise RuntimeError(f"--skip_pretrain 需要 {ckp_dir}/{model_name}-*.pth 存在")
            ckp_path = cands[-1]
            print(f"[{model_key}/N={N}] Step A skipped, reuse {ckp_path}")
        else:
            print(f"[{model_key}/N={N}] Step A: pretrain on {dataset_name} "
                  f"({pt1m.PRETRAIN_OVERRIDES['epochs']} epochs, L={args.max_seq_len})")
            t0 = time.perf_counter()
            ckp_path = run_pretrain(
                model_name=model_name,
                config_file=config_file,
                ckp_dir=ckp_dir,
                show_progress=args.show_progress,
                max_seq_len=args.max_seq_len,
            )
            record["pretrain_sec"] = time.perf_counter() - t0
            record["stages_run"]["pretrain"] = True
            print(f"[{model_key}/N={N}] Step A done in {record['pretrain_sec']:.0f}s, ckp={ckp_path}")
        record["ckp_path"] = ckp_path
        _save()

        # Step B: dump user_states on perN
        if not args.skip_dump:
            if os.path.isfile(states_path) and not args.force_redump:
                print(f"[{model_key}/N={N}] Step B: reuse {states_path}")
            else:
                print(f"[{model_key}/N={N}] Step B: dump user_states")
                t0 = time.perf_counter()
                run_state_dump(
                    ckp_path=ckp_path,
                    save_path=states_path,
                    model_name=model_name,
                    config_file=config_file,
                    max_seq_len=args.max_seq_len,
                    show_progress=args.show_progress,
                )
                record["dump_sec"] = time.perf_counter() - t0
                record["stages_run"]["dump"] = True
                print(f"[{model_key}/N={N}] Step B done in {record['dump_sec']:.0f}s")

        # Step C: streaming T2T（t2t 数据集不变，仍是 ml-1m-t2t）
        if not args.skip_t2t:
            print(f"[{model_key}/N={N}] Step C: streaming T2T")
            result, t_sec, mem_gb = run_t2t_from_ckp(
                ckp_path=ckp_path,
                show_progress=args.show_progress,
                t2t_model=None,
                t2t_lr=args.t2t_lr,
                max_seq_len=args.max_seq_len,
                user_states_path=states_path if os.path.isfile(states_path) else None,
            )
            record["result"] = {
                k: (float(v) if v is not None else None) for k, v in (result or {}).items()
            }
            record["t_sec"] = float(t_sec) if t_sec is not None else None
            record["mem_gb"] = float(mem_gb) if mem_gb is not None else None
            record["stages_run"]["t2t"] = True
            print(f"[{model_key}/N={N}] Step C done: {record['result']}")
    finally:
        _save()


# ============================================================
# orchestrator：(model, N) 矩阵并行
# ============================================================
def _orchestrate(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(PROJ, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, f"per_user_pretrain_streaming_L{args.max_seq_len}_{run_id}")
    os.makedirs(work_dir, exist_ok=True)

    Ns = args.Ns or DEFAULT_NS
    models = args.models or DEFAULT_MODELS

    # 预先生成所有 perN 数据集（serial，只做 IO）
    for N in Ns:
        _build_perN_dataset(N)

    # 任务矩阵
    jobs = [(m, N) for N in Ns for m in models]
    print(f"[orchestrator] {len(jobs)} jobs = {len(models)} models × {len(Ns)} Ns, GPUs={args.n_gpus}")
    print(f"              Ns={Ns} models={models} work_dir={work_dir}")

    gpus = list(range(args.n_gpus))
    pending = list(jobs)
    running = {}  # gpu_id -> (job_key, Popen, out_json, log_fh, t0)
    results = {}  # (model_key, N) -> record dict

    def _launch(model_key, N, gpu_id):
        key = f"{model_key}_N{N}"
        out_json = os.path.join(work_dir, f"{key}.json")
        log_path = os.path.join(work_dir, f"{key}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker",
            "--model", model_key,
            "--N", str(N),
            "--out_json", out_json,
            "--max_seq_len", str(args.max_seq_len),
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.t2t_lr is not None:
            cmd += ["--t2t_lr", str(args.t2t_lr)]
        if args.show_progress:
            cmd += ["--show_progress"]
        if args.skip_pretrain:
            cmd += ["--skip_pretrain"]
        if args.skip_dump:
            cmd += ["--skip_dump"]
        if args.skip_t2t:
            cmd += ["--skip_t2t"]
        if args.force_redump:
            cmd += ["--force_redump"]
        log_fh = open(log_path, "w", buffering=1)
        print(f"[launch] {key} on GPU {gpu_id}  -> log={log_path}")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        running[gpu_id] = ((model_key, N), p, out_json, log_fh, time.time())

    while pending or running:
        for gpu_id in gpus:
            if gpu_id not in running and pending:
                mk, N = pending.pop(0)
                _launch(mk, N, gpu_id)
        time.sleep(10)
        done = []
        for gpu_id, ((mk, N), p, out_json, log_fh, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log_fh.close()
                elapsed = time.time() - t0
                key = (mk, N)
                if rc == 0 and os.path.isfile(out_json):
                    with open(out_json, encoding="utf-8") as f:
                        results[key] = json.load(f)
                    print(f"[done] {mk}/N={N} GPU {gpu_id} rc=0 elapsed={elapsed:.0f}s")
                else:
                    print(f"[FAIL] {mk}/N={N} GPU {gpu_id} rc={rc} elapsed={elapsed:.0f}s  log={out_json.replace('.json','.log')}")
                    if os.path.isfile(out_json):
                        try:
                            with open(out_json, encoding="utf-8") as f:
                                results[key] = json.load(f)
                        except Exception:
                            results[key] = None
                    else:
                        results[key] = None
                done.append(gpu_id)
        for g in done:
            del running[g]

    _aggregate(args, models, Ns, results, work_dir, run_id, out_dir)


# ============================================================
# 汇总
# ============================================================
def _aggregate(args, models, Ns, results, work_dir, run_id, out_dir):
    txt_path = os.path.join(out_dir, f"per_user_pretrain_streaming_L{args.max_seq_len}_{run_id}.txt")
    csv_path = os.path.join(out_dir, f"per_user_pretrain_streaming_L{args.max_seq_len}_{run_id}.csv")

    def fmt(v, width=None, prec=4):
        if v is None:
            s = "N/A"
        elif isinstance(v, float):
            s = f"{v:.{prec}f}"
        else:
            s = str(v)
        return s.ljust(width) if width else s

    any_ok = next((r for r in results.values() if r), None)
    seed = any_ok.get("seed", 2020) if any_ok else 2020

    label_w, col_w, delta_w = 10, 12, 14

    lines = []
    lines.append("=" * 100)
    lines.append("Pretrain-lite (per-user last N) + Streaming T2T on ml-1m")
    lines.append("=" * 100)
    lines.append("")
    lines.append("[实验情景]")
    lines.append("  从 ml-1m-pretrain 派生 ml-1m-pretrain-perN：每 user 按 timestamp 保留最后 N 条真实交互；")
    lines.append("  所有 rating=0 占位行保留以维持 vocab 一致。")
    lines.append("  每个模型在 perN 子集上独立预训练（150 epoch，非嫁接），末态 dump per-user (S,M,V)。")
    lines.append("  Streaming T2T 阶段在原始 ml-1m-t2t（最后 20% 真实交互）上 prequential 评估。")
    lines.append("")
    lines.append("[研究问题]")
    lines.append("  预训练期 per-user 历史越稀疏（N 越小），streaming 阶段 Adam 式 (DamRec/FroRec)")
    lines.append("  相对 SGD (GDN) 的优势是否越大？若 Δ(Adam-GDN) 随 N 递减单调上升，")
    lines.append("  则支撑 paper 的「streaming 场景下自适应预条件子在稀疏 per-user 数据下更 sample-efficient」论点。")
    lines.append("")
    lines.append("[评估协议] Prequential streaming (Gama et al., 2009)")
    lines.append("  每 user 的 t2t 尾部 10% 为 test 点：到达 test 点先 full_sort_predict → 记录 Recall/NDCG/MRR；")
    lines.append("  整个 batch（含 test 点）参与 loss.backward() 模拟在线反馈；per-user S/M/V 持续演化不重置。")
    lines.append("  与 per_model_streaming 脚本的区别：本实验 pretrain 与 streaming_pretrain_dataset 均为 perN，")
    lines.append("  即 streaming 阶段的 user_history 只含 pretrain 末尾 N 条，而非原 80%。")
    lines.append("")
    lines.append("[模型对照]")
    lines.append("  GDN  = Gated Delta Net   (一阶 SGD 基线)")
    lines.append("  Adam = DamRec             (Adam 式，秩一分解 V_r⊙V_k^T)")
    lines.append("  Fro  = FroRec             (F-Adam，V 降维为 Frobenius 标量)")
    lines.append("")
    lines.append("[超参数]")
    lines.append(f"  N levels              = {Ns}")
    lines.append(f"  models                = {models}")
    lines.append(f"  max_seq_len L         = {args.max_seq_len}")
    lines.append(f"  pretrain_epochs       = {args.epochs or 150}  (早停 stopping_step=10)")
    lines.append(f"  t2t_lr                = {args.t2t_lr or 1e-4}")
    lines.append(f"  t2t_test_ratio        = 0.1")
    lines.append(f"  random seed           = {seed}  (单 seed 单次运行)")
    lines.append("")
    lines.append("[运行信息]")
    lines.append(f"  run_id   = {run_id}")
    lines.append(f"  work_dir = {work_dir}")
    lines.append("")

    # ------------------------------------------------------------
    # TEST 主表：指标 × (模型, N) + Δ(Adam-GDN) + Δ(Fro-GDN)
    # ------------------------------------------------------------
    def _delta_pct(adv, base):
        if adv is None or base is None or base == 0:
            return None
        return (adv - base) / base * 100.0

    lines.append("-" * 100)
    lines.append("TEST (streaming T2T)  —  每指标按 N 排行，附 Δ% 相对 GDN")
    lines.append("-" * 100)

    for mk in METRIC_KEYS:
        lines.append(f"  Metric: {mk}")
        header = ("N".ljust(label_w)
                  + "".join(m.ljust(col_w) for m in models)
                  + "Δ(Adam-GDN)".ljust(delta_w)
                  + "Δ(Fro-GDN)".ljust(delta_w))
        lines.append(header)
        lines.append("-" * (label_w + col_w * len(models) + delta_w * 2))
        for N in Ns:
            row = f"N={N}".ljust(label_w)
            vals = {}
            for m in models:
                r = results.get((m, N))
                v = None if r is None else r.get("result", {}).get(mk)
                vals[m] = v
                row += fmt(v, col_w)
            d_adam = _delta_pct(vals.get("Adam"), vals.get("GDN"))
            d_fro = _delta_pct(vals.get("Fro"), vals.get("GDN"))
            row += (f"{d_adam:+.1f}%" if d_adam is not None else "N/A").ljust(delta_w)
            row += (f"{d_fro:+.1f}%" if d_fro is not None else "N/A").ljust(delta_w)
            lines.append(row)
        lines.append("")

    # ------------------------------------------------------------
    # 耗时 / 显存
    # ------------------------------------------------------------
    lines.append("-" * 100)
    lines.append("pretrain_sec / t2t_sec / peak_mem_gb")
    lines.append("-" * 100)

    for field, label, prec in [("pretrain_sec", "pretrain(s)", 0),
                                 ("t_sec",        "t2t(s)",      0),
                                 ("mem_gb",       "mem(GB)",    2)]:
        lines.append(f"  {label}")
        header = "N".ljust(label_w) + "".join(m.ljust(col_w) for m in models)
        lines.append(header)
        lines.append("-" * (label_w + col_w * len(models)))
        for N in Ns:
            row = f"N={N}".ljust(label_w)
            for m in models:
                r = results.get((m, N))
                v = None if r is None else r.get(field)
                row += fmt(v, col_w, prec=prec)
            lines.append(row)
        lines.append("")

    lines.append("[判读]")
    lines.append("  支撑假设（Adam sample-efficient in sparse per-user regime）：")
    lines.append("    N 越小 → Δ(Adam-GDN) 越大，且 Δ>0 稳定")
    lines.append("  若全部 Δ ≈ 0  → streaming 协议下 ml-1m 无法分辨方法，需换数据集（Amazon Beauty）")
    lines.append("  若 Δ 随机起伏 → 单 seed 噪声，需补 multi-seed 取均值")
    lines.append("  若 N=5/10 Δ 严重为负 → Adam 在超短序列下可能有 chunk/CHUNK_SIZE 边界问题，需 debug")
    lines.append("=" * 100)

    table = "\n".join(lines)
    print("\n" + table)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nTXT: {txt_path}")

    csv_cols = (["model", "N", "seed", "pretrain_sec", "t2t_sec", "peak_mem_gb"]
                + [f"test_{k}" for k in METRIC_KEYS_FULL]
                + ["ckp_path", "states_path"])
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for N in Ns:
            for m in models:
                r = results.get((m, N))
                if r is None:
                    f.write(f"{m},{N}," + ",".join(["N/A"] * (len(csv_cols) - 2)) + "\n")
                    continue
                parts = [m, str(N), str(r.get("seed") or "N/A")]
                parts.append(f"{r.get('pretrain_sec'):.2f}" if r.get("pretrain_sec") is not None else "N/A")
                parts.append(f"{r.get('t_sec'):.2f}" if r.get("t_sec") is not None else "N/A")
                parts.append(f"{r.get('mem_gb'):.2f}" if r.get("mem_gb") is not None else "N/A")
                for k in METRIC_KEYS_FULL:
                    v = r.get("result", {}).get(k)
                    parts.append(f"{v:.4f}" if v is not None else "N/A")
                parts.append(str(r.get("ckp_path") or ""))
                parts.append(str(r.get("states_path") or ""))
                f.write(",".join(parts) + "\n")
    print(f"CSV: {csv_path}")


# ============================================================
# CLI
# ============================================================
def _parse_csv_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain-lite (per-user last N) + Streaming T2T on ml-1m，双卡并行"
    )
    parser.add_argument("--Ns", type=_parse_csv_int, default=None,
                        help=f"每 user 保留的历史长度，逗号分隔，默认 {DEFAULT_NS}")
    parser.add_argument("--models", type=_parse_csv_str, default=None,
                        help=f"模型子集，逗号分隔，默认 {DEFAULT_MODELS} (Adam=DamRec, Fro=FroRec)")
    parser.add_argument("-L", "--max_seq_len", type=int, default=64,
                        help="序列长度 L，默认 64（CHUNK_SIZE=16 兼容）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="预训练最大 epoch，默认 150")
    parser.add_argument("--t2t_lr", type=float, default=None,
                        help="streaming T2T 学习率，默认 1e-4")
    parser.add_argument("--n_gpus", type=int, default=2,
                        help="并行 GPU 数，默认 2")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)

    # 断点续跑
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--skip_dump", action="store_true")
    parser.add_argument("--skip_t2t", action="store_true")
    parser.add_argument("--force_redump", action="store_true")

    # Worker
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()

    if args.worker:
        if args.model is None or args.N is None or args.out_json is None:
            raise SystemExit("--worker 模式需要 --model --N --out_json")
        _worker_main(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
