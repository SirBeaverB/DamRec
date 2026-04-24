#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每模型独立预训练 + state_dump + 流式 T2T（ml-1m 80/20）。

与 run_gdn_pretrain_t2t_unified.py / run_t2t_from_ckp_unified.py 的关键区别：
  - **不嫁接**：5 个模型各自在 ml-1m-pretrain 上完整预训练 150 epoch，每个模型加载
    自己的 checkpoint（MoRec/NestRec/DamRec/FroRec 的独有参数不再是随机初始化）
  - **强制 dump_state**：每个模型预训练完立即导出 user_states_{MODEL}.pt，T2T 阶段
    per-user 内部状态 (S, M, V) 以预训练末态为起点，不从零开始吸收

双卡并行：每个模型独占一张卡跑完整流水线 (pretrain -> dump -> t2t)。
5 个模型在 2 张卡上排两波 (ceil(5/2)=3 轮 slot)，平均约 2.5h。

Usage:
  # 全量（默认 2 GPU、5 模型、L=64、epochs=150）
  python scripts/run_per_model_pretrain_t2t_1m.py

  # 指定 L 与 GPU 数
  python scripts/run_per_model_pretrain_t2t_1m.py -L 64 --n_gpus 2

  # 只跑部分模型
  python scripts/run_per_model_pretrain_t2t_1m.py --models Adam,Fro

  # 调参：预训练 epoch / 流式 lr
  python scripts/run_per_model_pretrain_t2t_1m.py --epochs 150 --t2t_lr 1e-4

  # 断点续跑（按需组合）
  python scripts/run_per_model_pretrain_t2t_1m.py --skip_pretrain            # 用已有 ckp
  python scripts/run_per_model_pretrain_t2t_1m.py --skip_pretrain --skip_dump # 只重跑 T2T

  # 单模型调试（手动指定 GPU，worker 模式）
  CUDA_VISIBLE_DEVICES=0 python scripts/run_per_model_pretrain_t2t_1m.py \\
      --worker --model Adam --out_json /tmp/adam.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)

MODEL_COLS = ["GDN", "Mo", "Nest", "Adam", "Fro"]
# 主表指标：Recall → NDCG → MRR（信息量从大到小）
# 删掉 hit@10（= recall@10）与 precision@10（= recall@10/10）——leave-one-out 下是冗余
METRIC_KEYS = ["recall@10", "ndcg@10", "mrr@10"]
# 完整指标（写入 CSV 备查，TXT 不展示）
METRIC_KEYS_FULL = ["recall@10", "ndcg@10", "mrr@10", "hit@10", "precision@10"]


# ---------------------------------------------------------------- worker 模式
# 单模型流水线：在 1 张 GPU 上跑 pretrain -> state_dump -> T2T，结果写 JSON。
# Orchestrator 通过 subprocess 调用此分支，每调用一次 = 一个模型占一张卡。

def _worker_main(args):
    # 延迟导入，避免 orchestrator 模式也吃 torch/recbole 冷启动
    import run_pretrain_t2t_1m as pt1m
    from run_pretrain_t2t_1m import (
        MODEL_CONFIGS,
        _ensure_split,
        run_pretrain,
        run_state_dump,
        run_t2t_from_ckp,
    )

    # 覆盖预训练 epoch（run_pretrain 没有显式 epochs 形参，通过 module-level dict）
    if args.epochs is not None:
        pt1m.PRETRAIN_OVERRIDES["epochs"] = int(args.epochs)

    model_key = args.model
    if model_key not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown --model {model_key}. Valid: {list(MODEL_CONFIGS.keys())}")
    model_name, config_file = MODEL_CONFIGS[model_key]

    _ensure_split()

    proj = os.path.dirname(_script_dir)
    ckp_dir = os.path.join(proj, "saved", f"per_model_pretrain_L{args.max_seq_len}")
    os.makedirs(ckp_dir, exist_ok=True)
    states_path = os.path.join(ckp_dir, f"user_states_{model_name}.pt")

    result_record = {
        "model": model_key,
        "model_name": model_name,
        "ckp_dir": ckp_dir,
        "ckp_path": None,
        "states_path": states_path,
        "pretrain_valid": {},   # pretrain 子集上常规划分 best valid（sequential 离线）
        "pretrain_test": {},    # pretrain 子集上 test（与 T2T 的 result 不同分布）
        "result": {},           # streaming T2T on 后 20%
        "t_sec": None,
        "mem_gb": None,
        "pretrain_epochs": pt1m.PRETRAIN_OVERRIDES.get("epochs"),
        "pretrain_sec": None,   # 预训练实际耗时（skip 时为 None）
        "dump_sec": None,
        "seed": 2020,           # RecBole 默认 seed（recbole/properties/overall.yaml）
        "stages_run": {"pretrain": False, "dump": False, "t2t": False},
    }

    def _save_and_maybe_exit():
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result_record, f, indent=2, ensure_ascii=False)

    try:
        # Step A: 预训练（或找已有 ckp）
        if args.skip_pretrain:
            cands = sorted(glob.glob(os.path.join(ckp_dir, f"{model_name}-*.pth")))
            if not cands:
                raise RuntimeError(
                    f"--skip_pretrain 需要 {ckp_dir}/{model_name}-*.pth 存在"
                )
            ckp_path = cands[-1]
            print(f"[{model_key}] Step A skipped, reuse {ckp_path}")
        else:
            print(f"[{model_key}] Step A: pretrain on ml-1m-pretrain ({pt1m.PRETRAIN_OVERRIDES['epochs']} epochs, L={args.max_seq_len})")
            _t0 = time.perf_counter()
            ckp_path, pv, pt = run_pretrain(
                model_name=model_name,
                config_file=config_file,
                ckp_dir=ckp_dir,
                show_progress=args.show_progress,
                max_seq_len=args.max_seq_len,
            )
            result_record["pretrain_sec"] = time.perf_counter() - _t0
            result_record["pretrain_valid"] = pv or {}
            result_record["pretrain_test"] = pt or {}
            result_record["stages_run"]["pretrain"] = True
            print(f"[{model_key}] Step A done in {result_record['pretrain_sec']:.0f}s, ckp={ckp_path}")
            print(
                f"[{model_key}]  pretrain 子集 valid@10={result_record['pretrain_valid'].get('recall@10', 'N/A')}, "
                f"test@10={result_record['pretrain_test'].get('recall@10', 'N/A')}"
            )
        result_record["ckp_path"] = ckp_path
        _save_and_maybe_exit()

        # Step B: dump per-user S/M/V
        if args.skip_dump:
            if not os.path.isfile(states_path):
                print(f"[{model_key}] Step B skipped, but {states_path} missing -> T2T will start with state=0")
        else:
            if os.path.isfile(states_path) and not args.force_redump:
                print(f"[{model_key}] Step B: reuse existing {states_path}")
            else:
                print(f"[{model_key}] Step B: dump user_states -> {states_path}")
                _t0 = time.perf_counter()
                run_state_dump(
                    ckp_path=ckp_path,
                    save_path=states_path,
                    model_name=model_name,
                    config_file=config_file,
                    max_seq_len=args.max_seq_len,
                    show_progress=args.show_progress,
                )
                result_record["dump_sec"] = time.perf_counter() - _t0
                result_record["stages_run"]["dump"] = True
                print(f"[{model_key}] Step B done in {result_record['dump_sec']:.0f}s")

        # Step C: T2T streaming（不传 t2t_model -> 走非嫁接路径）
        if args.skip_t2t:
            print(f"[{model_key}] Step C skipped")
        else:
            print(f"[{model_key}] Step C: streaming T2T from own checkpoint")
            result, t_sec, mem_gb = run_t2t_from_ckp(
                ckp_path=ckp_path,
                show_progress=args.show_progress,
                t2t_model=None,  # 关键：让 hot_swap=False
                t2t_lr=args.t2t_lr,
                max_seq_len=args.max_seq_len,
                user_states_path=states_path if os.path.isfile(states_path) else None,
            )
            result_record["result"] = {
                k: (float(v) if v is not None else None) for k, v in (result or {}).items()
            }
            result_record["t_sec"] = float(t_sec) if t_sec is not None else None
            result_record["mem_gb"] = float(mem_gb) if mem_gb is not None else None
            result_record["stages_run"]["t2t"] = True
            print(f"[{model_key}] Step C done: {result_record['result']}")

    finally:
        _save_and_maybe_exit()


# ------------------------------------------------------------ orchestrator 模式
# 双卡调度：把模型列表排到 GPU slot 上，每 slot 占一个 subprocess。
# 用 CUDA_VISIBLE_DEVICES 把该子进程的可见 GPU 限制到指定物理 id。

def _orchestrate(args):
    proj = os.path.dirname(_script_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(proj, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, f"per_model_streaming_L{args.max_seq_len}_{run_id}")
    os.makedirs(work_dir, exist_ok=True)

    models = args.models or list(MODEL_COLS)

    gpus = list(range(args.n_gpus))
    pending = list(models)
    running = {}   # gpu_id -> (model_key, Popen, out_json, log_fh, t0)
    results = {}

    print(f"[orchestrator] 模型={models} GPU={gpus} work_dir={work_dir}")

    def _launch(model_key, gpu_id):
        out_json = os.path.join(work_dir, f"{model_key}.json")
        log_path = os.path.join(work_dir, f"{model_key}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker",
            "--model", model_key,
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
        log_fh = open(log_path, "w", buffering=1)  # line-buffered
        print(f"[launch] {model_key} on GPU {gpu_id}  -> log={log_path}")
        p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        running[gpu_id] = (model_key, p, out_json, log_fh, time.time())

    while pending or running:
        for gpu_id in gpus:
            if gpu_id not in running and pending:
                _launch(pending.pop(0), gpu_id)
        time.sleep(10)
        done = []
        for gpu_id, (mk, p, out_json, log_fh, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log_fh.close()
                elapsed = time.time() - t0
                if rc == 0 and os.path.isfile(out_json):
                    with open(out_json, encoding="utf-8") as f:
                        results[mk] = json.load(f)
                    print(f"[done] {mk} (GPU {gpu_id}) rc={rc} elapsed={elapsed:.0f}s")
                else:
                    print(f"[FAIL] {mk} (GPU {gpu_id}) rc={rc} elapsed={elapsed:.0f}s, see {out_json.replace('.json', '.log')}")
                    # 若子进程中途落盘了部分 JSON，仍尝试读
                    if os.path.isfile(out_json):
                        try:
                            with open(out_json, encoding="utf-8") as f:
                                results[mk] = json.load(f)
                        except Exception:
                            results[mk] = None
                    else:
                        results[mk] = None
                done.append(gpu_id)
        for g in done:
            del running[g]

    _aggregate(args, models, results, work_dir, run_id, out_dir)


# ---------------------------------------------------------------- 汇总结果

def _aggregate(args, models, results, work_dir, run_id, out_dir):
    txt_path = os.path.join(out_dir, f"per_model_streaming_L{args.max_seq_len}_{run_id}.txt")
    csv_path = os.path.join(out_dir, f"per_model_streaming_L{args.max_seq_len}_{run_id}.csv")

    def fmt(v, width=None, prec=4):
        if v is None:
            s = "N/A"
        elif isinstance(v, float):
            s = f"{v:.{prec}f}"
        else:
            s = str(v)
        return s.ljust(width) if width else s

    # 从第一个成功的 result 里拿 seed（所有 worker 用同一 seed=2020）
    any_ok = next((r for r in results.values() if r), None)
    seed = any_ok.get("seed", 2020) if any_ok else 2020

    # 列宽：指标列 14，每个模型列 12
    label_w, col_w = 14, 12

    lines = []
    # ============================================================
    # 抬头说明：让读者看到文件就知道这是什么实验、怎么跑的
    # ============================================================
    lines.append("=" * 90)
    lines.append("Per-Model Pretrain + Streaming T2T on ml-1m (80/20 temporal split)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("[实验情景]")
    lines.append("  ml-1m 按 timestamp 全局排序后切 80/20：前 80% 作 pretrain，后 20% 作 streaming T2T。")
    lines.append("  每个模型独立在 pretrain 上跑完整预训练（非嫁接，所有独有参数都训），")
    lines.append("  预训练结束后 replay 一次 pretrain 导出 per-user 内部状态 (S, M, V) 到 user_states.pt，")
    lines.append("  再进入 T2T 阶段：模型加载 ckp + user_states，按真实时间顺序处理后 20% 交互。")
    lines.append("")
    lines.append("[评估协议] Prequential streaming (Gama et al., 2009)")
    lines.append("  每个 user 的 t2t 尾部 10% 交互被标为 test 点。流式迭代时：")
    lines.append("    1) 到达 test 点先 full_sort_predict → 记录 Recall/NDCG/MRR")
    lines.append("    2) 整个 batch（含 test 点）参与 loss.backward() → 模拟在线反馈")
    lines.append("    3) 交互加入 user_history；模型的 per-user S/M/V 持续演化，不重置")
    lines.append("  每 user 约 3 个 test 点，总 test 点 ~18K，所以 Recall@10 绝对值约在 0.01 量级。")
    lines.append("")
    lines.append("[模型对照]")
    lines.append("  GDN  = Gated Delta Net（基线，前向 ≡ 在线 SGD）")
    lines.append("  Mo   = MoRec（动量式前向更新）")
    lines.append("  Nest = NestRec（Nesterov 动量）")
    lines.append("  Adam = DamRec（Adam 式，秩一分解 V_r⊙V_k^T）")
    lines.append("  Fro  = FroRec（F-Adam，V 降维为 Frobenius 标量）")
    lines.append("")
    lines.append("[超参数]")
    lines.append(f"  max_seq_len L         = {args.max_seq_len}")
    lines.append(f"  pretrain_epochs       = {args.epochs or 150}  (实际受 stopping_step=10 早停控制)")
    lines.append(f"  pretrain_batch_size   = 2048 (default), lr = 1e-3 (yaml default)")
    lines.append(f"  t2t_lr                = {args.t2t_lr or 1e-4}  (流式阶段)")
    lines.append(f"  t2t_epochs            = 1 (prequential 单次 pass)")
    lines.append(f"  t2t_test_ratio        = 0.1  (每 user 尾部 10% 为 test)")
    lines.append(f"  random seed           = {seed}  (RecBole default; 单 seed 单次运行)")
    lines.append("")
    lines.append("[运行信息]")
    lines.append(f"  run_id       = {run_id}")
    lines.append(f"  work_dir     = {work_dir}")
    lines.append(f"  models       = {models}")
    skip_flags = [f for f, on in
                  [("skip_pretrain", args.skip_pretrain),
                   ("skip_dump", args.skip_dump),
                   ("skip_t2t", args.skip_t2t),
                   ("force_redump", args.force_redump)] if on]
    lines.append(f"  skip/force   = {skip_flags or 'none'}")
    lines.append("")

    # ============================================================
    # 预训练阶段：pretrain 子集上的常规划分（与下方 streaming 非同一考卷）
    # ============================================================
    lines.append("-" * 90)
    lines.append("PRETRAIN 离线 best-valid / test (pretrain 子集，RecBole leave-one-out+TO 等，见 yaml)")
    lines.append("-" * 90)
    for phase, key in [("valid (best)", "pretrain_valid"), ("test", "pretrain_test")]:
        lines.append(f"  [{phase}]")
        h = "Metric".ljust(label_w) + "".join(m.ljust(col_w) for m in models)
        lines.append(h)
        lines.append("-" * (label_w + col_w * len(models)))
        for mk in METRIC_KEYS:
            row = mk.ljust(label_w)
            for m in models:
                r = results.get(m)
                sub = (r or {}).get(key) if r else None
                val = None if not isinstance(sub, dict) else sub.get(mk)
                row += fmt(val, width=col_w)
            lines.append(row)
        lines.append("")

    # ============================================================
    # 主表：流式 T2T 指标 × 模型
    # ============================================================
    lines.append("-" * 90)
    lines.append("TEST (streaming T2T)  —  Recall / NDCG / MRR @10, 越大越好")
    lines.append("-" * 90)
    header = "Metric".ljust(label_w) + "".join(m.ljust(col_w) for m in models)
    lines.append(header)
    lines.append("-" * (label_w + col_w * len(models)))
    for mk in METRIC_KEYS:
        row = mk.ljust(label_w)
        for m in models:
            r = results.get(m)
            val = None if r is None else r.get("result", {}).get(mk)
            row += fmt(val, width=col_w)
        lines.append(row)
    lines.append("")

    # ============================================================
    # 耗时 / 显存
    # ============================================================
    lines.append("-" * 90)
    lines.append("训练耗时 / 流式推断耗时 / 峰值显存")
    lines.append("-" * 90)
    lines.append(header)
    lines.append("-" * (label_w + col_w * len(models)))

    def _row(label, key, prec=0, suffix=""):
        row = label.ljust(label_w)
        for m in models:
            r = results.get(m)
            v = None if r is None else r.get(key)
            row += (fmt(v, col_w, prec=prec) if v is None or not suffix
                    else (f"{v:.{prec}f}{suffix}").ljust(col_w))
        return row

    lines.append(_row("pretrain(s)", "pretrain_sec", prec=0))
    lines.append(_row("t2t(s)",      "t2t_sec" if any("t2t_sec" in (r or {}) for r in results.values()) else "t_sec", prec=0))
    lines.append(_row("peak_mem(GB)","mem_gb", prec=2))
    lines.append("")

    # ============================================================
    # 产物路径（调试/复核用）
    # ============================================================
    lines.append("-" * 90)
    lines.append("产物路径（复现 / 断点续跑用）")
    lines.append("-" * 90)
    for m in models:
        r = results.get(m)
        if r is None:
            lines.append(f"  {m:<6}  FAILED  log: {os.path.join(work_dir, m + '.log')}")
        else:
            lines.append(f"  {m:<6}  ckp   : {r.get('ckp_path')}")
            lines.append(f"          states: {r.get('states_path')}")
    lines.append("")
    lines.append("[注] 本次为单 seed 单次运行的可行性验证结果；正式论文实验需 3~5 seed 取均值±std。")
    lines.append("=" * 90)

    table = "\n".join(lines)
    print("\n" + table)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nTXT: {txt_path}")

    # CSV：pretrain 离线 + streaming + 耗时 + 路径
    extra_pre = [f"pretrain_valid_{mk}" for mk in METRIC_KEYS_FULL] + [
        f"pretrain_test_{mk}" for mk in METRIC_KEYS_FULL
    ]
    csv_cols = (
        ["model", "seed", "pretrain_sec", "t2t_sec", "peak_mem_gb"]
        + extra_pre
        + [f"test_{mk}" for mk in METRIC_KEYS_FULL]
        + ["ckp_path", "states_path"]
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for m in models:
            r = results.get(m)
            if r is None:
                f.write(m + "," + ",".join(["N/A"] * (len(csv_cols) - 1)) + "\n")
                continue
            parts = [m, str(r.get("seed") or "N/A")]
            parts.append(f"{r.get('pretrain_sec'):.2f}" if r.get("pretrain_sec") is not None else "N/A")
            parts.append(f"{r.get('t_sec'):.2f}" if r.get("t_sec") is not None else "N/A")
            parts.append(f"{r.get('mem_gb'):.2f}" if r.get("mem_gb") is not None else "N/A")
            for mk in METRIC_KEYS_FULL:
                v = (r.get("pretrain_valid") or {}).get(mk) if isinstance(r.get("pretrain_valid"), dict) else None
                parts.append(f"{v:.4f}" if v is not None else "N/A")
            for mk in METRIC_KEYS_FULL:
                v = (r.get("pretrain_test") or {}).get(mk) if isinstance(r.get("pretrain_test"), dict) else None
                parts.append(f"{v:.4f}" if v is not None else "N/A")
            for mk in METRIC_KEYS_FULL:
                v = r.get("result", {}).get(mk)
                parts.append(f"{v:.4f}" if v is not None else "N/A")
            parts.append(str(r.get("ckp_path") or ""))
            parts.append(str(r.get("states_path") or ""))
            f.write(",".join(parts) + "\n")
    print(f"CSV: {csv_path}")


# --------------------------------------------------------------- 命令行入口

def main():
    parser = argparse.ArgumentParser(
        description="每模型独立预训练 + state_dump + 流式 T2T (ml-1m 80/20)，双卡并行"
    )
    # 公共参数
    parser.add_argument("-L", "--max_seq_len", type=int, default=64,
                        help="序列长度 L，默认 64；与 yaml 中 MAX_ITEM_LIST_LENGTH 互覆")
    parser.add_argument("--epochs", type=int, default=None,
                        help="预训练最大 epoch，默认 150（受 stopping_step 早停控制）")
    parser.add_argument("--t2t_lr", type=float, default=None,
                        help="流式 T2T 学习率，默认 1e-4")
    parser.add_argument("--models", type=str, default=None,
                        help="逗号分隔子集，如 'Adam,Fro'，默认全部 GDN,Mo,Nest,Adam,Fro")
    parser.add_argument("--show_progress", action="store_true",
                        help="显示 tqdm 进度条（多卡并行时日志会乱，调试用）")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None,
                        help="结果输出目录，默认 experiment_results/")

    # Orchestrator 专用
    parser.add_argument("--n_gpus", type=int, default=2,
                        help="并行 GPU 数，默认 2；每模型独占一卡")

    # 断点续跑
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="跳过预训练，从 saved/per_model_pretrain_L{L}/{MODEL}-*.pth 加载")
    parser.add_argument("--skip_dump", action="store_true",
                        help="跳过 state dump；若 user_states_{MODEL}.pt 不存在则 T2T 从 0 开始")
    parser.add_argument("--skip_t2t", action="store_true",
                        help="跳过 T2T（只预训练 + dump）")
    parser.add_argument("--force_redump", action="store_true",
                        help="即使 user_states_{MODEL}.pt 已存在也重新 dump")

    # Worker 专用
    parser.add_argument("--worker", action="store_true",
                        help="内部 worker 模式：单模型流水线，写 JSON 到 --out_json")
    parser.add_argument("--model", type=str, default=None,
                        help="worker 模式下指定模型 (GDN/Mo/Nest/Adam/Fro)")
    parser.add_argument("--out_json", type=str, default=None,
                        help="worker 模式下结果 JSON 输出路径")

    args = parser.parse_args()

    if args.models:
        bad = [m for m in args.models.split(",") if m.strip() not in MODEL_COLS]
        if bad:
            raise SystemExit(f"Unknown models: {bad}. Valid: {MODEL_COLS}")
        args.models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.worker:
        if not args.model or not args.out_json:
            raise SystemExit("--worker 模式需要 --model 和 --out_json")
        _worker_main(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
