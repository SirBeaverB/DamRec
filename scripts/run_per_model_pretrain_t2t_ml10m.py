#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每模型独立预训练 + state_dump + 流式 T2T（ml-10m 80/20 时间切）。

目的：把 用于 ML-10M 稠密子集的流式实验，验证：
  (1) 离线评分高的 Fro/GDN 在 ml-10m streaming T2T 下能否超过 popularity baseline
  (2) Fro vs GDN 差距是否跨数据集稳定

与 ml-1m 版的区别：
  - dataset 默认 ml-10m-pretrain / ml-10m-t2t；可用 --data_tag 指向
    ml-10m-pretrain-{tag} / ml-10m-t2t-{tag}（与 prepare 的 --data_tag 一致）
  - 默认只跑 GDN / Adam / Fro 三模型（已去 perN，不再扫 N）
  - 预训练 checkpoint 与 state 落盘到 saved/per_model_pretrain_ml-10m[_tag]_L{L}/

前置：
  python scripts/prepare_ml10m_80_20_split.py    # 首次运行
  前提是 dataset/ml-10m/ml-10m.inter 已存在（RecBole atomic 格式）

Usage:
  # 默认：GDN/Adam/Fro × 1 seed × 2 GPU
  python scripts/run_per_model_pretrain_t2t_ml-10m.py

  # 多 seed
  python scripts/run_per_model_pretrain_t2t_ml-10m.py --seeds 2020,2021,2022

  # 改 L
  python scripts/run_per_model_pretrain_t2t_ml-10m.py -L 64

  # 断点续跑
  python scripts/run_per_model_pretrain_t2t_ml-10m.py --skip_pretrain
  python scripts/run_per_model_pretrain_t2t_ml-10m.py --skip_pretrain --skip_dump

  # 单模型调试
  CUDA_VISIBLE_DEVICES=0 python scripts/run_per_model_pretrain_t2t_ml-10m.py \\
      --worker --model Fro --seed 2020 --out_json /tmp/fro.json
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
PROJ = os.path.dirname(_script_dir)
sys.path.insert(0, PROJ)
sys.path.insert(0, _script_dir)

DATASET_NAME = "ml-10m"
# 由 _configure_data_paths() 根据 --data_tag 填写；无 tag 时为 ml-10m-pretrain / ml-10m-t2t
DATA_TAG = None
PRETRAIN_DATASET = "ml-10m-pretrain"
T2T_DATASET = "ml-10m-t2t"
PRETRAIN_INTER = os.path.join(PROJ, "dataset", PRETRAIN_DATASET, f"{PRETRAIN_DATASET}.inter")
T2T_INTER = os.path.join(PROJ, "dataset", T2T_DATASET, f"{T2T_DATASET}.inter")


def _configure_data_paths(data_tag=None):
    """与 prepare_ml10m_80_20_split.py 的 --data_tag 命名一致。"""
    global DATA_TAG, PRETRAIN_DATASET, T2T_DATASET, PRETRAIN_INTER, T2T_INTER
    t = (data_tag or "").strip() or None
    DATA_TAG = t
    if t:
        PRETRAIN_DATASET = f"{DATASET_NAME}-pretrain-{t}"
        T2T_DATASET = f"{DATASET_NAME}-t2t-{t}"
    else:
        PRETRAIN_DATASET = f"{DATASET_NAME}-pretrain"
        T2T_DATASET = f"{DATASET_NAME}-t2t"
    PRETRAIN_INTER = os.path.join(PROJ, "dataset", PRETRAIN_DATASET, f"{PRETRAIN_DATASET}.inter")
    T2T_INTER = os.path.join(PROJ, "dataset", T2T_DATASET, f"{T2T_DATASET}.inter")


def _saved_subdir_prefix():
    """saved/ 下目录名：per_model_pretrain_ml-10m 或 per_model_pretrain_ml-10m_u5000。"""
    return f"per_model_pretrain_{DATASET_NAME}_{DATA_TAG}" if DATA_TAG else f"per_model_pretrain_{DATASET_NAME}"

METRIC_KEYS = [
    "recall@10", "ndcg@10", "mrr@10",
    "recall@20", "ndcg@20", "mrr@20",
    "recall@50", "ndcg@50", "mrr@50",
]
METRIC_KEYS_FULL = (
    ["recall@10", "ndcg@10", "mrr@10", "hit@10", "precision@10"]
    + ["recall@20", "ndcg@20", "mrr@20", "hit@20", "precision@20"]
    + ["recall@50", "ndcg@50", "mrr@50", "hit@50", "precision@50"]
)

DEFAULT_MODELS = ["GDN", "Adam", "Fro"]
DEFAULT_SEEDS = [2020]


def _parse_csv_int(s):
    if s is None:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_str(s):
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _visible_gpu_ids_for_children(n_wanted):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [str(i) for i in range(n_wanted)]
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        return [str(i) for i in range(n_wanted)]
    return parts[:n_wanted]


def _ensure_ml10m_split():
    """缺省切分时：无 data_tag 则自动跑全量 prepare；有 data_tag 则只提示命令，避免误生成。"""
    if os.path.isfile(T2T_INTER) and os.path.isfile(PRETRAIN_INTER):
        return
    raw = os.path.join(PROJ, "dataset", DATASET_NAME, f"{DATASET_NAME}.inter")
    if not os.path.isfile(raw):
        raise SystemExit(
            f"未找到 {raw}。请先执行：\n"
            f"  python scripts/download_ml10m.py"
        )
    if DATA_TAG:
        raise SystemExit(
            f"未找到带 data_tag 的切分数据:\n  {PRETRAIN_INTER}\n  {T2T_INTER}\n"
            f"请先执行（标签需与当前 --data_tag 一致），例如:\n"
            f"  python scripts/prepare_ml10m_80_20_split.py --data_tag {DATA_TAG} --max_users 5000 --seed 42\n"
        )
    print(f"[ensure] 80/20 切分不存在，运行 prepare_ml10m_80_20_split.py ...")
    subprocess.run(
        [sys.executable, os.path.join(_script_dir, "prepare_ml10m_80_20_split.py")],
        check=True,
    )


def _set_ml10m_overrides(pt1m, t2t_lr=None, epochs=None, seed=None):
    """把 pt1m 模块的 PRETRAIN/T2T OVERRIDES 改成 ml-10m 版本。"""
    pt1m.PRETRAIN_OVERRIDES["dataset"] = PRETRAIN_DATASET
    pt1m.T2T_OVERRIDES["dataset"] = T2T_DATASET
    pt1m.T2T_OVERRIDES["streaming_pretrain_dataset"] = PRETRAIN_DATASET
    if epochs is not None:
        pt1m.PRETRAIN_OVERRIDES["epochs"] = int(epochs)
    if t2t_lr is not None:
        pt1m.T2T_OVERRIDES["learning_rate"] = float(t2t_lr)
    if seed is not None:
        pt1m.PRETRAIN_OVERRIDES["seed"] = int(seed)
        pt1m.T2T_OVERRIDES["seed"] = int(seed)


def _patch_ensure_split(pt1m):
    """替换 pt1m._ensure_split 让它不再去找 ml-1m-t2t。"""
    pt1m._ensure_split = _ensure_ml10m_split


def _worker_main(args):
    _configure_data_paths(getattr(args, "data_tag", None))
    import run_pretrain_t2t_1m as pt1m
    from run_pretrain_t2t_1m import (
        MODEL_CONFIGS,
        run_pretrain,
        run_state_dump,
        run_t2t_from_ckp,
    )

    model_key = args.model
    if model_key not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown --model {model_key}. Valid: {list(MODEL_CONFIGS.keys())}")
    model_name, config_file = MODEL_CONFIGS[model_key]

    seed = int(args.seed)
    _set_ml10m_overrides(pt1m, t2t_lr=args.t2t_lr, epochs=args.epochs, seed=seed)
    _patch_ensure_split(pt1m)
    _ensure_ml10m_split()

    ckp_dir = os.path.join(
        PROJ, "saved", f"{_saved_subdir_prefix()}_L{args.max_seq_len}_s{seed}"
    )
    os.makedirs(ckp_dir, exist_ok=True)
    states_path = os.path.join(ckp_dir, f"user_states_{model_name}.pt")

    record = {
        "dataset": DATASET_NAME,
        "data_tag": DATA_TAG,
        "model": model_key,
        "model_name": model_name,
        "seed": seed,
        "ckp_dir": ckp_dir,
        "ckp_path": None,
        "states_path": states_path,
        "pretrain_valid": {},
        "pretrain_test": {},
        "result": {},
        "t_sec": None,
        "mem_gb": None,
        "pretrain_sec": None,
        "dump_sec": None,
        "pretrain_epochs": pt1m.PRETRAIN_OVERRIDES.get("epochs"),
        "t2t_lr": pt1m.T2T_OVERRIDES.get("learning_rate"),
        "stages_run": {"pretrain": False, "dump": False, "t2t": False},
    }

    def _save():
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    try:
        # Step A: pretrain
        if args.skip_pretrain:
            cands = sorted(glob.glob(os.path.join(ckp_dir, f"{model_name}-*.pth")))
            if not cands:
                raise RuntimeError(f"--skip_pretrain 需要 {ckp_dir}/{model_name}-*.pth 存在")
            ckp_path = cands[-1]
            print(f"[{model_key}/seed={seed}] Step A skipped, reuse {ckp_path}")
        else:
            print(
                f"[{model_key}/seed={seed}] Step A: pretrain on {PRETRAIN_DATASET} "
                f"({pt1m.PRETRAIN_OVERRIDES['epochs']} epochs, L={args.max_seq_len})"
            )
            t0 = time.perf_counter()
            ckp_path, pv, pt = run_pretrain(
                model_name=model_name,
                config_file=config_file,
                ckp_dir=ckp_dir,
                show_progress=args.show_progress,
                max_seq_len=args.max_seq_len,
                seed=seed,
            )
            record["pretrain_sec"] = time.perf_counter() - t0
            record["pretrain_valid"] = pv or {}
            record["pretrain_test"] = pt or {}
            record["stages_run"]["pretrain"] = True
            print(f"[{model_key}/seed={seed}] Step A done in {record['pretrain_sec']:.0f}s, ckp={ckp_path}")
            print(
                f"[{model_key}/seed={seed}]  pretrain 子集 valid@10={record['pretrain_valid'].get('recall@10', 'N/A')}, "
                f"test@10={record['pretrain_test'].get('recall@10', 'N/A')}"
            )
        record["ckp_path"] = ckp_path
        _save()

        # Step B: state dump
        if not args.skip_dump:
            if os.path.isfile(states_path) and not args.force_redump:
                print(f"[{model_key}/seed={seed}] Step B: reuse {states_path}")
            else:
                print(f"[{model_key}/seed={seed}] Step B: dump user_states")
                t0 = time.perf_counter()
                run_state_dump(
                    ckp_path=ckp_path,
                    save_path=states_path,
                    model_name=model_name,
                    config_file=config_file,
                    max_seq_len=args.max_seq_len,
                    show_progress=args.show_progress,
                    seed=seed,
                )
                record["dump_sec"] = time.perf_counter() - t0
                record["stages_run"]["dump"] = True
                print(f"[{model_key}/seed={seed}] Step B done in {record['dump_sec']:.0f}s")

        # Step C: streaming T2T
        if not args.skip_t2t:
            print(f"[{model_key}/seed={seed}] Step C: streaming T2T on {T2T_DATASET}")
            result, t_sec, mem_gb = run_t2t_from_ckp(
                ckp_path=ckp_path,
                show_progress=args.show_progress,
                t2t_model=None,
                t2t_lr=args.t2t_lr,
                max_seq_len=args.max_seq_len,
                user_states_path=states_path if os.path.isfile(states_path) else None,
                seed=seed,
            )
            record["result"] = {
                k: (float(v) if v is not None else None) for k, v in (result or {}).items()
            }
            record["t_sec"] = float(t_sec) if t_sec is not None else None
            record["mem_gb"] = float(mem_gb) if mem_gb is not None else None
            record["stages_run"]["t2t"] = True
            print(f"[{model_key}/seed={seed}] Step C done: keys={list((result or {}).keys())[:6]}...")
    finally:
        _save()


def _aggregate(args, models, seeds, results, work_dir, run_id, out_dir):
    ts = f"_{DATA_TAG}" if DATA_TAG else ""
    txt_path = os.path.join(out_dir, f"per_model_streaming_{DATASET_NAME}{ts}_L{args.max_seq_len}_{run_id}.txt")
    csv_path = os.path.join(out_dir, f"per_model_streaming_{DATASET_NAME}{ts}_L{args.max_seq_len}_{run_id}.csv")

    def fmt(v, width=None, prec=4):
        if v is None:
            s = "N/A"
        elif isinstance(v, float):
            s = f"{v:.{prec}f}"
        else:
            s = str(v)
        return s.ljust(width) if width else s

    def _collect(m, key):
        vals = []
        for s in seeds:
            r = results.get((m, s))
            v = None if r is None else r.get("result", {}).get(key)
            if v is not None:
                vals.append(float(v))
        return vals

    def _collect_phase(m, key, phase):
        """phase: pretrain_valid | pretrain_test — pretrain 子集上常规划分指标。"""
        vals = []
        for s in seeds:
            r = results.get((m, s))
            sub = (r or {}).get(phase)
            v = None if not isinstance(sub, dict) else sub.get(key)
            if v is not None:
                vals.append(float(v))
        return vals

    def _collect_field(m, field):
        vals = []
        for s in seeds:
            r = results.get((m, s))
            v = None if r is None else r.get(field)
            if v is not None:
                vals.append(float(v))
        return vals

    def _mean_std(vals):
        if not vals:
            return None, None
        n = len(vals)
        mean = sum(vals) / n
        if n < 2:
            return mean, None
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        return mean, var ** 0.5

    def fmt_ms(mean, std, prec=4, width=None):
        if mean is None:
            s = "N/A"
        elif std is None:
            s = f"{mean:.{prec}f}"
        else:
            s = f"{mean:.{prec}f}±{std:.{prec}f}"
        return s.ljust(width) if width else s

    def _delta_pct(adv, base):
        if adv is None or base is None or base == 0:
            return None
        return (adv - base) / base * 100.0

    col_w, delta_w = 22, 14

    have_gdn = "GDN" in models
    show_d_adam = have_gdn and "Adam" in models
    show_d_fro = have_gdn and "Fro" in models

    lines = []
    lines.append("=" * 100)
    lines.append(f"Per-Model Pretrain + Streaming T2T on {DATASET_NAME} (80/20 temporal split)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("[超参数]")
    lines.append(
        f"  dataset          = {DATASET_NAME} (pretrain: {PRETRAIN_DATASET}, t2t: {T2T_DATASET})"
        + (f"  [data_tag={DATA_TAG}]" if DATA_TAG else "  (全量，无 data_tag)")
    )
    lines.append(f"  models           = {models}")
    lines.append(f"  seeds            = {seeds}")
    lines.append(f"  max_seq_len L    = {args.max_seq_len}")
    lines.append(f"  pretrain_epochs  = {args.epochs or 150}  (stopping_step=10)")
    lines.append(f"  t2t_lr           = {args.t2t_lr or 1e-4}")
    lines.append(f"  run_id           = {run_id}")
    lines.append(f"  work_dir         = {work_dir}")
    lines.append("")

    for title, phase in [
        ("PRETRAIN 离线 valid (best on pretrain split)", "pretrain_valid"),
        ("PRETRAIN 离线 test (on pretrain split)", "pretrain_test"),
    ]:
        lines.append("-" * 100)
        lines.append(f"{title}  —  mean±std 跨 seed，非 T2T")
        for mk in METRIC_KEYS:
            lines.append(f"  Metric: {mk}")
            header = "".ljust(6) + "".join(m.ljust(col_w) for m in models)
            if show_d_adam:
                header += "Δ(Adam-GDN)".ljust(delta_w)
            if show_d_fro:
                header += "Δ(Fro-GDN)".ljust(delta_w)
            lines.append(header)
            n_extra = (1 if show_d_adam else 0) + (1 if show_d_fro else 0)
            lines.append("-" * (6 + col_w * len(models) + delta_w * n_extra))
            row = "".ljust(6)
            means = {}
            for m in models:
                vals = _collect_phase(m, mk, phase)
                mean, std = _mean_std(vals)
                means[m] = mean
                row += fmt_ms(mean, std, prec=4, width=col_w)
            if show_d_adam:
                d = _delta_pct(means.get("Adam"), means.get("GDN"))
                row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
            if show_d_fro:
                d = _delta_pct(means.get("Fro"), means.get("GDN"))
                row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
            lines.append(row)
            lines.append("")

    lines.append("-" * 100)
    for mk in METRIC_KEYS:
        lines.append(f"  Metric: {mk}  (TEST streaming T2T, mean ± std 跨 {len(seeds)} seed)")
        header = "".ljust(6) + "".join(m.ljust(col_w) for m in models)
        if show_d_adam:
            header += "Δ(Adam-GDN)".ljust(delta_w)
        if show_d_fro:
            header += "Δ(Fro-GDN)".ljust(delta_w)
        lines.append(header)
        n_extra = (1 if show_d_adam else 0) + (1 if show_d_fro else 0)
        lines.append("-" * (6 + col_w * len(models) + delta_w * n_extra))
        row = "".ljust(6)
        means = {}
        for m in models:
            vals = _collect(m, mk)
            mean, std = _mean_std(vals)
            means[m] = mean
            row += fmt_ms(mean, std, prec=4, width=col_w)
        if show_d_adam:
            d = _delta_pct(means.get("Adam"), means.get("GDN"))
            row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
        if show_d_fro:
            d = _delta_pct(means.get("Fro"), means.get("GDN"))
            row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
        lines.append(row)
        lines.append("")

    lines.append("-" * 100)
    lines.append("Per-seed 明细 (recall@10)")
    lines.append("-" * 100)
    pshdr = "model".ljust(12)
    for s in seeds:
        pshdr += f"seed={s}".ljust(col_w)
    lines.append(pshdr)
    for m in models:
        row = m.ljust(12)
        for s in seeds:
            r = results.get((m, s))
            v = None if r is None else r.get("result", {}).get("recall@10")
            row += fmt(v, col_w)
        lines.append(row)
    lines.append("")

    lines.append("-" * 100)
    lines.append("pretrain_sec / t2t_sec / peak_mem_gb (mean 跨 seed)")
    lines.append("-" * 100)
    for field, label, prec in [("pretrain_sec", "pretrain(s)", 0),
                                 ("t_sec",        "t2t(s)",      0),
                                 ("mem_gb",       "mem(GB)",     2)]:
        lines.append(f"  {label}")
        header = "".ljust(6) + "".join(m.ljust(col_w) for m in models)
        lines.append(header)
        lines.append("-" * (6 + col_w * len(models)))
        row = "".ljust(6)
        for m in models:
            vals = _collect_field(m, field)
            mean, _ = _mean_std(vals)
            row += fmt(mean, col_w, prec=prec)
        lines.append(row)
        lines.append("")
    lines.append("=" * 100)

    table = "\n".join(lines)
    print("\n" + table)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nTXT: {txt_path}")

    extra_pre = [f"pretrain_valid_{k}" for k in METRIC_KEYS_FULL] + [
        f"pretrain_test_{k}" for k in METRIC_KEYS_FULL
    ]
    csv_cols = (
        ["model", "seed", "pretrain_sec", "t2t_sec", "peak_mem_gb"]
        + extra_pre
        + [f"test_{k}" for k in METRIC_KEYS_FULL]
        + ["ckp_path", "states_path"]
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for m in models:
            for s in seeds:
                r = results.get((m, s))
                if r is None:
                    f.write(f"{m},{s}," + ",".join(["N/A"] * (len(csv_cols) - 2)) + "\n")
                    continue
                parts = [m, str(r.get("seed", s))]
                parts.append(f"{r.get('pretrain_sec'):.2f}" if r.get("pretrain_sec") is not None else "N/A")
                parts.append(f"{r.get('t_sec'):.2f}" if r.get("t_sec") is not None else "N/A")
                parts.append(f"{r.get('mem_gb'):.2f}" if r.get("mem_gb") is not None else "N/A")
                for k in METRIC_KEYS_FULL:
                    v = (r.get("pretrain_valid") or {}).get(k) if isinstance(r.get("pretrain_valid"), dict) else None
                    parts.append(f"{v:.4f}" if v is not None else "N/A")
                for k in METRIC_KEYS_FULL:
                    v = (r.get("pretrain_test") or {}).get(k) if isinstance(r.get("pretrain_test"), dict) else None
                    parts.append(f"{v:.4f}" if v is not None else "N/A")
                for k in METRIC_KEYS_FULL:
                    v = r.get("result", {}).get(k)
                    parts.append(f"{v:.4f}" if v is not None else "N/A")
                parts.append(str(r.get("ckp_path") or ""))
                parts.append(str(r.get("states_path") or ""))
                f.write(",".join(parts) + "\n")
    print(f"CSV: {csv_path}")


def _orchestrate(args):
    import run_pretrain_t2t_1m as _pt1m

    _configure_data_paths(getattr(args, "data_tag", None))
    _ensure_ml10m_split()

    models = args.models or DEFAULT_MODELS
    seeds = args.seeds or DEFAULT_SEEDS
    bad = [m for m in models if m not in _pt1m.MODEL_CONFIGS]
    if bad:
        raise SystemExit(f"Unknown --models: {bad}. Valid: {list(_pt1m.MODEL_CONFIGS.keys())}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(PROJ, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    tag_sfx = f"_{DATA_TAG}" if DATA_TAG else ""
    work_dir = os.path.join(
        out_dir, f"per_model_streaming_{DATASET_NAME}{tag_sfx}_L{args.max_seq_len}_{run_id}"
    )
    os.makedirs(work_dir, exist_ok=True)

    olog_path = os.path.join(work_dir, "orchestrate.log")
    olog = open(olog_path, "w", buffering=1)

    def oprint(*a, **k):
        print(*a, **k, flush=True)
        print(*a, file=olog, **k, flush=True)

    try:
        slot_cuda_ids = _visible_gpu_ids_for_children(args.n_gpus)
        n_slots = len(slot_cuda_ids)

        manifest = os.path.join(work_dir, "00_manifest.txt")
        with open(manifest, "w", encoding="utf-8") as mf:
            mf.write(f"=== Per-model Pretrain + Streaming T2T on {DATASET_NAME} ===\n")
            mf.write(f"start     = {datetime.now().isoformat()}\n")
            mf.write(f"work_dir  = {work_dir}\n")
            mf.write(f"run_id    = {run_id}\n")
            mf.write(f"argv      = {sys.argv}\n")
            mf.write(f"data_tag  = {DATA_TAG!r}  (None=默认 ml-10m-pretrain / ml-10m-t2t)\n")
            mf.write(f"host CUDA = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n")
            mf.write(f"slot GPUs = {slot_cuda_ids}\n")
            mf.write(f"models    = {models}\n")
            mf.write(f"seeds     = {seeds}\n")
            mf.write(f"L         = {args.max_seq_len}\n")
            mf.write(f"epochs    = {args.epochs or 150}\n")
            mf.write(f"t2t_lr    = {args.t2t_lr or 1e-4}\n")

        jobs = [(m, s) for s in seeds for m in models]
        oprint(
            f"[orchestrator] {len(jobs)} jobs = {len(models)} models × {len(seeds)} seeds, "
            f"{n_slots} slot(s)"
        )
        oprint(f"              models={models} seeds={seeds} work_dir={work_dir}")

        gpus = list(range(n_slots))
        pending = list(jobs)
        running = {}
        results = {}

        def _launch(model_key, seed, slot_idx):
            key = f"{model_key}_s{seed}"
            cuda_id = slot_cuda_ids[slot_idx]
            out_json = os.path.join(work_dir, f"{key}.json")
            log_path = os.path.join(work_dir, f"{key}.log")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker",
                "--model", model_key,
                "--seed", str(seed),
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
            if getattr(args, "data_tag", None) and str(args.data_tag).strip():
                cmd += ["--data_tag", str(args.data_tag).strip()]
            log_fh = open(log_path, "w", buffering=1)
            oprint(f"[launch] {key} CUDA={cuda_id} (slot {slot_idx}) -> {log_path}")
            p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            running[slot_idx] = ((model_key, seed), p, out_json, log_fh, time.time(), str(cuda_id))

        while pending or running:
            for slot_idx in gpus:
                if slot_idx not in running and pending:
                    mk, sd = pending.pop(0)
                    _launch(mk, sd, slot_idx)
            time.sleep(10)
            done = []
            for slot_idx, ((mk, sd), p, out_json, log_fh, t0, cuda_s) in list(running.items()):
                rc = p.poll()
                if rc is not None:
                    log_fh.close()
                    elapsed = time.time() - t0
                    key = (mk, sd)
                    if rc == 0 and os.path.isfile(out_json):
                        with open(out_json, encoding="utf-8") as f:
                            results[key] = json.load(f)
                        oprint(f"[done] {mk}/seed={sd} CUDA {cuda_s} elapsed={elapsed:.0f}s")
                    else:
                        oprint(
                            f"[FAIL] {mk}/seed={sd} rc={rc} elapsed={elapsed:.0f}s  "
                            f"see {out_json.replace('.json', '.log')}"
                        )
                        if os.path.isfile(out_json):
                            try:
                                with open(out_json, encoding="utf-8") as f:
                                    results[key] = json.load(f)
                            except Exception:
                                results[key] = None
                        else:
                            results[key] = None
                    done.append(slot_idx)
            for s in done:
                del running[s]

        _aggregate(args, models, seeds, results, work_dir, run_id, out_dir)
    finally:
        olog.close()


def main():
    parser = argparse.ArgumentParser(
        description=f"Per-model Pretrain + Streaming T2T on {DATASET_NAME}"
    )
    parser.add_argument(
        "--models",
        type=_parse_csv_str,
        default=None,
        help=(
            f"模型子集，逗号分隔，默认 {DEFAULT_MODELS}。"
            "键名与 run_pretrain_t2t_1m.MODEL_CONFIGS 一致，含 GDN / Mo / Nest / Adam / Fro / FroNoV / GRU4Rec 等；"
            "例: --models GDN,Adam,Fro,GRU4Rec"
        ),
    )
    parser.add_argument("--seeds", type=_parse_csv_int, default=None,
                        help=f"seed 列表，默认 {DEFAULT_SEEDS}")
    parser.add_argument("-L", "--max_seq_len", type=int, default=64,
                        help="序列长度 L，默认 64 (CHUNK_SIZE=16 兼容)")
    parser.add_argument("--epochs", type=int, default=None, help="预训练最大 epoch，默认 150")
    parser.add_argument("--t2t_lr", type=float, default=None, help="流式 T2T 学习率，默认 1e-4")
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)

    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--skip_dump", action="store_true")
    parser.add_argument("--skip_t2t", action="store_true")
    parser.add_argument("--force_redump", action="store_true")
    parser.add_argument(
        "--data_tag",
        type=str,
        default=None,
        help=(
            "与 prepare_ml10m_80_20_split.py 的 --data_tag 一致："
            "使用 dataset/ml-10m-pretrain-{tag}/ 与 ml-10m-t2t-{tag}/；"
            "checkpoint 在 saved/per_model_pretrain_ml-10m_{tag}_L{L}_s{seed}/。默认无 tag=全量目录"
        ),
    )

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()

    if args.worker:
        if args.model is None or args.seed is None or args.out_json is None:
            raise SystemExit("--worker 需要 --model --seed --out_json")
        _worker_main(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
