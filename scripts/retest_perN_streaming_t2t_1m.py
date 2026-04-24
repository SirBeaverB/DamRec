#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
已训好的 perN ckp + user_states 重跑 Streaming T2T（只跑 Step C，重算指标）

场景：
  之前跑过 `run_pretrain_perN_streaming_t2t_1m.py`，产物已落盘：
    saved/per_user_pretrain_perN{N}_L{L}[_s{seed}]/{MODEL}-*.pth
    saved/per_user_pretrain_perN{N}_L{L}[_s{seed}]/user_states_{MODEL}.pt
  改了 yaml 里 topk / metrics、或只想换 t2t_lr 再评一次时，用本脚本跳过 pretrain/dump。

路径查找（按 seed）：
  1. saved/per_user_pretrain_perN{N}_L{L}_s{seed}/
  2. seed=2020 时可回退 legacy：saved/per_user_pretrain_perN{N}_L{L}/

默认在汇总 TXT 中输出 **@10、@20、@50** 九项（与 `T2T_OVERRIDES['topk']` 一致）；
`--at10-only` 仅 @10；`--txt-only-20-50` 仅 @20/@50（不再在 TXT 里列 @10，T2T 仍一次算全量 topk，不重复多跑）。
多 seed 时主表为 mean±std。

Usage:
  python scripts/retest_perN_streaming_t2t_1m.py

  python scripts/retest_perN_streaming_t2t_1m.py --txt-only-20-50

  python scripts/retest_perN_streaming_t2t_1m.py --at10-only --n_gpus 1

  CUDA_VISIBLE_DEVICES=0 python scripts/retest_perN_streaming_t2t_1m.py \\
      --worker --model Fro --N 100 --seed 2020 --out_json /tmp/out.json
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

# @10 仅表格（--at10-only）
METRIC_KEYS_MAIN = ["recall@10", "ndcg@10", "mrr@10"]
# 与 T2T topk=[10,20,50] 一致（默认汇总块会带）
METRIC_KEYS_EXTRA = [
    "recall@20", "ndcg@20", "mrr@20",
    "recall@50", "ndcg@50", "mrr@50",
]
METRIC_KEYS_FULL = (
    ["recall@10", "ndcg@10", "mrr@10", "hit@10", "precision@10"]
    + ["recall@20", "ndcg@20", "mrr@20", "hit@20", "precision@20"]
    + ["recall@50", "ndcg@50", "mrr@50", "hit@50", "precision@50"]
)

DEFAULT_NS = [10, 20, 50, 100]
DEFAULT_MODELS = ["GDN", "Adam", "Fro"]
DEFAULT_SEEDS = [2020]
LEGACY_SEED = 2020


def _parse_csv_int(s):
    if s is None:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_str(s):
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _find_ckp_and_states(model_name, N, seed, L):
    """返回 (ckp_path, states_path)；states 可缺失（为 None 仍返回 ckp）。"""
    candidates = [os.path.join(PROJ, "saved", f"per_user_pretrain_perN{N}_L{L}_s{seed}")]
    if seed == LEGACY_SEED:
        candidates.append(os.path.join(PROJ, "saved", f"per_user_pretrain_perN{N}_L{L}"))
    for ckp_dir in candidates:
        if not os.path.isdir(ckp_dir):
            continue
        pths = sorted(glob.glob(os.path.join(ckp_dir, f"{model_name}-*.pth")))
        if not pths:
            continue
        ckp_path = pths[-1]
        states_path = os.path.join(ckp_dir, f"user_states_{model_name}.pt")
        if not os.path.isfile(states_path):
            states_path = None
        return ckp_path, states_path
    return None, None


def _visible_gpu_ids_for_children(n_wanted):
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return [str(i) for i in range(n_wanted)]
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        return [str(i) for i in range(n_wanted)]
    if len(parts) < n_wanted:
        print(
            f"[orchestrator] 警告: CUDA_VISIBLE_DEVICES 仅有 {len(parts)} 张，"
            f"请求 n_gpus={n_wanted}，将只使用 {len(parts)} 个 slot"
        )
    return parts[:n_wanted]


def _worker_main(args):
    import run_pretrain_t2t_1m as pt1m
    from run_pretrain_t2t_1m import MODEL_CONFIGS, run_t2t_from_ckp

    model_key = args.model
    if model_key not in MODEL_CONFIGS:
        raise SystemExit(f"Unknown --model {model_key}. Valid: {list(MODEL_CONFIGS.keys())}")
    model_name, _ = MODEL_CONFIGS[model_key]

    N = int(args.N)
    seed = int(args.seed)

    ckp_path, states_path = _find_ckp_and_states(model_name, N, seed, args.max_seq_len)
    if ckp_path is None:
        raise SystemExit(
            f"[retest] 未找到 {model_name} N={N} seed={seed} 的 ckp："
            f"saved/per_user_pretrain_perN{N}_L{args.max_seq_len}[_s{seed}]/"
        )

    perN_dataset = f"ml-1m-pretrain-per{N}"
    pt1m.T2T_OVERRIDES["streaming_pretrain_dataset"] = perN_dataset
    if args.t2t_lr is not None:
        pt1m.T2T_OVERRIDES["learning_rate"] = float(args.t2t_lr)

    record = {
        "model": model_key,
        "model_name": model_name,
        "N": N,
        "seed": seed,
        "ckp_path": ckp_path,
        "states_path": states_path,
        "result": {},
        "t_sec": None,
        "mem_gb": None,
        "t2t_lr": args.t2t_lr if args.t2t_lr is not None else pt1m.T2T_OVERRIDES.get("learning_rate"),
    }

    print(f"[retest {model_key}/N={N}/seed={seed}] ckp={ckp_path}")
    print(f"[retest {model_key}/N={N}/seed={seed}] states={states_path}")

    result, t_sec, mem_gb = run_t2t_from_ckp(
        ckp_path=ckp_path,
        show_progress=args.show_progress,
        t2t_model=None,
        t2t_lr=args.t2t_lr,
        max_seq_len=args.max_seq_len,
        user_states_path=states_path,
        seed=seed,
    )
    record["result"] = {k: (float(v) if v is not None else None) for k, v in (result or {}).items()}
    record["t_sec"] = float(t_sec) if t_sec is not None else None
    record["mem_gb"] = float(mem_gb) if mem_gb is not None else None
    print(f"[retest {model_key}/N={N}/seed={seed}] done: keys in result = {list((result or {}).keys())[:8]}...")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def _aggregate(args, models, Ns, seeds, results, work_dir, run_id, out_dir):
    txt_path = os.path.join(out_dir, f"retest_perN_streaming_L{args.max_seq_len}_{run_id}.txt")
    csv_path = os.path.join(out_dir, f"retest_perN_streaming_L{args.max_seq_len}_{run_id}.csv")

    def fmt(v, width=None, prec=4):
        if v is None:
            s = "N/A"
        elif isinstance(v, float):
            s = f"{v:.{prec}f}"
        else:
            s = str(v)
        return s.ljust(width) if width else s

    def _collect(m, N, key):
        vals = []
        for s in seeds:
            r = results.get((m, N, s))
            v = None if r is None else r.get("result", {}).get(key)
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

    multi = len(seeds) > 1
    label_w, col_w, delta_w = 10, 12, 14

    have_gdn = "GDN" in models
    show_d_adam = have_gdn and "Adam" in models
    show_d_fro = have_gdn and "Fro" in models
    show_d_frnv = have_gdn and "FroNoV" in models

    lines = []
    lines.append("=" * 100)
    lines.append("Retest: Streaming T2T on existing perN checkpoints (no pretrain / no dump)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("[说明]")
    lines.append("  仅重跑 Step C：加载已保存的 .pth 与 user_states_*.pt（若存在），在 ml-1m-t2t 上 prequential 评估。")
    lines.append("  streaming_pretrain_dataset 与训练时一致，指向 ml-1m-pretrain-perN。")
    lines.append("")
    lines.append("[任务配置]")
    lines.append(f"  Ns     = {Ns}")
    lines.append(f"  models = {models}")
    lines.append(f"  seeds  = {seeds}" + ("  (多 seed 时主表为 mean±std)" if multi else "  (单 seed)"))
    lines.append(f"  L      = {args.max_seq_len}")
    lines.append(f"  t2t_lr = {args.t2t_lr if args.t2t_lr is not None else 1e-4}")
    if getattr(args, "at10_only", False):
        lines.append("  at10_only = True  (汇总 TXT 仅 @10)")
    if getattr(args, "txt_only_20_50", False):
        lines.append("  txt_only_20_50 = True  (汇总 TXT 仅 @20/@50，不列 @10)")
    lines.append("")
    lines.append("[运行信息]")
    lines.append(f"  run_id   = {run_id}")
    lines.append(f"  work_dir = {work_dir}")
    n_sl = getattr(args, "_n_slots", None)
    sids = getattr(args, "_slot_cuda_ids", None)
    if n_sl is not None and sids is not None:
        lines.append(f"  parallel = {n_sl} slot(s), CUDA per worker = {sids}")
    man = getattr(args, "_manifest_path", None)
    olp = getattr(args, "_orchestrate_log", None)
    if man:
        lines.append(f"  manifest = {man}")
    if olp:
        lines.append(f"  orchestrate.log = {olp}")
    lines.append("")

    def _emit_test_block(metric_keys, title_extra=""):
        lines.append("-" * 100)
        lines.append(
            f"TEST (streaming T2T)  —  每指标按 N 排行，附可选 Δ% 相对 GDN{title_extra}"
        )
        lines.append("-" * 100)
        for mk in metric_keys:
            sub = f"  Metric: {mk}"
            if multi:
                sub += f"  (mean ± std over {len(seeds)} seeds)"
            lines.append(sub)
            header = "N".ljust(label_w) + "".join(m.ljust(col_w) for m in models)
            if show_d_adam:
                header += "Δ(Adam-GDN)".ljust(delta_w)
            if show_d_fro:
                header += "Δ(Fro-GDN)".ljust(delta_w)
            if show_d_frnv:
                header += "Δ(FroNoV-GDN)".ljust(delta_w)
            lines.append(header)
            n_extra = (1 if show_d_adam else 0) + (1 if show_d_fro else 0) + (1 if show_d_frnv else 0)
            lines.append("-" * (label_w + col_w * len(models) + delta_w * n_extra))
            for N in Ns:
                row = f"N={N}".ljust(label_w)
                means = {}
                for m in models:
                    vals = _collect(m, N, mk)
                    mean, std = _mean_std(vals)
                    means[m] = mean
                    if multi:
                        row += fmt_ms(mean, std, prec=4, width=col_w)
                    else:
                        row += fmt(mean, col_w, prec=4)
                if show_d_adam:
                    d = _delta_pct(means.get("Adam"), means.get("GDN"))
                    row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
                if show_d_fro:
                    d = _delta_pct(means.get("Fro"), means.get("GDN"))
                    row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
                if show_d_frnv:
                    d = _delta_pct(means.get("FroNoV"), means.get("GDN"))
                    row += (f"{d:+.1f}%" if d is not None else "N/A").ljust(delta_w)
                lines.append(row)
            lines.append("")

    if getattr(args, "txt_only_20_50", False):
        _emit_test_block(METRIC_KEYS_EXTRA, "  (topk 20/50，无异 @10)")
    elif getattr(args, "at10_only", False):
        _emit_test_block(METRIC_KEYS_MAIN)
    else:
        _emit_test_block(METRIC_KEYS_MAIN)
        _emit_test_block(METRIC_KEYS_EXTRA, "  (topk 20/50)")

    if multi:
        _detail_metric = "recall@20" if getattr(args, "txt_only_20_50", False) else "recall@10"
        lines.append("-" * 100)
        lines.append(f"Per-seed 明细 ({_detail_metric})")
        lines.append("-" * 100)
        pshdr = "N_model".ljust(label_w + 6)
        for s in seeds:
            pshdr += f"seed={s}".ljust(col_w)
        lines.append(pshdr)
        lines.append("-" * (label_w + 6 + col_w * len(seeds)))
        for N in Ns:
            for m in models:
                row = f"N={N}/{m}".ljust(label_w + 6)
                for s in seeds:
                    r = results.get((m, N, s))
                    v = None if r is None else r.get("result", {}).get(_detail_metric)
                    row += fmt(v, col_w)
                lines.append(row)
        lines.append("")

    lines.append("=" * 100)

    table = "\n".join(lines)
    print("\n" + table)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nTXT: {txt_path}")

    csv_cols = (
        ["model", "N", "seed", "t2t_sec", "peak_mem_gb"]
        + [f"test_{k}" for k in METRIC_KEYS_FULL]
        + ["ckp_path", "states_path"]
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for N in Ns:
            for m in models:
                for s in seeds:
                    r = results.get((m, N, s))
                    if r is None:
                        f.write(
                            f"{m},{N},{s},"
                            + ",".join(["N/A"] * (len(csv_cols) - 3))
                            + "\n"
                        )
                        continue
                    parts = [m, str(N), str(r.get("seed", s))]
                    parts.append(
                        f"{r.get('t_sec'):.2f}" if r.get("t_sec") is not None else "N/A"
                    )
                    parts.append(
                        f"{r.get('mem_gb'):.2f}" if r.get("mem_gb") is not None else "N/A"
                    )
                    for k in METRIC_KEYS_FULL:
                        v = r.get("result", {}).get(k)
                        parts.append(f"{v:.4f}" if v is not None else "N/A")
                    parts.append(str(r.get("ckp_path") or ""))
                    parts.append(str(r.get("states_path") or ""))
                    f.write(",".join(parts) + "\n")
    print(f"CSV: {csv_path}")


def _orchestrate(args):
    if args.n_gpus < 1:
        raise SystemExit("--n_gpus 必须 >= 1")

    Ns = args.Ns or DEFAULT_NS
    models = args.models or DEFAULT_MODELS
    seeds = args.seeds or DEFAULT_SEEDS

    import run_pretrain_t2t_1m as _pt1m
    bad = [m for m in models if m not in _pt1m.MODEL_CONFIGS]
    if bad:
        raise SystemExit(
            f"Unknown --models: {bad}. Valid: {list(_pt1m.MODEL_CONFIGS.keys())}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(PROJ, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, f"retest_perN_streaming_L{args.max_seq_len}_{run_id}")
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

        jobs = []
        missing = []
        for N in Ns:
            for m in models:
                model_name, _ = _pt1m.MODEL_CONFIGS[m]
                for s in seeds:
                    ckp, _st = _find_ckp_and_states(model_name, N, s, args.max_seq_len)
                    if ckp is None:
                        missing.append((m, N, s))
                    else:
                        jobs.append((m, N, s))

        manifest = os.path.join(work_dir, "00_manifest.txt")
        args._manifest_path = manifest
        with open(manifest, "w", encoding="utf-8") as mf:
            mf.write("=== Retest perN streaming T2T ===\n")
            mf.write(f"start  = {datetime.now().isoformat()}\n")
            mf.write(f"argv   = {sys.argv}\n")
            mf.write(
                f"host CUDA = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n"
            )
            mf.write(f"slot GPUs = {slot_cuda_ids}\n")
            mf.write(f"Ns, models, seeds = {Ns}, {models}, {seeds}\n")
            mf.write(
                f"L, t2t_lr, at10_only, txt_only_20_50 = {args.max_seq_len}, {args.t2t_lr}, "
                f"{args.at10_only}, {args.txt_only_20_50}\n"
            )
            mf.write(f"runnable jobs = {len(jobs)}  missing = {len(missing)}\n")
            mf.write(f"orchestrate.log = {olog_path}\n")

        oprint(f"[retest] {len(jobs)} jobs runnable; {len(missing)} missing ckp")
        if missing:
            for m, N, s in missing:
                oprint(f"  [miss] {m}/N={N}/seed={s}")
        if not jobs:
            raise SystemExit("没有可重测的 ckp，退出")

        gpus = list(range(n_slots))
        pending = list(jobs)
        running = {}
        results = {}

        def _launch(model_key, N, seed, slot_idx):
            key = f"{model_key}_N{N}_s{seed}"
            cuda_id = slot_cuda_ids[slot_idx]
            out_json = os.path.join(work_dir, f"{key}.json")
            log_path = os.path.join(work_dir, f"{key}.log")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--model",
                model_key,
                "--N",
                str(N),
                "--seed",
                str(seed),
                "--out_json",
                out_json,
                "--max_seq_len",
                str(args.max_seq_len),
            ]
            if args.t2t_lr is not None:
                cmd += ["--t2t_lr", str(args.t2t_lr)]
            if args.show_progress:
                cmd += ["--show_progress"]
            log_fh = open(log_path, "w", buffering=1)
            cmd_txt = os.path.join(work_dir, f"{key}_cmd.txt")
            with open(cmd_txt, "w", encoding="utf-8") as cf:
                cf.write(
                    " ".join('"' + c + '"' if " " in c else c for c in cmd) + "\n"
                )
                cf.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            oprint(
                f"[launch] {key} CUDA={env['CUDA_VISIBLE_DEVICES']} (slot {slot_idx}) -> {log_path}"
            )
            p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            running[slot_idx] = (
                (model_key, N, seed),
                p,
                out_json,
                log_fh,
                time.time(),
                str(cuda_id),
            )

        while pending or running:
            for slot_idx in gpus:
                if slot_idx not in running and pending:
                    mk, N, sd = pending.pop(0)
                    _launch(mk, N, sd, slot_idx)
            time.sleep(5)
            done = []
            for slot_idx, ((mk, N, sd), p, out_json, log_fh, t0, cuda_s) in list(
                running.items()
            ):
                rc = p.poll()
                if rc is not None:
                    log_fh.close()
                    elapsed = time.time() - t0
                    k = (mk, N, sd)
                    if rc == 0 and os.path.isfile(out_json):
                        with open(out_json, encoding="utf-8") as f:
                            results[k] = json.load(f)
                        oprint(
                            f"[done] {mk}/N={N}/seed={sd} CUDA {cuda_s} (slot {slot_idx})"
                            f" {elapsed:.0f}s"
                        )
                    else:
                        oprint(
                            f"[FAIL] {mk}/N={N}/seed={sd} rc={rc}  see"
                            f" {out_json.replace('.json', '.log')}"
                        )
                        if os.path.isfile(out_json):
                            try:
                                with open(out_json, encoding="utf-8") as f:
                                    results[k] = json.load(f)
                            except Exception:
                                results[k] = None
                        else:
                            results[k] = None
                    done.append(slot_idx)
            for s in done:
                del running[s]

        for m, N, s in missing:
            results[(m, N, s)] = None

        _aggregate(args, models, Ns, seeds, results, work_dir, run_id, out_dir)
    finally:
        olog.close()


def main():
    parser = argparse.ArgumentParser(
        description="已训 perN ckp 重跑 Streaming T2T，输出与主实验汇总表同风格的 TEST 段"
    )
    parser.add_argument(
        "--Ns",
        type=_parse_csv_int,
        default=None,
        help=f"N 列表，逗号分隔，默认 {DEFAULT_NS}",
    )
    parser.add_argument(
        "--models",
        type=_parse_csv_str,
        default=None,
        help=f"模型键，逗号分隔，默认 {DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_csv_int,
        default=None,
        help=f"seed 列表，默认 {DEFAULT_SEEDS}；2020 可匹配 legacy 无 _s 后缀目录",
    )
    parser.add_argument("-L", "--max_seq_len", type=int, default=64)
    parser.add_argument(
        "--t2t_lr",
        type=float,
        default=None,
        help="流式 T2T 学习率，默认 1e-4（与 T2T_OVERRIDES 一致）",
    )
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument(
        "--at10-only",
        dest="at10_only",
        action="store_true",
        help="汇总 TXT 只含 recall/ndcg/mrr @10",
    )
    parser.add_argument(
        "--txt-only-20-50",
        dest="txt_only_20_50",
        action="store_true",
        help="汇总 TXT 只列 @20/@50（不列 @10）；流式仍一次算全 topk，仅省篇幅、不重复多开任务",
    )
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()

    if args.at10_only and args.txt_only_20_50:
        raise SystemExit("--at10-only 与 --txt-only-20-50 不能同时使用")

    if args.worker:
        if None in (args.model, args.N, args.seed, args.out_json):
            raise SystemExit("--worker 需要 --model --N --seed --out_json")
        _worker_main(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
