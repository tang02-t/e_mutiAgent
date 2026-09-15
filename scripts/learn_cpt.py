#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯网络参数学习 + 校准脚本（B-1）。

用真实标注 DGA 数据（data/real/dga/dga_records.jsonl）统计学习：
  - 先验  P(F)        = 各 fault_label 频率
  - 条件概率 CPT P(S|F):
        p_true  = P(症状=True | 故障=F)   —— 该故障样本中征兆出现的比例
        p_false = P(症状=True | 故障≠F)   —— 非该故障样本中征兆出现的比例
均采用拉普拉斯平滑（Laplace add-k），避免 0 概率导致似然连乘塌缩。

v2（B-1）新增：
  - `--kfold 5`：按故障类别分层 5 折交叉验证，逐折报告 Top-1 / Top-3 / 对数似然 / ECE / Brier；
  - 温度缩放：在每折验证集上网格搜索 T 使 NLL 最小，报告校准前后 ECE（15 桶）、Brier；
  - 按故障类别学习规则 / 贝叶斯融合权重 w_f（网格搜索，最大化验证折对数似然），与固定 0.7 对照；
  - 可靠性图 docs/figures/fig_b1_reliability.png，报告 docs/attribution_calibration.md；
  - 最终用全量数据重学 prior / CPT，并把 5 折平均的 T 与 w_f 写入 learned_params.json 的
    `calibration` / `fusion_weights` 字段，供 FaultBayesianNetwork 加载。

评测口径：只评估非 normal 样本（引擎不输出 normal），与 scripts/eval_fault_attribution.py 一致；
negative evidence（低于注意值的气体征兆记为 False）默认开启，可用 FAULT_ATTR_NEG_EVIDENCE=0 关闭做消融。

用法：
  python3 scripts/learn_cpt.py                       # 留出法 70/30（v1 行为）
  python3 scripts/learn_cpt.py --kfold 5             # 5 折交叉验证 + 校准 + 报告（B-1）
  python3 scripts/learn_cpt.py --kfold 5 --subset "dataset_(589).xlsx"   # 只看 589 公开基准子集
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.fault_attribution import (  # noqa: E402
    FAULT_IDS,
    SYMPTOM_IDS,
    FaultBayesianNetwork,
    derive_gas_evidence,
)
from src.tools.fault_attribution import _FAULT_SYMPTOM_CPT as _EXPERT_CPT  # noqa: E402

ALPHA = 1.0  # 拉普拉斯平滑系数
EXPERT_SHRINK = float(os.environ.get("EXPERT_SHRINK", "0.5"))  # 无数据征兆的专家 CPT 向 0.5 收缩系数 κ
T_GRID = [round(0.5 + 0.1 * i, 2) for i in range(0, 41)]          # 0.5 .. 4.5
W_GRID = [round(0.1 * i, 2) for i in range(0, 11)]                # 0.0 .. 1.0
N_BINS = 15


# ──────────────────────────────────────────────────────────────
# 数据与参数学习
# ──────────────────────────────────────────────────────────────

def load_records(jsonl: Path) -> list[dict]:
    recs = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def evidence_set(rec: dict) -> set[str]:
    """返回该记录中为 True 的征兆集合。"""
    return {k for k, v in (rec.get("evidence") or {}).items() if v}


def learn(records: list[dict]) -> dict:
    """从记录中统计先验与 CPT（含拉普拉斯平滑）。"""
    n = len(records)
    fault_counter = Counter(r["fault_label"] for r in records)

    prior = {}
    for f in FAULT_IDS:
        prior[f] = (fault_counter.get(f, 0) + ALPHA) / (n + ALPHA * len(FAULT_IDS))

    pos = defaultdict(lambda: defaultdict(int))
    n_f = Counter()
    total_pos = defaultdict(int)
    for r in records:
        f = r["fault_label"]
        ev = evidence_set(r)
        n_f[f] += 1
        for s in SYMPTOM_IDS:
            if s in ev:
                pos[f][s] += 1
                total_pos[s] += 1

    # 数据中从未出现过的征兆（CO / 油温 / 振动等，本数据集不含这些字段）：
    # 专家 CPT 未经数据验证，直接保留会在 EIG 中压过数据学到的气体征兆（专家值普遍比数据值更「自信」）。
    # 处理：写入向 0.5 收缩的专家值 p' = 0.5 + κ·(p - 0.5)（κ=EXPERT_SHRINK，默认 0.5，等价于对无数据支持的
    # 参数施加弱信息先验），并在 meta 中登记。κ=1 即原专家值。
    learnable = [s for s in SYMPTOM_IDS if total_pos[s] > 0]
    unobserved = [s for s in SYMPTOM_IDS if total_pos[s] == 0]

    cpt = {}
    for f in FAULT_IDS:
        nf = n_f.get(f, 0)
        n_not_f = n - nf
        row = {}
        for s in learnable:
            p_true = (pos[f][s] + ALPHA) / (nf + 2 * ALPHA)
            neg_count = total_pos[s] - pos[f][s]
            p_false = (neg_count + ALPHA) / (n_not_f + 2 * ALPHA)
            row[s] = {"p_true": round(p_true, 5), "p_false": round(p_false, 5)}
        for s in unobserved:
            exp_row = _EXPERT_CPT.get(f, {}).get(s)
            p_exp = exp_row.p_true if exp_row else 0.05
            p_shr = 0.5 + EXPERT_SHRINK * (p_exp - 0.5)
            row[s] = {"p_true": round(p_shr, 5), "p_false": round(exp_row.p_false if exp_row else 0.01, 5),
                      "source": "expert_shrunk"}
        cpt[f] = row

    return {
        "meta": {
            "n_samples": n,
            "alpha": ALPHA,
            "learned_symptoms": learnable,
            "expert_kept_symptoms": unobserved,
            "expert_shrink": EXPERT_SHRINK,
            "fault_distribution": dict(fault_counter),
        },
        "prior": {k: round(v, 5) for k, v in prior.items()},
        "cpt": cpt,
    }


# ──────────────────────────────────────────────────────────────
# 推理封装（直接用引擎对象，不走环境变量 / reload）
# ──────────────────────────────────────────────────────────────

def _record_inputs(rec: dict, engine: FaultBayesianNetwork) -> tuple[dict[str, bool], dict[str, Any]]:
    """把一条记录转成引擎输入：evidence（含负观测）与 symptom_context（DGA 数值）。"""
    g = rec["dga_data"]
    gases = {k: g.get(k) for k in ("H2", "CH4", "C2H2", "C2H4", "C2H6")}
    ev = dict(rec.get("evidence") or {})
    for s, v in derive_gas_evidence(gases, negative=engine.use_negative_evidence).items():
        ev.setdefault(s, v)
    ctx = {k: (v or 0.0) for k, v in gases.items()}
    return ev, ctx


def predict_probs(engine: FaultBayesianNetwork, records: list[dict]) -> list[dict[str, float]]:
    """返回每条记录的（融合 + 温度缩放后）故障分布。"""
    out = []
    for r in records:
        ev, ctx = _record_inputs(r, engine)
        out.append(engine.posterior_only(ev, ctx)["probs"])
    return out


def fused_components(engine: FaultBayesianNetwork, records: list[dict]) -> list[tuple[dict[str, float], dict[str, float]]]:
    """返回每条记录的 (bayes_probs, rule_boost_by_fault)，供融合权重搜索时快速重算。"""
    out = []
    for r in records:
        ev, ctx = _record_inputs(r, engine)
        bayes = engine._compute_posterior(ev)  # noqa: SLF001
        dga = engine._analyze_dga(ctx)         # noqa: SLF001
        boost: dict[str, float] = {}
        for fid in engine.faults:
            b = 0.0
            for rule in dga.get("matched_rules", []):
                if fid in rule["related_faults"]:
                    b = max(b, rule["confidence"])
            boost[fid] = b
        out.append((bayes, boost))
    return out


def fuse_with_weights(bayes: dict[str, float], boost: dict[str, float], w: dict[str, float] | float) -> dict[str, float]:
    fused = {}
    for fid, bp in bayes.items():
        wf = w if isinstance(w, float) else w.get(fid, FaultBayesianNetwork.DEFAULT_FUSION_WEIGHT)
        b = boost.get(fid, 0.0)
        fused[fid] = wf * bp + (1 - wf) * b if b > 0 else bp
    z = sum(fused.values()) or 1.0
    return {k: v / z for k, v in fused.items()}


def temperature_scale(probs: dict[str, float], T: float) -> dict[str, float]:
    if abs(T - 1.0) < 1e-9:
        return probs
    s = {k: max(v, 1e-12) ** (1.0 / T) for k, v in probs.items()}
    z = sum(s.values()) or 1.0
    return {k: v / z for k, v in s.items()}


# ──────────────────────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────────────────────

def metrics(probs_list: list[dict[str, float]], labels: list[str]) -> dict[str, Any]:
    n = len(labels)
    top1 = top3 = 0
    nll = 0.0
    brier = 0.0
    # ECE：按 Top-1 置信度分 15 桶
    bins = [[0, 0.0, 0.0] for _ in range(N_BINS)]  # count, sum_conf, sum_correct
    for probs, y in zip(probs_list, labels):
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        pred, conf = ranked[0]
        correct = int(pred == y)
        top1 += correct
        top3 += int(y in [k for k, _ in ranked[:3]])
        nll -= math.log(max(probs.get(y, 0.0), 1e-12))
        brier += sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in probs.items())
        b = min(int(conf * N_BINS), N_BINS - 1)
        bins[b][0] += 1
        bins[b][1] += conf
        bins[b][2] += correct
    ece = 0.0
    reliability = []
    for i, (cnt, sc, sco) in enumerate(bins):
        if cnt == 0:
            reliability.append({"bin": i, "n": 0, "conf": None, "acc": None})
            continue
        acc = sco / cnt
        conf = sc / cnt
        ece += (cnt / n) * abs(acc - conf)
        reliability.append({"bin": i, "n": cnt, "conf": round(conf, 4), "acc": round(acc, 4)})
    return {
        "n": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "nll": nll / n,
        "brier": brier / n,
        "ece": ece,
        "reliability": reliability,
    }


def per_class_recall(probs_list: list[dict[str, float]], labels: list[str]) -> dict[str, float]:
    hit = Counter()
    tot = Counter()
    for probs, y in zip(probs_list, labels):
        pred = max(probs.items(), key=lambda kv: kv[1])[0]
        tot[y] += 1
        hit[y] += int(pred == y)
    return {k: hit[k] / tot[k] for k in sorted(tot)}


# ──────────────────────────────────────────────────────────────
# 交叉验证 + 校准
# ──────────────────────────────────────────────────────────────

def stratified_folds(records: list[dict], k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_label[r["fault_label"]].append(i)
    folds: list[list[int]] = [[] for _ in range(k)]
    for idxs in by_label.values():
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def search_fusion_weights(components, labels, faults: list[str]) -> dict[str, float]:
    """坐标下降：每个故障类别独立搜索 w_f 使验证折 NLL 最小（其余类别固定当前值）。"""
    w = {f: FaultBayesianNetwork.DEFAULT_FUSION_WEIGHT for f in faults}

    def nll_of(wd):
        s = 0.0
        for (bayes, boost), y in zip(components, labels):
            p = fuse_with_weights(bayes, boost, wd)
            s -= math.log(max(p.get(y, 0.0), 1e-12))
        return s / len(labels)

    best = nll_of(w)
    for _ in range(2):  # 两轮坐标下降足够收敛
        for f in faults:
            cur = w[f]
            for cand in W_GRID:
                w[f] = cand
                v = nll_of(w)
                if v < best - 1e-9:
                    best, cur = v, cand
            w[f] = cur
    return w


def search_temperature(probs_list, labels) -> float:
    best_T, best = 1.0, float("inf")
    for T in T_GRID:
        s = 0.0
        for p, y in zip(probs_list, labels):
            q = temperature_scale(p, T)
            s -= math.log(max(q.get(y, 0.0), 1e-12))
        if s < best - 1e-9:
            best, best_T = s, T
    return best_T


def run_kfold(records: list[dict], k: int, seed: int) -> dict[str, Any]:
    """返回逐折与平均指标、以及平均 T / 融合权重。"""
    eval_recs = [r for r in records if r["fault_label"] != "normal"]
    folds = stratified_folds(records, k, seed)
    fold_reports = []
    Ts: list[float] = []
    Ws: list[dict[str, float]] = []

    for fi in range(k):
        val_idx = set(folds[fi])
        train = [r for i, r in enumerate(records) if i not in val_idx]
        val = [r for i, r in enumerate(records) if i in val_idx and r["fault_label"] != "normal"]
        random.Random(seed + fi).shuffle(val)  # 折内打乱，保证拟合半折 / 评估半折标签分布一致
        labels = [r["fault_label"] for r in val]

        params = learn(train)
        # (a) 专家默认（对照）
        expert = FaultBayesianNetwork(params={})
        expert.params_source = "expert_default"
        m_expert = metrics(predict_probs(expert, val), labels)

        # (b) 学习 CPT，固定 0.7，未校准
        eng = FaultBayesianNetwork(params=params)
        comps = fused_components(eng, val)
        base_probs = [fuse_with_weights(b, bo, 0.7) for b, bo in comps]
        m_base = metrics(base_probs, labels)

        # (c) + 按类别融合权重（在本折验证集上搜索 → 报告为「拟合折」指标；正式指标见下方 held-out）
        # 为避免在同一折上既拟合又评估，采用「折内二分」：验证折前半拟合 T/w，后半评估。
        half = len(val) // 2
        fit_c, fit_y = comps[:half], labels[:half]
        ev_c, ev_y = comps[half:], labels[half:]
        w_f = search_fusion_weights(fit_c, fit_y, eng.faults)
        fit_probs_w = [fuse_with_weights(b, bo, w_f) for b, bo in fit_c]
        T = search_temperature(fit_probs_w, fit_y)
        T_base = search_temperature([fuse_with_weights(b, bo, 0.7) for b, bo in fit_c], fit_y)
        Ts.append(T)
        Ws.append(w_f)

        ev_base = [fuse_with_weights(b, bo, 0.7) for b, bo in ev_c]
        ev_w = [fuse_with_weights(b, bo, w_f) for b, bo in ev_c]
        ev_wT = [temperature_scale(p, T) for p in ev_w]
        ev_T_only = [temperature_scale(p, T_base) for p in ev_base]
        m_ev_base = metrics(ev_base, ev_y)
        m_ev_w = metrics(ev_w, ev_y)
        m_ev_T = metrics(ev_T_only, ev_y)
        m_ev_wT = metrics(ev_wT, ev_y)

        fold_reports.append({
            "fold": fi,
            "n_train": len(train),
            "n_val": len(val),
            "n_heldout": len(ev_y),
            "T": T,
            "fusion_weights": w_f,
            "expert_full_val": m_expert,
            "learned_w0.7_full_val": m_base,
            "heldout": {
                "learned_w0.7": m_ev_base,
                "learned_wf": m_ev_w,
                "learned_w0.7_T": m_ev_T,
                "learned_wf_T": m_ev_wT,
            },
            "per_class_recall_heldout_wf_T": per_class_recall(ev_wT, ev_y),
        })
        print(f"[fold {fi}] n_val={len(val)} T={T:.2f} | expert Top1={m_expert['top1']:.3f} ECE={m_expert['ece']:.3f}"
              f" | learned Top1={m_base['top1']:.3f} ECE={m_base['ece']:.3f}"
              f" | heldout w0.7→wf+T: Top1 {m_ev_base['top1']:.3f}→{m_ev_wT['top1']:.3f}"
              f" ECE {m_ev_base['ece']:.3f}→{m_ev_wT['ece']:.3f} NLL {m_ev_base['nll']:.3f}→{m_ev_wT['nll']:.3f}")

    def avg(getter):
        vals = [getter(fr) for fr in fold_reports]
        return sum(vals) / len(vals)

    keys = ["top1", "top3", "nll", "brier", "ece"]
    summary = {
        "expert_full_val": {kk: avg(lambda fr: fr["expert_full_val"][kk]) for kk in keys},
        "learned_w0.7_full_val": {kk: avg(lambda fr: fr["learned_w0.7_full_val"][kk]) for kk in keys},
        "heldout": {
            name: {kk: avg(lambda fr, name=name: fr["heldout"][name][kk]) for kk in keys}
            for name in ("learned_w0.7", "learned_wf", "learned_w0.7_T", "learned_wf_T")
        },
    }
    T_avg = sum(Ts) / len(Ts)
    W_avg = {f: sum(w[f] for w in Ws) / len(Ws) for f in FAULT_IDS}
    # 汇总可靠性图（held-out 拼接）
    rel_before = defaultdict(lambda: [0, 0.0, 0.0])
    rel_after = defaultdict(lambda: [0, 0.0, 0.0])
    for fr in fold_reports:
        for name, tgt in (("learned_w0.7", rel_before), ("learned_wf_T", rel_after)):
            for b in fr["heldout"][name]["reliability"]:
                if b["n"]:
                    tgt[b["bin"]][0] += b["n"]
                    tgt[b["bin"]][1] += b["conf"] * b["n"]
                    tgt[b["bin"]][2] += b["acc"] * b["n"]

    def rel_list(d):
        out = []
        for i in range(N_BINS):
            cnt, sc, sa = d[i]
            out.append({"bin": i, "n": cnt, "conf": (sc / cnt) if cnt else None, "acc": (sa / cnt) if cnt else None})
        return out

    return {
        "k": k,
        "seed": seed,
        "n_records": len(records),
        "n_eval_records": len(eval_recs),
        "folds": fold_reports,
        "summary": summary,
        "T_avg": T_avg,
        "T_per_fold": Ts,
        "fusion_weights_avg": W_avg,
        "reliability_before": rel_list(rel_before),
        "reliability_after": rel_list(rel_after),
    }


# ──────────────────────────────────────────────────────────────
# 报告与图
# ──────────────────────────────────────────────────────────────

def plot_reliability(cv: dict[str, Any], out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib 不可用，跳过可靠性图: {exc}")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, key, title in ((axes[0], "reliability_before", "Before (learned CPT, w=0.7)"),
                           (axes[1], "reliability_after", "After (per-class w_f + temperature)")):
        xs, ys, ns = [], [], []
        for b in cv[key]:
            if b["n"]:
                xs.append(b["conf"]); ys.append(b["acc"]); ns.append(b["n"])
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.bar(xs, ys, width=1 / N_BINS * 0.9, alpha=0.6, label="accuracy")
        ax.scatter(xs, ys, s=[max(8, n / 5) for n in ns], c="tab:red", zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Top-1 confidence"); ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Empirical accuracy")
    ece_b = cv["summary"]["heldout"]["learned_w0.7"]["ece"]
    ece_a = cv["summary"]["heldout"]["learned_wf_T"]["ece"]
    fig.suptitle(f"Reliability diagram (5-fold held-out)  ECE {ece_b:.3f} → {ece_a:.3f}", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def write_report(cv: dict[str, Any], out_md: Path, fig_rel: Path | None, subset_tag: str, neg_ev: bool,
                 extra_sections: list[str] | None = None) -> None:
    s = cv["summary"]

    def row(name, m):
        return f"| {name} | {m['top1']:.3f} | {m['top3']:.3f} | {m['nll']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} |"

    lines = [
        "# 故障归因引擎校准报告（B-1）",
        "",
        f"> 生成：`python3 scripts/learn_cpt.py --kfold {cv['k']} --seed {cv['seed']}`"
        + (f" `--subset {subset_tag}`" if subset_tag else ""),
        f"> 数据：`data/real/dga/dga_records.jsonl` 共 {cv['n_records']} 条，其中非 normal 评测样本 {cv['n_eval_records']} 条"
        f"（引擎不输出 normal，口径与 `eval_fault_attribution.py` 一致）。",
        f"> 负观测（低于注意值的气体征兆记为 False）：{'开' if neg_ev else '关'}。",
        "",
        "## 1. 方法",
        "",
        "- 参数学习：按故障类别分层 5 折；每折用 4 折统计先验与 CPT（拉普拉斯 α=1）。",
        "- 校准参数拟合与评估分离：验证折按记录顺序二分，前半拟合按类别融合权重 w_f（坐标下降网格 0.0~1.0，目标 NLL）"
        " 与温度 T（网格 0.5~4.5，目标 NLL），后半（held-out）评估。",
        "- 指标：Top-1 / Top-3、NLL、Brier、ECE（Top-1 置信度 15 桶）。",
        "",
        "## 2. 5 折平均：全验证折（专家 CPT vs 学习 CPT，固定 w=0.7）",
        "",
        "| 配置 | Top-1 | Top-3 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
        row("专家 CPT（v1 默认）", s["expert_full_val"]),
        row("学习 CPT，w=0.7", s["learned_w0.7_full_val"]),
        "",
        "## 3. 5 折平均：held-out 半折（校准消融）",
        "",
        "| 配置 | Top-1 | Top-3 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
        row("学习 CPT，w=0.7（校准前）", s["heldout"]["learned_w0.7"]),
        row("+ 按类别 w_f", s["heldout"]["learned_wf"]),
        row("+ 温度 T（w=0.7）", s["heldout"]["learned_w0.7_T"]),
        row("+ 按类别 w_f + 温度 T（校准后）", s["heldout"]["learned_wf_T"]),
        "",
        f"- 温度 T 逐折：{', '.join(f'{t:.2f}' for t in cv['T_per_fold'])}；平均 **{cv['T_avg']:.2f}**（写入 `learned_params.json.calibration.T`）。",
        "- 融合权重 w_f（贝叶斯占比，5 折平均，写入 `learned_params.json.fusion_weights`）：",
        "",
        "| 故障 | w_f |",
        "|---|---:|",
    ]
    for f, w in cv["fusion_weights_avg"].items():
        lines.append(f"| `{f}` | {w:.2f} |")
    lines += ["", "## 4. 可靠性图", ""]
    if fig_rel:
        lines.append(f"![reliability]({os.path.relpath(fig_rel, out_md.parent)})")
    lines += ["", "| 桶 | n（前） | conf（前） | acc（前） | n（后） | conf（后） | acc（后） |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for b0, b1 in zip(cv["reliability_before"], cv["reliability_after"]):
        f0 = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {b0['bin']} | {b0['n']} | {f0(b0['conf'])} | {f0(b0['acc'])} | {b1['n']} | {f0(b1['conf'])} | {f0(b1['acc'])} |")
    lines += ["", "## 5. 逐折明细", "", "| 折 | n_val | n_heldout | T | Top-1 前→后 | ECE 前→后 | NLL 前→后 |", "|---:|---:|---:|---:|---|---|---|"]
    for fr in cv["folds"]:
        a, b = fr["heldout"]["learned_w0.7"], fr["heldout"]["learned_wf_T"]
        lines.append(f"| {fr['fold']} | {fr['n_val']} | {fr['n_heldout']} | {fr['T']:.2f} | {a['top1']:.3f}→{b['top1']:.3f}"
                     f" | {a['ece']:.3f}→{b['ece']:.3f} | {a['nll']:.3f}→{b['nll']:.3f} |")
    lines += ["", "### 各类召回（held-out，校准后，逐折）", ""]
    classes = sorted({c for fr in cv["folds"] for c in fr["per_class_recall_heldout_wf_T"]})
    lines.append("| 折 | " + " | ".join(f"`{c}`" for c in classes) + " |")
    lines.append("|---:|" + "---:|" * len(classes))
    for fr in cv["folds"]:
        pr = fr["per_class_recall_heldout_wf_T"]
        lines.append(f"| {fr['fold']} | " + " | ".join(f"{pr.get(c, float('nan')):.3f}" for c in classes) + " |")
    if extra_sections:
        lines += [""] + extra_sections
    lines += [
        "",
        "## 6. 说明与局限",
        "",
        "- 数据为文献汇编 DGA（见 `docs/data_inventory.md`），标签仅有 4 个故障类（局部放电 / 匝间短路 / 过载过热 / 绝缘老化）+ normal；"
        "引擎 8 类中的另 4 类（绕组变形 / 铁芯接地 / 套管 / 分接开关）无正样本，其先验被平滑到接近 0，Top-1 不会落在这些类上。",
        "- 朴素贝叶斯独立假设使多征兆同时出现时后验过尖，这是温度 T>1 的直接原因；报告的 ECE 下降即对此的修正。",
        f"- 数据集不含 CO / CO2 / 油温 / 绕组温度 / 负载 / 振动 / 局放告警字段，这 7 个征兆的 CPT 无法从数据学习；"
        f"写入向 0.5 收缩的专家值 p' = 0.5 + κ(p − 0.5)，κ = {EXPERT_SHRINK}（`EXPERT_SHRINK`），"
        "并在 CPT 条目标记 `source: expert_shrunk`。κ 对本报告的准确率 / ECE 无影响（评测数据不含这些征兆），"
        "但直接影响 B-2 EIG 对现场征兆的推荐强度，其敏感性见 `docs/eig_sanity_check.md`。",
        "- 校准参数（T、w_f）为 5 折平均值，最终 prior / CPT 用全量数据重学；线上引擎加载 `learned_params.json` 时自动生效。",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# 留出法（v1 兼容）
# ──────────────────────────────────────────────────────────────

def evaluate_holdout(records: list[dict], params: dict | None) -> tuple[float, float, int]:
    eng = FaultBayesianNetwork(params=params if params is not None else {})
    val = [r for r in records if r["fault_label"] != "normal"]
    labels = [r["fault_label"] for r in val]
    m = metrics(predict_probs(eng, val), labels)
    return m["top1"], m["top3"], m["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "data/real/dga/dga_records.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/real/dga/learned_params.json"))
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--kfold", type=int, default=0, help=">1 时启用分层 k 折交叉验证 + 校准（B-1）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", default="", help="只用 source 字段等于该值的记录（例如 dataset_(589).xlsx）")
    ap.add_argument("--report", default=str(ROOT / "docs/attribution_calibration.md"))
    ap.add_argument("--fig", default=str(ROOT / "docs/figures/fig_b1_reliability.png"))
    ap.add_argument("--no-write-params", action="store_true", help="只评测，不覆盖 learned_params.json")
    args = ap.parse_args()

    records = load_records(Path(args.jsonl))
    if args.subset:
        records = [r for r in records if r.get("source") == args.subset]
    print(f"载入记录: {len(records)} 条" + (f"（subset={args.subset}）" if args.subset else ""))
    neg_ev = os.environ.get("FAULT_ATTR_NEG_EVIDENCE", "1") not in ("0", "false", "False")

    if args.kfold and args.kfold > 1:
        cv = run_kfold(records, args.kfold, args.seed)
        s = cv["summary"]
        print("\n================ 5 折平均（held-out 半折） ================")
        for name in ("learned_w0.7", "learned_wf", "learned_w0.7_T", "learned_wf_T"):
            m = s["heldout"][name]
            print(f"{name:18s} Top1={m['top1']:.3f} Top3={m['top3']:.3f} NLL={m['nll']:.3f} Brier={m['brier']:.3f} ECE={m['ece']:.3f}")
        print(f"T_avg={cv['T_avg']:.2f}  w_f={ {k: round(v, 2) for k, v in cv['fusion_weights_avg'].items()} }")

        # 589 公开基准子集单列（A-3 要求），仅在全量运行时附加
        extra: list[str] = []
        if not args.subset and neg_ev:
            # 负观测消融：关闭负观测重跑一次 5 折（不写参数）
            print("\n[ablation] 关闭负观测重跑 5 折 …")
            os.environ["FAULT_ATTR_NEG_EVIDENCE"] = "0"
            try:
                cv_off = run_kfold(records, args.kfold, args.seed)
            finally:
                os.environ.pop("FAULT_ATTR_NEG_EVIDENCE", None)
            so = cv_off["summary"]["heldout"]
            extra += [
                "## 附：负观测消融（held-out 半折，5 折平均）",
                "",
                "| 负观测 | 配置 | Top-1 | Top-3 | NLL | Brier | ECE |",
                "|---|---|---:|---:|---:|---:|---:|",
                f"| 关（v1） | 学习 CPT，w=0.7 | {so['learned_w0.7']['top1']:.3f} | {so['learned_w0.7']['top3']:.3f} | {so['learned_w0.7']['nll']:.3f} | {so['learned_w0.7']['brier']:.3f} | {so['learned_w0.7']['ece']:.3f} |",
                f"| 关（v1） | + w_f + T | {so['learned_wf_T']['top1']:.3f} | {so['learned_wf_T']['top3']:.3f} | {so['learned_wf_T']['nll']:.3f} | {so['learned_wf_T']['brier']:.3f} | {so['learned_wf_T']['ece']:.3f} |",
                f"| 开（v2） | 学习 CPT，w=0.7 | {s['heldout']['learned_w0.7']['top1']:.3f} | {s['heldout']['learned_w0.7']['top3']:.3f} | {s['heldout']['learned_w0.7']['nll']:.3f} | {s['heldout']['learned_w0.7']['brier']:.3f} | {s['heldout']['learned_w0.7']['ece']:.3f} |",
                f"| 开（v2） | + w_f + T | {s['heldout']['learned_wf_T']['top1']:.3f} | {s['heldout']['learned_wf_T']['top3']:.3f} | {s['heldout']['learned_wf_T']['nll']:.3f} | {s['heldout']['learned_wf_T']['brier']:.3f} | {s['heldout']['learned_wf_T']['ece']:.3f} |",
                "",
            ]
            print(f"  neg-off: w0.7 Top1={so['learned_w0.7']['top1']:.3f} ECE={so['learned_w0.7']['ece']:.3f}"
                  f" | wf+T Top1={so['learned_wf_T']['top1']:.3f} ECE={so['learned_wf_T']['ece']:.3f}")
        if not args.subset:
            sub = [r for r in records if r.get("source") == "dataset_(589).xlsx"]
            if len(sub) >= 50:
                print(f"\n[subset dataset_(589).xlsx] n={len(sub)}，用全量学习参数 + 校准做外部检验")
                full_params = learn(records)
                full_params["fusion_weights"] = {k: round(v, 3) for k, v in cv["fusion_weights_avg"].items()}
                full_params["calibration"] = {"method": "temperature", "T": round(cv["T_avg"], 3)}
                eng_cal = FaultBayesianNetwork(params=full_params)
                eng_raw = FaultBayesianNetwork(params={"prior": full_params["prior"], "cpt": full_params["cpt"]})
                subv = [r for r in sub if r["fault_label"] != "normal"]
                labels = [r["fault_label"] for r in subv]
                m_raw = metrics(predict_probs(eng_raw, subv), labels)
                m_cal = metrics(predict_probs(eng_cal, subv), labels)
                extra = [
                    "## 附：R3 公开基准子集 `dataset_(589).xlsx` 单列",
                    "",
                    f"注意：该子集同时参与了全量参数学习（非严格外部检验），n={len(subv)}（非 normal）。",
                    "",
                    "| 配置 | Top-1 | Top-3 | NLL | Brier | ECE |",
                    "|---|---:|---:|---:|---:|---:|",
                    f"| 学习 CPT，w=0.7 | {m_raw['top1']:.3f} | {m_raw['top3']:.3f} | {m_raw['nll']:.3f} | {m_raw['brier']:.3f} | {m_raw['ece']:.3f} |",
                    f"| + w_f + T | {m_cal['top1']:.3f} | {m_cal['top3']:.3f} | {m_cal['nll']:.3f} | {m_cal['brier']:.3f} | {m_cal['ece']:.3f} |",
                ]
                print(f"  raw Top1={m_raw['top1']:.3f} ECE={m_raw['ece']:.3f} | cal Top1={m_cal['top1']:.3f} ECE={m_cal['ece']:.3f}")

        fig = Path(args.fig)
        ok = plot_reliability(cv, fig)
        write_report(cv, Path(args.report), fig if ok else None, args.subset, neg_ev, extra)
        print(f"\n报告: {args.report}" + (f"\n图: {fig}" if ok else ""))
        cv_json = Path(args.report).with_suffix(".json")
        with open(cv_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in cv.items() if k != "folds"} | {"folds": [
                {kk: vv for kk, vv in fr.items()} for fr in cv["folds"]]}, f, ensure_ascii=False, indent=1)
        print(f"CV 明细: {cv_json}")

        if not args.no_write_params and not args.subset:
            full_params = learn(records)
            full_params["fusion_weights"] = {k: round(v, 3) for k, v in cv["fusion_weights_avg"].items()}
            full_params["calibration"] = {
                "method": "temperature",
                "T": round(cv["T_avg"], 3),
                "kfold": cv["k"],
                "seed": cv["seed"],
                "negative_evidence": neg_ev,
                "ece_before": round(s["heldout"]["learned_w0.7"]["ece"], 4),
                "ece_after": round(s["heldout"]["learned_wf_T"]["ece"], 4),
                "top1_before": round(s["heldout"]["learned_w0.7"]["top1"], 4),
                "top1_after": round(s["heldout"]["learned_wf_T"]["top1"], 4),
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(full_params, f, ensure_ascii=False, indent=2)
            print(f"已写出全量学习参数 + 校准字段: {args.out}")
        return

    # ── v1 留出法 ──
    random.seed(args.seed)
    random.shuffle(records)
    n_test = int(len(records) * args.test_ratio)
    test = records[:n_test]
    train = records[n_test:]
    print(f"训练集: {len(train)} 条 | 测试集: {len(test)} 条 (test_ratio={args.test_ratio})")

    params = learn(train)
    print("\n================ 测试集评估对比 ================")
    a1, a3, n = evaluate_holdout(test, None)
    print(f"[学习前·专家CPT ] Top-1={a1:.1%}  Top-3={a3:.1%}  (n={n})")
    b1, b3, _ = evaluate_holdout(test, params)
    print(f"[学习后·数据CPT ] Top-1={b1:.1%}  Top-3={b3:.1%}  (n={n})")
    print(f"\n提升: Top-1 {a1:.1%} -> {b1:.1%} ({(b1 - a1) * 100:+.1f}pt)"
          f" | Top-3 {a3:.1%} -> {b3:.1%} ({(b3 - a3) * 100:+.1f}pt)")

    if not args.no_write_params:
        full_params = learn(records)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(full_params, f, ensure_ascii=False, indent=2)
        print(f"\n已用全量数据重学并覆盖参数文件，供线上加载: {args.out}")


if __name__ == "__main__":
    main()
