#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-5 主动规划对照实验（数据 D12，纯 CPU，无 LLM）。

在 1500 条部分观测样例上，让不同「问询策略」逐步揭示征兆（PartialObsEnv 代替用户），比较停止时的诊断质量与代价：
  random      每轮随机选一个未观测且可询问的征兆
  fixed       按 IEC 60599 / DL/T 722 常规检查顺序问（H2 → CH4 → C2H2 → C2H4 → C2H6 → 总烃 → 产气速率 → CO → CO2 → 现场）
  eig_greedy  每轮问期望信息增益 EIG 最大的征兆（不考虑成本，λ=0）
  eig_cost    每轮问 VoI = EIG − λ·cost 最大的征兆（本文方法，λ 取 symptom_cost.json 的 lambda_default）
  llm_*       需要真实 LLM，默认不跑；`--llm N` 时用 config.yaml 的模型在 N 条上跑完整工作流（decision=llm），结果单列

两种协议（同一条轨迹派生，策略选择不依赖停止规则，因此可事后套用任意停止配置）：
  A 固定预算：每个策略恰问 q 个问题（q = 0..q_max），画 Top-1 / 熵 随 q 的曲线（fig_b5_acc_vs_queries.png、fig_b5_entropy_curve.png）
  B 自适应停止：共享停止准则 H < τ_H 或 轮数 ≥ K 或 无候选；EIG 策略额外 VoI(EIG) < ε 即停。比较 Top-1 / Top-3 / 平均问询次数 / 成本 / 追问命中率 / ECE

消融：有 / 无校准参数（calibrated vs expert）、有 / 无成本项（eig_cost vs eig_greedy）、不同 τ_H。
显著性：问询次数差异用配对符号翻转置换检验；Top-1 差异用 McNemar 精确检验。
一致性：抽 N 条用完整 LangGraph 工作流（Planner decision=eig）跑，与轻量模拟的 Top-1 / 轮数比对。

用法：
  python3 scripts/eval/eval_active_planning.py                 # 全量 1500 条 + 消融 + 一致性核对
  python3 scripts/eval/eval_active_planning.py --limit 100     # 快速试跑
输出：docs/eval/active_planning_eval.md / .json、docs/figures/fig_b5_*.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
os.environ.setdefault("FAULT_ATTR_PARAMS", str(ROOT / "data/real/dga/learned_params.json"))

from src.tools.fault_attribution import FaultBayesianNetwork, SYMPTOM_IDS  # noqa: E402
from src.tools.eig import recommend, load_cost_table  # noqa: E402
import partial_obs_sim as sim  # noqa: E402

DOCS = ROOT / "docs" / "eval"
FIG_DIR = ROOT / "docs" / "figures"
REPORT_MD = DOCS / "active_planning_eval.md"
REPORT_JSON = DOCS / "active_planning_eval.json"
N_BINS = 15
STRATEGIES = ["random", "fixed", "eig_greedy", "eig_cost"]
STRATEGY_LABEL = {"random": "random", "fixed": "fixed (IEC 60599)", "eig_greedy": "eig_greedy (λ=0)",
                  "eig_cost": "eig_cost (VoI, ours)"}
# IEC 60599 / DL/T 722 常规顺序：五种特征气体 → 总烃 / 产气速率 → CO / CO2 → 现场检查
FIXED_ORDER = ["H2_elevated", "CH4_elevated", "C2H2_elevated", "C2H4_elevated", "C2H6_elevated",
               "TDCG_elevated", "gas_rate_rapid", "CO_elevated", "CO2_elevated",
               "winding_temp_elevated", "oil_temp_elevated", "load_elevated",
               "partial_discharge_alarm", "vibration_elevated"]
STOP_NAMES = {"entropy_below_tau": "熵低于 τ_H", "voi_below_eps": "VoI 低于 ε", "max_rounds": "达轮数上限",
              "no_candidates": "无候选", "budget_exhausted": "预算耗尽"}


# ──────────────────────────────────────────────────────────────
# 引擎 / 候选
# ──────────────────────────────────────────────────────────────

def make_engine(mode: str) -> FaultBayesianNetwork:
    if mode == "expert":
        e = FaultBayesianNetwork(params={})
        e.params_source = "expert_default"
        return e
    p = Path(os.environ["FAULT_ATTR_PARAMS"])
    e = FaultBayesianNetwork(params=json.loads(p.read_text(encoding="utf-8")))
    e.params_source = str(p)
    return e


def askable_symptoms(table: dict[str, Any]) -> list[str]:
    """D12 协议：只有 ask_user 类征兆可询问；call_tool 类（油温 / 负载）需要时序信号，D12 无此数据，视为不可获取。"""
    return [s for s in SYMPTOM_IDS if not str(table["symptoms"][s].get("how_to_obtain", "ask_user")).startswith("call_tool:")]


# ──────────────────────────────────────────────────────────────
# 轨迹生成（协议 A；协议 B 由轨迹事后派生）
# ──────────────────────────────────────────────────────────────

def _snapshot(engine: FaultBayesianNetwork, env: sim.PartialObsEnv, q: int, recs: list[dict], n_cands: int) -> dict[str, Any]:
    po = engine.posterior_only(env.evidence, env.context())
    return {
        "q": q, "entropy": round(po["entropy_bits"], 5), "probs": {k: round(v, 6) for k, v in po["probs"].items()},
        "top1": po["top1"], "n_cands": n_cands, "cost": env.total_cost, "n_queries": env.n_queries,
        "best_voi": recs[0]["voi"] if recs else None,
        "best_eig": max(r["eig"] for r in recs) if recs else None,
    }


def run_trajectory(sample: dict, strategy: str, engine: FaultBayesianNetwork, *, q_max: int, rng: random.Random,
                   askable: list[str], lam_ref: float) -> dict[str, Any]:
    env = sim.PartialObsEnv(sample)
    asked: set[str] = set()
    unavailable: set[str] = set()
    snaps: list[dict] = []
    picks: list[dict] = []
    for r in range(q_max + 1):
        cands = [s for s in askable if s not in env.evidence and s not in asked and s not in unavailable]
        recs: list[dict] = []
        if cands:
            rec = recommend(env.evidence, env.context(), engine=engine, lam=lam_ref, rounds_done=r,
                            candidates=cands, top_k=len(SYMPTOM_IDS), with_kg=False)
            recs = rec["recommendations"]  # 已按 VoI(λ_ref) 降序
        snaps.append(_snapshot(engine, env, r, recs, len(cands)))
        if r == q_max or not cands:
            break
        if strategy == "random":
            sym = rng.choice(cands)
        elif strategy == "fixed":
            sym = next(s for s in FIXED_ORDER if s in cands)
        elif strategy == "eig_greedy":
            sym = sorted(recs, key=lambda x: (-x["eig"], x["cost"], x["symptom"]))[0]["symptom"]
        elif strategy == "eig_cost":
            sym = recs[0]["symptom"]
        else:
            raise ValueError(strategy)
        ref_top2 = [x["symptom"] for x in recs[:2]]
        ans = env.answer(sym)
        asked.add(sym)
        if ans["answer"] is None:
            unavailable.add(sym)
        picks.append({"q": r + 1, "symptom": sym, "answer": ans["answer"], "cost": ans["cost"], "hit2": sym in ref_top2})
    return {"sim_id": sample["sim_id"], "level": sample["level"], "label": sample["fault_label"],
            "snaps": snaps, "picks": picks}


# ──────────────────────────────────────────────────────────────
# 协议 B：自适应停止 + 指标
# ──────────────────────────────────────────────────────────────

def stop_index(traj: dict, strategy: str, *, tau: float, eps: float, K: int) -> tuple[int, str]:
    for s in traj["snaps"]:
        r = s["q"]
        if s["entropy"] < tau:
            return r, "entropy_below_tau"
        if s["n_cands"] == 0:
            return r, "no_candidates"
        if strategy == "eig_cost" and s["best_voi"] is not None and s["best_voi"] < eps:
            return r, "voi_below_eps"
        if strategy == "eig_greedy" and s["best_eig"] is not None and s["best_eig"] < eps:
            return r, "voi_below_eps"
        if r >= K:
            return r, "max_rounds"
    return traj["snaps"][-1]["q"], "budget_exhausted"


def metrics(probs_list: list[dict[str, float]], labels: list[str]) -> dict[str, Any]:
    n = len(labels)
    top1 = top3 = 0
    nll = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(N_BINS)]
    for probs, y in zip(probs_list, labels):
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        pred, conf = ranked[0]
        c = int(pred == y)
        top1 += c
        top3 += int(y in [k for k, _ in ranked[:3]])
        nll -= math.log(max(probs.get(y, 0.0), 1e-12))
        b = min(int(conf * N_BINS), N_BINS - 1)
        bins[b][0] += 1
        bins[b][1] += conf
        bins[b][2] += c
    ece = sum((cnt / n) * abs(sco / cnt - sc / cnt) for cnt, sc, sco in bins if cnt)
    return {"n": n, "top1": top1 / n, "top3": top3 / n, "nll": nll / n, "ece": ece}


def evaluate_adaptive(trajs: list[dict], strategy: str, *, tau: float, eps: float, K: int) -> dict[str, Any]:
    probs, labels, nq, cost, stops, hits = [], [], [], [], Counter(), []
    per_level: dict[str, dict[str, list]] = {}
    correct_flags: list[int] = []
    for t in trajs:
        idx, reason = stop_index(t, strategy, tau=tau, eps=eps, K=K)
        s = t["snaps"][idx]
        probs.append(s["probs"]); labels.append(t["label"])
        nq.append(s["n_queries"]); cost.append(s["cost"]); stops[reason] += 1
        hits.extend(p["hit2"] for p in t["picks"][:idx])
        correct_flags.append(int(s["top1"] == t["label"]))
        pl = per_level.setdefault(t["level"], {"c": [], "q": []})
        pl["c"].append(correct_flags[-1]); pl["q"].append(s["n_queries"])
    m = metrics(probs, labels)
    m.update({
        "avg_queries": float(np.mean(nq)), "avg_cost": float(np.mean(cost)),
        "hit2": (sum(hits) / len(hits)) if hits else None, "n_picks": len(hits),
        "stop_reasons": dict(stops),
        "per_level": {lv: {"top1": float(np.mean(v["c"])), "avg_queries": float(np.mean(v["q"]))} for lv, v in sorted(per_level.items())},
        "_queries": nq, "_correct": correct_flags,
    })
    return m


def curve(trajs: list[dict], q_max: int) -> dict[str, list[float]]:
    top1, ent = [], []
    for q in range(q_max + 1):
        c = h = 0.0
        for t in trajs:
            s = t["snaps"][min(q, len(t["snaps"]) - 1)]
            c += int(s["top1"] == t["label"]); h += s["entropy"]
        top1.append(c / len(trajs)); ent.append(h / len(trajs))
    return {"top1": top1, "entropy": ent}


# ──────────────────────────────────────────────────────────────
# 配对检验
# ──────────────────────────────────────────────────────────────

def paired_permutation(a: list[float], b: list[float], n_perm: int = 20000, seed: int = 0) -> dict[str, float]:
    """配对符号翻转置换检验（H0：差值分布关于 0 对称），双侧 p 值。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    cnt = 0
    for start in range(0, n_perm, 2000):
        m = min(2000, n_perm - start)
        signs = rng.choice([-1.0, 1.0], size=(m, d.size))
        cnt += int(((signs * d).mean(axis=1).__abs__() >= obs - 1e-12).sum())
    return {"mean_diff": float(d.mean()), "p_value": (cnt + 1) / (n_perm + 1)}


def mcnemar_exact(a_correct: list[int], b_correct: list[int]) -> dict[str, float]:
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if not x and y)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b, "c": c, "p_value": p}


# ──────────────────────────────────────────────────────────────
# 完整工作流（一致性核对 / 可选 LLM 策略）
# ──────────────────────────────────────────────────────────────

NO_LLM_CFG = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 1, "planner_strategy": "active", "planner_decision": "eig",
                 "max_inquiry_rounds": 3, "attribution_mode": "calibrated"},
}


def run_workflow_strategy(samples: list[dict], cfg: dict, *, decision: str, K: int) -> list[dict[str, Any]]:
    from src.graph.state import AgentState
    from src.agents.planner import PlannerAgent
    from src.agents.retriever import RetrieverAgent
    from src.agents.generator import GeneratorAgent
    from src.agents.validator import ValidatorAgent
    from src.graph.workflow import run_diagnosis_workflow
    from src.tools.mcp_client import MCPClient
    from src.tools.fault_attribution import fault_attribution, configure_engine
    import logging
    logging.disable(logging.WARNING)

    configure_engine(cfg.get("workflow", {}).get("attribution_mode", "calibrated"))
    out = []
    for rec in samples:
        env = sim.PartialObsEnv(rec)
        mcp = MCPClient()
        mcp.register_tool("fault_attribution", fault_attribution)
        mcp.register_tool("rag_search", lambda query, **_: [])
        mcp.register_tool("kg_search", lambda **_: {"status": "no_match", "paths": []})
        planner = PlannerAgent(cfg, strategy="active", decision=decision)
        retriever = RetrieverAgent(cfg, mcp)
        state = AgentState(user_query="这台变压器油色谱有部分数据缺失，请判断故障类型。",
                           context={"dga": env.context(), "device_id": rec["sim_id"]}, max_iterations=1, max_inquiry_rounds=K)
        final = run_diagnosis_workflow(state, planner, retriever, generator=GeneratorAgent(cfg), validator=ValidatorAgent(cfg),
                                       answer_fn=env.answer)
        attr = None
        for tc in reversed(final.tool_calls):
            if tc.get("tool") == "fault_attribution" and tc.get("success"):
                attr = tc.get("result"); break
        out.append({"sim_id": rec["sim_id"], "label": rec["fault_label"], "top1": (attr or {}).get("primary_fault"),
                    "rounds": final.inquiry_rounds, "cost": final.inquiry_cost, "n_queries": env.n_queries,
                    "stop_reason": (final.inquiry_log[-1].get("stop_reason") if final.inquiry_log else None)})
    return out


# ──────────────────────────────────────────────────────────────
# 图表
# ──────────────────────────────────────────────────────────────

def plot_curves(curves: dict[str, dict[str, list[float]]], q_max: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    xs = list(range(q_max + 1))
    marks = {"random": "o--", "fixed": "s--", "eig_greedy": "^-", "eig_cost": "D-"}
    for key, ylabel, fname in (("top1", "Top-1 accuracy", "fig_b5_acc_vs_queries.png"),
                               ("entropy", "Mean posterior entropy H(F|E) / bit", "fig_b5_entropy_curve.png")):
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=160)
        for s in STRATEGIES:
            ax.plot(xs, curves[s][key], marks[s], label=STRATEGY_LABEL[s], ms=4, lw=1.3)
        ax.set_xlabel("Number of queries q")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────────────────

def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _row(name: str, m: dict[str, Any]) -> str:
    hit = f"{m['hit2']:.3f}" if m.get("hit2") is not None else "-"
    return (f"| {name} | {m['top1']:.3f} | {m['top3']:.3f} | {m['avg_queries']:.2f} | {m['avg_cost']:.2f} | {hit} | {m['ece']:.3f} | {m['nll']:.3f} |")


def write_report(res: dict[str, Any]) -> None:
    cfg = res["config"]
    main = res["adaptive"]["calibrated"]
    lines = [
        "# B-5 主动规划对照实验报告",
        "",
        f"- 生成时间：{res['generated_at']}；脚本 `scripts/eval/eval_active_planning.py`；数据 D12 `data/eval/d12/partial_obs.jsonl`"
        f"（{cfg['n_samples']} 条，sha256 `{cfg['d12_sha256'][:16]}…`）",
        f"- 归因引擎：`{cfg['params_source']}`（calibrated）与专家默认（expert）；候选征兆 {cfg['n_askable']} 个（`ask_user` 类；"
        "`call_tool` 类油温 / 负载需时序信号，D12 无此数据，不参与）",
        f"- 停止准则（协议 B）：H(F|E) < τ_H={cfg['tau']} bit、轮数 ≥ K={cfg['K']}、无候选；EIG 策略额外 VoI < ε={cfg['eps']}；λ={cfg['lam']}",
        f"- 所有策略确定性（random 固定 seed {cfg['seed']}）；不调用 LLM。`llm_free` 策略需真实模型，见第 6 节",
        "",
        "## 1. 策略",
        "",
        "| 策略 | 每轮选择 | 停止 |",
        "|---|---|---|",
        "| `random` | 未观测征兆中均匀随机 | τ_H / K / 无候选 |",
        "| `fixed` | IEC 60599 / DL/T 722 常规顺序：H2 → CH4 → C2H2 → C2H4 → C2H6 → 总烃 → 产气速率 → CO → CO2 → 现场 | 同上 |",
        "| `eig_greedy` | argmax EIG(s)（λ=0） | 同上 + EIG < ε |",
        "| `eig_cost`（本文） | argmax VoI(s) = EIG(s) − λ·cost(s) | 同上 + VoI < ε |",
        "",
        "## 2. 协议 B：自适应停止（calibrated 引擎）",
        "",
        "| 策略 | Top-1 | Top-3 | 平均问询 | 平均成本 | 追问命中率@2 | ECE | NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _row("无追问（q=0）", res["adaptive"]["baseline_q0"]),
    ]
    for s in STRATEGIES:
        lines.append(_row(f"`{s}`", main[s]))
    lines += [
        "",
        "追问命中率@2：所问征兆是否落在当轮 VoI 前 2（对 EIG 策略恒为 1，作为 random / fixed 的参照）。",
        "",
        "停止原因分布：",
        "",
        "| 策略 | " + " | ".join(STOP_NAMES[k] for k in ("entropy_below_tau", "voi_below_eps", "max_rounds", "no_candidates")) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in STRATEGIES:
        sr = main[s]["stop_reasons"]
        lines.append(f"| `{s}` | " + " | ".join(str(sr.get(k, 0)) for k in ("entropy_below_tau", "voi_below_eps", "max_rounds", "no_candidates")) + " |")
    lines += ["", "分档（遮蔽 1 / 2 / 3 种气体）：", "", "| 策略 | light Top-1 / 问询 | medium Top-1 / 问询 | heavy Top-1 / 问询 |", "|---|---:|---:|---:|"]
    for s in STRATEGIES:
        pl = main[s]["per_level"]
        lines.append(f"| `{s}` | " + " | ".join(f"{pl[lv]['top1']:.3f} / {pl[lv]['avg_queries']:.2f}" for lv in ("light", "medium", "heavy")) + " |")

    lines += ["", "## 3. 配对检验（calibrated，协议 B）", "",
              "| 对比 | 问询次数均差 | 置换检验 p | Top-1 不一致 (b, c) | McNemar p |", "|---|---:|---:|---|---:|"]
    for k, t in res["tests"].items():
        lines.append(f"| {k} | {t['queries']['mean_diff']:+.3f} | {_p(t['queries']['p_value'])} | ({t['top1']['b']}, {t['top1']['c']}) | {_p(t['top1']['p_value'])} |")
    lines += ["", "b = 前者对、后者错的条数；c 反之。问询次数均差为负表示前者问得更少。"]

    lines += ["", "## 4. 协议 A：固定预算曲线", "",
              f"![acc](../figures/fig_b5_acc_vs_queries.png)", "", f"![entropy](../figures/fig_b5_entropy_curve.png)", "",
              "| q | " + " | ".join(f"{s} Top-1" for s in STRATEGIES) + " | " + " | ".join(f"{s} H" for s in STRATEGIES) + " |",
              "|---:|" + "---:|" * (2 * len(STRATEGIES))]
    cv = res["curves"]["calibrated"]
    for q in range(cfg["q_max"] + 1):
        lines.append(f"| {q} | " + " | ".join(f"{cv[s]['top1'][q]:.3f}" for s in STRATEGIES) + " | " + " | ".join(f"{cv[s]['entropy'][q]:.3f}" for s in STRATEGIES) + " |")

    lines += ["", "## 5. 消融", "", "### 5.1 有 / 无校准参数（协议 B）", "",
              "| 引擎 | 策略 | Top-1 | Top-3 | 平均问询 | ECE |", "|---|---|---:|---:|---:|---:|"]
    for mode in ("expert", "calibrated"):
        for s in STRATEGIES:
            m = res["adaptive"][mode][s]
            lines.append(f"| {mode} | `{s}` | {m['top1']:.3f} | {m['top3']:.3f} | {m['avg_queries']:.2f} | {m['ece']:.3f} |")
    lines += ["", "### 5.2 有 / 无成本项", "",
              f"`eig_cost` 与 `eig_greedy` 的差别在于 λ·cost 惩罚：本数据集气体征兆成本 0、CO/CO2/产气速率/温度/振动成本 1、局放检测成本 3。",
              f"calibrated 下平均成本 eig_greedy {main['eig_greedy']['avg_cost']:.2f} vs eig_cost {main['eig_cost']['avg_cost']:.2f}，"
              f"Top-1 {main['eig_greedy']['top1']:.3f} vs {main['eig_cost']['top1']:.3f}。",
              "", "### 5.3 不同 τ_H（calibrated，`eig_cost`）", "",
              "| τ_H | Top-1 | Top-3 | 平均问询 | 平均成本 | 熵低于 τ 停止占比 |", "|---:|---:|---:|---:|---:|---:|"]
    for tau, m in res["tau_sweep"].items():
        n = sum(m["stop_reasons"].values())
        lines.append(f"| {tau} | {m['top1']:.3f} | {m['top3']:.3f} | {m['avg_queries']:.2f} | {m['avg_cost']:.2f} | {m['stop_reasons'].get('entropy_below_tau', 0) / n:.1%} |")

    lines += ["", "## 6. 完整工作流一致性核对与 LLM 策略", ""]
    wf = res.get("workflow_check")
    if wf:
        lines += [f"抽 {wf['n']} 条（每档均分）用完整 LangGraph 工作流（Planner `strategy=active, decision=eig`，PartialObsEnv 回答）运行：",
                  f"Top-1 {wf['top1']:.3f}，平均追问 {wf['avg_rounds']:.2f} 轮；与轻量模拟 `eig_cost` 在同一 {wf['n']} 条上 Top-1 {wf['sim_top1']:.3f}、"
                  f"平均问询 {wf['sim_avg_queries']:.2f}；Top-1 结论逐条一致 {wf['agree_top1']}/{wf['n']}，轮数一致 {wf['agree_rounds']}/{wf['n']}。",
                  "差异来源：工作流的候选集含 `call_tool` 类征兆（油温 / 负载），其 VoI 参与停止判断但在 D12 中无法获取被跳过。"]
    llm = res.get("llm_free")
    if llm:
        lines += ["", f"`llm_free`（Planner `decision=llm`，模型 `{llm['model']}`，{llm['n']} 条）：Top-1 {llm['top1']:.3f}，"
                  f"平均追问 {llm['avg_rounds']:.2f} 轮，成本 {llm['avg_cost']:.2f}。"]
    else:
        lines += ["", "`llm_free`（Planner 自由追问，`decision=llm`）需要真实 LLM，本次未运行；命令："
                  "`python3 scripts/eval/eval_active_planning.py --llm 100`（用 `config.yaml` 的模型，结果追加到本节）。"]

    c = res["conclusion"]
    lines += ["", "## 7. 结论", ""] + [f"- {x}" for x in c]
    lines += ["", "## 8. 局限", "",
              "- D12 现场征兆（CO / 温度 / 负载 / 振动 / 局放）无字段，询问返回「不可获取」并计成本；主动规划实际在 7 个气体征兆空间内进行，"
              "EIG 相对 fixed 的优势主要体现在「先问哪种气体」上。",
              "- 标签为文献汇编的四类（过热 / 局放 / 短路 / 绝缘老化），引擎 8 类中另 4 类无正样本，Top-1 上限受此约束。",
              "- 轻量模拟直接调用归因引擎与 EIG 模块，不经过 LLM；完整工作流一致性见第 6 节。"]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每档抽前 N 条（0=全量）")
    ap.add_argument("--q-max", type=int, default=6)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--workflow-check", type=int, default=30, help="完整工作流一致性核对条数（0 关闭）")
    ap.add_argument("--llm", type=int, default=0, help="用 config.yaml 的 LLM 跑 llm_free 策略的条数（默认 0 不跑）")
    ap.add_argument("--n-perm", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.time()
    table = load_cost_table()
    tau = args.tau if args.tau is not None else float(table["stop"]["tau_h_bits"])
    eps = args.eps if args.eps is not None else float(table["stop"]["eps_voi"])
    lam = float(table["lambda_default"])
    askable = askable_symptoms(table)

    all_samples = sim.load_d12()
    d12_sha = sim.sha256_of(all_samples)
    if args.limit:
        samples = [s for lv in sim.LEVELS for s in [x for x in all_samples if x["level"] == lv][:args.limit]]
    else:
        samples = all_samples
    labels = [s["fault_label"] for s in samples]
    print(f"D12 {len(samples)} 条；候选征兆 {len(askable)} 个；τ_H={tau} ε={eps} K={args.K} λ={lam}")

    engines = {"calibrated": make_engine("calibrated"), "expert": make_engine("expert")}
    trajs: dict[str, dict[str, list[dict]]] = {}
    for mode, eng in engines.items():
        trajs[mode] = {}
        for s in STRATEGIES:
            rng = random.Random(args.seed)
            tt = time.time()
            trajs[mode][s] = [run_trajectory(x, s, eng, q_max=args.q_max, rng=rng, askable=askable, lam_ref=lam) for x in samples]
            print(f"  [{mode}] {s:<11} 轨迹完成 {time.time() - tt:.1f}s")

    adaptive: dict[str, Any] = {}
    for mode in engines:
        adaptive[mode] = {s: evaluate_adaptive(trajs[mode][s], s, tau=tau, eps=eps, K=args.K) for s in STRATEGIES}
    q0 = [t["snaps"][0]["probs"] for t in trajs["calibrated"]["eig_cost"]]
    base = metrics(q0, labels)
    base.update({"avg_queries": 0.0, "avg_cost": 0.0, "hit2": None, "ece": base["ece"]})
    adaptive["baseline_q0"] = base

    curves = {mode: {s: curve(trajs[mode][s], args.q_max) for s in STRATEGIES} for mode in engines}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(curves["calibrated"], args.q_max)

    cal = adaptive["calibrated"]
    tests = {}
    for a, b in (("eig_cost", "fixed"), ("eig_cost", "random"), ("eig_greedy", "fixed"), ("eig_cost", "eig_greedy")):
        tests[f"`{a}` vs `{b}`"] = {
            "queries": paired_permutation(cal[a]["_queries"], cal[b]["_queries"], n_perm=args.n_perm, seed=args.seed),
            "top1": mcnemar_exact(cal[a]["_correct"], cal[b]["_correct"]),
        }

    tau_sweep = {}
    for tv in (0.5, 0.8, 1.0, 1.2, 1.5):
        tau_sweep[str(tv)] = evaluate_adaptive(trajs["calibrated"]["eig_cost"], "eig_cost", tau=tv, eps=eps, K=args.K)

    workflow_check = None
    if args.workflow_check:
        per = max(1, args.workflow_check // 3)
        sub = [s for lv in sim.LEVELS for s in [x for x in all_samples if x["level"] == lv][:per]]
        print(f"完整工作流一致性核对 {len(sub)} 条…")
        wf = run_workflow_strategy(sub, NO_LLM_CFG, decision="eig", K=args.K)
        eng = engines["calibrated"]
        sim_rows = []
        for x in sub:
            t = run_trajectory(x, "eig_cost", eng, q_max=args.q_max, rng=random.Random(args.seed), askable=askable, lam_ref=lam)
            idx, _ = stop_index(t, "eig_cost", tau=tau, eps=eps, K=args.K)
            sim_rows.append({"top1": t["snaps"][idx]["top1"], "n_queries": t["snaps"][idx]["n_queries"]})
        workflow_check = {
            "n": len(sub),
            "top1": sum(r["top1"] == r["label"] for r in wf) / len(wf),
            "avg_rounds": float(np.mean([r["rounds"] for r in wf])),
            "sim_top1": sum(r["top1"] == x["fault_label"] for r, x in zip(sim_rows, sub)) / len(sub),
            "sim_avg_queries": float(np.mean([r["n_queries"] for r in sim_rows])),
            "agree_top1": sum(r["top1"] == s["top1"] for r, s in zip(wf, sim_rows)),
            "agree_rounds": sum(r["rounds"] == s["n_queries"] for r, s in zip(wf, sim_rows)),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in wf)),
            "rows": wf,
        }
        print(f"  工作流 Top-1 {workflow_check['top1']:.3f} 平均 {workflow_check['avg_rounds']:.2f} 轮 | 模拟 Top-1 {workflow_check['sim_top1']:.3f}"
              f" 平均 {workflow_check['sim_avg_queries']:.2f} | 结论一致 {workflow_check['agree_top1']}/{len(sub)} 轮数一致 {workflow_check['agree_rounds']}/{len(sub)}")

    llm_free = None
    if args.llm:
        from src.utils.config import load_config
        cfg = load_config()
        cfg.setdefault("workflow", {}).update({"planner_strategy": "active", "planner_decision": "llm", "max_inquiry_rounds": args.K})
        per = max(1, args.llm // 3)
        sub = [s for lv in sim.LEVELS for s in [x for x in all_samples if x["level"] == lv][:per]]
        print(f"llm_free {len(sub)} 条（调用 LLM）…")
        rows = run_workflow_strategy(sub, cfg, decision="llm", K=args.K)
        llm_free = {"n": len(rows), "model": cfg["llms"]["default"].get("model_name", "?"),
                    "top1": sum(r["top1"] == r["label"] for r in rows) / len(rows),
                    "avg_rounds": float(np.mean([r["rounds"] for r in rows])),
                    "avg_cost": float(np.mean([r["cost"] for r in rows])), "rows": rows}

    # 结论
    ec, fx, rd, eg = cal["eig_cost"], cal["fixed"], cal["random"], cal["eig_greedy"]
    tq_f = tests["`eig_cost` vs `fixed`"]; tq_r = tests["`eig_cost` vs `random`"]
    conclusion = [
        f"无追问基线 Top-1 {base['top1']:.3f}；自适应停止后 eig_cost {ec['top1']:.3f}（平均 {ec['avg_queries']:.2f} 问）、"
        f"fixed {fx['top1']:.3f}（{fx['avg_queries']:.2f} 问）、random {rd['top1']:.3f}（{rd['avg_queries']:.2f} 问）、eig_greedy {eg['top1']:.3f}（{eg['avg_queries']:.2f} 问）。",
        f"eig_cost 相对 fixed：问询次数均差 {tq_f['queries']['mean_diff']:+.3f}（置换检验 p={_p(tq_f['queries']['p_value'])}），"
        f"Top-1 McNemar p={_p(tq_f['top1']['p_value'])}；相对 random：均差 {tq_r['queries']['mean_diff']:+.3f}（p={_p(tq_r['queries']['p_value'])}），McNemar p={_p(tq_r['top1']['p_value'])}。",
        f"固定预算下（协议 A）q=1 时 Top-1：eig_cost {curves['calibrated']['eig_cost']['top1'][1]:.3f} / fixed {curves['calibrated']['fixed']['top1'][1]:.3f} / random {curves['calibrated']['random']['top1'][1]:.3f}；"
        f"q=2 时 {curves['calibrated']['eig_cost']['top1'][2]:.3f} / {curves['calibrated']['fixed']['top1'][2]:.3f} / {curves['calibrated']['random']['top1'][2]:.3f}。",
        f"校准消融：expert 引擎下 eig_cost Top-1 {adaptive['expert']['eig_cost']['top1']:.3f} ECE {adaptive['expert']['eig_cost']['ece']:.3f}，"
        f"calibrated 下 {ec['top1']:.3f} / {ec['ece']:.3f}。",
    ]
    verdict_q = ec["avg_queries"] < fx["avg_queries"] and tq_f["queries"]["p_value"] < 0.05
    verdict_acc = ec["top1"] >= fx["top1"] - 0.01
    conclusion.append(("验收通过" if (verdict_q and verdict_acc) else "验收未完全达成") +
                      f"：eig_cost 在 Top-1 不低于 fixed（差 {ec['top1'] - fx['top1']:+.3f}）的前提下平均问询次数"
                      f"{'低于' if ec['avg_queries'] < fx['avg_queries'] else '不低于'} fixed（{ec['avg_queries']:.2f} vs {fx['avg_queries']:.2f}，"
                      f"减少 {(1 - ec['avg_queries'] / fx['avg_queries']):.0%}，p={_p(tq_f['queries']['p_value'])}）；平均获取成本 {ec['avg_cost']:.2f} vs {fx['avg_cost']:.2f}。"
                      "llm_free 对照待 LLM 可用时补跑。")

    def strip(m: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in m.items() if not k.startswith("_")}

    res = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"n_samples": len(samples), "d12_sha256": d12_sha, "q_max": args.q_max, "K": args.K, "tau": tau, "eps": eps,
                   "lam": lam, "seed": args.seed, "n_askable": len(askable), "askable": askable,
                   "params_source": engines["calibrated"].params_source},
        "adaptive": {mode: {s: strip(m) for s, m in adaptive[mode].items()} for mode in engines} | {"baseline_q0": strip(base)},
        "curves": curves, "tests": tests, "tau_sweep": {k: strip(v) for k, v in tau_sweep.items()},
        "workflow_check": workflow_check, "llm_free": llm_free, "conclusion": conclusion,
        "elapsed_s": round(time.time() - t0, 1),
    }
    REPORT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(res)
    print()
    for c in conclusion:
        print("-", c)
    print(f"\n报告：{REPORT_MD}  图：{FIG_DIR / 'fig_b5_acc_vs_queries.png'}  用时 {res['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
