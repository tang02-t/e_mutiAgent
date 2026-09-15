"""
期望信息增益（EIG）模块 —— 主线一「不确定性驱动的主动诊断规划」（B-2）。

给定当前观测 E（含正 / 负观测）与归因引擎的后验 P(F|E)，对每个未观测征兆 s 计算

    EIG(s) = H(F|E) - Σ_v P(s=v|E) · H(F|E, s=v),      v ∈ {True, False}
    P(s=v|E) = Σ_F P(s=v|F) · P(F|E)

再结合征兆获取成本 cost(s)（data/real/dga/symptom_cost.json）得到价值 VoI(s) = EIG(s) - λ·cost(s)，
按 VoI 排序给出推荐，并按停止准则给出建议动作 ask | call_tool | conclude。

两种计算基（basis）：
  - "bayes"：用纯朴素贝叶斯后验，EIG 即互信息 I(F; s | E) ≥ 0（有严格保证）；
  - "fused"：用规则融合 + 温度缩放后的系统实际置信分布（默认，与前端 / Planner 看到的一致），
    由于融合与缩放不是严格的贝叶斯更新，EIG 可能出现极小负值，这里裁剪到 0。

与图谱联动：每个推荐征兆附带 symptom_cost.json 中 kg_nodes 对应边的一句文献原文，供 Planner 生成可解释追问。
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from src.tools.fault_attribution import (
    FaultBayesianNetwork,
    SYMPTOM_IDS,
    GAS_DERIVED_SYMPTOMS,
    get_engine,
)

ROOT = Path(__file__).resolve().parents[2]
COST_PATH = ROOT / "data/real/dga/symptom_cost.json"

_DEFAULT_STOP = {"tau_h_bits": 0.8, "eps_voi": 0.02, "max_rounds": 3}
_DEFAULT_LAMBDA = 0.05


# ──────────────────────────────────────────────────────────────
# 成本表
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_cost_table(path: str | None = None) -> dict[str, Any]:
    p = Path(path or os.environ.get("SYMPTOM_COST_PATH") or COST_PATH)
    if not p.exists():
        # 无成本表时：气体派生征兆 0，其余 1
        return {
            "lambda_default": _DEFAULT_LAMBDA,
            "stop": dict(_DEFAULT_STOP),
            "symptoms": {s: {"cost": 0 if s in GAS_DERIVED_SYMPTOMS else 1, "how_to_obtain": "ask_user",
                             "ask_hint": "", "kg_nodes": []} for s in SYMPTOM_IDS},
        }
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("lambda_default", _DEFAULT_LAMBDA)
    data.setdefault("stop", dict(_DEFAULT_STOP))
    data.setdefault("symptoms", {})
    for s in SYMPTOM_IDS:
        data["symptoms"].setdefault(s, {"cost": 0 if s in GAS_DERIVED_SYMPTOMS else 1,
                                        "how_to_obtain": "ask_user", "ask_hint": "", "kg_nodes": []})
    return data


def symptom_cost(symptom: str, table: dict[str, Any] | None = None) -> float:
    table = table or load_cost_table()
    return float(table["symptoms"].get(symptom, {}).get("cost", 1))


# ──────────────────────────────────────────────────────────────
# 图谱原文
# ──────────────────────────────────────────────────────────────

_KG_REL_FOR_HINT = ("INDICATES", "DETECTED_BY", "PRODUCES", "CAUSES")


@lru_cache(maxsize=64)
def kg_evidence_for(symptom: str) -> list[dict[str, str]]:
    """返回该征兆关联图谱节点上 INDICATES / DETECTED_BY / PRODUCES / CAUSES 边的原文片段（最多 2 条）。"""
    nodes = load_cost_table()["symptoms"].get(symptom, {}).get("kg_nodes") or []
    if not nodes:
        return []
    try:
        from src.tools.kg_search import get_kg
        kg = get_kg()
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, str]] = []
    for nid in nodes:
        for idx in list(kg.out_adj.get(nid, [])) + list(kg.in_adj.get(nid, [])):
            e = kg.edges[idx]
            if e["relation"] not in _KG_REL_FOR_HINT:
                continue
            ev = (e.get("evidence") or [{}])[0]
            sent = (ev.get("sentence") or "").strip().replace("\n", " ")
            if not sent:
                continue
            out.append({
                "edge": f"{e['source']} -{e['relation']}-> {e['target']}",
                "sentence": sent[:160],
                "doc_id": ev.get("doc_id", ""),
                "unit_id": ev.get("unit_id", ""),
            })
            if len(out) >= 2:
                return out
    return out


# ──────────────────────────────────────────────────────────────
# EIG 计算
# ──────────────────────────────────────────────────────────────

def _entropy(p: dict[str, float]) -> float:
    return FaultBayesianNetwork.entropy_bits(p)


def _dist(engine: FaultBayesianNetwork, evidence: dict[str, Any], ctx: dict[str, Any] | None, basis: str) -> dict[str, float]:
    po = engine.posterior_only(evidence, ctx)
    return po["bayes_probs"] if basis == "bayes" else po["probs"]


def predictive_symptom_prob(engine: FaultBayesianNetwork, posterior: dict[str, float], symptom: str) -> float:
    """P(s=True | E) = Σ_F P(s=True|F) P(F|E)。"""
    return sum(engine._cpt_lookup(f, symptom).p_true * pf for f, pf in posterior.items())  # noqa: SLF001


def eig_for_symptom(
    engine: FaultBayesianNetwork,
    evidence: dict[str, Any],
    symptom: str,
    ctx: dict[str, Any] | None = None,
    *,
    basis: str = "fused",
    posterior: dict[str, float] | None = None,
    h_now: float | None = None,
) -> dict[str, float]:
    """返回 {eig, p_true, h_if_true, h_if_false, expected_h_after}。"""
    if posterior is None:
        posterior = _dist(engine, evidence, ctx, basis)
    if h_now is None:
        h_now = _entropy(posterior)
    p_t = min(max(predictive_symptom_prob(engine, posterior, symptom), 1e-9), 1 - 1e-9)
    ev_t = dict(evidence); ev_t[symptom] = True
    ev_f = dict(evidence); ev_f[symptom] = False
    h_t = _entropy(_dist(engine, ev_t, ctx, basis))
    h_f = _entropy(_dist(engine, ev_f, ctx, basis))
    exp_h = p_t * h_t + (1 - p_t) * h_f
    eig = h_now - exp_h
    if basis != "bayes":
        eig = max(eig, 0.0)
    return {"eig": eig, "p_true": p_t, "h_if_true": h_t, "h_if_false": h_f, "expected_h_after": exp_h}


def unobserved_symptoms(evidence: dict[str, Any], candidates: Optional[list[str]] = None) -> list[str]:
    pool = candidates or SYMPTOM_IDS
    return [s for s in pool if evidence.get(s) is None]


def recommend(
    evidence: dict[str, Any],
    ctx: dict[str, Any] | None = None,
    *,
    engine: FaultBayesianNetwork | None = None,
    lam: float | None = None,
    stop: dict[str, float] | None = None,
    rounds_done: int = 0,
    basis: str = "fused",
    candidates: Optional[list[str]] = None,
    top_k: int = 5,
    with_kg: bool = True,
) -> dict[str, Any]:
    """
    计算当前不确定性与推荐征兆。

    返回:
        {
          "entropy_bits": H(F|E), "top1": ..., "top1_prob": ..., "top1_top2_gap": ..., "calibrated": bool,
          "basis": basis, "lambda": λ, "stop": {...}, "rounds_done": n,
          "recommendations": [ {symptom, eig, cost, voi, p_true, expected_h_after, how_to_obtain, ask_hint, kg_evidence}, ... ],
          "suggested_action": "ask" | "call_tool" | "conclude",
          "suggested_tool": <tool_name> | None,
          "stop_reason": None | "entropy_below_tau" | "voi_below_eps" | "max_rounds" | "no_candidates",
        }
    """
    engine = engine or get_engine()
    table = load_cost_table()
    lam = table["lambda_default"] if lam is None else lam
    stop_cfg = dict(table["stop"]); stop_cfg.update(stop or {})

    po = engine.posterior_only(evidence, ctx)
    posterior = po["bayes_probs"] if basis == "bayes" else po["probs"]
    h_now = _entropy(posterior)

    recs: list[dict[str, Any]] = []
    for s in unobserved_symptoms(evidence, candidates):
        r = eig_for_symptom(engine, evidence, s, ctx, basis=basis, posterior=posterior, h_now=h_now)
        meta = table["symptoms"].get(s, {})
        cost = float(meta.get("cost", 1))
        recs.append({
            "symptom": s,
            "eig": round(r["eig"], 5),
            "cost": cost,
            "voi": round(r["eig"] - lam * cost, 5),
            "p_true": round(r["p_true"], 4),
            "expected_h_after": round(r["expected_h_after"], 4),
            "how_to_obtain": meta.get("how_to_obtain", "ask_user"),
            "ask_hint": meta.get("ask_hint", ""),
        })
    recs.sort(key=lambda x: (-x["voi"], x["cost"], x["symptom"]))
    recs = recs[:top_k]
    if with_kg:
        for r in recs:
            r["kg_evidence"] = kg_evidence_for(r["symptom"])

    # 停止准则
    stop_reason = None
    if h_now < stop_cfg["tau_h_bits"]:
        stop_reason = "entropy_below_tau"
    elif not recs:
        stop_reason = "no_candidates"
    elif recs[0]["voi"] < stop_cfg["eps_voi"]:
        stop_reason = "voi_below_eps"
    elif rounds_done >= int(stop_cfg["max_rounds"]):
        stop_reason = "max_rounds"

    if stop_reason:
        action, tool = "conclude", None
    else:
        how = recs[0]["how_to_obtain"]
        if how.startswith("call_tool:"):
            action, tool = "call_tool", how.split(":", 1)[1]
        else:
            action, tool = "ask", None

    return {
        "entropy_bits": round(h_now, 4),
        "top1": po["top1"],
        "top1_prob": round(po["top1_prob"], 4),
        "top1_top2_gap": round(po["top1_top2_gap"], 4),
        "calibrated": po["calibrated"],
        "basis": basis,
        "lambda": lam,
        "stop": stop_cfg,
        "rounds_done": rounds_done,
        "recommendations": recs,
        "suggested_action": action,
        "suggested_tool": tool,
        "stop_reason": stop_reason,
    }


def render_recommendations(rec: dict[str, Any], limit: int = 3) -> str:
    """把推荐渲染为 Markdown 段落（附加到归因报告尾部）。"""
    lines = [
        "**不确定性与下一步建议**",
        f"- 当前后验熵 H(F|E) = {rec['entropy_bits']:.2f} bit，Top-1 `{rec['top1']}` 概率 {rec['top1_prob']:.1%}，"
        f"与第二名差 {rec['top1_top2_gap']:.1%}" + ("（已校准）" if rec.get("calibrated") else "（未校准）"),
    ]
    if rec["suggested_action"] == "conclude":
        reason = {
            "entropy_below_tau": "后验熵已低于阈值",
            "voi_below_eps": "剩余征兆的信息价值过低",
            "max_rounds": "已达最大问询轮数",
            "no_candidates": "所有征兆均已观测",
        }.get(rec.get("stop_reason") or "", "")
        lines.append(f"- 建议动作：**直接给出结论**（{reason}）")
    else:
        top = rec["recommendations"][0]
        how = "追问用户" if rec["suggested_action"] == "ask" else f"调用工具 `{rec['suggested_tool']}`"
        lines.append(f"- 建议动作：**{how}** 获取 `{top['symptom']}`（EIG {top['eig']:.3f} bit，成本 {top['cost']:.0f}，VoI {top['voi']:.3f}）")
        if top.get("ask_hint"):
            lines.append(f"  - 追问话术参考：{top['ask_hint']}")
        if top.get("kg_evidence"):
            lines.append(f"  - 文献依据：{top['kg_evidence'][0]['sentence'][:80]}…")
    if rec["recommendations"]:
        lines.append("- 候选征兆（按 VoI）：" + "；".join(
            f"`{r['symptom']}` EIG={r['eig']:.3f} cost={r['cost']:.0f}" for r in rec["recommendations"][:limit]))
    return "\n".join(lines)
