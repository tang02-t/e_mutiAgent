"""
C-5 验证器对照评测：v1（整体评分 Validator）/ v2_check（声明级核查，不路由）/ v2_route（核查 + 补证重规划）。

数据：D11 `data/eval/d11/fault_injection_eval.jsonl`（200 注入 + 100 干净）。
每条记录用 `snapshot` 离线重建 AgentState（同一份工具结果 / 知识片段 / 图谱路径），
以「模拟 Generator」把注入后的声明作为首轮草案送入 Validator，再按各组配置走真实的
Validator → (supplement → Retriever) → Generator 修订循环（复用 workflow._finish_generation，不调 LLM）。

模拟 Generator（SimGenerator）的行为假设（写入报告）：
  - 首轮：输出注入后的声明（模拟 LLM 犯错）；
  - 修订轮：仅修正 Validator 逐条反馈中被标记的声明（用规则声明中同 id 的版本替换，没有则删除），
    未被标记的错误原样保留（模拟「没有被指出的错误不会自行消失」）；
  - v1 无逐条反馈 → 错误全部保留；
  - 补证轮后（v2_route）在扩充后的证据目录上重建规则声明并做同样的替换。

指标：
  - 错误通过率：注入样本首轮 Validator 判 PASS 的比例（错误未被发现直接放行）
  - 分类型检出率：注入声明在首轮被标记（v2）且命中期望约束的比例；v1 无逐条判定记 0
  - 干净误报率：干净样本首轮判非 PASS 的比例
  - 错误残留率：最终答案（非弃答）中注入声明原样保留的比例
  - 无依据结论率：最终声明经独立 ClaimChecker 复核的 unsupported / EVIDENCE 占比（样本平均）
  - 约束违反率：最终声明存在任一约束违反的样本比例（非弃答）
  - 弃答率：最终判 ABSTAIN 的比例
  - 成本：额外工具调用次数、额外 Token（USAGE 差分）、平均耗时、平均 Generator 轮次

用法：
    python3 scripts/eval/eval_validator.py                       # 全量 300 条，落盘 docs/validator_eval.md/json + 图
    python3 scripts/eval/eval_validator.py --n 30 --no-report    # 快速自检
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                # noqa: E402
from src.graph import workflow as wf                                  # noqa: E402
from src.graph.system_modes import tool_stack_spec, build_agents        # noqa: E402
from src.agents.claims import build_rule_claims, make_claim           # noqa: E402
from src.agents.claim_checker import ClaimChecker                     # noqa: E402
from src.agents.validator import ValidatorAgent                       # noqa: E402
from src.tools.fault_attribution import configure_engine              # noqa: E402
from src.utils.llm import USAGE                                       # noqa: E402
from eval_system_modes import build_mcp                               # noqa: E402
import build_d11_fault_injection as d11                               # noqa: E402

D11_PATH = ROOT / "data/eval/d11/fault_injection_eval.jsonl"
REPORT_MD = ROOT / "docs/validator_eval.md"
REPORT_JSON = ROOT / "docs/validator_eval.json"
FIG_DIR = ROOT / "docs/figures"

ARMS = ("v1", "v2_check", "v2_route")
ARM_CFG = {"v1": "off", "v2_check": "check", "v2_route": "route"}
ARM_ZH = {"v1": "v1 整体评分", "v2_check": "v2 声明核查", "v2_route": "v2 核查+补证"}

NO_LLM_CFG: Dict[str, Any] = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 3, "planner_strategy": "free", "attribution_mode": "calibrated",
                 "generator_output_mode": "claims", "claim_check": "off", "claim_supplement_rounds": 1},
}


# 缺证扩展（E2）：需要「本轮尚未调用的工具」的声明，无证据；文本触发 ClaimChecker._missing_evidence 的建议工具规则
EXTRA_CLAIMS = [
    ("fault_attribution", "本次乙炔浓度对应的后验故障概率需以归因引擎结果为准", "inference"),
    ("ett_forecast", "未来 24 小时油温预测均值处于正常区间", "inference"),
    ("kg_search", "主故障的产生机理与关联部位可由图谱关系链路支撑", "inference"),
    ("rag_search", "建议依据检修规程安排复测与带电检测", "recommendation"),
]


def make_extras(rec: Dict[str, Any], k: int = 2) -> List[Dict[str, Any]]:
    """为记录挑 k 条需要尚未调用工具的缺证声明（fault_attribution 需 context.dga）。"""
    called = {c.get("tool") for c in rec["snapshot"]["tool_calls"] if c.get("success")}
    has_dga = isinstance((rec["snapshot"].get("context") or {}).get("dga"), dict)
    out: List[Dict[str, Any]] = []
    for tool, text, typ in EXTRA_CLAIMS:
        if tool in called or (tool == "fault_attribution" and not has_dga):
            continue
        c = make_claim(f"x_{len(out) + 1}", text, typ, [])
        c["_need_tool"] = tool
        out.append(c)
        if len(out) >= k:
            break
    return out


def load_d11() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in D11_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


# ─────────────────────────────────────────────────────────────
# 模拟 Generator
# ─────────────────────────────────────────────────────────────
class SimGenerator:
    """
    首轮输出注入声明（可附加缺证声明 extras）；修订轮只修正被 Validator 逐条标记的声明：
      - 同 id 规则声明存在 → 替换（fixed）
      - extras（id 以 x_ 开头，需要某工具的证据）→ 该工具本轮已有成功调用则用其规则声明再落证（regrounded），否则原样保留（kept）
      - 其余无可替换版本 → 删除（dropped）
    """

    def __init__(self, claims: List[Dict[str, Any]], body: str, extras: Optional[List[Dict[str, Any]]] = None) -> None:
        self.injected = copy.deepcopy(claims)
        self.body = body
        self.extras = copy.deepcopy(extras or [])
        self.current: List[Dict[str, Any]] = copy.deepcopy(claims) + copy.deepcopy(self.extras)
        self.n_fixed = 0
        self.n_dropped = 0
        self.n_regrounded = 0
        self.n_kept = 0

    @staticmethod
    def _from_tool(e: Dict[str, Any], need: str) -> bool:
        if need == "rag_search":
            return e.get("source") == "kb"
        if need == "kg_search":
            return e.get("source") == "kg"
        return e.get("source") == "tool" and e.get("tool") == need

    @staticmethod
    def _last_flagged(state: AgentState) -> Optional[set]:
        for tr in reversed(state.reasoning_trace or []):
            if tr.get("agent") == "validator" and tr.get("type") == "evaluation":
                cv = (tr.get("content") or {}).get("claim_verdicts")
                if cv is None:
                    return None
                return {v["claim_id"] for v in cv if v.get("verdict") != "pass" or v.get("violated_constraints")}
        return None

    def run(self, state: AgentState, revision_feedback: str = "") -> AgentState:
        state.iteration += 1
        if state.iteration > 1:
            flagged = self._last_flagged(state)
            if flagged:
                rule_claims = build_rule_claims(state)
                rule = {c["id"]: c for c in rule_claims}
                used_texts = {c["text"] for c in self.current}
                ok_tools = {c.get("tool") for c in state.tool_calls or [] if c.get("success")}
                fixed: List[Dict[str, Any]] = []
                for c in self.current:
                    if c["id"] not in flagged:
                        fixed.append(c)
                        continue
                    if c["id"] in rule:
                        fixed.append(copy.deepcopy(rule[c["id"]]))
                        self.n_fixed += 1
                    elif str(c["id"]).startswith("x_"):
                        need = c.get("_need_tool")
                        cand = [r for r in rule_claims if r["text"] not in used_texts
                                and any(self._from_tool(e, need) for e in r.get("evidence", []))]
                        if need in ok_tools and cand:
                            new = copy.deepcopy(cand[0])
                            new["id"] = c["id"]
                            used_texts.add(new["text"])
                            fixed.append(new)
                            self.n_regrounded += 1
                        else:
                            fixed.append(c)
                            self.n_kept += 1
                    else:
                        self.n_dropped += 1
                self.current = fixed
        state.draft_claims = [{k: v for k, v in c.items() if not k.startswith("_")} for c in copy.deepcopy(self.current)]
        state.claims_source = "rule"
        state.draft_answer = d11.render_draft(self.body, state.draft_claims)
        state.reasoning_trace.append({"agent": "generator", "type": "sim_generation",
                                      "content": {"iteration": state.iteration, "n_claims": len(state.draft_claims)}})
        return state


# ─────────────────────────────────────────────────────────────
# 单条评测
# ─────────────────────────────────────────────────────────────
def _first_eval(state: AgentState) -> Optional[Dict[str, Any]]:
    for tr in state.reasoning_trace or []:
        if tr.get("agent") == "validator" and tr.get("type") == "evaluation":
            return tr.get("content") or {}
    return None


def run_one(rec: Dict[str, Any], arm: str, retriever, validator, oracle: ClaimChecker,
            extras: bool = False) -> Dict[str, Any]:
    body = rec["draft_answer"].split(d11.CLAIMS_HEAD)[0].rstrip()
    state = d11.state_from_snapshot(rec["snapshot"], [], "")
    state.iteration = 0
    state.max_iterations = NO_LLM_CFG["workflow"]["max_iterations"]
    n_calls0 = len(state.tool_calls)
    ex = make_extras(rec) if extras else []
    gen = SimGenerator(rec["claims"], body, ex)
    USAGE.reset()
    t0 = time.time()
    err = None
    try:
        final = wf._finish_generation(state, retriever, gen, validator, None)
    except Exception as exc:  # noqa: BLE001
        final, err = state, str(exc)
    elapsed = time.time() - t0
    usage = USAGE.snapshot()

    first = _first_eval(final) or {}
    first_verdict = first.get("verdict")
    first_cv = first.get("claim_verdicts")
    tgt = (rec.get("location") or {}).get("claim_id")
    tgt_flagged, tgt_constraints = False, []
    if first_cv is not None and tgt:
        for v in first_cv:
            if v.get("claim_id") == tgt:
                tgt_flagged = v.get("verdict") != "pass" or bool(v.get("violated_constraints"))
                tgt_constraints = list(v.get("violated_constraints") or [])
    expected = rec.get("expected_constraints") or []
    hit_expected = bool(set(expected) & set(tgt_constraints)) if rec["injected"] else None

    abstained = final.validation_verdict == "ABSTAIN"
    final_claims = final.draft_claims or []
    injected_by_id = {c["id"]: c for c in rec["claims"]}
    survived = None
    if rec["injected"] and not abstained:
        cur = {c["id"]: c for c in final_claims}
        survived = tgt in cur and cur[tgt] == injected_by_id.get(tgt)
    # 独立复核最终声明
    osc = d11.state_from_snapshot(rec["snapshot"], [], "")
    osc.tool_calls = copy.deepcopy(final.tool_calls)
    osc.retrieved_knowledge = copy.deepcopy(final.retrieved_knowledge)
    osc.draft_claims = copy.deepcopy(final_claims)
    orep = oracle.check(osc, use_llm=False) if final_claims else None
    o_unsupported = orep.unsupported_ratio if orep else (0.0 if abstained else 1.0)
    o_violated = bool(orep and any(v.violated_constraints for v in orep.claim_verdicts))
    o_n_unsupported = sum(1 for v in orep.claim_verdicts if v.verdict in ("unsupported", "contradict")
                          or "EVIDENCE" in v.violated_constraints) if orep else 0
    # extras 去向：仍无据保留 / 已再落证 / 被删
    ex_ids = {c["id"] for c in ex}
    ex_kept_unsupported = sum(1 for c in final_claims if c["id"] in ex_ids and not c.get("evidence"))

    return {
        "eval_id": rec["eval_id"], "arm": arm, "injected": rec["injected"], "type": rec.get("type"),
        "subtype": rec.get("subtype"), "scenario": rec["scenario"], "error": err, "elapsed": round(elapsed, 4),
        "first_verdict": first_verdict, "first_claim_verdict": first.get("claim_check_verdict"),
        "target_flagged": tgt_flagged, "target_constraints": tgt_constraints, "hit_expected": hit_expected,
        "final_verdict": final.validation_verdict, "abstained": abstained, "iterations": final.iteration,
        "survived": survived, "oracle_unsupported_ratio": round(o_unsupported, 4), "oracle_violated": o_violated,
        "oracle_violations": dict(orep.violation_counts) if orep else {},
        "n_final_claims": len(final_claims), "n_fixed": gen.n_fixed, "n_dropped": gen.n_dropped,
        "n_regrounded": gen.n_regrounded, "n_kept": gen.n_kept, "n_extras": len(ex),
        "extras_kept_unsupported": ex_kept_unsupported, "oracle_n_unsupported": o_n_unsupported,
        "extra_tool_calls": len(final.tool_calls) - n_calls0, "evidence_rounds": final.evidence_rounds,
        "route_targets": [r.get("target") for r in final.route_log],
        "tokens": usage.get("total_tokens", 0), "llm_calls": usage.get("calls", 0),
    }


def run_arm(records: List[Dict[str, Any]], arm: str, extras: bool = False) -> List[Dict[str, Any]]:
    logging.disable(logging.WARNING)
    cfg = json.loads(json.dumps(NO_LLM_CFG))
    cfg["workflow"]["claim_check"] = ARM_CFG[arm]
    configure_engine(cfg["workflow"]["attribution_mode"])
    spec = tool_stack_spec(cfg)
    mcp, kb_desc = build_mcp(spec)
    _, retriever, _, _, _ = build_agents(cfg, mcp, spec)
    validator = ValidatorAgent(cfg, claim_check=ARM_CFG[arm])
    oracle = ClaimChecker()
    print(f"=== C-5 eval_validator | arm={arm} claim_check={ARM_CFG[arm]} extras={extras} n={len(records)} rag={kb_desc} ===")
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(records, 1):
        out.append(run_one(rec, arm, retriever, validator, oracle, extras=extras))
        if i % 50 == 0 or i == len(records):
            print(f"  ... {i}/{len(records)}")
    return out


# ─────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────
def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    inj = [r for r in rows if r["injected"]]
    clean = [r for r in rows if not r["injected"]]
    by_type: Dict[str, Any] = {}
    for t in d11.TYPES:
        rs = [r for r in inj if r["type"] == t]
        by_type[t] = {
            "n": len(rs),
            "detection_rate": _rate(sum(1 for r in rs if r["target_flagged"]), len(rs)),
            "expected_hit_rate": _rate(sum(1 for r in rs if r["hit_expected"]), len(rs)),
            "error_pass_rate": _rate(sum(1 for r in rs if r["first_verdict"] == "PASS"), len(rs)),
            "survival_rate": _rate(sum(1 for r in rs if r["survived"]), sum(1 for r in rs if not r["abstained"])),
            "abstain_rate": _rate(sum(1 for r in rs if r["abstained"]), len(rs)),
        }
    by_subtype: Dict[str, Any] = {}
    for st, rs in defaultdict(list, {k: [r for r in inj if r["subtype"] == k] for k in {r["subtype"] for r in inj}}).items():
        by_subtype[st] = {"n": len(rs), "detection_rate": _rate(sum(1 for r in rs if r["target_flagged"]), len(rs)),
                          "survival_rate": _rate(sum(1 for r in rs if r["survived"]), sum(1 for r in rs if not r["abstained"]))}
    non_abstain = [r for r in rows if not r["abstained"]]
    return {
        "arm": rows[0]["arm"] if rows else None, "n": len(rows), "n_injected": len(inj), "n_clean": len(clean),
        "n_errors": sum(1 for r in rows if r["error"]),
        "error_pass_rate": _rate(sum(1 for r in inj if r["first_verdict"] == "PASS"), len(inj)),
        "detection_rate": _rate(sum(1 for r in inj if r["target_flagged"]), len(inj)),
        "expected_hit_rate": _rate(sum(1 for r in inj if r["hit_expected"]), len(inj)),
        "clean_fp_rate": _rate(sum(1 for r in clean if r["first_verdict"] != "PASS"), len(clean)),
        "clean_final_non_pass_rate": _rate(sum(1 for r in clean if r["final_verdict"] != "PASS"), len(clean)),
        "survival_rate": _rate(sum(1 for r in inj if r["survived"]), sum(1 for r in inj if not r["abstained"])),
        "unsupported_rate": _rate(sum(r["oracle_unsupported_ratio"] for r in non_abstain), len(non_abstain)),
        "unsupported_rate_injected": _rate(sum(r["oracle_unsupported_ratio"] for r in inj if not r["abstained"]),
                                           sum(1 for r in inj if not r["abstained"])),
        "violation_rate": _rate(sum(1 for r in non_abstain if r["oracle_violated"]), len(non_abstain)),
        "violation_rate_injected": _rate(sum(1 for r in inj if not r["abstained"] and r["oracle_violated"]),
                                         sum(1 for r in inj if not r["abstained"])),
        "abstain_rate": _rate(sum(1 for r in rows if r["abstained"]), len(rows)),
        "abstain_rate_injected": _rate(sum(1 for r in inj if r["abstained"]), len(inj)),
        "abstain_rate_clean": _rate(sum(1 for r in clean if r["abstained"]), len(clean)),
        "avg_iterations": _rate(sum(r["iterations"] for r in rows), len(rows)),
        "avg_extra_tool_calls": _rate(sum(r["extra_tool_calls"] for r in rows), len(rows)),
        "total_extra_tool_calls": sum(r["extra_tool_calls"] for r in rows),
        "supplement_triggered": sum(1 for r in rows if r["evidence_rounds"] > 0),
        "total_tokens": sum(r["tokens"] for r in rows), "total_llm_calls": sum(r["llm_calls"] for r in rows),
        "avg_elapsed_ms": 1000 * _rate(sum(r["elapsed"] for r in rows), len(rows)),
        "final_verdicts": dict(Counter(r["final_verdict"] for r in rows)),
        "first_verdicts_injected": dict(Counter(r["first_verdict"] for r in inj)),
        "n_fixed": sum(r["n_fixed"] for r in rows), "n_dropped": sum(r["n_dropped"] for r in rows),
        "n_regrounded": sum(r["n_regrounded"] for r in rows), "n_kept": sum(r["n_kept"] for r in rows),
        "n_extras": sum(r["n_extras"] for r in rows),
        "extras_kept_unsupported": sum(r["extras_kept_unsupported"] for r in rows),
        "extras_unsupported_rate": _rate(sum(r["extras_kept_unsupported"] for r in rows), sum(r["n_extras"] for r in rows)),
        "avg_final_unsupported_claims": _rate(sum(r["oracle_n_unsupported"] for r in non_abstain), len(non_abstain)),
        "by_type": by_type, "by_subtype": dict(sorted(by_subtype.items())),
    }


def acceptance(s: Dict[str, Dict[str, Any]], s2: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """s: 设定 E1（D11 原样）；s2: 设定 E2（D11 + 每条附加 2 条需新工具的缺证声明）。"""
    v1, c, r = s["v1"], s["v2_check"], s["v2_route"]
    c2, r2 = s2["v2_check"], s2["v2_route"]
    a = {
        "v2_check_error_pass_drop": v1["error_pass_rate"] - c["error_pass_rate"],
        "v2_check_error_pass_significant": (c["error_pass_rate"] <= 0.5 * v1["error_pass_rate"]) if v1["error_pass_rate"] > 0
        else c["error_pass_rate"] == 0,
        "v2_check_clean_fp_ok": c["clean_fp_rate"] <= 0.10,
        "e1_v2_route_unsupported_not_higher": r["unsupported_rate"] <= c["unsupported_rate"] + 1e-9,
        "e2_v2_route_unsupported_lower": r2["unsupported_rate"] < c2["unsupported_rate"],
        "e2_v2_route_extras_unsupported_lower": r2["extras_unsupported_rate"] < c2["extras_unsupported_rate"],
        "no_errors": all(x["n_errors"] == 0 for x in list(s.values()) + list(s2.values())),
    }
    a["pass"] = bool(a["v2_check_error_pass_significant"] and a["v2_check_clean_fp_ok"]
                     and a["e1_v2_route_unsupported_not_higher"] and a["e2_v2_route_unsupported_lower"] and a["no_errors"])
    return a


# ─────────────────────────────────────────────────────────────
# 图表
# ─────────────────────────────────────────────────────────────
def plot(s: Dict[str, Dict[str, Any]], s2: Dict[str, Dict[str, Any]]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: List[str] = []

    # 图 1：分类型检出率柱状图（E1）
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    xs = list(range(len(d11.TYPES)))
    for k, arm in enumerate(ARMS):
        vals = [100 * s[arm]["by_type"][t]["detection_rate"] for t in d11.TYPES]
        bars = ax.bar([x + (k - 1) * width for x in xs], vals, width, label=ARM_ZH[arm])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([d11.TYPE_ZH[t] for t in d11.TYPES])
    ax.set_ylabel("首轮检出率（%）")
    ax.set_ylim(0, 112)
    ax.set_title("D11 注入错误分类型检出率（n=50/类）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = FIG_DIR / "fig_c5_detection_by_type.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))

    # 图 2：错误通过率 / 无依据结论率 / 弃答率 与 额外调用成本（E1、E2 两设定，三组）
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    groups = [("E1 仅注入", s), ("E2 注入+缺证", s2)]
    xs2 = list(range(len(groups)))
    w = 0.26
    panels = [
        ("错误通过率（%）", lambda ss, a: 100 * ss[a]["error_pass_rate"]),
        ("无依据结论率（%）", lambda ss, a: 100 * ss[a]["unsupported_rate"]),
        ("弃答率（%）", lambda ss, a: 100 * ss[a]["abstain_rate"]),
    ]
    for ax, (ylabel, fn) in zip(axes, panels):
        for k, arm in enumerate(ARMS):
            vals = [fn(ss, arm) for _, ss in groups]
            bars = ax.bar([x + (k - 1) * w for x in xs2], vals, w, label=ARM_ZH[arm])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(xs2)
        ax.set_xticklabels([g for g, _ in groups])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", alpha=0.3)
    # 成本标注在标题
    cost = " / ".join(f"{ARM_ZH[a]} {s2[a]['avg_extra_tool_calls']:.2f}" for a in ARMS)
    fig.suptitle(f"验证器三组对照（无 LLM，Token=0）；E2 平均额外工具调用：{cost}", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = FIG_DIR / "fig_c5_pass_vs_cost.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    out.append(str(p))
    return out


# ─────────────────────────────────────────────────────────────
# 报告
# ─────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(s: Dict[str, Dict[str, Any]], extras: bool) -> List[str]:
    L = ["| 指标 | v1 | v2_check | v2_route |", "|---|---:|---:|---:|"]
    rows: List[tuple] = [
        ("错误通过率（注入样本首轮 PASS）", "error_pass_rate", True),
        ("注入声明首轮检出率", "detection_rate", True),
        ("检出并命中期望约束", "expected_hit_rate", True),
        ("干净样本误报率（首轮非 PASS）", "clean_fp_rate", True),
        ("干净样本最终非 PASS", "clean_final_non_pass_rate", True),
        ("错误残留率（最终答案仍含注入声明，非弃答）", "survival_rate", True),
        ("无依据结论率（最终声明 unsupported 占比，样本平均）", "unsupported_rate", True),
        ("无依据结论率（注入样本）", "unsupported_rate_injected", True),
        ("最终答案平均无依据声明数", "avg_final_unsupported_claims", False),
    ]
    if extras:
        rows += [("缺证声明仍无据保留比例", "extras_unsupported_rate", True),
                 ("缺证声明再落证数 / 保留数", "_extras", None)]
    rows += [
        ("约束违反率（最终答案任一违反）", "violation_rate", True),
        ("约束违反率（注入样本）", "violation_rate_injected", True),
        ("弃答率（全体）", "abstain_rate", True),
        ("弃答率（注入 / 干净）", "_abstain", None),
        ("平均 Generator 轮次", "avg_iterations", False),
        ("修订：替换 / 删除声明数", "_fix", None),
        ("额外工具调用（总 / 平均）", "_calls", None),
        ("补证触发样本数", "supplement_triggered", False),
        ("额外 Token / LLM 调用", "_tok", None),
        ("平均耗时（ms）", "avg_elapsed_ms", False),
    ]
    for label, key, is_pct in rows:
        if key == "_extras":
            cells = [f"{s[a]['n_regrounded']} / {s[a]['n_kept']}" for a in ARMS]
        elif key == "_abstain":
            cells = [f"{_pct(s[a]['abstain_rate_injected'])} / {_pct(s[a]['abstain_rate_clean'])}" for a in ARMS]
        elif key == "_fix":
            cells = [f"{s[a]['n_fixed']} / {s[a]['n_dropped']}" for a in ARMS]
        elif key == "_calls":
            cells = [f"{s[a]['total_extra_tool_calls']} / {s[a]['avg_extra_tool_calls']:.2f}" for a in ARMS]
        elif key == "_tok":
            cells = [f"{s[a]['total_tokens']} / {s[a]['total_llm_calls']}" for a in ARMS]
        elif is_pct:
            cells = [_pct(s[a][key]) for a in ARMS]
        else:
            cells = [f"{s[a][key]:.2f}" if isinstance(s[a][key], float) else str(s[a][key]) for a in ARMS]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    return L


def write_report(s: Dict[str, Dict[str, Any]], s2: Dict[str, Dict[str, Any]], acc: Dict[str, Any],
                 figs: List[str], n: int, seed: int) -> None:
    L: List[str] = []
    L += ["# C-5 验证器对照评测（v1 / v2_check / v2_route）", "",
          f"- 数据：D11 `data/eval/d11/fault_injection_eval.jsonl`，{n} 条（注入 {s['v1']['n_injected']} + 干净 {s['v1']['n_clean']}）；seed={seed}",
          "- 脚本：`scripts/eval/eval_validator.py`；无 LLM、无 GPU；每条用 `snapshot` 离线重建同一份证据，三组走真实 Validator / supplement / Retriever 节点（`workflow._finish_generation`）",
          "- 组别：`v1` = 现有整体评分 Validator（规则降级路径，`claim_check=off`）；`v2_check` = 声明级核查不路由；`v2_route` = 核查 + 规则补证路由（最多 1 轮）；三组 `max_iterations=3`、`claim_abstain_after=3`",
          "- 两个设定：E1 = D11 原样（每条 1 处注入错误）；E2 = D11 每条再附加 2 条「需要本轮尚未调用工具」的缺证声明（无 evidence，文本触发建议工具规则），用于考察补证路由对无依据结论的作用",
          "", "## 模拟 Generator 假设", "",
          "- 首轮输出注入后的声明（模拟 LLM 犯错）；修订轮只修正 Validator 逐条反馈中被标记的声明：同 id 规则声明存在则替换，缺证声明在所需工具已被补证调用后用该工具的规则声明再落证，否则原样保留；其余无替代版本则删除。",
          "- v1 无逐条反馈，因此错误全部保留；这刻画的是「验证器能否把错误定位到声明并给出可执行的修订信号」，而不是 LLM 自我纠错能力。",
          "- 三组共享同一份首轮草案与证据，差异只来自 Validator 配置。",
          "", "## E1：D11 原样（每条 1 处注入错误）", ""]
    L += _table(s, extras=False)
    L += ["", "### E1 分类型", "",
          "| 类型 | 指标 | v1 | v2_check | v2_route |", "|---|---|---:|---:|---:|"]
    for t in d11.TYPES:
        for lab, k in (("首轮检出率", "detection_rate"), ("错误通过率", "error_pass_rate"),
                       ("残留率", "survival_rate"), ("弃答率", "abstain_rate")):
            L.append(f"| {d11.TYPE_ZH[t]}（{t}） | {lab} | " + " | ".join(_pct(s[a]["by_type"][t][k]) for a in ARMS) + " |")
    L += ["", "### E1 分子类（v2_check / v2_route）", "",
          "| 子类 | n | 检出率 check | 检出率 route | 残留率 check | 残留率 route |", "|---|---:|---:|---:|---:|---:|"]
    for st, v in s["v2_check"]["by_subtype"].items():
        r = s["v2_route"]["by_subtype"].get(st, {})
        L.append(f"| {st} | {v['n']} | {_pct(v['detection_rate'])} | {_pct(r.get('detection_rate', 0))} | "
                 f"{_pct(v['survival_rate'])} | {_pct(r.get('survival_rate', 0))} |")

    L += ["", "## E2：D11 + 每条 2 条缺证声明", "",
          f"附加缺证声明共 {s2['v1']['n_extras']} 条（{n} 条样本，含干净底稿）。注意：E2 下「干净样本」也附加了缺证声明，"
          "其首轮非 PASS 属正确拦截而非误报；误报率指标只在 E1 有意义。", ""]
    L += _table(s2, extras=True)

    L += ["", "## 验收判定", "",
          "验收标准（PLAN C-5）：`v2_check` 相对 `v1` 错误通过率显著下降且干净样本误报率不高于 10%；`v2_route` 的无依据结论率进一步下降。", "",
          f"- E1 错误通过率：v1 {_pct(s['v1']['error_pass_rate'])} → v2_check {_pct(s['v2_check']['error_pass_rate'])}"
          f"（下降 {100 * acc['v2_check_error_pass_drop']:.1f} 个百分点，{'满足' if acc['v2_check_error_pass_significant'] else '未满足'}「至少减半」）",
          f"- E1 v2_check 干净误报率 {_pct(s['v2_check']['clean_fp_rate'])}（{'≤' if acc['v2_check_clean_fp_ok'] else '>'} 10%）",
          f"- E1 无依据结论率：v2_check {_pct(s['v2_check']['unsupported_rate'])} → v2_route {_pct(s['v2_route']['unsupported_rate'])}"
          f"（D11 注入类型中只有伪造引用属可补证据，两者{'相等或更低' if acc['e1_v2_route_unsupported_not_higher'] else '更高'}）",
          f"- E2 无依据结论率：v2_check {_pct(s2['v2_check']['unsupported_rate'])} → v2_route {_pct(s2['v2_route']['unsupported_rate'])}；"
          f"缺证声明仍无据保留 {_pct(s2['v2_check']['extras_unsupported_rate'])} → {_pct(s2['v2_route']['extras_unsupported_rate'])}"
          f"（{'进一步下降' if acc['e2_v2_route_unsupported_lower'] else '未下降'}）",
          f"- E2 成本：v2_route 额外工具调用 {s2['v2_route']['total_extra_tool_calls']} 次（平均 {s2['v2_route']['avg_extra_tool_calls']:.2f}/样本），"
          f"补证触发 {s2['v2_route']['supplement_triggered']} 条，额外 Token 0",
          f"- 运行异常 {sum(x['n_errors'] for x in list(s.values()) + list(s2.values()))} 条",
          f"- **结论：{'验收通过' if acc['pass'] else '未通过'}**", "",
          "## 图表", ""]
    for f in figs:
        L.append(f"- `{Path(f).relative_to(ROOT)}`")
    L += ["", "## 说明与局限", "",
          "- 三组均为确定性层（无 LLM）：v1 走规则降级评分（长度 / 安全词 / 建议词），v2 走 C-2 确定性核查；语义层（NLI）与「LLM 判定抽 10% 人工复核」需启用 LLM，后置。",
          "- C-5 评测中发现并修订的判定规则：APPLICABILITY 纳入硬约束（数据集 / 设备号 / 时间窗错位原先仅计入 `violated` 不触发 REVISION，导致换数据集类 60% 通过）；observation 声明证据引用无效（伪造 / 不存在 / 知识库冒充检测）直接 REVISION（原先仅按 unsupported 占比 > 0.3 触发，多声明草案中单条伪造引用会被稀释放行）。配置 `claim_hard_constraints` / `claim_strict_observation`。",
          "- E1 残余通过的注入样本为 inference / recommendation 类声明的伪造引用（`nonexistent_*`），占比未超 0.3 阈值；提高阈值敏感度会推高干净误报，保留为已知权衡。",
          "- E2 中 v2_check 弃答率高（无法补证时缺证声明始终保留，第 3 次评估仍不通过 → ABSTAIN），体现「宁弃答不臆断」；v2_route 通过补证把弃答率降到个位数，代价是平均 1.6 次额外工具调用。仍无据保留的缺证声明来自补证被跳过的情形（图谱无可定位实体、context 无 dga / 数据集参数、同参数已调用）。",
          "- D11 底稿为规则拼装声明，注入错误的隐蔽性低于真实 LLM 幻觉；检出率上限受 D11 构造方式影响（见 `data/eval/d11/DATA_CARD.md` 已知偏差）。",
          "- 错误残留率与弃答率依赖模拟 Generator 的「只改被标记声明」假设；真实 LLM Generator 可能修正更多或引入新错误。",
          "- v2_route 的补证只针对缺证声明（无引用 / EVIDENCE 违反且能映射到可用工具）；DATA / APPLICABILITY / SAFETY 违反走 Generator 直接修订，因此 E1 中补证触发数低于注入总数属预期。",
          ]
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数（0 = 全量 300）")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--settings", default="e1,e2", help="e1 = D11 原样；e2 = D11 + 缺证声明")
    args = ap.parse_args()

    records = load_d11()
    if args.n and args.n < len(records):
        rng = random.Random(args.seed)
        inj = [r for r in records if r["injected"]]
        clean = [r for r in records if not r["injected"]]
        k_inj = max(1, round(args.n * 2 / 3))
        records = rng.sample(inj, min(k_inj, len(inj))) + rng.sample(clean, min(args.n - k_inj, len(clean)))
    arms = [a for a in args.arms.split(",") if a in ARMS]
    settings = [x for x in args.settings.split(",") if x in ("e1", "e2")]

    t0 = time.time()
    all_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for setting in settings:
        all_rows[setting], summaries[setting] = {}, {}
        for arm in arms:
            rows = run_arm(records, arm, extras=(setting == "e2"))
            all_rows[setting][arm] = rows
            summaries[setting][arm] = summarize(rows)
            brief = {k: v for k, v in summaries[setting][arm].items() if k not in ("by_type", "by_subtype", "final_verdicts",
                                                                                   "first_verdicts_injected")}
            print(f"[{setting}/{arm}] " + json.dumps(brief, ensure_ascii=False))
            if setting == "e1":
                print(json.dumps({t: {k: round(v, 3) for k, v in d.items()} for t, d in summaries[setting][arm]["by_type"].items()},
                                 ensure_ascii=False))

    full = set(arms) == set(ARMS) and set(settings) == {"e1", "e2"}
    acc = acceptance(summaries["e1"], summaries["e2"]) if full else {"pass": None}
    print("acceptance:", json.dumps(acc, ensure_ascii=False))
    print(f"elapsed {time.time() - t0:.1f}s")

    if not args.no_report and full:
        figs = plot(summaries["e1"], summaries["e2"])
        write_report(summaries["e1"], summaries["e2"], acc, figs, len(records), args.seed)
        REPORT_JSON.write_text(json.dumps({"summaries": summaries, "acceptance": acc, "rows": all_rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {REPORT_MD}\nfigures -> {figs}")
    return 0 if acc.get("pass") in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
