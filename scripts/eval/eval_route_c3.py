"""
C-3 验收脚本：补证与重规划路由在 D10 上的去重与路径统计。

流程：D10 分层抽样 → OraclePlanner（金标动作，不调 LLM）→ 真实工具 → Generator(claims，模板路径)
    → 首轮草案注入 4 条无依据声明（分别指向 fault_attribution / ett_forecast / kg_search / rag_search）
    → Validator(claim_check=route) → supplement → Retriever → Generator 修订 → Validator。

验收标准（PLAN C-3）：补证路由触发的调用中不出现重复调用同一工具同一参数。
本脚本同时统计：补证触发率、补证工具分布、去重跳过原因、修订后 verdict、额外调用数 / 耗时 / Token、
以及 claim_check=check（不路由）的对照。

用法：
    python3 scripts/eval/eval_route_c3.py                    # 30 条，落盘 docs/eval/route_acceptance.md/json
    python3 scripts/eval/eval_route_c3.py --n 6 --no-report  # 快速自检
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                # noqa: E402
from src.graph import workflow as wf                                  # noqa: E402
from src.graph.system_modes import tool_stack_spec, build_agents        # noqa: E402
from src.agents.generator import GeneratorAgent                       # noqa: E402
from src.agents.claims import make_claim                              # noqa: E402
from src.tools.fault_attribution import configure_engine              # noqa: E402
from eval_system_modes import build_mcp, OraclePlanner                # noqa: E402
from eval_claims_c1 import load_d10, stratified_sample                # noqa: E402

REPORT_MD = ROOT / "docs/eval/route_acceptance.md"
REPORT_JSON = ROOT / "docs/eval/route_acceptance.json"

NO_LLM_CFG: Dict[str, Any] = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 3, "planner_strategy": "free", "attribution_mode": "calibrated",
                 "generator_output_mode": "claims", "claim_check": "route", "claim_supplement_rounds": 1},
}

# 注入的无依据声明：文本触发 ClaimChecker._missing_evidence 的四条建议工具规则
INJECT = [
    ("fault_attribution", "本次乙炔浓度对应的后验故障概率需以归因引擎结果为准", "inference"),
    ("ett_forecast", "未来 24 小时油温预测均值处于正常区间", "inference"),
    ("kg_search", "主故障的产生机理与关联部位可由图谱关系链路支撑", "inference"),
    ("rag_search", "建议依据检修规程安排复测与带电检测", "recommendation"),
]


class InjectGenerator(GeneratorAgent):
    """首轮草案追加 4 条无 evidence 的声明（unsupported），修订轮按规则重建。"""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg, output_mode="claims")
        self._first = True

    def reset(self) -> None:
        self._first = True

    def run(self, state: AgentState, revision_feedback: str = "") -> AgentState:
        state = super().run(state, revision_feedback=revision_feedback)
        if self._first:
            state.draft_claims = list(state.draft_claims) + [
                make_claim(f"inj_{i}", txt, typ, []) for i, (_, txt, typ) in enumerate(INJECT, 1)]
            self._first = False
        return state


def _sig(c: Dict[str, Any]) -> tuple:
    args = c.get("args") if isinstance(c.get("args"), dict) else (c.get("raw_arguments") or {})
    return (c.get("tool"), wf._canonical_args(c.get("tool"), args))


def run(samples: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    logging.disable(logging.WARNING)
    cfg = json.loads(json.dumps(NO_LLM_CFG))
    cfg["workflow"]["claim_check"] = mode
    configure_engine(cfg["workflow"]["attribution_mode"])
    spec = tool_stack_spec(cfg)
    mcp, kb_desc = build_mcp(spec)
    _, retriever, _, validator, _ = build_agents(cfg, mcp, spec)
    generator = InjectGenerator(cfg)
    planner = OraclePlanner(spec.allowed_tools)
    print(f"=== C-3 route acceptance | mode={mode} n={len(samples)} rag={kb_desc} ===")

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        planner.bind(s)
        generator.reset()
        state = AgentState(user_query=s["user_query"], context=dict(s.get("context") or {}), max_iterations=3)
        t0 = time.time()
        err = None
        try:
            final = wf.run_diagnosis_workflow(state, planner, retriever, generator, validator)
        except Exception as exc:  # noqa: BLE001
            final, err = state, str(exc)
        elapsed = time.time() - t0

        sup_entries = [r for r in final.route_log if r.get("target") == "supplement"]
        sup_tools: List[Dict[str, Any]] = [t for r in sup_entries for t in (r.get("tools") or [])]
        skipped: List[Dict[str, Any]] = [k for r in sup_entries for k in (r.get("skipped") or [])]
        # 补证调用 = tool_calls 尾部（Retriever 补证轮累积追加）
        n_first = len(final.tool_calls) - len(sup_tools)
        sup_calls = final.tool_calls[n_first:]
        first_calls = final.tool_calls[:n_first]
        first_sigs = {_sig(c) for c in first_calls}
        sup_sigs = [_sig(c) for c in sup_calls]
        dup_with_first = [sg for sg in sup_sigs if sg in first_sigs]
        dup_within = [sg for sg, n in Counter(sup_sigs).items() if n > 1]
        all_sigs = [_sig(c) for c in final.tool_calls]
        dup_all = len(all_sigs) - len(set(all_sigs))
        sup_ok = sum(1 for c in sup_calls if c.get("success"))

        rec = {
            "eval_id": s["eval_id"], "scenario": s["scenario"], "error": err, "elapsed": round(elapsed, 3),
            "verdict": final.validation_verdict, "iteration": final.iteration, "evidence_rounds": final.evidence_rounds,
            "route_targets": [r.get("target") for r in final.route_log],
            "route_reasons": [r.get("reason") for r in final.route_log],
            "claim_verdicts": [e.get("verdict") for e in final.claim_check_log],
            "first_tools": sorted(c.get("tool") for c in first_calls),
            "sup_tools": [t["tool"] for t in sup_tools],
            "sup_args": [t["arguments"] for t in sup_tools],
            "sup_success": sup_ok, "n_sup_calls": len(sup_calls),
            "skipped_reasons": dict(Counter(k.get("reason") for k in skipped)),
            "skipped_tools": dict(Counter(f"{k.get('tool')}:{k.get('reason')}" for k in skipped)),
            "dup_with_first": len(dup_with_first), "dup_within_sup": len(dup_within), "dup_all": dup_all,
            "extra_tokens": sum(int(r.get("extra_tokens") or 0) for r in sup_entries),
            "extra_llm_calls": sum(int(r.get("extra_llm_calls") or 0) for r in sup_entries),
            "sup_latency_ms": round(sum(float(r.get("latency_ms") or 0) for r in sup_entries), 1),
            "n_claims_final": len(final.draft_claims or []),
            "final_sources": dict(Counter(e.get("source") for c in (final.draft_claims or []) for e in c.get("evidence", []))),
            "n_tool_calls": len(final.tool_calls),
        }
        out.append(rec)
        print(f"  [{i}/{len(samples)}] {s['eval_id']} {s['scenario']:<20} first={','.join(rec['first_tools']) or '-':<38} "
              f"sup={','.join(rec['sup_tools']) or '-':<32} skip={sum(rec['skipped_reasons'].values())} "
              f"dup={dup_all} verdict={rec['verdict']} route={'>'.join(rec['route_targets'])} err={err or '-'}")
    return out


def summarize(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    n = len(rows)
    trig = [r for r in rows if r["evidence_rounds"] > 0]
    sup_tool_counts: Counter = Counter()
    skip_counts: Counter = Counter()
    for r in rows:
        sup_tool_counts.update(r["sup_tools"])
        skip_counts.update(r["skipped_reasons"])
    n_sup_calls = sum(r["n_sup_calls"] for r in rows)
    return {
        "mode": mode, "n": n, "n_errors": sum(1 for r in rows if r["error"]),
        "triggered": len(trig), "trigger_rate": len(trig) / n if n else 0.0,
        "n_sup_calls": n_sup_calls, "avg_sup_calls": n_sup_calls / n if n else 0.0,
        "sup_success": sum(r["sup_success"] for r in rows),
        "sup_tool_counts": dict(sup_tool_counts), "skipped_reasons": dict(skip_counts),
        "dup_with_first_total": sum(r["dup_with_first"] for r in rows),
        "dup_within_sup_total": sum(r["dup_within_sup"] for r in rows),
        "dup_all_total": sum(r["dup_all"] for r in rows),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "first_claim_verdicts": dict(Counter(r["claim_verdicts"][0] if r["claim_verdicts"] else None for r in rows)),
        "avg_iteration": sum(r["iteration"] for r in rows) / n if n else 0.0,
        "avg_tool_calls": sum(r["n_tool_calls"] for r in rows) / n if n else 0.0,
        "extra_tokens": sum(r["extra_tokens"] for r in rows),
        "extra_llm_calls": sum(r["extra_llm_calls"] for r in rows),
        "avg_sup_latency_ms": (sum(r["sup_latency_ms"] for r in trig) / len(trig)) if trig else 0.0,
        "avg_elapsed": sum(r["elapsed"] for r in rows) / n if n else 0.0,
        "route_paths": dict(Counter(">".join(r["route_targets"]) for r in rows)),
        "final_sources": dict(sum((Counter(r["final_sources"]) for r in rows), Counter())),
        "pass": (sum(r["dup_all"] for r in rows) == 0) and len(trig) > 0 and sum(1 for r in rows if r["error"]) == 0,
    }


def write_report(s_route: Dict[str, Any], rows_route: List[Dict[str, Any]], s_check: Dict[str, Any], seed: int) -> None:
    s = s_route
    lines = [
        "# C-3 补证与重规划路由验收报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/eval/eval_route_c3.py`；seed={seed}。",
        "",
        "## 1. 设置",
        "",
        f"- 样本：D10 `data/eval/d10/end2end_eval.jsonl` 按 scenario 分层抽 {s['n']} 条；Planner 为 OraclePlanner（金标动作，不调 LLM）",
        "- Generator：`output_mode=claims` 模板路径（规则声明）；首轮草案额外注入 4 条无 evidence 的声明，文本分别命中 `fault_attribution / ett_forecast / kg_search / rag_search` 四条补证建议规则",
        "- Validator：`claim_check=route`，`claim_supplement_rounds=1`，`max_iterations=3`；对照组 `claim_check=check`（核查但不路由）",
        "- 去重签名：工具名 + 规范化参数（`ett_forecast` 补默认 dataset/lookback/horizon，`fault_attribution` 忽略 `query`，剔除 None，键排序）",
        "",
        "## 2. 主结果（route 模式）",
        "",
        "| 指标 | 值 | 验收线 |",
        "|---|---|---|",
        f"| 补证路由触发样本 | {s['triggered']}/{s['n']}（{s['trigger_rate'] * 100:.0f}%） | > 0 |",
        f"| 补证调用总数 / 平均每样本 | {s['n_sup_calls']} / {s['avg_sup_calls']:.2f} | - |",
        f"| 补证调用与首轮调用重复（同工具同参数） | {s['dup_with_first_total']} | 0 |",
        f"| 补证调用内部重复 | {s['dup_within_sup_total']} | 0 |",
        f"| 全部工具调用重复数 | {s['dup_all_total']} | 0 |",
        f"| 补证调用成功数 | {s['sup_success']}/{s['n_sup_calls']} | - |",
        f"| 最终 verdict 分布 | {s['verdicts']} | - |",
        f"| 首轮声明级核查 verdict | {s['first_claim_verdicts']} | - |",
        f"| 平均 Generator 轮次 / 平均工具调用数 | {s['avg_iteration']:.2f} / {s['avg_tool_calls']:.2f} | - |",
        f"| 补证轮额外 Token / LLM 调用 | {s['extra_tokens']} / {s['extra_llm_calls']} | 无 LLM 时为 0 |",
        f"| 补证轮平均耗时（触发样本） | {s['avg_sup_latency_ms']:.0f} ms | - |",
        f"| 工作流异常样本 | {s['n_errors']} | 0 |",
        f"| 平均总耗时 / 样本 | {s['avg_elapsed']:.2f} s | - |",
        "",
        f"**验收判定：{'通过' if s['pass'] else '未通过'}**（补证路由触发的调用中同工具同参数重复数为 0，且至少一条样本触发补证，无工作流异常）。",
        "",
        "## 3. 补证工具分布与去重跳过原因",
        "",
        "| 补证工具 | 调用数 |", "|---|---|",
    ]
    for k, v in sorted(s["sup_tool_counts"].items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "| 跳过原因 | 次数 |", "|---|---|"]
    for k, v in sorted(s["skipped_reasons"].items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "路径分布（Validator 路由目标序列）：", ""]
    for k, v in sorted(s["route_paths"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{k}`：{v}")
    lines += [
        "",
        "## 4. 对照：claim_check=check（核查不路由）",
        "",
        "| 指标 | route | check |",
        "|---|---|---|",
        f"| 补证触发样本 | {s['triggered']} | {s_check['triggered']} |",
        f"| 平均工具调用数 | {s['avg_tool_calls']:.2f} | {s_check['avg_tool_calls']:.2f} |",
        f"| 平均 Generator 轮次 | {s['avg_iteration']:.2f} | {s_check['avg_iteration']:.2f} |",
        f"| 最终 verdict | {s['verdicts']} | {s_check['verdicts']} |",
        f"| 最终声明证据来源 | {s['final_sources']} | {s_check['final_sources']} |",
        f"| 平均耗时 | {s['avg_elapsed']:.2f} s | {s_check['avg_elapsed']:.2f} s |",
        "",
        "说明：注入声明只出现在首轮，修订轮由规则重建，因此两种模式最终都会 PASS；差别在 route 模式为缺证声明补充了新的证据来源（kb / kg / ett 工具结果进入证据目录），check 模式仅删掉无据声明。",
        "",
        "## 5. 样例（前 3 条）",
        "",
    ]
    for r in rows_route[:3]:
        lines.append(f"### {r['eval_id']}（{r['scenario']}）")
        lines.append("")
        lines.append(f"- 首轮工具：{', '.join(r['first_tools']) or '无'}")
        lines.append(f"- 补证工具：{', '.join(r['sup_tools']) or '无'}；跳过：{r['skipped_tools'] or '无'}")
        lines.append(f"- 路由：{' → '.join(r['route_targets'])}；原因：{r['route_reasons']}")
        lines.append(f"- 声明级核查：{r['claim_verdicts']}；最终 verdict={r['verdict']}，iteration={r['iteration']}")
        lines.append("")
    lines += [
        "## 6. 说明与局限",
        "",
        "- 补证计划由规则构造（`workflow._build_supplement_plan`，不调 LLM）：按 `missing_evidence.suggested_tool` 选工具，`kg_search` 以最新归因的主故障名为 query，`fault_attribution` 复用上下文 DGA / evidence，`ett_forecast` 取上下文 dataset/horizon，`rag_search` 用 `suggested_query`。",
        "- 去重覆盖两层：与本轮已执行调用同签名 → `duplicate_of_existing_call`；计划内部同签名 → `duplicate_in_plan`；无建议工具 → `no_tool`；模式外工具 → `tool_disabled`；`kg_search` 候选查询词（主故障名 / 建议词 / 用户问题）均无法在图谱中定位实体 → `no_entity_in_kg`（避免必然 no_match 的无效调用）。",
        "- 未触发补证的样本为首轮规则声明较多（multi_tool）的场景：4 条注入声明占比未超过 `claim_unsupported_threshold=0.3`，声明级核查直接 PASS，属阈值设计内的行为。",
        "- Planner `active` 模板已加入 `missing_evidence` 处理规则（LLM 重规划路径），但本报告验证的是规则补证路径；LLM 重规划的去重效果需启用 LLM 后在 C-5 一并评测。",
        "- 每次 Validator 路由写入 `state.route_log`（触发原因 / 目标 / 补证工具 / 跳过 / 额外 Token / LLM 调用 / 耗时）。",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({"summary_route": s_route, "summary_check": s_check, "rows_route": rows_route},
                                      ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    samples = stratified_sample(load_d10(), args.n, args.seed)
    t0 = time.time()
    rows_route = run(samples, "route")
    s_route = summarize(rows_route, "route")
    rows_check = run(samples, "check")
    s_check = summarize(rows_check, "check")
    s_route["elapsed_total"] = round(time.time() - t0, 2)
    print(json.dumps({k: v for k, v in s_route.items() if k != "route_paths"}, ensure_ascii=False, indent=2))
    if not args.no_report:
        write_report(s_route, rows_route, s_check, args.seed)
        print(f"report -> {REPORT_MD}")
    return 0 if s_route["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
