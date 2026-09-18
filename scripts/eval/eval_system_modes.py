#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-1 / E-2：五级系统模式对比评测（D10 → mode1..mode5）。

模式定义见 src/graph/system_modes.py（PLAN v2 §0.5）：
  mode1 llm_only（无工具）/ mode2 tool_base（五工具 + 两路检索 + 图谱 + 词法反思，expert 归因，v1 Validator）
  / mode3 active_plan（calibrated 归因 + EIG + Planner active）/ mode4 claim_verify（claims + claim_check=route）
  / mode5 dpo_planner（Planner finetuned）
每级只比上一级多开一项能力；--planner-mode / --validator-mode / --attribution-mode / --planner-strategy /
--generator-output 可对所有被评测模式做显式覆盖（交叉对照，如 mode4 + baseline planner）。

指标（PLAN P6-2）：
  任务成功率  = 动作正确 ∧ 关键参数正确 ∧ 工具结果有效 ∧ 答案忠实于证据（四项同时满足）
    - 动作正确    ：required_actions 中每个 tool_call 的工具都被调用；ask_user / direct_answer 样本不得调用工具
                    （ask_user 还要求答案含追问措辞）；不必要调用（模式外或非必要工具）不影响“动作正确”，
                    但单独统计 不必要调用率
    - 关键参数正确：key_params 逐键比对（dga_data 数值容差 1e-6；relations 集合比较；query 词法重合 ≥ 0.5）
    - 工具结果有效：必要工具全部 exec_success ∧ business_success
    - 答案忠实    ：离线代理 = 证据锚点覆盖（kb：金标文档标题/章节词出现；kg：期望实体出现；
                    dga：工具返回的 primary_fault_name 出现在答案；ett：预测数值/趋势词出现）；
                    LLM 裁判（--judge llm）不可用时回退该代理并在报告中标注
  证据忠实度    ：同上代理的连续分值（0-1）
  平均工具调用次数 / 平均延迟 / Token 成本（src.utils.llm.USAGE）
  错误恢复成功率：expected_recovery 非空样本中，最终 recovered_call 工具业务成功的比例
  追问正确率    ：ask_user 样本中 未调用工具 ∧ 答案含追问措辞
  无依据结论率  ：mode4/5 下 Validator 声明级核查的 unsupported_ratio（首轮）均值；平均问询次数：active 策略的 inquiry_rounds

主动问询（mode3+）：D10 样本没有隐藏征兆真值，评测时 answer_fn 一律回答 None（「用户无法提供」），
追问循环因此最多问 K 轮后结论；若工作流因中断留下 pending_questions，则把追问文本视作最终答案参与打分。

用法：
  python3 scripts/eval/eval_system_modes.py --modes mode1,mode2,mode5 --limit 20      # 冒烟
  python3 scripts/eval/eval_system_modes.py --modes all --planner-mode baseline      # 微调模型未就绪时的对照
  python3 scripts/eval/eval_system_modes.py --modes mode4 --validator-mode off       # 交叉对照：mode4 + v1 validator
  python3 scripts/eval/eval_system_modes.py --modes all --no-llm                     # 仅流程连通性（Planner 禁用 → 全部 no plan）
输出：
  data/eval/d10/results/<mode>.jsonl   逐条结果
  docs/eval/end2end_eval.md                 汇总表（追加模式列，保留已有结果）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.state import AgentState                          # noqa: E402
from src.graph.workflow import run_diagnosis_workflow           # noqa: E402
from src.graph.system_modes import (                            # noqa: E402
    MODE_ORDER, SYSTEM_MODES, resolve_mode, register_rag_for_mode, build_agents, SystemModeSpec,
)
from src.tools.mcp_client import MCPClient                      # noqa: E402
from src.tools.fault_attribution import fault_attribution       # noqa: E402
from src.tools.kg_search import kg_search                       # noqa: E402
from src.tools.ett_forecasting import ett_forecast              # noqa: E402
from src.tools.timeseries import timeseries_anomaly             # noqa: E402
from src.utils.config import load_config                        # noqa: E402
from src.utils.data_guard import assert_not_synthetic           # noqa: E402
from src.utils.llm import USAGE                                 # noqa: E402

D10 = ROOT / "data/eval/d10/end2end_eval.jsonl"
RES_DIR = ROOT / "data/eval/d10/results"
REPORT = ROOT / "docs/eval/end2end_eval.md"

ASK_PATTERNS = ["请提供", "请补充", "需要您", "需要你", "缺少", "请告知", "请说明", "能否提供", "麻烦提供", "请问", "是否可以提供"]
TREND_WORDS = ["上升", "下降", "平稳", "趋势", "预测", "℃", "°C"]


def build_mcp(spec: SystemModeSpec) -> tuple[MCPClient, str]:
    mcp = MCPClient()
    mcp.register_tool("fault_attribution", fault_attribution)
    mcp.register_tool("timeseries_anomaly", timeseries_anomaly)
    mcp.register_tool("ett_forecast", ett_forecast)
    # kg_search 始终注册真实图谱；模式外调用由 Planner/Retriever 的 allowed_tools 拦截并计入不必要调用
    mcp.register_tool("kg_search", kg_search)
    kb_desc = register_rag_for_mode(mcp, spec)
    return mcp, kb_desc


# ──────────────────────────────────────────────────────────────
# 评分
# ──────────────────────────────────────────────────────────────
def _tokens(s: str) -> set:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(s or ""))
    toks = set()
    for w in s.split():
        if re.fullmatch(r"[A-Za-z0-9_.-]+", w):
            toks.add(w.lower())
        else:
            toks.update(w[i:i + 2] for i in range(max(len(w) - 1, 1)))
    return toks


def _query_match(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(tb) >= 0.5 or b in a or a in b


def _param_ok(tool: str, key: str, expect: Any, got: Any) -> bool:
    if key == "query":
        return _query_match(str(got or ""), str(expect or ""))
    if key == "dga_data":
        if not isinstance(got, dict) or not isinstance(expect, dict):
            return False
        return all(k in got and abs(float(got[k]) - float(v)) < 1e-6 for k, v in expect.items())
    if key == "relations":
        return set(got or []) == set(expect or [])
    if key == "signal":
        return isinstance(got, list) and len(got) == len(expect or []) and all(
            abs(float(x) - float(y)) < 1e-6 for x, y in zip(got, expect))
    if key in ("hops", "horizon"):
        return got is None and expect is None or (got is not None and int(got) == int(expect))
    return got == expect


def _match_calls(required: List[Dict[str, Any]], traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    把 required tool_call 与实际 trajectory 按工具名贪心匹配（同一工具多次调用取参数匹配最好的一次）。
    """
    used = set()
    matched, params_ok, results_ok = [], [], []
    for req in [a for a in required if a["type"] == "tool_call"]:
        cands = [(i, t) for i, t in enumerate(traj) if i not in used and t.get("tool") == req["tool"]]
        if not cands:
            matched.append(False); params_ok.append(False); results_ok.append(False)
            continue
        best, best_score = None, -1
        for i, t in cands:
            args = t.get("raw_arguments") or {}
            score = sum(1 for k, v in req["key_params"].items() if _param_ok(req["tool"], k, v, args.get(k)))
            if score > best_score:
                best, best_score = (i, t), score
        i, t = best
        used.add(i)
        matched.append(True)
        params_ok.append(best_score == len(req["key_params"]))
        results_ok.append(bool(t.get("exec_success") and t.get("business_success")))
    return {"matched": matched, "params_ok": params_ok, "results_ok": results_ok, "used_idx": used}


def _faithfulness_proxy(sample: Dict[str, Any], answer: str, tool_calls: List[Dict[str, Any]]) -> Optional[float]:
    """离线证据忠实度代理，返回 0-1；无法判定返回 None。"""
    ev = sample.get("evidence_source") or {}
    et = ev.get("type")
    ans = answer or ""
    if et == "kb_chunk":
        rag_calls = [c for c in tool_calls if c.get("tool") == "rag_search" and c.get("success")]
        if not rag_calls:
            return 0.0
        anchors = []
        if ev.get("doc_title"):
            anchors.append(ev["doc_title"])
        for p in ev.get("section_path") or []:
            anchors.append(re.sub(r"^[\d.．\s]+", "", p))
        # 检索到金标块本身也算命中
        hit_chunk = any(
            (c.get("tool") == "rag_search") and any(
                (r.get("chunk_id") in set(ev.get("chunk_ids") or [])) for r in (c.get("result") or []) if isinstance(r, dict))
            for c in tool_calls)
        cov = [a for a in anchors if a and _query_match(ans, a)]
        base = len(cov) / len(anchors) if anchors else 0.0
        return max(base, 1.0 if hit_chunk and base > 0 else (0.5 if hit_chunk else base))
    if et in ("kg_entities", "kg_path"):
        names = [e.split(":", 1)[-1] for e in (ev.get("expected_entities") or ev.get("path_prefix") or []) if ":" in e]
        if not names:
            return None
        return sum(1 for n in names if n in ans) / len(names)
    if et == "dga_label":
        for c in tool_calls:
            if c.get("tool") == "fault_attribution" and isinstance(c.get("result"), dict):
                pf = c["result"].get("primary_fault_name")
                if pf:
                    return 1.0 if pf in ans else 0.0
        return 0.0
    if et == "ett_dataset":
        for c in tool_calls:
            if c.get("tool") == "ett_forecast" and isinstance(c.get("result"), dict) and c["result"].get("status") == "ok":
                fc = (c["result"].get("forecast") or {}).get("mean_predicted_ot")
                num_hit = fc is not None and (f"{fc:.1f}" in ans or f"{fc:.0f}" in ans or str(fc) in ans)
                trend_hit = any(w in ans for w in TREND_WORDS)
                return 1.0 if num_hit else (0.6 if trend_hit else 0.0)
        return 0.0
    if et == "signal_inline":
        for c in tool_calls:
            if c.get("tool") == "timeseries_anomaly" and isinstance(c.get("result"), dict):
                idx = c["result"].get("anomaly_indices") or []
                return 1.0 if (not idx and any(w in ans for w in ["未发现", "无异常", "正常"])) or any(str(i) in ans for i in idx) else 0.5
        return 0.0
    if et == "multi":
        for c in tool_calls:
            if c.get("tool") == "fault_attribution" and isinstance(c.get("result"), dict):
                pf = c["result"].get("primary_fault_name")
                if pf:
                    return 1.0 if pf in ans else 0.0
        for c in tool_calls:
            if c.get("tool") == "ett_forecast" and isinstance(c.get("result"), dict) and c["result"].get("status") == "ok":
                fc = (c["result"].get("forecast") or {}).get("mean_predicted_ot")
                num_hit = fc is not None and (f"{fc:.1f}" in ans or f"{fc:.0f}" in ans or str(fc) in ans)
                trend_hit = any(w in ans for w in TREND_WORDS)
                return 1.0 if num_hit else (0.6 if trend_hit else 0.0)
        return 0.0
    if et == "none":
        return None
    if et == "trajectory":
        return None
    return None


def score_sample(sample: Dict[str, Any], final: AgentState, elapsed: float, usage: Dict[str, Any],
                 spec: SystemModeSpec) -> Dict[str, Any]:
    answer = final.final_answer or final.draft_answer or ""
    if not answer and getattr(final, "pending_questions", None):
        # active 策略在无 answer_fn 时遇 ask 中断：把追问文本当作系统的最终输出
        answer = "；".join(str(q.get("question") or "") for q in final.pending_questions if isinstance(q, dict))
        if answer and not any(p in answer for p in ASK_PATTERNS):
            answer = "请提供：" + answer
    traj = list(getattr(final, "trajectory", []) or [])
    tool_calls = list(final.tool_calls or [])
    required = sample["required_actions"]
    req_tools = [a for a in required if a["type"] == "tool_call"]
    kinds = {a["type"] for a in required}

    n_calls = len(traj)
    n_business_ok = sum(1 for t in traj if t.get("exec_success") and t.get("business_success"))
    n_disabled = sum(1 for t in traj if t.get("error_code") == "tool_disabled")

    m = _match_calls(required, traj)
    if req_tools:
        action_ok = all(m["matched"])
        params_ok = all(m["params_ok"])
        results_ok = all(m["results_ok"])
        unnecessary = n_calls - len([x for x in m["matched"] if x])
    else:
        action_ok = n_calls == 0
        params_ok = True
        results_ok = True
        unnecessary = n_calls
    ask_ok = None
    if "ask_user" in kinds:
        ask_ok = (n_calls == 0) and any(p in answer for p in ASK_PATTERNS)
        action_ok = action_ok and bool(ask_ok)

    faith = _faithfulness_proxy(sample, answer, tool_calls)
    faith_ok = True if faith is None else faith >= 0.5
    success = bool(action_ok and params_ok and results_ok and faith_ok)

    recovery_ok = None
    if sample.get("expected_recovery"):
        # 最终是否成功调用了恢复目标工具
        rec_tool = sample["expected_recovery"][-1]["recovered_call"]["tool"]
        recovery_ok = any(t.get("tool") == rec_tool and t.get("business_success") for t in traj)

    plan_content = next((it.get("content") for it in final.reasoning_trace
                         if it.get("agent") == "planner" and it.get("type") == "llm_plan"), {}) or {}
    # E-1 新增：主动问询次数（mode3+）、声明级核查首轮无依据比例（mode4+）、补证轮数（route）
    first_check = (final.claim_check_log or [{}])[0] if getattr(final, "claim_check_log", None) else {}
    return {
        "eval_id": sample["eval_id"], "scenario": sample["scenario"], "sub_type": sample["sub_type"],
        "mode": spec.key,
        "success": success, "action_ok": action_ok, "params_ok": params_ok, "results_ok": results_ok,
        "faithfulness": faith, "faith_ok": faith_ok, "ask_ok": ask_ok, "recovery_ok": recovery_ok,
        "n_tool_calls": n_calls, "n_business_ok": n_business_ok, "n_unnecessary": max(unnecessary, 0),
        "n_disabled": n_disabled,
        "plan_status": final.plan_status, "planner_model": plan_content.get("planner_model"),
        "verdict": final.validation_verdict, "iterations": final.iteration,
        "inquiry_rounds": int(getattr(final, "inquiry_rounds", 0) or 0),
        "pending_question": bool(getattr(final, "pending_questions", None)),
        "unsupported_ratio": first_check.get("unsupported_ratio"),
        "claim_verdict": first_check.get("verdict"),
        "evidence_rounds": int(getattr(final, "evidence_rounds", 0) or 0),
        "elapsed": elapsed, "tokens": usage.get("total_tokens", 0), "llm_calls": usage.get("calls", 0),
        "answer_len": len(answer),
        "tools_called": [t.get("tool") for t in traj],
        "answer_head": answer[:200],
    }


class OraclePlanner:
    """
    金标 Planner：直接把 D10 的 required_actions 转成计划（不调用 LLM）。
    用途：(1) 校验评测脚本的动作/参数匹配逻辑应得到 100%；(2) 在无 LLM 时测量工具层与 Generator 的上限。
    结果不得当作任何模式的正式成绩。
    """

    planner_mode = "oracle"
    model_name = "oracle"

    def __init__(self, allowed_tools: Optional[List[str]] = None) -> None:
        self.allowed = set(allowed_tools) if allowed_tools is not None else None
        self._sample: Dict[str, Any] = {}

    def bind(self, sample: Dict[str, Any]) -> None:
        self._sample = sample

    def run(self, state: AgentState) -> AgentState:
        from src.tools.tool_registry import validate_arguments_strict
        steps = []
        for i, a in enumerate(self._sample.get("required_actions", []), 1):
            if a["type"] != "tool_call":
                continue
            args = dict(a.get("key_params") or {})
            if a["tool"] == "fault_attribution":
                args["query"] = state.user_query
            step = {"id": i, "stage": "oracle", "description": "金标动作", "tool": a["tool"], "arguments": args}
            if self.allowed is not None and a["tool"] not in self.allowed:
                step["validation"] = {"valid": False, "errors": [{"field": None, "code": "tool_disabled", "message": "模式外"}]}
            else:
                step["validation"] = validate_arguments_strict(a["tool"], args)
            steps.append(step)
        kinds = {a["type"] for a in self._sample.get("required_actions", [])}
        intent = "（oracle）" + ("请提供缺失信息。" if "ask_user" in kinds else "")
        plan = {"intent_analysis": intent, "steps": steps, "plan_status": "ok" if steps else "no_tool",
                "planner_mode": "oracle", "planner_experiment_mode": "oracle", "planner_model": "oracle"}
        state.plan_status = plan["plan_status"]
        state.plan = intent
        state.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": plan})
        return state


def _no_answer(symptom: str, question: Dict[str, Any]) -> None:
    """D10 无隐藏征兆真值：追问一律回答「无法提供」（None），使 active 策略在 K 轮内自行结论。"""
    return None


def run_mode(spec: SystemModeSpec, samples: List[Dict[str, Any]], config: Dict[str, Any],
             oracle: bool = False, planner_decision: Optional[str] = None) -> List[Dict[str, Any]]:
    mcp, kb_desc = build_mcp(spec)
    planner, retriever, generator, validator, reflector = build_agents(config, mcp, spec,
                                                                       planner_decision=planner_decision)
    if oracle:
        planner = OraclePlanner(spec.allowed_tools)
    answer_fn = _no_answer if getattr(planner, "strategy", "free") == "active" else None
    print(f"\n=== {spec.label} [{spec.name}] | {spec.summary()} | rag_impl={kb_desc} "
          f"planner_model={getattr(planner, 'model_name', '?')} ===")
    out = []
    for i, s in enumerate(samples, 1):
        if oracle:
            planner.bind(s)
        state = AgentState(user_query=s["user_query"], context=dict(s.get("context") or {}),
                           max_iterations=config.get("workflow", {}).get("max_iterations", 3),
                           max_inquiry_rounds=int(config.get("workflow", {}).get("max_inquiry_rounds", 3)))
        USAGE.reset()
        t0 = time.time()
        try:
            final = run_diagnosis_workflow(state, planner, retriever, generator, validator, reflector=reflector,
                                           answer_fn=answer_fn)
            r = score_sample(s, final, time.time() - t0, USAGE.snapshot(), spec)
        except Exception as exc:  # noqa: BLE001
            r = {"eval_id": s["eval_id"], "scenario": s["scenario"], "sub_type": s["sub_type"], "mode": spec.key,
                 "success": False, "error": str(exc), "elapsed": time.time() - t0, "n_tool_calls": 0}
        out.append(r)
        print(f"  [{i}/{len(samples)}] {s['eval_id']} {s['scenario']:<20} ok={r.get('success')} "
              f"act={r.get('action_ok')} par={r.get('params_ok')} res={r.get('results_ok')} "
              f"faith={r.get('faithfulness')} calls={r.get('n_tool_calls')} plan={r.get('plan_status')}"
              + (f" ask={r.get('inquiry_rounds')}" if spec.planner_strategy == "active" else "")
              + (f" unsup={r.get('unsupported_ratio')}" if spec.validator_mode != "off" else "")
              + (f" ERR={r['error'][:80]}" if r.get("error") else ""))
    return out


# ──────────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────────
def _rate(rows: List[Dict[str, Any]], key: str, pred=lambda v: bool(v)) -> Optional[float]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return (sum(1 for v in vals if pred(v)) / len(vals)) if vals else None


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "task_success": _rate(rows, "success"),
        "action_acc": _rate(rows, "action_ok"),
        "param_acc": _rate(rows, "params_ok"),
        "tool_result_valid": _rate(rows, "results_ok"),
        "faithfulness": _mean(rows, "faithfulness"),
        "ask_acc": _rate(rows, "ask_ok"),
        "recovery_rate": _rate(rows, "recovery_ok"),
        "avg_tool_calls": _mean(rows, "n_tool_calls"),
        "unnecessary_call_rate": (sum(r.get("n_unnecessary", 0) for r in rows) / max(sum(r.get("n_tool_calls", 0) for r in rows), 1)),
        "avg_latency_s": _mean(rows, "elapsed"),
        "avg_tokens": _mean(rows, "tokens"),
        "avg_inquiry_rounds": _mean(rows, "inquiry_rounds"),
        "unsupported_rate": _mean(rows, "unsupported_ratio"),
        "avg_evidence_rounds": _mean(rows, "evidence_rounds"),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "plan_status": dict(sorted(__import__("collections").Counter(str(r.get("plan_status")) for r in rows).items())),
        "by_scenario": {
            sc: _rate([r for r in rows if r.get("scenario") == sc], "success")
            for sc in sorted({r.get("scenario") for r in rows})
        },
    }


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1%}" if pct else f"{v:.2f}"
    return str(v)


def write_report(summaries: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    modes = [m for m in MODE_ORDER if m in summaries]
    overrides = {k: v for k, v in (meta.get("overrides") or {}).items() if v}
    L = ["# 端到端评测：五级系统模式对比（E-1 / E-2）\n",
         f"- 评测集：`data/eval/d10/end2end_eval.jsonl`（D10，{meta.get('n_samples')} 条{'，--limit 截断' if meta.get('limited') else ''}）",
         f"- 运行时间：{meta.get('time')}；LLM：{'禁用（仅流程连通性）' if meta.get('no_llm') else meta.get('llm_desc')}",
         "- 显式覆盖：" + (", ".join(f"`--{k.replace('_', '-')} {v}`" for k, v in overrides.items()) if overrides
                         else "无（按模式默认；mode5 的 finetuned Planner 未配置时自动回退 baseline）"),
         "- 忠实度：" + ("LLM 裁判" if meta.get("judge") == "llm" else "离线证据锚点代理（非 LLM 裁判，见指标说明）"),
         "", "## 模式开关", "",
         "| 模式 | 名称 | 工具 | 检索 | 反思 | 归因 | EIG | Planner | 策略 | Generator | Validator |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in modes:
        sp = summaries[m].get("spec") or SYSTEM_MODES[m].to_dict()
        tools = "无" if not sp["allowed_tools"] else ("五工具" if len(sp["allowed_tools"]) == 5 else "、".join(sp["allowed_tools"]))
        L.append(f"| {m} | `{sp['name']}` | {tools} | {sp['rag_mode'] or '—'} | {sp['reflection']} | {sp['attribution_mode']} | "
                 f"{'开' if sp['with_eig'] else '关'} | {sp['planner_mode']} | {sp['planner_strategy']} | "
                 f"{sp['generator_output']} | {sp['validator_mode']} |")
    L += ["", "## 主表", "",
          "| 指标 | " + " | ".join(summaries[m]["label"] for m in modes) + " |",
          "|---|" + "---:|" * len(modes)]
    rows = [("任务成功率", "task_success", True), ("动作正确率", "action_acc", True), ("关键参数正确率", "param_acc", True),
            ("工具结果有效率", "tool_result_valid", True), ("证据忠实度（0-1）", "faithfulness", False),
            ("追问正确率", "ask_acc", True), ("错误恢复成功率", "recovery_rate", True),
            ("无依据结论率（首轮声明核查）", "unsupported_rate", True),
            ("平均问询次数", "avg_inquiry_rounds", False), ("平均补证轮数", "avg_evidence_rounds", False),
            ("平均工具调用次数", "avg_tool_calls", False), ("不必要调用率", "unnecessary_call_rate", True),
            ("平均延迟 (s)", "avg_latency_s", False), ("平均 Token", "avg_tokens", False),
            ("运行异常条数", "n_errors", False)]
    for name, key, pct in rows:
        L.append(f"| {name} | " + " | ".join(_fmt(summaries[m].get(key), pct) for m in modes) + " |")
    L += ["", "## 分场景任务成功率", "",
          "| 场景 | " + " | ".join(summaries[m]["label"] for m in modes) + " |", "|---|" + "---:|" * len(modes)]
    scenarios = sorted({sc for m in modes for sc in summaries[m]["by_scenario"]})
    for sc in scenarios:
        L.append(f"| {sc} | " + " | ".join(_fmt(summaries[m]["by_scenario"].get(sc), True) for m in modes) + " |")
    L += ["", "## Planner 计划状态分布", ""]
    for m in modes:
        L.append(f"- {summaries[m]['label']}：{summaries[m]['plan_status']}")
    L += ["", "## 指标说明", "",
          "- 任务成功率 = 动作正确 ∧ 关键参数正确 ∧ 工具结果有效 ∧ 答案忠实（四项同时满足）。",
          "- 动作正确：必要工具全部被调用；ask_user/direct_answer 样本不得调用工具；ask_user 还需答案含追问措辞。",
          "- 关键参数：只比对 D10 `key_params`；`query` 用词法重合 ≥ 0.5 判定，`dga_data` 数值容差 1e-6，`relations` 集合相等。",
          "- 证据忠实度（离线代理）：kb 样本看金标文档标题/章节词与检索命中；kg 样本看期望实体出现；dga 样本看工具返回的主故障名是否被答案引用；"
          "ett 看预测数值/趋势词。该代理偏向词面匹配，正式报告需 LLM 裁判 + 10% 人工复核。",
          "- 无依据结论率：Validator 声明级核查（mode4/5，或 `--validator-mode check|route`）首轮 `unsupported_ratio` 均值；v1 Validator 不产出该值（—）。",
          "- 平均问询次数：active 策略（mode3+）的 `inquiry_rounds` 均值；D10 无隐藏征兆真值，追问一律按「无法提供」回答。",
          "- 不必要调用率 = 非必要调用次数 / 总调用次数；模式外工具被 Retriever 拒绝执行（tool_disabled）也计入。",
          "- Token 为 `src.utils.llm.USAGE` 累计的 prompt+completion；LLM 禁用时为 0。",
          "", "## 已知局限", "",
          "- LLM 禁用时 free 策略的 Planner 不产出计划，需要工具的样本均失败；active 策略回退 EIG 规则，仅在有 DGA 的样本上调用归因。此时只验证流程连通性与 no_tool/ask_user 的“不调用”口径。",
          "- mode5 依赖 `llms.planner_finetuned`（百炼部署的 DPO 模型）；未部署时自动回退 baseline，mode5 与 mode4 等价。",
          "- D10 查询未经口语化改写（A-2 rewrite 需 LLM），对 Planner 偏乐观。",
          ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(D10))
    ap.add_argument("--modes", default="all", help="逗号分隔：mode1,mode2,... 或 all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--planner-mode", default=None, choices=["baseline", "finetuned"], help="覆盖所有被评测模式的 planner_mode")
    ap.add_argument("--validator-mode", default=None, choices=["off", "check", "route"], help="覆盖 Validator 核查模式（交叉对照：mode5 + v1 validator）")
    ap.add_argument("--attribution-mode", default=None, choices=["expert", "calibrated"], help="覆盖归因引擎参数")
    ap.add_argument("--planner-strategy", default=None, choices=["free", "active"], help="覆盖 Planner 策略")
    ap.add_argument("--generator-output", default=None, choices=["text", "claims"], help="覆盖 Generator 输出结构")
    ap.add_argument("--planner-decision", default=None, choices=["llm", "eig"], help="active 策略的动作决策来源（默认取 config）")
    ap.add_argument("--reflection", default=None, choices=["off", "lexical", "llm"], help="覆盖反思评分器（mode2+）")
    ap.add_argument("--judge", default="proxy", choices=["proxy", "llm"], help="忠实度判定：proxy 离线代理；llm 裁判（待接入）")
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM（清空 api_key），仅验证流程")
    ap.add_argument("--scenario", default="", help="只评测某些场景，逗号分隔")
    ap.add_argument("--oracle-planner", action="store_true",
                    help="用 D10 金标动作替代 Planner（校验评分链路 / 工具层上限），结果写入 results/oracle_<mode>.jsonl，不进主表")
    ap.add_argument("--tag", default="", help="结果文件后缀（交叉对照实验时避免覆盖主结果，且不写入主报告）")
    args = ap.parse_args()

    assert_not_synthetic(args.eval, purpose="eval_set")
    samples = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    if args.scenario:
        keep = set(args.scenario.split(","))
        samples = [s for s in samples if s["scenario"] in keep]
    if args.limit:
        samples = samples[: args.limit]

    config = load_config()
    if args.no_llm:
        for v in config.get("llms", {}).values():
            if isinstance(v, dict):
                v["api_key"] = ""
                v["api_key_env"] = "___DISABLED___"
    if args.judge == "llm":
        print("[warn] LLM 裁判尚未接入，本次回退离线代理")
        args.judge = "proxy"

    overrides = {"planner_mode": args.planner_mode, "validator_mode": args.validator_mode,
                 "attribution_mode": args.attribution_mode, "planner_strategy": args.planner_strategy,
                 "generator_output": args.generator_output, "reflection": args.reflection}
    modes = MODE_ORDER if args.modes == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]
    RES_DIR.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, Any]] = {}
    side_run = bool(args.oracle_planner or args.tag)
    # 保留历史结果，只覆盖本次运行的模式
    if not side_run:
        for m in MODE_ORDER:
            p = RES_DIR / f"{m}.jsonl"
            if p.exists() and m not in modes:
                rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                if rows:
                    summaries[m] = {**summarize(rows), "label": resolve_mode(m).label, "spec": resolve_mode(m).to_dict()}
    for m in modes:
        spec = resolve_mode(m, **overrides)
        rows = run_mode(spec, samples, config, oracle=args.oracle_planner, planner_decision=args.planner_decision)
        fname = f"{'oracle_' if args.oracle_planner else ''}{m}{('_' + args.tag) if args.tag else ''}.jsonl"
        with open(RES_DIR / fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries[m] = {**summarize(rows), "label": spec.label, "spec": spec.to_dict()}
        print(json.dumps({k: v for k, v in summaries[m].items() if k not in ("by_scenario", "spec")}, ensure_ascii=False))

    if args.oracle_planner:
        print("\n[oracle] 金标 Planner 结果仅用于校验评分链路，不写入 docs/eval/end2end_eval.md")
        return
    if args.tag:
        print(f"\n[tag={args.tag}] 交叉对照结果已写入 {RES_DIR}/<mode>_{args.tag}.jsonl，不写入主报告")
        return
    llm_desc = (config.get("llms", {}).get("default", {}) or {}).get("model_name", "—")
    write_report(summaries, {"n_samples": len(samples), "limited": bool(args.limit or args.scenario),
                             "time": time.strftime("%Y-%m-%d %H:%M"), "no_llm": args.no_llm,
                             "llm_desc": llm_desc, "overrides": overrides, "judge": args.judge})
    print(f"\n报告已写入 {REPORT}")


if __name__ == "__main__":
    main()
