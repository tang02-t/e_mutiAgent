"""
C-3 补证与重规划路由 单元测试（无 LLM、无 GPU）。

运行：python3 tests/test_c3_route.py

覆盖：
[1] _canonical_args / _called_signatures：同工具同参数签名（默认值补齐、None 剔除、query 忽略）
[2] _build_supplement_plan：按 missing_evidence 构造补证计划；去重（已执行 / 计划内重复）、无工具、模式外工具
[3] _decide_supplement_route：PASS / ABSTAIN / FAIL / REVISION(off|check|route) / 无 missing / 轮数耗尽 → 目标与原因
[4] _supplement_node：写 reasoning_trace 合成计划、evidence_rounds+1、route_log 记录工具与 skipped、清空 draft_claims、
    revision_feedback 追加补证说明、无可执行步骤时直接回 generator
[5] Retriever 补证轮累积 tool_calls / retrieved_knowledge 且 call_index 连续（free 策略）
[6] 端到端（真实工具 + 本地 BM25 + 模板 Generator）：注入无依据声明 → Validator route → supplement → retriever → generator → PASS；
    route_log 含 extra_tokens / latency_ms；补证调用无重复同工具同参数
[7] Planner _render_context 渲染 missing_evidence；active 模板含补证规则
[8] claim_check=off / check 下工作流行为不变（不出现 supplement）
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                              # noqa: E402
from src.graph import workflow as wf                                                # noqa: E402
from src.agents.validator import ValidatorAgent                                     # noqa: E402
from src.agents.generator import GeneratorAgent                                     # noqa: E402
from src.agents.planner import PlannerAgent                                         # noqa: E402
from src.agents.claims import build_rule_claims, make_claim, make_evidence          # noqa: E402

logging.disable(logging.WARNING)

PASS = 0
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


NO_LLM_CFG: Dict[str, Any] = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 3, "planner_strategy": "free", "attribution_mode": "calibrated",
                 "generator_output_mode": "claims", "claim_check": "route", "claim_supplement_rounds": 1},
}
DGA = {"H2": 150.0, "CH4": 40.0, "C2H2": 25.0, "C2H4": 30.0, "C2H6": 10.0}


def _cfg(**over: Any) -> Dict[str, Any]:
    c = json.loads(json.dumps(NO_LLM_CFG))
    c["workflow"].update(over)
    return c


def _tc(tool: str, args: Dict[str, Any], idx: int, result: Any = None, success: bool = True) -> Dict[str, Any]:
    return {"tool": tool, "args": args, "raw_arguments": args, "success": success, "exec_success": success,
            "business_success": success, "result": result if result is not None else {"status": "ok"},
            "call_index": idx, "error": None, "error_code": None, "latency_ms": 1.0}


class _FakeValidator:
    def __init__(self, claim_check: str = "route", supplement_rounds: int = 1) -> None:
        self.claim_check = claim_check
        self.supplement_rounds = supplement_rounds


# ─────────────────────────────────────────────────────────────
print("[1] canonical args / signatures")
check(wf._canonical_args("ett_forecast", {}) == wf._canonical_args("ett_forecast", {"dataset": "ETTh1", "horizon": 24, "lookback": 96}),
      "ett_forecast 默认参数补齐后签名一致")
check(wf._canonical_args("ett_forecast", {"dataset": "ETTm1"}) != wf._canonical_args("ett_forecast", {}),
      "ett_forecast 换数据集签名不同")
check(wf._canonical_args("fault_attribution", {"query": "a", "dga_data": DGA}) ==
      wf._canonical_args("fault_attribution", {"query": "b", "dga_data": DGA}), "fault_attribution 忽略 query")
check(wf._canonical_args("rag_search", {"query": "x", "k": None}) == wf._canonical_args("rag_search", {"query": "x"}),
      "None 参数剔除")
st = AgentState(user_query="q")
st.tool_calls = [_tc("rag_search", {"query": "局部放电 处理"}, 0), _tc("kg_search", {"query": "局部放电", "hops": 1}, 1)]
sigs = wf._called_signatures(st)
check(("rag_search", wf._canonical_args("rag_search", {"query": "局部放电 处理"})) in sigs, "签名集合包含已执行 rag_search")
check(len(sigs) == 2, "签名集合大小 2")

# ─────────────────────────────────────────────────────────────
print("[2] build_supplement_plan")
st = AgentState(user_query="变压器乙炔升高怎么处理", context={"dga": DGA, "evidence": {"C2H2_elevated": True}})
st.tool_calls = [
    _tc("fault_attribution", {"query": st.user_query, "dga_data": DGA, "evidence": {"C2H2_elevated": True}}, 0,
        {"status": "ok", "primary_fault_name": "局部放电", "fault_ranking": []}),
    _tc("rag_search", {"query": "乙炔 处理"}, 1),
]
st.missing_evidence = [
    {"claim_id": "c1", "suggested_tool": "fault_attribution", "suggested_query": "乙炔 25 ppm", "reason": "无证据引用"},
    {"claim_id": "c2", "suggested_tool": "kg_search", "suggested_query": "局部放电 导致", "reason": "无证据引用"},
    {"claim_id": "c3", "suggested_tool": "kg_search", "suggested_query": "放电 机理", "reason": "无证据引用"},
    {"claim_id": "c4", "suggested_tool": "rag_search", "suggested_query": "停电检修", "reason": "无证据引用"},
    {"claim_id": "c5", "suggested_tool": None, "suggested_query": "xx", "reason": "无证据引用"},
    {"claim_id": "c6", "suggested_tool": "ett_forecast", "suggested_query": "油温", "reason": "无证据引用"},
]
plan = wf._build_supplement_plan(st)
tools = [s["tool"] for s in plan["steps"]]
check(plan["decision_source"] == "supplement" and plan["action"] == "call_tool", "计划 decision_source=supplement / call_tool")
check("fault_attribution" not in tools, "与已执行归因同参数 → 跳过")
check(tools.count("kg_search") == 1, "两条 kg_search 建议（同一 primary_fault 作 query）计划内去重为 1")
check("rag_search" in tools and "ett_forecast" in tools, "rag_search（新检索词）与 ett_forecast 进入计划")
reasons = {k["claim_id"]: k["reason"] for k in plan["skipped"]}
check(reasons.get("c1") == "duplicate_of_existing_call", "c1 跳过原因 duplicate_of_existing_call")
check(reasons.get("c3") == "duplicate_in_plan", "c3 跳过原因 duplicate_in_plan")
check(reasons.get("c5") == "no_tool", "c5 无建议工具 → no_tool")
check(all(s["validation"]["valid"] for s in plan["steps"]), "所有补证步骤通过参数严格校验")
check(all(s.get("for_claims") for s in plan["steps"]), "步骤带 for_claims")
kg_step = next(s for s in plan["steps"] if s["tool"] == "kg_search")
check(kg_step["arguments"]["query"] == "局部放电", "kg_search 用最新归因 primary_fault_name 作 query")
rag_step = next(s for s in plan["steps"] if s["tool"] == "rag_search")
check(rag_step["arguments"]["query"] == "停电检修", "rag_search 用 suggested_query")
plan2 = wf._build_supplement_plan(st, allowed_tools=["fault_attribution", "rag_search"])
check([s["tool"] for s in plan2["steps"]] == ["rag_search"], "allowed_tools 限制模式外工具")
check(any(k["reason"] == "tool_disabled" for k in plan2["skipped"]), "模式外工具 skipped 原因 tool_disabled")
st.missing_evidence = []
plan3 = wf._build_supplement_plan(st)
check(plan3["steps"] == [] and plan3["action"] == "conclude" and plan3["plan_status"] == "empty", "无 missing_evidence → 空计划 conclude")

# ─────────────────────────────────────────────────────────────
print("[3] decide_supplement_route")


def _routed(verdict: str, next_node: str, missing: List[Dict[str, Any]], validator, rounds: int = 0):
    s = AgentState(user_query="q")
    s.validation_verdict = verdict
    s.next_node = next_node
    s.missing_evidence = missing
    s.evidence_rounds = rounds
    s.iteration = 1
    s.claim_check_log = [{"verdict": "REVISION" if verdict == "REVISION" else verdict, "unsupported_ratio": 0.5}]
    wf._decide_supplement_route(s, validator)
    return s


m1 = [{"claim_id": "c1", "suggested_tool": "kg_search", "suggested_query": "x", "reason": "r"}]
s = _routed("PASS", "END", [], _FakeValidator())
check(s.next_node == "END" and s.route_log[-1]["reason"] == "pass", "PASS → END")
s = _routed("ABSTAIN", "END", m1, _FakeValidator())
check(s.next_node == "END" and s.route_log[-1]["reason"] == "abstain", "ABSTAIN → END（即便有 missing）")
s = _routed("FAIL", "END", m1, _FakeValidator())
check(s.next_node == "END" and s.route_log[-1]["reason"] == "fail", "FAIL → END")
s = _routed("REVISION", "generator", m1, _FakeValidator("route"))
check(s.next_node == "supplement" and s.route_log[-1]["target"] == "supplement" and s.route_log[-1]["reason"].startswith("unsupported"),
      "REVISION + route + missing → supplement")
s = _routed("REVISION", "generator", [], _FakeValidator("route"))
check(s.next_node == "generator" and s.route_log[-1]["reason"] == "revision:contradict_or_constraint", "REVISION 无 missing（矛盾/约束）→ generator")
s = _routed("REVISION", "generator", m1, _FakeValidator("check"))
check(s.next_node == "generator" and "claim_check!=route" in s.route_log[-1]["reason"], "claim_check=check → 不补证")
s = _routed("REVISION", "generator", m1, _FakeValidator("off"))
check(s.next_node == "generator", "claim_check=off → generator")
s = _routed("REVISION", "generator", m1, _FakeValidator("route", 1), rounds=1)
check(s.next_node == "generator" and "exhausted" in s.route_log[-1]["reason"], "补证轮数耗尽 → generator")
s = _routed("REVISION", "END", m1, _FakeValidator("route"))
check(s.next_node == "END", "REVISION 但 Validator 已判 END（达 max_iterations）→ 保持 END")
check(all(k in s.route_log[-1] for k in ("iteration", "verdict", "claim_check_verdict", "target", "reason", "n_missing")),
      "route_log 字段完整")

# ─────────────────────────────────────────────────────────────
print("[4] supplement_node")
st = AgentState(user_query="乙炔升高", context={"dga": DGA})
st.tool_calls = [_tc("fault_attribution", {"query": "乙炔升高", "dga_data": DGA}, 0, {"status": "ok", "primary_fault_name": "局部放电"})]
st.missing_evidence = [{"claim_id": "c9", "suggested_tool": "kg_search", "suggested_query": "放电", "reason": "无证据引用"},
                       {"claim_id": "c10", "suggested_tool": None, "suggested_query": "", "reason": "无证据引用"}]
st.draft_claims = [{"id": "c9", "text": "x", "type": "inference", "evidence": []}]
st.revision_feedback = "旧反馈"
st.validation_verdict = "REVISION"
st.next_node = "supplement"
st.route_log = [{"iteration": 1, "verdict": "REVISION", "target": "supplement", "reason": "unsupported:1"}]
st = wf._supplement_node(st)
check(st.evidence_rounds == 1, "evidence_rounds +1")
check(st.next_node == "retriever", "有可执行步骤 → retriever")
check(st.reasoning_trace and st.reasoning_trace[-1]["content"]["decision_source"] == "supplement", "合成补证计划写入 reasoning_trace")
check(wf._latest_plan(st)["steps"][0]["tool"] == "kg_search", "_latest_plan 取到补证计划")
check(st.draft_claims == [] and st.missing_evidence == [], "清空 draft_claims / missing_evidence")
check(st.route_log[-1]["tools"][0]["tool"] == "kg_search" and st.route_log[-1]["round"] == 1, "route_log 记录补证工具与轮次")
check(st.route_log[-1]["skipped"][0]["claim_id"] == "c10", "route_log 记录跳过声明")
check("t_start" in st.route_log[-1] and "tokens_before" in st.route_log[-1], "route_log 记录起始 token / 时间快照")
check("【补证结果】" in (st.revision_feedback or "") and "旧反馈" in st.revision_feedback, "revision_feedback 追加补证说明且保留旧反馈")
check("c10 无法补证" in st.revision_feedback, "反馈中列出无法补证的声明")
check(st.context.get("missing_evidence") and st.context["missing_evidence"][0]["claim_id"] == "c9", "missing_evidence 写入 context 供 Planner 渲染")
wf._finalize_supplement_cost(st)
e = st.route_log[-1]
check("latency_ms" in e and "extra_tokens" in e and "extra_llm_calls" in e and "t_start" not in e, "回填 latency_ms / extra_tokens / extra_llm_calls")
check(e["extra_tokens"] == 0 and e["extra_llm_calls"] == 0, "无 LLM 时额外 Token / 调用为 0")
# 全部去重 → 直接 generator
st2 = AgentState(user_query="乙炔升高", context={"dga": DGA})
st2.tool_calls = [_tc("kg_search", {"query": "局部放电", "hops": 1}, 0),
                  _tc("fault_attribution", {"query": "乙炔升高", "dga_data": DGA}, 1, {"status": "ok", "primary_fault_name": "局部放电"})]
st2.missing_evidence = [{"claim_id": "c1", "suggested_tool": "kg_search", "suggested_query": "放电", "reason": "r"}]
st2 = wf._supplement_node(st2)
check(st2.next_node == "generator", "去重后无步骤 → 直接 generator")
check(wf._route_after_supplement(st2) == "generator" and wf._route_after_supplement(st) == "retriever", "_route_after_supplement 路由正确")

# ─────────────────────────────────────────────────────────────
print("[5] Retriever 补证轮累积（free 策略）")
from src.graph.system_modes import resolve_mode, build_agents                        # noqa: E402
from eval_system_modes import build_mcp                                             # noqa: E402
from src.tools.fault_attribution import configure_engine                           # noqa: E402

configure_engine("calibrated")
spec = resolve_mode("mode5", planner_mode="baseline", reflection="off")
mcp, _kb = build_mcp(spec)
_, retriever, _, _, _ = build_agents(_cfg(), mcp, spec)

st = AgentState(user_query="变压器乙炔 25 ppm，可能是什么故障？", context={"dga": DGA})
st.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": {
    "steps": [{"id": 1, "tool": "fault_attribution", "arguments": {"query": st.user_query, "dga_data": DGA}},
              {"id": 2, "tool": "rag_search", "arguments": {"query": "乙炔升高 处理"}}], "plan_status": "ok"}})
st = retriever.run(st)
n0, kb0 = len(st.tool_calls), len(st.retrieved_knowledge)
check(n0 == 2 and kb0 > 0, f"首轮 2 次调用，KB 命中 {kb0}")
st.missing_evidence = [{"claim_id": "c1", "suggested_tool": "kg_search", "suggested_query": "放电", "reason": "r"},
                       {"claim_id": "c2", "suggested_tool": "rag_search", "suggested_query": "局部放电 停电 检修 依据", "reason": "r"}]
st = wf._supplement_node(st)
st = retriever.run(st)
check(len(st.tool_calls) == n0 + 2, "补证轮 tool_calls 累积（2 + 2）")
check([c["call_index"] for c in st.tool_calls] == [0, 1, 2, 3], "call_index 连续")
check(len(st.retrieved_knowledge) >= kb0, "retrieved_knowledge 累积不减少")
ids = [str(k.get("chunk_id") or k.get("id") or k.get("text", "")[:80]) for k in st.retrieved_knowledge]
check(len(ids) == len(set(ids)), "累积 KB 片段无重复 chunk")
# 空补证计划不清空
st3 = AgentState(user_query="q")
st3.tool_calls = [_tc("rag_search", {"query": "x"}, 0)]
st3.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": {"decision_source": "supplement", "steps": []}})
st3 = retriever.run(st3)
check(len(st3.tool_calls) == 1, "补证空计划不清空既有 tool_calls")

# ─────────────────────────────────────────────────────────────
print("[6] 端到端补证路由（route 模式，注入无依据声明）")


class InjectGenerator(GeneratorAgent):
    """首轮在规则声明上注入无依据声明（触发 unsupported），修订轮正常重建。"""

    def __init__(self, cfg, inject: List[Dict[str, Any]]):
        super().__init__(cfg, output_mode="claims")
        self.inject = inject
        self.calls = 0

    def run(self, state, revision_feedback: str = ""):
        state = super().run(state, revision_feedback=revision_feedback)
        self.calls += 1
        if self.calls == 1:
            state.draft_claims = list(state.draft_claims) + list(self.inject)
        return state


class OracleLike:
    strategy = "free"

    def __init__(self, steps):
        self.steps = steps

    def run(self, state):
        from src.tools.tool_registry import validate_arguments_strict
        steps = [{**s, "id": i, "validation": validate_arguments_strict(s["tool"], s["arguments"])} for i, s in enumerate(self.steps, 1)]
        state.plan_status = "ok"
        state.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": {"steps": steps, "plan_status": "ok"}})
        return state


inject = [
    make_claim("x1", "局部放电与乙炔生成存在图谱关系链路", "inference", []),
    make_claim("x2", "依据规程建议停电后进行局部放电定位检测", "recommendation", []),
    make_claim("x3", "未来 24 小时油温预测平均约 30 ℃", "inference", []),
    make_claim("x4", "建议按规程复测油色谱并安排带电局放检测", "recommendation", []),
]
cfg = _cfg()
_, retriever, _, validator, _ = build_agents(cfg, mcp, spec)
gen = InjectGenerator(cfg, inject)
planner = OracleLike([{"tool": "fault_attribution", "arguments": {"query": "变压器乙炔 25 ppm，可能是什么故障？", "dga_data": DGA}}])
state = AgentState(user_query="变压器乙炔 25 ppm，可能是什么故障？", context={"dga": DGA}, max_iterations=3)
final = wf.run_diagnosis_workflow(state, planner, retriever, gen, validator)
targets = [r["target"] for r in final.route_log]
check("supplement" in targets, f"路由经过 supplement：{targets}")
check(final.evidence_rounds == 1, "补证 1 轮")
sup = next(r for r in final.route_log if r["target"] == "supplement")
sup_tools = [t["tool"] for t in sup["tools"]]
check(set(sup_tools) <= {"kg_search", "rag_search", "ett_forecast"} and len(sup_tools) >= 2, f"补证调用工具：{sup_tools}")
check("fault_attribution" not in sup_tools, "补证不重复已执行的归因")
sig_all = [(c["tool"], wf._canonical_args(c["tool"], c.get("args") or {})) for c in final.tool_calls]
check(len(sig_all) == len(set(sig_all)), "全部工具调用无重复同工具同参数")
check("latency_ms" in sup and "extra_tokens" in sup and sup["extra_tokens"] == 0, "补证轮回填耗时与额外 Token（0）")
check(final.validation_verdict == "PASS", f"补证后修订通过（verdict={final.validation_verdict}）")
check(final.claim_check_log[0]["verdict"] == "REVISION" and final.claim_check_log[-1]["verdict"] == "PASS", "claim_check_log 首轮 REVISION、末轮 PASS")
check(final.iteration == 2, f"Generator 共运行 2 次（iteration={final.iteration}）")
check(len(final.tool_calls) == 1 + len(sup_tools), "tool_calls 累积 = 首轮 1 + 补证数")
srcs = {e["source"] for c in final.draft_claims for e in c.get("evidence", [])}
check("kg" in srcs or "kb" in srcs, f"修订后声明引用了补证证据来源：{sorted(srcs)}")
check(final.route_log[-1]["target"] == "END" and final.route_log[-1]["reason"] == "pass", "最后一条路由 END/pass")
check("[c" in (final.final_answer or "") or "【声明与证据清单】" in (final.final_answer or ""), "最终答案含声明清单")

# 无建议工具的缺证声明：REVISION 但不进 supplement（no_actionable），交 Generator 删改
inject2 = [make_claim(f"y{i}", f"该结论与第 {i} 份运行记录一致", "inference", []) for i in range(1, 6)]


class Inject2(InjectGenerator):
    def run(self, state, revision_feedback: str = ""):
        state = GeneratorAgent.run(self, state, revision_feedback=revision_feedback)
        self.calls += 1
        state.draft_claims = list(state.draft_claims) + list(self.inject)  # 每轮都注入
        return state


gen2 = Inject2(cfg, inject2)
state2 = AgentState(user_query="变压器乙炔 25 ppm，可能是什么故障？", context={"dga": DGA}, max_iterations=3)
final2 = wf.run_diagnosis_workflow(state2, planner, retriever, gen2, validator)
t2 = [r["target"] for r in final2.route_log]
check(t2.count("supplement") == 0, f"无建议工具的声明不触发补证：{t2}")
check(any(r["reason"] == "revision:no_actionable_evidence" for r in final2.route_log), "路由记录 no_actionable_evidence 原因")
check(len(final2.tool_calls) == 1, "不产生额外工具调用")
check(final2.validation_verdict == "ABSTAIN", f"持续无据 → ABSTAIN（verdict={final2.validation_verdict}）")

# 每轮都注入可补声明：只补 1 轮，第二次走 generator（轮数耗尽）
class Inject3(Inject2):
    pass


inject3 = [make_claim(f"z{i}", f"局部放电与乙炔生成存在第 {i} 条图谱关系链路", "inference", []) for i in range(1, 8)]
gen3 = Inject3(cfg, inject3)
state3 = AgentState(user_query="变压器乙炔 25 ppm，可能是什么故障？", context={"dga": DGA}, max_iterations=3)
final3 = wf.run_diagnosis_workflow(state3, planner, retriever, gen3, validator)
t3 = [r["target"] for r in final3.route_log]
check(t3.count("supplement") == 1, f"持续注入可补声明时仅补证 1 轮：{t3}")
check(any("exhausted" in r["reason"] for r in final3.route_log), "第二次 REVISION 记录轮数耗尽原因")
sig3 = [(c["tool"], wf._canonical_args(c["tool"], c.get("args") or {})) for c in final3.tool_calls]
check(len(sig3) == len(set(sig3)), "持续注入场景全部调用仍无重复")
check(final3.iteration == 3 and final3.validation_verdict == "ABSTAIN", f"第三次评估仍不通过 → ABSTAIN（iteration={final3.iteration}, verdict={final3.validation_verdict}）")
check("证据不足" in (final3.final_answer or ""), "ABSTAIN 输出证据不足文案")

# 串行降级路径
state4 = AgentState(user_query="变压器乙炔 25 ppm，可能是什么故障？", context={"dga": DGA}, max_iterations=3)
gen4 = InjectGenerator(cfg, inject)
final4 = wf._run_sequential_workflow(state4, planner, retriever, gen4, validator)
check([r["target"] for r in final4.route_log].count("supplement") == 1 and final4.validation_verdict == "PASS",
      "串行降级工作流同样完成补证并 PASS")

# ─────────────────────────────────────────────────────────────
print("[7] Planner 渲染与模板")
ctx_txt = PlannerAgent._render_context({"dga": DGA, "missing_evidence": [
    {"claim_id": "c1", "suggested_tool": "kg_search", "suggested_query": "局部放电 导致"}]})
check("missing_evidence" in ctx_txt and "c1→kg_search" in ctx_txt, "_render_context 渲染 missing_evidence")
check("missing_evidence" not in PlannerAgent._render_context({"dga": DGA}), "无 missing_evidence 时不渲染")
tpl = (ROOT / "templates/planner/system_active.txt").read_text(encoding="utf-8")
check("【补证规则（missing_evidence）】" in tpl and "最多 1 轮" in tpl, "active 模板含补证规则")
check("不要重复调用本轮已经执行过且参数相同的工具" in tpl, "模板要求不重复同工具同参数")

# ─────────────────────────────────────────────────────────────
print("[8] off / check 模式行为不变")
for mode in ("off", "check"):
    cfg_m = _cfg(claim_check=mode)
    _, retriever_m, _, validator_m, _ = build_agents(cfg_m, mcp, spec)
    gen_m = InjectGenerator(cfg_m, inject)
    s_m = AgentState(user_query="变压器乙炔 25 ppm，可能是什么故障？", context={"dga": DGA}, max_iterations=3)
    f_m = wf.run_diagnosis_workflow(s_m, planner, retriever_m, gen_m, validator_m)
    check("supplement" not in [r["target"] for r in f_m.route_log] and f_m.evidence_rounds == 0,
          f"claim_check={mode} 不触发补证（targets={[r['target'] for r in f_m.route_log]}）")
    check(len(f_m.tool_calls) == 1, f"claim_check={mode} 仅首轮 1 次工具调用")
    check(not f_m.missing_evidence, f"claim_check={mode} 不残留 missing_evidence")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
