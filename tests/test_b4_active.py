"""
B-4 主动追问循环回归测试（无 LLM，Planner decision=eig）。

覆盖：
  - AgentState 新字段默认值
  - Planner active：eig_decide 首轮 / 有 uncertainty 时的 ask / conclude；_validate_action 各错误码
  - 工作流：在 D12 抽 30 条（每档 10 条）用 PartialObsEnv 代替用户，完整走通
    planner → retriever → planner → inquiry → … → generator → validator，
    轨迹 inquiry_log 每轮含 entropy_bits / recommendations / action；追问轮数 ≤ K；
    交互式（answer_fn=None）遇 ask 中断并留下 pending_questions，resume_after_answer 可续跑
  - attribution_mode 开关：configure_engine("expert") / ("calibrated")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
os.environ["FAULT_ATTR_PARAMS"] = str(ROOT / "data/real/dga/learned_params.json")

from src.graph.state import AgentState  # noqa: E402
from src.agents.planner import PlannerAgent  # noqa: E402
from src.agents.retriever import RetrieverAgent  # noqa: E402
from src.agents.generator import GeneratorAgent  # noqa: E402
from src.agents.validator import ValidatorAgent  # noqa: E402
from src.graph.workflow import run_diagnosis_workflow, resume_after_answer  # noqa: E402
from src.tools.mcp_client import MCPClient  # noqa: E402
from src.tools.fault_attribution import fault_attribution, configure_engine, get_engine  # noqa: E402
import partial_obs_sim as sim  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name} {detail}")


NO_LLM_CFG = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 1, "planner_strategy": "active", "planner_decision": "eig",
                 "max_inquiry_rounds": 3, "attribution_mode": "calibrated"},
}


def make_mcp() -> MCPClient:
    m = MCPClient()
    m.register_tool("fault_attribution", fault_attribution)
    m.register_tool("rag_search", lambda query, **_: [])
    m.register_tool("kg_search", lambda **_: {"status": "no_match", "paths": []})
    m.register_tool("timeseries_anomaly", lambda signal, **_: {"status": "ok", "mean": 0, "std": 0, "anomaly_indices": []})
    m.register_tool("ett_forecast", lambda **_: {"status": "ok"})
    return m


def build_agents(cfg=NO_LLM_CFG, allowed=None, **planner_kw):
    planner = PlannerAgent(cfg, allowed_tools=allowed, **planner_kw)
    retriever = RetrieverAgent(cfg, make_mcp(), allowed_tools=allowed)
    return planner, retriever, GeneratorAgent(cfg), ValidatorAgent(cfg)


def run_sample(rec: dict, K: int = 3, answer_fn=None):
    env = sim.PartialObsEnv(rec)
    state = AgentState(user_query="这台变压器油色谱有部分数据缺失，请判断故障类型。",
                       context={"dga": env.context(), "device_id": rec["sim_id"]},
                       max_iterations=1, max_inquiry_rounds=K)
    planner, retriever, generator, validator = build_agents()
    fn = answer_fn if answer_fn is not None else env.answer
    final = run_diagnosis_workflow(state, planner, retriever, generator, validator, answer_fn=fn)
    return final, env


def main() -> int:
    print("[1] AgentState 新字段")
    s = AgentState(user_query="q")
    check("默认 planner_strategy=free", s.planner_strategy == "free")
    check("pending_questions / asked_symptoms / user_answers 为空", s.pending_questions == [] and s.asked_symptoms == [] and s.user_answers == {})
    check("inquiry_rounds=0, max_inquiry_rounds=3", s.inquiry_rounds == 0 and s.max_inquiry_rounds == 3)

    print("[2] Planner active 构造与提示词变体")
    p = PlannerAgent(NO_LLM_CFG)
    check("从 config 读取 strategy=active decision=eig", p.strategy == "active" and p.decision == "eig")
    try:
        PlannerAgent(NO_LLM_CFG, strategy="bogus")
        check("非法 strategy 抛错", False)
    except ValueError:
        check("非法 strategy 抛错", True)
    from src.utils.prompts import PLANNER_SYSTEM_PROMPT
    base = PLANNER_SYSTEM_PROMPT()
    act = PLANNER_SYSTEM_PROMPT(variant="active")
    check("active 变体包含动作规则与 uncertainty 说明", "call_tool | ask | conclude" in act and "entropy_bits" in act and "rationale" in act)
    check("基础模板不含动作字段", '"action"' not in base)
    check("active 变体已替换工具清单", "{{tools}}" not in act and "fault_attribution" in act)

    print("[3] eig_decide 与 _validate_action")
    d0 = p.eig_decide({"dga": {"H2": 100, "CH4": None}}, "q")
    check("首轮有 DGA → call_tool fault_attribution", d0["action"] == "call_tool" and d0["steps"][0]["tool"] == "fault_attribution")
    check("首轮 dga_data 不含 None 气体", "CH4" not in d0["steps"][0]["arguments"]["dga_data"])
    check("首轮步骤通过参数校验", d0["steps"][0]["validation"]["valid"], str(d0["steps"][0]["validation"]))
    d1 = p.eig_decide({}, "q")
    check("首轮无数据 → conclude", d1["action"] == "conclude")
    unc = {"entropy_bits": 1.5, "suggested_action": "ask", "recommendations": [
        {"symptom": "oil_temp_elevated", "eig": 0.3, "cost": 1, "voi": 0.25, "how_to_obtain": "call_tool:ett_forecast"},
        {"symptom": "C2H2_elevated", "eig": 0.2, "cost": 0, "voi": 0.2, "how_to_obtain": "ask_user", "ask_hint": "请问乙炔浓度？"},
        {"symptom": "load_elevated", "eig": 0.1, "cost": 1, "voi": 0.05, "how_to_obtain": "ask_user"},
    ]}
    d2 = p.eig_decide({"uncertainty": unc}, "q")
    check("首推 call_tool:ett_forecast → call_tool", d2["action"] == "call_tool" and d2["steps"][0]["tool"] == "ett_forecast", str(d2))
    p_num = PlannerAgent(NO_LLM_CFG, allowed_tools=["fault_attribution"])
    d3 = p_num.eig_decide({"uncertainty": unc}, "q")
    check("工具不在名单 → 跳到下一个 ask 征兆", d3["action"] == "ask" and d3["symptom"] == "C2H2_elevated" and d3["question"] == "请问乙炔浓度？", str(d3))
    d4 = p_num.eig_decide({"uncertainty": unc, "asked_symptoms": ["C2H2_elevated"]}, "q")
    check("已问过的征兆跳过", d4["action"] == "ask" and d4["symptom"] == "load_elevated")
    d5 = p_num.eig_decide({"uncertainty": unc, "asked_symptoms": ["C2H2_elevated"], "unavailable_symptoms": ["load_elevated"]}, "q")
    check("全部不可用 → conclude", d5["action"] == "conclude")
    d6 = p.eig_decide({"uncertainty": {"suggested_action": "conclude", "stop_reason": "entropy_below_tau", "entropy_bits": 0.3}}, "q")
    check("suggested_action=conclude → conclude", d6["action"] == "conclude" and "entropy_below_tau" in d6["rationale"])

    pj = {"action": "ask", "steps": []}
    p._validate_action(pj, {"uncertainty": unc})
    check("ask 缺 symptom → missing_symptom（致命）", not pj["action_validation"]["valid"] and any(e["code"] == "missing_symptom" for e in pj["action_validation"]["errors"]))
    pj = {"action": "ask", "symptom": "vibration_elevated", "question": "x", "steps": []}
    p._validate_action(pj, {"uncertainty": unc})
    check("symptom 不在推荐列表 → symptom_not_recommended", any(e["code"] == "symptom_not_recommended" for e in pj["action_validation"]["errors"]))
    pj = {"action": "ask", "symptom": "C2H2_elevated", "steps": []}
    p._validate_action(pj, {"uncertainty": unc, "asked_symptoms": ["C2H2_elevated"]})
    codes = {e["code"] for e in pj["action_validation"]["errors"]}
    check("重复追问 + 缺 question → symptom_repeated, missing_question", {"symptom_repeated", "missing_question"} <= codes, str(codes))
    pj = {"steps": [{"id": 1, "tool": "fault_attribution", "arguments": {}}]}
    p._validate_action(pj, {})
    check("缺 action 按 steps 推断 call_tool", pj["action"] == "call_tool" and pj["action_validation"]["valid"])
    pj = {"action": "call_tool", "steps": []}
    p._validate_action(pj, {})
    check("call_tool 无 steps → 致命", not pj["action_validation"]["valid"])

    print("[4] attribution_mode 开关")
    e1 = configure_engine("expert")
    check("expert 引擎未校准", e1.params_source == "expert_default" and not e1.is_calibrated)
    e2 = configure_engine("calibrated")
    check("calibrated 引擎已校准 T>1", e2.is_calibrated and e2.temperature > 1.0 and get_engine() is e2)
    try:
        configure_engine("bogus")
        check("非法 attribution_mode 抛错", False)
    except ValueError:
        check("非法 attribution_mode 抛错", True)

    print("[5] D12 抽 30 条走通追问循环（eig 决策，模拟器回答）")
    rows = []
    for level in ("light", "medium", "heavy"):
        rows += sim.load_d12(level=level, limit=10)
    check("抽样 30 条", len(rows) == 30)
    n_ok = n_asked = n_hit = 0
    max_rounds_seen = 0
    total_rounds = 0
    log_ok = True
    stop_reasons: dict[str, int] = {}
    for rec in rows:
        final, env = run_sample(rec, K=3)
        asked = [e for e in final.inquiry_log if e["action"] == "ask"]
        calls = [e for e in final.inquiry_log if e["action"] == "call_tool"]
        concl = [e for e in final.inquiry_log if e["action"] == "conclude"]
        ok = (final.final_answer is not None and final.draft_answer is not None
              and len(calls) >= 1 and len(concl) == 1 and final.inquiry_rounds == len(asked) <= 3
              and final.inquiry_rounds == env.n_queries)
        if not ok:
            print(f"    ! {rec['sim_id']} answer={final.final_answer is not None} calls={len(calls)} asked={len(asked)} concl={len(concl)} rounds={final.inquiry_rounds} env={env.n_queries}")
        n_ok += ok
        n_asked += bool(asked)
        total_rounds += len(asked)
        max_rounds_seen = max(max_rounds_seen, len(asked))
        for e in calls:
            if e.get("entropy_bits") is None or not isinstance(e.get("recommendations"), list):
                log_ok = False
        seen: set = set()
        for e in asked:
            if e.get("symptom") is None or e.get("entropy_bits") is None or not e["recommendations"]:
                log_ok = False
                continue
            # eig_greedy 必须问「尚未问过 / 非不可获取 / 可由用户回答」中 VoI 最高的
            cand = [r["symptom"] for r in e["recommendations"]
                    if r["symptom"] not in seen and str(r.get("how_to_obtain", "ask_user")).startswith("ask_user")]
            if cand and e["symptom"] != cand[0]:
                log_ok = False
                print(f"    ! {rec['sim_id']} 问了 {e['symptom']} 但首推 {cand[0]}")
            seen.add(e["symptom"])
        r = concl[0].get("rationale", "") if concl else ""
        key = next((k for k in ("entropy_below_tau", "voi_below_eps", "max_rounds", "no_candidates", "达最大追问") if k in r), r[:20])
        stop_reasons[key] = stop_reasons.get(key, 0) + 1
        attr = next((c["result"] for c in reversed(final.tool_calls) if c["tool"] == "fault_attribution" and c["success"]), None)
        if attr and attr.get("primary_fault") == rec["fault_label"]:
            n_hit += 1
    check("30 条全部走通（有归因、有结论、轮数一致）", n_ok == 30, f"{n_ok}/30")
    check("追问轮数 ≤ K=3", max_rounds_seen <= 3, str(max_rounds_seen))
    check("至少一半样例发生了追问", n_asked >= 15, f"{n_asked}/30")
    check("轨迹每轮记录 entropy_bits / recommendations，ask 命中 VoI 最高征兆", log_ok)
    print(f"    平均追问 {total_rounds / 30:.2f} 轮，Top-1 命中 {n_hit}/30，停止原因 {stop_reasons}")

    print("[5b] mode3（active_plan：全工具 + calibrated + EIG + active 策略）名单下走通")
    from src.graph.system_modes import resolve_mode
    spec3 = resolve_mode("mode3")
    check("mode3 定义为主动规划级", spec3.planner_strategy == "active" and spec3.with_eig
          and spec3.attribution_mode == "calibrated" and "fault_attribution" in spec3.allowed_tools)
    n3 = 0
    for rec in rows[:6]:
        env = sim.PartialObsEnv(rec)
        state = AgentState(user_query="请判断故障类型", context={"dga": env.context()}, max_iterations=1, max_inquiry_rounds=3)
        planner, retriever, generator, validator = build_agents(allowed=list(spec3.allowed_tools))
        final = run_diagnosis_workflow(state, planner, retriever, generator, validator, answer_fn=env.answer)
        disabled = [c for c in final.tool_calls if c.get("error_code") == "tool_disabled"]
        n3 += final.final_answer is not None and not disabled and final.inquiry_rounds <= 3
    check("mode3 名单 6 条走通且无越权调用", n3 == 6, f"{n3}/6")

    print("[6] 追问后后验熵下降、证据写回 context")
    rec = next(r for r in rows if r["level"] == "heavy")
    final, env = run_sample(rec, K=3)
    ents = [e["entropy_bits"] for e in final.inquiry_log if e["action"] == "call_tool"]
    check("多次归因（≥2 次 call_tool）", len(ents) >= 2, str(ents))
    check("最终熵 ≤ 首轮熵", ents[-1] <= ents[0] + 1e-9, str(ents))
    ev = final.context.get("evidence") or {}
    check("追问回答写入 context.evidence", all(final.user_answers[s] is None or ev.get(s) == final.user_answers[s] for s in final.asked_symptoms))
    check("asked_symptoms 写回 context", final.context.get("asked_symptoms") == final.asked_symptoms)
    check("tool_calls 累积且 call_index 连续", [c["call_index"] for c in final.tool_calls] == list(range(len(final.tool_calls))))
    check("inquiry_cost 累计 = 模拟器成本", abs(final.inquiry_cost - env.total_cost) < 1e-9, f"{final.inquiry_cost} vs {env.total_cost}")
    check("最终答案含不确定性描述", "后验熵" in (final.draft_answer or ""))

    print("[7] 不可获取征兆：回答 None 记入 unavailable，不重复追问")
    final, env = run_sample(rec, K=3, answer_fn=lambda sym, meta: None)
    asked = [e["symptom"] for e in final.inquiry_log if e["action"] == "ask"]
    check("每个征兆最多问一次", len(asked) == len(set(asked)))
    check("None 回答进入 unavailable_symptoms", set(asked) <= set(final.context.get("unavailable_symptoms") or []))
    check("仍能给出结论", final.final_answer is not None)

    print("[8] 交互式中断与续跑")
    env = sim.PartialObsEnv(rec)
    state = AgentState(user_query="请诊断", context={"dga": env.context()}, max_iterations=1, max_inquiry_rounds=3)
    planner, retriever, generator, validator = build_agents()
    mid = run_diagnosis_workflow(state, planner, retriever, generator, validator, answer_fn=None)
    check("answer_fn=None 遇 ask 中断，留下 pending_questions", bool(mid.pending_questions) and mid.final_answer is None, str(mid.pending_questions)[:80])
    check("pending 问题含 question 与 rationale", mid.pending_questions and mid.pending_questions[0].get("question") and mid.pending_questions[0].get("rationale"))
    sym = mid.pending_questions[0]["symptom"]
    done = resume_after_answer(mid, sym, env.answer(sym), planner, retriever, generator, validator)
    n_resume = 1
    while done.final_answer is None and done.pending_questions and n_resume < 5:
        s2 = done.pending_questions[0]["symptom"]
        done = resume_after_answer(done, s2, env.answer(s2), planner, retriever, generator, validator)
        n_resume += 1
    check("续跑（可多次）后得到最终答案", done.final_answer is not None and done.inquiry_rounds == n_resume, f"resumes={n_resume} rounds={done.inquiry_rounds}")
    check("续跑轮数 ≤ K", done.inquiry_rounds <= 3)
    check("续跑后 user_answers 含该征兆", sym in done.user_answers)

    print("[9] free 策略行为不变")
    cfg_free = {**NO_LLM_CFG, "workflow": {**NO_LLM_CFG["workflow"], "planner_strategy": "free"}}
    pf = PlannerAgent(cfg_free)
    st = AgentState(user_query="q", context={"dga": {"H2": 100}})
    st = pf.run(st)
    check("free + 无 LLM → llm_disabled 空计划", st.plan_status == "llm_disabled" and st.planner_action == "")

    print("[10] 前端演示案例 6 / 7（信息不足 → 追问 → 确诊）离线走通")
    demo = {
        "案例6": {"H2": 180.0, "CH4": 95.0, "C2H2": None, "C2H4": 60.0, "C2H6": None},
        "案例7": {"H2": 420.0, "CH4": None, "C2H2": 38.0, "C2H4": None, "C2H6": None},
    }
    for name, d in demo.items():
        state = AgentState(user_query="请判断故障类型，如需补充信息请问我", context={"dga": d},
                           max_iterations=1, max_inquiry_rounds=3)
        planner, retriever, generator, validator = build_agents()
        mid = run_diagnosis_workflow(state, planner, retriever, generator, validator, answer_fn=None)
        check(f"{name} 首轮即进入追问", bool(mid.pending_questions) and mid.pending_questions[0]["symptom"] in {"C2H2_elevated", "CH4_elevated", "C2H4_elevated", "C2H6_elevated", "TDCG_elevated", "gas_rate_rapid"}, str(mid.pending_questions)[:100])
        done, n = mid, 0
        while done.final_answer is None and done.pending_questions and n < 5:
            s2 = done.pending_questions[0]["symptom"]
            done = resume_after_answer(done, s2, True if s2 == "C2H2_elevated" else False, planner, retriever, generator, validator)
            n += 1
        check(f"{name} 回答后得到结论且熵下降", done.final_answer is not None and
              [e["entropy_bits"] for e in done.inquiry_log if e["action"] == "call_tool"][-1]
              < [e["entropy_bits"] for e in done.inquiry_log if e["action"] == "call_tool"][0])

    print(f"结果：{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
