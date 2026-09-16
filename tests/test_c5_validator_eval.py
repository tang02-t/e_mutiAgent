"""
C-5 验证器对照评测 自检测试（无 LLM、无 GPU）。

运行：python3 tests/test_c5_validator_eval.py

覆盖：
[1] ClaimChecker 判定规则修订：APPLICABILITY 硬约束、observation 伪造引用直接 REVISION、可配置关闭；Validator 读取配置
[2] SimGenerator：首轮输出注入声明 + extras；修订轮只替换被标记声明；extras 在工具补证后再落证，否则保留
[3] make_extras：只挑本轮未调用的工具；fault_attribution 需 context.dga；最多 k 条
[4] run_one：三组在同一条注入记录上的行为（v1 PASS 放行；v2 首轮标记目标声明并命中期望约束；v2_route 对缺证 extras 触发补证）
[5] summarize / acceptance：字段完整、比率区间合法、验收逻辑
[6] 报告产物：docs/validator_eval.md / .json / 两张图存在且 JSON 中 acceptance.pass 为 True
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
from src.graph.system_modes import tool_stack_spec, build_agents                      # noqa: E402
from src.agents.claims import make_claim, make_evidence                             # noqa: E402
from src.agents.claim_checker import ClaimChecker                                   # noqa: E402
from src.agents.validator import ValidatorAgent                                     # noqa: E402
from src.tools.fault_attribution import configure_engine                            # noqa: E402
from eval_system_modes import build_mcp                                             # noqa: E402
import eval_validator as ev                                                         # noqa: E402
import build_d11_fault_injection as d11                                             # noqa: E402

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


def ev_tool(ref: str, span: Dict[str, Any], tool: str = "ett_forecast") -> Dict[str, Any]:
    return make_evidence("tool", ref, span, tool=tool)


def base_state() -> AgentState:
    st = AgentState(user_query="ETTh1 数据预测未来 24 小时油温")
    st.tool_calls = [{
        "tool": "ett_forecast", "call_index": 0, "call_id": "c0", "success": True, "exec_success": True,
        "business_success": True, "args": {"dataset": "ETTh1", "lookback": 96, "horizon": 24},
        "raw_arguments": {"dataset": "ETTh1", "lookback": 96, "horizon": 24},
        "result": {"status": "ok", "dataset": "ETTh1", "freq": "H", "lookback": 96, "horizon": 24,
                   "history_summary": {"ot_mean": 30.5, "ot_std": 2.1},
                   "forecast": {"mean_predicted_ot": 31.2, "min_predicted_ot": 30.0, "max_predicted_ot": 32.4, "values": [31.2] * 24},
                   "anomalies": {"count": 0}},
    }]
    st.retrieved_knowledge = [{"chunk_id": "kb1", "text": "变压器油温持续升高时应检查冷却系统。", "metadata": {"title": "规程"}}]
    st.iteration = 1
    return st


def test_1_rules() -> None:
    print("[1] 判定规则修订")
    good = make_claim("g", "基于 ETTh1 数据集回看 96 点、预测 24 步。", "observation",
                      [ev_tool("call:0", {"dataset": "ETTh1", "lookback": 96, "horizon": 24})])
    bad_ap = make_claim("a", "基于 ETTm1 数据集预测 24 步。", "observation", [ev_tool("call:0", {"dataset": "ETTh1", "horizon": 24})])
    st = base_state()
    st.draft_claims = [good, good | {"id": "g2"}, good | {"id": "g3"}, good | {"id": "g4"}, bad_ap]
    r = ClaimChecker(llm=None).check(st)
    check(r.verdict == "REVISION" and any("APPLICABILITY" in x for x in r.reasons),
          f"单条 APPLICABILITY 违反（占比 0.2）→ REVISION（reasons={r.reasons}）")
    r2 = ClaimChecker(llm=None, hard_constraints=("SAFETY", "DATA")).check(st)
    check(r2.verdict == "PASS", "hard_constraints 不含 APPLICABILITY 时恢复旧行为 → PASS")

    fake_obs = make_claim("f", "历史油温均值 30.5℃。", "observation", [ev_tool("call:9", {"history_summary": {"ot_mean": 30.5}})])
    st.draft_claims = [good, good | {"id": "g2"}, good | {"id": "g3"}, good | {"id": "g4"}, fake_obs]
    r = ClaimChecker(llm=None).check(st)
    check(r.verdict == "REVISION" and any("observation" in x for x in r.reasons),
          f"observation 引用不存在的 call（占比 0.2）→ REVISION（reasons={r.reasons}）")
    r2 = ClaimChecker(llm=None, strict_observation=False).check(st)
    check(r2.verdict == "PASS", "strict_observation=False 时按占比阈值 → PASS")
    fake_inf = make_claim("f", "油温趋势平稳。", "inference", [ev_tool("call:9", {"x": 1})])
    st.draft_claims = [good, good | {"id": "g2"}, good | {"id": "g3"}, good | {"id": "g4"}, fake_inf]
    r = ClaimChecker(llm=None).check(st)
    check(r.verdict == "PASS", "inference 引用不存在的 call 且占比 0.2 → 仍按阈值 PASS（已知权衡）")

    cfg = json.loads(json.dumps(ev.NO_LLM_CFG))
    cfg["workflow"]["claim_check"] = "check"
    cfg["workflow"]["claim_hard_constraints"] = ["SAFETY"]
    cfg["workflow"]["claim_strict_observation"] = False
    v = ValidatorAgent(cfg)
    check(v.checker.hard_constraints == ("SAFETY",) and v.checker.strict_observation is False, "Validator 读取 claim_hard_constraints / claim_strict_observation")
    v2 = ValidatorAgent(json.loads(json.dumps(ev.NO_LLM_CFG)) | {"workflow": {**ev.NO_LLM_CFG["workflow"], "claim_check": "check"}})
    check(v2.checker.hard_constraints == ("SAFETY", "DATA", "APPLICABILITY") and v2.checker.strict_observation, "默认硬约束含 APPLICABILITY 且 strict_observation=True")


def test_2_sim_generator() -> None:
    print("[2] SimGenerator")
    st = base_state()
    st.iteration = 0
    inj = [make_claim("c1", "基于 ETTm1 数据集预测。", "observation", [ev_tool("call:0", {"dataset": "ETTh1"})])]
    extra = make_claim("x_1", "主故障的产生机理与关联部位可由图谱关系链路支撑", "inference", [])
    extra["_need_tool"] = "kg_search"
    gen = ev.SimGenerator(inj, "正文", [extra])
    st = gen.run(st)
    check(len(st.draft_claims) == 2 and all("_need_tool" not in c for c in st.draft_claims), "首轮输出注入声明 + extras，内部字段不外泄")
    check(st.iteration == 1 and d11.CLAIMS_HEAD in st.draft_answer, "iteration+1，draft_answer 含声明清单")
    st.reasoning_trace.append({"agent": "validator", "type": "evaluation", "content": {"claim_verdicts": [
        {"claim_id": "c1", "verdict": "violated", "violated_constraints": ["APPLICABILITY"]},
        {"claim_id": "x_1", "verdict": "unsupported", "violated_constraints": ["EVIDENCE"]}]}})
    st = gen.run(st)
    ids = {c["id"]: c for c in st.draft_claims}
    check("c1" in ids and "ETTh1" in ids["c1"]["text"] and gen.n_fixed == 1, "被标记的 c1 用同 id 规则声明替换")
    check("x_1" in ids and not ids["x_1"]["evidence"] and gen.n_kept == 1, "kg_search 未调用 → extras 保留无据")
    st.tool_calls.append({
        "tool": "kg_search", "call_index": 1, "call_id": "c1", "success": True, "exec_success": True, "business_success": True,
        "args": {"query": "过热"}, "raw_arguments": {"query": "过热"},
        "result": {"status": "ok", "matched_entities": ["Fault:过热故障"],
                   "paths": [{"path": "Fault:过热故障 → LOCATED_IN → Component:绕组", "support_count": 2, "confidence": 0.6, "evidence": "文献"}]}})
    st.reasoning_trace.append({"agent": "validator", "type": "evaluation", "content": {"claim_verdicts": [
        {"claim_id": "c1", "verdict": "pass", "violated_constraints": []},
        {"claim_id": "x_1", "verdict": "unsupported", "violated_constraints": ["EVIDENCE"]}]}})
    st = gen.run(st)
    ids = {c["id"]: c for c in st.draft_claims}
    check("x_1" in ids and ids["x_1"]["evidence"] and ids["x_1"]["evidence"][0]["source"] == "kg" and gen.n_regrounded == 1,
          "kg_search 补证后 extras 用 kg 规则声明再落证（保留 id）")
    st.reasoning_trace.append({"agent": "validator", "type": "evaluation", "content": {"claim_verdicts": [
        {"claim_id": "c1", "verdict": "pass", "violated_constraints": []},
        {"claim_id": "x_1", "verdict": "pass", "violated_constraints": []}]}})
    n0 = len(st.draft_claims)
    st = gen.run(st)
    check(len(st.draft_claims) == n0, "无标记时声明不变")
    st.reasoning_trace.append({"agent": "validator", "type": "evaluation", "content": {"verdict": "PASS"}})
    check(ev.SimGenerator._last_flagged(st) is None, "v1（无 claim_verdicts）→ 无逐条反馈")


def test_3_extras() -> None:
    print("[3] make_extras")
    recs = ev.load_d11()
    n_ok = 0
    for r in recs[:80]:
        ex = ev.make_extras(r)
        called = {c.get("tool") for c in r["snapshot"]["tool_calls"] if c.get("success")}
        has_dga = isinstance((r["snapshot"].get("context") or {}).get("dga"), dict)
        ok = len(ex) <= 2 and all(e["_need_tool"] not in called for e in ex) and all(
            e["_need_tool"] != "fault_attribution" or has_dga for e in ex) and all(not e["evidence"] for e in ex)
        n_ok += ok
    check(n_ok == 80, f"80 条记录 extras 规则全部满足（{n_ok}/80）")
    check(all(len(ev.make_extras(r)) == 2 for r in recs[:80]), "每条恰有 2 条 extras（工具组合最多 2 个，候选 4 个）")


def test_4_run_one() -> None:
    print("[4] run_one 三组行为")
    recs = ev.load_d11()
    inj = next(r for r in recs if r["injected"] and r["type"] == "numeric_tamper")
    ap = next(r for r in recs if r["injected"] and r["subtype"] == "dataset_swap")
    clean = next(r for r in recs if not r["injected"])
    configure_engine("calibrated")
    spec = tool_stack_spec(ev.NO_LLM_CFG)
    mcp, _ = build_mcp(spec)
    oracle = ClaimChecker()
    res: Dict[str, Dict[str, Any]] = {}
    for arm in ev.ARMS:
        cfg = json.loads(json.dumps(ev.NO_LLM_CFG))
        cfg["workflow"]["claim_check"] = ev.ARM_CFG[arm]
        _, retriever, _, _, _ = build_agents(cfg, mcp, spec)
        validator = ValidatorAgent(cfg, claim_check=ev.ARM_CFG[arm])
        res[arm] = {"inj": ev.run_one(inj, arm, retriever, validator, oracle),
                    "ap": ev.run_one(ap, arm, retriever, validator, oracle),
                    "clean": ev.run_one(clean, arm, retriever, validator, oracle),
                    "e2": ev.run_one(inj, arm, retriever, validator, oracle, extras=True)}
    v1 = res["v1"]
    check(v1["inj"]["first_verdict"] == "PASS" and v1["inj"]["survived"] and not v1["inj"]["target_flagged"], "v1：注入样本首轮 PASS、错误残留、无逐条标记")
    check(v1["inj"]["oracle_violated"], "v1：独立复核确认最终答案仍含约束违反")
    for arm in ("v2_check", "v2_route"):
        r = res[arm]
        check(r["inj"]["first_verdict"] == "REVISION" and r["inj"]["target_flagged"] and r["inj"]["hit_expected"],
              f"{arm}：numeric_tamper 首轮 REVISION、目标声明标记并命中 DATA")
        check(r["inj"]["final_verdict"] == "PASS" and r["inj"]["survived"] is False and not r["inj"]["oracle_violated"],
              f"{arm}：修订后 PASS 且注入声明已被替换")
        check(r["ap"]["first_verdict"] == "REVISION" and "APPLICABILITY" in r["ap"]["target_constraints"],
              f"{arm}：dataset_swap 首轮 REVISION（APPLICABILITY 硬约束）")
        check(r["clean"]["first_verdict"] == "PASS" and r["clean"]["iterations"] == 1, f"{arm}：干净样本首轮 PASS")
    check(res["v2_check"]["e2"]["extra_tool_calls"] == 0 and res["v2_check"]["e2"]["n_extras"] == 2, "v2_check E2：不补证、extras=2")
    r2 = res["v2_route"]["e2"]
    check(r2["evidence_rounds"] >= 1 and r2["extra_tool_calls"] >= 1 and "supplement" in r2["route_targets"], "v2_route E2：触发补证并产生额外调用")
    check(r2["oracle_unsupported_ratio"] <= res["v2_check"]["e2"]["oracle_unsupported_ratio"], "v2_route E2：无依据占比不高于 v2_check")
    check(all(r[k]["error"] is None for r in res.values() for k in r), "无运行异常")
    check(all(r[k]["tokens"] == 0 and r[k]["llm_calls"] == 0 for r in res.values() for k in r), "全部零 Token / 零 LLM 调用")


def test_5_summary() -> None:
    print("[5] summarize / acceptance")
    rows: List[Dict[str, Any]] = []
    for i in range(6):
        rows.append({"eval_id": f"D11-{i:04d}", "arm": "v2_check", "injected": i < 4, "type": d11.TYPES[i % 4] if i < 4 else None,
                     "subtype": f"s{i % 4}" if i < 4 else None, "scenario": "x", "error": None, "elapsed": 0.001,
                     "first_verdict": "REVISION" if i < 3 else "PASS", "first_claim_verdict": None,
                     "target_flagged": i < 3, "target_constraints": ["DATA"] if i < 3 else [], "hit_expected": (i < 3) if i < 4 else None,
                     "final_verdict": "PASS", "abstained": False, "iterations": 2 if i < 3 else 1, "survived": (i == 3) if i < 4 else None,
                     "oracle_unsupported_ratio": 0.0, "oracle_violated": i == 3, "oracle_violations": {}, "n_final_claims": 5,
                     "n_fixed": 1 if i < 3 else 0, "n_dropped": 0, "n_regrounded": 0, "n_kept": 0, "n_extras": 0,
                     "extras_kept_unsupported": 0, "oracle_n_unsupported": 0, "extra_tool_calls": 0, "evidence_rounds": 0,
                     "route_targets": ["END"], "tokens": 0, "llm_calls": 0})
    s = ev.summarize(rows)
    check(abs(s["error_pass_rate"] - 0.25) < 1e-9 and abs(s["detection_rate"] - 0.75) < 1e-9, "错误通过率 / 检出率计算")
    check(s["clean_fp_rate"] == 0.0 and abs(s["survival_rate"] - 0.25) < 1e-9, "干净误报率 / 残留率计算")
    check(set(s["by_type"]) == set(d11.TYPES) and all(0 <= v <= 1 for k, v in s.items() if k.endswith("_rate")), "by_type 齐全、比率在 [0,1]")
    v1 = dict(s, error_pass_rate=1.0, clean_fp_rate=0.0, unsupported_rate=0.3, n_errors=0)
    c = dict(s, error_pass_rate=0.1, clean_fp_rate=0.05, unsupported_rate=0.2, extras_unsupported_rate=1.0, n_errors=0)
    r = dict(c, unsupported_rate=0.1, extras_unsupported_rate=0.3)
    acc = ev.acceptance({"v1": v1, "v2_check": c, "v2_route": r}, {"v1": v1, "v2_check": c, "v2_route": r})
    check(acc["pass"] is True, "验收逻辑：通过用例")
    acc2 = ev.acceptance({"v1": v1, "v2_check": dict(c, clean_fp_rate=0.2), "v2_route": r}, {"v1": v1, "v2_check": c, "v2_route": r})
    check(acc2["pass"] is False and acc2["v2_check_clean_fp_ok"] is False, "验收逻辑：干净误报 > 10% 不通过")
    acc3 = ev.acceptance({"v1": v1, "v2_check": c, "v2_route": r}, {"v1": v1, "v2_check": c, "v2_route": dict(r, unsupported_rate=0.25)})
    check(acc3["pass"] is False, "验收逻辑：E2 无依据率未下降不通过")


def test_6_artifacts() -> None:
    print("[6] 报告产物")
    md = ROOT / "docs/validator_eval.md"
    js = ROOT / "docs/validator_eval.json"
    check(md.exists() and "## 验收判定" in md.read_text(encoding="utf-8"), "validator_eval.md 存在且含验收判定")
    d = json.loads(js.read_text(encoding="utf-8"))
    check(d["acceptance"]["pass"] is True, "validator_eval.json acceptance.pass = True")
    check(set(d["summaries"]) == {"e1", "e2"} and all(set(d["summaries"][k]) == set(ev.ARMS) for k in d["summaries"]), "两设定 × 三组汇总齐全")
    s1 = d["summaries"]["e1"]
    check(s1["v1"]["n"] == 300 and s1["v2_check"]["clean_fp_rate"] <= 0.10 and s1["v2_check"]["error_pass_rate"] <= 0.5 * s1["v1"]["error_pass_rate"],
          f"E1 全量 300 条：v2_check 误报 {s1['v2_check']['clean_fp_rate']:.3f}、错误通过率 {s1['v2_check']['error_pass_rate']:.3f}")
    s2 = d["summaries"]["e2"]
    check(s2["v2_route"]["unsupported_rate"] < s2["v2_check"]["unsupported_rate"], "E2：v2_route 无依据结论率低于 v2_check")
    for f in ("fig_c5_detection_by_type.png", "fig_c5_pass_vs_cost.png"):
        check((ROOT / "docs/figures" / f).exists(), f"图表 {f} 存在")


def main() -> int:
    test_1_rules()
    test_2_sim_generator()
    test_3_extras()
    test_4_run_one()
    test_5_summary()
    test_6_artifacts()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
