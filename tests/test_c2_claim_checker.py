"""
C-2 单元测试：声明级约束核查器（确定性层）。

运行：python3 tests/test_c2_claim_checker.py
验收：四类约束各 5 个构造违反样例检出率 100%；对应正例（合规声明）误报 0。
另测：判定规则（SAFETY/DATA → REVISION；unsupported_ratio > 0.3 → REVISION；iteration ≥ 3 → ABSTAIN）、
ValidationResult v2 字段与兼容、ValidatorAgent 集成（claim_check=off/check）、missing_evidence 建议。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.state import AgentState                                  # noqa: E402
from src.agents.claims import make_claim, make_evidence, build_rule_claims   # noqa: E402
from src.agents.claim_checker import ClaimChecker, ClaimCheckResult     # noqa: E402
from src.agents.validator import ValidatorAgent, ValidationResult       # noqa: E402

PASSED = FAILED = 0


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {msg}")
    else:
        FAILED += 1
        print(f"  FAIL {msg}")


NO_LLM_CFG = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 3, "claim_check": "check"},
}

DGA = {"H2": 80.0, "CH4": 20.0, "C2H2": 62.0, "C2H4": 6.0, "C2H6": 37.0}
ATTR = {
    "status": "ok", "primary_fault": "partial_discharge", "primary_fault_name": "局部放电", "primary_probability": 0.596,
    "fault_ranking": [
        {"fault_id": "partial_discharge", "fault_name": "局部放电", "probability": 0.596, "probability_pct": "59.6%", "rank": 1, "severity": "高风险"},
        {"fault_id": "winding_short_circuit", "fault_name": "绕组匝间短路", "probability": 0.391, "probability_pct": "39.1%", "rank": 2, "severity": "中等风险"},
        {"fault_id": "overload_overheating", "fault_name": "过载过热", "probability": 0.007, "probability_pct": "0.7%", "rank": 3, "severity": "低风险"},
    ],
    "dga_analysis": {"interpretation": "C₂H₂=62μL/L 超过注意值", "ratio_codes": {"code_C2H2_C2H4": 2, "code_CH4_H2": 0, "code_C2H4_C2H6": 0},
                     "matched_rules": [{"rule_name": "电弧放电", "confidence": 0.93}], "ratios_incomplete": False},
    "evidence_used": ["C2H2_elevated"], "evidence_negative": ["H2_elevated"],
    "uncertainty": {"entropy_bits": 1.0757, "top1_top2_gap": 0.205, "calibrated": True},
}
FC = {"status": "ok", "dataset": "ETTh1", "freq": "1h", "lookback": 96, "horizon": 24,
      "forecast": {"mean_predicted_ot": 31.8, "min_predicted_ot": 30.1, "max_predicted_ot": 33.2},
      "history_summary": {"ot_mean": 30.5, "ot_std": 1.1}, "anomalies": {"count": 2}}
NORMAL_ATTR = {"status": "ok", "primary_fault": "overload_overheating", "primary_fault_name": "过载过热", "primary_probability": 0.52,
               "fault_ranking": [{"fault_id": "overload_overheating", "fault_name": "过载过热", "probability": 0.52, "probability_pct": "52.0%", "rank": 1, "severity": "低风险"}],
               "dga_analysis": {"interpretation": "轻微过热", "matched_rules": []}, "uncertainty": {"entropy_bits": 1.5}}


def call(idx, tool, result, args):
    return {"tool": tool, "call_index": idx, "call_id": f"id{idx}", "raw_arguments": args, "args": args,
            "result": result, "success": True, "exec_success": True, "business_success": True, "inquiry_round": 0}


def base_state(attr=ATTR, with_fc=True) -> AgentState:
    st = AgentState(user_query="#2 主变 DGA 异常，请诊断。", context={"device_id": "#2"})
    st.tool_calls = [call(0, "fault_attribution", attr, {"dga_data": DGA, "device_id": "#2主变"})]
    if with_fc:
        st.tool_calls.append(call(1, "ett_forecast", FC, {"dataset": "ETTh1", "horizon": 24, "lookback": 96}))
    st.tool_calls.append(call(2, "kg_search", {"status": "ok", "paths": [{"path": "放电 → 产生 → C2H2", "confidence": 0.8, "support_count": 4, "evidence": "放电性故障主要产生乙炔"}]}, {"query": "放电"}))
    st.retrieved_knowledge = [{"chunk_id": "chunk_7f3a", "text": "乙炔含量超过 5 μL/L 时应缩短取样周期，并结合三比值法判断。",
                               "metadata": {"title": "DGA 导则"}}]
    st.iteration = 1
    return st


def ev_tool(ref, obj):
    return make_evidence("tool", ref, obj, tool="x")


def run_one(claim, st=None):
    st = st or base_state()
    st.draft_claims = [claim]
    return ClaimChecker(llm=None).check(st)


def violated(res: ClaimCheckResult, ctype: str) -> bool:
    return any(ctype in v.violated_constraints for v in res.claim_verdicts)


def clean(res: ClaimCheckResult) -> bool:
    return all(v.verdict == "pass" and not v.violated_constraints for v in res.claim_verdicts)


# ────────────────────────────────────────────────────────────
print("[DATA] 5 违反 + 5 合规")
data_bad = [
    make_claim("c1", "本次检测 C2H2=125 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0}})]),
    make_claim("c1", "主要故障局部放电，后验概率 39.1%；绕组匝间短路 59.6%。", "inference", [ev_tool("call:0", {"primary_probability": 0.596})]),
    make_claim("c1", "三比值法编码：1-0-2。", "observation", [ev_tool("call:0", {"ratio_codes": {"code_C2H2_C2H4": 2, "code_CH4_H2": 0, "code_C2H4_C2H6": 0}})]),
    make_claim("c1", "未来 48 步油温预测均值 35.0℃。", "inference", [ev_tool("call:1", {"horizon": 24, "forecast": {"mean_predicted_ot": 31.8}})]),
    make_claim("c1", "3σ 检出 7 个异常点，后验熵 2.50 bit。", "observation", [ev_tool("call:1", {"anomalies": {"count": 2}}), ev_tool("call:0", {"uncertainty": {"entropy_bits": 1.0757}})]),
]
data_good = [
    make_claim("c1", "本次检测 C2H2=62.0 ppm、H2=80 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0, "H2": 80.0}})]),
    make_claim("c1", "主要故障局部放电，后验概率 59.6%（Top-1 与 Top-2 差 20.5%）。", "inference", [ev_tool("call:0", {"primary_probability": 0.596, "top1_top2_gap": 0.205})]),
    make_claim("c1", "三比值法编码：2-0-0。", "observation", [ev_tool("call:0", {"ratio_codes": {"code_C2H2_C2H4": 2, "code_CH4_H2": 0, "code_C2H4_C2H6": 0}})]),
    make_claim("c1", "未来 24 步油温预测均值 31.8℃，范围 30.1℃~33.2℃。", "inference", [ev_tool("call:1", {"horizon": 24, "forecast": FC["forecast"]})]),
    make_claim("c1", "3σ 检出 2 个异常点；归因后验熵 1.08 bit。", "observation", [ev_tool("call:1", {"anomalies": {"count": 2}}), ev_tool("call:0", {"uncertainty": {"entropy_bits": 1.0757}})]),
]
for i, c in enumerate(data_bad, 1):
    r = run_one(c)
    check(violated(r, "DATA") and r.verdict == "REVISION", f"DATA 违反 #{i} 检出 → REVISION")
for i, c in enumerate(data_good, 1):
    r = run_one(c)
    check(clean(r) and r.verdict == "PASS", f"DATA 合规 #{i} 无误报")

# ────────────────────────────────────────────────────────────
print("[EVIDENCE] 5 违反 + 5 合规")
ev_bad = [
    make_claim("c1", "C2H2=62 ppm。", "observation", [ev_tool("call:9", {"dga_data": {"C2H2": 62.0}})]),
    make_claim("c1", "应缩短取样周期。", "recommendation", [make_evidence("kb", "kb:chunk_7f3a", "乙炔超过 5 ppm 应立即停电吊罩")]),
    make_claim("c1", "本次检测数据显示乙炔升高。", "observation", [make_evidence("kb", "kb:chunk_7f3a", "乙炔含量超过 5 μL/L 时应缩短取样周期")]),
    make_claim("c1", "过热产生乙烯。", "inference", [make_evidence("kg", "kg:过热 → 产生 → C2H4", "过热故障主要产生乙烯")]),
    make_claim("c1", "C2H2=62 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0, "CO": 300.0}})]),
]
ev_good = [
    make_claim("c1", "C2H2=62 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0}})]),
    make_claim("c1", "应缩短取样周期并结合三比值法。", "recommendation", [make_evidence("kb", "kb:chunk_7f3a", "乙炔含量超过 5 μL/L 时应缩短取样周期")]),
    make_claim("c1", "放电性故障主要产生乙炔。", "inference", [make_evidence("kg", "kg:放电 → 产生 → C2H2", "放电性故障主要产生乙炔")]),
    make_claim("c1", "用户提问设备为 #2 主变。", "observation", [make_evidence("user", "user:0", "#2 主变 DGA 异常")]),
    make_claim("c1", "参考导则片段（截断）。", "recommendation", [make_evidence("kb", "kb:chunk_7f3a", "乙炔含量超过 5 μL/L 时应缩短取样周期，并结合…")]),
]
for i, c in enumerate(ev_bad, 1):
    r = run_one(c)
    check(violated(r, "EVIDENCE"), f"EVIDENCE 违反 #{i} 检出")
for i, c in enumerate(ev_good, 1):
    r = run_one(c)
    check(clean(r), f"EVIDENCE 合规 #{i} 无误报")

# ────────────────────────────────────────────────────────────
print("[APPLICABILITY] 5 违反 + 5 合规")
ap_bad = [
    make_claim("c1", "基于 ETTm1 数据集预测 24 步。", "observation", [ev_tool("call:1", {"dataset": "ETTh1", "horizon": 24})]),
    make_claim("c1", "#1 主变 DGA 显示乙炔 62 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0}})]),
    make_claim("c1", "ETTh2 油温历史均值 30.5℃。", "observation", [ev_tool("call:1", {"history_summary": {"ot_mean": 30.5}})]),
    make_claim("c1", "3 号主变主要故障为局部放电（59.6%）。", "inference", [ev_tool("call:0", {"primary_probability": 0.596})]),
    make_claim("c1", "预测基于 ETTm2（15 分钟采样），未来 24 步均值 31.8℃。", "inference", [ev_tool("call:1", {"dataset": "ETTh1", "forecast": {"mean_predicted_ot": 31.8}})]),
]
ap_good = [
    make_claim("c1", "基于 ETTh1 数据集预测 24 步。", "observation", [ev_tool("call:1", {"dataset": "ETTh1", "horizon": 24})]),
    make_claim("c1", "#2 主变 DGA 显示 C2H2=62 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0}})]),
    make_claim("c1", "ETTh1 油温历史均值 30.5℃。", "observation", [ev_tool("call:1", {"history_summary": {"ot_mean": 30.5}})]),
    make_claim("c1", "2 号主变主要故障为局部放电（59.6%）。", "inference", [ev_tool("call:0", {"primary_probability": 0.596})]),
    make_claim("c1", "回看 96 点、预测 24 步。", "observation", [ev_tool("call:1", {"lookback": 96, "horizon": 24})]),
]
for i, c in enumerate(ap_bad, 1):
    r = run_one(c)
    check(violated(r, "APPLICABILITY"), f"APPLICABILITY 违反 #{i} 检出")
for i, c in enumerate(ap_good, 1):
    r = run_one(c)
    check(clean(r), f"APPLICABILITY 合规 #{i} 无误报")

# ────────────────────────────────────────────────────────────
print("[SAFETY] 5 违反 + 5 合规")
st_normal = base_state(attr=NORMAL_ATTR)
st_no_obs = base_state()
sa_bad = [
    (make_claim("c1", "建议立即停电吊罩检查。", "safety", [make_evidence("user", "user:0", "#2 主变 DGA 异常")]), st_no_obs),
    (make_claim("c1", "建议立即停电检修。", "safety", [ev_tool("call:0", {"primary_fault_name": "过载过热", "primary_probability": 0.52})]), st_normal),
    (make_claim("c1", "当前可继续运行，无需处理。", "recommendation", [ev_tool("call:0", {"primary_probability": 0.596})]), base_state()),
    (make_claim("c1", "建议更换套管。", "recommendation", [make_evidence("kb", "kb:chunk_7f3a", "乙炔含量超过 5 μL/L 时应缩短取样周期")]), st_no_obs),
    (make_claim("c1", "建议退出运行并返厂大修。", "safety", [make_evidence("kg", "kg:放电 → 产生 → C2H2", "放电性故障主要产生乙炔")]), st_no_obs),
]
sa_good = [
    (make_claim("c1", "鉴于局部放电概率 59.6% 且属高危，建议复测后申请停电检查。", "safety", [ev_tool("call:0", {"primary_fault_name": "局部放电", "primary_probability": 0.596, "severity": "高风险"}), ev_tool("call:0", {"dga_data": DGA})]), base_state()),
    (make_claim("c1", "建议缩短取样周期跟踪趋势。", "recommendation", [make_evidence("kb", "kb:chunk_7f3a", "乙炔含量超过 5 μL/L 时应缩短取样周期")]), base_state()),
    (make_claim("c1", "过载过热概率 52.0%，建议核查冷却系统与负载，暂不停电。", "recommendation", [ev_tool("call:0", {"primary_probability": 0.52})]), st_normal),
    (make_claim("c1", "停电检修前须执行工作票、验电、接地。", "safety", [ev_tool("call:0", {"primary_fault_name": "局部放电", "primary_probability": 0.596})]), base_state()),
    (make_claim("c1", "建议加强巡视并关注油温。", "recommendation", [ev_tool("call:1", {"anomalies": {"count": 2}})]), base_state()),
]
for i, (c, st) in enumerate(sa_bad, 1):
    r = run_one(c, st)
    check(violated(r, "SAFETY") and r.verdict == "REVISION", f"SAFETY 违反 #{i} 检出 → REVISION")
for i, (c, st) in enumerate(sa_good, 1):
    r = run_one(c, st)
    check(clean(r), f"SAFETY 合规 #{i} 无误报")

# ────────────────────────────────────────────────────────────
print("[规则声明] build_rule_claims 全量通过核查（自洽）")
st = base_state()
st.draft_claims = build_rule_claims(st)
r = ClaimChecker(llm=None).check(st)
check(r.verdict == "PASS" and clean(r) and r.n_claims >= 8, f"规则声明 {r.n_claims} 条全部 pass（{r.violation_counts}）")
if not clean(r):
    print(json.dumps([v.to_dict() for v in r.claim_verdicts if v.verdict != "pass"], ensure_ascii=False, indent=1))

# ────────────────────────────────────────────────────────────
print("[判定规则]")
st = base_state()
good = make_claim("g", "C2H2=62 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0}})])
unsup = make_claim("u", "可能存在铁芯多点接地。", "inference", [ev_tool("call:7", {"x": 1})])
st.draft_claims = [good, good | {"id": "g2"}, good | {"id": "g3"}, unsup]
r = ClaimChecker(llm=None).check(st)
check(abs(r.unsupported_ratio - 0.25) < 1e-9 and r.verdict == "PASS", "unsupported_ratio 0.25 ≤ 0.3 → PASS（仅 EVIDENCE 违反不单独触发）")
st.draft_claims = [good, unsup, unsup | {"id": "u2"}]
r = ClaimChecker(llm=None).check(st)
check(r.unsupported_ratio > 0.3 and r.verdict == "REVISION", "unsupported_ratio 0.67 > 0.3 → REVISION")
st.iteration = 3
r = ClaimChecker(llm=None).check(st)
check(r.verdict == "ABSTAIN", "iteration=3 仍不通过 → ABSTAIN")
st.iteration = 3
st.draft_claims = [good]
r = ClaimChecker(llm=None).check(st)
check(r.verdict == "PASS", "iteration=3 但通过 → PASS")
st.draft_claims = []
st.iteration = 1
r = ClaimChecker(llm=None).check(st)
check(r.verdict == "REVISION" and r.n_claims == 0, "无声明 → REVISION")
st.draft_claims = [good, unsup, unsup | {"id": "u2"}]
r = ClaimChecker(llm=None).check(st)
check(len(r.missing_evidence) == 2 and all(m["claim_id"] in ("u", "u2") for m in r.missing_evidence), "missing_evidence 覆盖无依据声明")
d = r.to_dict()
check(set(d) >= {"verdict", "claim_verdicts", "unsupported_ratio", "violation_counts", "missing_evidence"}, "ClaimCheckResult.to_dict 字段")

# ────────────────────────────────────────────────────────────
print("[ValidationResult v2 兼容]")
v1 = ValidationResult("PASS", 8, ["a"], [], [], "ok")
check(set(v1.to_dict()) == {"verdict", "score", "strengths", "weaknesses", "improvement_suggestions", "summary"}, "v1 构造 to_dict 无新增字段")
v2 = ValidationResult("REVISION", 5, [], ["[c1] DATA：x"], ["修正"], "s", claim_verdicts=[{"claim_id": "c1", "verdict": "violated", "violated_constraints": ["DATA"], "detail": ["x"]}],
                      unsupported_ratio=0.0, violation_counts={"DATA": 1}, claim_check_verdict="REVISION")
check("claim_verdicts" in v2.to_dict() and "【声明级核查】" in v2.to_report(), "v2 字段进入 to_dict / to_report")

# ────────────────────────────────────────────────────────────
print("[ValidatorAgent 集成]")
cfg_off = json.loads(json.dumps(NO_LLM_CFG)); cfg_off["workflow"]["claim_check"] = "off"
va_off = ValidatorAgent(cfg_off)
check(va_off.checker is None and va_off.claim_check == "off", "claim_check=off 不创建 checker")
va = ValidatorAgent(NO_LLM_CFG)
check(va.checker is not None and va.claim_check == "check", "claim_check=check 创建 checker")

st = base_state()
st.draft_claims = build_rule_claims(st)
st.draft_answer = "一、诊断结论：局部放电（可能）。二、处理建议：安全优先，建议复测 DGA。"
va.run(st)
check(st.validation_verdict == "PASS" and st.claim_check_log and st.claim_check_log[0]["verdict"] == "PASS", "合规草案 → PASS 且记录 claim_check_log")
check("【声明级核查】" in (st.validation_report or ""), "验证报告含声明级核查段")

st = base_state()
st.draft_claims = [make_claim("c1", "本次检测 C2H2=125 ppm。", "observation", [ev_tool("call:0", {"dga_data": {"C2H2": 62.0}})])]
st.draft_answer = "一、诊断结论：… 安全 建议 …"
va.run(st)
check(st.validation_verdict == "REVISION" and st.next_node == "generator" and "c1" in (st.revision_feedback or ""), "DATA 违反 → REVISION 并携带声明级反馈")

st = base_state()
st.iteration = 3
st.draft_claims = [make_claim("c1", "建议立即停电吊罩。", "safety", [make_evidence("user", "user:0", "#2 主变 DGA 异常")])]
st.draft_answer = "建议立即停电吊罩。安全。"
va.run(st)
check(st.validation_verdict == "ABSTAIN" and st.next_node == "END" and "证据不足" in (st.final_answer or ""), "第 3 次评估仍违反 → ABSTAIN 弃答")
check(st.missing_evidence == [], "claim_check=check 不写 missing_evidence（route 才写）")

cfg_route = json.loads(json.dumps(NO_LLM_CFG)); cfg_route["workflow"]["claim_check"] = "route"
va_r = ValidatorAgent(cfg_route)
st = base_state()
st.draft_claims = [make_claim("u", "可能存在铁芯多点接地。", "inference", [ev_tool("call:7", {"x": 1})])]
st.draft_answer = "可能存在铁芯多点接地。安全 建议。"
va_r.run(st)
check(st.missing_evidence and st.missing_evidence[0]["claim_id"] == "u", "claim_check=route 写入 missing_evidence")

st = base_state()
st.draft_claims = []
st.draft_answer = "一、诊断结论：可能为局部放电。二、处理建议：安全优先，确认保护装置正常后复测 DGA，并结合三比值法进一步判断，必要时申请停电试验。"
va_off.run(st)
check(st.validation_verdict == "PASS" and not st.claim_check_log, "off 模式行为与 v1 一致")

print(f"\n结果：{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
