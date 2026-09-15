"""
C-1 单元测试：声明与证据输出规范。

运行：python3 tests/test_c1_claims.py
覆盖：
[1] validate_claims：合法 / 各类非法（缺 evidence、type 非法、ref 前缀不匹配、span 为空、id 重复、未知 ref 记 warning）
[2] evidence_catalog：tool / kb / kg / user 四类 ref 枚举与 render
[3] build_rule_claims：fault_attribution 结果 → observation / inference / safety；高危才产 safety 且双引用；
    inquiry_log → user:n；无证据 → user:0 信息不足声明；timeseries / forecast / kg / kb 各自产出
[4] GeneratorAgent：text 模式不产 claims；claims 模式模板回退产 claims 且草案末尾附清单；
    _split_claims_output 解析（带分隔符 / 无分隔符 / 非法 JSON / 代码块包裹）
[5] GENERATOR_SYSTEM_PROMPT("claims") 加载 system_claims.txt
[6] 端到端：eval_claims_c1.py --n 6 --no-report 子进程退出码 0
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.state import AgentState                          # noqa: E402
from src.agents.claims import (                                 # noqa: E402
    validate_claims, evidence_catalog, render_evidence_catalog, build_rule_claims,
    render_claims_markdown, make_claim, make_evidence, CLAIM_TYPES,
)
from src.agents.generator import GeneratorAgent, CLAIMS_SEPARATOR   # noqa: E402
from src.utils.prompts import GENERATOR_SYSTEM_PROMPT           # noqa: E402

PASSED = 0
FAILED = 0


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
    "workflow": {"max_iterations": 1},
}


def _attr_result(primary="partial_discharge", name="局部放电", prob=0.6, sev="高风险"):
    return {
        "status": "ok", "primary_fault": primary, "primary_fault_name": name, "primary_probability": prob,
        "fault_ranking": [
            {"fault_id": primary, "fault_name": name, "probability": prob, "probability_pct": f"{prob*100:.1f}%", "rank": 1, "severity": sev},
            {"fault_id": "winding_short_circuit", "fault_name": "绕组匝间短路", "probability": 0.3, "probability_pct": "30.0%", "rank": 2, "severity": "中等风险"},
        ],
        "dga_analysis": {"interpretation": "C₂H₂ 超过注意值", "matched_rules": [{"rule_name": "电弧放电", "confidence": 0.9}]},
        "evidence_used": ["C2H2_elevated"], "evidence_negative": ["H2_elevated"],
        "uncertainty": {"entropy_bits": 1.1, "top1_top2_gap": 0.3, "calibrated": True},
    }


def _call(idx, tool, result, args=None, success=True):
    return {"tool": tool, "call_index": idx, "call_id": f"id{idx}", "raw_arguments": args or {}, "args": args or {},
            "result": result, "success": success, "exec_success": success, "business_success": success,
            "inquiry_round": 0}


def make_state_full() -> AgentState:
    st = AgentState(user_query="#2 主变 DGA 异常，请诊断。")
    st.tool_calls = [
        _call(0, "fault_attribution", _attr_result(), {"dga_data": {"H2": 80.0, "C2H2": 62.0}}),
        _call(1, "timeseries_anomaly", {"status": "ok", "n": 100, "mean": 1.2, "std": 0.3, "anomaly_indices": [3, 7]}, {"signal": [1, 2]}),
        _call(2, "ett_forecast", {"status": "ok", "dataset": "ETTh1", "freq": "1h", "lookback": 96, "horizon": 24,
                                  "forecast": {"mean_predicted_ot": 31.8, "min_predicted_ot": 30.1, "max_predicted_ot": 33.2},
                                  "history_summary": {"ot_mean": 30.5, "ot_std": 1.1}, "anomalies": {"count": 2}},
              {"dataset": "ETTh1", "horizon": 24}),
        _call(3, "kg_search", {"status": "ok", "matched_entities": [{"id": "f1", "name": "过热", "type": "fault"}],
                               "paths": [{"path": "过热 → 产生 → C2H4", "support_count": 5, "confidence": 0.8, "evidence": "过热故障主要产生乙烯"}],
                               "summary": "过热 → 产生 → C2H4"}, {"query": "过热"}),
        _call(4, "rag_search", None, {"query": "x"}, success=False),
    ]
    st.retrieved_knowledge = [
        {"chunk_id": "chunk_7f3a", "text": "乙炔含量超过 5 μL/L 时应缩短取样周期。", "score": 0.9,
         "metadata": {"title": "DGA 导则", "doc_name": "GB/T 7252"}},
    ]
    st.inquiry_log = [{"round": 1, "action": "ask", "symptom": "partial_discharge_alarm", "answer": True}]
    return st


# ─────────────────────────────────────────────────────────────
print("[1] validate_claims")
good = {"claims": [make_claim("c1", "H2=80 ppm", "observation", [make_evidence("tool", "call:0", "{\"H2\": 80}")])]}
ok, errs, norm = validate_claims(good)
check(ok and not errs and len(norm) == 1, "合法 claims 通过")
ok, errs, _ = validate_claims({"claims": [{"id": "c1", "text": "x", "type": "observation", "evidence": []}]})
check(not ok and any("evidence 缺失" in e for e in errs), "缺 evidence 判非法")
ok, errs, _ = validate_claims([{"id": "c1", "text": "x", "type": "guess", "evidence": [make_evidence("tool", "call:0", "y")]}])
check(not ok and any("type 非法" in e for e in errs), "type 非法")
ok, errs, _ = validate_claims([{"id": "c1", "text": "x", "type": "inference", "evidence": [{"source": "kb", "ref": "call:0", "span": "y"}]}])
check(not ok and any("前缀与 source 不匹配" in e for e in errs), "ref 前缀与 source 不匹配")
ok, errs, _ = validate_claims([{"id": "c1", "text": "x", "type": "inference", "evidence": [{"source": "tool", "ref": "call:0", "span": ""}]}])
check(not ok and any("span 为空" in e for e in errs), "span 为空")
ok, errs, _ = validate_claims([make_claim("c1", "a", "observation", [make_evidence("user", "user:0", "q")]),
                               make_claim("c1", "b", "observation", [make_evidence("user", "user:0", "q")])])
check(not ok and any("id 重复" in e for e in errs), "id 重复")
ok, errs, _ = validate_claims(good, known_refs={"call:1"})
check(ok and errs and errs[0].startswith("[warning]"), "未知 ref 记 warning 且不影响 ok")
ok, errs, _ = validate_claims({"claims": "oops"})
check(not ok and "claims 不是列表" in errs[0], "claims 非列表")
ok, errs, _ = validate_claims([{"id": "c1", "text": "", "type": "safety", "evidence": [make_evidence("tool", "call:0", "y")]}])
check(not ok and any("text 为空" in e for e in errs), "text 为空")

# ─────────────────────────────────────────────────────────────
print("[2] evidence_catalog")
st = make_state_full()
cat = evidence_catalog(st)
check({"call:0", "call:1", "call:2", "call:3"} <= cat["refs"], "成功工具调用均入目录")
check("call:4" not in cat["refs"], "失败调用不入目录")
check("kb:chunk_7f3a" in cat["refs"], "kb chunk_id 入目录")
check("kg:过热 → 产生 → C2H4" in cat["refs"], "kg path 入目录")
check({"user:0", "user:1"} <= cat["refs"], "用户原问与追问轮次入目录")
txt = render_evidence_catalog(cat)
check("call:0" in txt and "kb:chunk_7f3a" in txt and "user:1" in txt, "证据目录可渲染")
empty_cat = evidence_catalog(AgentState(user_query="q"))
check(empty_cat["refs"] == {"user:0"} and "无任何可引用证据" in render_evidence_catalog(empty_cat), "空状态目录只含 user:0")

# ─────────────────────────────────────────────────────────────
print("[3] build_rule_claims")
claims = build_rule_claims(st)
ok, errs, _ = validate_claims(claims, cat["refs"])
check(ok and not errs, f"规则声明通过 schema 且 ref 全在目录（{len(claims)} 条）")
types = {c["type"] for c in claims}
check(types == set(CLAIM_TYPES), f"四类声明齐全：{sorted(types)}")
check(any(c["type"] == "observation" and "H2=80.0 ppm" in c["text"] and "C2H2=62.0 ppm" in c["text"] for c in claims), "DGA 入参 observation")
check(any(c["type"] == "inference" and "局部放电" in c["text"] and "60.0%" in c["text"] for c in claims), "主要故障 inference 概率一致")
saf = [c for c in claims if c["type"] == "safety"]
check(len(saf) == 1 and len(saf[0]["evidence"]) == 2 and all(e["source"] == "tool" for e in saf[0]["evidence"]), "高危故障产 1 条 safety 且双 tool 引用")
check(any(e["ref"] == "user:1" for c in claims for e in c["evidence"]), "追问回答 → user:1 引用")
check(any(e["ref"] == "kg:过热 → 产生 → C2H4" for c in claims for e in c["evidence"]), "图谱关系 → kg 引用")
check(any(e["ref"] == "kb:chunk_7f3a" and "乙炔" in e["span"] for c in claims for e in c["evidence"]), "知识片段 → kb 引用且 span 为原文")
check(any("ETTh1" in c["text"] and "24 步" in c["text"] for c in claims), "预测 observation 含数据集与步数")
check(any("3σ 检出 2 个异常点" in c["text"] for c in claims), "时序 observation 含异常点数")
ids = [c["id"] for c in claims]
check(len(ids) == len(set(ids)), "id 唯一")

st_low = AgentState(user_query="q")
st_low.tool_calls = [_call(0, "fault_attribution", _attr_result("overload_overheating", "过载过热", 0.55, "中等风险"), {"dga_data": {"C2H4": 100.0}})]
claims_low = build_rule_claims(st_low)
check(not any(c["type"] == "safety" for c in claims_low), "非高危故障不产 safety")
st_lowp = AgentState(user_query="q")
st_lowp.tool_calls = [_call(0, "fault_attribution", _attr_result("partial_discharge", "局部放电", 0.3, "低风险"), {"dga_data": {"C2H2": 3.0}})]
check(not any(c["type"] == "safety" for c in build_rule_claims(st_lowp)), "高危类型但概率 < 0.4 不产 safety")

st_empty = AgentState(user_query="设备有异响")
ce = build_rule_claims(st_empty)
check(len(ce) == 1 and ce[0]["type"] == "observation" and ce[0]["evidence"][0]["ref"] == "user:0" and "信息不足" in ce[0]["text"] or "无法给出" in ce[0]["text"],
      "无证据 → 一条 user:0 信息不足声明")
st_pend = AgentState(user_query="q")
st_pend.pending_questions = [{"symptom": "oil_temp_high", "question": "油温是否偏高？"}]
cp = build_rule_claims(st_pend)
check("oil_temp_high" in cp[0]["text"], "待追问时声明列出缺失项")
st_fail = AgentState(user_query="q")
st_fail.tool_calls = [_call(0, "fault_attribution", {"status": "error", "message": "x"}, {}, success=False)]
cf = build_rule_claims(st_fail)
check(len(cf) == 1 and cf[0]["evidence"][0]["ref"] == "user:0", "工具失败不产生工具声明")
md = render_claims_markdown(claims)
check(md.startswith("【声明与证据清单】") and "[c1]" in md and "⟨证据：call:0⟩" in md, "声明清单渲染")

# ─────────────────────────────────────────────────────────────
print("[4] GeneratorAgent")
g_text = GeneratorAgent(NO_LLM_CFG)
check(g_text.output_mode == "text" and not g_text.llm.enabled, "默认 text 模式、LLM 未启用")
s1 = make_state_full()
g_text.run(s1)
check(s1.draft_answer and "【声明与证据清单】" not in s1.draft_answer, "text 模式草案不附声明清单")
check(s1.draft_claims != [] and s1.claims_source == "rule", "模板回退路径始终产 claims（供 C-2 使用）")

cfg_claims = json.loads(json.dumps(NO_LLM_CFG))
cfg_claims["workflow"]["generator_output_mode"] = "claims"
g_claims = GeneratorAgent(cfg_claims)
check(g_claims.output_mode == "claims", "配置 generator_output_mode=claims 生效")
s2 = make_state_full()
g_claims.run(s2)
check(len(s2.draft_claims) >= 8 and s2.claims_source == "rule" and s2.claims_json_valid is True, "claims 模式模板回退产 claims 且校验通过")
check("【声明与证据清单】" in (s2.draft_answer or ""), "claims 模式草案末尾附声明清单")
check(GeneratorAgent(NO_LLM_CFG, output_mode="bogus").output_mode == "text", "非法 output_mode 回退 text")

# _split_claims_output
payload = {"claims": [make_claim("c1", "H2=80 ppm", "observation", [make_evidence("tool", "call:0", "{\"H2\": 80}")])]}
raw = json.dumps(payload, ensure_ascii=False) + f"\n{CLAIMS_SEPARATOR}\n一、诊断结论：… [c1]"
c, a, v = GeneratorAgent._split_claims_output(raw, {"call:0"})
check(v and len(c) == 1 and "诊断结论" in a, "带分隔符输出解析")
raw2 = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```\n正文…"
c, a, v = GeneratorAgent._split_claims_output(raw2)
check(v and len(c) == 1 and "正文" in a, "无分隔符 + 代码块包裹解析")
c, a, v = GeneratorAgent._split_claims_output("这不是 JSON\n" + CLAIMS_SEPARATOR + "\n正文")
check(not v and c == [] and "正文" in a, "非法 JSON 判 invalid 且保留正文")
bad = {"claims": [{"id": "c1", "text": "x", "type": "observation", "evidence": []}]}
c, a, v = GeneratorAgent._split_claims_output(json.dumps(bad) + CLAIMS_SEPARATOR + "正文")
check(not v, "缺 evidence 判 invalid")
c, a, v = GeneratorAgent._split_claims_output("")
check(not v and c == [] and a == "", "空输出")

# ─────────────────────────────────────────────────────────────
print("[5] prompts")
sp = GENERATOR_SYSTEM_PROMPT("claims")
check("===ANSWER===" in sp and '"claims"' in sp and "safety" in sp, "system_claims.txt 加载")
check("===ANSWER===" not in GENERATOR_SYSTEM_PROMPT(), "基础模板不含 claims 规范")
check(GENERATOR_SYSTEM_PROMPT("nonexistent") == GENERATOR_SYSTEM_PROMPT(), "缺失 variant 回退基础模板")

# ─────────────────────────────────────────────────────────────
print("[6] eval_claims_c1.py 端到端（子进程，--no-report）")
proc = subprocess.run([sys.executable, str(ROOT / "scripts/eval/eval_claims_c1.py"), "--n", "6", "--no-report"],
                      capture_output=True, text=True, cwd=str(ROOT), timeout=300)
check(proc.returncode == 0, f"退出码 0（实际 {proc.returncode}）")
check('"pass": true' in proc.stdout and '"json_valid_rate": 1.0' in proc.stdout, "6 条 JSON 合法率 100% 且 pass")
if proc.returncode != 0:
    print(proc.stdout[-2000:], proc.stderr[-2000:])

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
