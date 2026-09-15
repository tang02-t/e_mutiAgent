#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0-1 验收测试：确认「会污染实验的缺陷」已被修复。

不依赖 LLM、不依赖网络：通过桩 Planner 直接注入 reasoning_trace 中的 llm_plan。
运行：python3 tests/test_p0_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.state import AgentState                                # noqa: E402
from src.tools.mcp_client import MCPClient                            # noqa: E402
from src.tools.tool_registry import validate_arguments_strict         # noqa: E402
from src.agents.retriever import RetrieverAgent                       # noqa: E402
from src.utils.prompts import render_planner_user                     # noqa: E402


PASSED, FAILED = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


# ──────────────────────────────────────────────────────────────
# 桩工具
# ──────────────────────────────────────────────────────────────
CALLS: list = []


def stub_rag(query):
    CALLS.append(("rag_search", query))
    return [{"text": f"hit for {query}", "score": 0.9}] if "空" not in query else []


def stub_ts(signal):
    CALLS.append(("timeseries_anomaly", list(signal)))
    return {"status": "ok", "mean": 1.0, "std": 0.1, "anomaly_indices": []}


def stub_ett(**kw):
    CALLS.append(("ett_forecast", kw))
    if kw.get("dataset") == "ETTh2":
        return {"status": "error", "message": "模拟业务错误"}
    return {"status": "ok", "forecast": {"values": [1, 2]}}


def stub_fa(dga_data=None, evidence=None, query=""):
    CALLS.append(("fault_attribution", dga_data))
    return {"status": "ok", "primary_fault_name": "x"}


def stub_kg(query, relations=None, hops=1, direction="both", max_paths=20):
    CALLS.append(("kg_search", {"query": query, "relations": relations, "hops": hops, "direction": direction}))
    if "无实体" in query:
        return {"status": "no_match", "matched_entities": [], "paths": []}
    if "孤立" in query:
        return {"status": "no_relation", "matched_entities": [{"id": "Fault:x", "name": "x"}], "paths": []}
    return {"status": "ok", "matched_entities": [{"id": "Fault:铁心多点接地", "name": "铁心多点接地", "type": "Fault"}],
            "paths": [{"path": ["Fault:铁心多点接地", "CAUSES", "Fault:过热故障"], "support_count": 8,
                       "confidence": 0.6, "evidence": "多点接地引起过热"}],
            "summary": "铁心多点接地 → 导致 → 过热故障（8 篇支持，置信 0.6）"}


def make_mcp() -> MCPClient:
    m = MCPClient()
    m.register_tool("rag_search", stub_rag)
    m.register_tool("timeseries_anomaly", stub_ts)
    m.register_tool("ett_forecast", stub_ett)
    m.register_tool("fault_attribution", stub_fa)
    m.register_tool("kg_search", stub_kg)
    return m


def run_with_plan(steps, plan_status="ok") -> AgentState:
    CALLS.clear()
    st = AgentState(user_query="测试问题")
    st.plan_status = plan_status
    st.reasoning_trace.append({"agent": "planner", "type": "llm_plan",
                               "content": {"steps": steps, "plan_status": plan_status}})
    r = RetrieverAgent({"workflow": {"parallel_tools": 2}}, make_mcp())
    return r.run(st)


# ──────────────────────────────────────────────────────────────
print("\n== 1. 严格校验不再静默修正 ==")
v = validate_arguments_strict("ett_forecast", {"dataset": "ETTx9", "horizon": 24})
check("非法枚举被报告而非删除", not v["valid"] and v["errors"][0]["code"] == "invalid_enum", str(v))
check("原始参数原样保留", v["arguments"] == {"dataset": "ETTx9", "horizon": 24})
v = validate_arguments_strict("ett_forecast", {"dataset": "ETTh1", "foo": 1})
check("未知字段被报告", any(e["code"] == "unknown_field" for e in v["errors"]))
v = validate_arguments_strict("ett_forecast", {"dataset": "ETTh1", "horizon": "24"})
check("类型不匹配被报告", any(e["code"] == "type_mismatch" for e in v["errors"]))
v = validate_arguments_strict("timeseries_anomaly", {})
check("timeseries_anomaly 缺少 signal 报 missing_required",
      any(e["code"] == "missing_required" for e in v["errors"]))
v = validate_arguments_strict("timeseries_anomaly", {"signal": [1, 2, "a"]})
check("signal 元素类型被检查", any(e["code"] == "invalid_item_type" for e in v["errors"]))
v = validate_arguments_strict("rag_search", {"query": "x"})
check("合法参数通过", v["valid"])
v = validate_arguments_strict("no_such_tool", {})
check("未知工具报 unknown_tool", v["errors"][0]["code"] == "unknown_tool")

print("\n== 2. 空计划不再触发固定工具 ==")
st = run_with_plan([], plan_status="no_tool")
check("空计划零工具调用", len(CALLS) == 0 and st.tool_calls == [], f"CALLS={CALLS}")
check("fallback_mode 恒为 False", st.fallback_mode is False)
st = run_with_plan([], plan_status="parse_failed")
check("解析失败同样零调用", len(CALLS) == 0)

print("\n== 3. timeseries_anomaly 不再使用示例信号 ==")
st = run_with_plan([{"id": 1, "tool": "timeseries_anomaly", "arguments": {}}])
check("缺 signal → 不执行、记录 validation_failed",
      len(CALLS) == 0 and st.tool_calls[0]["error_code"] == "validation_failed", str(st.tool_calls))
st = run_with_plan([{"id": 1, "tool": "timeseries_anomaly", "arguments": {"signal": [5, 6, 7, 8]}}])
check("有 signal → 使用模型传入的序列",
      CALLS and CALLS[0] == ("timeseries_anomaly", [5, 6, 7, 8]), str(CALLS))

print("\n== 4. 非法参数不执行且保留原始值 ==")
st = run_with_plan([{"id": 1, "tool": "ett_forecast", "arguments": {"dataset": "ETTx9"}}])
rec = st.tool_calls[0]
check("非法枚举不执行", len(CALLS) == 0)
check("记录 raw_arguments 原值", rec["raw_arguments"] == {"dataset": "ETTx9"})
check("success=False, error_code=validation_failed",
      rec["success"] is False and rec["error_code"] == "validation_failed")

print("\n== 5. 业务错误不再记 success=True ==")
st = run_with_plan([{"id": 1, "tool": "ett_forecast", "arguments": {"dataset": "ETTh2", "horizon": 6}}])
rec = st.tool_calls[0]
check("exec_success=True 但 business_success=False",
      rec["exec_success"] is True and rec["business_success"] is False, str(rec))
check("兼容字段 success=False", rec["success"] is False)
check("error_code=tool_error", rec["error_code"] == "tool_error")
check("state.errors 记录业务错误", any(e.get("error_code") == "tool_error" for e in st.errors))

st = run_with_plan([{"id": 1, "tool": "rag_search", "arguments": {"query": "空结果"}}])
check("rag 空结果记 empty_result", st.tool_calls[0]["error_code"] == "empty_result")

print("\n== 6. 正常调用与统一轨迹 ==")
st = run_with_plan([
    {"id": 1, "tool": "rag_search", "arguments": {"query": "乙炔升高"}},
    {"id": 2, "tool": "fault_attribution", "arguments": {"dga_data": {"H2": 100}}},
    {"id": 3, "tool": "ett_forecast", "arguments": {"dataset": "ETTm1", "horizon": 24}},
])
check("三个工具都执行", len(CALLS) == 3)
check("全部 success", all(r["success"] for r in st.tool_calls))
check("retrieved_knowledge 聚合", len(st.retrieved_knowledge) == 1)
check("trajectory 条数 = 调用数", len(st.trajectory) == 3)
need = {"call_index", "tool", "raw_arguments", "validation", "exec_success",
        "business_success", "error_code", "latency_ms"}
check("trajectory 字段完整", all(need <= set(t) for t in st.trajectory))
ett_kw = [c for c in CALLS if c[0] == "ett_forecast"][0][1]
check("ett_forecast 只传显式参数", ett_kw == {"dataset": "ETTm1", "horizon": 24}, str(ett_kw))

print("\n== 7. Planner 输入包含上下文 ==")
txt = render_planner_user(query="Q", context="- 前端填写的 DGA 数据：H2=1")
check("用户模板渲染出 context", "前端填写的 DGA 数据" in txt and "{{context}}" not in txt)

print("\n== 8. ETT 15 分钟频率映射 ==")
try:
    from src.tools.ett_loader import ETTDatasetCache
    import pandas as pd
    df = ETTDatasetCache().load("ETTm1")
    delta = (df.index[1] - df.index[0])
    check("ETTm1 索引间隔为 15 分钟", delta == pd.Timedelta(minutes=15), str(delta))
    check("ETTm1 行数未被扩张（≈69680）", 69000 <= len(df) <= 70500, str(len(df)))
except Exception as exc:  # noqa: BLE001
    check("ETT 加载", False, f"异常：{exc}")

print("\n== 9. kg_search 接入（P2） ==")
v = validate_arguments_strict("kg_search", {"query": "铁心多点接地", "hops": 2, "direction": "out"})
check("kg_search 合法参数通过", v["valid"], str(v))
v = validate_arguments_strict("kg_search", {"query": "x", "relations": ["FOO"]})
check("非法关系枚举被报告", not v["valid"] and any(e["code"] == "invalid_enum" for e in v["errors"]), str(v))
v = validate_arguments_strict("kg_search", {"query": "x", "direction": "up"})
check("非法 direction 被报告", not v["valid"], str(v))

st = run_with_plan([{"id": 1, "tool": "kg_search",
                     "arguments": {"query": "铁心多点接地", "relations": ["CAUSES"], "hops": 2, "direction": "out"}}])
check("kg_search 正常执行并透传参数",
      CALLS and CALLS[0][0] == "kg_search" and CALLS[0][1]["relations"] == ["CAUSES"] and CALLS[0][1]["hops"] == 2,
      str(CALLS))
check("kg_search ok → success=True", st.tool_calls[0]["success"] is True, str(st.tool_calls[0]))

st = run_with_plan([{"id": 1, "tool": "kg_search", "arguments": {"query": "无实体"}}])
check("no_match → business_success=False, error_code=tool_no_match",
      st.tool_calls[0]["business_success"] is False and st.tool_calls[0]["error_code"] == "tool_no_match",
      str(st.tool_calls[0]))
st = run_with_plan([{"id": 1, "tool": "kg_search", "arguments": {"query": "孤立"}}])
check("no_relation → error_code=tool_no_relation",
      st.tool_calls[0]["error_code"] == "tool_no_relation", str(st.tool_calls[0]))

try:
    from src.agents.generator import GeneratorAgent
    from src.utils.prompts import render_generator_user
    g = GeneratorAgent({})
    st = run_with_plan([{"id": 1, "tool": "kg_search", "arguments": {"query": "铁心多点接地"}}])
    kb_t, ts_t, at_t, fc_t, kg_t = g._extract_tool_results(st)
    check("Generator 提取到图谱文本", "铁心多点接地" in kg_t and "过热故障" in kg_t, kg_t)
    u = render_generator_user(query="q", kb_text=kb_t, ts_text=ts_t, attribution_text=at_t,
                              forecast_text=fc_t, kg_text=kg_t)
    check("Generator 用户提示词含图谱段且无未填占位", "【故障关系图谱】" in u and "{{" not in u)
    st = run_with_plan([{"id": 1, "tool": "kg_search", "arguments": {"query": "无实体"}}])
    kg_t = g._extract_tool_results(st)[4]
    check("no_match 时给出明确说明而非空串", "未识别到图谱实体" in kg_t, kg_t)
    draft = g._template_generate(st)
    check("模板生成包含图谱章节", "故障关系图谱" in draft)
except Exception as exc:  # noqa: BLE001
    check("Generator 图谱接入", False, f"异常：{exc}")

print(f"\n结果：{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
