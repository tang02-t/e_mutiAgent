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

# ──────────────────────────────────────────────────────────────
# 10. P3 反思模块：评分丢弃 / 同章节补召回 / 全丢改写重检索 / 无输入 / 工作流接入
# ──────────────────────────────────────────────────────────────
print("\n[10] P3 反思模块")
try:
    from src.agents.reflection import ReflectionModule, Scorer, LexicalScorer
    from src.graph.workflow import run_diagnosis_workflow

    class KeywordScorer(Scorer):
        """桩评分器：块正文含 'good' 记 3 分，含 'mid' 记 2 分，否则 0 分。"""
        name = "stub_keyword"

        def score(self, query, chunk_text, section_title=""):
            return 3 if "good" in chunk_text else (2 if "mid" in chunk_text else 0)

    class StubKB:
        """桩知识库：c1 的同章节邻块为 n1；重检索固定返回 r1（good）。"""
        RESEARCH_CALLS: list = []

        def neighbors(self, chunk_id, window=1, same_section=True):
            return [{"chunk_id": "n1", "text": "neighbor mid text", "doc_id": "d", "section_path": ["s"]}] \
                if chunk_id == "c1" else []

        def format_chunk(self, c, score=0.0):
            return {"chunk_id": c["chunk_id"], "text": c["text"], "doc_id": c.get("doc_id"),
                    "section_path": c.get("section_path", []), "score": score}

        def search(self, query):
            self.RESEARCH_CALLS.append(query)
            return [{"chunk_id": "r1", "text": "good rewritten hit", "score": 0.5}]

    def mk_state(items):
        s = AgentState(user_query="变压器铁心多点接地 的原因")
        s.retrieved_knowledge = items
        return s

    kb = StubKB()
    refl = ReflectionModule(scorer=KeywordScorer(), kb=kb, max_rounds=2, max_total_chunks=8)

    # (a) 评分丢弃 + 补召回：c1 good 保留，c2 bad 丢弃，c1 邻块 n1（mid=2）补入
    st = refl.run(mk_state([{"chunk_id": "c1", "text": "good text", "score": 0.9},
                            {"chunk_id": "c2", "text": "bad text", "score": 0.8}]))
    ids = [it["chunk_id"] for it in st.retrieved_knowledge]
    check("低分块被丢弃", "c2" not in ids, str(ids))
    check("高分块保留并带 reflection_score", "c1" in ids and st.retrieved_knowledge[0]["reflection_score"] == 3,
          str(st.retrieved_knowledge))
    check("同章节邻块补召回并标记 expanded_from",
          "n1" in ids and any(it.get("expanded_from") == "c1" for it in st.retrieved_knowledge), str(ids))
    log = st.reflection_log[-1]
    check("反思日志 decision=kept 且两轮", log["decision"] == "kept" and len(log["rounds"]) == 2, str(log))
    check("reasoning_trace 记录反思", any(t.get("agent") == "reflection" for t in st.reasoning_trace))
    check("输出按 reflection_score 降序", [it["reflection_score"] for it in st.retrieved_knowledge]
          == sorted([it["reflection_score"] for it in st.retrieved_knowledge], reverse=True))

    # (b) 全丢 → 改写重检索一次
    kb.RESEARCH_CALLS.clear()
    st = refl.run(mk_state([{"chunk_id": "c9", "text": "bad", "score": 0.7}]))
    log = st.reflection_log[-1]
    check("全丢触发改写重检索", log["rewrite_triggered"] and len(kb.RESEARCH_CALLS) == 1, str(log))
    check("改写后查询不等于原查询", log.get("rewritten_query") and log["rewritten_query"] != st.user_query,
          str(log.get("rewritten_query")))
    check("重检索结果被采纳并标记 from_rewrite",
          [it["chunk_id"] for it in st.retrieved_knowledge] == ["r1"] and st.retrieved_knowledge[0].get("from_rewrite"),
          str(st.retrieved_knowledge))

    # (c) 无输入
    st = refl.run(mk_state([]))
    log = st.reflection_log[-1]
    check("无输入 decision=no_input 且 output_count=0", log["decision"] == "no_input" and log["output_count"] == 0, str(log))

    # (d) 重检索仍全丢 → all_dropped，检索结果为空
    class BadKB(StubKB):
        def search(self, query):
            return [{"chunk_id": "r2", "text": "still bad", "score": 0.1}]
    st = ReflectionModule(scorer=KeywordScorer(), kb=BadKB()).run(mk_state([{"chunk_id": "c9", "text": "bad"}]))
    check("重检索仍全丢 → all_dropped 且结果为空",
          st.reflection_log[-1]["decision"] == "all_dropped" and st.retrieved_knowledge == [],
          str(st.reflection_log[-1]))

    # (e) enabled=False 不改动状态
    st = ReflectionModule(scorer=KeywordScorer(), kb=kb, enabled=False).run(
        mk_state([{"chunk_id": "c2", "text": "bad"}]))
    check("禁用时不过滤、不写日志", len(st.retrieved_knowledge) == 1 and not st.reflection_log)

    # (f) 评分器异常 → 回退 lexical
    class BoomScorer(Scorer):
        name = "boom"

        def score_batch(self, query, items):
            raise RuntimeError("llm down")
    st = ReflectionModule(scorer=BoomScorer(), kb=kb).run(
        mk_state([{"chunk_id": "c1", "text": "铁心多点接地会引起铁心过热。", "score": 0.9}]))
    check("评分器异常回退到 lexical", "fallback" in st.reflection_log[-1]["rounds"][0]["scorer"],
          str(st.reflection_log[-1]))

    # (g) LexicalScorer 基本区分度
    lx = LexicalScorer()
    q = "铁心多点接地的原因"
    check("LexicalScorer 相关块 > 无关块",
          lx.score(q, "铁心多点接地的主要原因是铁心绝缘损坏或夹件与铁心短接。", "铁心故障")
          > lx.score(q, "本标准规定了绝缘油色谱分析的取样周期。", "取样"))

    # (h) 工作流接入：reflector 非 None 时 retriever→reflection→generator
    class StubAgent:
        def __init__(self, name, fn=None):
            self.name, self.fn = name, fn

        def run(self, state, **kwargs):
            state.reasoning_trace.append({"agent": self.name})
            if self.fn:
                self.fn(state)
            return state

    def fill_kb(state):
        state.retrieved_knowledge = [{"chunk_id": "c1", "text": "good", "score": 1.0},
                                     {"chunk_id": "c2", "text": "bad", "score": 0.9}]

    def gen(state):
        state.final_answer = f"n_kb={len(state.retrieved_knowledge)}"

    st = run_diagnosis_workflow(mk_state([]), StubAgent("planner"), StubAgent("retriever", fill_kb),
                                StubAgent("generator", gen), StubAgent("validator"),
                                reflector=ReflectionModule(scorer=KeywordScorer(), kb=kb))
    order = [t.get("agent") for t in st.reasoning_trace]
    check("工作流中反思位于 retriever 与 generator 之间",
          order.index("retriever") < order.index("reflection") < order.index("generator"), str(order))
    check("Generator 拿到的是反思过滤后的知识", st.final_answer == "n_kb=2", str(st.final_answer))
    st = run_diagnosis_workflow(mk_state([]), StubAgent("planner"), StubAgent("retriever", fill_kb),
                                StubAgent("generator", gen), StubAgent("validator"))
    check("无 reflector 时工作流不含反思节点",
          "reflection" not in [t.get("agent") for t in st.reasoning_trace] and st.final_answer == "n_kb=2")
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("反思模块用例", False, f"异常：{exc}")

# ──────────────────────────────────────────────────────────────
# P5-5 / P6：planner_mode 开关与五级系统模式
# ──────────────────────────────────────────────────────────────
print("\n[P5-5/P6] planner_mode 与五级系统模式")
try:
    from src.agents.planner import PlannerAgent
    from src.tools.tool_registry import to_openai_tools, render_tools_for_prompt
    from src.utils.prompts import PLANNER_SYSTEM_PROMPT
    from src.graph.system_modes import SYSTEM_MODES, MODE_ORDER, resolve_mode, register_rag_for_mode

    base_cfg = {"llms": {"planner": {"provider": "openai", "base_url": "http://x", "api_key": "",
                                     "model_name": "m-base"}},
                "workflow": {"planner_mode": "baseline"}}
    p = PlannerAgent(base_cfg)
    check("planner_mode 默认 baseline 且记录模型名", p.planner_mode == "baseline" and p.model_name == "m-base")
    p = PlannerAgent(base_cfg, planner_mode="finetuned")
    check("finetuned 缺配置时回退 baseline", p.planner_mode == "baseline")
    cfg2 = {**base_cfg, "llms": {**base_cfg["llms"], "planner_finetuned": {"model_name": "m-lora", "api_key": "k"}}}
    p = PlannerAgent(cfg2, planner_mode="finetuned")
    check("finetuned 有配置时切换模型", p.planner_mode == "finetuned" and p.model_name == "m-lora")
    try:
        PlannerAgent(base_cfg, planner_mode="xxx")
        check("非法 planner_mode 抛错", False)
    except ValueError:
        check("非法 planner_mode 抛错", True)

    # 工具过滤
    names = [t["function"]["name"] for t in to_openai_tools(["rag_search", "fault_attribution"])]
    check("to_openai_tools 按名单过滤", names == ["rag_search", "fault_attribution"], str(names))
    txt = render_tools_for_prompt(["fault_attribution"])
    check("render_tools_for_prompt 只渲染名单内工具", "fault_attribution" in txt and "kg_search" not in txt)
    sp = PLANNER_SYSTEM_PROMPT(["fault_attribution"])
    check("系统提示词不出现被屏蔽工具", "kg_search" not in sp and "rag_search" not in sp)

    # Planner 校验：名单外工具标记 tool_disabled
    p = PlannerAgent(base_cfg, allowed_tools=["fault_attribution"])
    steps = [{"id": 1, "tool": "kg_search", "arguments": {"query": "x"}},
             {"id": 2, "tool": "fault_attribution", "arguments": {"dga_data": {"H2": 1.0}}}]
    p._validate_steps(steps)
    check("Planner 名单外工具 code=tool_disabled",
          steps[0]["validation"]["errors"][0]["code"] == "tool_disabled" and steps[1]["validation"]["valid"])

    # Retriever 守卫：名单外工具不执行
    CALLS.clear()
    mcp = MCPClient()
    mcp.register_tool("rag_search", stub_rag)
    mcp.register_tool("fault_attribution", stub_fa)
    st = AgentState(user_query="q")
    st.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": {"steps": [
        {"id": 1, "tool": "rag_search", "arguments": {"query": "a"}, "validation": {"valid": True, "errors": []}},
        {"id": 2, "tool": "fault_attribution", "arguments": {"dga_data": {"H2": 1.0}},
         "validation": {"valid": True, "errors": []}},
    ]}})
    st = RetrieverAgent({}, mcp, allowed_tools=["fault_attribution"]).run(st)
    codes = {c["tool"]: c["error_code"] for c in st.tool_calls}
    check("Retriever 拒绝名单外工具且不实际调用",
          codes.get("rag_search") == "tool_disabled" and all(c[0] != "rag_search" for c in CALLS)
          and codes.get("fault_attribution") is None, str(codes))
    check("拒绝记录进入 trajectory", any(t["error_code"] == "tool_disabled" for t in st.trajectory))

    # 五级模式定义单调递增
    check("五级模式能力单调递增",
          all(set(SYSTEM_MODES[a].allowed_tools) <= set(SYSTEM_MODES[b].allowed_tools)
              for a, b in zip(MODE_ORDER, MODE_ORDER[1:]))
          and SYSTEM_MODES["mode1"].rag_mode is None and SYSTEM_MODES["mode2"].rag_mode == "naive"
          and "kg_search" not in SYSTEM_MODES["mode3"].allowed_tools and "kg_search" in SYSTEM_MODES["mode4"].allowed_tools
          and SYSTEM_MODES["mode5"].reflection != "off")
    spec = resolve_mode("4", planner_mode="baseline", reflection="llm")
    check("resolve_mode 支持别名与覆盖", spec.key == "mode4" and spec.planner_mode == "baseline" and spec.reflection == "llm")
    m = MCPClient()
    check("mode1 的 rag_search 桩返回空", register_rag_for_mode(m, SYSTEM_MODES["mode1"]) == "disabled"
          and m.call_tool("rag_search", "x") == [])
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("P5-5/P6 用例", False, f"异常：{exc}")

print(f"\n结果：{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
