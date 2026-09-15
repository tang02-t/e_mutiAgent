#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变压器故障诊断多智能体系统 —— Streamlit 可视化前端。

运行方式：
    streamlit run app.py

功能：
    - 输入诊断问题 / 设备信息 / DGA 油色谱数据
    - 一键运行 Planner → Retriever → Generator → Validator 工作流
    - 可视化展示：诊断结论、质量验证、规划计划、工具调用结果、
      故障归因排序、油温预测、检索知识、推理轨迹与错误信息
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# ── 项目根路径注入 ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.state import AgentState                       # noqa: E402
from src.graph.workflow import run_diagnosis_workflow        # noqa: E402
from src.tools.mcp_client import MCPClient                   # noqa: E402
from src.tools.fault_attribution import fault_attribution    # noqa: E402
from src.tools.ett_forecasting import ett_forecast           # noqa: E402
from src.tools.timeseries import timeseries_anomaly          # noqa: E402
from src.utils.config import load_config                     # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Mock 工具（无外部依赖，便于离线演示）
# ══════════════════════════════════════════════════════════════════════════════
CASES_PATH = ROOT / "data/synthetic/cases/fault_cases.jsonl"


@st.cache_data(show_spinner=False)
def _build_case_index() -> List[Dict[str, Any]]:
    """从故障案例库构造一个简单的 mock 知识库。"""
    cases: List[Dict[str, Any]] = []
    if CASES_PATH.exists():
        with open(CASES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def mock_rag_search(query: str) -> List[Dict[str, Any]]:
    """按 query 与案例的故障中文名/征兆做朴素关键词匹配，返回 top 知识片段。"""
    cases = _build_case_index()
    scored = []
    for c in cases:
        score = 0
        text = c.get("fault_label_cn", "") + " " + " ".join(c.get("symptoms", []))
        for ch in set(query):
            if ch in text:
                score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    results = []
    for score, c in scored[:3]:
        results.append({
            "text": f"案例{c['case_id']}：{c['fault_location']}。"
                    f"征兆：{'、'.join(c.get('symptoms', []))}。"
                    f"处理：{c.get('handling', '')}。参考 {c.get('reference', '')}。",
            "score": float(score),
            "metadata": {"doc_name": c["case_id"], "title": c.get("fault_label_cn", "")},
        })
    return results


def build_mcp(kb_mode: str, user_dga: Dict[str, Any] | None, rag_mode: str | None = "two_way") -> MCPClient:
    """
    构造 MCP 客户端并注册工具。
    kb_mode: local_kb | mock；rag_mode：local_kb 的检索方式 naive | two_way | two_way_rerank，
    None 表示当前系统模式不开放文献检索（注册空桩，越权调用得到 empty_result）。
    """
    mcp = MCPClient()

    # fault_attribution：用闭包注入前端填写的 DGA（当 plan 未带 dga_data 时生效）
    def fault_attribution_with_dga(dga_data=None, evidence=None, query: str = "", **kwargs):
        if not dga_data and user_dga:
            dga_data = user_dga
        return fault_attribution(dga_data=dga_data, evidence=evidence, query=query, **kwargs)

    mcp.register_tool("fault_attribution", fault_attribution_with_dga)
    mcp.register_tool("timeseries_anomaly", timeseries_anomaly)
    mcp.register_tool("ett_forecast", ett_forecast)

    # kg_search：故障关系链路图谱（本地 graph.json）
    try:
        from src.tools.kg_search import kg_search
        mcp.register_tool("kg_search", kg_search)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"图谱工具初始化失败：{exc}")

    if rag_mode is None:
        mcp.register_tool("rag_search", lambda query, **_: [])
        return mcp

    if kb_mode == "local_kb":
        try:
            from src.tools.local_kb import local_kb_search
            _mode = rag_mode or "two_way"
            mcp.register_tool("rag_search", lambda query, top_k=5, **_: local_kb_search(query, top_k=top_k, mode=_mode))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"本地文献库初始化失败，已回退到合成案例 mock：{exc}")
            kb_mode = "mock"
    if kb_mode == "mock":
        mcp.register_tool("rag_search", mock_rag_search)

    return mcp


# ══════════════════════════════════════════════════════════════════════════════
# 工作流执行
# ══════════════════════════════════════════════════════════════════════════════
def run_workflow(
    user_query: str,
    context: Dict[str, Any],
    config: Dict[str, Any],
    mcp: MCPClient,
    reflection_mode: str = "off",
    planner_mode: str = "baseline",
    allowed_tools: List[str] | None = None,
    planner_strategy: str = "free",
    planner_decision: str = "llm",
    max_inquiry_rounds: int = 3,
    attribution_mode: str = "calibrated",
) -> tuple[AgentState | None, float, str | None]:
    """执行完整诊断工作流，返回 (final_state, elapsed, error)。
    planner_strategy=active 时遇到追问会中断并把问题留在 final_state.pending_questions，由前端收集回答后续跑。
    """
    from src.agents.planner import PlannerAgent
    from src.agents.retriever import RetrieverAgent
    from src.agents.generator import GeneratorAgent
    from src.agents.validator import ValidatorAgent
    from src.tools.fault_attribution import configure_engine

    try:
        configure_engine(attribution_mode)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"归因参数切换失败（{exc}），沿用当前引擎")

    state = AgentState(
        user_query=user_query,
        context=context,
        max_iterations=config.get("workflow", {}).get("max_iterations", 3),
        max_inquiry_rounds=int(max_inquiry_rounds),
    )

    planner = PlannerAgent(config, planner_mode=planner_mode, allowed_tools=allowed_tools,
                           strategy=planner_strategy, decision=planner_decision)
    retriever = RetrieverAgent(config, mcp, allowed_tools=allowed_tools)
    generator = GeneratorAgent(config)
    validator = ValidatorAgent(config)
    reflector = None
    if reflection_mode != "off":
        try:
            from src.agents.reflection import ReflectionModule, LexicalScorer
            from src.tools.local_kb import get_local_kb
            reflector = ReflectionModule(scorer=LexicalScorer(), kb=get_local_kb())
        except Exception as exc:  # noqa: BLE001
            st.warning(f"反思模块初始化失败，已关闭：{exc}")
    st.session_state["agents"] = (planner, retriever, generator, validator, reflector)

    t0 = time.time()
    try:
        final = run_diagnosis_workflow(state, planner, retriever, generator, validator, reflector=reflector,
                                       answer_fn=None)
        return final, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        import traceback
        return None, time.time() - t0, traceback.format_exc()


def resume_workflow(state: AgentState, symptom: str, answer: bool | None) -> tuple[AgentState | None, float, str | None]:
    """用户回答追问后续跑（复用本轮构造的智能体实例）。"""
    from src.graph.workflow import resume_after_answer
    agents = st.session_state.get("agents")
    if not agents:
        return None, 0.0, "智能体实例已失效，请重新开始诊断。"
    planner, retriever, generator, validator, reflector = agents
    t0 = time.time()
    try:
        final = resume_after_answer(state, symptom, answer, planner, retriever, generator, validator, reflector)
        return final, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        import traceback
        return None, time.time() - t0, traceback.format_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 渲染辅助
# ══════════════════════════════════════════════════════════════════════════════
VERDICT_STYLE = {
    "PASS": ("通过", "green"),
    "REVISION": ("需改进", "orange"),
    "FAIL": ("失败", "red"),
    "PENDING": ("待评估", "gray"),
}


def _extract_validation(state: AgentState) -> Dict[str, Any] | None:
    for item in reversed(state.reasoning_trace):
        if item.get("agent") == "validator" and item.get("type") == "evaluation":
            return item["content"]
    return None


def render_overview(state: AgentState, elapsed: float) -> None:
    val = _extract_validation(state)
    verdict = state.validation_verdict
    label, color = VERDICT_STYLE.get(verdict, (verdict, "gray"))
    score = val.get("score") if val else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("验证结论", label)
    c2.metric("质量评分", f"{score}/10" if score is not None else "—")
    c3.metric("迭代轮次", state.iteration)
    c4.metric("耗时", f"{elapsed:.1f}s")

    badges = []
    sm = st.session_state.get("system_mode")
    if sm:
        badges.append(f"系统模式：{sm.get('label')}")
    plan_content = next((it.get("content") for it in state.reasoning_trace
                         if it.get("agent") == "planner" and it.get("type") == "llm_plan"), None) or {}
    if plan_content.get("planner_model"):
        badges.append(f"Planner：{plan_content.get('planner_experiment_mode')}（{plan_content.get('planner_model')}）")
    badges.append(f"计划状态：{state.plan_status}")
    try:
        from src.tools.fault_attribution import get_engine
        src = get_engine().params_source
        badges.append("诊断参数：" + ("专家默认 CPT" if src == "expert_default" else f"数据学习 ({Path(src).name})"))
    except Exception:  # noqa: BLE001
        pass
    if state.errors:
        badges.append(f"❗ {len(state.errors)} 条错误")
    if badges:
        st.caption("　|　".join(badges))


def render_tool_calls(state: AgentState) -> None:
    if not state.tool_calls:
        st.info("本次未调用任何工具。")
        return

    for call in state.tool_calls:
        tool = call.get("tool", "unknown")
        ok = call.get("success")
        code = call.get("error_code")
        icon = "✅" if ok else ("⚠️" if code == "validation_failed" else "❌")
        with st.expander(f"{icon} 工具：{tool}" + (f"（{code}）" if code else ""), expanded=False):
            if call.get("raw_arguments"):
                st.caption("模型原始参数")
                st.json(call["raw_arguments"])
            if not ok:
                st.error(call.get("error", "执行失败"))
                val = call.get("validation") or {}
                if val.get("errors"):
                    st.json(val["errors"])
                continue
            result = call.get("result")

            if tool == "fault_attribution" and isinstance(result, dict):
                _render_fault_attribution(result)
            elif tool == "ett_forecast" and isinstance(result, dict):
                _render_ett_forecast(result)
            elif tool == "timeseries_anomaly" and isinstance(result, dict):
                _render_timeseries(result)
            elif tool == "rag_search":
                _render_rag(result)
            else:
                st.json(result)


def _render_fault_attribution(result: Dict[str, Any]) -> None:
    if result.get("status") != "ok":
        st.json(result)
        return
    primary = result.get("primary_fault_name", "未确定")
    prob = result.get("primary_probability", 0)
    st.markdown(f"**主推断故障：** {primary}　（后验概率 {prob * 100:.1f}%）")

    ranking = result.get("fault_ranking", [])
    if ranking:
        st.markdown("**故障概率排序（贝叶斯网络 × DGA 融合）：**")
        rows = [
            {
                "排名": r.get("rank"),
                "故障类型": r.get("fault_name"),
                "概率": r.get("probability_pct"),
                "严重度": r.get("severity"),
            }
            for r in ranking
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    dga = result.get("dga_analysis") or {}
    if dga.get("interpretation"):
        st.markdown(f"**DGA 特征解释：** {dga['interpretation']}")
    if result.get("evidence_negative"):
        st.caption("已排除征兆（负观测）：" + "、".join(result["evidence_negative"]))

    unc = result.get("uncertainty") or {}
    if unc:
        _render_uncertainty(unc, ranking)


def _render_uncertainty(unc: Dict[str, Any], ranking: List[Dict[str, Any]] | None = None) -> None:
    """后验分布 / 熵 / EIG 推荐征兆（B-4）。"""
    st.markdown("**不确定性与下一步建议**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("后验熵 H(F|E)", f"{unc.get('entropy_bits', 0):.2f} bit")
    c2.metric("Top-1 与 Top-2 差", f"{unc.get('top1_top2_gap', 0) * 100:.1f}%")
    c3.metric("参数校准", "已校准" if unc.get("calibrated") else "未校准")
    action_cn = {"ask": "追问用户", "call_tool": "调用工具", "conclude": "可下结论"}
    c4.metric("建议动作", action_cn.get(unc.get("suggested_action"), unc.get("suggested_action") or "—"))
    if ranking:
        probs = {r.get("fault_name"): float(str(r.get("probability_pct", "0")).rstrip("%")) for r in ranking}
        st.bar_chart(probs)
    recs = unc.get("recommendations") or []
    if recs:
        st.caption("候选征兆（按信息价值 VoI = EIG − λ·cost 降序）")
        st.dataframe([
            {"征兆": r.get("symptom"), "EIG (bit)": round(r.get("eig", 0), 3), "成本": r.get("cost"),
             "VoI": round(r.get("voi", 0), 3), "获取方式": r.get("how_to_obtain"), "追问话术": r.get("ask_hint", "")}
            for r in recs[:5]
        ], use_container_width=True, hide_index=True)
    if unc.get("stop_reason"):
        st.caption(f"停止原因：{unc['stop_reason']}")


def _render_inquiry(state: AgentState) -> None:
    """追问循环轨迹：每轮熵 / 动作 / 征兆 / 回答。"""
    if not state.inquiry_log:
        st.info("本次未进入主动追问循环（planner_strategy=free 或无需追问）。")
        return
    st.markdown(f"追问 {state.inquiry_rounds} 轮，累计获取成本 {state.inquiry_cost:.0f}")
    rows = []
    for e in state.inquiry_log:
        rows.append({
            "轮": e.get("round"), "动作": e.get("action"), "决策来源": e.get("decision_source"),
            "征兆": e.get("symptom") or "", "回答": {True: "有", False: "无", None: ""}.get(e.get("answer"), ""),
            "后验熵": None if e.get("entropy_bits") is None else round(e["entropy_bits"], 3),
            "Top-1": e.get("top1") or "", "首推征兆": (e.get("recommendations") or [{}])[0].get("symptom", ""),
            "理由": (e.get("rationale") or "")[:60],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    ents = [(e["round"], e["entropy_bits"]) for e in state.inquiry_log
            if e.get("action") == "call_tool" and e.get("entropy_bits") is not None]
    if len(ents) >= 2:
        st.caption("后验熵随归因轮次变化")
        st.line_chart([h for _, h in ents])


def _render_ett_forecast(result: Dict[str, Any]) -> None:
    if result.get("status") != "ok":
        st.json(result)
        return
    hist = result.get("history_summary", {})
    fc = result.get("forecast", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("历史油温均值", f"{hist.get('ot_mean', 'N/A')}℃")
    c2.metric("预测均值", f"{fc.get('mean_predicted_ot', 'N/A')}℃")
    c3.metric("拟合 R²", f"{hist.get('r_squared', 'N/A')}")

    values = fc.get("values")
    if isinstance(values, list) and values:
        st.markdown("**未来油温预测曲线：**")
        st.line_chart(values)
    if result.get("text_report"):
        with st.expander("完整预测报告"):
            st.markdown(result["text_report"])


def _render_timeseries(result: Dict[str, Any]) -> None:
    if result.get("status") != "ok":
        st.json(result)
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("均值", f"{result.get('mean', 0):.2f}")
    c2.metric("标准差", f"{result.get('std', 0):.2f}")
    c3.metric("异常点数", len(result.get("anomaly_indices", [])))
    series = result.get("series")
    if isinstance(series, list) and series:
        st.markdown("**油温时序（最近窗口）：**")
        st.line_chart(series)


def _render_rag(result: Any) -> None:
    if not isinstance(result, list) or not result:
        st.info("未检索到相关知识。")
        return
    for i, item in enumerate(result, 1):
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        title = meta.get("title") or meta.get("doc_name") or f"片段 {i}"
        score = item.get("score")
        score_txt = f"（score={score:.3f}）" if isinstance(score, (int, float)) else ""
        sec = meta.get("section_path")
        sec_txt = f"　·　{' > '.join(sec)}" if isinstance(sec, list) and sec else ""
        tags = []
        if item.get("reflection_score") is not None:
            tags.append(f"反思评分 {item['reflection_score']}/3")
        if item.get("expanded_from"):
            tags.append("邻块补召回")
        if item.get("from_rewrite"):
            tags.append("改写重检索")
        ch = item.get("channels")
        if isinstance(ch, dict) and ch:
            tags.append("命中通道 " + "/".join(sorted(ch)))
        tag_txt = f"　`{'` `'.join(tags)}`" if tags else ""
        st.markdown(f"**{i}. {title}** {score_txt}{sec_txt}{tag_txt}")
        st.caption(item.get("text", ""))


def _render_kg(state: AgentState) -> None:
    """图谱链路可视化：把 kg_search 返回的 paths 画成 Graphviz 有向图，并列出每条边的原文证据。"""
    calls = [c for c in (state.tool_calls or []) if c.get("tool") == "kg_search" and c.get("success")]
    if not calls:
        disabled = any(c.get("tool") == "kg_search" and c.get("error_code") == "tool_disabled" for c in (state.tool_calls or []))
        st.info("当前模式未开放图谱工具（Planner 越权调用已被拦截）。" if disabled else "本次未调用图谱工具。")
        return
    REL_CN = {"CAUSES": "导致", "PRODUCES": "产生", "INDICATES": "指示", "LOCATED_IN": "位于",
              "DETECTED_BY": "检测", "TREATED_BY": "处理", "SPECIFIED_IN": "依据"}
    for call in calls:
        res = call.get("result") or {}
        st.markdown(f"**查询：** `{(call.get('raw_arguments') or {}).get('query', '')}`　匹配实体："
                    + "、".join(f"{e.get('name')}({e.get('type')})" for e in res.get("matched_entities", [])[:3]))
        paths = res.get("paths") or []
        if not paths:
            st.warning(res.get("message") or "无关系路径。")
            continue
        lines = ["digraph G {", "rankdir=LR; node [shape=box, style=rounded, fontname=\"PingFang SC\"];",
                 "edge [fontname=\"PingFang SC\", fontsize=10];"]
        seen = set()
        for p in paths[:20]:
            path = p.get("path") or []
            for i in range(0, len(path) - 2, 2):
                s, r, t = path[i], path[i + 1], path[i + 2]
                key = (s, r, t)
                if key in seen:
                    continue
                seen.add(key)
                sn, tn = s.split(":", 1)[-1], t.split(":", 1)[-1]
                lines.append(f'"{sn}" -> "{tn}" [label="{REL_CN.get(r, r)}（{p.get("support_count", 0)}篇）"];')
        lines.append("}")
        try:
            st.graphviz_chart("\n".join(lines))
        except Exception:  # noqa: BLE001
            st.code("\n".join(lines))
        with st.expander("每条关系的文献证据", expanded=False):
            for p in paths[:20]:
                path = p.get("path") or []
                chain = " → ".join(x.split(":", 1)[-1] if i % 2 == 0 else REL_CN.get(x, x) for i, x in enumerate(path))
                st.markdown(f"- **{chain}**（支持 {p.get('support_count', 0)} 篇，置信 {p.get('confidence', 0)}）")
                if p.get("evidence"):
                    st.caption(p["evidence"])


def _render_reflection(state: AgentState) -> None:
    """反思评分展示：每轮评分明细、丢弃/保留、补召回与改写决策。"""
    logs = getattr(state, "reflection_log", None) or []
    if not logs:
        st.info("当前模式未启用反思模块。")
        return
    for log in logs:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("决策", str(log.get("decision")))
        c2.metric("输入块", log.get("input_count", 0))
        c3.metric("输出块", log.get("output_count", 0))
        c4.metric("耗时", f"{log.get('latency_ms', 0):.0f} ms")
        st.caption(f"评分器：{log.get('scorer')}　|　改写重检索：{'是' if log.get('rewrite_triggered') else '否'}"
                   + (f"　|　改写后的查询：{log['rewritten_query']}" if log.get("rewritten_query") else ""))
        for rnd in log.get("rounds") or []:
            title = (f"第 {rnd.get('round')} 轮：评分 {rnd.get('scored')} 块，保留 {rnd.get('kept')}，丢弃 {rnd.get('dropped')}"
                     + (f"，补召回 {rnd['expanded']} 块" if rnd.get("expanded") else "")
                     + (f"，重检索新增 {rnd['research_added']} 块" if rnd.get("research_added") is not None else ""))
            with st.expander(title, expanded=(rnd.get("round") == 1)):
                scores = rnd.get("scores") or []
                cids = rnd.get("chunk_ids") or [None] * len(scores)
                titles = rnd.get("titles") or [""] * len(scores)
                rows = [{"块": cid or f"#{i + 1}", "文献": t, "评分(0-3)": s}
                        for i, (cid, t, s) in enumerate(zip(cids, titles, scores))]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.json(rnd)
        kept_items = [it for it in (state.retrieved_knowledge or []) if it.get("reflection_score") is not None]
        if kept_items:
            st.markdown("**最终保留块（按反思评分排序）**")
            st.dataframe([{"块": it.get("chunk_id"), "评分": it.get("reflection_score"),
                           "来源": "邻块补召回" if it.get("expanded_from") else ("改写重检索" if it.get("from_rewrite") else "首轮检索"),
                           "文献": (it.get("metadata") or {}).get("title", "")} for it in kept_items],
                         use_container_width=True, hide_index=True)


def _render_trajectory(state: AgentState) -> None:
    """工具调用轨迹表：顺序、工具、关键参数、校验、结果状态、耗时。"""
    traj = getattr(state, "trajectory", None) or []
    if not traj:
        st.info("无工具调用轨迹。")
        return
    rows = []
    for t in traj:
        args = t.get("raw_arguments") or {}
        brief = json.dumps({k: (v if not isinstance(v, list) or len(v) <= 3 else f"[{len(v)} 项]") for k, v in args.items()},
                           ensure_ascii=False)
        rows.append({
            "#": t.get("call_index"),
            "工具": t.get("tool"),
            "参数": brief[:80] + ("…" if len(brief) > 80 else ""),
            "校验": "✅" if (t.get("validation") or {}).get("valid", True) else "❌",
            "执行": "✅" if t.get("exec_success") else "❌",
            "业务": "✅" if t.get("business_success") else "❌",
            "错误码": t.get("error_code") or "",
            "耗时(ms)": t.get("latency_ms"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


TRACE_ICON = {
    "llm_plan": "🧭",
    "thought": "💭",
    "tool_call": "🔧",
    "evaluation": "🔎",
    "reflection": "🪞",
}
AGENT_CN = {
    "planner": "规划智能体 Planner",
    "retriever": "检索智能体 Retriever",
    "reflection": "反思模块 Reflection",
    "generator": "生成智能体 Generator",
    "validator": "验证智能体 Validator",
}


def render_trace(state: AgentState) -> None:
    if not state.reasoning_trace:
        st.info("无推理轨迹记录。")
        return
    for i, item in enumerate(state.reasoning_trace, 1):
        agent = item.get("agent", "")
        ttype = item.get("type", "")
        icon = TRACE_ICON.get(ttype, "•")
        title = f"{i}. {icon} {AGENT_CN.get(agent, agent)} · {ttype}"
        with st.expander(title, expanded=False):
            content = item.get("content")
            if ttype == "thought" and isinstance(content, str):
                st.write(content)
            elif isinstance(content, (dict, list)):
                st.json(content)
            else:
                st.write(content)


# ══════════════════════════════════════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="变压器故障诊断多智能体系统",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ 变压器故障诊断多智能体系统")
st.caption("Planner → Retriever → Generator → Validator　|　基于 LangGraph 的图式编排与迭代优化")

# ── 侧边栏：配置 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 运行配置")
    config = load_config()
    from src.graph.system_modes import SYSTEM_MODES, MODE_ORDER, resolve_mode

    enable_llm = st.toggle(
        "启用 LLM",
        value=True,
        help="关闭后各智能体将使用模板/规则降级，可离线演示流程连通性。",
    )
    _ft_ready = isinstance(config.get("llms", {}).get("planner_finetuned"), dict)
    system_mode_key = st.selectbox(
        "系统模式（P6 五级对比）",
        options=MODE_ORDER,
        index=len(MODE_ORDER) - 1,
        format_func=lambda k: SYSTEM_MODES[k].label,
        help="每级只在上一级基础上打开一个能力：无RAG → 朴素RAG → +规划微调 → +图谱 → +反思。",
    )
    _spec = SYSTEM_MODES[system_mode_key]
    st.caption(_spec.description)
    st.caption("可用工具：" + "、".join(_spec.allowed_tools)
               + f"　|　检索：{_spec.rag_mode or '关闭'}　|　Planner：{_spec.planner_mode}"
               + ("（未配置微调模型，回退基线）" if _spec.planner_mode == "finetuned" and not _ft_ready else "")
               + f"　|　反思：{_spec.reflection}")

    with st.expander("高级：覆盖模式子开关", expanded=False):
        kb_mode = st.radio(
            "知识库来源",
            options=["local_kb", "mock"],
            index=0,
            format_func=lambda x: {"local_kb": "本地文献库（182 篇，多层索引）",
                                   "mock": "合成案例 mock（仅演示）"}[x],
            help="local_kb 由 data/kb 离线构建，不依赖网络；mock 为程序合成案例，不得用于评测。",
        )
        max_iter = st.slider("最大迭代轮次", 1, 5, value=config.get("workflow", {}).get("max_iterations", 3))
        _pm_opts = ["（按模式）", "baseline", "finetuned"]
        _pm_pick = st.selectbox("Planner 模型覆盖", _pm_opts, index=0,
                                help="finetuned 需在 config.yaml 的 llms.planner_finetuned 配置微调模型接口；未配置时自动回退 baseline。")
        _rf_pick = st.selectbox("反思模块覆盖", ["（按模式）", "off", "lexical"], index=0,
                                help="lexical 为离线词法评分器（阈值 3 分）；LLM 评分需接口。")
    _spec = resolve_mode(system_mode_key,
                         planner_mode=None if _pm_pick == "（按模式）" else _pm_pick,
                         reflection=None if _rf_pick == "（按模式）" else _rf_pick)
    planner_mode = _spec.planner_mode
    reflection_mode = _spec.reflection if _spec.reflection in ("off", "lexical") else "lexical"
    allowed_tools = list(_spec.allowed_tools)
    rag_mode = _spec.rag_mode

    st.divider()
    st.subheader("🎯 主动规划（B-4）")
    _wf = config.get("workflow", {}) or {}
    planner_strategy = st.radio(
        "Planner 策略", options=["free", "active"],
        index=["free", "active"].index(_wf.get("planner_strategy", "free")),
        format_func=lambda x: {"free": "free：一次规划直接结论", "active": "active：按信息增益决定追问 / 调工具 / 结论"}[x],
        horizontal=False,
    )
    planner_decision = "llm"
    max_inquiry_rounds = int(_wf.get("max_inquiry_rounds", 3))
    if planner_strategy == "active":
        planner_decision = st.radio(
            "动作决策来源", options=["llm", "eig"],
            index=["llm", "eig"].index(_wf.get("planner_decision", "llm")),
            format_func=lambda x: {"llm": "LLM 决策（读 uncertainty 块，失败回退 EIG）", "eig": "EIG 规则（不调 LLM，问 VoI 最高征兆）"}[x],
        )
        max_inquiry_rounds = st.slider("最大追问轮数 K", 1, 5, value=max_inquiry_rounds)
    attribution_mode = st.selectbox(
        "归因参数", options=["calibrated", "expert"],
        index=["calibrated", "expert"].index(_wf.get("attribution_mode", "calibrated")),
        format_func=lambda x: {"calibrated": "calibrated：数据学习 + 温度缩放（B-1）", "expert": "expert：专家默认 CPT"}[x],
    )

    st.divider()
    st.subheader("🧪 DGA 油色谱（可选）")
    st.caption("填写后将注入故障归因引擎（单位 μL/L）")
    use_dga = st.checkbox("提供 DGA 数据", value=True)
    dga: Dict[str, Any] = {}
    if use_dga:
        st.caption("勾选「未检测」的气体将作为缺失值传入（用于演示信息不足 → 追问）")
        _defaults = {"H2": 105.0, "CH4": 160.0, "C2H2": 1.7, "C2H4": 375.0, "C2H6": 40.0}
        _labels = {"H2": "H₂", "CH4": "CH₄", "C2H2": "C₂H₂", "C2H4": "C₂H₄", "C2H6": "C₂H₆"}
        _steps = {"H2": 1.0, "CH4": 1.0, "C2H2": 0.1, "C2H4": 1.0, "C2H6": 1.0}
        for g in ("H2", "CH4", "C2H2", "C2H4", "C2H6"):
            ca, cb = st.columns([3, 1])
            val = ca.number_input(_labels[g], min_value=0.0, value=_defaults[g], step=_steps[g], key=f"dga_{g}")
            miss = cb.checkbox("未检测", key=f"miss_{g}")
            dga[g] = None if miss else val

    st.divider()
    st.subheader("🏷️ 设备上下文（可选）")
    device_id = st.text_input("设备编号", value="TR-0050")
    voltage = st.number_input("电压等级 (kV)", min_value=0, value=220, step=1)

    st.divider()
    model_name = config.get("llms", {}).get("default", {}).get("model_name", "—")
    st.caption(f"默认模型：`{model_name}`")


# ── 主输入区 ──────────────────────────────────────────────────────────────────
EXAMPLES = {
    "—（手动输入）—": "",
    "案例1 过载过热（DGA 归因 + 规程）": "220kV主变（SFSZ-240000/220，运行6年）油色谱检测：H2=105.0, CH4=160.3, C2H2=1.7, C2H4=374.9 μL/L，请诊断故障类型并结合规程给出处理建议。",
    "案例2 高能放电（含安全要求）": "110kV主变油色谱异常：C2H2=45, H2=680, C2H4=230 μL/L，并伴随轻瓦斯报警，请判断故障性质并给出现场处理与安全措施。",
    "案例3 图谱多跳推理（部位 + 处理）": "铁心多点接地会导致什么后果？一般发生在哪个部件、如何检测和处理？",
    "案例4 油温趋势预测": "请基于 ETTh1 数据集预测该变压器未来 24 小时的油温走势，并判断是否存在过热风险。",
    "案例5 信息不足需追问": "帮我看看这台变压器有没有问题。",
    "案例6 主动追问 A（缺乙炔，放电 vs 过热）": "110kV主变本次色谱只测到 H2=180, CH4=95, C2H4=60 μL/L（乙炔、乙烷未出结果），现场反映近期负载偏高，请判断故障类型；如需补充信息请直接问我。",
    "案例7 主动追问 B（仅两种气体，需多轮确认）": "这台主变只拿到 H2=420 和 C2H2=38 μL/L 两个数，其余气体还没测，请先给出可能的故障并告诉我还需要补什么检测。",
}
# 追问演示案例对应的 DGA 缺失设置（选中案例后自动勾选「未检测」）
EXAMPLE_DGA = {
    "案例6 主动追问 A（缺乙炔，放电 vs 过热）": {"H2": 180.0, "CH4": 95.0, "C2H2": None, "C2H4": 60.0, "C2H6": None},
    "案例7 主动追问 B（仅两种气体，需多轮确认）": {"H2": 420.0, "CH4": None, "C2H2": 38.0, "C2H4": None, "C2H6": None},
}

example = st.selectbox("示例问题", list(EXAMPLES.keys()))
default_query = EXAMPLES[example]
user_query = st.text_area(
    "诊断问题",
    value=default_query,
    height=120,
    placeholder="例如：220kV主变油色谱检测 H2=105, CH4=160, C2H2=1.7, C2H4=375 μL/L，请诊断故障类型并给出处理建议。",
)

run_clicked = st.button("🚀 开始诊断", type="primary", use_container_width=True)


# ── 执行 ──────────────────────────────────────────────────────────────────────
if run_clicked:
    if not user_query.strip():
        st.error("请输入诊断问题。")
        st.stop()

    # 根据开关动态调整配置（禁用 LLM 时清空 api_key）
    run_cfg = json.loads(json.dumps(config))  # 深拷贝
    run_cfg.setdefault("workflow", {})["max_iterations"] = max_iter
    if not enable_llm:
        for v in run_cfg.get("llms", {}).values():
            if isinstance(v, dict):
                v["api_key"] = ""
                v["api_key_env"] = "___DISABLED___"

    context: Dict[str, Any] = {"device_id": device_id, "voltage_level_kv": voltage}
    _dga_for_run: Dict[str, Any] | None = None
    if example in EXAMPLE_DGA:
        _dga_for_run = dict(EXAMPLE_DGA[example])
        st.caption("已按演示案例设置 DGA 缺失项：" + ", ".join(f"{k}={'未检测' if v is None else v}" for k, v in _dga_for_run.items()))
    elif use_dga and dga:
        _dga_for_run = dict(dga)
    if _dga_for_run:
        # 显式进入 Planner 输入：模型必须能看到前端填写的数据，才能被要求填对 dga_data
        context["dga"] = _dga_for_run
    mcp = build_mcp(kb_mode=kb_mode,
                    user_dga={k: v for k, v in (_dga_for_run or {}).items() if v is not None} or None,
                    rag_mode=rag_mode)

    with st.spinner(f"多智能体协同诊断中…（{_spec.label}）"):
        final, elapsed, err = run_workflow(user_query, context, run_cfg, mcp,
                                           reflection_mode=reflection_mode, planner_mode=planner_mode,
                                           allowed_tools=allowed_tools,
                                           planner_strategy=planner_strategy, planner_decision=planner_decision,
                                           max_inquiry_rounds=max_inquiry_rounds, attribution_mode=attribution_mode)

    if err:
        st.error("工作流执行失败：")
        st.code(err)
        st.stop()

    st.session_state["final"] = final
    st.session_state["elapsed"] = elapsed
    st.session_state["system_mode"] = _spec.to_dict()


# ── 追问交互（active 策略遇 ask 中断）────────────────────────────────────────
_pending: AgentState | None = st.session_state.get("final")
if _pending is not None and _pending.pending_questions and _pending.final_answer is None:
    q = _pending.pending_questions[0]
    st.divider()
    st.subheader(f"❓ 系统追问（第 {_pending.inquiry_rounds + 1}/{_pending.max_inquiry_rounds} 轮）")
    st.markdown(f"**{q.get('question')}**")
    st.caption(f"为什么问：{q.get('rationale', '')}　|　征兆 `{q.get('symptom')}`"
               + (f"　|　EIG {q['eig']:.3f} bit，成本 {q.get('cost', 0):.0f}" if q.get("eig") is not None else ""))
    unc_now = (_pending.context or {}).get("uncertainty") or {}
    if unc_now:
        with st.expander("当前后验与候选征兆", expanded=False):
            _render_uncertainty(unc_now)
    ca, cb, cc = st.columns(3)
    ans: bool | None = None
    clicked = False
    if ca.button("✅ 是 / 存在该现象", use_container_width=True):
        ans, clicked = True, True
    if cb.button("❌ 否 / 已排除", use_container_width=True):
        ans, clicked = False, True
    if cc.button("🤷 不知道 / 无法提供", use_container_width=True):
        ans, clicked = None, True
    if clicked:
        with st.spinner("已纳入回答，重新归因中…"):
            final, elapsed2, err = resume_workflow(_pending, q["symptom"], ans)
        if err:
            st.error("续跑失败：")
            st.code(err)
            st.stop()
        st.session_state["final"] = final
        st.session_state["elapsed"] = st.session_state.get("elapsed", 0.0) + elapsed2
        st.rerun()


# ── 结果展示 ──────────────────────────────────────────────────────────────────
final: AgentState | None = st.session_state.get("final")
if final is not None:
    elapsed = st.session_state.get("elapsed", 0.0)
    st.divider()
    render_overview(final, elapsed)

    tabs = st.tabs([
        "📋 诊断结论",
        "✅ 质量验证",
        "🧭 规划计划",
        "🎯 追问轨迹",
        "🔧 工具调用",
        "📚 检索知识",
        "🕸️ 图谱链路",
        "🪞 反思评分",
        "🧩 推理轨迹",
        "🐞 错误",
    ])

    with tabs[0]:
        if final.pending_questions and final.final_answer is None:
            st.info("系统仍在追问中，请在上方回答后查看结论。")
        st.markdown(final.final_answer or final.draft_answer or "（无结果）")

    with tabs[1]:
        val = _extract_validation(final)
        if val:
            label, _ = VERDICT_STYLE.get(val.get("verdict"), (val.get("verdict"), "gray"))
            st.markdown(f"**结论：** {label}　**评分：** {val.get('score')}/10")
            if val.get("summary"):
                st.markdown(f"**总体说明：** {val['summary']}")
            if val.get("strengths"):
                st.markdown("**优点**")
                for s in val["strengths"]:
                    st.markdown(f"- {s}")
            if val.get("weaknesses"):
                st.markdown("**问题**")
                for w in val["weaknesses"]:
                    st.markdown(f"- {w}")
            if val.get("improvement_suggestions"):
                st.markdown("**改进建议**")
                for s in val["improvement_suggestions"]:
                    st.markdown(f"- {s}")
        elif final.validation_report:
            st.text(final.validation_report)
        else:
            st.info("无验证报告。")

    with tabs[2]:
        st.markdown(f"```\n{final.plan or '（LLM 未启用或未生成计划）'}\n```")

    with tabs[3]:
        _render_inquiry(final)

    with tabs[4]:
        _render_trajectory(final)
        st.divider()
        render_tool_calls(final)

    with tabs[5]:
        _render_rag(final.retrieved_knowledge)

    with tabs[6]:
        _render_kg(final)

    with tabs[7]:
        _render_reflection(final)

    with tabs[8]:
        render_trace(final)

    with tabs[9]:
        if final.errors:
            for e in final.errors:
                st.error(f"[{e.get('agent', '?')}{('/' + e['tool']) if e.get('tool') else ''}] {e.get('error', '')}")
        else:
            st.success("无错误记录。")
else:
    st.info("在上方填写诊断问题与 DGA 数据，点击「开始诊断」运行多智能体工作流。")
