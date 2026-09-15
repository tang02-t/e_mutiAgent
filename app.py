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

import csv
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
from src.utils.config import load_config                     # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Mock 工具（无外部依赖，便于离线演示）
# ══════════════════════════════════════════════════════════════════════════════
TS_DIR = ROOT / "data/synthetic/timeseries"
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


def timeseries_anomaly_tool(signal=None) -> Dict[str, Any]:
    """
    对传入的数值序列做 3σ 异常检测。

    P0 修复：不再读取模拟设备文件 TR-0001.csv、不再使用示例序列；
    signal 缺失时直接返回业务错误，避免「参数错了也能成功」。
    """
    if not isinstance(signal, (list, tuple)) or len(signal) == 0:
        return {"status": "error", "message": "timeseries_anomaly 需要非空 signal 序列"}
    try:
        ot = [float(x) for x in signal]
    except (TypeError, ValueError) as exc:
        return {"status": "error", "message": f"signal 含非数值元素：{exc}"}
    if len(ot) < 3:
        return {"status": "error", "message": f"signal 长度 {len(ot)} 过短，至少需要 3 个点"}
    mean = sum(ot) / len(ot)
    var = sum((v - mean) ** 2 for v in ot) / max(len(ot) - 1, 1)
    std = var ** 0.5
    anomalies = [i for i, v in enumerate(ot) if std and abs(v - mean) > 3 * std]
    return {
        "status": "ok",
        "n": len(ot),
        "mean": mean,
        "std": std,
        "anomaly_indices": anomalies,
        "series": ot,
    }


def build_mcp(kb_mode: str, user_dga: Dict[str, Any] | None, rag_mode: str | None = "two_way") -> MCPClient:
    """
    构造 MCP 客户端并注册工具。
    kb_mode: local_kb | milvus | mock；rag_mode：local_kb 的检索方式 naive | two_way | two_way_rerank，
    None 表示当前系统模式不开放文献检索（注册空桩，越权调用得到 empty_result）。
    """
    mcp = MCPClient()

    # fault_attribution：用闭包注入前端填写的 DGA（当 plan 未带 dga_data 时生效）
    def fault_attribution_with_dga(dga_data=None, evidence=None, query: str = ""):
        if not dga_data and user_dga:
            dga_data = user_dga
        return fault_attribution(dga_data=dga_data, evidence=evidence, query=query)

    mcp.register_tool("fault_attribution", fault_attribution_with_dga)
    mcp.register_tool("timeseries_anomaly", timeseries_anomaly_tool)
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

    if kb_mode == "milvus":
        try:
            from src.tools.rag_engine import RAGEngine
            cfg = load_config()
            mv = cfg["knowledge_base"]["milvus"]
            engine = RAGEngine(
                uri=mv.get("uri"), token=mv.get("token"),
                host=mv.get("host"), port=mv.get("port"),
                collection_name=mv["collection"], text_field=mv["text_field"],
                vector_field=mv["vector_field"],
                top_k=cfg["knowledge_base"].get("top_k", 5),
                metric_type=mv.get("metric_type", "L2"), nprobe=mv.get("nprobe", 10),
            )
            mcp.register_tool("rag_search", engine.search)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Milvus 初始化失败，已回退到本地文献库：{exc}")
            kb_mode = "local_kb"
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
) -> tuple[AgentState | None, float, str | None]:
    """执行完整诊断工作流，返回 (final_state, elapsed, error)。"""
    from src.agents.planner import PlannerAgent
    from src.agents.retriever import RetrieverAgent
    from src.agents.generator import GeneratorAgent
    from src.agents.validator import ValidatorAgent

    state = AgentState(
        user_query=user_query,
        context=context,
        max_iterations=config.get("workflow", {}).get("max_iterations", 3),
    )

    planner = PlannerAgent(config, planner_mode=planner_mode, allowed_tools=allowed_tools)
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

    t0 = time.time()
    try:
        final = run_diagnosis_workflow(state, planner, retriever, generator, validator, reflector=reflector)
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
            options=["local_kb", "milvus", "mock"],
            index=0,
            format_func=lambda x: {"local_kb": "本地文献库（182 篇，多层索引）",
                                   "milvus": "Milvus 向量库（需 embedding 接口）",
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
    st.subheader("🧪 DGA 油色谱（可选）")
    st.caption("填写后将注入故障归因引擎（单位 μL/L）")
    use_dga = st.checkbox("提供 DGA 数据", value=True)
    dga: Dict[str, Any] = {}
    if use_dga:
        cda, cdb = st.columns(2)
        dga["H2"] = cda.number_input("H₂", min_value=0.0, value=105.0, step=1.0)
        dga["CH4"] = cdb.number_input("CH₄", min_value=0.0, value=160.0, step=1.0)
        dga["C2H2"] = cda.number_input("C₂H₂", min_value=0.0, value=1.7, step=0.1)
        dga["C2H4"] = cdb.number_input("C₂H₄", min_value=0.0, value=375.0, step=1.0)
        dga["C2H6"] = cda.number_input("C₂H₆", min_value=0.0, value=40.0, step=1.0)

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
    if use_dga and dga:
        # 显式进入 Planner 输入：模型必须能看到前端填写的数据，才能被要求填对 dga_data
        context["dga"] = dict(dga)
    mcp = build_mcp(kb_mode=kb_mode, user_dga=dga if use_dga else None, rag_mode=rag_mode)

    with st.spinner(f"多智能体协同诊断中…（{_spec.label}）"):
        final, elapsed, err = run_workflow(user_query, context, run_cfg, mcp,
                                           reflection_mode=reflection_mode, planner_mode=planner_mode,
                                           allowed_tools=allowed_tools)

    if err:
        st.error("工作流执行失败：")
        st.code(err)
        st.stop()

    st.session_state["final"] = final
    st.session_state["elapsed"] = elapsed
    st.session_state["system_mode"] = _spec.to_dict()


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
        "🔧 工具调用",
        "📚 检索知识",
        "🕸️ 图谱链路",
        "🪞 反思评分",
        "🧩 推理轨迹",
        "🐞 错误",
    ])

    with tabs[0]:
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
        _render_trajectory(final)
        st.divider()
        render_tool_calls(final)

    with tabs[4]:
        _render_rag(final.retrieved_knowledge)

    with tabs[5]:
        _render_kg(final)

    with tabs[6]:
        _render_reflection(final)

    with tabs[7]:
        render_trace(final)

    with tabs[8]:
        if final.errors:
            for e in final.errors:
                st.error(f"[{e.get('agent', '?')}{('/' + e['tool']) if e.get('tool') else ''}] {e.get('error', '')}")
        else:
            st.success("无错误记录。")
else:
    st.info("在上方填写诊断问题与 DGA 数据，点击「开始诊断」运行多智能体工作流。")
