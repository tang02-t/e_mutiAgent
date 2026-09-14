from typing import Dict, Any, List

from src.graph.state import AgentState
from src.utils.config import get_llm_config
from src.utils.logging import get_logger
from src.utils.llm import LLMClient, LLMConfig
from src.utils.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_REVISION_SYSTEM_PROMPT,
    render_generator_user,
    render_generator_revision_user,
)


logger = get_logger(__name__)


class GeneratorAgent:
    """
    对齐生成智能体：
    - 根据检索到的知识和时序分析结果，生成面向一线的诊断与操作建议
    - 支持 Validator 触发重生成，根据改进建议优化诊断草案
    - 优先使用 LLM 生成，若 LLM 未启用则使用模板填充
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        model_cfg = get_llm_config(config, "generator")
        self.llm = LLMClient(
            LLMConfig(
                provider=model_cfg.get("provider", "openai"),
                model_name=model_cfg.get("model_name", ""),
                temperature=float(model_cfg.get("temperature", 0.1)),
                max_tokens=int(model_cfg.get("max_tokens", 2048)),
                base_url=model_cfg.get("base_url", "https://api.openai.com/v1"),
                api_key=model_cfg.get("api_key", ""),
                api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
                enable_thinking=model_cfg.get("enable_thinking", False),
                timeout=float(model_cfg.get("timeout", 60.0)),
                max_retries=int(model_cfg.get("max_retries", 2)),
            )
        )

    def _format_kb_snippets(self, kb_results: List[Dict[str, Any]]) -> str:
        if not kb_results:
            return "（知识库中未检索到高度相关的案例，以下结论基于通用经验。）"

        lines = []
        for idx, item in enumerate(kb_results[:3], start=1):
            # 兼容不同检索后端的返回结构：
            # - 旧本地 txt 版：{path, snippet, score}
            # - Milvus/Zilliz 版：{id, text, score, child, parent, metadata}
            # - 并行检索聚合后的扁平结构
            source = item.get("path") or item.get("id") or ""
            text = item.get("snippet") or item.get("text") or ""

            # 父子分块模式：优先使用子块文本
            if "child" in item and item["child"]:
                text = item["child"].get("text", text)
            if "metadata" in item and item["metadata"]:
                source = item["metadata"].get("doc_name") or item["metadata"].get("title") or source

            score = item.get("score")
            score_text = f"（score={score:.4f}）" if isinstance(score, (int, float)) else ""
            lines.append(f"{idx}. 来源：{source} {score_text}".rstrip())
            lines.append(f"   片段：{str(text)[:200]}...")
        return "\n".join(lines)

    def _extract_tool_results(self, state: AgentState) -> tuple[str, str, str, str]:
        """从 tool_calls 中提取各类工具的结果文本。"""
        ts_calls = [c for c in state.tool_calls if c.get("tool") == "timeseries_anomaly"]
        ts_result = ts_calls[0]["result"] if ts_calls else None

        attr_calls = [c for c in state.tool_calls if c.get("tool") == "fault_attribution"]
        attr_result = attr_calls[0]["result"] if attr_calls else None

        forecast_calls = [c for c in state.tool_calls if c.get("tool") == "ett_forecast"]
        forecast_result = forecast_calls[0]["result"] if forecast_calls else None

        kb_text = self._format_kb_snippets(state.retrieved_knowledge)

        ts_text = ""
        if ts_result and ts_result.get("status") == "ok":
            anomaly_cnt = len(ts_result.get("anomaly_indices", []))
            ts_text = (
                f"- 信号平均值约为 {ts_result.get('mean'):.2f}，标准差约为 {ts_result.get('std'):.2f}。"
                f"\n- 检测到 {anomaly_cnt} 个疑似异常点，可能对应短时冲击或工况突变。"
            )
        else:
            ts_text = "- 当前未获得有效时序分析结果。"

        attribution_text = ""
        if attr_result and attr_result.get("status") == "ok":
            ranked = attr_result.get("fault_ranking", [])
            primary = attr_result.get("primary_fault_name") or "未确定"
            primary_prob = attr_result.get("primary_probability", 0)
            dga_analysis = attr_result.get("dga_analysis") or {}
            interpretation = dga_analysis.get("interpretation", "") or ""

            top_faults = "\n".join(
                f"  {r['rank']}. {r['fault_name']}（{r['probability_pct']}，{r['severity']}）"
                for r in ranked[:3]
            ) if ranked else "无"

            attribution_text = (
                f"- 主要故障推断：{primary}（后验概率 {primary_prob*100:.1f}%）\n"
                f"- 故障概率排序（贝叶斯网络 × DGA规则融合）：\n{top_faults}\n"
                f"- DGA特征解释：{interpretation}"
            )
        else:
            attribution_text = "- 当前无故障归因分析结果。"

        forecast_text = ""
        if forecast_result and forecast_result.get("status") == "ok":
            ds = forecast_result.get("dataset", "")
            freq = forecast_result.get("freq", "")
            hist = forecast_result.get("history_summary", {})
            fc = forecast_result.get("forecast", {})
            anomalies = forecast_result.get("anomalies", {})

            ot_mean = hist.get("ot_mean", "N/A")
            ot_std = hist.get("ot_std", "N/A")
            pred_mean = fc.get("mean_predicted_ot", "N/A")
            pred_min = fc.get("min_predicted_ot", "N/A")
            pred_max = fc.get("max_predicted_ot", "N/A")
            r2 = hist.get("r_squared", "N/A")
            anom_cnt = anomalies.get("count", 0)
            horizon = forecast_result.get("horizon", 0)

            forecast_text = (
                f"- 数据集：{ds}（{freq}采样），回看窗口 {forecast_result.get('lookback', '?')} 点，预测步数 {horizon}\n"
                f"- 历史油温统计：均值 {ot_mean}℃，标准差 {ot_std}℃，"
                f"模型拟合 R²={r2}\n"
                f"- 未来 {horizon} 步预测：均值 {pred_mean}℃，范围 {pred_min}℃~{pred_max}℃\n"
                f"- 3σ异常检测：{'发现 ' + str(anom_cnt) + ' 个异常点，请关注油温突变风险' if anom_cnt > 0 else '未检测到异常，油温运行在正常区间'}"
            )
        else:
            forecast_text = "- 当前无有效油温预测分析结果。"
            if forecast_result and forecast_result.get("status") == "error":
                forecast_text = f"- 油温预测分析失败：{forecast_result.get('message', '未知错误')}"

        return kb_text, ts_text, attribution_text, forecast_text

    def _llm_revision(self, state: AgentState, revision_feedback: str) -> str:
        """调用 LLM 根据 Validator 反馈重新生成诊断草案。"""
        kb_text, ts_text, attribution_text, forecast_text = self._extract_tool_results(state)
        previous_draft = state.draft_answer or ""

        system_msg = GENERATOR_REVISION_SYSTEM_PROMPT()
        user_msg = render_generator_revision_user(
            query=state.user_query,
            kb_text=kb_text,
            ts_text=ts_text,
            attribution_text=attribution_text,
            forecast_text=forecast_text,
            revision_feedback=revision_feedback,
            previous_draft=previous_draft,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        choice = self.llm.chat(messages)
        return choice.get("content", "") if isinstance(choice, dict) else ""

    def _llm_generate(self, state: AgentState) -> str:
        """调用 LLM 生成诊断草案（首次生成）。"""
        kb_text, ts_text, attribution_text, forecast_text = self._extract_tool_results(state)

        system_msg = GENERATOR_SYSTEM_PROMPT()
        user_msg = render_generator_user(
            query=state.user_query,
            kb_text=kb_text,
            ts_text=ts_text,
            attribution_text=attribution_text,
            forecast_text=forecast_text,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        choice = self.llm.chat(messages)
        return choice.get("content", "") if isinstance(choice, dict) else ""

    def _template_generate(self, state: AgentState) -> str:
        """使用模板填充生成诊断草案。"""
        kb = state.retrieved_knowledge
        ts_calls = [c for c in state.tool_calls if c.get("tool") == "timeseries_anomaly"]
        ts_result = ts_calls[0]["result"] if ts_calls else None

        kb_text = self._format_kb_snippets(kb)

        draft_parts: List[str] = [
            GENERATOR_SYSTEM_PROMPT(),
            "",
            "用户问题：" + state.user_query,
            "",
            "一、相关案例与知识片段：",
            kb_text,
            "",
            "二、时序信号分析要点：",
        ]

        if ts_result and ts_result.get("status") == "ok":
            anomaly_cnt = len(ts_result.get("anomaly_indices", []))
            draft_parts.append(
                f"- 信号平均值约为 {ts_result.get('mean'):.2f}，标准差约为 {ts_result.get('std'):.2f}。"
            )
            draft_parts.append(f"- 检测到 {anomaly_cnt} 个疑似异常点，可能对应负载突变或设备异常。")
        else:
            draft_parts.append("- 当前未获得有效时序分析结果。")

        draft_parts.extend(
            [
                "",
                "三、诊断结论与处理建议（示例）：",
                "1. 【安全优先】在进行任何检修前，确认相关保护装置（差动保护、瓦斯保护）处于正常状态。",
                "2. 建议取油样进行 DGA（溶解气体分析），根据 GB/T 7252 标准判定故障类型。",
                "3. 结合红外热成像检测套管及引线接头温升，关注是否超过额定温升限值（油浸式 A 级绝缘绕组温升≤65K）。",
                "4. 检查冷却系统（油泵、风扇）运行状态，确认油温是否正常。",
                "5. 如有必要申请停电进行绕组直流电阻、绝缘电阻、变比等电气试验。",
            ]
        )

        return "\n".join(draft_parts)

    def run(self, state: AgentState, revision_feedback: str = "") -> AgentState:
        """
        执行生成任务。

        参数：
        - state: 当前状态
        - revision_feedback: Validator 提供的改进建议（非空时触发重生成模式）

        备注：当 revision_feedback 为空时，方法会自动从 state.validation_report 中提取改进建议。
        """
        state.iteration += 1

        # 自动从 state.revision_feedback 中获取改进建议（由 Validator 设置）
        if not revision_feedback and state.revision_feedback and state.iteration > 1:
            revision_feedback = state.revision_feedback
            # 清除已使用的 feedback，避免重复使用
            state.revision_feedback = None

        is_revision = bool(revision_feedback)

        logger.info(
            "[yellow]Generator Input:[/yellow] iteration=%d, revision=%s, user_query=%s, retrieved_knowledge=%d items, tool_calls=%d",
            state.iteration,
            "yes" if revision_feedback else "no",
            state.user_query,
            len(state.retrieved_knowledge),
            len(state.tool_calls)
        )

        if self.llm.enabled:
            try:
                if is_revision:
                    logger.info("Generator revision mode (LLM): applying Validator feedback.")
                    draft = self._llm_revision(state, revision_feedback)
                else:
                    logger.info("Generator initial mode (LLM): generating draft answer.")
                    draft = self._llm_generate(state)

                if draft:
                    state.draft_answer = draft
                    logger.info("Generator %s succeeded, draft length=%d chars.",
                                "revision" if is_revision else "generation", len(draft))
                else:
                    logger.warning("Generator LLM returned empty, falling back to template.")
                    state.errors.append({
                        "agent": "generator",
                        "error": "LLM 返回空内容，已降级到模板生成。",
                    })
                    state.draft_answer = self._template_generate(state)
            except Exception as exc:
                logger.warning("Generator LLM call failed: %s, falling back to template.", exc)
                state.errors.append({
                    "agent": "generator",
                    "error": f"LLM 调用失败：{exc}",
                })
                state.draft_answer = self._template_generate(state)
        else:
            logger.info("Generator LLM not enabled, using template.")
            state.draft_answer = self._template_generate(state)

        logger.info("[yellow]Generator Output:[/yellow] draft_answer length=%d chars", len(state.draft_answer))
        return state

