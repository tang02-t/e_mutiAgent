from typing import Dict, Any, List, Optional, Tuple

from src.graph.state import AgentState
from src.utils.config import get_llm_config
from src.utils.logging import get_logger
from src.utils.llm import LLMClient, LLMConfig
from src.utils.json_parse import parse_llm_json
from src.utils.prompts import VALIDATOR_SYSTEM_PROMPT


logger = get_logger(__name__)


class ValidationResult:
    """Validator 结构化评估结果。"""

    def __init__(
        self,
        verdict: str,  # PASS / REVISION / FAIL
        score: int,
        strengths: List[str],
        weaknesses: List[str],
        improvement_suggestions: List[str],
        summary: str,
    ) -> None:
        self.verdict = verdict
        self.score = score
        self.strengths = strengths
        self.weaknesses = weaknesses
        self.improvement_suggestions = improvement_suggestions
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_suggestions": self.improvement_suggestions,
            "summary": self.summary,
        }

    def to_report(self) -> str:
        """生成可读的验证报告文本。"""
        lines = [
            VALIDATOR_SYSTEM_PROMPT(),
            "",
            f"评估结论：{self.verdict}（评分：{self.score}/10）",
            f"总体说明：{self.summary}",
            "",
        ]
        if self.strengths:
            lines.append("【优点】")
            lines.extend(f"- {s}" for s in self.strengths)
            lines.append("")
        if self.weaknesses:
            lines.append("【问题】")
            lines.extend(f"- {w}" for w in self.weaknesses)
            lines.append("")
        if self.improvement_suggestions:
            lines.append("【改进建议】")
            lines.extend(f"- {s}" for s in self.improvement_suggestions)
            lines.append("")
        return "\n".join(lines)

    def get_feedback_text(self) -> str:
        """获取用于 Generator 重生成的反馈文本。"""
        parts = []
        if self.weaknesses:
            parts.append("【发现的问题】")
            parts.extend(self.weaknesses)
            parts.append("")
        if self.improvement_suggestions:
            parts.append("【改进建议】")
            parts.extend(self.improvement_suggestions)
        return "\n".join(parts)


class ValidatorAgent:
    """
    质量验证智能体：
    - 对 Generator 的输出做结构化质量评估
    - 支持 PASS / REVISION / FAIL 三级判定
    - REVISION 时输出改进建议，触发 Generator 重生成
    - FAIL 时标记工作流终止
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        model_cfg = get_llm_config(config, "validator")
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

    def _parse_structured_response(self, content: str) -> Optional[ValidationResult]:
        """从 LLM 返回内容中解析结构化 JSON（复用统一的 parse_llm_json）。"""
        data = parse_llm_json(content)
        if not data or "verdict" not in data:
            logger.warning("Validator 无法从 LLM 输出解析出合法 JSON，原始内容前 200 字符: %s", (content or "")[:200])
            return None

        try:
            return ValidationResult(
                verdict=data.get("verdict", "FAIL"),
                score=int(data.get("score", 0)),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                improvement_suggestions=data.get("improvement_suggestions", []),
                summary=data.get("summary", ""),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Validator 解析 JSON 字段失败: %s", exc)
            return None

    def _rule_validate(self, answer: str) -> ValidationResult:
        """使用规则进行验证（LLM 不可用时的降级方案）。"""
        issues: List[str] = []

        if not answer.strip():
            issues.append("未生成有效诊断结论。")
            return ValidationResult(
                verdict="FAIL",
                score=0,
                strengths=[],
                weaknesses=issues,
                improvement_suggestions=["请重新生成诊断草案。"],
                summary="诊断草案为空，验证失败。",
            )

        content_length = len(answer)

        # 内容长度检查
        if content_length < 30:
            issues.append("诊断草案内容过短，信息不足。")

        # 关键词检查（适用于故障诊断类查询）
        has_safety = "安全" in answer or "保护" in answer or "注意" in answer
        has_suggestion = "建议" in answer or "处理" in answer or "步骤" in answer

        # 如果同时缺少安全提示和建议，则记录为问题
        if not has_safety and not has_suggestion:
            issues.append("未明确给出安全注意事项或操作建议。")
        elif not has_safety:
            issues.append("未明显提及安全或保护相关的注意事项。")
        elif not has_suggestion:
            issues.append("建议不够明确或缺少具体操作步骤。")

        # 计算评分（更宽松的评分标准）
        score = 10
        if content_length < 30:
            score -= 3
        if not has_safety:
            score -= 2
        if not has_suggestion:
            score -= 2
        if content_length < 100:
            score -= 1
        score = max(0, min(10, score))

        # 判定结论
        if not issues:
            verdict = "PASS"
            summary = "通过规则检查。"
        elif score >= 5:
            verdict = "REVISION"
            summary = "存在部分问题，建议改进。"
        else:
            verdict = "REVISION"  # 改为 REVISION 而非 FAIL，给 Generator 改进的机会
            summary = "存在一些问题，需要改进。"

        return ValidationResult(
            verdict=verdict,
            score=score,
            strengths=["内容基本完整。" if score >= 5 else "提供了部分内容。"],
            weaknesses=issues,
            improvement_suggestions=issues if issues else ["建议进一步完善诊断内容。"],
            summary=summary,
        )

    def _llm_validate(self, answer: str, revision_context: str = "") -> ValidationResult:
        """使用 LLM 进行结构化评估。"""
        system_msg = VALIDATOR_SYSTEM_PROMPT()

        if revision_context:
            user_msg = f"""请对以下诊断草案进行质量评估：

【诊断草案】
{answer}

【上一轮改进建议】
{revision_context}

请以 JSON 格式输出评估结果："""
        else:
            user_msg = f"""请对以下诊断草案进行质量评估：

【诊断草案】
{answer}

请以 JSON 格式输出评估结果："""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        choice = self.llm.chat(messages)
        content = choice.get("content", "") if isinstance(choice, dict) else ""

        if not content:
            logger.warning("Validator LLM returned empty.")
            return self._rule_validate(answer)

        result = self._parse_structured_response(content)
        if result is None:
            logger.warning("Validator failed to parse LLM response, falling back to rule-based.")
            return self._rule_validate(answer)

        return result

    def run(self, state: AgentState) -> AgentState:
        """
        执行验证任务，返回状态中包含：
        - validation_report: 可读的验证报告
        - validation_verdict: PASS / REVISION / FAIL
        - next_node: 路由目标（generator 表示需要重生成，END 表示结束）
        """
        answer = state.draft_answer or ""
        # 仅回传上一轮的"改进建议"，而非整份验证报告（避免把 system prompt 也喂回 LLM）
        revision_context = state.last_improvement_suggestions or ""

        logger.info(
            "[yellow]Validator Input:[/yellow] iteration=%d, revision_context=%s, draft length=%d chars",
            state.iteration,
            "yes" if revision_context else "no",
            len(answer)
        )

        if self.llm.enabled:
            try:
                logger.info("Validator using LLM for structured evaluation.")
                result = self._llm_validate(answer, revision_context if state.iteration > 1 else "")

                # 当 LLM 返回 score=0 时，使用规则验证作为补充
                if result.score == 0:
                    logger.warning("LLM returned score=0, using rule-based validation as supplement.")
                    rule_result = self._rule_validate(answer)
                    # 合并结果：优先使用 LLM 结果，但使用规则验证的评分
                    if rule_result.score > result.score:
                        logger.info("Rule-based score (%d) higher than LLM score (%d), using rule-based result.",
                                   rule_result.score, result.score)
                        result = rule_result
            except Exception as exc:
                logger.warning("Validator LLM call failed: %s, falling back to rule-based.", exc)
                result = self._rule_validate(answer)
        else:
            logger.info("Validator LLM not enabled, using rule-based validation.")
            result = self._rule_validate(answer)

        # 填充状态
        state.validation_report = result.to_report()
        state.validation_verdict = result.verdict
        # 记录本轮改进建议，供下一轮 Validator 评估对比使用
        state.last_improvement_suggestions = "\n".join(result.improvement_suggestions)

        # 路由决策
        if result.verdict == "PASS":
            state.next_node = "END"
            state.final_answer = answer
            logger.info("Validator: PASS (score=%d), routing to END.", result.score)
        elif result.verdict == "REVISION":
            # 保存改进建议到 state，供 Generator 下次迭代使用
            state.revision_feedback = result.get_feedback_text()

            # 检查是否超过最大迭代次数
            if state.iteration >= state.max_iterations:
                logger.warning(
                    "Validator: REVISION but max iterations (%d) reached. "
                    "Accepting current draft with warning.",
                    state.max_iterations
                )
                state.next_node = "END"
                state.final_answer = (
                    answer
                    + "\n\n【系统提示】已达到最大改进次数上限，请结合现场经验审慎采纳以上建议。"
                )
                state.validation_report += "\n\n【系统提示】已达到最大改进次数上限。"
            else:
                # 将改进建议存入 state 中供 Generator 使用
                state.next_node = "generator"
                logger.info(
                    "Validator: REVISION (score=%d), routing to generator for revision %d/%d.",
                    result.score, state.iteration, state.max_iterations
                )
        else:  # FAIL
            state.next_node = "END"
            state.errors.append({
                "agent": "validator",
                "error": f"诊断草案质量不达标（score={result.score}），验证判定 FAIL。",
            })
            state.final_answer = (
                "【系统诊断失败】当前诊断草案质量不达标，无法给出有效建议。"
                "\n\n建议："
                + "\n".join(f"- {s}" for s in result.improvement_suggestions)
                + "\n\n请补充更多信息后重新提交诊断请求。"
            )
            logger.warning("Validator: FAIL (score=%d), routing to END.", result.score)

        # 记录推理轨迹
        state.reasoning_trace.append({
            "agent": "validator",
            "type": "evaluation",
            "content": result.to_dict(),
        })

        logger.info(
            "[yellow]Validator Output:[/yellow] verdict=%s, score=%d, next_node=%s",
            result.verdict, result.score, state.next_node
        )
        return state
