from typing import Dict, Any, List, Optional, Tuple

from src.graph.state import AgentState
from src.utils.config import get_llm_config
from src.utils.logging import get_logger
from src.utils.llm import LLMClient, LLMConfig
from src.utils.json_parse import parse_llm_json
from src.utils.prompts import VALIDATOR_SYSTEM_PROMPT


logger = get_logger(__name__)


class ValidationResult:
    """Validator 结构化评估结果（v2：可选携带声明级核查字段，旧字段全部保留）。"""

    def __init__(
        self,
        verdict: str,  # PASS / REVISION / FAIL / ABSTAIN
        score: int,
        strengths: List[str],
        weaknesses: List[str],
        improvement_suggestions: List[str],
        summary: str,
        claim_verdicts: Optional[List[Dict[str, Any]]] = None,
        unsupported_ratio: Optional[float] = None,
        violation_counts: Optional[Dict[str, int]] = None,
        claim_check_verdict: Optional[str] = None,
        missing_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.verdict = verdict
        self.score = score
        self.strengths = strengths
        self.weaknesses = weaknesses
        self.improvement_suggestions = improvement_suggestions
        self.summary = summary
        # ── v2（C-2）声明级核查 ──
        self.claim_verdicts = claim_verdicts
        self.unsupported_ratio = unsupported_ratio
        self.violation_counts = violation_counts
        self.claim_check_verdict = claim_check_verdict
        self.missing_evidence = missing_evidence or []

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "verdict": self.verdict,
            "score": self.score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_suggestions": self.improvement_suggestions,
            "summary": self.summary,
        }
        if self.claim_verdicts is not None:
            d.update({
                "claim_verdicts": self.claim_verdicts,
                "unsupported_ratio": self.unsupported_ratio,
                "violation_counts": self.violation_counts,
                "claim_check_verdict": self.claim_check_verdict,
                "missing_evidence": self.missing_evidence,
            })
        return d

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
        if self.claim_verdicts is not None:
            n = len(self.claim_verdicts)
            n_bad = sum(1 for v in self.claim_verdicts if v.get("verdict") != "pass")
            vc = self.violation_counts or {}
            lines.append(
                f"【声明级核查】结论 {self.claim_check_verdict}；声明 {n} 条，未通过 {n_bad} 条，"
                f"无依据占比 {(self.unsupported_ratio or 0) * 100:.0f}%；"
                f"约束违反 DATA {vc.get('DATA', 0)} / EVIDENCE {vc.get('EVIDENCE', 0)} / "
                f"APPLICABILITY {vc.get('APPLICABILITY', 0)} / SAFETY {vc.get('SAFETY', 0)}"
            )
            for v in self.claim_verdicts:
                if v.get("verdict") != "pass":
                    lines.append(f"- [{v.get('claim_id')}] {v.get('verdict')} {'/'.join(v.get('violated_constraints') or [])}："
                                 f"{'；'.join(v.get('detail') or [])}")
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

    def __init__(self, config: Dict[str, Any], claim_check: Optional[str] = None) -> None:
        self.config = config
        wf = config.get("workflow", {}) or {}
        # C-2 声明级核查：off | check（核查并按规则判定，不产出补证路由）| route（C-3：核查 + 补证 / 重规划路由）
        mode = claim_check or wf.get("claim_check", "off")
        self.claim_check = mode if mode in ("off", "check", "route") else "off"
        self.abstain_after = int(wf.get("claim_abstain_after", 3))
        # C-3：route 模式下最多补证轮数（每轮 = supplement → Retriever → Generator 修订）
        self.supplement_rounds = int(wf.get("claim_supplement_rounds", 1))
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
        self.checker = None
        if self.claim_check != "off":
            from src.agents.claim_checker import ClaimChecker
            self.checker = ClaimChecker(
                rel_tol=float(wf.get("claim_rel_tol", 0.02)),
                abs_tol=float(wf.get("claim_abs_tol", 0.05)),
                unsupported_threshold=float(wf.get("claim_unsupported_threshold", 0.3)),
                abstain_after=self.abstain_after,
                llm=self.llm if wf.get("claim_semantic_layer", True) else None,
                hard_constraints=tuple(wf.get("claim_hard_constraints") or ("SAFETY", "DATA", "APPLICABILITY")),
                strict_observation=bool(wf.get("claim_strict_observation", True)),
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

    def _merge_claim_check(self, base: ValidationResult, check) -> ValidationResult:
        """
        把声明级核查结论合并进整体评估（C-2 判定规则优先）：
        - check.verdict == PASS：保留 base（整体评分仍可触发 REVISION），仅附加 v2 字段
        - check.verdict == REVISION：覆盖为 REVISION，并把逐条问题写入 weaknesses / improvement_suggestions
        - check.verdict == ABSTAIN：覆盖为 ABSTAIN
        """
        weaknesses = list(base.weaknesses)
        suggestions = list(base.improvement_suggestions)
        verdict, score, summary = base.verdict, base.score, base.summary
        if check.verdict != "PASS":
            lines = check.feedback_lines()
            weaknesses = lines + weaknesses
            suggestions = [f"修正声明 {l.split(']')[0].lstrip('[')}：{l.split('：', 1)[-1]}" for l in lines] + suggestions
            verdict = check.verdict
            score = min(score, 5 if check.verdict == "REVISION" else 3)
            summary = "声明级核查未通过（" + "；".join(check.reasons) + "）。" + (summary or "")
        elif base.verdict == "FAIL":
            # 声明全部有据但整体评分 FAIL：降为 REVISION 给 Generator 修正机会
            verdict = "REVISION"
        return ValidationResult(
            verdict=verdict, score=score, strengths=list(base.strengths), weaknesses=weaknesses,
            improvement_suggestions=suggestions, summary=summary,
            claim_verdicts=[v.to_dict() for v in check.claim_verdicts],
            unsupported_ratio=check.unsupported_ratio, violation_counts=dict(check.violation_counts),
            claim_check_verdict=check.verdict, missing_evidence=list(check.missing_evidence),
        )

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

        # ── C-2 声明级核查（claim_check != off 且草案带声明时）──
        check_res = None
        if self.checker is not None and getattr(state, "draft_claims", None):
            try:
                check_res = self.checker.check(state)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ClaimChecker failed: %s", exc)
                state.errors.append({"agent": "validator", "error": f"声明级核查异常：{exc}"})
        if check_res is not None:
            result = self._merge_claim_check(result, check_res)
            state.claim_check_log.append({
                "iteration": state.iteration, "verdict": check_res.verdict,
                "n_claims": check_res.n_claims, "unsupported_ratio": round(check_res.unsupported_ratio, 4),
                "violation_counts": dict(check_res.violation_counts), "reasons": list(check_res.reasons),
                "n_checked_llm": check_res.n_checked_llm,
            })
            state.missing_evidence = list(check_res.missing_evidence) if self.claim_check == "route" else []

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
        elif result.verdict == "ABSTAIN":
            # C-2：两轮修订仍不通过 → 弃答，输出「证据不足，建议补充 X」
            state.next_node = "END"
            missing = state.missing_evidence or (check_res.missing_evidence if check_res else [])
            need = "、".join(
                f"{m.get('suggested_tool') or '相关检测'}（声明 {m.get('claim_id')}）" for m in missing[:5]) or "更多检测数据或知识依据"
            bad_lines = [w for w in result.weaknesses if w.startswith("[")][:6]
            state.final_answer = (
                "【证据不足，暂不给出诊断结论】经多轮修订，草案中仍有声明缺乏可核对的证据支撑或与检测结果矛盾。\n\n"
                "未通过核查的声明：\n" + ("\n".join(f"- {l}" for l in bad_lines) if bad_lines else "- （见验证报告）")
                + f"\n\n建议补充：{need}。补充后可重新提交诊断请求。"
            )
            logger.warning("Validator: ABSTAIN after %d iterations, routing to END.", state.iteration)
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
