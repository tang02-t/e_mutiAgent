from typing import Dict, Any, List, Optional

from src.graph.state import AgentState
from src.utils.logging import get_logger
from src.utils.config import get_llm_config
from src.utils.prompts import PLANNER_SYSTEM_PROMPT, render_planner_user
from src.utils.llm import LLMClient, LLMConfig
from src.utils.json_parse import parse_llm_json
from src.tools.tool_registry import to_openai_tools, validate_arguments_strict, get_tool_names


logger = get_logger(__name__)


class PlannerAgent:
    """
    策略规划智能体：
    - 使用大模型 + ReAct 风格思维链规划诊断流程
    - 若大模型不可用或解析失败，返回空计划

    实验口径说明（P0 修复后）：
    - 模型给出的参数只做「严格校验」，不做任何静默修正/删除；
      校验失败的步骤会带上 `validation.errors`，由 Retriever 决定是否执行。
    - 计划状态 `plan_status` 区分：
        ok            正常解析，至少一个步骤
        no_tool       正常解析，模型明确不调用任何工具
        parse_failed  模型输出无法解析
        llm_error     LLM 调用异常
        llm_disabled  LLM 未启用
    - state.context（前端 DGA、设备信息等）会显式注入 Planner 输入，
      避免「模型没看到的数据却被要求填对参数」。
    """

    def __init__(self, config: Dict[str, Any], planner_mode: Optional[str] = None,
                 allowed_tools: Optional[List[str]] = None, *,
                 strategy: Optional[str] = None, decision: Optional[str] = None) -> None:
        """
        planner_mode（P5-5 / P6 对比开关）：
          baseline   使用 llms.planner（或 default）配置的通用模型
          finetuned  使用 llms.planner_finetuned 配置的微调 Planner（百炼部署的 LoRA 模型或本地服务）；
                     若该节点不存在则回退 baseline 并记录 warning
        取值优先级：构造参数 > config["workflow"]["planner_mode"] > "baseline"

        allowed_tools（P6 五级模式）：None 表示全部工具可用；否则只向模型暴露名单内的工具，
        模型仍调用名单外工具时标记 validation.errors code=tool_disabled，由 Retriever 拒绝执行。

        strategy（B-4 主动规划开关）：
          free    只做意图分析 + 工具决策（v1 行为），不输出 action
          active  使用 system_active.txt 变体，输出 action: ask | call_tool | conclude 与 rationale，
                  可读取 context.uncertainty（上一轮归因的熵 / 推荐征兆）决定是否追问
        取值优先级：构造参数 > config["workflow"]["planner_strategy"] > "free"

        decision（仅 strategy=active 有效）：
          llm     由大模型决策动作（默认）；LLM 不可用或调用失败时回退 eig 规则
          eig     不调用 LLM，直接按 EIG 推荐（suggested_action / 最高 VoI 征兆）决策，
                  即 B-5 的 eig_greedy 策略，也是无 LLM 环境下走通追问循环的路径
        """
        self.config = config
        self.allowed_tools: Optional[List[str]] = list(allowed_tools) if allowed_tools is not None else None
        wf_cfg = (config.get("workflow", {}) or {})
        mode = planner_mode or wf_cfg.get("planner_mode") or "baseline"
        if mode not in ("baseline", "finetuned"):
            raise ValueError(f"planner_mode 只能是 baseline|finetuned，得到 {mode!r}")
        self.planner_mode = mode
        strategy = strategy or wf_cfg.get("planner_strategy") or "free"
        if strategy not in ("free", "active"):
            raise ValueError(f"planner_strategy 只能是 free|active，得到 {strategy!r}")
        self.strategy = strategy
        decision = decision or wf_cfg.get("planner_decision") or "llm"
        if decision not in ("llm", "eig"):
            raise ValueError(f"planner_decision 只能是 llm|eig，得到 {decision!r}")
        self.decision = decision
        if mode == "finetuned":
            model_cfg = (config.get("llms", {}) or {}).get("planner_finetuned")
            if not model_cfg:
                logger.warning("planner_mode=finetuned 但 config.llms.planner_finetuned 缺失，回退 baseline")
                self.planner_mode = "baseline"
                model_cfg = get_llm_config(config, "planner")
            else:
                model_cfg = {**get_llm_config(config, "planner"), **model_cfg}
        else:
            model_cfg = get_llm_config(config, "planner")
        self.model_name = model_cfg.get("model_name", "")
        self.llm = LLMClient(
            LLMConfig(
                provider=model_cfg.get("provider", "openai"),
                model_name=model_cfg.get("model_name", ""),
                temperature=float(model_cfg.get("temperature", 0.2)),
                max_tokens=int(model_cfg.get("max_tokens", 2048)),
                base_url=model_cfg.get("base_url", "https://api.openai.com/v1"),
                api_key=model_cfg.get("api_key", ""),
                api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
                enable_thinking=model_cfg.get("enable_thinking", False),
                timeout=float(model_cfg.get("timeout", 60.0)),
                max_retries=int(model_cfg.get("max_retries", 2)),
            )
        )

    # ──────────────────────────────────────────────────────────────
    # 上下文渲染
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _render_context(context: Dict[str, Any]) -> str:
        """把 state.context 渲染成给模型看的简短文本；为空时返回占位说明。"""
        if not context:
            return "（无额外上下文）"
        lines: List[str] = []
        dga = context.get("dga")
        if isinstance(dga, dict) and dga:
            gas_txt = ", ".join(f"{k}={v}" for k, v in dga.items() if v is not None)
            missing = [k for k, v in dga.items() if v is None]
            lines.append(f"- 前端填写的 DGA 数据（μL/L）：{gas_txt}"
                         + (f"（未提供：{', '.join(missing)}）" if missing else ""))
        ev = context.get("evidence")
        if isinstance(ev, dict) and ev:
            pos = [k for k, v in ev.items() if v is True]
            neg = [k for k, v in ev.items() if v is False]
            if pos:
                lines.append(f"- 已确认出现的征兆：{', '.join(pos)}")
            if neg:
                lines.append(f"- 已确认排除的征兆：{', '.join(neg)}")
        unavailable = context.get("unavailable_symptoms")
        if isinstance(unavailable, list) and unavailable:
            lines.append(f"- 用户无法提供的征兆（不要再问）：{', '.join(unavailable)}")
        asked = context.get("asked_symptoms")
        if isinstance(asked, list) and asked:
            lines.append(f"- 已追问过的征兆：{', '.join(asked)}")
        unc = context.get("uncertainty")
        if isinstance(unc, dict) and unc:
            lines.append(PlannerAgent._render_uncertainty(unc))
        skip = {"dga", "evidence", "unavailable_symptoms", "asked_symptoms", "uncertainty"}
        for k, v in context.items():
            if k in skip:
                continue
            if v in (None, "", [], {}):
                continue
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) if lines else "（无额外上下文）"

    @staticmethod
    def _render_uncertainty(unc: Dict[str, Any]) -> str:
        """渲染上一轮归因的不确定性块（B-4）。"""
        parts = [
            "- 上一轮故障归因的不确定性（uncertainty）："
            f"entropy_bits={unc.get('entropy_bits', 0):.2f}, top1={unc.get('top1')}, "
            f"top1_prob={unc.get('top1_prob', 0):.2f}, top1_top2_gap={unc.get('top1_top2_gap', 0):.2f}, "
            f"rounds_done={unc.get('rounds_done', 0)}, "
            f"suggested_action={unc.get('suggested_action')}"
            + (f", suggested_tool={unc.get('suggested_tool')}" if unc.get("suggested_tool") else "")
            + (f", stop_reason={unc.get('stop_reason')}" if unc.get("stop_reason") else "")
        ]
        recs = unc.get("recommendations") or []
        if recs:
            parts.append("  recommendations（按 VoI 降序）：")
            for r in recs[:5]:
                parts.append(
                    f"    - {r.get('symptom')}: EIG={r.get('eig', 0):.3f} bit, cost={r.get('cost', 0):.0f}, "
                    f"VoI={r.get('voi', 0):.3f}, how_to_obtain={r.get('how_to_obtain')}"
                    + (f", 话术参考：{r.get('ask_hint')}" if r.get("ask_hint") else "")
                )
        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────────
    # LLM 规划
    # ──────────────────────────────────────────────────────────────
    def _llm_plan(self, query: str, context: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """
        调用大模型生成计划，返回 (plan_json, raw_content)。

        plan_json 结构：
        {
          "intent_analysis": "...",
          "plan_status": "ok | no_tool | parse_failed",
          "planner_mode": "function_calling | json_text",
          "steps": [
            {"id": 1, "stage": "...", "description": "...",
             "tool": "rag_search",
             "arguments": {...原始参数...},
             "validation": {"valid": true, "errors": []}},
            ...
          ]
        }
        """
        system_msg = PLANNER_SYSTEM_PROMPT(self.allowed_tools, variant="active" if self.strategy == "active" else None)
        user_msg = render_planner_user(query=query, context=self._render_context(context))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # ① 优先尝试原生 Function Calling
        try:
            tools = to_openai_tools(self.allowed_tools)
            choice = self.llm.chat(messages, tools=tools) if tools else self.llm.chat(messages)
        except Exception:
            # 个别网关不支持 tools 参数时，退回无 tools 调用
            choice = self.llm.chat(messages)

        content = choice.get("content") if isinstance(choice, dict) else ""
        content = content or ""
        tool_calls = choice.get("tool_calls") if isinstance(choice, dict) else None

        if tool_calls:
            plan_json = self._build_plan_from_tool_calls(tool_calls, content)
            plan_json["planner_mode"] = "function_calling"
            if self.strategy == "active":
                # 模型走了函数调用即视为 call_tool；若 content 里同时带了 JSON，尽量提取 action / rationale
                extra = parse_llm_json(content) if content and "{" in content else None
                if isinstance(extra, dict):
                    for k in ("action", "rationale", "symptom", "question"):
                        if extra.get(k):
                            plan_json[k] = extra[k]
                plan_json.setdefault("action", "call_tool")
            logger.info(
                "Planner 使用原生 Function Calling，模型给出 %d 个工具调用。",
                len(plan_json.get("steps", [])),
            )
            return plan_json, content

        # ② 回退：解析 JSON 文本
        intent_part = ""
        json_start = content.find("{")
        if json_start > 0:
            intent_part = content[:json_start].strip()

        plan_json = parse_llm_json(content)
        if plan_json is not None and isinstance(plan_json, dict):
            if intent_part and "intent_analysis" not in plan_json:
                plan_json["intent_analysis"] = intent_part
            steps = plan_json.get("steps") or []
            if not isinstance(steps, list):
                steps = []
            self._validate_steps(steps)
            plan_json["steps"] = steps
            plan_json["planner_mode"] = "json_text"
            plan_json["plan_status"] = "ok" if any(s.get("tool") for s in steps) else "no_tool"
            if self.strategy == "active":
                self._validate_action(plan_json, context)
            return plan_json, content

        logger.warning("LLM 规划结果解析失败，返回空计划。")
        return {
            "steps": [],
            "intent_analysis": intent_part or "（解析失败）",
            "planner_mode": "json_text",
            "plan_status": "parse_failed",
        }, content

    # ──────────────────────────────────────────────────────────────
    # B-4 主动动作：校验与规则决策
    # ──────────────────────────────────────────────────────────────
    ACTIONS = ("ask", "call_tool", "conclude")

    @staticmethod
    def _recommended_symptoms(context: Dict[str, Any]) -> List[str]:
        unc = (context or {}).get("uncertainty") or {}
        return [r.get("symptom") for r in (unc.get("recommendations") or []) if r.get("symptom")]

    def _validate_action(self, plan_json: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        校验 active 输出的 action 字段（原地补充 action_validation，不改写模型输出）：
          - action 必须是 ask | call_tool | conclude；缺失时按 steps 推断（有工具 → call_tool，否则 conclude）
          - ask 必须携带 symptom，且 symptom 属于上一轮 recommendations；缺失 / 越界记 code
          - ask 时 question 缺失记 code=missing_question（仍可执行，用 ask_hint 兜底）
        """
        errors: List[Dict[str, Any]] = []
        action = plan_json.get("action")
        steps = plan_json.get("steps") or []
        if action not in self.ACTIONS:
            inferred = "call_tool" if any(isinstance(s, dict) and s.get("tool") for s in steps) else "conclude"
            if action:
                errors.append({"field": "action", "code": "unknown_action",
                               "message": f"未知动作 {action!r}，按 steps 推断为 {inferred}"})
            else:
                errors.append({"field": "action", "code": "missing_action",
                               "message": f"缺少 action，按 steps 推断为 {inferred}"})
            plan_json["action"] = action = inferred
        if action == "ask":
            symptom = plan_json.get("symptom")
            recs = self._recommended_symptoms(context)
            if not symptom:
                errors.append({"field": "symptom", "code": "missing_symptom",
                               "message": "action=ask 必须携带 symptom"})
            elif recs and symptom not in recs:
                errors.append({"field": "symptom", "code": "symptom_not_recommended",
                               "message": f"symptom {symptom!r} 不在推荐列表 {recs}"})
            asked = (context or {}).get("asked_symptoms") or []
            if symptom and symptom in asked:
                errors.append({"field": "symptom", "code": "symptom_repeated",
                               "message": f"symptom {symptom!r} 已追问过"})
            if not plan_json.get("question"):
                errors.append({"field": "question", "code": "missing_question",
                               "message": "action=ask 缺少 question，将用 ask_hint 兜底"})
        if action == "call_tool" and not any(isinstance(s, dict) and s.get("tool") for s in steps):
            errors.append({"field": "steps", "code": "call_tool_without_steps",
                           "message": "action=call_tool 但 steps 为空"})
        fatal = {"missing_symptom", "symptom_not_recommended", "symptom_repeated", "call_tool_without_steps"}
        plan_json["action_validation"] = {
            "valid": not any(e["code"] in fatal for e in errors),
            "errors": errors,
        }

    def eig_decide(self, context: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """
        不调用 LLM 的规则决策（decision=eig，或 LLM 不可用时的回退）：
          - 上下文里没有 uncertainty（首轮）：有 DGA 或已确认征兆 → call_tool fault_attribution；否则 conclude
          - 有 uncertainty：suggested_action=conclude → conclude；
            否则取 recommendations 中第一个未追问且 how_to_obtain=ask_user 的征兆 → ask；
            若首推是 call_tool:<tool> → call_tool 该工具（工具不在 allowed_tools 时跳到下一个）；
            无可行推荐 → conclude
        """
        ctx = context or {}
        unc = ctx.get("uncertainty") or {}
        dga = ctx.get("dga") if isinstance(ctx.get("dga"), dict) else None
        ev = ctx.get("evidence") if isinstance(ctx.get("evidence"), dict) else None
        allowed = set(self.allowed_tools) if self.allowed_tools is not None else None

        def _fa_step() -> Dict[str, Any]:
            args: Dict[str, Any] = {"query": query}
            if dga:
                args["dga_data"] = {k: v for k, v in dga.items() if v is not None}
            if ev:
                args["evidence"] = dict(ev)
            step = {"id": 1, "stage": "故障归因", "description": "基于当前可见证据做贝叶斯归因并评估不确定性",
                    "tool": "fault_attribution", "arguments": args}
            self._validate_steps([step])
            return step

        if not unc:
            if (dga and any(v is not None for v in dga.values())) or ev:
                if allowed is not None and "fault_attribution" not in allowed:
                    return {"action": "conclude", "rationale": "当前模式不可用 fault_attribution", "steps": []}
                return {"action": "call_tool", "rationale": "首轮：有 DGA / 征兆数据，先做归因并获取不确定性",
                        "steps": [_fa_step()]}
            return {"action": "conclude", "rationale": "无 DGA 与征兆数据，无法归因", "steps": []}

        if unc.get("suggested_action") == "conclude":
            return {"action": "conclude", "steps": [],
                    "rationale": f"停止准则触发（{unc.get('stop_reason')}），entropy_bits={unc.get('entropy_bits', 0):.2f}"}

        asked = set(ctx.get("asked_symptoms") or [])
        unavailable = set(ctx.get("unavailable_symptoms") or [])
        for r in unc.get("recommendations") or []:
            sym = r.get("symptom")
            if not sym or sym in asked or sym in unavailable:
                continue
            how = str(r.get("how_to_obtain") or "ask_user")
            if how.startswith("call_tool:"):
                tool = how.split(":", 1)[1].strip()
                if allowed is not None and tool not in allowed:
                    continue
                step = {"id": 1, "stage": "工具调用", "description": f"调用 {tool} 获取 {sym}",
                        "tool": tool, "arguments": {}}
                self._validate_steps([step])
                if not step.get("validation", {}).get("valid", False):
                    continue  # 该工具需要用户提供的必填参数（如时序信号），当前无法调用
                return {"action": "call_tool", "steps": [step], "symptom": sym,
                        "rationale": f"最高 VoI 征兆 {sym}（EIG={r.get('eig', 0):.3f}）需由工具 {tool} 获取"}
            question = r.get("ask_hint") or f"请确认现场是否观察到 {sym}？"
            return {"action": "ask", "symptom": sym, "question": question, "steps": [],
                    "rationale": f"entropy_bits={unc.get('entropy_bits', 0):.2f} 仍高，"
                                 f"追问 VoI 最高的 {sym}（EIG={r.get('eig', 0):.3f} bit, cost={r.get('cost', 0):.0f}）"}
        return {"action": "conclude", "steps": [], "rationale": "无可追问的征兆，直接结论"}

    def _validate_steps(self, steps: List[Dict[str, Any]]) -> None:
        """
        对 steps 中带 tool 的步骤做严格校验（原地补充 validation 字段，不修改 arguments）。
        未知工具同样保留原始 tool 名，只标记 validation.valid=False；
        allowed_tools 名单外的工具标记 code=tool_disabled（P6 五级模式下的“越权调用”统计口径）。
        """
        valid_tools = set(get_tool_names())
        allowed = set(self.allowed_tools) if self.allowed_tools is not None else None
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            if not tool:
                continue
            raw_args = step.get("arguments") or {}
            if not isinstance(raw_args, dict):
                step["validation"] = {
                    "valid": False,
                    "errors": [{"field": None, "code": "arguments_not_object",
                                "message": "arguments 必须是 JSON 对象"}],
                }
                continue
            if tool not in valid_tools:
                step["validation"] = {
                    "valid": False,
                    "errors": [{"field": None, "code": "unknown_tool",
                                "message": f"未知工具：{tool}"}],
                }
                continue
            if allowed is not None and tool not in allowed:
                step["validation"] = {
                    "valid": False,
                    "errors": [{"field": None, "code": "tool_disabled",
                                "message": f"工具 {tool} 在当前系统模式下不可用"}],
                }
                continue
            step["validation"] = validate_arguments_strict(tool, raw_args)

    def _build_plan_from_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        content: str,
    ) -> Dict[str, Any]:
        """
        将原生 Function Calling 的 tool_calls 转换为统一的 plan_json.steps 结构。
        未知工具不再跳过，而是保留并标记校验失败，便于评测统计。
        """
        steps: List[Dict[str, Any]] = []

        for idx, call in enumerate(tool_calls, start=1):
            name = call.get("name", "") or ""
            raw_args = call.get("arguments") or {}
            if not isinstance(raw_args, dict):
                raw_args = {"_raw": raw_args}
            step = {
                "id": idx,
                "stage": "工具调用",
                "description": f"调用工具 {name} 收集诊断证据",
                "tool": name,
                "arguments": raw_args,
                "call_id": call.get("id"),
            }
            steps.append(step)

        self._validate_steps(steps)

        return {
            "intent_analysis": content.strip() if content else "（模型通过函数调用直接给出工具规划）",
            "steps": steps,
            "plan_status": "ok" if steps else "no_tool",
        }

    # ──────────────────────────────────────────────────────────────
    # 入口
    # ──────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        query = state.user_query
        logger.info("[yellow]Planner Input:[/yellow] user_query=%s strategy=%s", query, self.strategy)
        state.planner_strategy = self.strategy
        context = state.context or {}

        # ── active + eig：不调用 LLM 的规则决策 ──
        if self.strategy == "active" and self.decision == "eig":
            plan_json = self.eig_decide(context, query)
            plan_json["decision_source"] = "eig"
            plan_json["plan_status"] = "ok" if plan_json.get("steps") else "no_tool"
            plan_json["planner_mode"] = "rule"
            plan_json.setdefault("intent_analysis", "（EIG 规则决策）")
            self._validate_action(plan_json, context)
            return self._finish(state, plan_json, raw_content="")

        if not self.llm.enabled:
            if self.strategy == "active":
                # LLM 不可用时回退 EIG 规则，保证追问循环可离线走通
                plan_json = self.eig_decide(context, query)
                plan_json["decision_source"] = "eig_fallback"
                plan_json["plan_status"] = "ok" if plan_json.get("steps") else "no_tool"
                plan_json["planner_mode"] = "rule"
                plan_json.setdefault("intent_analysis", "（LLM 未启用，EIG 规则决策）")
                self._validate_action(plan_json, context)
                return self._finish(state, plan_json, raw_content="")
            state.plan = ""
            state.plan_status = "llm_disabled"
            state.reasoning_trace.append({
                "agent": "planner",
                "type": "llm_plan",
                "content": {"steps": [], "plan_status": "llm_disabled",
                            "intent_analysis": "（LLM 未启用）"},
            })
            logger.info("[yellow]Planner Output:[/yellow] plan=(LLM not enabled)")
            return state

        try:
            plan_json, raw_content = self._llm_plan(query, context)
        except Exception as exc:
            logger.error("Planner LLM 调用异常: %s", exc)
            state.errors.append({"agent": "planner", "error": f"LLM 调用失败：{exc}"})
            if self.strategy == "active":
                plan_json = self.eig_decide(context, query)
                plan_json["decision_source"] = "eig_fallback"
                plan_json["plan_status"] = "ok" if plan_json.get("steps") else "no_tool"
                plan_json["planner_mode"] = "rule"
                plan_json.setdefault("intent_analysis", f"（LLM 调用失败：{exc}；EIG 规则决策）")
                self._validate_action(plan_json, context)
                return self._finish(state, plan_json, raw_content="")
            state.plan = ""
            state.plan_status = "llm_error"
            state.reasoning_trace.append({
                "agent": "planner",
                "type": "llm_plan",
                "content": {"steps": [], "plan_status": "llm_error",
                            "intent_analysis": f"（LLM 调用失败：{exc}）"},
            })
            logger.info("[yellow]Planner Output:[/yellow] plan=(empty due to error)")
            return state

        if self.strategy == "active":
            plan_json.setdefault("decision_source", "llm")
            if "action_validation" not in plan_json:
                self._validate_action(plan_json, context)
        return self._finish(state, plan_json, raw_content=raw_content)

    def _finish(self, state: AgentState, plan_json: Dict[str, Any], *, raw_content: str) -> AgentState:
        """把 plan_json 写入 state（trace / plan 文本 / 动作字段）。"""
        plan_json.setdefault("plan_status", "ok" if plan_json.get("steps") else "no_tool")
        plan_json["raw_content"] = raw_content
        plan_json["planner_experiment_mode"] = self.planner_mode
        plan_json["planner_model"] = self.model_name if plan_json.get("decision_source", "llm") == "llm" else "eig_rule"
        plan_json["planner_strategy"] = self.strategy
        if self.allowed_tools is not None:
            plan_json["allowed_tools"] = list(self.allowed_tools)
        state.plan_status = plan_json["plan_status"]
        state.reasoning_trace.append({
            "agent": "planner",
            "type": "llm_plan",
            "content": plan_json,
        })

        if self.strategy == "active":
            action = plan_json.get("action") or ("call_tool" if plan_json.get("steps") else "conclude")
            state.planner_action = action
            if action == "ask" and plan_json.get("action_validation", {}).get("valid", True):
                unc = (state.context or {}).get("uncertainty") or {}
                rec = next((r for r in unc.get("recommendations") or [] if r.get("symptom") == plan_json.get("symptom")), {})
                state.pending_questions = [{
                    "symptom": plan_json.get("symptom"),
                    "question": plan_json.get("question") or rec.get("ask_hint") or f"请确认是否存在 {plan_json.get('symptom')}",
                    "rationale": plan_json.get("rationale", ""),
                    "eig": rec.get("eig"), "voi": rec.get("voi"), "cost": rec.get("cost"),
                    "how_to_obtain": rec.get("how_to_obtain", "ask_user"),
                }]
            else:
                state.pending_questions = []

        # 将结构化计划转成易读文本写入 state.plan
        steps = plan_json.get("steps", [])
        intent_analysis = plan_json.get("intent_analysis", "")

        lines: List[str] = []
        if intent_analysis:
            lines.append("【意图分析】")
            lines.append(intent_analysis)
            lines.append("")
        if self.strategy == "active":
            lines.append(f"【动作】{plan_json.get('action')}（{plan_json.get('decision_source')}）"
                         + (f" symptom={plan_json.get('symptom')}" if plan_json.get("symptom") else ""))
            if plan_json.get("rationale"):
                lines.append(f"  理由：{plan_json['rationale']}")
            if plan_json.get("question"):
                lines.append(f"  追问：{plan_json['question']}")
            av = plan_json.get("action_validation") or {}
            if av.get("errors"):
                lines.append("  ⚠ 动作校验：" + "; ".join(e.get("message", "") for e in av["errors"]))
            lines.append("")
        lines.append(f"【诊断计划】（状态：{state.plan_status}）")
        if not steps:
            lines.append("  （模型未规划任何工具调用）")

        for step in steps:
            desc = step.get("description", "")
            tool = step.get("tool")
            stage = step.get("stage", "")
            val = step.get("validation") or {}
            if tool:
                flag = "" if val.get("valid", True) else "  ⚠ 参数校验失败"
                lines.append(f"  {step.get('id', '?')}. [{stage}] {desc} → 工具：{tool}{flag}")
            else:
                lines.append(f"  {step.get('id', '?')}. [{stage}] {desc}")

        state.plan = "\n".join(lines)
        logger.info("[yellow]Planner Output:[/yellow] plan=%s", state.plan)
        logger.info("Planner produced plan: status=%s, steps=%d", state.plan_status, len(steps))
        return state
