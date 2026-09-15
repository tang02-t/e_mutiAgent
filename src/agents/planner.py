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
                 allowed_tools: Optional[List[str]] = None) -> None:
        """
        planner_mode（P5-5 / P6 对比开关）：
          baseline   使用 llms.planner（或 default）配置的通用模型
          finetuned  使用 llms.planner_finetuned 配置的微调 Planner（百炼部署的 LoRA 模型或本地服务）；
                     若该节点不存在则回退 baseline 并记录 warning
        取值优先级：构造参数 > config["workflow"]["planner_mode"] > "baseline"

        allowed_tools（P6 五级模式）：None 表示全部工具可用；否则只向模型暴露名单内的工具，
        模型仍调用名单外工具时标记 validation.errors code=tool_disabled，由 Retriever 拒绝执行。
        """
        self.config = config
        self.allowed_tools: Optional[List[str]] = list(allowed_tools) if allowed_tools is not None else None
        mode = planner_mode or (config.get("workflow", {}) or {}).get("planner_mode") or "baseline"
        if mode not in ("baseline", "finetuned"):
            raise ValueError(f"planner_mode 只能是 baseline|finetuned，得到 {mode!r}")
        self.planner_mode = mode
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
            gas_txt = ", ".join(f"{k}={v}" for k, v in dga.items())
            lines.append(f"- 前端填写的 DGA 数据（μL/L）：{gas_txt}")
        for k, v in context.items():
            if k == "dga":
                continue
            if v in (None, "", [], {}):
                continue
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) if lines else "（无额外上下文）"

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
        system_msg = PLANNER_SYSTEM_PROMPT(self.allowed_tools)
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
            return plan_json, content

        logger.warning("LLM 规划结果解析失败，返回空计划。")
        return {
            "steps": [],
            "intent_analysis": intent_part or "（解析失败）",
            "planner_mode": "json_text",
            "plan_status": "parse_failed",
        }, content

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
        logger.info("[yellow]Planner Input:[/yellow] user_query=%s", query)

        if not self.llm.enabled:
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
            plan_json, raw_content = self._llm_plan(query, state.context or {})
        except Exception as exc:
            logger.error("Planner LLM 调用异常: %s", exc)
            state.errors.append({"agent": "planner", "error": f"LLM 调用失败：{exc}"})
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

        plan_json.setdefault("plan_status", "ok" if plan_json.get("steps") else "no_tool")
        plan_json["raw_content"] = raw_content
        plan_json["planner_experiment_mode"] = self.planner_mode
        plan_json["planner_model"] = self.model_name
        if self.allowed_tools is not None:
            plan_json["allowed_tools"] = list(self.allowed_tools)
        state.plan_status = plan_json["plan_status"]
        state.reasoning_trace.append({
            "agent": "planner",
            "type": "llm_plan",
            "content": plan_json,
        })

        # 将结构化计划转成易读文本写入 state.plan
        steps = plan_json.get("steps", [])
        intent_analysis = plan_json.get("intent_analysis", "")

        lines: List[str] = []
        if intent_analysis:
            lines.append("【意图分析】")
            lines.append(intent_analysis)
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
