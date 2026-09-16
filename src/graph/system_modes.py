"""
五级系统模式定义（system modes，PLAN v2 §0.5 口径）。

每级只在上一级基础上打开一项能力；知识库两路检索、kg_search、词法反思属于工程基座，
从 mode2 起全部开启且不再单独归因。

  mode1  llm_only      无任何工具，LLM 直接回答（参照）
  mode2  tool_base     全部五个工具 + 两路检索 + 图谱 + 词法反思；归因引擎用专家默认参数、
                       不输出不确定性（EIG 关闭）；Validator 为整体评分版（claim_check=off）；
                       Planner 为通用基座、free 策略（工程基座）
  mode3  active_plan   归因引擎切换为校准参数并输出 EIG 推荐；Planner 启用主动问询 / 补证策略（主线一）
  mode4  claim_verify  Generator 输出 claim-evidence 结构；Validator 切换为声明级约束核查 + 补证重规划路由（主线二）
  mode5  dpo_planner   Planner 切换为百炼部署的 DPO 模型 llms.planner_finetuned（主线三）

所有开关都能被 resolve_mode 的显式参数覆盖，用于交叉对照（例如 mode4 + baseline planner、
mode5 + v1 validator）。

工程注意：归因引擎参数（configure_engine）与 EIG 开关（环境变量 FAULT_ATTR_EIG）是进程级全局状态，
build_agents 会按 spec 设置它们；同一进程内切换模式时必须重新调用 build_agents（或 apply_engine_settings）。

用法：
    from src.graph.system_modes import SYSTEM_MODES, resolve_mode, build_agents
    spec = resolve_mode("mode4", planner_mode="baseline")
    planner, retriever, generator, validator, reflector = build_agents(config, mcp, spec)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional


ALL_TOOLS = ["rag_search", "kg_search", "fault_attribution", "ett_forecast", "timeseries_anomaly"]

ATTRIBUTION_MODES = ("expert", "calibrated")
PLANNER_STRATEGIES = ("free", "active")
GENERATOR_OUTPUTS = ("text", "claims")
VALIDATOR_MODES = ("off", "check", "route")
PLANNER_MODES = ("baseline", "finetuned")
REFLECTIONS = ("off", "lexical", "llm")


@dataclass(frozen=True)
class SystemModeSpec:
    key: str                          # mode1..mode5
    name: str                         # 英文短名
    label: str                        # 中文展示名
    allowed_tools: List[str]          # Planner 可见 / Retriever 可执行的工具；[] 表示无工具
    rag_mode: Optional[str]           # LocalKBRetriever mode：None 表示不启用 rag_search
    planner_mode: str                 # baseline | finetuned
    reflection: str                   # off | lexical | llm
    attribution_mode: str = "expert"  # expert | calibrated（B-1 校准参数）
    with_eig: bool = False            # fault_attribution 是否输出 EIG 推荐块（B-2）
    planner_strategy: str = "free"    # free | active（B-4 主动问询 / 补证）
    generator_output: str = "text"    # text | claims（C-1 声明级输出）
    validator_mode: str = "off"       # off | check | route（C-2 / C-3，即 ValidatorAgent.claim_check）
    description: str = ""
    mainline: str = ""                # 对应论文主线

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "label": self.label,
            "allowed_tools": list(self.allowed_tools), "rag_mode": self.rag_mode,
            "planner_mode": self.planner_mode, "reflection": self.reflection,
            "attribution_mode": self.attribution_mode, "with_eig": self.with_eig,
            "planner_strategy": self.planner_strategy, "generator_output": self.generator_output,
            "validator_mode": self.validator_mode, "mainline": self.mainline,
        }

    def summary(self) -> str:
        """一行开关摘要（评测脚本 / 前端展示）。"""
        tools = "无" if not self.allowed_tools else ("全部五工具" if set(self.allowed_tools) == set(ALL_TOOLS)
                                                    else "、".join(self.allowed_tools))
        return (f"tools={tools} rag={self.rag_mode or 'off'} reflection={self.reflection} "
                f"attribution={self.attribution_mode}{'+eig' if self.with_eig else ''} "
                f"planner={self.planner_mode}/{self.planner_strategy} "
                f"generator={self.generator_output} validator={self.validator_mode}")


SYSTEM_MODES: Dict[str, SystemModeSpec] = {
    "mode1": SystemModeSpec(
        key="mode1", name="llm_only", label="模式 1：LLM 直答",
        allowed_tools=[], rag_mode=None, planner_mode="baseline", reflection="off",
        attribution_mode="expert", with_eig=False, planner_strategy="free",
        generator_output="text", validator_mode="off",
        description="无任何工具，答案完全依赖模型自身知识；用作参照。",
        mainline="参照",
    ),
    "mode2": SystemModeSpec(
        key="mode2", name="tool_base", label="模式 2：工具基座",
        allowed_tools=list(ALL_TOOLS), rag_mode="two_way", planner_mode="baseline", reflection="lexical",
        attribution_mode="expert", with_eig=False, planner_strategy="free",
        generator_output="text", validator_mode="off",
        description="全部五个工具 + 两路检索 + 故障图谱 + 词法反思；归因引擎用专家默认参数且不输出不确定性；"
                    "Validator 为整体评分版；Planner 为通用基座、一次规划直接结论。",
        mainline="工程基座",
    ),
    "mode3": SystemModeSpec(
        key="mode3", name="active_plan", label="模式 3：+ 主动规划",
        allowed_tools=list(ALL_TOOLS), rag_mode="two_way", planner_mode="baseline", reflection="lexical",
        attribution_mode="calibrated", with_eig=True, planner_strategy="active",
        generator_output="text", validator_mode="off",
        description="在模式 2 上：归因引擎切换为校准参数并输出 EIG 推荐；Planner 启用主动问询 / 补证策略（按信息增益决定追问、调工具或结论）。",
        mainline="主线一",
    ),
    "mode4": SystemModeSpec(
        key="mode4", name="claim_verify", label="模式 4：+ 声明级验证",
        allowed_tools=list(ALL_TOOLS), rag_mode="two_way", planner_mode="baseline", reflection="lexical",
        attribution_mode="calibrated", with_eig=True, planner_strategy="active",
        generator_output="claims", validator_mode="route",
        description="在模式 3 上：Generator 输出 claim-evidence 结构；Validator 切换为声明级约束核查 + 补证重规划路由。",
        mainline="主线二",
    ),
    "mode5": SystemModeSpec(
        key="mode5", name="dpo_planner", label="模式 5：+ DPO Planner（完整系统）",
        allowed_tools=list(ALL_TOOLS), rag_mode="two_way", planner_mode="finetuned", reflection="lexical",
        attribution_mode="calibrated", with_eig=True, planner_strategy="active",
        generator_output="claims", validator_mode="route",
        description="在模式 4 上：Planner 切换为百炼部署的 DPO 模型（llms.planner_finetuned；未配置时自动回退 baseline）。",
        mainline="主线三",
    ),
}

MODE_ORDER = ["mode1", "mode2", "mode3", "mode4", "mode5"]
_ALIAS = {spec.name: key for key, spec in SYSTEM_MODES.items()}
_ALIAS.update({"1": "mode1", "2": "mode2", "3": "mode3", "4": "mode4", "5": "mode5", "full": "mode5"})


def _check(value: Any, allowed: tuple, field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} 只能是 {'|'.join(map(str, allowed))}，得到 {value!r}")


def resolve_mode(mode: str, *,
                 planner_mode: Optional[str] = None,
                 reflection: Optional[str] = None,
                 rag_mode: Optional[str] = None,
                 validator_mode: Optional[str] = None,
                 attribution_mode: Optional[str] = None,
                 planner_strategy: Optional[str] = None,
                 generator_output: Optional[str] = None,
                 with_eig: Optional[bool] = None) -> SystemModeSpec:
    """
    按 key / 英文名 / 数字解析模式，并允许显式覆盖子开关（交叉对照实验）。
    覆盖项彼此独立：例如 mode4 + planner_mode=baseline、mode5 + validator_mode=off。
    rag_mode 覆盖仅对启用了 rag_search 的模式生效。
    """
    key = _ALIAS.get(str(mode), str(mode))
    if key not in SYSTEM_MODES:
        raise ValueError(f"未知系统模式 {mode!r}，可选 {MODE_ORDER} 或 {sorted(_ALIAS)}")
    spec = SYSTEM_MODES[key]
    overrides: Dict[str, Any] = {}
    if planner_mode:
        _check(planner_mode, PLANNER_MODES, "planner_mode")
        overrides["planner_mode"] = planner_mode
    if reflection:
        _check(reflection, REFLECTIONS, "reflection")
        overrides["reflection"] = reflection
    if rag_mode and spec.rag_mode is not None:
        overrides["rag_mode"] = rag_mode
    if validator_mode:
        _check(validator_mode, VALIDATOR_MODES, "validator_mode")
        overrides["validator_mode"] = validator_mode
    if attribution_mode:
        _check(attribution_mode, ATTRIBUTION_MODES, "attribution_mode")
        overrides["attribution_mode"] = attribution_mode
    if planner_strategy:
        _check(planner_strategy, PLANNER_STRATEGIES, "planner_strategy")
        overrides["planner_strategy"] = planner_strategy
    if generator_output:
        _check(generator_output, GENERATOR_OUTPUTS, "generator_output")
        overrides["generator_output"] = generator_output
    if with_eig is not None:
        overrides["with_eig"] = bool(with_eig)
    return replace(spec, **overrides) if overrides else spec


def tool_stack_spec(config: Dict[str, Any], *, reflection: str = "off", planner_mode: str = "baseline",
                    with_eig: bool = True, **overrides: Any) -> SystemModeSpec:
    """
    组件级评测用的「工具栈」spec：全部五个工具 + 两路检索（同 mode2），但归因参数 / Planner 策略 /
    Generator 输出 / Validator 核查模式一律取自 config.workflow（attribution_mode / planner_strategy /
    generator_output_mode / claim_check），反思默认关闭、EIG 默认开启。
    用途：C-1 ~ C-5、D-2 等只考察单个组件、需要把其它开关交给脚本自身配置控制的场景；
    五级模式对比请用 resolve_mode。
    """
    wf = (config.get("workflow", {}) or {})
    base = dict(
        attribution_mode=wf.get("attribution_mode", "calibrated"),
        planner_strategy=wf.get("planner_strategy", "free"),
        generator_output=wf.get("generator_output_mode", "text"),
        validator_mode=wf.get("claim_check", "off"),
    )
    base.update({k: v for k, v in overrides.items() if v is not None})
    return resolve_mode("mode2", planner_mode=planner_mode, reflection=reflection, with_eig=with_eig, **base)


def register_rag_for_mode(mcp, spec: SystemModeSpec, *, fallback_mock=None) -> str:
    """
    按模式为 MCPClient 注册 rag_search：
      - rag_mode None → 注册一个恒返回空列表的桩（Planner 本不该调用；若越权调用会得到 empty_result）
      - 其余 → LocalKBRetriever(mode=rag_mode)
    返回实际生效的 kb 描述字符串。
    """
    if spec.rag_mode is None:
        mcp.register_tool("rag_search", lambda query, **_: [])
        return "disabled"
    try:
        from src.tools.local_kb import local_kb_search
        mode = spec.rag_mode
        mcp.register_tool("rag_search", lambda query, top_k=5, **_: local_kb_search(query, top_k=top_k, mode=mode))
        return f"local_kb:{mode}"
    except Exception as exc:  # noqa: BLE001
        if fallback_mock is not None:
            mcp.register_tool("rag_search", fallback_mock)
            return f"mock（local_kb 不可用：{exc}）"
        raise


def apply_engine_settings(spec: SystemModeSpec) -> None:
    """按模式设置进程级全局开关：归因引擎参数（expert|calibrated）与 EIG 输出（FAULT_ATTR_EIG）。"""
    from src.tools.fault_attribution import configure_engine
    configure_engine(spec.attribution_mode)
    os.environ["FAULT_ATTR_EIG"] = "1" if spec.with_eig else "0"


def build_reflector(config: Dict[str, Any], spec: SystemModeSpec):
    """按模式构建反思模块；off 返回 None。LLM 评分器不可用时回退词法评分器。"""
    if spec.reflection == "off":
        return None
    from src.agents.reflection import ReflectionModule, LexicalScorer
    from src.tools.local_kb import get_local_kb
    kb = get_local_kb(mode=spec.rag_mode or "two_way")
    scorer = LexicalScorer()
    if spec.reflection == "llm":
        try:
            from src.agents.reflection import LLMScorer
            from src.utils.config import get_llm_config
            from src.utils.llm import LLMClient, LLMConfig
            m = get_llm_config(config, "reflection")
            client = LLMClient(LLMConfig(
                provider=m.get("provider", "openai"), model_name=m.get("model_name", ""),
                temperature=0.0, max_tokens=int(m.get("max_tokens", 512)),
                base_url=m.get("base_url", ""), api_key=m.get("api_key", ""),
                api_key_env=m.get("api_key_env", "OPENAI_API_KEY"),
                enable_thinking=m.get("enable_thinking", False),
            ))
            if client.enabled:
                scorer = LLMScorer(client)
        except Exception:  # noqa: BLE001
            pass
    return ReflectionModule(scorer=scorer, kb=kb)


def build_agents(config: Dict[str, Any], mcp, spec: SystemModeSpec, *, planner_decision: Optional[str] = None):
    """
    构建五个组件：planner, retriever, generator, validator, reflector(None 表示无反思)，
    并按 spec 设置归因引擎与 EIG 开关（进程级）。
    planner_decision 仅对 planner_strategy=active 有效（llm | eig），None 取 config.workflow.planner_decision。
    """
    from src.agents.planner import PlannerAgent
    from src.agents.retriever import RetrieverAgent
    from src.agents.generator import GeneratorAgent
    from src.agents.validator import ValidatorAgent

    apply_engine_settings(spec)
    planner = PlannerAgent(config, planner_mode=spec.planner_mode, allowed_tools=spec.allowed_tools,
                           strategy=spec.planner_strategy, decision=planner_decision)
    retriever = RetrieverAgent(config, mcp, allowed_tools=spec.allowed_tools)
    generator = GeneratorAgent(config, output_mode=spec.generator_output)
    validator = ValidatorAgent(config, claim_check=spec.validator_mode)
    reflector = build_reflector(config, spec)
    return planner, retriever, generator, validator, reflector
