"""
P6 五级系统模式定义（system modes）。

论文口径（PLAN P6-2）：
  mode1  no_rag        无 RAG：Planner 只能调用数值工具（fault_attribution / ett_forecast / timeseries_anomaly），
                        不查文献、不查图谱、无反思。
  mode2  naive_rag     朴素 RAG：开放 rag_search，但知识库走单路 BM25（LocalKBRetriever mode=naive），
                        无图谱、无反思，Planner 为通用基座。
  mode3  planner_ft    + 规划微调：在 mode2 基础上 Planner 切换为 llms.planner_finetuned（P5 产物），
                        同时知识库升级为两路检索（P1-6 正式配置 two_way）。
  mode4  kg            + 图谱：开放 kg_search（P2）。
  mode5  full          + 反思：插入 ReflectionModule（P3），即完整系统。

每级只在上一级基础上打开一个能力，便于做增量归因。所有开关都能被显式参数覆盖
（例如 `--planner-mode baseline` 可在 mode3~5 上做“无微调”的对照）。

用法：
    from src.graph.system_modes import SYSTEM_MODES, resolve_mode, build_agents
    spec = resolve_mode("mode4")
    planner, retriever, generator, validator, reflector = build_agents(config, mcp, spec)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


ALL_TOOLS = ["rag_search", "kg_search", "fault_attribution", "ett_forecast", "timeseries_anomaly"]
NUMERIC_TOOLS = ["fault_attribution", "ett_forecast", "timeseries_anomaly"]


@dataclass(frozen=True)
class SystemModeSpec:
    key: str                      # mode1..mode5
    name: str                     # 英文短名
    label: str                    # 中文展示名
    allowed_tools: List[str]      # Planner 可见 / Retriever 可执行的工具
    rag_mode: Optional[str]       # LocalKBRetriever mode：None 表示不启用 rag_search
    planner_mode: str             # baseline | finetuned
    reflection: str               # off | lexical | llm
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "label": self.label,
            "allowed_tools": list(self.allowed_tools), "rag_mode": self.rag_mode,
            "planner_mode": self.planner_mode, "reflection": self.reflection,
        }


SYSTEM_MODES: Dict[str, SystemModeSpec] = {
    "mode1": SystemModeSpec(
        key="mode1", name="no_rag", label="模式 1：无 RAG",
        allowed_tools=list(NUMERIC_TOOLS), rag_mode=None,
        planner_mode="baseline", reflection="off",
        description="仅数值工具，答案完全依赖模型自身知识与工具计算结果。",
    ),
    "mode2": SystemModeSpec(
        key="mode2", name="naive_rag", label="模式 2：朴素 RAG",
        allowed_tools=NUMERIC_TOOLS + ["rag_search"], rag_mode="naive",
        planner_mode="baseline", reflection="off",
        description="开放文献检索，单路 BM25、按标题分块基线口径；无图谱、无反思。",
    ),
    "mode3": SystemModeSpec(
        key="mode3", name="planner_ft", label="模式 3：+ 规划微调",
        allowed_tools=NUMERIC_TOOLS + ["rag_search"], rag_mode="two_way",
        planner_mode="finetuned", reflection="off",
        description="Planner 切换为 LoRA 微调模型，知识库升级为两路检索（子块 + 摘要 + BM25 + 锚点 RRF）。",
    ),
    "mode4": SystemModeSpec(
        key="mode4", name="kg", label="模式 4：+ 故障图谱",
        allowed_tools=NUMERIC_TOOLS + ["rag_search", "kg_search"], rag_mode="two_way",
        planner_mode="finetuned", reflection="off",
        description="在模式 3 上开放 kg_search 多跳关系检索。",
    ),
    "mode5": SystemModeSpec(
        key="mode5", name="full", label="模式 5：完整系统（+ 反思）",
        allowed_tools=list(ALL_TOOLS), rag_mode="two_way",
        planner_mode="finetuned", reflection="lexical",
        description="在模式 4 上插入反思模块（评分过滤 + 邻块补召回 + 改写重检索）。",
    ),
}

MODE_ORDER = ["mode1", "mode2", "mode3", "mode4", "mode5"]
_ALIAS = {spec.name: key for key, spec in SYSTEM_MODES.items()}
_ALIAS.update({"1": "mode1", "2": "mode2", "3": "mode3", "4": "mode4", "5": "mode5"})


def resolve_mode(mode: str, *, planner_mode: Optional[str] = None,
                 reflection: Optional[str] = None, rag_mode: Optional[str] = None) -> SystemModeSpec:
    """按 key / 英文名 / 数字解析模式，并允许显式覆盖子开关（用于对照实验）。"""
    key = _ALIAS.get(str(mode), str(mode))
    if key not in SYSTEM_MODES:
        raise ValueError(f"未知系统模式 {mode!r}，可选 {MODE_ORDER} 或 {sorted(_ALIAS)}")
    spec = SYSTEM_MODES[key]
    overrides: Dict[str, Any] = {}
    if planner_mode:
        overrides["planner_mode"] = planner_mode
    if reflection:
        overrides["reflection"] = reflection
    if rag_mode and spec.rag_mode is not None:
        overrides["rag_mode"] = rag_mode
    return replace(spec, **overrides) if overrides else spec


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


def build_agents(config: Dict[str, Any], mcp, spec: SystemModeSpec):
    """构建五个组件：planner, retriever, generator, validator, reflector(None 表示无反思)。"""
    from src.agents.planner import PlannerAgent
    from src.agents.retriever import RetrieverAgent
    from src.agents.generator import GeneratorAgent
    from src.agents.validator import ValidatorAgent

    planner = PlannerAgent(config, planner_mode=spec.planner_mode, allowed_tools=spec.allowed_tools)
    retriever = RetrieverAgent(config, mcp, allowed_tools=spec.allowed_tools)
    generator = GeneratorAgent(config)
    validator = ValidatorAgent(config)
    reflector = build_reflector(config, spec)
    return planner, retriever, generator, validator, reflector
