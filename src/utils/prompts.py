"""
提示词模块（兼容层）。

提供与旧代码兼容的常量，同时也支持从模板文件动态加载。
优先使用 templates/ 目录下的模板文件，若不存在则回退到这里的默认值。

旧代码直接引用 PLANNER_SYSTEM_PROMPT 等常量无需改动。
"""

import warnings
from typing import Any, Dict, Optional

# ─── 默认提示词（模板文件不存在时的回退值）─────────────────────────
_PLANNER_SYSTEM = """你是一名选冶领域生产过程专家，擅长故障诊断与控制策略规划。
请根据用户问题，将诊断过程拆分为若干可执行的小步骤，并指明哪些步骤需要：
- 检索专业知识（控制方法、故障案例、设备说明）
- 调用时序分析工具（如电流、电压、振动、给矿量信号等）
"""

_GENERATOR_SYSTEM = """你是一名电力变压器故障诊断专家，负责根据检索到的专业知识和分析结果，
给出面向电力运维/检修人员的诊断结论与处理建议，要求：
- 语言简洁专业，符合电力行业规范
- 优先给出安全相关的建议，确保人员和设备安全
- 诊断结论要明确，区分不同置信度（确定/可能/待确认）
- 如果信息不足，要诚实说明并给出进一步检查建议
- 涉及停电检修时必须注明安全注意事项
"""


_GENERATOR_REVISION_SYSTEM = """你是一名电力变压器故障诊断专家。
Validator 对你的诊断草案提出了改进意见，请根据反馈重新生成诊断建议。

改进要求：
- 认真分析 Validator 指出的每个问题
- 保留原诊断中正确的部分
- 重点改进 Validator 指出的不足
- 生成更准确、更专业的诊断结论
"""

_PLANNER_SYSTEM = """你是一名电力变压器领域高级工程师，负责对用户提交的变压器故障或异常问题进行深度意图分析与任务规划。
请根据用户问题，判断问题类型并进行工具决策，诊断步骤通常 3-6 步。
"""

_VALIDATOR_SYSTEM = """你是一名电力变压器资深专家（高级工程师），负责对诊断结果进行严格质量检查。
请从以下维度进行评估：
- 是否与用户问题高度相关，诊断结论是否针对用户实际场景
- 是否充分参考了检索到的专业知识或分析数据
- 是否存在明显逻辑错误或安全风险
- 建议的处理措施是否符合电力安全规程

请以结构化 JSON 格式输出评估结果（不要用 markdown 代码块包裹）：
{
  "verdict": "PASS | REVISION | FAIL",
  "score": 0-10 的整数评分,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "improvement_suggestions": ["改进建议1", "改进建议2"],
  "summary": "总体评价的简要说明（50字以内）"
}

规则：
- PASS（通过）：score >= 7，无需改进
- REVISION（需改进）：score 4-6，建议重生成
- FAIL（失败）：score < 4，诊断结论无效，需重新检索和生成
"""


class _LazyTemplateLoader:
    """延迟加载的模板加载器，仅在第一次访问提示词时才尝试加载模板。"""

    _instance: "_LazyTemplateLoader | None" = None
    _loader: Any = None

    def __new__(cls) -> "_LazyTemplateLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_loader(self):
        if self._loader is None:
            try:
                from src.utils.template_loader import TemplateLoader
                self._loader = TemplateLoader()
            except Exception:
                self._loader = None
        return self._loader

    def get_system(self, agent: str, variant: Optional[str] = None) -> str:
        loader = self._get_loader()
        if loader:
            content = loader.load_system(agent, variant) if variant else loader.load_system(agent)
            if content:
                return content
            if variant:
                # 变体缺失时回退到基础模板
                content = loader.load_system(agent)
                if content:
                    return content
        # 回退到默认值
        fallbacks = {
            "planner": _PLANNER_SYSTEM,
            "generator": _GENERATOR_SYSTEM,
            "validator": _VALIDATOR_SYSTEM,
        }
        return fallbacks.get(agent, "")

    def render_user(self, agent: str, **kwargs: Any) -> str:
        loader = self._get_loader()
        if loader:
            return loader.render_user(agent, **kwargs)
        return ""


_loader = _LazyTemplateLoader()


def PLANNER_SYSTEM_PROMPT(allowed_tools: Optional[list] = None, variant: Optional[str] = None) -> str:
    """
    返回 Planner 系统提示词，并将模板中的 {{tools}} 占位符替换为工具注册表的清单文本。

    工具清单来自 src.tools.tool_registry（单一数据源），新增工具无需改动本提示词。
    allowed_tools 非 None 时只渲染名单内的工具（P6 五级模式）。
    variant="active" 时加载 templates/planner/system_active.txt（B-4 主动追问变体；缺失则回退基础模板）。
    """
    template = _loader.get_system("planner", variant)
    if "{{tools}}" in template:
        try:
            from src.tools.tool_registry import render_tools_for_prompt
            template = template.replace("{{tools}}", render_tools_for_prompt(allowed_tools))
        except Exception:
            template = template.replace("{{tools}}", "（工具清单加载失败）")
    if allowed_tools is not None:
        # 模板「工具选择原则」中形如「… → rag_search」的规则行，若指向被屏蔽的工具则整行删除
        try:
            from src.tools.tool_registry import get_tool_names
            disabled = [t for t in get_tool_names() if t not in set(allowed_tools)]
        except Exception:
            disabled = []
        if disabled:
            kept = []
            for line in template.splitlines():
                if any(("→ " + t) in line or ("→" + t) in line for t in disabled):
                    continue
                kept.append(line)
            template = "\n".join(kept)
    return template


def GENERATOR_SYSTEM_PROMPT(variant: Optional[str] = None) -> str:
    """
    variant="claims" 时加载 templates/generator/system_claims.txt（C-1 声明级输出；缺失则回退基础模板）。
    """
    return _loader.get_system("generator", variant)


def VALIDATOR_SYSTEM_PROMPT() -> str:
    return _loader.get_system("validator")


def render_planner_user(query: str, context: str = "（无额外上下文）") -> str:
    return _loader.render_user("planner", query=query, context=context)


def _generator_user_fallback(
    query: str,
    kb_text: str,
    kg_text: str,
    ts_text: str,
    attribution_text: str,
    forecast_text: str,
) -> str:
    """templates/generator/user.txt 缺失时的内置回退，保证 LLM 永远收到非空用户消息。"""
    return f"""请根据以下信息生成诊断建议：

【用户问题】
{query}

【知识库检索结果】
{kb_text}

【故障关系图谱】
{kg_text}

【时序分析结果】
{ts_text}

【故障归因分析】
{attribution_text}

【油温预测结果】
{forecast_text}

请按以下结构输出：
一、诊断结论（区分确定/可能/待确认，并说明依据来源）
二、处理与检修建议（安全优先，按优先级排序）
三、进一步检查建议
"""


def render_generator_user(
    query: str,
    kb_text: str,
    ts_text: str,
    attribution_text: str = "",
    forecast_text: str = "",
    kg_text: str = "",
) -> str:
    kwargs = dict(
        query=query,
        kb_text=kb_text or "（无知识库检索结果）",
        kg_text=kg_text or "（无图谱关系检索结果）",
        ts_text=ts_text or "（无时序分析结果）",
        attribution_text=attribution_text or "（无故障归因分析结果）",
        forecast_text=forecast_text or "（无油温预测分析结果）",
    )
    rendered = _loader.render_user("generator", **kwargs)
    if not rendered.strip():
        rendered = _generator_user_fallback(**kwargs)
    return rendered


def render_validator_user(draft_answer: str) -> str:
    return _loader.render_user("validator", draft_answer=draft_answer)


def GENERATOR_REVISION_SYSTEM_PROMPT() -> str:
    return _GENERATOR_REVISION_SYSTEM


def render_generator_revision_user(
    query: str,
    kb_text: str,
    ts_text: str,
    attribution_text: str = "",
    forecast_text: str = "",
    revision_feedback: str = "",
    previous_draft: str = "",
    kg_text: str = "",
) -> str:
    """
    渲染 Generator 重生成时的用户提示词。

    参数：
    - query: 用户原始问题
    - kb_text: 知识库检索结果
    - ts_text: 时序分析结果
    - attribution_text: 故障归因结果
    - forecast_text: 油温预测结果
    - revision_feedback: Validator 的改进建议
    - previous_draft: 上一版本的诊断草案
    - kg_text: 故障关系图谱检索结果
    """
    feedback_section = f"""
【Validator 改进建议】
{revision_feedback}
""" if revision_feedback else ""

    previous_section = f"""
【上一版本诊断草案】
{previous_draft}
""" if previous_draft else ""

    return f"""请根据以下信息重新生成诊断建议：

【用户问题】
{query}

【知识库检索结果】
{kb_text}

【故障关系图谱】
{kg_text or "（无）"}

【时序分析结果】
{ts_text}

【故障归因分析】
{attribution_text or "（无）"}

【油温预测结果】
{forecast_text or "（无）"}
{feedback_section}
{previous_section}

请生成改进后的诊断建议：
"""


def render_validator_structured_user(
    draft_answer: str,
    revision_context: str = "",
) -> str:
    """
    渲染 Validator 结构化评估的用户提示词。

    参数：
    - draft_answer: 待评估的诊断草案
    - revision_context: 重生成场景下的上下文（Validator 改进建议等）
    """
    context_section = f"""
【Validator 上一轮改进建议】
{revision_context}
""" if revision_context else ""

    return f"""请对以下诊断草案进行质量评估：

【诊断草案】
{draft_answer}
{context_section}

请以 JSON 格式输出评估结果：
"""
