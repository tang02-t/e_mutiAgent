"""
工作流编排模块（支持 LangGraph 图式编排）。

提供两种工作流模式：
1. run_diagnosis_workflow()：简化版串行工作流（向后兼容）
2. run_langgraph_workflow()：LangGraph 图式编排，支持动态路由和迭代优化

LangGraph 工作流架构：
    ┌──────────────────────────────────────────────────────────────┐
    │                         START                                 │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                       Planner                                │
    │              意图分析 + 工具决策（一次执行）                  │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                       Retriever                              │
    │          并行工具调用（RAG / 时序分析 / 故障归因）           │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                       Generator                              │
    │              生成诊断草案（首次 / 改进版）                    │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                       Validator                              │
    │          PASS → 结束  |  REVISION → Generator（迭代）       │
    │          FAIL → 结束  |  max_iter → 结束（带警告）          │
    └──────────────────────────────────────────────────────────────┘
                               │
                               ▼
                              END
"""

from typing import Literal, Callable, Dict, Any, Optional

from src.graph.state import AgentState
from src.utils.logging import get_logger


logger = get_logger(__name__)


# ─── 节点函数定义 ────────────────────────────────────────────────────────────

def _planner_node(state: AgentState, planner) -> AgentState:
    """Planner 节点：意图分析 + 工具决策。"""
    logger.info("[bold cyan]LangGraph Node: planner[/bold cyan]")
    return planner.run(state)


def _retriever_node(state: AgentState, retriever) -> AgentState:
    """Retriever 节点：并行工具调用。"""
    logger.info("[bold cyan]LangGraph Node: retriever[/bold cyan]")
    return retriever.run(state)


def _reflection_node(state: AgentState, reflector) -> AgentState:
    """Reflection 节点（P3，可选）：对检索片段评分、丢弃低分块、同章节补召回。"""
    logger.info("[bold cyan]LangGraph Node: reflection[/bold cyan]")
    return reflector.run(state)


def _generator_node(state: AgentState, generator, revision_feedback: str = "") -> AgentState:
    """Generator 节点：生成诊断草案。"""
    logger.info(
        "[bold cyan]LangGraph Node: generator[/bold cyan] (revision_feedback=%s)",
        "yes" if revision_feedback else "no"
    )
    return generator.run(state, revision_feedback=revision_feedback)


def _validator_node(state: AgentState, validator) -> AgentState:
    """Validator 节点：质量验证 + 路由决策。"""
    logger.info("[bold cyan]LangGraph Node: validator[/bold cyan]")
    return validator.run(state)


# ─── 路由函数 ───────────────────────────────────────────────────────────────

def _route_after_validator(state: AgentState) -> str:
    """
    Validator 之后的路由决策：
    - PASS / FAIL → END
    - REVISION 且未达上限 → generator（迭代）
    - REVISION 但已达上限 → END
    """
    verdict = state.validation_verdict
    next_node = state.next_node

    if next_node == "END":
        logger.info("Route: validator → END (verdict=%s)", verdict)
    elif next_node == "generator":
        logger.info(
            "Route: validator → generator (verdict=%s, iteration=%d/%d)",
            verdict, state.iteration, state.max_iterations
        )
    else:
        logger.warning("Route: validator → END (unknown next_node=%s)", next_node)
        next_node = "END"

    return next_node


# ─── LangGraph 工作流 ────────────────────────────────────────────────────────

def run_langgraph_workflow(
    state: AgentState,
    planner,
    retriever,
    generator,
    validator,
    reflector=None,
) -> AgentState:
    """
    基于 LangGraph 的图式编排工作流。

    支持：
    - 动态路由（Validator 根据评估结果决定下一步）
    - 迭代优化（REVISION 时 Generator 重生成，最多 max_iterations 轮）
    - 可选 Reflection 节点（reflector 非 None 时插入 Retriever → Reflection → Generator）
    - 清晰的执行轨迹
    """
    try:
        from langgraph.graph import StateGraph, END, START
        _LANGGRAPH_AVAILABLE = True
    except ImportError:
        logger.warning(
            "LangGraph 未安装，将使用简化版串行工作流。"
            "安装方式：pip install langgraph"
        )
        _LANGGRAPH_AVAILABLE = False

    if not _LANGGRAPH_AVAILABLE:
        # 降级到串行工作流
        return _run_sequential_workflow(state, planner, retriever, generator, validator, reflector)

    # 构建图
    graph = StateGraph(AgentState)

    # 注册节点（使用 partial 绑定智能体实例）
    from functools import partial

    graph.add_node("planner", partial(_planner_node, planner=planner))
    graph.add_node("retriever", partial(_retriever_node, retriever=retriever))
    graph.add_node("generator", partial(_generator_node, generator=generator))
    graph.add_node("validator", partial(_validator_node, validator=validator))

    # 设置入口和边
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    if reflector is not None:
        graph.add_node("reflection", partial(_reflection_node, reflector=reflector))
        graph.add_edge("retriever", "reflection")
        graph.add_edge("reflection", "generator")
    else:
        graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "validator")

    # 条件路由：Validator → (generator | END)
    graph.add_conditional_edges(
        "validator",
        _route_after_validator,
        {
            "generator": "generator",  # REVISION → 迭代重生成
            "END": END,                 # PASS / FAIL / 达上限 → 结束
        }
    )

    # 编译图
    compiled_graph = graph.compile()

    # 注入配置：每次 generator 节点需要传递 revision_feedback
    # 由于 AgentState 包含 revision_feedback（在 validator 评估后由 validator 写入），
    # 我们通过 state.validation_report 传递改进建议

    # 执行图
    # 注意：LangGraph 的 invoke() 返回字典而非 AgentState 对象，需要转换
    result_dict = compiled_graph.invoke(state)

    # 将字典转换为 AgentState
    from dataclasses import dataclass, fields
    final_state = AgentState(**result_dict)

    # 确保 final_answer 被填充
    if final_state.final_answer is None:
        final_state.final_answer = final_state.draft_answer or "(无结果)"

    return final_state


def _run_sequential_workflow(
    state: AgentState,
    planner,
    retriever,
    generator,
    validator,
    reflector=None,
) -> AgentState:
    """
    简化版串行工作流（LangGraph 不可用时的降级方案）。
    不支持迭代优化，仅执行一次完整流程。
    """
    logger.info("[bold cyan]Running sequential workflow (fallback mode)[/bold cyan]")

    state = planner.run(state)
    state = retriever.run(state)
    if reflector is not None:
        state = reflector.run(state)
    state = generator.run(state)
    state = validator.run(state)

    if state.final_answer is None:
        state.final_answer = state.draft_answer or "(无结果)"

    return state


# ─── 向后兼容的简化接口 ─────────────────────────────────────────────────────

def run_diagnosis_workflow(
    state: AgentState,
    planner,
    retriever,
    generator,
    validator,
    reflector=None,
) -> AgentState:
    """
    简化的串行工作流入口（向后兼容）。

    注意：新版本优先使用 LangGraph，若需要迭代优化请使用 run_langgraph_workflow()。
    当前实现会自动检测 LangGraph 是否可用：
    - 可用：使用 LangGraph 工作流（支持迭代优化）
    - 不可用：降级到串行工作流
    reflector 为 None 时不启用反思节点（保持与基线一致）。
    """
    return run_langgraph_workflow(state, planner, retriever, generator, validator, reflector)
