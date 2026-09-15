from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """
    多智能体间共享的状态结构（支持 LangGraph 图式编排）。
    """

    # 用户输入
    user_query: str

    # 其它上下文（如设备 ID、工艺段信息）
    context: Dict[str, Any] = field(default_factory=dict)

    # ── Planner 输出 ────────────────────────────────────────────────
    plan: Optional[str] = None
    # 计划状态：ok / no_tool / parse_failed / llm_error / llm_disabled / PENDING
    plan_status: str = "PENDING"

    # ── Retriever 输出 ──────────────────────────────────────────────
    retrieved_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # 统一工具调用轨迹（每条包含 call_index / tool / raw_arguments / validation /
    # exec_success / business_success / error_code / latency_ms 等），供造数与评测使用
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    # ── Reflection 输出（P3，可选节点）──────────────────────────────
    # 每轮反思日志：scorer / rounds / kept / dropped / rewrite / latency_ms
    reflection_log: List[Dict[str, Any]] = field(default_factory=list)

    # ── Generator 输出 ──────────────────────────────────────────────
    draft_answer: Optional[str] = None
    # C-1 声明级输出（generator_output_mode=claims 时填充；模板回退亦按规则拼装）
    # 每条：{id, text, type: observation|inference|recommendation|safety, evidence: [{source, ref, span}]}
    draft_claims: List[Dict[str, Any]] = field(default_factory=list)
    # 声明产出方式：llm | rule | ""（text 模式未产出）；以及 LLM claims JSON 是否解析合法
    claims_source: str = ""
    claims_json_valid: Optional[bool] = None

    # ── Validator 输出 ──────────────────────────────────────────────
    validation_report: Optional[str] = None
    # 验证结论：PASS（通过）/ REVISION（需改进，触发重生成）/ FAIL（失败，直接输出）
    validation_verdict: str = "PENDING"
    # Validator 给出的改进建议（供 Generator 迭代重生成使用）
    revision_feedback: Optional[str] = None
    # 上一轮 Validator 输出的改进建议（仅用于 Validator 下一轮评估对比，避免回传整份报告）
    last_improvement_suggestions: Optional[str] = None

    # ── 工作流元数据 ────────────────────────────────────────────────
    # 当前迭代轮次（Generator 重生成次数）
    iteration: int = 0
    # 最大允许迭代次数（防止无限循环）
    max_iterations: int = 3
    # 各步骤的错误记录（用于排查）
    errors: List[Dict[str, Any]] = field(default_factory=list)
    # 是否使用 fallback 降级模式
    fallback_mode: bool = False

    # ── 推理轨迹（ReAct / Chain-of-Thought）─────────────────────────
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)

    # ── B-4 主动追问（planner_strategy=active / free 时启用）────────
    # 规划策略：free（不做主动决策）| active（按 EIG 推荐决定 ask / call_tool / conclude）
    planner_strategy: str = "free"
    # 最近一次主动决策动作：ask | call_tool | conclude | ""（尚未决策）
    planner_action: str = ""
    # 待向用户提出的问题（每条：symptom / question / rationale / eig / voi / how_to_obtain / tool）
    pending_questions: List[Dict[str, Any]] = field(default_factory=list)
    # 已追问过的征兆（无论是否得到回答），避免重复追问
    asked_symptoms: List[str] = field(default_factory=list)
    # 用户（或模拟器）给出的征兆真值：True / False；None 表示用户无法提供，该征兆视为不可获取
    user_answers: Dict[str, Optional[bool]] = field(default_factory=dict)
    # 已完成的追问轮数与上限 K
    inquiry_rounds: int = 0
    max_inquiry_rounds: int = 3
    # 每轮追问日志：round / entropy_bits / top1 / top1_prob / recommendations / action /
    # symptom / answer / cost / decision_source（llm | eig | rule）/ stop_reason
    inquiry_log: List[Dict[str, Any]] = field(default_factory=list)
    # 追问累计获取成本（按 symptom_cost.json）
    inquiry_cost: float = 0.0

    # ── 工作流最终结果 ──────────────────────────────────────────────
    final_answer: Optional[str] = None

    # ── 节点路由标记（LangGraph 内部使用）───────────────────────────
    # 标记下一步应该执行的节点
    next_node: str = "planner"

