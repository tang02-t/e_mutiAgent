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
    │          REVISION 且有可补证据 → Supplement → Retriever → Generator（C-3）│
    │          FAIL / ABSTAIN → 结束  |  max_iter → 结束（带警告）│
    └──────────────────────────────────────────────────────────────┘
                               │
                               ▼
                              END
"""

from typing import Literal, Callable, Dict, Any, Optional, List

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
    """Validator 节点：质量验证 + 路由决策（C-3：REVISION 且有可补证据时改走 supplement）。"""
    logger.info("[bold cyan]LangGraph Node: validator[/bold cyan]")
    state = validator.run(state)
    _finalize_supplement_cost(state)
    _decide_supplement_route(state, validator)
    return state


# ─── C-3 补证与重规划路由 ────────────────────────────────────────────────────

SUPPLEMENT_TOOLS = ("fault_attribution", "ett_forecast", "kg_search", "rag_search")
_ETT_DEFAULTS = {"dataset": "ETTh1", "lookback": 96, "horizon": 24}


def _canonical_args(tool: str, args: Dict[str, Any]) -> str:
    """工具参数规范化（补默认值、去 None、键排序），用于「同工具同参数」去重。"""
    import json
    clean = {k: v for k, v in (args or {}).items() if v is not None}
    if tool == "ett_forecast":
        clean = {**_ETT_DEFAULTS, **clean}
    if tool == "fault_attribution":
        clean.pop("query", None)  # query 仅作上下文，不影响归因结果
    return json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)


def _called_signatures(state: AgentState) -> set:
    sigs = set()
    for c in state.tool_calls or []:
        tool = c.get("tool")
        if not tool:
            continue
        args = c.get("args") if isinstance(c.get("args"), dict) else c.get("raw_arguments")
        if not isinstance(args, dict):
            args = {}
        sigs.add((tool, _canonical_args(tool, args)))
    return sigs


def _supplement_args(state: AgentState, tool: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """按待补证声明构造工具参数；返回 None 表示该工具对当前上下文必然无结果，应跳过。"""
    ctx = state.context or {}
    q = str(item.get("suggested_query") or "").strip() or state.user_query
    if tool == "fault_attribution":
        args: Dict[str, Any] = {"query": state.user_query}
        dga = ctx.get("dga")
        if isinstance(dga, dict):
            clean = {k: v for k, v in dga.items() if v is not None}
            if clean:
                args["dga_data"] = clean
        ev = ctx.get("evidence")
        if isinstance(ev, dict) and ev:
            args["evidence"] = {k: v for k, v in ev.items() if v is not None}
        return args
    if tool == "ett_forecast":
        args = {}
        for k in ("dataset", "lookback", "horizon"):
            if ctx.get(k) is not None:
                args[k] = ctx[k]
        return args
    if tool == "kg_search":
        attr = _latest_attribution(state)
        name = (attr or {}).get("primary_fault_name")
        candidates = [c for c in (name, q, state.user_query) if c]
        try:
            from src.tools.kg_search import get_kg
            kg = get_kg()
            for c in candidates:
                if kg.locate(c):
                    return {"query": c, "hops": 1}
            return None  # 图谱中无可定位实体：调用必然 no_match，跳过
        except Exception:  # noqa: BLE001
            return {"query": candidates[0] if candidates else q, "hops": 1}
    return {"query": q}


def _build_supplement_plan(state: AgentState, allowed_tools=None) -> Dict[str, Any]:
    """
    由 state.missing_evidence 构造合成补证计划（decision_source=supplement）。
    去重规则：跳过与本轮已执行调用「同工具同参数」的步骤，以及计划内部重复的步骤；
    跳过原因记入 skipped，供轨迹日志与验收统计。
    """
    from src.tools.tool_registry import validate_arguments_strict
    called = _called_signatures(state)
    steps: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    planned: set = set()
    for item in state.missing_evidence or []:
        tool = item.get("suggested_tool")
        cid = item.get("claim_id")
        if tool not in SUPPLEMENT_TOOLS:
            skipped.append({"claim_id": cid, "tool": tool, "reason": "no_tool"})
            continue
        if allowed_tools is not None and tool not in allowed_tools:
            skipped.append({"claim_id": cid, "tool": tool, "reason": "tool_disabled"})
            continue
        args = _supplement_args(state, tool, item)
        if args is None:
            skipped.append({"claim_id": cid, "tool": tool, "reason": "no_entity_in_kg"})
            continue
        sig = (tool, _canonical_args(tool, args))
        if sig in called:
            skipped.append({"claim_id": cid, "tool": tool, "reason": "duplicate_of_existing_call"})
            continue
        if sig in planned:
            skipped.append({"claim_id": cid, "tool": tool, "reason": "duplicate_in_plan"})
            continue
        planned.add(sig)
        steps.append({
            "id": len(steps) + 1, "stage": "补证", "tool": tool, "arguments": args,
            "description": f"为声明 {cid} 补充证据：{str(item.get('reason') or '')[:80]}",
            "for_claims": [cid],
            "validation": validate_arguments_strict(tool, args),
        })
    return {
        "intent_analysis": "（Validator 判定存在无依据声明，按建议工具补证）",
        "action": "call_tool" if steps else "conclude",
        "decision_source": "supplement",
        "rationale": f"第 {state.evidence_rounds + 1} 轮补证：{len(steps)} 步执行，{len(skipped)} 步去重/跳过",
        "steps": steps, "skipped": skipped,
        "plan_status": "ok" if steps else "empty", "planner_mode": "rule",
    }


def _decide_supplement_route(state: AgentState, validator) -> None:
    """
    C-3 路由决策（在 Validator 之后执行，写 state.next_node 与 state.route_log）：
    - ABSTAIN / PASS / FAIL → END（Validator 已决定）
    - REVISION 且 claim_check=route 且 missing_evidence 非空 且 未超补证轮数 → supplement
    - 其余 REVISION（矛盾 / 数据错误 / 安全违反）→ generator 修订
    """
    verdict = state.validation_verdict
    last_check = state.claim_check_log[-1] if state.claim_check_log else {}
    target = state.next_node
    reason = ""
    max_rounds = int(getattr(validator, "supplement_rounds", 1) or 0)
    route_mode = getattr(validator, "claim_check", "off") == "route"
    if verdict == "REVISION" and target == "generator":
        actionable = [m for m in (state.missing_evidence or []) if m.get("suggested_tool") in SUPPLEMENT_TOOLS]
        if not route_mode:
            reason = "revision:claim_check!=route"
        elif not state.missing_evidence:
            reason = "revision:contradict_or_constraint"  # 无可补证据：矛盾 / DATA / SAFETY / 整体评分
        elif not actionable:
            reason = "revision:no_actionable_evidence"  # 缺证声明均无可用建议工具 → 交 Generator 删改
        elif state.evidence_rounds >= max_rounds:
            reason = f"revision:supplement_rounds_exhausted({state.evidence_rounds}/{max_rounds})"
        else:
            target = "supplement"
            reason = f"unsupported:{len(actionable)} claims need evidence"
            state.next_node = "supplement"
    elif verdict == "ABSTAIN":
        reason = "abstain"
    elif verdict == "PASS":
        reason = "pass"
    elif verdict == "FAIL":
        reason = "fail"
    else:
        reason = f"{verdict.lower()}:{target}"
    state.route_log.append({
        "iteration": state.iteration, "verdict": verdict,
        "claim_check_verdict": last_check.get("verdict"),
        "unsupported_ratio": last_check.get("unsupported_ratio"),
        "n_missing": len(state.missing_evidence or []),
        "target": target, "reason": reason,
    })
    logger.info("Route decision: validator → %s (%s)", target, reason)


def _finalize_supplement_cost(state: AgentState) -> None:
    """补证轮结束（再次进入 Validator）时，回填该轮额外 Token / LLM 调用 / 耗时。"""
    import time
    from src.utils.llm import USAGE
    for entry in reversed(state.route_log or []):
        if entry.get("target") == "supplement" and "t_start" in entry and "latency_ms" not in entry:
            snap = USAGE.snapshot()
            entry["extra_tokens"] = snap["total_tokens"] - int(entry.pop("tokens_before", 0))
            entry["extra_llm_calls"] = snap["calls"] - int(entry.pop("calls_before", 0))
            entry["latency_ms"] = round((time.time() - entry.pop("t_start")) * 1000, 1)
            break


def _supplement_node(state: AgentState, allowed_tools=None) -> AgentState:
    """
    补证节点（规则型，不调 LLM）：把 missing_evidence 转成合成补证计划交给 Retriever；
    若去重后无可执行步骤，直接回 Generator 修订。补证后清空 draft_claims，让 Generator 基于扩充后的证据目录重建声明。
    """
    import time
    from src.utils.llm import USAGE
    logger.info("[bold cyan]LangGraph Node: supplement[/bold cyan]")
    plan = _build_supplement_plan(state, allowed_tools)
    state.evidence_rounds += 1
    snap = USAGE.snapshot()
    entry = state.route_log[-1] if state.route_log and state.route_log[-1].get("target") == "supplement" else None
    if entry is None:
        entry = {"iteration": state.iteration, "verdict": state.validation_verdict, "target": "supplement",
                 "reason": "supplement"}
        state.route_log.append(entry)
    entry.update({
        "round": state.evidence_rounds,
        "tools": [{"tool": s["tool"], "arguments": s["arguments"], "for_claims": s.get("for_claims")} for s in plan["steps"]],
        "skipped": plan["skipped"],
        "tokens_before": snap["total_tokens"], "calls_before": snap["calls"], "t_start": time.time(),
    })
    state.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": plan})
    state.context["missing_evidence"] = list(state.missing_evidence)
    # 反馈给 Generator：说明补了哪些证据 / 哪些无法补
    lines = [f"- 已为声明 {','.join(s.get('for_claims') or [])} 调用 {s['tool']} 补充证据" for s in plan["steps"]]
    lines += [f"- 声明 {k.get('claim_id')} 无法补证（{k.get('reason')}），请删除该声明或改为有据的表述" for k in plan["skipped"]]
    note = "【补证结果】\n" + "\n".join(lines) if lines else ""
    if note:
        state.revision_feedback = (state.revision_feedback or "") + ("\n\n" if state.revision_feedback else "") + note
    state.draft_claims = []
    state.missing_evidence = []
    if plan["steps"]:
        state.plan_status = "ok"
        state.planner_action = "call_tool"
        state.next_node = "retriever"
    else:
        logger.info("supplement: 去重后无可执行步骤，直接回 Generator 修订")
        state.next_node = "generator"
    return state


def _route_after_supplement(state: AgentState) -> str:
    nxt = "retriever" if state.next_node == "retriever" else "generator"
    logger.info("Route: supplement → %s", nxt)
    return nxt


def _is_supplement_round(state: AgentState) -> bool:
    return _latest_plan(state).get("decision_source") == "supplement"


# ─── B-4 主动追问循环 ────────────────────────────────────────────────────────

AnswerFn = Callable[[str, Dict[str, Any]], Optional[bool]]
"""回答函数签名：(symptom, question_meta) -> True | False | None（None 表示用户无法提供）。"""


def _latest_attribution(state: AgentState) -> Optional[Dict[str, Any]]:
    for c in reversed(state.tool_calls):
        if c.get("tool") == "fault_attribution" and c.get("success") and isinstance(c.get("result"), dict):
            return c["result"]
    return None


def _latest_plan(state: AgentState) -> Dict[str, Any]:
    for item in reversed(state.reasoning_trace):
        if item.get("agent") == "planner" and item.get("type") == "llm_plan":
            return item.get("content") or {}
    return {}


def _sync_uncertainty_to_context(state: AgentState) -> None:
    """把最新一次归因的 uncertainty / 已确认征兆写回 context，供下一轮 Planner 读取。"""
    attr = _latest_attribution(state)
    if attr is None:
        return
    unc = dict(attr.get("uncertainty") or {})
    unc["rounds_done"] = state.inquiry_rounds
    state.context["uncertainty"] = unc
    ev = dict(state.context.get("evidence") or {})
    for k in attr.get("evidence_used") or []:
        ev.setdefault(k, True)
    for k in attr.get("evidence_negative") or []:
        ev.setdefault(k, False)
    ev.update({k: v for k, v in state.user_answers.items() if v is not None})
    state.context["evidence"] = ev
    state.context["asked_symptoms"] = list(state.asked_symptoms)
    unavailable = [k for k, v in state.user_answers.items() if v is None]
    if unavailable:
        prev = list(state.context.get("unavailable_symptoms") or [])
        state.context["unavailable_symptoms"] = sorted(set(prev) | set(unavailable))


def _log_round(state: AgentState, action: str, *, symptom: Optional[str] = None,
               answer: Optional[bool] = None, cost: float = 0.0, plan: Optional[Dict[str, Any]] = None) -> None:
    unc = (state.context or {}).get("uncertainty") or {}
    plan = plan or _latest_plan(state)
    state.inquiry_log.append({
        "round": state.inquiry_rounds,
        "action": action,
        "decision_source": plan.get("decision_source"),
        "symptom": symptom,
        "answer": answer,
        "cost": cost,
        "entropy_bits": unc.get("entropy_bits"),
        "top1": unc.get("top1"),
        "top1_prob": unc.get("top1_prob"),
        "top1_top2_gap": unc.get("top1_top2_gap"),
        "recommendations": [
            {"symptom": r.get("symptom"), "eig": r.get("eig"), "cost": r.get("cost"), "voi": r.get("voi"),
             "how_to_obtain": r.get("how_to_obtain", "ask_user")}
            for r in (unc.get("recommendations") or [])[:5]
        ],
        "suggested_action": unc.get("suggested_action"),
        "stop_reason": unc.get("stop_reason"),
        "rationale": plan.get("rationale"),
        "tools": [s.get("tool") for s in plan.get("steps") or [] if isinstance(s, dict) and s.get("tool")],
    })


def _inquiry_node(state: AgentState, answer_fn: Optional[AnswerFn]) -> AgentState:
    """
    追问节点：处理 Planner 给出的 ask 动作。
      - 有 answer_fn（评测用模拟器 / 前端回调）：立即取回答，写入 user_answers，轮数 +1；
      - 无 answer_fn（交互式前端）：保留 pending_questions，标记 next_node=END 让上层中断等待用户。
    """
    logger.info("[bold cyan]LangGraph Node: inquiry[/bold cyan]")
    from src.tools.eig import symptom_cost

    if not state.pending_questions:
        state.next_node = "planner"
        return state
    q = state.pending_questions[0]
    symptom = q.get("symptom")
    if answer_fn is None:
        # 交互式：等待外部回答，本轮结束
        state.next_node = "END"
        return state
    try:
        raw = answer_fn(symptom, q)
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer_fn 异常：%s", exc)
        raw = None
    # 回答可以是 bool/None，也可以是 {"answer": bool|None, "dga": {...}, "evidence": {...}}：
    # 用户回答「乙炔 12 μL/L」时不仅确定 C2H2_elevated，还可能派生 TDCG / 产气速率等征兆
    extra_dga: Dict[str, Any] = {}
    extra_ev: Dict[str, Any] = {}
    if isinstance(raw, dict):
        answer = raw.get("answer")
        extra_dga = raw.get("dga") if isinstance(raw.get("dga"), dict) else {}
        extra_ev = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    else:
        answer = raw
    if answer is not None:
        answer = bool(answer)
    cost = float(q.get("cost") if q.get("cost") is not None else symptom_cost(symptom))
    if isinstance(raw, dict) and raw.get("cost") is not None:
        cost = float(raw["cost"])
    state.user_answers[symptom] = answer
    if symptom not in state.asked_symptoms:
        state.asked_symptoms.append(symptom)
    state.inquiry_rounds += 1
    state.inquiry_cost += cost
    _log_round(state, "ask", symptom=symptom, answer=answer, cost=cost)
    state.reasoning_trace.append({
        "agent": "inquiry", "type": "user_answer",
        "content": {"round": state.inquiry_rounds, "symptom": symptom, "question": q.get("question"),
                    "answer": answer, "cost": cost,
                    "extra_observations": sorted(set(extra_ev) | set(k for k, v in extra_dga.items() if v is not None))},
    })
    state.pending_questions = []
    # 回答写回 context
    ev = dict(state.context.get("evidence") or {})
    for k, v in extra_ev.items():
        if v is not None:
            ev[k] = bool(v)
    if answer is not None:
        ev[symptom] = answer
    state.context["evidence"] = ev
    if extra_dga:
        dga = dict(state.context.get("dga") or {})
        for k, v in extra_dga.items():
            if v is not None:
                dga[k] = v
        state.context["dga"] = dga
    state.context["asked_symptoms"] = list(state.asked_symptoms)
    if answer is None:
        prev = list(state.context.get("unavailable_symptoms") or [])
        state.context["unavailable_symptoms"] = sorted(set(prev) | {symptom})
    # 「追问 → 回答 → 重新归因」：注入一条合成的 fault_attribution 计划，直接交给 Retriever 执行，
    # 归因完成后再回到 Planner 决策（不占用 LLM 调用，也不计入追问轮数）
    state.reasoning_trace.append({
        "agent": "planner", "type": "llm_plan",
        "content": _reattribution_plan(state),
    })
    state.plan_status = "ok"
    state.planner_action = "call_tool"
    state.next_node = "retriever"
    return state


def _reattribution_plan(state: AgentState) -> Dict[str, Any]:
    """构造追问后重新归因的合成计划（带 validation，Retriever 可直接执行）。"""
    from src.tools.tool_registry import validate_arguments_strict
    ctx = state.context or {}
    args: Dict[str, Any] = {"query": state.user_query}
    dga = ctx.get("dga")
    if isinstance(dga, dict):
        clean = {k: v for k, v in dga.items() if v is not None}
        if clean:
            args["dga_data"] = clean
    ev = ctx.get("evidence")
    if isinstance(ev, dict) and ev:
        args["evidence"] = {k: v for k, v in ev.items() if v is not None}
    step = {"id": 1, "stage": "故障归因", "description": "追问得到新证据，重新归因并更新不确定性",
            "tool": "fault_attribution", "arguments": args,
            "validation": validate_arguments_strict("fault_attribution", args)}
    return {"intent_analysis": "（追问后重新归因）", "action": "call_tool", "decision_source": "reattribution",
            "rationale": f"纳入第 {state.inquiry_rounds} 轮追问回答后重新计算后验",
            "steps": [step], "plan_status": "ok", "planner_mode": "rule"}


def _planner_node_active(state: AgentState, planner) -> AgentState:
    """active 策略的 Planner 节点：运行 Planner 后做 K 轮上限 / 校验兜底，并把本轮决策写入 inquiry_log。"""
    logger.info("[bold cyan]LangGraph Node: planner (active)[/bold cyan]")
    state = planner.run(state)
    action = state.planner_action or ("call_tool" if state.plan_status == "ok" else "conclude")
    plan = _latest_plan(state)
    if action == "ask":
        if state.inquiry_rounds >= state.max_inquiry_rounds:
            logger.info("追问已达上限 K=%d，强制结论", state.max_inquiry_rounds)
            state.planner_action = "conclude"
            state.pending_questions = []
            _log_round(state, "conclude", plan={**plan, "rationale": f"达最大追问轮数 K={state.max_inquiry_rounds}"})
        elif not plan.get("action_validation", {}).get("valid", True) or not state.pending_questions:
            logger.info("ask 动作校验失败，直接结论")
            state.planner_action = "conclude"
            state.pending_questions = []
            _log_round(state, "conclude", plan={**plan, "rationale": "ask 校验失败：" + "; ".join(
                e.get("message", "") for e in plan.get("action_validation", {}).get("errors", []))})
    elif action == "call_tool" and state.plan_status != "ok":
        state.planner_action = "conclude"
        _log_round(state, "conclude", plan={**plan, "rationale": "call_tool 但无可执行步骤"})
    elif action != "call_tool":
        state.planner_action = "conclude"
        _log_round(state, "conclude", plan=plan)
    return state


def _retriever_node_active(state: AgentState, retriever) -> AgentState:
    """
    active 策略的 Retriever 节点：执行工具后记录 call_tool 轮，若本轮做了归因则回写 uncertainty
    并交回 Planner 重新决策（call_tool 轮不计入追问轮数）；否则进入结论。防御：工具轮数超过 2K+2 强制结束。
    """
    logger.info("[bold cyan]LangGraph Node: retriever (active)[/bold cyan]")
    state = retriever.run(state)
    plan = _latest_plan(state)
    if plan.get("decision_source") == "supplement":
        # C-3 补证轮：不重开追问循环，直接回 Generator 修订
        state.next_node = "generator"
        return state
    tools = [s.get("tool") for s in plan.get("steps") or [] if isinstance(s, dict)]
    n_tool_rounds = sum(1 for e in state.inquiry_log if e.get("action") == "call_tool")
    did_attr = "fault_attribution" in tools and _latest_attribution(state) is not None
    if did_attr:
        _sync_uncertainty_to_context(state)
    _log_round(state, "call_tool", plan=plan)
    # 由工具获取的征兆（how_to_obtain=call_tool:*）：工具结果不直接产出该征兆真值，记为已尝试，
    # 后续不再为它调用工具；若仍未进入 evidence，视为不可获取
    target = plan.get("symptom")
    if target and plan.get("decision_source") != "reattribution":
        if target not in state.asked_symptoms:
            state.asked_symptoms.append(target)
        state.context["asked_symptoms"] = list(state.asked_symptoms)
        if target not in (state.context.get("evidence") or {}):
            prev = list(state.context.get("unavailable_symptoms") or [])
            state.context["unavailable_symptoms"] = sorted(set(prev) | {target})
    if did_attr or target:
        if n_tool_rounds + 1 > 2 * state.max_inquiry_rounds + 2:
            logger.warning("工具轮数超限，强制结论")
            _log_round(state, "conclude", plan={**plan, "rationale": "工具轮数超限"})
            state.next_node = "generator"
        else:
            state.next_node = "planner"
    else:
        _log_round(state, "conclude", plan={**plan, "rationale": "工具执行完毕，进入结论"})
        state.next_node = "generator"
    return state


def _route_after_planner_active(state: AgentState) -> str:
    """active 策略下 Planner 之后的路由（只读）：call_tool → retriever；ask → inquiry；其余 → generator。"""
    action = state.planner_action
    if action == "ask" and state.pending_questions:
        logger.info("Route: planner → inquiry (symptom=%s)", state.pending_questions[0].get("symptom"))
        return "inquiry"
    if action == "call_tool" and state.plan_status == "ok":
        logger.info("Route: planner → retriever")
        return "retriever"
    logger.info("Route: planner → generator (action=%s)", action)
    return "generator"


def _route_after_retriever_active(state: AgentState) -> str:
    """active 策略下 Retriever 之后的路由（只读）：由 _retriever_node_active 写入 next_node。"""
    if state.next_node == "planner":
        nxt = "planner"
    elif _is_supplement_round(state):
        nxt = "generator_direct"  # C-3 补证轮：跳过 Reflection 直达 Generator
    else:
        nxt = "generator"
    logger.info("Route: retriever → %s", nxt)
    return nxt


# ─── 路由函数 ───────────────────────────────────────────────────────────────

def _route_after_validator(state: AgentState) -> str:
    """
    Validator 之后的路由决策：
    - PASS / FAIL / ABSTAIN → END
    - REVISION 且有可补证据（claim_check=route，C-3）→ supplement（补证后回 Generator）
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
    elif next_node == "supplement":
        logger.info("Route: validator → supplement (verdict=%s, missing=%d, round=%d)",
                    verdict, len(state.missing_evidence), state.evidence_rounds + 1)
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
    *,
    answer_fn: Optional[AnswerFn] = None,
) -> AgentState:
    """
    基于 LangGraph 的图式编排工作流。

    支持：
    - 动态路由（Validator 根据评估结果决定下一步）
    - 迭代优化（REVISION 时 Generator 重生成，最多 max_iterations 轮）
    - 可选 Reflection 节点（reflector 非 None 时插入 Retriever → Reflection → Generator）
    - B-4 主动追问循环（planner.strategy == "active"）：
        planner ─call_tool→ retriever ─(归因后)→ planner ─ask→ inquiry → planner … ─conclude→ generator
      最多 state.max_inquiry_rounds 轮追问；answer_fn 为回答来源（评测用模拟器），None 时遇 ask 中断等待用户。
    - 清晰的执行轨迹
    """
    active = getattr(planner, "strategy", "free") == "active"
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
        return _run_sequential_workflow(state, planner, retriever, generator, validator, reflector,
                                        answer_fn=answer_fn)

    # 构建图
    graph = StateGraph(AgentState)

    # 注册节点（使用 partial 绑定智能体实例）
    from functools import partial

    graph.add_node("planner", partial(_planner_node_active if active else _planner_node, planner=planner))
    graph.add_node("retriever", partial(_retriever_node_active if active else _retriever_node, retriever=retriever))
    graph.add_node("generator", partial(_generator_node, generator=generator))
    graph.add_node("validator", partial(_validator_node, validator=validator))

    # 设置入口和边
    graph.add_edge(START, "planner")
    after_tools = "generator"
    if reflector is not None:
        graph.add_node("reflection", partial(_reflection_node, reflector=reflector))
        graph.add_edge("reflection", "generator")
        after_tools = "reflection"

    if active:
        graph.add_node("inquiry", partial(_inquiry_node, answer_fn=answer_fn))
        graph.add_conditional_edges(
            "planner", _route_after_planner_active,
            {"retriever": "retriever", "inquiry": "inquiry", "generator": "generator"},
        )
        graph.add_conditional_edges(
            "retriever", _route_after_retriever_active,
            {"planner": "planner", "generator": after_tools, "generator_direct": "generator"},
        )
        graph.add_conditional_edges(
            "inquiry", lambda s: s.next_node,
            {"planner": "planner", "retriever": "retriever", "END": END},
        )
    else:
        graph.add_edge("planner", "retriever")
        # C-3 补证轮的 Retriever 直达 Generator（不重跑 Reflection）
        graph.add_conditional_edges(
            "retriever", lambda s: "generator" if _is_supplement_round(s) else "after_tools",
            {"after_tools": after_tools, "generator": "generator"},
        )
    graph.add_edge("generator", "validator")

    # C-3 补证节点：Validator REVISION 且有可补证据 → supplement → retriever → generator
    supplement_allowed = getattr(retriever, "allowed_tools", None)
    graph.add_node("supplement", partial(_supplement_node, allowed_tools=supplement_allowed))
    graph.add_conditional_edges(
        "supplement", _route_after_supplement,
        {"retriever": "retriever", "generator": "generator"},
    )

    # 条件路由：Validator → (generator | supplement | END)
    graph.add_conditional_edges(
        "validator",
        _route_after_validator,
        {
            "generator": "generator",      # REVISION → 迭代重生成
            "supplement": "supplement",    # REVISION 且可补证据 → 补证（C-3）
            "END": END,                    # PASS / FAIL / ABSTAIN / 达上限 → 结束
        }
    )

    # 编译图
    compiled_graph = graph.compile()

    # 执行图
    # 注意：LangGraph 的 invoke() 返回字典而非 AgentState 对象，需要转换
    # 追问循环节点较多，递归上限按 K 放宽
    result_dict = compiled_graph.invoke(state, config={"recursion_limit": 25 + 6 * state.max_inquiry_rounds})

    # 将字典转换为 AgentState
    final_state = AgentState(**result_dict)

    # 交互式 ask 中断：pending_questions 非空且未生成答案时不填 final_answer 兜底文案
    if final_state.pending_questions and final_state.draft_answer is None:
        q = final_state.pending_questions[0]
        final_state.final_answer = None
        final_state.plan_status = final_state.plan_status or "ok"
        logger.info("工作流在追问处中断，等待用户回答：%s", q.get("symptom"))
        return final_state

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
    *,
    answer_fn: Optional[AnswerFn] = None,
) -> AgentState:
    """
    简化版串行工作流（LangGraph 不可用时的降级方案）。
    支持 Validator REVISION 迭代与 C-3 补证路由；active 策略下仍执行追问循环（与图版同一套路由函数）。
    """
    logger.info("[bold cyan]Running sequential workflow (fallback mode)[/bold cyan]")
    active = getattr(planner, "strategy", "free") == "active"

    if not active:
        state = planner.run(state)
        state = retriever.run(state)
    else:
        guard = 0
        node = "planner"
        while node != "generator" and guard < 25 + 6 * state.max_inquiry_rounds:
            guard += 1
            if node == "planner":
                state = _planner_node_active(state, planner)
                node = _route_after_planner_active(state)
            elif node == "retriever":
                state = _retriever_node_active(state, retriever)
                node = _route_after_retriever_active(state)
            elif node == "inquiry":
                state = _inquiry_node(state, answer_fn)
                if state.next_node == "END":
                    return state  # 交互式中断
                node = state.next_node if state.next_node in ("planner", "retriever") else "planner"
    return _finish_generation(state, retriever, generator, validator, reflector)


def _finish_generation(state: AgentState, retriever, generator, validator, reflector=None) -> AgentState:
    """串行模式的收尾：Reflection → Generator → Validator，支持 REVISION 迭代与 C-3 补证路由（与图版同一套节点函数）。"""
    if reflector is not None:
        state = reflector.run(state)
    state = generator.run(state)
    state = _validator_node(state, validator)
    guard = 0
    supplement_allowed = getattr(retriever, "allowed_tools", None)
    while state.next_node in ("generator", "supplement") and guard < 2 * state.max_iterations + 4:
        guard += 1
        if state.next_node == "supplement":
            state = _supplement_node(state, supplement_allowed)
            if state.next_node == "retriever":
                state = retriever.run(state)
        state = generator.run(state)
        state = _validator_node(state, validator)

    if state.final_answer is None:
        state.final_answer = state.draft_answer or "(无结果)"

    return state


def resume_after_answer(state: AgentState, symptom: str, answer: Optional[bool],
                        planner, retriever, generator, validator, reflector=None) -> AgentState:
    """
    交互式前端：用户回答了 pending_questions 中的追问后，写入回答并继续工作流。
    answer=None 表示用户无法提供。
    """
    q = state.pending_questions[0] if state.pending_questions else {"symptom": symptom}
    if q.get("symptom") != symptom:
        logger.warning("resume_after_answer: symptom 不匹配（pending=%s, got=%s）", q.get("symptom"), symptom)
    state = _inquiry_node(state, lambda s, meta: answer)
    # 回答已注入合成的重新归因计划：先执行归因，再交回图（图入口是 planner，会基于新 uncertainty 决策）
    if state.next_node == "retriever":
        state = _retriever_node_active(state, retriever)
        if state.next_node != "planner":
            return _finish_generation(state, retriever, generator, validator, reflector)
    return run_langgraph_workflow(state, planner, retriever, generator, validator, reflector, answer_fn=None)


# ─── 向后兼容的简化接口 ─────────────────────────────────────────────────────

def run_diagnosis_workflow(
    state: AgentState,
    planner,
    retriever,
    generator,
    validator,
    reflector=None,
    *,
    answer_fn: Optional[AnswerFn] = None,
) -> AgentState:
    """
    简化的串行工作流入口（向后兼容）。

    注意：新版本优先使用 LangGraph，若需要迭代优化请使用 run_langgraph_workflow()。
    当前实现会自动检测 LangGraph 是否可用：
    - 可用：使用 LangGraph 工作流（支持迭代优化）
    - 不可用：降级到串行工作流
    reflector 为 None 时不启用反思节点（保持与基线一致）。
    answer_fn 仅在 planner.strategy == "active" 时使用（评测用模拟器回答追问）。
    """
    return run_langgraph_workflow(state, planner, retriever, generator, validator, reflector,
                                  answer_fn=answer_fn)
