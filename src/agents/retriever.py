"""
检索调用智能体（Retriever）。

P0 修复后的行为口径：
- 只执行 Planner 明确规划、且参数严格校验通过的工具；不再有「空计划自动执行固定工具」的兜底。
- 参数校验失败的步骤不执行，记录 exec_success=False / error_code=validation_failed。
- timeseries_anomaly 必须使用模型传入的 signal；缺失即报错，不再使用示例序列。
- 区分「调用成功」(exec_success：函数没抛异常) 与「业务成功」(business_success：工具返回不是业务错误)。
  旧字段 `success` 保留，语义等于 business_success，以兼容下游 Generator / 前端。
- 所有调用写入 state.trajectory（统一轨迹），包含原始参数、校验结果、耗时、错误码。

本智能体仅做工具编排，不直接调用 LLM。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

from src.graph.state import AgentState
from src.tools.mcp_client import MCPClient
from src.tools.tool_registry import validate_arguments_strict, get_tool_names
from src.utils.logging import get_logger


logger = get_logger(__name__)


_DEFAULT_MAX_PARALLEL_TOOLS = 3

# 工具返回中表示「业务失败」的 status 取值
_BUSINESS_ERROR_STATUS = {"error", "no_data", "failed", "no_match", "no_relation"}


def _judge_business_result(tool: str, result: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    根据工具返回判断业务是否成功。
    返回 (business_success, error_code, error_message)。
    """
    if tool == "rag_search":
        if isinstance(result, list):
            if not result:
                return False, "empty_result", "检索未返回任何片段"
            return True, None, None
        return False, "unexpected_result_type", f"rag_search 返回类型异常：{type(result).__name__}"

    if isinstance(result, dict):
        status = result.get("status")
        if status is None:
            # 没有 status 字段的字典，视为成功（部分旧工具）
            return True, None, None
        if status in _BUSINESS_ERROR_STATUS:
            return False, f"tool_{status}", str(result.get("message") or result.get("error") or status)
        return True, None, None

    if result is None:
        return False, "null_result", "工具返回 None"
    return True, None, None


class RetrieverAgent:
    def __init__(self, config: Dict[str, Any], mcp_client: MCPClient,
                 allowed_tools: Optional[List[str]] = None) -> None:
        self.config = config
        self.mcp = mcp_client
        # P6 五级模式：名单外工具即使 Planner 规划了也不执行，记录 error_code=tool_disabled
        self.allowed_tools: Optional[set] = set(allowed_tools) if allowed_tools is not None else None
        workflow_cfg = config.get("workflow", {}) if isinstance(config, dict) else {}
        self.max_parallel_tools = int(
            workflow_cfg.get("parallel_tools", _DEFAULT_MAX_PARALLEL_TOOLS)
        )

    # ──────────────────────────────────────────────────────────────
    # 单工具执行
    # ──────────────────────────────────────────────────────────────
    def _make_record(
        self,
        tool: str,
        raw_args: Dict[str, Any],
        *,
        validation: Optional[Dict[str, Any]] = None,
        exec_success: bool,
        business_success: bool,
        result: Any = None,
        error_code: Optional[str] = None,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
        call_id: Optional[str] = None,
        step_id: Any = None,
    ) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "tool": tool,
            "args": raw_args,                 # 兼容旧字段：现在保存的是模型原始参数
            "raw_arguments": raw_args,
            "validation": validation or {"valid": True, "errors": []},
            "result": result,
            "exec_success": exec_success,
            "business_success": business_success,
            "success": exec_success and business_success,   # 兼容旧字段
            "error_code": error_code,
            "error": error,
            "latency_ms": round(latency_ms, 2),
            "call_id": call_id,
            "step_id": step_id,
        }
        if tool == "rag_search":
            rec["result_count"] = len(result) if isinstance(result, list) else 0
        return rec

    def _execute_single_tool(
        self,
        tool: str,
        args: Dict[str, Any],
        state: AgentState,
        *,
        validation: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        step_id: Any = None,
    ) -> Dict[str, Any]:
        """执行单个工具，返回标准化调用记录。参数在此之前必须已经通过严格校验。"""
        t0 = time.perf_counter()
        try:
            if tool == "rag_search":
                query = args.get("query")
                result = self.mcp.call_tool("rag_search", query)

            elif tool == "timeseries_anomaly":
                signal = args.get("signal")
                if not isinstance(signal, list) or not signal:
                    # 校验层应已拦截；这里是二次保险，绝不再回退到示例序列
                    return self._make_record(
                        tool, args, validation=validation, exec_success=False,
                        business_success=False, error_code="missing_signal",
                        error="timeseries_anomaly 需要非空 signal 序列，未使用任何默认数据",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        call_id=call_id, step_id=step_id,
                    )
                result = self.mcp.call_tool("timeseries_anomaly", signal)

            elif tool == "ett_forecast":
                # 只传模型显式给出的参数，其余交给工具函数自身的默认值
                kwargs = {k: v for k, v in args.items() if v is not None}
                result = self.mcp.call_tool("ett_forecast", **kwargs)

            elif tool == "fault_attribution":
                dga_data = args.get("dga_data") or None
                evidence = args.get("evidence") or None
                query = args.get("query") or state.user_query
                kwargs = {"dga_data": dga_data, "evidence": evidence, "query": query}
                if getattr(state, "planner_strategy", "free") == "active":
                    # 追问循环：把已完成轮数传给引擎的停止准则；注册的工具函数不接受该参数时省略
                    fn = getattr(self.mcp, "_tools", {}).get("fault_attribution")
                    try:
                        import inspect
                        params = inspect.signature(fn).parameters
                        if "rounds_done" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
                            kwargs["rounds_done"] = int(getattr(state, "inquiry_rounds", 0))
                    except (TypeError, ValueError):
                        pass
                result = self.mcp.call_tool("fault_attribution", **kwargs)

            elif tool == "kg_search":
                kwargs = {k: v for k, v in args.items() if v is not None}
                result = self.mcp.call_tool("kg_search", **kwargs)

            else:
                return self._make_record(
                    tool, args, validation=validation, exec_success=False,
                    business_success=False, error_code="unknown_tool",
                    error=f"未知工具：{tool}",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    call_id=call_id, step_id=step_id,
                )

            latency = (time.perf_counter() - t0) * 1000
            biz_ok, err_code, err_msg = _judge_business_result(tool, result)
            return self._make_record(
                tool, args, validation=validation, exec_success=True,
                business_success=biz_ok, result=result,
                error_code=err_code, error=err_msg, latency_ms=latency,
                call_id=call_id, step_id=step_id,
            )

        except Exception as e:  # noqa: BLE001
            logger.warning("工具 %s 执行异常: %s", tool, e)
            return self._make_record(
                tool, args, validation=validation, exec_success=False,
                business_success=False, error_code="exception",
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                call_id=call_id, step_id=step_id,
            )

    # ──────────────────────────────────────────────────────────────
    # 并行执行
    # ──────────────────────────────────────────────────────────────
    def _execute_tools_parallel(
        self,
        tool_tasks: List[Dict[str, Any]],
        state: AgentState,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = [{} for _ in tool_tasks]
        kb_results_acc: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_parallel_tools) as executor:
            future_to_idx = {
                executor.submit(
                    self._execute_single_tool,
                    task["tool"],
                    task.get("arguments") or {},
                    state,
                    validation=task.get("validation"),
                    call_id=task.get("call_id"),
                    step_id=task.get("step_id"),
                ): idx
                for idx, task in enumerate(tool_tasks)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("并行工具执行异常（索引 %d）: %s", idx, e)
                    task = tool_tasks[idx]
                    result = self._make_record(
                        task.get("tool", "unknown"), task.get("arguments") or {},
                        validation=task.get("validation"), exec_success=False,
                        business_success=False, error_code="exception", error=str(e),
                        call_id=task.get("call_id"), step_id=task.get("step_id"),
                    )
                results[idx] = result
                if result.get("tool") == "rag_search" and result.get("business_success"):
                    kb_results_acc.extend(result.get("result") or [])

        return results, kb_results_acc

    # ──────────────────────────────────────────────────────────────
    # 入口
    # ──────────────────────────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        logger.info(
            "[yellow]Retriever Input:[/yellow] plan_status=%s plan=%s",
            state.plan_status,
            (state.plan[:100] + "...") if state.plan and len(state.plan) > 100 else state.plan,
        )

        # 读取 Planner 结构化 steps（取最近一轮：B-4 追问循环中 Planner 会多次运行）
        steps: List[Dict[str, Any]] = []
        plan_meta: Dict[str, Any] = {}
        for item in reversed(state.reasoning_trace):
            if item.get("agent") == "planner" and item.get("type") == "llm_plan":
                plan_meta = item.get("content") or {}
                steps = plan_meta.get("steps", []) or []
                break
        # C-3 补证轮：在既有证据上追加，而非覆盖
        is_supplement = plan_meta.get("decision_source") == "supplement"

        # 不再有 fallback：空计划就是「不调用工具」
        state.fallback_mode = False
        if not steps:
            logger.info("Retriever: 计划为空（status=%s），不执行任何工具。", state.plan_status)
            if not is_supplement:
                state.tool_calls = []
            state.reasoning_trace.append({
                "agent": "retriever",
                "type": "thought",
                "content": f"Planner 未规划工具调用（plan_status={state.plan_status}），Retriever 跳过执行。",
            })
            return state

        valid_tools = set(get_tool_names())
        runnable: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []

        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            raw_args = step.get("arguments") or {}
            desc = step.get("description", "")
            step_id = step.get("id")
            call_id = step.get("call_id")

            if not tool:
                state.reasoning_trace.append({"agent": "retriever", "type": "thought", "content": desc})
                continue

            # 严格校验：优先用 Planner 附带的结果，没有则现场校验
            validation = step.get("validation")
            if not isinstance(validation, dict):
                if tool in valid_tools and isinstance(raw_args, dict):
                    validation = validate_arguments_strict(tool, raw_args)
                else:
                    validation = {"valid": False, "errors": [
                        {"field": None, "code": "unknown_tool", "message": f"未知工具：{tool}"}]}

            if self.allowed_tools is not None and tool not in self.allowed_tools:
                rec = self._make_record(
                    tool, raw_args if isinstance(raw_args, dict) else {"_raw": raw_args},
                    validation={"valid": False, "errors": [
                        {"field": None, "code": "tool_disabled",
                         "message": f"工具 {tool} 在当前系统模式下不可用"}]},
                    exec_success=False, business_success=False,
                    error_code="tool_disabled", error=f"工具 {tool} 在当前系统模式下不可用",
                    call_id=call_id, step_id=step_id,
                )
                tool_calls.append(rec)
                logger.warning("Retriever 拒绝执行模式外工具 %s", tool)
                continue

            if not validation.get("valid", False):
                rec = self._make_record(
                    tool, raw_args if isinstance(raw_args, dict) else {"_raw": raw_args},
                    validation=validation, exec_success=False, business_success=False,
                    error_code="validation_failed",
                    error="; ".join(e.get("message", "") for e in validation.get("errors", [])),
                    call_id=call_id, step_id=step_id,
                )
                tool_calls.append(rec)
                state.errors.append({"agent": "retriever", "tool": tool,
                                     "error_code": "validation_failed", "error": rec["error"]})
                logger.warning("Retriever 跳过校验失败的工具调用 %s: %s", tool, rec["error"])
                continue

            runnable.append({
                "tool": tool, "arguments": raw_args, "validation": validation,
                "call_id": call_id, "step_id": step_id,
            })

        # 目前四个工具彼此独立，全部并行执行；顺序依赖留给后续「有限反馈规划」实现
        kb_results_acc: List[Dict[str, Any]] = []
        if runnable:
            logger.info(
                "Retriever parallel execution: %d tools (%s)",
                len(runnable), ", ".join(t["tool"] for t in runnable),
            )
            results, kb_results_acc = self._execute_tools_parallel(runnable, state)
            tool_calls.extend(results)

        # 写入 state（active 追问循环 / C-3 补证轮中 Retriever 会多次运行：累积而非覆盖，call_index 连续）
        accumulate = getattr(state, "planner_strategy", "free") == "active" or is_supplement
        base = len(state.tool_calls) if accumulate else 0
        for idx, rec in enumerate(tool_calls, start=base):
            rec["call_index"] = idx
            rec["inquiry_round"] = getattr(state, "inquiry_rounds", 0)
            state.reasoning_trace.append({"agent": "retriever", "type": "tool_call", "content": rec})
            state.trajectory.append({
                "call_index": idx,
                "tool": rec["tool"],
                "raw_arguments": rec["raw_arguments"],
                "validation": rec["validation"],
                "exec_success": rec["exec_success"],
                "business_success": rec["business_success"],
                "error_code": rec["error_code"],
                "error": rec["error"],
                "latency_ms": rec["latency_ms"],
                "call_id": rec.get("call_id"),
                "step_id": rec.get("step_id"),
                "inquiry_round": rec["inquiry_round"],
            })
            if not rec["success"] and rec["error_code"] != "validation_failed":
                state.errors.append({
                    "agent": "retriever", "tool": rec["tool"],
                    "error_code": rec["error_code"], "error": rec["error"] or "unknown",
                })

        if kb_results_acc:
            if is_supplement and state.retrieved_knowledge:
                seen = {str(k.get("chunk_id") or k.get("id") or k.get("text", "")[:80]) for k in state.retrieved_knowledge}
                extra = [k for k in kb_results_acc
                         if str(k.get("chunk_id") or k.get("id") or k.get("text", "")[:80]) not in seen]
                state.retrieved_knowledge = list(state.retrieved_knowledge) + extra
            else:
                state.retrieved_knowledge = kb_results_acc
        state.tool_calls = (list(state.tool_calls) + tool_calls) if accumulate else tool_calls

        n_ok = sum(1 for r in tool_calls if r["success"])
        logger.info(
            "Retriever finished: %d KB hits, %d tool calls (%d success, %d failed).",
            len(state.retrieved_knowledge), len(tool_calls), n_ok, len(tool_calls) - n_ok,
        )
        return state
