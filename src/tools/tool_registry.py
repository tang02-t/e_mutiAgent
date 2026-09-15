"""
工具注册表（Tool Registry）。

集中声明所有可供智能体调用的工具元信息，作为「单一数据源」同时服务于：
1. Planner 的提示词 —— 让大模型清楚知道有哪些工具、各自的用途、参数与适用场景；
2. LLM 原生 Function Calling —— 直接把 `to_openai_tools()` 的结果传给 `chat(tools=...)`；
3. Retriever 的参数校验与默认值填充 —— 通过 `validate_arguments()` 规范模型给出的参数。

新增工具时，只需在 `TOOL_SPECS` 中追加一条声明，Planner / Retriever 即可自动感知，
无需再到各处修改 if/elif 分支或手写提示词。

JSON Schema 字段遵循 OpenAI tools（function calling）规范的 `parameters` 子集。
"""

from __future__ import annotations

from typing import Any, Dict, List


# ──────────────────────────────────────────────────────────────────────────────
# 工具声明
#   - description    : 给大模型看的用途说明（越具体，工具选择越准）
#   - when_to_use    : 适用场景，进一步降低误选概率
#   - parameters     : OpenAI tools 规范的 JSON Schema（type=object）
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "rag_search",
        "description": (
            "在变压器专业知识库（检修导则、故障案例、设备说明、标准规程）中做语义检索，"
            "返回最相关的知识片段。用于补充诊断所需的领域知识与处理建议依据。"
        ),
        "when_to_use": (
            "当用户问题涉及『如何处理 / 规程 / 标准 / 案例 / 依据』或需要文献支撑时调用；"
            "纯数据计算类问题（只要故障概率或油温数值）不必调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或问题，通常可直接用用户原始问题，必要时可改写以提升召回。",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "kg_search",
        "description": (
            "在变压器故障关系链路图谱中检索实体及其关系链路。图谱由文献抽取，节点类型包括故障、部件、"
            "症状、指标、方法、工况、处理措施、标准；关系包括 CAUSES（导致）、PRODUCES（产生气体）、"
            "INDICATES（症状指示故障）、LOCATED_IN（故障部位）、DETECTED_BY（检测手段）、"
            "TREATED_BY（处理措施）、SPECIFIED_IN（依据标准）。返回带文献支持数的关系路径。"
        ),
        "when_to_use": (
            "需要回答『X 会导致什么 / X 由什么引起 / X 发生在哪个部件 / X 如何检测或处理 / "
            "某症状指示哪些故障』这类结构化因果与关系问题时调用；已得到故障归因结果后，"
            "可用它补充部位、处理措施与复核手段。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要定位的实体或包含实体的短语，如『铁心多点接地』『乙炔升高』『突发短路』。",
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["CAUSES", "PRODUCES", "INDICATES", "LOCATED_IN",
                                 "DETECTED_BY", "TREATED_BY", "SPECIFIED_IN"],
                    },
                    "description": "限定关系类型；省略则返回全部关系。",
                },
                "hops": {
                    "type": "integer",
                    "description": "扩展跳数，1~3，默认 1。因果链追溯用 2。",
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "description": "out=从实体出发（X 导致什么），in=指向实体（什么导致 X），both=双向，默认 both。",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fault_attribution",
        "description": (
            "基于贝叶斯网络 + DL/T 722-2014 三比值法的故障归因引擎。输入 DGA 油色谱气体浓度"
            "（H2/CH4/C2H2/C2H4/C2H6，单位 μL/L）和/或征兆布尔标记，输出各故障类型的后验概率排序"
            "（如绕组短路、局部放电、过载过热、绝缘老化等）与诊断报告。"
        ),
        "when_to_use": (
            "当用户提供了 DGA 油色谱数据，或问题需要『判断故障类型 / 故障原因 / 哪种故障可能性最大』时必选。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dga_data": {
                    "type": "object",
                    "description": (
                        "DGA 气体浓度字典，单位 μL/L。字段：H2, CH4, C2H2, C2H4, C2H6。"
                        "若用户未提供具体数值，可留空（系统会使用前端填写的 DGA 数据）。"
                    ),
                    "properties": {
                        "H2": {"type": "number"},
                        "CH4": {"type": "number"},
                        "C2H2": {"type": "number"},
                        "C2H4": {"type": "number"},
                        "C2H6": {"type": "number"},
                    },
                },
                "evidence": {
                    "type": "object",
                    "description": (
                        "征兆布尔字典，可选。键为征兆 ID，值为 true 表示该征兆出现。"
                        "常用键：H2_elevated, CH4_elevated, C2H2_elevated, C2H4_elevated, "
                        "C2H6_elevated, oil_temp_elevated, winding_temp_elevated, "
                        "load_elevated, vibration_elevated, partial_discharge_alarm, "
                        "gas_rate_rapid, TDCG_elevated。"
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "用户原始问题描述，用于上下文。",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ett_forecast",
        "description": (
            "基于 ETT 电力变压器历史时序数据的油温（OT）预测工具。用线性回归拟合负载特征与油温关系，"
            "输出未来若干步的油温预测值、历史统计摘要与 3σ 异常检测结果。"
        ),
        "when_to_use": (
            "当用户问题涉及『油温趋势 / 预测 / 未来温度 / 过热风险预估 / 负载与温度关系』时选用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "enum": ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                    "description": "数据集：ETTh1/ETTh2 为小时级，ETTm1/ETTm2 为 15 分钟级。默认 ETTh1。",
                },
                "lookback": {
                    "type": "integer",
                    "description": "回看窗口大小（数据点数），默认 96。",
                },
                "horizon": {
                    "type": "integer",
                    "description": "预测步数，默认 24。",
                },
                "start_time": {
                    "type": "string",
                    "description": "可选，ISO 格式起始时间（如 2018-06-01），不填则取最新数据。",
                },
                "end_time": {
                    "type": "string",
                    "description": "可选，ISO 格式截止时间（如 2018-06-07）。",
                },
                "train_window_size": {
                    "type": "integer",
                    "description": "训练窗口大小（数据点数），默认 168。",
                },
            },
            "required": [],
        },
    },
    {
        "name": "timeseries_anomaly",
        "description": (
            "通用时序信号异常检测工具，对一段数值序列（如电流、振动、给矿量等）使用 3σ 规则"
            "检测异常点，返回均值、标准差与异常点索引。"
        ),
        "when_to_use": (
            "当需要对某条监测信号做快速异常筛查、且不属于油温预测（ett_forecast）场景时选用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "signal": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "待检测的数值序列（必填）。必须来自用户提供或上下文中明确给出的数据；"
                        "若用户未提供任何序列，不要调用本工具，而应说明缺少数据。"
                    ),
                },
            },
            "required": ["signal"],
        },
    },
]


# JSON Schema 基础类型 → Python 类型 的映射（bool 单独处理，避免被当作 int）
_JSON_TYPE_CHECK = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


# 便于按名称查找
_SPEC_BY_NAME: Dict[str, Dict[str, Any]] = {spec["name"]: spec for spec in TOOL_SPECS}


def get_tool_names() -> List[str]:
    """返回所有已声明工具的名称列表。"""
    return [spec["name"] for spec in TOOL_SPECS]


def get_tool_spec(name: str) -> Dict[str, Any] | None:
    """按名称返回工具声明，不存在时返回 None。"""
    return _SPEC_BY_NAME.get(name)


def to_openai_tools(names: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    转换为 OpenAI Chat Completions 的 `tools` 参数格式，可直接传给 LLMClient.chat(tools=...)。
    names 非 None 时只导出名单内的工具（P6 五级模式按需屏蔽 rag_search / kg_search）。
    """
    tools: List[Dict[str, Any]] = []
    for spec in TOOL_SPECS:
        if names is not None and spec["name"] not in names:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
        )
    return tools


def render_tools_for_prompt(names: List[str] | None = None) -> str:
    """
    渲染成给 Planner 系统提示词嵌入的工具清单文本（人类/模型友好）。
    names 非 None 时只渲染名单内的工具。
    """
    lines: List[str] = []
    specs = [s for s in TOOL_SPECS if names is None or s["name"] in names]
    for i, spec in enumerate(specs, start=1):
        lines.append(f"{i}. {spec['name']}")
        lines.append(f"   - 用途：{spec['description']}")
        if spec.get("when_to_use"):
            lines.append(f"   - 适用场景：{spec['when_to_use']}")

        props = spec.get("parameters", {}).get("properties", {})
        required = set(spec.get("parameters", {}).get("required", []))
        if props:
            lines.append("   - 参数：")
            for pname, pdef in props.items():
                req = "必填" if pname in required else "可选"
                ptype = pdef.get("type", "any")
                enum = pdef.get("enum")
                enum_str = f"，取值 {enum}" if enum else ""
                desc = pdef.get("description", "")
                lines.append(f"       · {pname}（{ptype}，{req}{enum_str}）：{desc}")
        else:
            lines.append("   - 参数：无")
        lines.append("")
    return "\n".join(lines).rstrip()


def validate_arguments_strict(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    依据工具的 JSON Schema 对参数做严格校验（不修正、不填默认值）。

    与旧版 validate_arguments 的区别：
    - 不静默丢弃/修正任何值；所有问题都以错误条目形式返回；
    - 保留模型给出的原始参数，便于评测时区分「模型错误」与「程序修正」。

    返回：
    {
      "valid": bool,
      "arguments": 原始参数（浅拷贝，未做任何修改）,
      "errors": [ {"field": str, "code": str, "message": str}, ... ],
    }
    错误码：
      unknown_tool / unknown_field / missing_required / invalid_enum / type_mismatch / invalid_item_type
    """
    raw = dict(arguments or {})
    errors: List[Dict[str, Any]] = []

    spec = get_tool_spec(name)
    if spec is None:
        errors.append({"field": None, "code": "unknown_tool", "message": f"未知工具：{name}"})
        return {"valid": False, "arguments": raw, "errors": errors}

    params = spec.get("parameters", {})
    props: Dict[str, Any] = params.get("properties", {})
    required = params.get("required", []) or []

    for req in required:
        if req not in raw or raw[req] is None:
            errors.append({"field": req, "code": "missing_required", "message": f"缺少必填参数 {req}"})

    for key, value in raw.items():
        if key not in props:
            errors.append({"field": key, "code": "unknown_field", "message": f"未声明的参数 {key}"})
            continue
        pdef = props[key]
        if value is None:
            # 可选参数显式为 null 视为未提供
            continue
        ptype = pdef.get("type")
        checker = _JSON_TYPE_CHECK.get(ptype)
        if checker and not checker(value):
            errors.append({
                "field": key,
                "code": "type_mismatch",
                "message": f"参数 {key} 期望类型 {ptype}，实际为 {type(value).__name__}",
            })
            continue
        enum = pdef.get("enum")
        if enum and value not in enum:
            errors.append({
                "field": key,
                "code": "invalid_enum",
                "message": f"参数 {key} 取值 {value!r} 不在允许范围 {enum}",
            })
        if ptype == "array":
            item_def = pdef.get("items") or {}
            item_type = item_def.get("type")
            item_checker = _JSON_TYPE_CHECK.get(item_type)
            if item_checker:
                bad = [i for i, v in enumerate(value) if not item_checker(v)]
                if bad:
                    errors.append({
                        "field": key,
                        "code": "invalid_item_type",
                        "message": f"参数 {key} 中第 {bad[:5]} 项不是 {item_type}",
                    })
            item_enum = item_def.get("enum")
            if item_enum:
                bad_vals = [v for v in value if v not in item_enum]
                if bad_vals:
                    errors.append({
                        "field": key,
                        "code": "invalid_enum",
                        "message": f"参数 {key} 中的元素 {bad_vals[:5]!r} 不在允许范围 {item_enum}",
                    })

    return {"valid": not errors, "arguments": raw, "errors": errors}


def validate_arguments(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    【已弃用】旧版宽松清洗：丢弃未声明键、丢弃非法枚举值。

    该行为会掩盖模型的参数错误，不应再用于 Planner / Retriever 主路径。
    仅为向后兼容保留；新代码请使用 validate_arguments_strict()。
    """
    spec = get_tool_spec(name)
    if spec is None:
        return dict(arguments or {})

    props: Dict[str, Any] = spec.get("parameters", {}).get("properties", {})
    cleaned: Dict[str, Any] = {}

    for key, value in (arguments or {}).items():
        if key not in props:
            continue
        pdef = props[key]
        enum = pdef.get("enum")
        if enum and value not in enum:
            continue
        cleaned[key] = value

    return cleaned
