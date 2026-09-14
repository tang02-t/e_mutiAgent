"""
统一的 LLM JSON 输出解析工具。

不同智能体（Planner / Validator 等）从大模型拿到的内容里，常见以下几种"脏"包装：
- ``<think>...</think>`` 思考过程
- ```` ```json ... ``` ```` 之类的 markdown 代码块
- JSON 前后夹带的解释性文字

本模块提供一个健壮的 :func:`parse_llm_json`，按多策略尝试解析，
避免在多个 agent 中重复维护各自的正则清洗逻辑。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from src.utils.logging import get_logger


logger = get_logger(__name__)


def strip_wrappers(text: str) -> str:
    """去除 <think> 标签与 markdown 代码块包装，返回尽可能"干净"的 JSON 文本。"""
    if not text:
        return ""

    cleaned = text.strip()

    # 1) 去除 <think>...</think> 思考内容（保留其后的正文）
    if "<think>" in cleaned:
        # 优先取最外层花括号之间的内容
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return cleaned[first_brace:last_brace + 1]

    # 2) 去除 markdown 代码块标记
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    return cleaned.strip()


def parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 原始输出中解析出 JSON 对象。

    返回解析得到的 dict；若所有策略均失败，返回 None（由调用方决定降级行为）。
    """
    if not content or not content.strip():
        return None

    candidates = [
        content,
        strip_wrappers(content),
    ]

    for text in candidates:
        if not text:
            continue
        # 优先尝试直接解析
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # 再尝试提取首个完整的 JSON 对象
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                continue

    logger.warning("parse_llm_json 解析失败，原始内容前 200 字符: %s", content[:200])
    return None
