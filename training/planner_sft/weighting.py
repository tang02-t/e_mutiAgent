# -*- coding: utf-8 -*-
"""
P5-2 损失加权核心（纯 Python，不依赖 ms-swift / torch，可在本地单测）。

规则（对应 PLAN P5-2「损失加权」与叶金涛论文 custom_agent 思路）：
  权重 0：输入（system/user）与工具返回（tool / tool_response）——由 ms-swift 的 loss_scale 机制在 context_type 层面完成，
          本模块只处理 assistant 响应文本；
  权重 1：普通文本（意图分析、思考说明、自由文本参数值）；
  权重 2：工具调用结构 token：<tool_call> / </tool_call> 标签、JSON 键（"name": / "arguments": / "tool": ...）、
          花括号方括号、键值分隔符；
  权重 3：领域关键标识：工具名、参数名、枚举值、故障/征兆 ID 与中文名、DGA 气体符号、ETT 数据集名、
          图谱实体规范名与同义词（training/planner_sft/domain_terms.json，由 build_domain_terms.py 生成）。

对外接口：
  split_weighted(text, struct_weight=2.0, domain_weight=3.0, domain_terms=None) -> (segments, weights)
      把一段响应文本切成若干连续片段并给出每段权重；片段拼接后与原文完全相同（ms-swift loss_scale 契约）。
  load_domain_terms(path=None) -> List[str]
  normalize_weights(weights, counts) -> 归一化说明见 docstring
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_TERMS_PATH = HERE / "domain_terms.json"

W_TEXT = 1.0
W_STRUCT = 2.0
W_DOMAIN = 3.0

# 结构 token：标签、JSON 键（带引号与冒号）、括号
_STRUCT_PATTERNS = [
    r"</?tool_call>",
    r"</?tool_response>",
    r"\"[A-Za-z_][A-Za-z0-9_]*\"\s*:",   # "key":
    r"[{}\[\]]",
]
_STRUCT_RE = re.compile("|".join(f"(?:{p})" for p in _STRUCT_PATTERNS))

# 自由文本参数：这些键的字符串值视为普通文本（权重 1），其内部的领域词仍可升为 3
FREE_TEXT_KEYS = ("query", "description", "intent_analysis", "content", "stage")


def load_domain_terms(path: Optional[Path] = None) -> List[str]:
    p = Path(path) if path else DEFAULT_TERMS_PATH
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    terms = data.get("terms") if isinstance(data, dict) else data
    return sorted({t for t in terms if t}, key=lambda s: (-len(s), s))


def _compile_terms(terms: Sequence[str]) -> Optional[re.Pattern]:
    if not terms:
        return None
    parts = []
    for t in terms:  # 已按长度降序，保证最长匹配优先
        esc = re.escape(t)
        if re.fullmatch(r"[A-Za-z0-9_./μ-]+", t) and len(t) < 4:
            # 短 ASCII 词（in/out/both/H2/CO 等）要求词边界，避免误命中普通英文
            esc = r"(?<![A-Za-z0-9_])" + esc + r"(?![A-Za-z0-9_])"
        parts.append(esc)
    return re.compile("|".join(parts))


def _mark(spans: List[float], start: int, end: int, weight: float, override: bool) -> None:
    for i in range(start, end):
        if override or spans[i] < weight:
            spans[i] = weight


def char_weights(text: str, struct_weight: float = W_STRUCT, domain_weight: float = W_DOMAIN,
                 domain_terms: Optional[Sequence[str]] = None, text_weight: float = W_TEXT) -> List[float]:
    """逐字符权重（后续按连续相等权重合并为片段）。优先级：领域 3 > 结构 2 > 文本 1。"""
    n = len(text)
    w = [text_weight] * n
    if n == 0:
        return w
    for m in _STRUCT_RE.finditer(text):
        _mark(w, m.start(), m.end(), struct_weight, override=True)
    # JSON 中非自由文本键的标量值（工具名、枚举、数值、布尔）也视作结构/参数 token（权重 2）
    for m in re.finditer(r"\"([A-Za-z_][A-Za-z0-9_]*)\"\s*:\s*(\"(?:[^\"\\]|\\.)*\"|-?\d+(?:\.\d+)?|true|false|null)", text):
        key = m.group(1)
        if key in FREE_TEXT_KEYS:
            continue
        _mark(w, m.start(2), m.end(2), struct_weight, override=False)
    pat = _compile_terms(domain_terms if domain_terms is not None else load_domain_terms())
    if pat is not None:
        for m in pat.finditer(text):
            _mark(w, m.start(), m.end(), domain_weight, override=True)
    return w


def split_weighted(text: str, struct_weight: float = W_STRUCT, domain_weight: float = W_DOMAIN,
                   domain_terms: Optional[Sequence[str]] = None,
                   text_weight: float = W_TEXT) -> Tuple[List[str], List[float]]:
    """把响应切成片段与权重；保证 ''.join(segments) == text。"""
    if not text:
        return [text], [text_weight]
    w = char_weights(text, struct_weight, domain_weight, domain_terms, text_weight)
    segs: List[str] = []
    ws: List[float] = []
    start = 0
    for i in range(1, len(text) + 1):
        if i == len(text) or w[i] != w[start]:
            segs.append(text[start:i])
            ws.append(w[start])
            start = i
    return segs, ws


def summarize(segments: Iterable[str], weights: Iterable[float]) -> dict:
    """统计各权重覆盖的字符数（用于 dry-run 检查与数据卡片）。"""
    out: dict = {}
    for s, w in zip(segments, weights):
        out[w] = out.get(w, 0) + len(s)
    total = sum(out.values()) or 1
    return {"chars_by_weight": {str(k): v for k, v in sorted(out.items())},
            "ratio_by_weight": {str(k): round(v / total, 4) for k, v in sorted(out.items())}}


def normalized_weighted_ce(losses: Sequence[float], weights: Sequence[float]) -> float:
    """
    归一化加权交叉熵（参考实现，供单测与文档说明）：
        L = Σ_i w_i·CE_i / Σ_i w_i
    与 ms-swift 默认的 Σ w_i·CE_i / N（按 token 数归一）不同：按权重和归一后，加权不会整体放大 loss 量级，
    学习率无需随权重方案改动而调整；训练侧对应 plugin_loss_scale.py 中注册的 loss_func。
    """
    num = sum(l * w for l, w in zip(losses, weights))
    den = sum(weights)
    return num / den if den > 0 else 0.0
