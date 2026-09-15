#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5-2：生成「领域关键标识」词典 domain_terms.json（供损失加权插件 M3 使用）。

来源（全部来自项目内已有资产，不手工编造）：
  1. 工具 Schema（src/tools/tool_registry.py）：工具名、参数名、枚举取值
  2. 故障归因引擎（src/tools/fault_attribution.py）：故障 ID / 中文名、征兆 ID
  3. 图谱同义词表（data/kg/synonyms.json）：8 类实体的规范名与同义词
  4. 图谱节点（data/kg/graph.json）：节点名
  5. DGA 气体与 ETT 数据集标识

输出结构：
  {"version": ..., "sources": {...}, "groups": {"tool_name": [...], "param_name": [...], "enum_value": [...],
   "fault": [...], "symptom": [...], "kg_entity": [...], "gas": [...], "dataset": [...]}, "terms": [去重后全量]}

用法：python3 training/planner_sft/build_domain_terms.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.tool_registry import get_tool_names, get_tool_spec  # noqa: E402
from src.tools import fault_attribution as fa                       # noqa: E402

OUT = Path(__file__).resolve().parent / "domain_terms.json"


def _walk_schema(schema, param_names: set, enum_values: set) -> None:
    if not isinstance(schema, dict):
        return
    for k, v in (schema.get("properties") or {}).items():
        param_names.add(k)
        if isinstance(v, dict):
            for e in v.get("enum") or []:
                enum_values.add(str(e))
            items = v.get("items")
            if isinstance(items, dict):
                for e in items.get("enum") or []:
                    enum_values.add(str(e))
            _walk_schema(v, param_names, enum_values)


def build() -> dict:
    groups: dict[str, set] = {k: set() for k in
                              ("tool_name", "param_name", "enum_value", "fault", "symptom", "kg_entity", "gas", "dataset")}
    for name in get_tool_names():
        groups["tool_name"].add(name)
        spec = get_tool_spec(name) or {}
        _walk_schema(spec.get("parameters"), groups["param_name"], groups["enum_value"])

    groups["fault"].update(fa.FAULT_IDS)
    groups["symptom"].update(fa.SYMPTOM_IDS)
    # 故障中文名：从源码中的 FAULT_NAMES 字面量提取（该字典定义在方法内部，无法直接 import）
    src = (ROOT / "src/tools/fault_attribution.py").read_text(encoding="utf-8")
    m = re.search(r"FAULT_NAMES\s*=\s*\{(.*?)\}", src, re.S)
    if m:
        for cn in re.findall(r":\s*\"([^\"]+)\"", m.group(1)):
            for part in re.split(r"[/、]", cn):
                if part.strip():
                    groups["fault"].add(part.strip())

    syn = json.loads((ROOT / "data/kg/synonyms.json").read_text(encoding="utf-8"))
    for _etype, mapping in (syn.get("entities") or {}).items():
        for canon, alias in mapping.items():
            groups["kg_entity"].add(canon)
            groups["kg_entity"].update(a for a in alias if a)
    graph = json.loads((ROOT / "data/kg/graph.json").read_text(encoding="utf-8"))
    for node in graph.get("nodes") or []:
        if node.get("name"):
            groups["kg_entity"].add(node["name"])

    groups["gas"].update(["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "TDCG", "μL/L"])
    groups["dataset"].update(["ETTh1", "ETTh2", "ETTm1", "ETTm2"])

    # 过滤：过短的纯汉字（单字）与纯数字不作为领域标识，避免大面积误命中
    def keep(t: str) -> bool:
        t = t.strip()
        if not t or t.isdigit():
            return False
        if re.fullmatch(r"[\u4e00-\u9fff]", t):
            return False
        return True

    groups_out = {k: sorted(t for t in v if keep(t)) for k, v in groups.items()}
    all_terms = sorted({t for v in groups_out.values() for t in v}, key=lambda s: (-len(s), s))
    return {
        "version": 1,
        "description": "领域关键标识词典：命中的 token 在 M3 加权损失中权重 3；按长度降序做最长匹配",
        "sources": {
            "tool_schema": "src/tools/tool_registry.py",
            "fault_engine": "src/tools/fault_attribution.py",
            "kg_synonyms": "data/kg/synonyms.json",
            "kg_graph": "data/kg/graph.json",
        },
        "counts": {k: len(v) for k, v in groups_out.items()} | {"total": len(all_terms)},
        "groups": groups_out,
        "terms": all_terms,
    }


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {OUT.relative_to(ROOT)}：{data['counts']}")


if __name__ == "__main__":
    main()
