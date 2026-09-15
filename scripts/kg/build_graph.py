#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2-3/P2-4 图谱合并与构建：triples_rule.jsonl (+ triples_llm.jsonl 若存在, + triples_human.jsonl 若存在)
→ 约束检查 → 合并 → data/kg/graph.json

约束（见 docs/kg_schema.md §3）：
- 头/尾类型必须满足关系的允许类型；
- CAUSES 尾不能是 Condition，头不能是 Symptom/Indicator；
- 自环拒绝；
- 同一 (head, relation, tail) 合并：support_count 取并集文献数，confidence 取最大，evidence 合并去重（≤5）。

人工确认（triples_human.jsonl）中 `verdict: reject` 的三元组会被移除。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "data/kg"

ALLOWED = {
    "CAUSES": ({"Condition", "Fault"}, {"Fault", "Symptom"}),
    "PRODUCES": ({"Fault"}, {"Indicator"}),
    "INDICATES": ({"Symptom", "Indicator"}, {"Fault"}),
    "TREATED_BY": ({"Fault"}, {"Action"}),
    "DETECTED_BY": ({"Fault", "Symptom"}, {"Method"}),
    "LOCATED_IN": ({"Fault"}, {"Component"}),
    "MEASURES": ({"Method"}, {"Indicator"}),
    "SPECIFIED_IN": ({"Indicator", "Method"}, {"Standard"}),
}
CONF = {"rule": 0.6, "llm": 0.75, "human": 1.0}


def check(t: Dict[str, Any]) -> Tuple[bool, str]:
    rel = t.get("relation")
    if rel not in ALLOWED:
        return False, f"unknown_relation:{rel}"
    ht, tt = ALLOWED[rel]
    if t.get("head_type") not in ht:
        return False, f"head_type:{t.get('head_type')}"
    if t.get("tail_type") not in tt:
        return False, f"tail_type:{t.get('tail_type')}"
    if t["head"] == t["tail"]:
        return False, "self_loop"
    return True, ""


def load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> int:
    sources = [("rule", KG / "triples_rule.jsonl"), ("llm", KG / "triples_llm.jsonl"), ("human", KG / "triples_human.jsonl")]
    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    rejected = collections.Counter()
    human_reject: set = set()

    for tag, path in sources:
        for t in load(path):
            if tag == "human" and t.get("verdict") == "reject":
                human_reject.add((t["head"], t["relation"], t["tail"]))
                continue
            ok, why = check(t)
            if not ok:
                rejected[why] += 1
                continue
            key = (t["head"], t["relation"], t["tail"])
            rec = merged.get(key)
            ev = t.get("evidence") or []
            if isinstance(ev, dict):
                ev = [ev]
            docs = set(t.get("doc_ids") or [e.get("doc_id") for e in ev if e.get("doc_id")])
            if rec is None:
                merged[key] = {
                    "head": t["head"], "head_type": t["head_type"], "relation": t["relation"],
                    "tail": t["tail"], "tail_type": t["tail_type"],
                    "evidence": ev[:5], "doc_ids": docs, "extractors": {tag},
                    "confidence": max(CONF.get(tag, 0.5), float(t.get("confidence", 0))),
                }
            else:
                seen = {e.get("unit_id") for e in rec["evidence"]}
                for e in ev:
                    if e.get("unit_id") not in seen and len(rec["evidence"]) < 5:
                        rec["evidence"].append(e)
                        seen.add(e.get("unit_id"))
                rec["doc_ids"] |= docs
                rec["extractors"].add(tag)
                rec["confidence"] = max(rec["confidence"], CONF.get(tag, 0.5), float(t.get("confidence", 0)))

    for k in human_reject:
        merged.pop(k, None)

    nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    for (h, r, t), rec in merged.items():
        for name, typ in ((h, rec["head_type"]), (t, rec["tail_type"])):
            n = nodes.setdefault((typ, name), {"id": f"{typ}:{name}", "name": name, "type": typ, "degree": 0, "doc_ids": set()})
            n["degree"] += 1
            n["doc_ids"] |= rec["doc_ids"]
        edges.append({
            "source": f"{rec['head_type']}:{h}", "target": f"{rec['tail_type']}:{t}", "relation": r,
            "support_count": len(rec["doc_ids"]), "confidence": round(rec["confidence"], 2),
            "extractors": sorted(rec["extractors"]),
            "evidence": rec["evidence"],
        })
    for n in nodes.values():
        n["n_docs"] = len(n.pop("doc_ids"))

    graph = {
        "schema_version": "0.1",
        "n_nodes": len(nodes), "n_edges": len(edges),
        "nodes": sorted(nodes.values(), key=lambda x: (-x["degree"], x["id"])),
        "edges": sorted(edges, key=lambda e: (-e["support_count"], e["relation"], e["source"])),
        "rejected": dict(rejected), "human_rejected": len(human_reject),
    }
    (KG / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")

    rel_cnt = collections.Counter(e["relation"] for e in edges)
    type_cnt = collections.Counter(n["type"] for n in nodes.values())
    print(f"graph: {len(nodes)} nodes, {len(edges)} edges; rejected={dict(rejected)}; human_rejected={len(human_reject)}")
    print("relations:", dict(rel_cnt))
    print("node types:", dict(type_cnt))
    print("top nodes:", [(n["id"], n["degree"]) for n in graph["nodes"][:12]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
