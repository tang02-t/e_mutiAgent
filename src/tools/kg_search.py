"""
kg_search 工具：在故障关系链路图谱（data/kg/graph.json）上做实体定位 + 多跳关系检索。

用法（作为工具函数注册）：
    from src.tools.kg_search import kg_search
    kg_search(query="铁心多点接地", relations=["CAUSES", "TREATED_BY"], hops=1)

返回：
{
  "status": "ok" | "no_match",
  "matched_entities": [{"id","name","type","score"}],
  "paths": [ {"path": ["Fault:铁心多点接地", "CAUSES", "Fault:过热故障"], "support_count": 13,
              "confidence": 0.6, "evidence": "…"} ],
  "summary": "铁心多点接地 → 导致 → 过热故障（13 篇文献支持）…",
  "n_nodes": ..., "n_edges": ...
}

实体定位：先用同义词表精确/包含匹配，再用字符 2-gram 相似度兜底。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "data/kg"

REL_ZH = {
    "CAUSES": "导致", "PRODUCES": "产生", "INDICATES": "指示", "LOCATED_IN": "部位",
    "DETECTED_BY": "检测手段", "MEASURES": "测量", "TREATED_BY": "处理措施", "SPECIFIED_IN": "依据标准",
}


class KnowledgeGraph:
    def __init__(self, graph_path: Path = KG / "graph.json", syn_path: Path = KG / "synonyms.json") -> None:
        g = json.loads(graph_path.read_text(encoding="utf-8"))
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in g["nodes"]}
        self.edges: List[Dict[str, Any]] = g["edges"]
        self.out_adj: Dict[str, List[int]] = {}
        self.in_adj: Dict[str, List[int]] = {}
        for i, e in enumerate(self.edges):
            self.out_adj.setdefault(e["source"], []).append(i)
            self.in_adj.setdefault(e["target"], []).append(i)
        # surface → node ids
        self.surface: Dict[str, Set[str]] = {}
        syn = json.loads(syn_path.read_text(encoding="utf-8"))["entities"] if syn_path.exists() else {}
        for typ, m in syn.items():
            for canon, alias in m.items():
                nid = f"{typ}:{canon}"
                if nid in self.nodes:
                    for s in [canon, *alias]:
                        self.surface.setdefault(s.lower(), set()).add(nid)
        for nid, n in self.nodes.items():
            self.surface.setdefault(n["name"].lower(), set()).add(nid)
        self._surfaces_sorted = sorted(self.surface, key=len, reverse=True)

    # ── 实体定位 ──
    @staticmethod
    def _bigrams(s: str) -> Set[str]:
        s = re.sub(r"\s+", "", s.lower())
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    def locate(self, query: str, max_entities: int = 4) -> List[Dict[str, Any]]:
        q = query.lower()
        hits: Dict[str, float] = {}
        # ① 词典包含匹配（长优先，不重叠）
        occupied = [False] * len(q)
        for s in self._surfaces_sorted:
            if len(s) < 2:
                continue
            start = 0
            while True:
                i = q.find(s, start)
                if i < 0:
                    break
                if not any(occupied[i:i + len(s)]):
                    for nid in self.surface[s]:
                        hits[nid] = max(hits.get(nid, 0), 1.0 + len(s) / 20)
                    for k in range(i, i + len(s)):
                        occupied[k] = True
                start = i + len(s)
        # ② 兜底：2-gram 相似
        if not hits:
            qb = self._bigrams(q)
            for nid, n in self.nodes.items():
                nb = self._bigrams(n["name"])
                j = len(qb & nb) / max(1, len(qb | nb))
                if j >= 0.34:
                    hits[nid] = j
        out = [{"id": nid, "name": self.nodes[nid]["name"], "type": self.nodes[nid]["type"], "score": round(s, 3)}
               for nid, s in sorted(hits.items(), key=lambda kv: -kv[1])[:max_entities]]
        return out

    # ── 多跳 ──
    def expand(self, start_ids: List[str], relations: Optional[List[str]] = None, hops: int = 1,
               direction: str = "both", max_paths: int = 30) -> List[Dict[str, Any]]:
        rel_set = set(relations) if relations else None
        paths: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()

        def walk(nid: str, path: List[str], sup: int, conf: float, ev: List[Dict[str, Any]], depth: int,
                 lock: Optional[bool] = None):
            if depth == 0:
                return
            cand: List[Tuple[int, bool]] = []
            if direction in ("out", "both") and lock in (None, True):
                cand += [(i, True) for i in self.out_adj.get(nid, [])]
            if direction in ("in", "both") and lock in (None, False):
                cand += [(i, False) for i in self.in_adj.get(nid, [])]
            for i, fwd in cand:
                e = self.edges[i]
                if rel_set and e["relation"] not in rel_set:
                    continue
                nxt = e["target"] if fwd else e["source"]
                if nxt in path:
                    continue
                step = e["relation"] if fwd else f"←{e['relation']}"
                new_path = path + [step, nxt]
                key = tuple(new_path)
                if key not in seen:
                    seen.add(key)
                    paths.append({
                        "path": new_path,
                        "support_count": min(sup, e["support_count"]) if sup else e["support_count"],
                        "confidence": round(min(conf, e["confidence"]), 2),
                        "evidence": (ev + e.get("evidence", [])[:1])[-1].get("sentence", "") if (ev or e.get("evidence")) else "",
                    })
                # 多跳时锁定方向：因果链只沿同一方向延伸
                walk(nxt, new_path, min(sup, e["support_count"]) if sup else e["support_count"],
                     min(conf, e["confidence"]), ev + e.get("evidence", [])[:1], depth - 1, lock=fwd)

        for s in start_ids:
            walk(s, [s], 0, 1.0, [], hops)
        paths.sort(key=lambda p: (-p["support_count"], -p["confidence"], len(p["path"])))
        return paths[:max_paths]

    @staticmethod
    def render(paths: List[Dict[str, Any]], limit: int = 12) -> str:
        lines = []
        for p in paths[:limit]:
            parts = []
            for i, x in enumerate(p["path"]):
                if i % 2 == 0:
                    parts.append(x.split(":", 1)[1])
                else:
                    rel = x.lstrip("←")
                    parts.append(("←" if x.startswith("←") else "") + REL_ZH.get(rel, rel))
            lines.append(" → ".join(parts) + f"（{p['support_count']} 篇支持，置信 {p['confidence']}）")
        return "\n".join(lines)


_kg: Optional[KnowledgeGraph] = None


def get_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg


def kg_search(query: str, relations: Optional[List[str]] = None, hops: int = 1,
              direction: str = "both", max_paths: int = 20) -> Dict[str, Any]:
    kg = get_kg()
    ents = kg.locate(query)
    if not ents:
        return {"status": "no_match", "message": f"图谱中未找到与「{query}」匹配的实体",
                "matched_entities": [], "paths": [], "n_nodes": len(kg.nodes), "n_edges": len(kg.edges)}
    hops = max(1, min(int(hops or 1), 3))
    paths = kg.expand([e["id"] for e in ents], relations, hops, direction, max_paths)
    return {
        "status": "ok" if paths else "no_relation",
        "matched_entities": ents,
        "paths": paths,
        "summary": kg.render(paths) if paths else "实体已定位，但在指定关系下没有出边/入边。",
        "n_nodes": len(kg.nodes), "n_edges": len(kg.edges),
    }
