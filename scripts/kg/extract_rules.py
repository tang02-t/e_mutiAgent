#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2-2 阶段 1：规则三元组抽取（不依赖 LLM）。

流程：
1. 读取 data/kb/units.jsonl 的 paragraph/list 单元，分句。
2. 用同义词表做实体匹配（最长匹配，映射到规范名 + 类型）。
3. 对每个含触发词的句子，按关系模板确定 头/尾 的候选类型与在触发词左右的位置：
   - CAUSES      : 左 Condition|Fault  → 右 Fault|Symptom
   - PRODUCES    : 左 Fault            → 右 Indicator
   - INDICATES   : 左 Symptom|Indicator→ 右 Fault      （"表明/说明/…"）
   - DETECTED_BY : 句中同时含 Method 与 Fault|Symptom 且含"检测/判断/诊断/发现/测量"
   - TREATED_BY  : 左 Fault            → 右 Action      （"应/需/必须/建议 … 动作"）
   - LOCATED_IN  : 「Component 的 Fault」紧邻模式
   - SPECIFIED_IN: 句中同时含 Standard 与 Indicator|Method
4. 输出 data/kg/triples_rule.jsonl；未命中但含触发词的句子写入 data/kg/llm_candidates.jsonl 供阶段 2。
5. 生成 data/kg/extract_report.md。

规则抽取的定位是「高精度小召回」，并给阶段 2 LLM 抽取提供候选句子。
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "data/kb"
KG = ROOT / "data/kg"

TRIGGERS = {
    "CAUSES": r"(导致|引起|引发|造成|使得|会使|致使|从而产生|进而引起)",
    "PRODUCES": r"(产生|析出|生成|分解出|以.{1,6}为主|主要成分是|特征气体为|特征气体是)",
    "INDICATES": r"(表明|说明|反映|意味着|征兆|判断为|诊断为|可判断|可以判断|提示|预示)",
    "TREATED_BY": r"(应|需|须|必须|建议|宜|要|采取|进行)",
    "DETECT_VERB": r"(检测|判断|诊断|发现|测量|测得|识别|监测|判别|确定)",
}
SENT_SPLIT = re.compile(r"(?<=[。；！？])")
ALLOWED = {
    "CAUSES": ({"Condition", "Fault"}, {"Fault", "Symptom"}),
    "PRODUCES": ({"Fault"}, {"Indicator"}),
    "INDICATES": ({"Symptom", "Indicator"}, {"Fault"}),
    "TREATED_BY": ({"Fault"}, {"Action"}),
    "DETECTED_BY": ({"Fault", "Symptom"}, {"Method"}),
    "LOCATED_IN": ({"Fault"}, {"Component"}),
    "SPECIFIED_IN": ({"Indicator", "Method"}, {"Standard"}),
}


def load_lexicon() -> List[Tuple[str, str, str]]:
    """返回 [(surface, canonical, type)]，按 surface 长度降序。"""
    syn = json.loads((KG / "synonyms.json").read_text(encoding="utf-8"))["entities"]
    lex: List[Tuple[str, str, str]] = []
    for etype, m in syn.items():
        for canon, alias in m.items():
            lex.append((canon, canon, etype))
            for a in alias:
                lex.append((a, canon, etype))
    # 同一 surface 多类型时（如 吊罩检查 既是 Method 又是 Action、过负荷 既是 Fault 又是 Condition），保留全部
    lex.sort(key=lambda x: -len(x[0]))
    return lex


def find_entities(sent: str, lex: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    """最长匹配、不重叠；返回 [{surface, canon, type, start, end}]（同一 span 多类型会产生多条）。"""
    found: List[Dict[str, Any]] = []
    occupied = [False] * len(sent)
    for surface, canon, etype in lex:
        if len(surface) < 2 and not surface.isascii():
            continue
        start = 0
        while True:
            i = sent.find(surface, start)
            if i < 0:
                break
            j = i + len(surface)
            # 英文缩写要求边界（仅 ASCII 字母数字视为粘连；汉字 isalnum() 为 True，需排除）
            def _ascii_alnum(ch: str) -> bool:
                return ch.isascii() and ch.isalnum()
            if surface.isascii() and ((i > 0 and _ascii_alnum(sent[i - 1])) or (j < len(sent) and _ascii_alnum(sent[j]))):
                start = j
                continue
            if not any(occupied[i:j]):
                found.append({"surface": surface, "canon": canon, "type": etype, "start": i, "end": j})
                for k in range(i, j):
                    occupied[k] = True
            else:
                # 允许同 span 的其他类型（完全相同 span）
                same = [f for f in found if f["start"] == i and f["end"] == j]
                if same and not any(f["type"] == etype for f in same):
                    found.append({"surface": surface, "canon": canon, "type": etype, "start": i, "end": j})
            start = j
    found.sort(key=lambda f: f["start"])
    return found


_GAS_INDICATORS = {"氢气", "甲烷", "乙炔", "乙烯", "乙烷", "一氧化碳", "二氧化碳", "总烃"}

_HYPERNYM = {
    ("局部放电", "放电故障"), ("低能放电", "放电故障"), ("高能放电", "放电故障"), ("悬浮放电", "放电故障"),
    ("低温过热", "过热故障"), ("中温过热", "过热故障"), ("高温过热", "过热故障"), ("铁心过热", "过热故障"),
    ("匝间短路", "相间短路"), ("绝缘受潮", "绝缘老化"),
}


def _is_hypernym_pair(a: str, b: str) -> bool:
    return (a, b) in _HYPERNYM or (b, a) in _HYPERNYM


_CLAUSE_SEP = "，；,;。：:"


def _clause_bound(sent: str, pos: int, backward: bool, max_clauses: int = 3, max_chars: int = 80) -> int:
    """
    从触发词位置向前/向后扫描，最多跨 max_clauses 个分句、max_chars 个字符，返回边界下标。
    解决「A…，B，说明 C」中把远处的 A 当作头实体的问题。
    """
    if backward:
        i, seen = pos - 1, 0
        while i >= 0 and pos - i <= max_chars:
            if sent[i] in _CLAUSE_SEP:
                seen += 1
                if seen >= max_clauses:
                    return i + 1
            i -= 1
        return max(0, i + 1) if i >= 0 else 0
    i, seen = pos, 0
    while i < len(sent) and i - pos <= max_chars:
        if sent[i] in _CLAUSE_SEP:
            seen += 1
            if seen >= max_clauses:
                return i
        i += 1
    return min(len(sent), i)


# LOCATED_IN 黑名单：这些「故障-部位」组合是表面紧邻造成的伪关系（如「变压器油渗漏」）
_LOCATED_IN_BLOCK = {("渗漏油", "绝缘油"), ("油位异常", "绝缘油")}


def extract_from_sentence(sent: str, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if len(ents) < 2:
        return out
    faults = [e for e in ents if e["type"] == "Fault"]
    is_enum = len(faults) >= 4 or len(ents) >= 7

    # 方向性关系：以触发词位置切分左右，只取触发词两侧「最近」的合法实体，
    # 并跳过枚举句（同类型实体 ≥4 个，如“故障原因可能为 A、B、C、D…”）
    for rel in ("CAUSES", "PRODUCES", "INDICATES"):
        m = re.search(TRIGGERS[rel], sent)
        if not m:
            continue
        ht, tt = ALLOWED[rel]
        left_bound = _clause_bound(sent, m.start(), backward=True)
        right_bound = _clause_bound(sent, m.end(), backward=False)
        heads = [e for e in ents if e["type"] in ht and left_bound <= e["start"] and e["end"] <= m.start()]
        tails = [e for e in ents if e["type"] in tt and m.end() <= e["start"] and e["end"] <= right_bound]
        if not heads or not tails:
            continue
        type_cnt = collections.Counter(e["type"] for e in heads + tails)
        if max(type_cnt.values()) >= 4:
            continue
        h = max(heads, key=lambda e: e["end"])          # 触发词左侧最近
        t = min(tails, key=lambda e: e["start"])        # 触发词右侧最近
        if h["canon"] == t["canon"] or _is_hypernym_pair(h["canon"], t["canon"]):
            continue
        if rel == "PRODUCES" and t["canon"] not in _GAS_INDICATORS:
            continue
        out.append({"head": h["canon"], "head_type": h["type"], "relation": rel,
                    "tail": t["canon"], "tail_type": t["type"]})

    # TREATED_BY：Fault 在前，Action 在后，且中间出现情态词
    actions = [e for e in ents if e["type"] == "Action"]
    if not is_enum:
        for f in faults:
            for a in actions:
                if f["end"] <= a["start"] and a["start"] - f["end"] <= 40 \
                        and re.search(TRIGGERS["TREATED_BY"], sent[f["end"]:a["start"]]):
                    out.append({"head": f["canon"], "head_type": "Fault", "relation": "TREATED_BY",
                                "tail": a["canon"], "tail_type": "Action"})

    # DETECTED_BY：Method 附近（≤15 字）出现检测动词，且与 Fault|Symptom 距离 ≤ 40；
    # 句中方法 ≥3 个（检测手段枚举）时，每个 Fault 只配最近的一个 Method
    if not is_enum:
        methods = [e for e in ents if e["type"] == "Method"]
        method_enum = len(methods) >= 3
        for f in [e for e in ents if e["type"] in ("Fault", "Symptom")]:
            cands = []
            for m_ in methods:
                window = sent[max(0, m_["start"] - 15): m_["end"] + 15]
                if not re.search(TRIGGERS["DETECT_VERB"], window):
                    continue
                dist = max(f["start"], m_["start"]) - min(f["end"], m_["end"])
                if dist > 40:
                    continue
                between = sent[min(f["end"], m_["end"]):max(f["start"], m_["start"])]
                # 「超声波局部放电、油色谱分析、红外测温」这类并列项之间只有顿号，不是检测关系
                if "、" in between and not re.search(TRIGGERS["DETECT_VERB"], between):
                    continue
                cands.append((dist, m_))
            if not cands:
                continue
            cands.sort(key=lambda x: x[0])
            for _, m_ in (cands[:1] if method_enum else cands):
                out.append({"head": f["canon"], "head_type": f["type"], "relation": "DETECTED_BY",
                            "tail": m_["canon"], "tail_type": "Method"})

    # LOCATED_IN：Component 紧邻 Fault（≤3 字符间隔，如 "铁心的多点接地"、"绕组匝间短路"）
    for c in [e for e in ents if e["type"] == "Component"]:
        for f in faults:
            if (f["canon"], c["canon"]) in _LOCATED_IN_BLOCK:
                continue
            gap = sent[c["end"]:f["start"]]
            if 0 <= f["start"] - c["end"] <= 3 and re.fullmatch(r"(的|内部|中|上|处)?", gap):
                out.append({"head": f["canon"], "head_type": "Fault", "relation": "LOCATED_IN",
                            "tail": c["canon"], "tail_type": "Component"})

    # SPECIFIED_IN：
    #  - 句中若出现带编号的标准（如 GB/T 7252），则只认编号匹配的标准，忽略仅靠标准名称匹配到的其他标准
    #    （GB/T 7252 与 DL/T 722 同名《变压器油中溶解气体分析和判断导则》，否则会串）
    #  - Indicator|Method 与标准距离 ≤ 80；规程条目天然是枚举句，故不受 is_enum 限制
    stds = [e for e in ents if e["type"] == "Standard"]
    if stds:
        coded = [s for s in stds if any(ch.isdigit() for ch in s["surface"])]
        use_stds = coded if coded else stds
        for s in use_stds:
            for x in [e for e in ents if e["type"] in ("Indicator", "Method")]:
                dist = max(s["start"], x["start"]) - min(s["end"], x["end"])
                if dist <= 80:
                    out.append({"head": x["canon"], "head_type": x["type"], "relation": "SPECIFIED_IN",
                                "tail": s["canon"], "tail_type": "Standard"})
    # 去重
    uniq = {}
    for t in out:
        uniq[(t["head"], t["relation"], t["tail"])] = t
    return list(uniq.values())


def main() -> int:
    KG.mkdir(parents=True, exist_ok=True)
    lex = load_lexicon()
    units = [json.loads(l) for l in open(KB / "units.jsonl", encoding="utf-8")]
    any_trigger = re.compile("|".join(v for k, v in TRIGGERS.items() if k != "DETECT_VERB") + "|" + TRIGGERS["DETECT_VERB"])

    triples: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    n_sent = n_trig = n_hit = 0
    rel_cnt = collections.Counter()
    ent_cnt = collections.Counter()

    for u in units:
        if u["type"] not in ("paragraph", "list"):
            continue
        for s in SENT_SPLIT.split(u["text"]):
            s = s.strip()
            if not (10 <= len(s) <= 200) or s.count("$") >= 4:
                continue
            n_sent += 1
            if not any_trigger.search(s):
                continue
            n_trig += 1
            ents = find_entities(s, lex)
            for e in ents:
                ent_cnt[(e["type"], e["canon"])] += 1
            tr = extract_from_sentence(s, ents)
            if tr:
                n_hit += 1
                for t in tr:
                    key = (t["head"], t["relation"], t["tail"])
                    rec = triples.get(key)
                    if rec is None:
                        rec = {**t, "evidence": [], "doc_ids": set(), "extractor": "rule", "confidence": 0.6}
                        triples[key] = rec
                    if len(rec["evidence"]) < 3:
                        rec["evidence"].append({"sentence": s, "unit_id": u["unit_id"], "doc_id": u["doc_id"]})
                    rec["doc_ids"].add(u["doc_id"])
                    rel_cnt[t["relation"]] += 1
            elif len(ents) >= 1 and any(e["type"] in ("Fault", "Symptom", "Indicator") for e in ents):
                candidates.append({"sentence": s, "unit_id": u["unit_id"], "doc_id": u["doc_id"],
                                   "entities": [{"canon": e["canon"], "type": e["type"]} for e in ents]})

    with (KG / "triples_rule.jsonl").open("w", encoding="utf-8") as f:
        for rec in sorted(triples.values(), key=lambda r: (-len(r["doc_ids"]), r["relation"], r["head"])):
            rec["support_count"] = len(rec["doc_ids"])
            rec["doc_ids"] = sorted(rec["doc_ids"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (KG / "llm_candidates.jsonl").open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    nodes = {(r["head_type"], r["head"]) for r in triples.values()} | {(r["tail_type"], r["tail"]) for r in triples.values()}
    node_by_type = collections.Counter(t for t, _ in nodes)
    multi = sum(1 for r in triples.values() if r["support_count"] >= 2)
    rep = ["# P2-2 规则三元组抽取报告", "",
           f"- 句子总数：{n_sent}；含触发词：{n_trig}；规则命中：{n_hit}",
           f"- 唯一三元组：{len(triples)}（≥2 篇文献支持：{multi}）",
           f"- 节点：{len(nodes)}  按类型：{dict(node_by_type)}",
           f"- 关系分布（去重前计数）：{dict(rel_cnt)}",
           f"- 待 LLM 抽取候选句：{len(candidates)}", "",
           "## 支持度最高的 25 条", "", "| head | relation | tail | support |", "|---|---|---|---:|"]
    for r in sorted(triples.values(), key=lambda r: -r["support_count"])[:25]:
        rep.append(f"| {r['head']} | {r['relation']} | {r['tail']} | {r['support_count']} |")
    rep += ["", "## 实体命中 Top 30", "", "| type | entity | count |", "|---|---|---:|"]
    for (t, e), c in ent_cnt.most_common(30):
        rep.append(f"| {t} | {e} | {c} |")
    (KG / "extract_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
