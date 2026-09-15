#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2-5（离线部分）：图谱关系问答评测集构建 + 精度抽检表 + kg_search 自动评测。

三个产物（均写入 data/kg/eval/）：
  1. kg_qa_seed.jsonl  —— 关系问答种子。对 support_count >= --min-support 的边，用关系模板生成
                          「问题 → 期望实体集合」形式的样例，并按 doc 集合切 train/dev/test，
                          评测 kg_search 的实体定位 + 一跳/多跳检索能否命中期望尾实体。
  2. precision_sample.jsonl —— 分层随机抽样（按关系类型、按 support 分桶）的三元组，附证据句，
                          供人工标注 correct / wrong / unsure；标注后由 eval_kg.py --precision 汇总。
  3. report.md          —— 自动评测结果（自动部分不依赖 LLM）。

注意：
  - 期望答案来自图谱自身的边，因此 kg_qa_seed 只能评测「检索是否忠实于图谱」（工具正确性），
    不能评测「图谱是否正确」——后者靠 precision_sample 人工抽检。
  - 用 doc_id 切分：某条边的 evidence 中任一 doc 落入 test 集合则该样例归 test，防止规则/LLM
    后续在 train 文档上调参时泄漏。

用法：
  python3 scripts/kg/build_kg_eval.py                # 构建 + 自动评测
  python3 scripts/kg/build_kg_eval.py --min-support 2 --n-sample 60
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tools.kg_search import kg_search, get_kg, REL_ZH  # noqa: E402

KG_DIR = ROOT / "data/kg"
OUT_DIR = KG_DIR / "eval"

# 关系 → (问法模板列表, 期望方向)。模板中 {h} 为头实体名。方向 out 表示问头实体的出边。
QA_TEMPLATES: Dict[str, List[str]] = {
    "CAUSES": ["{h}会导致什么故障？", "{h}可能引起哪些问题？", "{h}的后果是什么？"],
    "PRODUCES": ["{h}会产生哪些特征气体？", "{h}时油中哪些气体会升高？"],
    "INDICATES": ["{h}指示什么故障？", "出现{h}说明可能是什么故障？"],
    "LOCATED_IN": ["{h}通常发生在哪个部位？", "{h}的故障部位有哪些？"],
    "DETECTED_BY": ["{h}可以用什么方法检测？", "如何检测{h}？"],
    "MEASURES": ["{h}测量哪些指标？", "{h}关注什么参数？"],
    "TREATED_BY": ["{h}应如何处理？", "{h}的处理措施有哪些？"],
    "SPECIFIED_IN": ["{h}依据哪个标准？", "{h}在哪份规程中有规定？"],
}

# 反向问法：问尾实体的入边（direction=in），用于评测反向检索
QA_TEMPLATES_REV: Dict[str, List[str]] = {
    "CAUSES": ["哪些原因会导致{t}？", "{t}是由什么引起的？"],
    "PRODUCES": ["哪些故障会产生{t}？"],
    "LOCATED_IN": ["{t}部位常见哪些故障？"],
    "DETECTED_BY": ["{t}能检测哪些故障？"],
    "TREATED_BY": ["{t}适用于处理哪些故障？"],
}


def _name(nid: str) -> str:
    return nid.split(":", 1)[1]


def _split_docs(doc_ids: List[str], seed: int = 42) -> Dict[str, str]:
    """按 7/1/2 把 doc_id 分到 train/dev/test。"""
    rng = random.Random(seed)
    docs = sorted(set(doc_ids))
    rng.shuffle(docs)
    n = len(docs)
    n_tr, n_dev = int(n * 0.7), int(n * 0.1)
    out = {}
    for i, d in enumerate(docs):
        out[d] = "train" if i < n_tr else ("dev" if i < n_tr + n_dev else "test")
    return out


def build_qa_seed(graph: Dict[str, Any], min_support: int, seed: int) -> List[Dict[str, Any]]:
    edges = graph["edges"]
    all_docs = [ev["doc_id"] for e in edges for ev in e.get("evidence", [])]
    doc_split = _split_docs(all_docs, seed)

    # 按 (head, relation) 聚合期望尾实体；按 (tail, relation) 聚合期望头实体
    fwd: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {"expected": set(), "docs": set(), "support": 0})
    rev: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {"expected": set(), "docs": set(), "support": 0})
    for e in edges:
        if e["support_count"] < min_support:
            continue
        docs = {ev["doc_id"] for ev in e.get("evidence", [])}
        k = (e["source"], e["relation"])
        fwd[k]["expected"].add(e["target"]); fwd[k]["docs"] |= docs
        fwd[k]["support"] = max(fwd[k]["support"], e["support_count"])
        if e["relation"] in QA_TEMPLATES_REV:
            k2 = (e["target"], e["relation"])
            rev[k2]["expected"].add(e["source"]); rev[k2]["docs"] |= docs
            rev[k2]["support"] = max(rev[k2]["support"], e["support_count"])

    def _split_of(docs: set) -> str:
        splits = {doc_split.get(d, "train") for d in docs}
        if "test" in splits:
            return "test"
        if "dev" in splits:
            return "dev"
        return "train"

    rng = random.Random(seed)
    syn_path = KG_DIR / "synonyms.json"
    aliases: Dict[str, List[str]] = {}
    if syn_path.exists():
        syn = json.loads(syn_path.read_text(encoding="utf-8"))["entities"]
        for typ, m in syn.items():
            for canon, al in m.items():
                aliases[f"{typ}:{canon}"] = [a for a in al if a != canon and len(a) >= 2]  # 单字别名工具有意不匹配

    samples: List[Dict[str, Any]] = []
    idx = 0

    def _emit(anchor: str, rel: str, direction: str, agg: Dict[str, Any], tmpl: str, fmt_key: str) -> None:
        nonlocal idx
        variants = [(_name(anchor), "canonical")]
        al = aliases.get(anchor) or []
        if al:
            variants.append((rng.choice(al), "alias"))
        for surface, kind in variants:
            idx += 1
            samples.append({
                "qa_id": f"KGQA-{idx:04d}", "question": tmpl.format(**{fmt_key: surface}),
                "anchor_entity": anchor, "anchor_surface": surface, "surface_kind": kind,
                "relation": rel, "direction": direction,
                "expected_entities": sorted(agg["expected"]), "max_support": agg["support"],
                "split": _split_of(agg["docs"]), "n_docs": len(agg["docs"]),
            })

    for (h, rel), agg in sorted(fwd.items()):
        _emit(h, rel, "out", agg, rng.choice(QA_TEMPLATES[rel]), "h")
    for (t, rel), agg in sorted(rev.items()):
        _emit(t, rel, "in", agg, rng.choice(QA_TEMPLATES_REV[rel]), "t")
    return samples


def build_precision_sample(graph: Dict[str, Any], n_sample: int, seed: int) -> List[Dict[str, Any]]:
    """分层抽样：每种关系至少抽 min(可用, n_sample//关系数)，support=1 与 >=2 各占一半左右。"""
    rng = random.Random(seed)
    edges = graph["edges"]
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        by_rel[e["relation"]].append(e)
    per_rel = max(3, n_sample // max(len(by_rel), 1))
    chosen: List[Dict[str, Any]] = []
    for rel, es in sorted(by_rel.items()):
        low = [e for e in es if e["support_count"] == 1]
        high = [e for e in es if e["support_count"] >= 2]
        rng.shuffle(low); rng.shuffle(high)
        k_high = min(len(high), per_rel // 2 + per_rel % 2)
        k_low = min(len(low), per_rel - k_high)
        chosen += high[:k_high] + low[:k_low]
    rng.shuffle(chosen)
    out = []
    for i, e in enumerate(chosen, 1):
        ev = e.get("evidence", [])[:2]
        out.append({
            "sample_id": f"KGP-{i:03d}",
            "triple": f"{_name(e['source'])} --{e['relation']}({REL_ZH.get(e['relation'], e['relation'])})--> {_name(e['target'])}",
            "source": e["source"], "relation": e["relation"], "target": e["target"],
            "support_count": e["support_count"], "confidence": e["confidence"],
            "evidence": [x.get("sentence", "") for x in ev],
            "label": "",            # 人工填写：correct / wrong / unsure
            "note": "",
        })
    return out


def auto_eval(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """用 kg_search 跑问答种子：实体定位率、一跳命中率（期望尾实体是否全部/部分出现在一跳路径中）。"""
    res = {"n": 0, "located": 0, "hit_any": 0, "hit_all": 0, "by_rel": defaultdict(lambda: [0, 0, 0]),
           "by_split": defaultdict(lambda: [0, 0, 0]), "by_kind": defaultdict(lambda: [0, 0, 0]), "failures": []}
    for s in samples:
        r = kg_search(s["question"], relations=[s["relation"]], hops=1, direction=s["direction"], max_paths=50)
        res["n"] += 1
        located = r["status"] != "no_match" and any(e["id"] == s["anchor_entity"] for e in r.get("matched_entities", []))
        found = set()
        for p in r.get("paths", []):
            path = p["path"]
            if path[0] == s["anchor_entity"] and len(path) >= 3:
                found.add(path[2])
        exp = set(s["expected_entities"])
        any_hit = bool(found & exp)
        all_hit = exp <= found
        res["located"] += located
        res["hit_any"] += any_hit
        res["hit_all"] += all_hit
        for key, bucket in ((s["relation"], res["by_rel"]), (s["split"], res["by_split"]),
                            (s.get("surface_kind", "canonical"), res["by_kind"])):
            bucket[key][0] += 1; bucket[key][1] += any_hit; bucket[key][2] += all_hit
        if not all_hit and len(res["failures"]) < 30:
            res["failures"].append({
                "qa_id": s["qa_id"], "question": s["question"], "anchor": s["anchor_entity"],
                "located": located,
                "matched": [e["id"] for e in r.get("matched_entities", [])][:5],
                "expected": sorted(exp), "found": sorted(found), "status": r["status"],
            })
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--n-sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    graph = json.loads((KG_DIR / "graph.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qa = build_qa_seed(graph, args.min_support, args.seed)
    with open(OUT_DIR / "kg_qa_seed.jsonl", "w", encoding="utf-8") as f:
        for s in qa:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    prec = build_precision_sample(graph, args.n_sample, args.seed)
    prec_path = OUT_DIR / "precision_sample.jsonl"
    if prec_path.exists():
        # 已有人工标注则不覆盖
        existing = [json.loads(l) for l in prec_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if any(x.get("label") or x.get("ai_prelabel") for x in existing):
            print(f"[skip] {prec_path} 已含标注/预标注，不覆盖（如需重抽请删除该文件）")
            prec = existing
        else:
            prec_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prec), encoding="utf-8")
    else:
        prec_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prec), encoding="utf-8")

    ev = auto_eval(qa)
    n = ev["n"] or 1
    kg = get_kg()

    L = ["# 图谱评测报告（P2-5 离线部分）\n"]
    L.append(f"- 图谱规模：{len(kg.nodes)} 节点 / {len(kg.edges)} 边；min_support={args.min_support}")
    L.append(f"- 关系问答种子：{len(qa)} 条（split 分布：{dict(Counter(s['split'] for s in qa))}；"
             f"方向：{dict(Counter(s['direction'] for s in qa))}）")
    L.append(f"- 精度抽检样本：{len(prec)} 条（已标注 {sum(1 for x in prec if x.get('label'))} 条）\n")
    L.append("## kg_search 自动评测（工具对图谱的忠实度）")
    L.append(f"- 实体定位率：{ev['located']/n:.1%}")
    L.append(f"- 一跳命中率 HitAny：{ev['hit_any']/n:.1%}（至少一个期望实体被检回）")
    L.append(f"- 一跳命中率 HitAll：{ev['hit_all']/n:.1%}（全部期望实体被检回）\n")
    L.append("| 关系 | n | HitAny | HitAll |")
    L.append("|---|---|---|---|")
    for rel, (c, a, b) in sorted(ev["by_rel"].items()):
        L.append(f"| {rel} | {c} | {a/c:.1%} | {b/c:.1%} |")
    L.append("")
    L.append("| split | n | HitAny | HitAll |")
    L.append("|---|---|---|---|")
    for sp, (c, a, b) in sorted(ev["by_split"].items()):
        L.append(f"| {sp} | {c} | {a/c:.1%} | {b/c:.1%} |")
    L.append("")
    L.append("| 问法 | n | HitAny | HitAll |")
    L.append("|---|---|---|---|")
    for k, (c, a, b) in sorted(ev["by_kind"].items()):
        L.append(f"| {k} | {c} | {a/c:.1%} | {b/c:.1%} |")
    L.append("")
    if ev["failures"]:
        L.append("## 未全命中样例（前 30）")
        for f_ in ev["failures"]:
            L.append(f"- {f_['qa_id']} 「{f_['question']}」 located={f_['located']} status={f_['status']} "
                     f"matched={f_['matched']} expected={f_['expected']} found={f_['found']}")
        L.append("")
    labeled = [x for x in prec if x.get("label")]
    L.append("## 三元组精度（人工抽检）")

    def _prec_block(rows: List[Dict[str, Any]], key: str, title: str) -> None:
        c = Counter(x[key] for x in rows)
        correct = c.get("correct", 0)
        L.append(f"- {title}：{len(rows)} 条，correct={correct} wrong={c.get('wrong',0)} unsure={c.get('unsure',0)}")
        L.append(f"- **精度（correct / 已标注）：{correct/len(rows):.1%}**；"
                 f"严格精度（unsure 计为错）同上；宽松精度（unsure 计为对）：{(correct + c.get('unsure',0))/len(rows):.1%}")
        by_rel = defaultdict(lambda: [0, 0, 0])
        for x in rows:
            by_rel[x["relation"]][0] += 1
            by_rel[x["relation"]][1] += x[key] == "correct"
            by_rel[x["relation"]][2] += x[key] == "wrong"
        by_sup = defaultdict(lambda: [0, 0])
        for x in rows:
            b = "support=1" if x["support_count"] == 1 else "support>=2"
            by_sup[b][0] += 1; by_sup[b][1] += x[key] == "correct"
        L.append("")
        L.append("| 关系 | 标注数 | correct | wrong | 精度 |")
        L.append("|---|---|---|---|---|")
        for rel, (t, k, w) in sorted(by_rel.items()):
            L.append(f"| {rel} | {t} | {k} | {w} | {k/t:.1%} |")
        L.append("")
        L.append("| 支持度分桶 | 标注数 | 精度 |")
        L.append("|---|---|---|")
        for b, (t, k) in sorted(by_sup.items()):
            L.append(f"| {b} | {t} | {k/t:.1%} |")
        L.append("")

    if labeled:
        _prec_block(labeled, "label", "人工标注")
    else:
        L.append(f"- 人工尚未标注。请在 `{prec_path.relative_to(ROOT)}` 的 `label` 字段填 correct/wrong/unsure 后重跑本脚本。")
    ai_labeled = [x for x in prec if x.get("ai_prelabel")]
    if ai_labeled:
        edge_keys = {(e["source"], e["relation"], e["target"]) for e in graph["edges"]}
        in_graph = [x for x in ai_labeled if (x["source"], x["relation"], x["target"]) in edge_keys]
        dropped = [x for x in ai_labeled if (x["source"], x["relation"], x["target"]) not in edge_keys]
        L.append("")
        L.append("### AI 预标注（待人工复核，不作为正式结论）")
        _prec_block(ai_labeled, "ai_prelabel", "AI 预标注（抽样时全部样本）")
        if dropped:
            dc = Counter(x["ai_prelabel"] for x in dropped)
            L.append(f"规则修订后已从图谱剔除的抽样边：{len(dropped)} 条（correct={dc.get('correct',0)} "
                     f"wrong={dc.get('wrong',0)} unsure={dc.get('unsure',0)}）")
            L.append("")
            _prec_block(in_graph, "ai_prelabel", "AI 预标注（仅当前图谱仍保留的边）")
        wrongs = [x for x in ai_labeled if x["ai_prelabel"] == "wrong"]
        if wrongs:
            L.append("预标注为 wrong 的样例（提示规则抽取的典型错误模式）：")
            for x in wrongs:
                L.append(f"- {x['sample_id']} {x['triple']}（s={x['support_count']}）：{x.get('ai_note','')}")
            L.append("")
    L.append("")
    L.append("## 说明")
    L.append("- 自动评测的期望答案来自图谱本身，仅衡量 kg_search 的实体定位与检索忠实度，不衡量图谱正确性。")
    L.append("- 图谱正确性由人工抽检精度衡量；LLM 抽取阶段（P2 阶段 2）完成后应重跑并对比 rule-only 与 rule+llm。")
    (OUT_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
