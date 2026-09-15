#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3-3 评分校验集（D9）构建 + 评分器一致性评测。

产物（data/kb/eval/reflection/）：
  1. d9_pairs.jsonl     ——（查询，文本块）对，含 chunk 章节标题与正文，`human_score` 留空供人工 0-3 打分。
                          采样分层：金标块（期望 3）/ 检回非金标（期望 1-3 混合）/ 同文献随机块（期望 0-1）/ 全库随机块（期望 0）。
                          仅使用 retrieval_seed 中 split=dev|test 的查询，避免与训练侧重叠。
  2. report.md          —— 对已标注对计算：各评分器与人工的一致率、加权 Kappa（二次权重）、混淆矩阵；
                          未标注时输出评分器之间的分布对比（lexical vs. 期望桶）。

用法：
  python3 scripts/kb/build_reflection_evalset.py --n 300           # 构建（已存在且含人工分则不覆盖）
  python3 scripts/kb/build_reflection_evalset.py --eval-only        # 只评测
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tools.local_kb import get_local_kb  # noqa: E402
from src.agents.reflection import LexicalScorer  # noqa: E402

SEED_PATH = ROOT / "data/kb/eval/retrieval_seed.jsonl"
OUT_DIR = ROOT / "data/kb/eval/reflection"
PAIRS = OUT_DIR / "d9_pairs.jsonl"

BUCKETS = {"gold": 0.35, "retrieved_nongold": 0.35, "same_doc_random": 0.15, "global_random": 0.15}


def _sec(c: Dict[str, Any]) -> str:
    return " / ".join(c.get("section_path", [])[-2:])


def build_pairs(n: int, seed: int) -> List[Dict[str, Any]]:
    kb = get_local_kb()
    rng = random.Random(seed)
    rows = [json.loads(l) for l in SEED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("split") in ("dev", "test")]
    rng.shuffle(rows)
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in kb.chunks:
        by_doc[c["doc_id"]].append(c)

    quota = {k: int(n * v) for k, v in BUCKETS.items()}
    pairs: List[Dict[str, Any]] = []
    used = set()

    def _emit(q_row: Dict[str, Any], c: Dict[str, Any], bucket: str, expected_hint: str) -> None:
        key = (q_row["qid"], c["chunk_id"])
        if key in used:
            return
        used.add(key)
        pairs.append({
            "pair_id": f"D9-{len(pairs) + 1:04d}",
            "qid": q_row["qid"], "query": q_row["query"], "query_type": q_row.get("type"),
            "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "doc_title": c.get("doc_title", ""),
            "section": _sec(c), "text": c["body"] if c.get("body") else c["text"],
            "bucket": bucket, "expected_hint": expected_hint,
            "is_gold": c["chunk_id"] in q_row["gold_chunk_ids"],
            "human_score": None, "human_score_2": None, "adjudicated": None, "note": "",
        })

    i = 0
    while any(quota[b] > 0 for b in quota) and i < len(rows) * 3:
        r = rows[i % len(rows)]
        i += 1
        gold_ids = r["gold_chunk_ids"]
        gold = kb.get_chunk(gold_ids[0])
        if not gold:
            continue
        if quota["gold"] > 0:
            _emit(r, gold, "gold", "2-3"); quota["gold"] -= 1
        if quota["retrieved_nongold"] > 0:
            hits = [h for h in kb.search(r["query"], top_k=5) if h["chunk_id"] not in gold_ids]
            if hits:
                h = rng.choice(hits)
                _emit(r, kb.get_chunk(h["chunk_id"]), "retrieved_nongold", "0-3"); quota["retrieved_nongold"] -= 1
        if quota["same_doc_random"] > 0:
            cands = [c for c in by_doc[gold["doc_id"]] if c["chunk_id"] not in gold_ids]
            if cands:
                _emit(r, rng.choice(cands), "same_doc_random", "0-2"); quota["same_doc_random"] -= 1
        if quota["global_random"] > 0:
            c = rng.choice(kb.chunks)
            if c["doc_id"] != gold["doc_id"]:
                _emit(r, c, "global_random", "0-1"); quota["global_random"] -= 1
    rng.shuffle(pairs)
    for k, p in enumerate(pairs, 1):
        p["pair_id"] = f"D9-{k:04d}"
    return pairs


def weighted_kappa(a: List[int], b: List[int], k: int = 4) -> Optional[float]:
    """二次加权 Kappa。"""
    n = len(a)
    if n == 0:
        return None
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    ra = [sum(O[i]) for i in range(k)]
    rb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    W = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * ra[i] * rb[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


def evaluate(pairs: List[Dict[str, Any]]) -> str:
    sc = LexicalScorer()
    for p in pairs:
        p["_lexical"] = sc.score(p["query"], p["text"], p["section"])

    L = ["# 反思评分校验报告（P3-3 / D9）\n"]
    L.append(f"- 样本对：{len(pairs)}；桶分布：{dict(Counter(p['bucket'] for p in pairs))}")
    human = [p for p in pairs if isinstance(p.get("adjudicated") if p.get("adjudicated") is not None else p.get("human_score"), int)]
    L.append(f"- 已人工标注：{len(human)}\n")

    L.append("## 评分器输出分布（按采样桶）")
    L.append("| 桶 | n | lexical=0 | 1 | 2 | 3 |")
    L.append("|---|---|---|---|---|---|")
    for b in BUCKETS:
        ps = [p for p in pairs if p["bucket"] == b]
        c = Counter(p["_lexical"] for p in ps)
        L.append(f"| {b} | {len(ps)} | {c.get(0,0)} | {c.get(1,0)} | {c.get(2,0)} | {c.get(3,0)} |")
    L.append("")

    if human:
        def _h(p):
            return p["adjudicated"] if p.get("adjudicated") is not None else p["human_score"]
        hs = [int(_h(p)) for p in human]
        ls = [p["_lexical"] for p in human]
        agree = sum(1 for x, y in zip(hs, ls) if x == y) / len(hs)
        agree1 = sum(1 for x, y in zip(hs, ls) if abs(x - y) <= 1) / len(hs)
        kw = weighted_kappa(hs, ls)
        keep_agree = sum(1 for x, y in zip(hs, ls) if (x >= 2) == (y >= 2)) / len(hs)
        L.append("## lexical 评分器 vs 人工")
        L.append(f"- 完全一致率：{agree:.1%}；±1 一致率：{agree1:.1%}；保留/丢弃（阈值 2）二分一致率：{keep_agree:.1%}")
        L.append(f"- 二次加权 Kappa：{kw:.3f}" if kw is not None else "- 二次加权 Kappa：N/A")
        L.append("")
        L.append("混淆矩阵（行=人工，列=lexical）")
        L.append("| 人工\\lex | 0 | 1 | 2 | 3 |")
        L.append("|---|---|---|---|---|")
        for i in range(4):
            row = Counter(y for x, y in zip(hs, ls) if x == i)
            L.append(f"| {i} | {row.get(0,0)} | {row.get(1,0)} | {row.get(2,0)} | {row.get(3,0)} |")
        L.append("")
        h2 = [p for p in human if isinstance(p.get("human_score_2"), int)]
        if h2:
            a = [int(p["human_score"]) for p in h2]; b = [int(p["human_score_2"]) for p in h2]
            L.append(f"## 标注者间一致性（n={len(h2)}）")
            kk = weighted_kappa(a, b)
            L.append(f"- 完全一致率：{sum(1 for x,y in zip(a,b) if x==y)/len(a):.1%}；二次加权 Kappa：{kk:.3f}" if kk is not None else "- N/A")
            L.append("")
        L.append("LLM 评分器（LLMScorer）一致性需在接口恢复后运行 `--with-llm` 补充。")
    else:
        L.append("## 说明")
        L.append(f"- 尚无人工标注。请在 `{PAIRS.relative_to(ROOT)}` 中填写 `human_score`（0-3，标准见 src/agents/reflection.py SCORE_RUBRIC），"
                 "两人标注时第二人填 `human_score_2`，争议由第三人填 `adjudicated`。")
        L.append("- `expected_hint` 只是采样桶的先验提示，不是答案；标注时请以 0-3 标准独立判断。")
    for p in pairs:
        p.pop("_lexical", None)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if PAIRS.exists():
        pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
        has_human = any(isinstance(p.get("human_score"), int) for p in pairs)
    else:
        pairs, has_human = [], False

    if not args.eval_only and (not pairs or not has_human):
        pairs = build_pairs(args.n, args.seed)
        PAIRS.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        print(f"已写入 {len(pairs)} 对 → {PAIRS}")
    elif not args.eval_only:
        print(f"[skip] {PAIRS} 已含人工标注，不覆盖")

    report = evaluate(pairs)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
