#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3-3：在检索评测集 D3 上对比「无反思 / 有反思（lexical）」的检索质量（离线，无需 LLM）。

流程：对每条查询 → local_kb.search(top_k=5) → 构造最小 AgentState → ReflectionModule.run →
比较过滤/补召回后的 retrieved_knowledge 与金标块。

指标：
  Recall@5（原 top5 命中）vs Recall@out（反思输出集合命中）
  Precision（金标块占输出块比例）—— 反思的核心收益应体现在精度而非召回
  金标被误丢率（原本命中、反思后丢弃）
  补召回增益（原本未命中、经邻块补召回后命中）
  平均输出块数、平均耗时

用法：python3 scripts/kb/eval_reflection_recall.py [--split test] [--min-keep 2]
输出：data/kb/eval/reflection/recall_compare_<split>.json + 追加至 docs/reflection_eval.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.reflection import LexicalScorer, ReflectionModule  # noqa: E402
from src.graph.state import AgentState  # noqa: E402
from src.tools.local_kb import get_local_kb  # noqa: E402
from src.utils.data_guard import assert_not_synthetic  # noqa: E402

EVAL = ROOT / "data/kb/eval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--min-keep", type=int, default=None, help="默认取评分器推荐阈值（LexicalScorer=3）")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()

    p = EVAL / "retrieval_seed.jsonl"
    assert_not_synthetic(str(p), purpose="eval_set")
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    if a.split != "all":
        recs = [r for r in recs if r.get("split") == a.split]

    kb = get_local_kb(top_k=a.top_k)
    reflector = ReflectionModule(scorer=LexicalScorer(), kb=kb,
                                 research_fn=lambda q: kb.search(q, top_k=a.top_k),
                                 min_keep_score=a.min_keep, max_rounds=2, neighbor_window=1, max_total_chunks=8)

    agg = {"n": 0, "base_hit": 0, "refl_hit": 0, "base_prec": 0.0, "refl_prec": 0.0,
           "gold_dropped": 0, "gold_gained": 0, "out_sizes": 0, "all_dropped": 0, "rewrite": 0, "lat": 0.0}
    by_type: Dict[str, Dict[str, float]] = {}
    for r in recs:
        gold = set(r["gold_chunk_ids"])
        base = kb.search(r["query"], top_k=a.top_k)
        base_ids = [x["chunk_id"] for x in base]
        st = AgentState(user_query=r["query"])
        st.retrieved_knowledge = [dict(x) for x in base]
        t0 = time.perf_counter()
        reflector.run(st)
        lat = (time.perf_counter() - t0) * 1000
        out_ids = [x.get("chunk_id") for x in st.retrieved_knowledge]
        log = st.reflection_log[-1] if st.reflection_log else {}

        bh = bool(gold & set(base_ids))
        rh = bool(gold & set(out_ids))
        bp = len(gold & set(base_ids)) / max(1, len(base_ids))
        rp = len(gold & set(out_ids)) / max(1, len(out_ids)) if out_ids else 0.0
        t = r["type"]
        for key in (t, "ALL"):
            d = by_type.setdefault(key, {"n": 0, "base_hit": 0, "refl_hit": 0, "base_prec": 0.0, "refl_prec": 0.0,
                                         "gold_dropped": 0, "gold_gained": 0, "out_sizes": 0, "all_dropped": 0, "rewrite": 0, "lat": 0.0})
            d["n"] += 1; d["base_hit"] += bh; d["refl_hit"] += rh; d["base_prec"] += bp; d["refl_prec"] += rp
            d["gold_dropped"] += (bh and not rh); d["gold_gained"] += (rh and not bh)
            d["out_sizes"] += len(out_ids); d["all_dropped"] += (log.get("decision") == "all_dropped")
            d["rewrite"] += bool(log.get("rewrite_triggered")); d["lat"] += lat

    rows = []
    for t, d in sorted(by_type.items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        n = d["n"]
        rows.append({"type": t, "n": n, "R@5_base": round(d["base_hit"] / n, 4), "R_refl": round(d["refl_hit"] / n, 4),
                     "P_base": round(d["base_prec"] / n, 4), "P_refl": round(d["refl_prec"] / n, 4),
                     "gold_dropped": d["gold_dropped"], "gold_gained": d["gold_gained"],
                     "avg_out": round(d["out_sizes"] / n, 2), "all_dropped": d["all_dropped"], "rewrite": d["rewrite"],
                     "avg_ms": round(d["lat"] / n, 2)})
        print(f"{t:14s} n={n:4d} R@5 {rows[-1]['R@5_base']:.3f}->{rows[-1]['R_refl']:.3f}  P {rows[-1]['P_base']:.3f}->{rows[-1]['P_refl']:.3f}  "
              f"dropped={d['gold_dropped']} gained={d['gold_gained']} avg_out={rows[-1]['avg_out']} all_dropped={d['all_dropped']}")

    out_dir = EVAL / "reflection"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"recall_compare_{a.split}.json").write_text(json.dumps({"min_keep": reflector.min_keep_score, "top_k": a.top_k, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    all_row = rows[0]
    md = [f"# 反思模块有/无对比（P3-3，D3 {a.split} 切分，LexicalScorer，min_keep={reflector.min_keep_score}）", "",
          "- 基线：`local_kb.search(top_k=5)`（two_way）；反思：0-3 词法评分、<2 丢弃、全丢改写重检索、同章节邻块补召回、最多 8 块。",
          "- 说明：评分器为离线词法基线，此处结论描述的是反思机制本身在词法评分下的行为；LLMScorer 版本待接口预算允许后复跑。", "",
          "| 类型 | n | R@5 无反思 | R 有反思 | 精度 无反思 | 精度 有反思 | 金标误丢 | 补召回增益 | 平均输出块 | 全丢 | 改写 | 耗时 ms |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['type']} | {r['n']} | {r['R@5_base']:.3f} | {r['R_refl']:.3f} | {r['P_base']:.3f} | {r['P_refl']:.3f} | "
                  f"{r['gold_dropped']} | {r['gold_gained']} | {r['avg_out']} | {r['all_dropped']} | {r['rewrite']} | {r['avg_ms']} |")
    md += ["", "## 解读", "",
           f"- 召回：{all_row['R@5_base']:.3f} → {all_row['R_refl']:.3f}（金标误丢 {all_row['gold_dropped']} 条，邻块补召回新增命中 {all_row['gold_gained']} 条）。",
           f"- 精度：{all_row['P_base']:.3f} → {all_row['P_refl']:.3f}；平均输出块数 {all_row['avg_out']}（基线固定 5）。",
           "- 反思模块的设计目标是在不明显损失召回的前提下提高送入 Generator 的证据精度并补齐上下文；金标误丢条数是评分器阈值 t2 的直接约束，"
           "后续用 D9 人工标注校准 LexicalScorer/LLMScorer 阈值时以该指标为主。"]
    (ROOT / "docs/reflection_eval.md").write_text("\n".join(md), encoding="utf-8")
    print("written docs/reflection_eval.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
