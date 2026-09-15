#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索评测：在 retrieval_seed.jsonl 上计算 Recall@k / MRR，支持按通道消融。

用法：
  python3 scripts/kb/eval_retrieval.py                     # 默认 test 切分，全部通道
  python3 scripts/kb/eval_retrieval.py --split dev --no-anchor
  python3 scripts/kb/eval_retrieval.py --ablation          # 依次跑 bm25 / bm25+anchor（dense 需嵌入文件）
输出：data/kb/eval/retrieval_result_<tag>.json 与控制台表格。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tools.local_kb import LocalKBRetriever  # noqa: E402
from src.utils.data_guard import assert_not_synthetic  # noqa: E402

EVAL = ROOT / "data/kb/eval"
KS = (1, 3, 5, 10)


def run(retriever: LocalKBRetriever, recs: List[Dict[str, Any]], tag: str) -> Dict[str, Any]:
    per_type: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    n_type: Dict[str, int] = collections.Counter()
    lat: List[float] = []
    doc_hit = collections.Counter()
    for r in recs:
        t0 = time.perf_counter()
        res = retriever.search(r["query"], top_k=max(KS))
        lat.append((time.perf_counter() - t0) * 1000)
        ids = [x["chunk_id"] for x in res]
        gold = set(r["gold_chunk_ids"])
        n_type[r["type"]] += 1
        n_type["ALL"] += 1
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in gold), None)
        for k in KS:
            hit = 1.0 if rank is not None and rank <= k else 0.0
            per_type[r["type"]][f"R@{k}"] += hit
            per_type["ALL"][f"R@{k}"] += hit
        rr = 1.0 / rank if rank else 0.0
        per_type[r["type"]]["MRR"] += rr
        per_type["ALL"]["MRR"] += rr
        # 文献级命中（更宽松的参考指标）
        if any(x["metadata"]["doc_id"] == r["gold_doc_id"] for x in res[:5]):
            doc_hit[r["type"]] += 1
            doc_hit["ALL"] += 1

    out: Dict[str, Any] = {"tag": tag, "n": n_type["ALL"], "avg_latency_ms": round(sum(lat) / max(1, len(lat)), 2), "by_type": {}}
    for t, agg in per_type.items():
        n = n_type[t]
        out["by_type"][t] = {k: round(v / n, 4) for k, v in agg.items()}
        out["by_type"][t]["Doc@5"] = round(doc_hit[t] / n, 4)
        out["by_type"][t]["n"] = n
    return out


def print_table(results: List[Dict[str, Any]]) -> None:
    types = sorted({t for r in results for t in r["by_type"]}, key=lambda x: (x != "ALL", x))
    print(f"\n{'tag':16s} {'type':14s} {'n':>5s} " + " ".join(f"{f'R@{k}':>6s}" for k in KS) + f" {'MRR':>6s} {'Doc@5':>6s}")
    for r in results:
        for t in types:
            m = r["by_type"].get(t)
            if not m:
                continue
            print(f"{r['tag']:16s} {t:14s} {m['n']:5d} " + " ".join(f"{m[f'R@{k}']:6.3f}" for k in KS) + f" {m['MRR']:6.3f} {m['Doc@5']:6.3f}")
        print(f"{'':16s} avg latency {r['avg_latency_ms']} ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(EVAL / "retrieval_seed.jsonl"))
    ap.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    assert_not_synthetic(args.file, purpose="eval_set")
    recs = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    if args.split != "all":
        recs = [r for r in recs if r.get("split") == args.split]

    configs = []
    if args.ablation:
        configs = [("bm25_only", dict(use_anchor=False, use_dense=False)),
                   ("bm25+anchor", dict(use_anchor=True, use_dense=False))]
    else:
        configs = [(args.tag or ("bm25_only" if args.no_anchor else "bm25+anchor"),
                    dict(use_anchor=not args.no_anchor, use_dense=False))]

    results = []
    for tag, kw in configs:
        retr = LocalKBRetriever(top_k=max(KS), candidate_k=50, **kw)
        res = run(retr, recs, tag)
        res["split"] = args.split
        results.append(res)
        (EVAL / f"retrieval_result_{tag}_{args.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
