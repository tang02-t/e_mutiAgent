#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-6 RRF 通道权重网格（在 dev 切分上调参，test 只报告一次）。"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tools.local_kb import LocalKBRetriever  # noqa: E402
from src.utils.data_guard import assert_not_synthetic  # noqa: E402

EVAL = ROOT / "data/kb/eval"


def r_at(retr, recs, k=5):
    hit = 0
    mrr = 0.0
    for r in recs:
        ids = [x["chunk_id"] for x in retr.search(r["query"], top_k=10)]
        gold = set(r["gold_chunk_ids"])
        rank = next((i + 1 for i, c in enumerate(ids) if c in gold), None)
        hit += 1 if rank and rank <= k else 0
        mrr += 1 / rank if rank else 0
    return hit / len(recs), mrr / len(recs)


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    p = EVAL / "retrieval_seed.jsonl"
    assert_not_synthetic(str(p), purpose="eval_set")
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if json.loads(l).get("split") == split]
    retr = LocalKBRetriever(top_k=10, candidate_k=30, use_dense=False, mode="two_way")
    rows = []
    for sub, summ in itertools.product([0.4, 0.8, 1.2], [0.0, 0.3, 0.6]):
        retr.rrf_weights = {"bm25": 1.0, "anchor": 1.2, "sub": sub, "summary": summ}
        r5, mrr = r_at(retr, recs)
        rows.append({"sub": sub, "summary": summ, "R@5": round(r5, 4), "MRR": round(mrr, 4)})
        print(f"sub={sub:.1f} summary={summ:.1f}  R@5={r5:.3f} MRR={mrr:.3f}")
    best = max(rows, key=lambda x: (x["R@5"], x["MRR"]))
    print("best:", best)
    (EVAL / f"rrf_weight_grid_{split}.json").write_text(json.dumps({"rows": rows, "best": best}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
