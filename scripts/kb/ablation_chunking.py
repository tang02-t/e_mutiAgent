#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-4 / P1-7 分块策略消融（离线，无需 LLM）。

对比策略：
  title            按最底层章节整体成块（超长按 max 硬切）
  fixed_512_128    固定窗口 512 字 / 重叠 128
  dp_a0.3 / dp_a0.5 / dp_a0.7   动态规划分块，切割惩罚权重 α 网格

流程：对每种策略在临时目录重新分块 → 内存 BM25 索引 → 在 retrieval_seed.jsonl 上评测。
金标映射：评测集记录 `source_unit_id`（查询派生自的原子单元），对每种分块把 source_unit_id
映射到其所在块，作为该策略下的金标块；因此不同分块方案之间的 Recall@k 可直接比较。

输出：data/kb/eval/chunking_ablation.json 与 docs/kb_ablation.md（分块部分）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/kb"))
from build_index import tokenize  # noqa: E402
from src.utils.data_guard import assert_not_synthetic  # noqa: E402

EVAL = ROOT / "data/kb/eval"
KS = (1, 3, 5, 10)

STRATEGIES: Dict[str, List[str]] = {
    "title": ["--strategy", "title"],
    "fixed_512_128": ["--strategy", "fixed", "--window", "512", "--overlap", "128"],
    "dp_a0.3": ["--strategy", "dp", "--alpha", "0.3"],
    "dp_a0.5": ["--strategy", "dp", "--alpha", "0.5"],
    "dp_a0.7": ["--strategy", "dp", "--alpha", "0.7"],
}


def build(strategy_args: List[str], out: Path) -> List[Dict[str, Any]]:
    subprocess.run([sys.executable, str(ROOT / "scripts/kb/build_chunks.py"), "--out", str(out),
                    "--report", str(out.with_suffix(".md")), *strategy_args],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [json.loads(l) for l in open(out, encoding="utf-8")]


def resolve_source_units(recs: List[Dict[str, Any]]) -> None:
    """标题单元不进入块；A_title 查询的金标改为该标题之后的第一个正文单元（章节首块）。"""
    units = [json.loads(l) for l in open(ROOT / "data/kb/units.jsonl", encoding="utf-8")]
    pos = {u["unit_id"]: i for i, u in enumerate(units)}
    for r in recs:
        uid = r["source_unit_id"]
        i = pos.get(uid)
        if i is None or units[i]["type"] != "title":
            r["_gold_unit"] = uid
            continue
        j = i + 1
        while j < len(units) and units[j]["doc_id"] == units[i]["doc_id"] and units[j]["type"] == "title":
            j += 1
        r["_gold_unit"] = units[j]["unit_id"] if j < len(units) and units[j]["doc_id"] == units[i]["doc_id"] else uid


def evaluate(chunks: List[Dict[str, Any]], recs: List[Dict[str, Any]], tag: str) -> Dict[str, Any]:
    unit_to_chunks: Dict[str, List[str]] = {}
    for c in chunks:
        for uid in c["unit_ids"]:
            unit_to_chunks.setdefault(uid.split("#")[0], []).append(c["chunk_id"])
    ids = [c["chunk_id"] for c in chunks]
    t0 = time.perf_counter()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    build_ms = (time.perf_counter() - t0) * 1000

    agg: Dict[str, Dict[str, float]] = {}
    n_by: Dict[str, int] = {}
    lat: List[float] = []
    skipped = 0
    for r in recs:
        gold = set(unit_to_chunks.get(r.get("_gold_unit", r["source_unit_id"]), []))
        if not gold:
            skipped += 1
            continue
        toks = tokenize(r["query"])
        t1 = time.perf_counter()
        scores = bm25.get_scores(toks) if toks else []
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(KS)] if len(scores) else []
        lat.append((time.perf_counter() - t1) * 1000)
        ranked = [ids[i] for i in order if scores[i] > 0]
        rank = next((i + 1 for i, cid in enumerate(ranked) if cid in gold), None)
        for t in (r["type"], "ALL"):
            n_by[t] = n_by.get(t, 0) + 1
            a = agg.setdefault(t, {})
            for k in KS:
                a[f"R@{k}"] = a.get(f"R@{k}", 0.0) + (1.0 if rank and rank <= k else 0.0)
            a["MRR"] = a.get("MRR", 0.0) + (1.0 / rank if rank else 0.0)
    by_type = {t: {k: round(v / n_by[t], 4) for k, v in a.items()} | {"n": n_by[t]} for t, a in agg.items()}
    lens = sorted(c["char_len"] for c in chunks)
    return {
        "tag": tag, "n_chunks": len(chunks), "skipped_no_gold": skipped,
        "len_p50": lens[len(lens) // 2] if lens else 0, "len_p90": lens[int(len(lens) * 0.9)] if lens else 0,
        "index_size_tokens": sum(len(tokenize(c["text"])) for c in chunks),
        "index_build_ms": round(build_ms), "avg_latency_ms": round(sum(lat) / max(1, len(lat)), 2),
        "by_type": by_type,
    }


def render_md(results: List[Dict[str, Any]], split: str) -> str:
    L = ["# 知识库分块策略消融（P1-4 / P1-7 离线部分）", "",
         f"- 评测集：`data/kb/eval/retrieval_seed.jsonl` split={split}；金标按 `source_unit_id` 映射到各策略的所在块",
         "- 检索通道：仅 BM25（jieba + 领域词典），固定 candidate=10；不含锚点通道与稠密向量，以隔离分块本身的影响",
         "- 注意：A_title / B_caption 查询目前为标题/题注原文，BM25 偏乐观；LLM 改写后的自然问句版本待接口预算允许后补跑", "",
         "| 策略 | 块数 | 长度 p50/p90 | 索引 token 数 | R@1 | R@3 | R@5 | R@10 | MRR | 延迟 ms |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        a = r["by_type"]["ALL"]
        L.append(f"| {r['tag']} | {r['n_chunks']} | {r['len_p50']}/{r['len_p90']} | {r['index_size_tokens']} | "
                 f"{a['R@1']:.3f} | {a['R@3']:.3f} | {a['R@5']:.3f} | {a['R@10']:.3f} | {a['MRR']:.3f} | {r['avg_latency_ms']} |")
    L += ["", "## 按查询类型 R@5", "", "| 策略 | A_title | B_caption | C_definition |", "|---|---:|---:|---:|"]
    for r in results:
        bt = r["by_type"]
        L.append(f"| {r['tag']} | " + " | ".join(f"{bt[t]['R@5']:.3f} (n={bt[t]['n']})" if t in bt else "-" for t in ("A_title", "B_caption", "C_definition")) + " |")
    L += ["", "## 结论", ""]
    best = max(results, key=lambda r: r["by_type"]["ALL"]["R@5"])
    base = next((r for r in results if r["tag"] == "title"), None)
    fixed = next((r for r in results if r["tag"].startswith("fixed")), None)
    L.append(f"- R@5 最优策略：`{best['tag']}`（{best['by_type']['ALL']['R@5']:.3f}）。")
    if base:
        L.append(f"- 相对标题切分（R@5={base['by_type']['ALL']['R@5']:.3f}，块数 {base['n_chunks']}）：最优策略 R@5 变化 {best['by_type']['ALL']['R@5'] - base['by_type']['ALL']['R@5']:+.3f}。")
    if fixed:
        L.append(f"- 固定窗口 512/128 产生 {fixed['n_chunks']} 块（索引膨胀），R@5={fixed['by_type']['ALL']['R@5']:.3f}；重叠导致同一单元出现在多块中，金标按任一命中计。")
    dps = [r for r in results if r["tag"].startswith("dp_")]
    if dps:
        L.append("- α 网格：" + "；".join(f"α={r['tag'][4:]} R@5={r['by_type']['ALL']['R@5']:.3f} 块数={r['n_chunks']}" for r in dps)
                 + "。α 越大越保留语义粘连、块更长更少；选择兼顾 R@5 与块数的 α 作为正式配置。")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ap.add_argument("--only", default="", help="逗号分隔的策略名子集")
    a = ap.parse_args()
    seed_path = EVAL / "retrieval_seed.jsonl"
    assert_not_synthetic(str(seed_path), purpose="eval_set")
    recs = [json.loads(l) for l in open(seed_path, encoding="utf-8")]
    if a.split != "all":
        recs = [r for r in recs if r.get("split") == a.split]
    resolve_source_units(recs)
    names = [n for n in a.only.split(",") if n] or list(STRATEGIES)
    results = []
    with tempfile.TemporaryDirectory() as td:
        for name in names:
            chunks = build(STRATEGIES[name], Path(td) / f"chunks_{name}.jsonl")
            res = evaluate(chunks, recs, name)
            results.append(res)
            print(f"{name:14s} chunks={res['n_chunks']:5d} R@1={res['by_type']['ALL']['R@1']:.3f} "
                  f"R@5={res['by_type']['ALL']['R@5']:.3f} MRR={res['by_type']['ALL']['MRR']:.3f} skipped={res['skipped_no_gold']}")
    (EVAL / f"chunking_ablation_{a.split}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_md(results, a.split)
    md += ("\n- 解读注意：A_title 查询是章节标题原文，其金标是章节首块；按标题整体成块时章节文本全部落在一块内，"
           "天然有利于该类查询，因此本表对 `title` 策略存在评测口径偏向。动态规划分块的收益主要体现在块长可控"
           "（供 LLM 上下文预算）、锚点单元与解释段落不被切开（B_caption 类）、以及反思模块邻块补召回的粒度；"
           "最终选型以 LLM 改写后的自然问句评测集与端到端指标为准。\n")
    (ROOT / f"docs/kb_ablation_{a.split}.md").write_text(md, encoding="utf-8")
    print(f"written docs/kb_ablation_{a.split}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
