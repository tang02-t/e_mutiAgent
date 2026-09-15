#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-5 多层索引构建（无需 LLM 的部分）。

层级：
  L0 doc      文献级：标题、抽取式摘要（摘要段/首段）、关键词、章节目录
  L1 section  章节级：section_path → 该章节下所有块的拼接摘要（前 N 字）+ 块 id 列表
  L2 chunk    检索块（chunks.jsonl，主检索层）
  L3 unit     原子单元（units.jsonl，用于精确定位表/图/公式）

输出 data/kb/index/
  docs_index.jsonl      L0
  sections_index.jsonl  L1
  bm25_chunks.pkl       L2 BM25 索引（jieba 分词）
  bm25_units.pkl        L3 BM25 索引（仅 table/equation/image caption，用于精确定位）
  vocab_stats.json      词表统计

LLM 生成的摘要（论文中的 summary 层）留待接口恢复后用 scripts/kb/gen_summaries.py 覆盖 `summary` 字段。
"""
from __future__ import annotations

import collections
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "data/kb"
IDX = KB / "index"

# 领域词典：保证关键术语不被切碎
DOMAIN_TERMS = [
    "油中溶解气体", "油色谱", "色谱分析", "三比值法", "改良三比值", "大卫三角形", "Duval三角", "罗杰斯比值",
    "局部放电", "高能放电", "低能放电", "火花放电", "电弧放电", "绕组变形", "匝间短路", "相间短路",
    "绝缘老化", "绝缘受潮", "铁心多点接地", "铁心接地", "分接开关", "有载分接开关", "套管", "冷却器",
    "轻瓦斯", "重瓦斯", "瓦斯保护", "差动保护", "过流保护", "零序保护", "绕组温度", "油温", "顶层油温",
    "绕组直流电阻", "介质损耗", "介损", "绝缘电阻", "吸收比", "极化指数", "局放量", "频响法", "频率响应",
    "短路阻抗", "低电压短路阻抗", "总烃", "乙炔", "氢气", "甲烷", "乙烯", "乙烷", "一氧化碳", "二氧化碳",
    "产气速率", "绝对产气速率", "相对产气速率", "糠醛", "聚合度", "油纸绝缘", "固体绝缘", "热点温度",
    "过热故障", "放电故障", "高温过热", "中温过热", "低温过热", "突发短路", "短路电流", "轴向力", "辐向力",
    "漏磁", "励磁涌流", "谐波", "振动信号", "状态检修", "状态评估", "健康指数", "贝叶斯网络", "支持向量机",
    "神经网络", "故障树", "模糊综合评判", "灰色关联", "云模型", "DGA", "IEC", "GB/T", "DL/T",
]
for t in DOMAIN_TERMS:
    jieba.add_word(t, freq=2000)

_STOP = set("的 了 在 是 和 与 及 或 等 对 中 为 以 由 将 于 而 也 有 其 之 并 被 从 到 把 个 这 那 一 不 上 下 后 前 时 可 能 会 要 用 如 所 该 各 种 进行 通过 根据 由于 因此 以及 可以 一般 主要 应 需 即 但 当 则 且 若 又 再 更 最 很 较 都 已 未 无 非 含 使 让 向 至 给 就 只 才 还 又 又 因 故 此 其中 同时 以上 以下 之间 情况 方面 问题 方法 分析 研究".split())
_TOKEN_OK = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9./%°]+")
_LATEX_CMD = re.compile(r"\\[A-Za-z]+|[\$\{\}\^_\\]")
_PURE_NUM = re.compile(r"^[\d.]+$")


def tokenize(text: str) -> List[str]:
    text = _LATEX_CMD.sub(" ", text)
    toks = []
    for w in jieba.cut(text):
        w = w.strip().lower()
        if not w or w in _STOP or not _TOKEN_OK.fullmatch(w):
            continue
        if len(w) == 1:
            continue                     # 单字（中/英）一律丢弃
        if _PURE_NUM.match(w):
            continue                     # 纯数字不作为检索词
        if w.isascii() and len(w) <= 2 and w not in _ASCII_KEEP:
            continue                     # 丢弃 i/j/c 等单双字母噪声，保留 h2/co/kv 等领域缩写
        toks.append(w)
    return toks


_ASCII_KEEP = {"h2", "co", "kv", "mv", "ma", "kg", "mm", "cm", "ml", "ul", "μl", "hz", "pd", "d1", "d2",
               "t1", "t2", "t3", "nf", "ac", "dc", "iec", "ppm", "hv", "lv", "mw", "kva", "mva"}


def extract_abstract(units: List[Dict[str, Any]]) -> str:
    """抽取式摘要：优先「摘要/提要」段，否则取前两段正文。"""
    for i, u in enumerate(units):
        if u["type"] == "paragraph" and re.match(r"^[\[【（(]?(摘要|提要|内容摘要|Abstract)", u["text"]):
            txt = re.sub(r"^[\[【（(]?(摘要|提要|内容摘要|Abstract)[\]】）)：:\s]*", "", u["text"])
            if len(txt) < 40 and i + 1 < len(units) and units[i + 1]["type"] == "paragraph":
                txt += units[i + 1]["text"]
            return txt[:500]
        if u["type"] == "title" and re.match(r"^(摘要|提要|Abstract)", u["text"]):
            body = [x["text"] for x in units[i + 1:i + 3] if x["type"] == "paragraph"]
            if body:
                return "".join(body)[:500]
    paras = [u["text"] for u in units if u["type"] == "paragraph" and u["char_len"] > 40]
    return "".join(paras[:2])[:500]


def extract_keywords_line(units: List[Dict[str, Any]]) -> List[str]:
    for u in units[:40]:
        if u["type"] == "paragraph":
            m = re.match(r"^[\[【（(]?(关键词|关键字|Key\s*words?)[\]】）)：:\s]*(.+)$", u["text"])
            if m:
                return [k.strip() for k in re.split(r"[;；,，\s]+", m.group(2)) if k.strip()][:10]
    return []


def main() -> int:
    IDX.mkdir(parents=True, exist_ok=True)
    units = [json.loads(l) for l in open(KB / "units.jsonl", encoding="utf-8")]
    chunks = [json.loads(l) for l in open(KB / "chunks.jsonl", encoding="utf-8")]
    docs = [json.loads(l) for l in open(KB / "docs.jsonl", encoding="utf-8")]

    u_by_doc: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for u in units:
        u_by_doc[u["doc_id"]].append(u)
    c_by_doc: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in chunks:
        c_by_doc[c["doc_id"]].append(c)

    # ── L0 docs ──
    with (IDX / "docs_index.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            us = u_by_doc[d["doc_id"]]
            toc = [{"level": u["level"], "title": u["text"]} for u in us if u["type"] == "title" and u["level"] > 0]
            kw = extract_keywords_line(us)
            abstract = extract_abstract(us)
            if not kw:
                cnt = collections.Counter(t for t in tokenize(abstract + " " + " ".join(x["title"] for x in toc))
                                          if len(t) >= 2)
                kw = [w for w, _ in cnt.most_common(8)]
            rec = {**d, "summary": abstract, "summary_source": "extractive",
                   "keywords": kw, "toc": toc[:80], "chunk_ids": [c["chunk_id"] for c in c_by_doc[d["doc_id"]]]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── L1 sections ──
    sec_map: Dict[tuple, Dict[str, Any]] = collections.OrderedDict()
    for c in chunks:
        key = (c["doc_id"], tuple(c["section_path"]))
        s = sec_map.setdefault(key, {"section_id": f"{c['doc_id']}-s{len([k for k in sec_map if k[0] == c['doc_id']]):03d}",
                                      "doc_id": c["doc_id"], "doc_title": c["doc_title"],
                                      "section_path": c["section_path"], "chunk_ids": [], "n_chars": 0,
                                      "preview": ""})
        s["chunk_ids"].append(c["chunk_id"])
        s["n_chars"] += c["char_len"]
        if len(s["preview"]) < 300:
            s["preview"] = (s["preview"] + " " + c["body"])[:300]
    with (IDX / "sections_index.jsonl").open("w", encoding="utf-8") as f:
        for s in sec_map.values():
            s["summary"] = s["preview"]
            s["summary_source"] = "extractive"
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── L2 chunk BM25 ──
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    with (IDX / "bm25_chunks.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["chunk_id"] for c in chunks], "tokens_len": [len(t) for t in corpus_tokens]}, f)

    # ── L3 anchor-unit BM25（表/图/公式）──
    anchors = [u for u in units if u["type"] in ("table", "image", "equation")]
    anchor_tokens = [tokenize(u["text"]) for u in anchors]
    bm25u = BM25Okapi(anchor_tokens)
    with (IDX / "bm25_units.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25u, "ids": [u["unit_id"] for u in anchors]}, f)

    vocab = collections.Counter(t for toks in corpus_tokens for t in toks)
    stats = {
        "n_docs": len(docs), "n_sections": len(sec_map), "n_chunks": len(chunks), "n_anchor_units": len(anchors),
        "vocab_size": len(vocab), "avg_tokens_per_chunk": round(sum(len(t) for t in corpus_tokens) / max(1, len(chunks)), 1),
        "top_terms": vocab.most_common(40),
        "domain_terms_hit": {t: vocab.get(t.lower(), 0) for t in DOMAIN_TERMS if vocab.get(t.lower(), 0)},
    }
    (IDX / "vocab_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("top_terms", "domain_terms_hit")}, ensure_ascii=False))
    print("top terms:", [w for w, _ in stats["top_terms"][:25]])
    print("domain terms hit:", len(stats["domain_terms_hit"]), "/", len(DOMAIN_TERMS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
