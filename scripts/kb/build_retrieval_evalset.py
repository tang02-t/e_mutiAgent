#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-3 检索评测集（种子版，不依赖 LLM）。

思路：从内容单元自动派生「查询 → 金标块」对，覆盖四类：
  A. 章节标题型：用叶子章节标题作为查询（去掉编号），金标 = 该章节的块（Recall 以任一命中计）
  B. 表/图标题型：用表/图 caption 作为查询，金标 = 含该锚点单元的块
  C. 定义/结论句型：正文中形如「X 是指 / X 是 / 表明 / 说明」的句子，遮盖主语后作查询
  D. 数值事实型：含具体数值 + 领域术语的句子（如「乙炔含量达到 X μL/L」），去掉数值后作查询

每条记录：
  {"qid", "type", "query", "gold_chunk_ids": [...], "gold_doc_id", "source_unit_id", "needs_llm_rewrite": bool}
LLM 改写（把标题/句子改写成自然提问）留待接口恢复后执行，字段 needs_llm_rewrite=True 标记。

同时按 doc_id 做 train/dev/test 切分（7/1/2），保证同一文献不跨集合。

输出：data/kb/eval/retrieval_seed.jsonl、split.json、report.md
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "data/kb"
OUT = KB / "eval"
random.seed(20260915)

DOMAIN = re.compile(r"(变压器|绕组|铁心|铁芯|油|气体|乙炔|氢气|甲烷|乙烯|乙烷|总烃|放电|过热|短路|绝缘|套管|分接开关|瓦斯|温度|电阻|介损|色谱|故障|检修|试验)")
NUM_TERM = re.compile(r"\d+(\.\d+)?\s*(μL/L|uL/L|ppm|℃|°C|kV|kVA|MVA|MΩ|%|Hz|mm|A|V)")
DEF_PAT = re.compile(r"^(.{2,20}?)(是指|指的是|是一种|是变压器|主要是指|通常是指)(.{10,120})$")
CONC_PAT = re.compile(r"^(.{6,60}?)(表明|说明|证明|可以判断|可判断|判断为|诊断为)(.{6,80})$")
STRIP_NUM = re.compile(r"^[\d一二三四五六七八九十（）()．.、\s]+|^第[一二三四五六七八九十\d]+[章节]\s*")


def strip_title(t: str) -> str:
    t = STRIP_NUM.sub("", t).strip()
    t = re.sub(r"[A-Za-z][A-Za-z\s\-]{8,}$", "", t).strip()  # 去掉英文对照标题
    return t


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    units = [json.loads(l) for l in open(KB / "units.jsonl", encoding="utf-8")]
    chunks = [json.loads(l) for l in open(KB / "chunks.jsonl", encoding="utf-8")]
    docs = [json.loads(l) for l in open(KB / "docs.jsonl", encoding="utf-8")]

    unit_to_chunk: Dict[str, str] = {}
    sec_to_chunks: Dict[tuple, List[str]] = collections.defaultdict(list)
    for c in chunks:
        for uid in c["unit_ids"]:
            unit_to_chunk[uid.split("#")[0]] = c["chunk_id"]
        sec_to_chunks[(c["doc_id"], tuple(c["section_path"]))].append(c["chunk_id"])

    recs: List[Dict[str, Any]] = []
    seen_q: set = set()

    def add(qtype: str, query: str, gold: List[str], doc_id: str, uid: str, rewrite: bool, extra: Dict[str, Any] | None = None):
        q = query.strip()
        if len(q) < 4 or q in seen_q or not gold:
            return
        seen_q.add(q)
        recs.append({"qid": f"{qtype}-{len(recs):05d}", "type": qtype, "query": q, "gold_chunk_ids": sorted(set(gold)),
                     "gold_doc_id": doc_id, "source_unit_id": uid, "needs_llm_rewrite": rewrite, **(extra or {})})

    # A. 叶子章节标题
    titles_by_doc: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for u in units:
        if u["type"] == "title" and u["level"] > 0:
            titles_by_doc[u["doc_id"]].append(u)
    for doc_id, ts in titles_by_doc.items():
        for u in ts:
            key = (doc_id, tuple(u["section_path"]))
            golds = sec_to_chunks.get(key)
            q = strip_title(u["text"])
            if not golds or len(q) < 5 or not DOMAIN.search(q):
                continue
            if re.search(r"(参考文献|致谢|结论|结语|引言|前言|摘要|概述|目录)$", q):
                continue
            add("A_title", q, golds, doc_id, u["unit_id"], True)

    # B. 表/图 caption
    for u in units:
        if u["type"] in ("table", "image") and u["caption"] and not u.get("caption_inferred"):
            cap = re.sub(r"^(表|图|Table|Fig\.?)\s*[\d\.\-－—]+\s*", "", u["caption"]).strip()
            cap = re.sub(r"\$.*?\$", "", cap).strip()
            cap = re.sub(r"[A-Za-z][A-Za-z\s\-\.]{8,}$", "", cap).strip()
            if len(cap) < 5 or not DOMAIN.search(cap):
                continue
            cid = unit_to_chunk.get(u["unit_id"])
            if cid:
                add("B_caption", cap, [cid], u["doc_id"], u["unit_id"], True, {"anchor_type": u["type"]})

    # C/D. 句子型
    for u in units:
        if u["type"] != "paragraph" or u["char_len"] < 30:
            continue
        cid = unit_to_chunk.get(u["unit_id"])
        if not cid:
            continue
        for s in re.split(r"(?<=[。；！？])", u["text"]):
            s = s.strip().rstrip("。；！？")
            if not (15 <= len(s) <= 140) or "$" in s or not DOMAIN.search(s):
                continue
            m = DEF_PAT.match(s)
            if m:
                subj = m.group(1).strip()
                if re.search(r"[，,、：:；;（）()]|^(其中|其|该|这|它|此|本文|文中|所谓)", subj) or len(subj) < 2:
                    continue
                add("C_definition", f"什么是{subj}", [cid], u["doc_id"], u["unit_id"], False,
                    {"answer_span": s})
                continue
            m = CONC_PAT.match(s)
            if m and DOMAIN.search(m.group(1)) and not re.search(r"[，,；;]", m.group(1)) and len(m.group(1)) <= 40:
                add("C_conclusion", m.group(1).strip() + "说明什么", [cid], u["doc_id"], u["unit_id"], True,
                    {"answer_span": s})
                continue
            nums = NUM_TERM.findall(s)
            if nums and len(nums) <= 2 and len(s) <= 80 and not re.search(r"(摘\s*要|提要|变电站|供电局|电厂)", s):
                # 只保留与故障判据直接相关的量：气体浓度/温度/电阻/介损
                if not re.search(r"(μL/L|uL/L|ppm|℃|°C|MΩ|%)", s):
                    continue
                if not re.search(r"(乙炔|氢|甲烷|乙烯|乙烷|总烃|一氧化碳|二氧化碳|CO|H2|温度|温升|电阻|介损|含水|微水|吸收比)", s):
                    continue
                q = NUM_TERM.sub("多少", s)
                q = re.sub(r"多少(多少)+", "多少", q)
                add("D_numeric", q, [cid], u["doc_id"], u["unit_id"], True, {"answer_span": s})

    # 采样：控制每类上限与每篇文献上限，避免长论文占满
    # 注：C_conclusion / D_numeric 的规则派生质量不足（抽检多为无效问句），本版不纳入，
    #     这两类改由接口恢复后用 LLM 从 answer_span 生成（见 gen_eval_questions.py）。
    cap_per_type = {"A_title": 700, "B_caption": 400, "C_definition": 300, "C_conclusion": 0, "D_numeric": 0}
    per_doc_cap = 25
    by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in recs:
        by_type[r["type"]].append(r)
    final: List[Dict[str, Any]] = []
    for t, lst in by_type.items():
        if cap_per_type.get(t, 0) <= 0:
            continue
        random.shuffle(lst)
        cnt: Dict[str, int] = collections.Counter()
        for r in lst:
            if t == "C_definition" and re.search(r"(同样|实际上|通常|一般|主要|往往|则|也|就|都)$", r["query"]):
                continue
            if cnt[r["gold_doc_id"]] >= per_doc_cap:
                continue
            cnt[r["gold_doc_id"]] += 1
            final.append(r)
            if sum(1 for x in final if x["type"] == t) >= cap_per_type[t]:
                break

    # 切分（按文献）
    doc_ids = sorted({d["doc_id"] for d in docs})
    random.shuffle(doc_ids)
    n = len(doc_ids)
    split = {"train": doc_ids[: int(n * .7)], "dev": doc_ids[int(n * .7): int(n * .8)], "test": doc_ids[int(n * .8):]}
    doc_split = {d: s for s, ds in split.items() for d in ds}
    for r in final:
        r["split"] = doc_split[r["gold_doc_id"]]

    final.sort(key=lambda r: r["qid"])
    with (OUT / "retrieval_seed.jsonl").open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")

    tc = collections.Counter(r["type"] for r in final)
    sc = collections.Counter(r["split"] for r in final)
    rep = ["# P1-3 检索评测集（种子版）", "",
           f"- 候选总数：{len(recs)}，采样后：{len(final)}",
           f"- 类型分布：{dict(tc)}", f"- 切分（按文献 7/1/2）：{dict(sc)}",
           f"- 需要 LLM 改写为自然问句：{sum(1 for r in final if r['needs_llm_rewrite'])}",
           f"- 覆盖文献数：{len({r['gold_doc_id'] for r in final})} / {len(docs)}", "",
           "## 示例", ""]
    for t in tc:
        for r in [x for x in final if x["type"] == t][:3]:
            rep.append(f"- [{t}] {r['query']}  →  {r['gold_chunk_ids'][0]}")
    rep += ["", "## 说明", "",
            "- 金标为「查询派生自的块」；同章节其他块可能也相关，因此 Recall@k 是保守下界。",
            "- A/B 类查询目前是标题/题注原文，与块文本高度重合，BM25 会偏乐观；LLM 改写后才是有效难度。",
            "- C_definition 类查询已是自然问句，可直接用作基线。"]
    (OUT / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
