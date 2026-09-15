#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-4 动态规划分块：把内容单元（units.jsonl）组合成检索块（chunks.jsonl）。

思想（对应论文的“动态规划分块”）：
- 在同一文献、同一最底层章节内，把连续单元切成若干块；
- 目标：块长接近 target，且尽量不在语义紧密的单元之间切断；
- 用 DP 最小化代价：cost = 长度偏离惩罚 + 切割惩罚（表/图/公式与其相邻解释段落之间切断代价高）。

规则：
- 章节边界必然切断（不同 section_path 不进同一块）。
- 表格 / 公式 / 图片单元视为“锚点单元”，其前后一个段落与它的粘合代价 = 高（不希望被切开）。
- 超过 max_len 的单个单元按句子硬切。
- 每块携带：doc_id、doc_title、section_path、unit_ids、类型集合、文本、char_len。

用法：python3 scripts/kb/build_chunks.py [--target 400] [--max 700] [--min 120]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]

ANCHOR_TYPES = {"table", "equation", "image"}
SENT_SPLIT = re.compile(r"(?<=[。！？；.!?;])")


def split_long(text: str, max_len: int) -> List[str]:
    if len(text) <= max_len:
        return [text]
    sents = [s for s in SENT_SPLIT.split(text) if s]
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > max_len and cur:
            out.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        out.append(cur)
    # 仍超长（无标点）→ 硬切
    final = []
    for seg in out:
        final.extend(seg[i:i + max_len] for i in range(0, len(seg), max_len))
    return final


def glue_cost(prev: Dict[str, Any], nxt: Dict[str, Any]) -> float:
    """在 prev 与 nxt 之间切断的代价（越高越不希望切）。"""
    pt, nt = prev["type"], nxt["type"]
    if pt in ANCHOR_TYPES or nt in ANCHOR_TYPES:
        # 锚点与解释段落之间
        if nt == "paragraph" and re.match(r"^(由表|由图|从表|从图|表\s*\d|图\s*\d|如表|如图|上表|上图|式中|其中)", nxt["text"]):
            return 3.0
        if pt == "paragraph" and re.search(r"(如下表|见表|如表|如下图|见图|如图|列于表|示于图)[^。]*$", prev["text"][-40:]):
            return 3.0
        return 1.5
    if pt == "list" and nt == "list":
        return 2.0
    if pt == "paragraph" and nt == "list":
        return 2.0
    if pt == "paragraph" and prev["text"] and prev["text"][-1] in "：:":
        return 2.5
    return 0.6


def dp_segment(units: List[Dict[str, Any]], target: int, max_len: int, min_len: int, alpha: float = 0.5) -> List[List[int]]:
    """返回每块包含的 units 下标列表。

    alpha ∈ [0,1]：切割惩罚的相对权重。总代价 = 2(1-α)·长度代价 + 2α·切割代价；α=0.5 与旧版等价。
    α 越大越倾向保持语义粘连（块长更不均匀），α 越小越贴近目标长度。
    """
    n = len(units)
    if n == 0:
        return []
    w_len, w_cut = 2.0 * (1.0 - alpha), 2.0 * alpha
    lens = [u["char_len"] for u in units]
    prefix = [0]
    for L in lens:
        prefix.append(prefix[-1] + L + 1)

    INF = float("inf")
    best = [INF] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        # 块 = units[i:j]
        for i in range(j - 1, -1, -1):
            L = prefix[j] - prefix[i] - 1
            if L > max_len and j - i > 1:
                break
            # 长度代价
            if L < min_len:
                len_cost = (min_len - L) / min_len * 2.0
            elif L <= target:
                len_cost = (target - L) / target * 0.5
            else:
                len_cost = (L - target) / target * 1.5
            if L > max_len:
                len_cost += 5.0
            cut_cost = glue_cost(units[i - 1], units[i]) if i > 0 else 0.0
            c = best[i] + w_len * len_cost + w_cut * cut_cost
            if c < best[j]:
                best[j], back[j] = c, i
    segs: List[List[int]] = []
    j = n
    while j > 0:
        i = back[j]
        segs.append(list(range(i, j)))
        j = i
    return segs[::-1]


def fixed_window_segment(units: List[Dict[str, Any]], window: int, overlap: int) -> List[Tuple[str, List[str]]]:
    """固定窗口切分：把章节内正文拼接后按字符窗口滑动。返回 [(text, unit_ids)]。"""
    body = "\n".join(u["text"] for u in units)
    if not body:
        return []
    # 记录每个字符所属 unit，便于回填 unit_ids
    owner: List[str] = []
    for u in units:
        owner.extend([u["unit_id"]] * (len(u["text"]) + 1))
    out: List[Tuple[str, List[str]]] = []
    step = max(1, window - overlap)
    start = 0
    while start < len(body):
        end = min(len(body), start + window)
        seg = body[start:end]
        uids = sorted({owner[k] for k in range(start, end) if k < len(owner)}, key=lambda x: owner.index(x))
        out.append((seg, uids))
        if end >= len(body):
            break
        start += step
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=str(ROOT / "data/kb/units.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/kb/chunks.jsonl"))
    ap.add_argument("--report", default="", help="报告路径；默认与 out 同目录 build_chunks_report.md")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--max", dest="max_len", type=int, default=700)
    ap.add_argument("--min", dest="min_len", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.5, help="DP 切割惩罚权重 α（0.3/0.5/0.7 网格）")
    ap.add_argument("--strategy", default="dp", choices=["dp", "title", "fixed"],
                    help="dp=动态规划；title=按最底层章节整体成块（超长按 max 硬切）；fixed=固定窗口")
    ap.add_argument("--window", type=int, default=512, help="fixed 策略窗口大小")
    ap.add_argument("--overlap", type=int, default=128, help="fixed 策略重叠")
    args = ap.parse_args()

    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    by_doc: Dict[str, List[Dict[str, Any]]] = collections.OrderedDict()
    for u in units:
        by_doc.setdefault(u["doc_id"], []).append(u)

    chunks: List[Dict[str, Any]] = []
    stats = collections.Counter()
    for doc_id, ulist in by_doc.items():
        doc_title = ulist[0]["doc_title"]
        # 按 section_path 分组（保持顺序）；标题单元并入其所属章节开头
        groups: List[Tuple[Tuple[str, ...], List[Dict[str, Any]]]] = []
        for u in ulist:
            if u["type"] == "title":
                continue  # 标题信息已在 section_path 中
            key = tuple(u["section_path"])
            # 超长单元先硬切
            pieces = split_long(u["text"], args.max_len)
            for k, p in enumerate(pieces):
                uu = dict(u)
                uu["text"] = p
                uu["char_len"] = len(p)
                if len(pieces) > 1:
                    uu["unit_id"] = f"{u['unit_id']}#{k}"
                    stats["hard_split_pieces"] += 1
                if groups and groups[-1][0] == key:
                    groups[-1][1].append(uu)
                else:
                    groups.append((key, [uu]))
        for key, us in groups:
            header = " > ".join(key)
            pieces: List[Tuple[str, List[Dict[str, Any]], List[str]]] = []  # (body, members, unit_ids)
            if args.strategy == "dp":
                for seg in dp_segment(us, args.target, args.max_len, args.min_len, alpha=args.alpha):
                    members = [us[i] for i in seg]
                    pieces.append(("\n".join(m["text"] for m in members), members, [m["unit_id"] for m in members]))
            elif args.strategy == "title":
                body_all = "\n".join(m["text"] for m in us)
                for p in split_long(body_all, args.max_len):
                    pieces.append((p, us, [m["unit_id"] for m in us]))
            else:
                for body_seg, uids in fixed_window_segment(us, args.window, args.overlap):
                    members = [m for m in us if m["unit_id"] in set(uids)] or us[:1]
                    pieces.append((body_seg, members, uids))
            for body, members, uids in pieces:
                text = (f"《{doc_title}》 {header}\n{body}" if header else f"《{doc_title}》\n{body}")
                types = sorted({m["type"] for m in members})
                ch = {
                    "chunk_id": f"{doc_id}-c{len([c for c in chunks if c['doc_id'] == doc_id]):04d}",
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "section_path": list(key),
                    "unit_ids": uids,
                    "unit_types": types,
                    "has_anchor": any(t in ANCHOR_TYPES for t in types),
                    "text": text,
                    "body": body,
                    "char_len": len(body),
                    "page_start": members[0]["page"],
                    "page_end": members[-1]["page"],
                }
                chunks.append(ch)
                stats["chunks"] += 1
                stats[f"len_bucket_{min(ch['char_len'] // 100 * 100, 900)}"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lens = sorted(c["char_len"] for c in chunks)
    def pct(p): return lens[int(len(lens) * p)] if lens else 0
    rep = [
        "# P1-4 动态规划分块报告", "",
        f"- 策略：{args.strategy}（dp: alpha={args.alpha}; fixed: window={args.window} overlap={args.overlap}）",
        f"- 参数：target={args.target} max={args.max_len} min={args.min_len}",
        f"- 输入单元：{len(units)}（不含标题 {sum(1 for u in units if u['type'] != 'title')}）",
        f"- 输出块数：{len(chunks)}",
        f"- 块长（正文字符）：p10={pct(.1)} p50={pct(.5)} p90={pct(.9)} max={lens[-1] if lens else 0}",
        f"- 含锚点（表/图/公式）的块：{sum(1 for c in chunks if c['has_anchor'])}",
        f"- 超长单元硬切片段：{stats.get('hard_split_pieces', 0)}",
        f"- 短块（<{args.min_len}）：{sum(1 for L in lens if L < args.min_len)}（多为章节末尾/独立图表说明）",
        "", "## 块长分布", "", "| 区间 | 数量 |", "|---|---:|",
    ]
    for b in range(0, 1000, 100):
        rep.append(f"| {b}-{b+99 if b < 900 else '+'} | {stats.get(f'len_bucket_{b}', 0)} |")
    rep_path = Path(args.report) if args.report else (Path(args.out).parent / "build_chunks_report.md")
    rep_path.write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
