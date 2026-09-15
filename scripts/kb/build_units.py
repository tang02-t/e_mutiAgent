#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-1 内容单元清洗：把 MinerU `content_list_v2.json` 转成统一的「内容单元库」。

输入：data/fast_md/<doc_dir>/content_list_v2.json  （结构：页列表 → 块列表）
输出：
  data/kb/units.jsonl      每行一个内容单元
  data/kb/docs.jsonl       每行一篇文献的元数据
  data/kb/build_units_report.md  统计报告

处理规则：
1. 丢弃 page_header / page_footer / page_number / page_aside_text（噪声）；page_footnote 单独保留为 footnote 类型。
2. 标题层级：MinerU 输出全部为 level 1，按编号模式重建（1 / 1.1 / 1.1.1 / 一、/（一）/ 第X章）。
3. 段落跨页合并：上一页末尾段落不以句末标点结尾，且下一页首段落以小写/中文续写 → 合并。
4. 表格：html 转为 Markdown；caption 为空时从相邻段落中匹配「表N ...」推断。
5. 图片：保留路径与 caption；caption 缺失时从相邻段落匹配「图N ...」推断。
6. 行间公式：与前一段落（引出语）及后一段落（以「式中」开头的变量解释）合并成一个 equation 单元。
7. 每个单元携带 section_path（所属章节路径），供后续动态分块与多层索引使用。

用法：python3 scripts/kb/build_units.py [--src data/fast_md] [--out data/kb]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NOISE_TYPES = {"page_header", "page_footer", "page_number", "page_aside_text"}
SENT_END = "。！？；：.!?;:）)」』】"

# ──────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────
def _spans_to_text(spans: Iterable[Dict[str, Any]]) -> str:
    out: List[str] = []
    for sp in spans or []:
        t = sp.get("type")
        c = sp.get("content", "")
        if t == "equation_inline":
            out.append(f"${c}$")
        else:
            out.append(str(c))
    return "".join(out).strip()


def _clean_text(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


class _TableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_clean_text("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(html: str) -> Tuple[str, int, int]:
    p = _TableHTML()
    try:
        p.feed(html or "")
    except Exception:  # noqa: BLE001
        return "", 0, 0
    rows = [r for r in p.rows if any(c for c in r)]
    if not rows:
        return "", 0, 0
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncol]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines), len(rows), ncol


# ──────────────────────────────────────────────────────────────
# 标题层级重建
# ──────────────────────────────────────────────────────────────
_CN_NUM = "一二三四五六七八九十百"
_TITLE_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(rf"^第[{_CN_NUM}\d]+章"), 1),
    (re.compile(rf"^第[{_CN_NUM}\d]+节"), 2),
    (re.compile(rf"^[{_CN_NUM}]+[、．.]"), 1),
    (re.compile(rf"^[（(][{_CN_NUM}]+[)）]"), 2),
    (re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]"), 4),
    (re.compile(r"^\d+[)）]"), 3),          # "3)" 形式：段内小点
    (re.compile(r"^[（(]\d+[)）]"), 3),
    (re.compile(r"^\d+(\.\d+){3}\b"), 4),
    (re.compile(r"^\d+(\.\d+){2}\b"), 3),
    (re.compile(r"^\d+\.\d+\b"), 2),
    (re.compile(r"^\d+[、．.]?\s*\S"), 1),
]
_SPECIAL_L1 = ("摘要", "abstract", "引言", "前言", "结论", "结语", "参考文献", "致谢", "附录", "目录", "结束语")


def infer_title_level(text: str, is_first: bool) -> int:
    t = text.strip()
    if is_first:
        return 0  # 文献主标题
    low = t.lower()
    if any(low.startswith(k) or low.rstrip("：:") == k for k in _SPECIAL_L1):
        return 1
    for pat, lv in _TITLE_PATTERNS:
        if pat.search(t):
            return lv
    return 2  # 无编号的小标题默认次级


# ──────────────────────────────────────────────────────────────
# 单文档处理
# ──────────────────────────────────────────────────────────────
_RE_TABLE_CAP = re.compile(r"^(表\s*\d+[\.\-－—]?\d*|Table\s*\d+)[\s：:．.、]*(.{0,60})")
_RE_FIG_CAP = re.compile(r"^(图\s*\d+[\.\-－—]?\d*|Fig(?:ure)?\.?\s*\d+)[\s：:．.、]*(.{0,60})")


def _flatten(content_list: List[Any]) -> List[Dict[str, Any]]:
    """页列表 → 带 page_idx 的扁平块列表"""
    blocks: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(content_list):
        items = page if isinstance(page, list) else [page]
        for b in items:
            if isinstance(b, dict):
                b = dict(b)
                b["_page"] = page_idx
                blocks.append(b)
    return blocks


def _block_text(b: Dict[str, Any]) -> str:
    t = b.get("type")
    c = b.get("content") or {}
    if t == "title":
        return _clean_text(_spans_to_text(c.get("title_content")))
    if t == "paragraph":
        return _clean_text(_spans_to_text(c.get("paragraph_content")))
    if t == "list":
        items = []
        for it in c.get("list_items") or []:
            items.append("- " + _clean_text(_spans_to_text(it.get("item_content"))))
        return "\n".join(items)
    if t == "page_footnote":
        return _clean_text(_spans_to_text(c.get("page_footnote_content")))
    if t in ("code", "algorithm"):
        key = f"{t}_content"
        raw = c.get(key)
        if isinstance(raw, list):
            return _clean_text(_spans_to_text(raw))
        return _clean_text(str(raw or c.get("code") or ""))
    return ""


def process_doc(doc_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    stats: Dict[str, int] = collections.Counter()
    cl = json.loads((doc_dir / "content_list_v2.json").read_text(encoding="utf-8"))
    blocks = _flatten(cl)
    n_pages = len(cl)
    doc_id = hashlib.md5(doc_dir.name.encode("utf-8")).hexdigest()[:12]

    # ── 1. 去噪 + 转成中间单元 ──
    mid: List[Dict[str, Any]] = []
    first_title_seen = False
    for b in blocks:
        t = b.get("type")
        if t in NOISE_TYPES:
            stats["dropped_noise"] += 1
            continue
        c = b.get("content") or {}
        u: Dict[str, Any] = {"type": t, "page": b["_page"], "bbox": b.get("bbox")}

        if t == "title":
            text = _block_text(b)
            if not text:
                stats["dropped_empty"] += 1
                continue
            u["text"] = text
            if first_title_seen and len(text) > 60:
                # 过长的「标题」几乎都是误判的正文
                u["type"] = "paragraph"
                stats["title_demoted"] += 1
            else:
                u["level"] = infer_title_level(text, not first_title_seen)
                first_title_seen = True
        elif t in ("paragraph", "list", "page_footnote", "code", "algorithm"):
            text = _block_text(b)
            if not text:
                stats["dropped_empty"] += 1
                continue
            u["text"] = text
            if t == "page_footnote":
                u["type"] = "footnote"
            if t == "algorithm":
                u["type"] = "code"
        elif t == "image":
            cap = _clean_text(_spans_to_text(c.get("image_caption")))
            foot = _clean_text(_spans_to_text(c.get("image_footnote")))
            u.update({"image_path": (c.get("image_source") or {}).get("path"),
                      "caption": cap, "footnote": foot, "caption_inferred": False})
        elif t == "table":
            cap = _clean_text(_spans_to_text(c.get("table_caption")))
            foot = _clean_text(_spans_to_text(c.get("table_footnote")))
            md, nrow, ncol = html_table_to_markdown(c.get("html", ""))
            u.update({"image_path": (c.get("image_source") or {}).get("path"),
                      "caption": cap, "footnote": foot, "caption_inferred": False,
                      "html": c.get("html", ""), "markdown": md,
                      "n_rows": nrow, "n_cols": ncol})
        elif t == "equation_interline":
            u["type"] = "equation"
            u["latex"] = c.get("math_content", "")
            u["image_path"] = (c.get("image_source") or {}).get("path")
        else:
            stats[f"dropped_other_{t}"] += 1
            continue
        mid.append(u)

    # ── 2. 跨页/断行段落合并 ──
    merged: List[Dict[str, Any]] = []
    for u in mid:
        if (merged and u["type"] == "paragraph" and merged[-1]["type"] == "paragraph"
                and u["page"] == merged[-1]["page"] + 1):
            prev = merged[-1]["text"]
            if prev and prev[-1] not in SENT_END and not _RE_TABLE_CAP.match(u["text"]) \
                    and not _RE_FIG_CAP.match(u["text"]):
                merged[-1]["text"] = prev + u["text"]
                merged[-1]["merged_pages"] = merged[-1].get("merged_pages", [merged[-1]["page"]]) + [u["page"]]
                stats["merged_cross_page"] += 1
                continue
        merged.append(u)

    # ── 3. 表/图 caption 推断 ──
    def _infer_caption(idx: int, pat: re.Pattern) -> Optional[Tuple[int, str]]:
        for j in (idx - 1, idx + 1, idx - 2, idx + 2):
            if 0 <= j < len(merged) and merged[j]["type"] == "paragraph":
                txt = merged[j]["text"]
                if len(txt) <= 80 and pat.match(txt):
                    return j, txt
        return None

    consumed: set = set()
    for i, u in enumerate(merged):
        if u["type"] == "table" and not u["caption"]:
            r = _infer_caption(i, _RE_TABLE_CAP)
            if r:
                u["caption"], u["caption_inferred"] = r[1], True
                consumed.add(r[0]); stats["table_caption_inferred"] += 1
            else:
                stats["table_caption_missing"] += 1
        elif u["type"] == "image" and not u["caption"]:
            r = _infer_caption(i, _RE_FIG_CAP)
            if r:
                u["caption"], u["caption_inferred"] = r[1], True
                consumed.add(r[0]); stats["image_caption_inferred"] += 1
            else:
                stats["image_caption_missing"] += 1

    # ── 4. 公式与上下文合并 ──
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(merged):
        u = merged[i]
        if i in consumed:
            i += 1
            continue
        if u["type"] == "equation":
            ctx_before = ""
            if out and out[-1]["type"] == "paragraph" and len(out[-1]["text"]) <= 300:
                ctx_before = out.pop()["text"]
            latex_list = [u["latex"]]
            j = i + 1
            while j < len(merged) and merged[j]["type"] == "equation":
                latex_list.append(merged[j]["latex"]); j += 1
            ctx_after = ""
            if j < len(merged) and merged[j]["type"] == "paragraph" and \
                    re.match(r"^(式中|其中|where)", merged[j]["text"]):
                ctx_after = merged[j]["text"]; j += 1
            u = dict(u)
            u["latex"] = latex_list
            u["context_before"], u["context_after"] = ctx_before, ctx_after
            u["text"] = "\n".join(filter(None, [ctx_before, *[f"$${x}$$" for x in latex_list], ctx_after]))
            stats["equation_units"] += 1
            out.append(u)
            i = j
            continue
        out.append(u)
        i += 1

    # ── 5. section_path + 文本渲染 + 编号 ──
    # 文献标题：目录名（原 PDF 文件名）最可靠；学位论文封面的第一个 title 往往是学校名/“学位论文”
    fname_title = re.sub(r"\.pdf-[0-9a-f-]{36}$", "", doc_dir.name)
    fname_title = re.sub(r"\s*\(\d+\)$", "", fname_title).strip()
    parsed_title = next((u["text"] for u in out if u["type"] == "title" and u.get("level") == 0), "")
    doc_title = fname_title or parsed_title or doc_dir.name
    stack: List[Tuple[int, str]] = []
    units: List[Dict[str, Any]] = []
    for order, u in enumerate(out):
        if u["type"] == "title":
            lv = u["level"]
            if lv == 0:
                stack = []
            else:
                while stack and stack[-1][0] >= lv:
                    stack.pop()
                stack.append((lv, u["text"]))
        if u["type"] == "image":
            u["text"] = f"[图] {u['caption'] or '（无标题）'}" + (f"\n{u['footnote']}" if u["footnote"] else "")
        elif u["type"] == "table":
            head = f"[表] {u['caption'] or '（无标题）'}"
            u["text"] = "\n".join(filter(None, [head, u["markdown"], u["footnote"]]))
        u["section_path"] = [s[1] for s in stack]
        u["unit_id"] = f"{doc_id}-{order:04d}"
        u["doc_id"] = doc_id
        u["doc_title"] = doc_title
        u["order"] = order
        u["char_len"] = len(u.get("text", ""))
        units.append(u)
        stats[f"kept_{u['type']}"] += 1

    doc_meta = {
        "doc_id": doc_id,
        "doc_dir": doc_dir.name,
        "title": doc_title,
        "parsed_first_title": parsed_title,
        "n_pages": n_pages,
        "n_units": len(units),
        "doc_kind": "thesis" if n_pages >= 20 else "article",
        "n_chars": sum(u["char_len"] for u in units),
        "n_images": stats.get("kept_image", 0),
        "n_tables": stats.get("kept_table", 0),
        "n_equations": stats.get("kept_equation", 0),
    }
    return doc_meta, units, dict(stats)


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/fast_md"))
    ap.add_argument("--out", default=str(ROOT / "data/kb"))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc_dirs = sorted(d for d in src.iterdir() if d.is_dir() and (d / "content_list_v2.json").exists())

    total: Dict[str, int] = collections.Counter()
    docs: List[Dict[str, Any]] = []
    n_units = 0
    with (out / "units.jsonl").open("w", encoding="utf-8") as fu, \
         (out / "docs.jsonl").open("w", encoding="utf-8") as fd:
        for d in doc_dirs:
            try:
                meta, units, st = process_doc(d)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR] {d.name}: {exc}", file=sys.stderr)
                total["doc_failed"] += 1
                continue
            docs.append(meta)
            fd.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for u in units:
                u.pop("bbox", None)
                fu.write(json.dumps(u, ensure_ascii=False) + "\n")
            n_units += len(units)
            for k, v in st.items():
                total[k] += v

    kinds = collections.Counter(m["doc_kind"] for m in docs)
    lines = [
        "# P1-1 内容单元清洗报告", "",
        f"- 输入文献目录：{len(doc_dirs)}（失败 {total.get('doc_failed', 0)}）",
        f"- 文献类型：期刊/短文 {kinds.get('article', 0)}，学位论文/长文 {kinds.get('thesis', 0)}",
        f"- 输出单元总数：{n_units}", "",
        "## 单元类型分布", "",
        "| 类型 | 数量 |", "|---|---:|",
    ]
    for k in sorted(k for k in total if k.startswith("kept_")):
        lines.append(f"| {k[5:]} | {total[k]} |")
    lines += ["", "## 处理统计", "", "| 项 | 数量 |", "|---|---:|"]
    for k in ("dropped_noise", "dropped_empty", "merged_cross_page", "table_caption_inferred",
              "table_caption_missing", "image_caption_inferred", "image_caption_missing", "equation_units"):
        lines.append(f"| {k} | {total.get(k, 0)} |")
    other = [k for k in total if k.startswith("dropped_other_")]
    for k in other:
        lines.append(f"| {k} | {total[k]} |")
    lines += ["", "## 说明", "",
              "- 标题层级由编号模式重建（level 0 = 文献主标题）。",
              "- 表格 caption 由相邻「表N」段落推断，`caption_inferred=true` 标记。",
              "- 公式单元合并了引出段落与「式中」解释段落，原 LaTeX 保存在 `latex` 列表中。",
              "- 图片当前仅有 caption 与路径，图像描述生成留待 P1-2（需多模态模型）。"]
    (out / "build_units_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
