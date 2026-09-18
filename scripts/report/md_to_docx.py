#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Markdown 技术报告转换为 Word(.docx) —— 白底黑字、排版规范版。

关键修复（解决"黑底白字"问题）：
  - 所有文字显式设置为黑色 RGB(0,0,0)，不继承主题色（深色主题 Word 下不再显示为白字）
  - 文档背景显式设为白色
  - 表格使用浅色表头底纹 + 黑色加粗文字，确保对比清晰
  - 代码块浅灰底 + 深色等宽字体

依赖：python-docx
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[2]
# 支持命令行传入 源md 与 输出docx；默认转换技术报告
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "thesis" / "技术报告.md"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "thesis" / "电力变压器故障诊断多智能体系统技术报告.docx"

FONT_CN = "宋体"
FONT_CODE = "Consolas"
FONT_HEAD = "黑体"

BLACK = RGBColor(0x00, 0x00, 0x00)
HEAD_BLUE = RGBColor(0x1F, 0x3A, 0x5F)
CODE_RED = RGBColor(0xB0, 0x2A, 0x1E)

# 表头底纹（浅蓝，与黑字对比清晰）
HEADER_FILL = "DCE6F1"
CODE_FILL = "F4F4F4"


def _set_doc_white_background(doc):
    """显式设置文档背景为白色，避免主题深色背景。"""
    settings = doc.settings.element
    bg = OxmlElement("w:background")
    bg.set(qn("w:color"), "FFFFFF")
    # background 需放在 document 元素上才生效，这里同时写 document body
    doc.element.insert(0, bg)
    # displayBackgroundShape
    dbg = OxmlElement("w:displayBackgroundShape")
    settings.append(dbg)


def set_cell_bg(cell, color_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def style_run(run, font=FONT_CN, size=10.5, bold=False, color=BLACK, italic=False):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), font)
    rfonts.set(qn("w:hAnsi"), font)
    rfonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color  # 始终显式着色，杜绝主题白字


def add_runs_with_inline(paragraph, text, base_font=FONT_CN, size=10.5, base_color=BLACK):
    """解析 **加粗** 、 `代码` 与 <sup>上标</sup> 行内标记。"""
    parts = re.split(r"(\*\*.+?\*\*|`.+?`|<sup>.+?</sup>)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2]); style_run(r, base_font, size, bold=True, color=base_color)
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1]); style_run(r, FONT_CODE, size - 0.5, color=CODE_RED)
        elif part.startswith("<sup>") and part.endswith("</sup>"):
            r = paragraph.add_run(part[5:-6]); style_run(r, base_font, size, color=base_color)
            r.font.superscript = True
        else:
            r = paragraph.add_run(part); style_run(r, base_font, size, color=base_color)


def add_code_block(doc, lines):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Inches(0.15)
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.0
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), CODE_FILL)
    pPr.append(shd)
    for i, line in enumerate(lines):
        if i > 0:
            p.add_run().add_break()
        r = p.add_run(line.replace("\t", "    "))
        style_run(r, FONT_CODE, 9, color=RGBColor(0x33, 0x33, 0x33))


def add_table(doc, rows):
    n_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(n_cols):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            text = row[j] if j < len(row) else ""
            add_runs_with_inline(p, text, size=9.5)
            if i == 0:
                set_cell_bg(cell, HEADER_FILL)
                for run in p.runs:
                    run.font.bold = True
                    run.font.color.rgb = BLACK
            else:
                set_cell_bg(cell, "FFFFFF")  # 数据行显式白底
    doc.add_paragraph()


def parse_table_block(block_lines):
    rows = []
    for ln in block_lines:
        if "---" in ln and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def main():
    text = SRC.read_text(encoding="utf-8")
    lines = text.split("\n")

    doc = Document()
    _set_doc_white_background(doc)

    # 正文默认样式：黑字
    normal = doc.styles["Normal"]
    normal.font.name = FONT_CN
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(6)

    # 强制覆盖内置标题样式的颜色，避免主题色（深色主题下显示白字）
    for sname, scolor in [
        ("Title", BLACK), ("Heading 1", BLACK), ("Heading 2", HEAD_BLUE),
        ("Heading 3", HEAD_BLUE), ("Heading 4", HEAD_BLUE),
        ("List Bullet", BLACK), ("List Number", BLACK),
    ]:
        try:
            st = doc.styles[sname]
            st.font.color.rgb = scolor
            st.font.name = FONT_HEAD if sname.startswith(("Heading", "Title")) else FONT_CN
            st.element.rPr.rFonts.set(qn("w:eastAsia"), st.font.name)
        except KeyError:
            pass

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        if line.strip().startswith("```"):
            code = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            add_code_block(doc, code); i += 1
            continue

        # 图片：![alt](relative/path.png)
        m_img = re.match(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$", line)
        if m_img:
            img_rel = m_img.group(1)
            img_path = (SRC.parent / img_rel).resolve()
            if img_path.exists():
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                try:
                    run.add_picture(str(img_path), width=Inches(6.0))
                except Exception as e:
                    add_runs_with_inline(doc.add_paragraph(), f"[图片插入失败: {img_rel} ({e})]")
            else:
                add_runs_with_inline(doc.add_paragraph(), f"[缺少图片: {img_rel}]")
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < n and "---" in lines[i + 1]:
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, parse_table_block(block))
            continue

        s = line.strip()

        # 块级公式 \[ ... \]（可能跨行）
        if s == r"\[":
            formula = []
            i += 1
            while i < n and lines[i].strip() != r"\]":
                formula.append(lines[i].strip()); i += 1
            i += 1
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(" ".join(formula)); style_run(r, FONT_CODE, 11, italic=True, color=BLACK)
            continue

        if s.startswith("#### "):
            p = doc.add_heading(level=4); p.paragraph_format.space_before = Pt(8)
            r = p.add_run(s[5:]); style_run(r, FONT_HEAD, 11, bold=True, color=HEAD_BLUE)
        elif s.startswith("### "):
            p = doc.add_heading(level=3); p.paragraph_format.space_before = Pt(10)
            r = p.add_run(s[4:]); style_run(r, FONT_HEAD, 12.5, bold=True, color=HEAD_BLUE)
        elif s.startswith("## "):
            p = doc.add_heading(level=2); p.paragraph_format.space_before = Pt(14)
            r = p.add_run(s[3:]); style_run(r, FONT_HEAD, 15, bold=True, color=HEAD_BLUE)
        elif s.startswith("# "):
            p = doc.add_heading(level=1); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(16)
            r = p.add_run(s[2:]); style_run(r, FONT_HEAD, 19, bold=True, color=BLACK)
        elif s == "---":
            pass
        elif re.match(r"^\d+\.\s", s):
            p = doc.add_paragraph(style="List Number")
            add_runs_with_inline(p, re.sub(r"^\d+\.\s", "", s))
        elif s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_runs_with_inline(p, s[2:])
        elif s.startswith("> "):
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Inches(0.3)
            add_runs_with_inline(p, s[2:], base_color=RGBColor(0x44, 0x44, 0x44))
            for r in p.runs:
                r.font.italic = True
        elif s == "":
            pass
        else:
            p = doc.add_paragraph()
            add_runs_with_inline(p, s)

        i += 1

    doc.save(OUT)
    print(f"已生成 Word 文档: {OUT}")


if __name__ == "__main__":
    main()
