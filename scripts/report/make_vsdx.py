#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Visio 可编辑的 .vsdx 文件：图3-1 系统总体架构图。

.vsdx 为 OPC(ZIP) 包，内部为符合 Visio Schema 的 XML。
本脚本手工构造最小但合法的包结构，包含：
  - 三个分层容器（编排层 / 智能体层 / 工具层）
  - 9 个可编辑形状（含文字）
  - 连接关系（顺序流、迭代回退、工具调度）
所有形状均为标准矩形 + 文本，可在 Visio 中自由拖动、改字、改色。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "fig3_1_architecture.vsdx"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Visio 页面坐标单位为英寸。页面尺寸约 11 x 8.5。
PAGE_W, PAGE_H = 11.0, 8.5

# ── 形状定义：(id, 文字, 中心x, 中心y, 宽, 高, 填充色, 线色) ──
# 注意 Visio 的 y 轴向上，故顶部 y 大。
SHAPES = [
    # 分层容器（大矩形，浅色，无文字标题单独放）
    (1,  "编排层",  2.6, 7.7, 4.6, 0.45, "F2F6FB", "9DB4CE", 0),   # 标题贴左
    (2,  "智能体层", 2.6, 5.55, 4.6, 0.45, "F4FBF6", "8FC0A2", 0),
    (3,  "工具层",  2.6, 2.0, 4.6, 0.45, "FDF7F0", "D2A679", 0),

    # 编排层节点
    (10, "LangGraph 图式编排\n有向状态图 / 条件路由 / 共享状态 AgentState",
         5.5, 7.0, 5.4, 0.8, "DCE6F1", "1F3A5F", 1),

    # 智能体层 四个节点
    (20, "Planner\n意图分析+任务分解", 1.7, 5.0, 2.0, 0.95, "D9EAD3", "2E7D54", 1),
    (21, "Retriever\n工具调度(并行/串行)", 4.2, 5.0, 2.0, 0.95, "D9EAD3", "2E7D54", 1),
    (22, "Generator\n诊断结论生成", 6.7, 5.0, 2.0, 0.95, "D9EAD3", "2E7D54", 1),
    (23, "Validator\n质量验证+路由", 9.2, 5.0, 2.0, 0.95, "D9EAD3", "2E7D54", 1),

    # 工具层 四个节点
    (30, "RAG\n知识检索", 1.7, 1.4, 2.0, 0.85, "FCE5CD", "B5651D", 1),
    (31, "贝叶斯\n故障归因", 4.2, 1.4, 2.0, 0.85, "FCE5CD", "B5651D", 1),
    (32, "ETT\n油温预测", 6.7, 1.4, 2.0, 0.85, "FCE5CD", "B5651D", 1),
    (33, "时序\n异常检测", 9.2, 1.4, 2.0, 0.85, "FCE5CD", "B5651D", 1),
]

# 连接线：(起点shape, 终点shape, 标签)
CONNECTS = [
    (10, 20, ""),
    (20, 21, ""), (21, 22, ""), (22, 23, ""),
    (23, 22, "REVISION 迭代"),
    (21, 30, ""), (21, 31, ""), (21, 32, ""), (21, 33, ""),
]


def shape_xml(sid, text, cx, cy, w, h, fill, line, is_box):
    # 容器标题（is_box=0）用较小字号靠上；普通框居中
    # Visio Shape XML
    fontsize = "8" if is_box == 0 else "9"
    text_esc = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    # 多行文字用 \n -> 软换行（Visio 用 <Field> 较复杂，简单用文本+换行字符）
    lines = text_esc.split("\\n")
    text_runs = ""
    for idx, ln in enumerate(lines):
        if idx > 0:
            text_runs += "<cp IX='0'/>"  # 占位
        text_runs += ln
    # 这里直接用换行实体
    text_joined = "&#10;".join(lines)

    return f"""    <Shape ID='{sid}' Type='Shape'>
      <Cell N='PinX' V='{cx}'/>
      <Cell N='PinY' V='{cy}'/>
      <Cell N='Width' V='{w}'/>
      <Cell N='Height' V='{h}'/>
      <Cell N='LocPinX' V='{w/2}' F='Width*0.5'/>
      <Cell N='LocPinY' V='{h/2}' F='Height*0.5'/>
      <Cell N='FillForegnd' V='#{fill}'/>
      <Cell N='FillBkgnd' V='#{fill}'/>
      <Cell N='LineColor' V='#{line}'/>
      <Cell N='LineWeight' V='0.013888888888889'/>
      <Cell N='Rounding' V='0.06'/>
      <Cell N='Char.Size' V='{float(fontsize)/72.0:.6f}'/>
      <Cell N='Para.HorzAlign' V='1'/>
      <Cell N='VerticalAlign' V='{1 if is_box else 0}'/>
      <Section N='Geometry' IX='0'>
        <Cell N='NoFill' V='0'/>
        <Row T='RelMoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>
        <Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0'/></Row>
        <Row T='RelLineTo' IX='3'><Cell N='X' V='1'/><Cell N='Y' V='1'/></Row>
        <Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='1'/></Row>
        <Row T='RelLineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>
      </Section>
      <Text>{text_joined}</Text>
    </Shape>"""


def connect_shapes_xml(start_id, conn_base):
    return ""


def build_page_xml():
    shapes = "\n".join(
        shape_xml(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]) for s in SHAPES
    )

    # 连接线作为动态连接器形状
    conn_shapes = []
    conns = []
    cid = 100
    for (a, b, label) in CONNECTS:
        sa = next(s for s in SHAPES if s[0] == a)
        sb = next(s for s in SHAPES if s[0] == b)
        x1, y1 = sa[2], sa[3]
        x2, y2 = sb[2], sb[3]
        label_esc = label.replace("&", "&amp;").replace("<", "&lt;")
        conn_shapes.append(f"""    <Shape ID='{cid}' Type='Shape'>
      <Cell N='PinX' V='{(x1+x2)/2}'/>
      <Cell N='PinY' V='{(y1+y2)/2}'/>
      <Cell N='Width' V='{abs(x2-x1) or 0.1}'/>
      <Cell N='Height' V='{abs(y2-y1) or 0.1}'/>
      <Cell N='BeginX' V='{x1}'/>
      <Cell N='BeginY' V='{y1}'/>
      <Cell N='EndX' V='{x2}'/>
      <Cell N='EndY' V='{y2}'/>
      <Cell N='LineColor' V='#2E7D54'/>
      <Cell N='EndArrow' V='4'/>
      <Section N='Geometry' IX='0'>
        <Cell N='NoFill' V='1'/>
        <Row T='MoveTo' IX='1'><Cell N='X' V='{x1}'/><Cell N='Y' V='{y1}'/></Row>
        <Row T='LineTo' IX='2'><Cell N='X' V='{x2}'/><Cell N='Y' V='{y2}'/></Row>
      </Section>
      <Text>{label_esc}</Text>
    </Shape>""")
        conns.append(f"""    <Connect FromSheet='{cid}' FromCell='BeginX' ToSheet='{a}'/>
    <Connect FromSheet='{cid}' FromCell='EndX' ToSheet='{b}'/>""")
        cid += 1

    conn_xml = "\n".join(conn_shapes)
    connects_xml = "\n".join(conns)

    return f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<PageContents xmlns='http://schemas.microsoft.com/office/visio/2012/main' xml:space='preserve'>
  <Shapes>
{shapes}
{conn_xml}
  </Shapes>
  <Connects>
{connects_xml}
  </Connects>
</PageContents>"""


CONTENT_TYPES = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
  <Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
  <Default Extension='xml' ContentType='application/xml'/>
  <Override PartName='/visio/document.xml' ContentType='application/vnd.ms-visio.drawing.main+xml'/>
  <Override PartName='/visio/pages/pages.xml' ContentType='application/vnd.ms-visio.pages+xml'/>
  <Override PartName='/visio/pages/page1.xml' ContentType='application/vnd.ms-visio.page+xml'/>
  <Override PartName='/visio/windows.xml' ContentType='application/vnd.ms-visio.windows+xml'/>
  <Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>
  <Override PartName='/docProps/app.xml' ContentType='application/vnd.openxmlformats-officedocument.extended-properties+xml'/>
</Types>"""

ROOT_RELS = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/document' Target='visio/document.xml'/>
  <Relationship Id='rId2' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/core-properties' Target='docProps/core.xml'/>
  <Relationship Id='rId3' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties' Target='docProps/app.xml'/>
</Relationships>"""

DOCUMENT_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<VisioDocument xmlns='http://schemas.microsoft.com/office/visio/2012/main' xml:space='preserve'>
  <DocumentSettings TopPage='0'/>
  <Colors/>
  <FaceNames/>
  <StyleSheets/>
</VisioDocument>"""

DOCUMENT_RELS = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/pages' Target='pages/pages.xml'/>
  <Relationship Id='rId2' Type='http://schemas.microsoft.com/visio/2010/relationships/windows' Target='windows.xml'/>
</Relationships>"""

WINDOWS_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Windows xmlns='http://schemas.microsoft.com/office/visio/2012/main' ClientWidth='1000' ClientHeight='700'>
  <Window ID='0' WindowType='Drawing' WindowState='1073741824' Page='0'
          ViewScale='-1' ViewCenterX='5.5' ViewCenterY='4.25'/>
</Windows>"""

PAGES_XML = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Pages xmlns='http://schemas.microsoft.com/office/visio/2012/main' xml:space='preserve'>
  <Page ID='0' Name='系统总体架构图' NameU='系统总体架构图'>
    <PageSheet>
      <Cell N='PageWidth' V='{PAGE_W}'/>
      <Cell N='PageHeight' V='{PAGE_H}'/>
    </PageSheet>
    <Rel xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' r:id='rId1'/>
  </Page>
</Pages>"""

PAGES_RELS = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/page' Target='page1.xml'/>
</Relationships>"""

CORE_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:dcterms='http://purl.org/dc/terms/' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
  <dc:title>系统总体架构图</dc:title>
  <dc:creator>thesis</dc:creator>
</cp:coreProperties>"""

APP_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Properties xmlns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'>
  <Application>Microsoft Visio</Application>
</Properties>"""


def main():
    page1 = build_page_xml()
    files = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": ROOT_RELS,
        "docProps/core.xml": CORE_XML,
        "docProps/app.xml": APP_XML,
        "visio/document.xml": DOCUMENT_XML,
        "visio/_rels/document.xml.rels": DOCUMENT_RELS,
        "visio/windows.xml": WINDOWS_XML,
        "visio/pages/pages.xml": PAGES_XML,
        "visio/pages/_rels/pages.xml.rels": PAGES_RELS,
        "visio/pages/page1.xml": page1,
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    print("已生成 Visio 文件:", OUT)


if __name__ == "__main__":
    main()
