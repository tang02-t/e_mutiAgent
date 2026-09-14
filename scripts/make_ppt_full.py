#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 ts-研究进展汇报-0615.pptx 模板风格，生成完整内容 PPT：
  1) 总体架构设计      2) 工作流与质量闭环
  3) 数据体系说明      4) 验证方法与评测结果
  5) 混合故障归因·原理 6) 混合故障归因·公式
风格：紫色 #7E4381 + 微软雅黑 + 卡片 + 标题栏 + 右上页码。
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "研究进展汇报_整合版.pptx"

PURPLE  = RGBColor(0x7E, 0x43, 0x81)
PURPLE_D= RGBColor(0x5E, 0x30, 0x61)
T1 = RGBColor(0x21, 0x25, 0x29); T2 = RGBColor(0x49, 0x50, 0x57); T3 = RGBColor(0x6C, 0x75, 0x7D)
CARD1 = RGBColor(0xF9, 0xFA, 0xFB); CARD2 = RGBColor(0xF0, 0xF7, 0xFF)
PURPLE_L = RGBColor(0xF3, 0xEC, 0xF4); WHITE = RGBColor(0xFF, 0xFF, 0xFF); LINEC = RGBColor(0xE9, 0xEC, 0xEF)
GREEN = RGBColor(0x2E, 0x7D, 0x4F); ORANGE = RGBColor(0xD9, 0x6A, 0x2B)
F = "微软雅黑"

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def rect(s, x, y, w, h, fill, line=None, round_=False, lw=1.0):
    sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None: sp.fill.background()
    else: sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None: sp.line.fill.background()
    else: sp.line.color.rgb = line; sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    return sp


def oval(s, x, y, d, fill):
    sp = s.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill; sp.line.fill.background(); sp.shadow.inherit = False
    return sp


def txt(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, sa=5, ls=1.08):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(2)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(sa); p.line_spacing = ls
        for (t, sz, c, b, it) in para:
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.color.rgb = c; r.font.bold = b; r.font.name = F; r.font.italic = it
    return tb


def Rn(t, sz=13, c=T1, b=False, it=False): return (t, sz, c, b, it)


def header(s, title, tag):
    rect(s, 0, 0, SW, Inches(0.12), PURPLE)
    txt(s, Inches(0.8), Inches(0.42), Inches(11.0), Inches(0.6), [[Rn(title, 23, PURPLE, True)]], anchor=MSO_ANCHOR.MIDDLE)
    rect(s, Inches(0.8), Inches(1.12), Inches(11.73), Pt(1.5), LINEC)
    txt(s, SW - Inches(1.9), Inches(0.45), Inches(1.3), Inches(0.5), [[Rn(tag, 14, T3)]], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def chip(s, x, y, w, text):
    rect(s, x, y + Inches(0.02), Pt(4), Inches(0.32), PURPLE)
    txt(s, x + Inches(0.16), y, w - Inches(0.2), Inches(0.38), [[Rn(text, 15, T1, True)]], anchor=MSO_ANCHOR.MIDDLE)


def new(title, tag):
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, SW, SH, WHITE); header(s, title, tag); return s


# ════════ 1. 总体架构设计 ════════
s = new("系统架构：四层架构与多智能体协作", "/ 架构设计")
txt(s, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.55), PARAS := [
    [Rn("核心思想：", 14, PURPLE, True), Rn("将变压器故障诊断的专家决策过程，解构为四个职责单一的智能体，以 LangGraph 图式编排实现协作与质量闭环。", 14, T2)]],
    anchor=MSO_ANCHOR.MIDDLE)
# 四层架构（纵向堆叠）
layers = [
    ("用户查询层", "变压器故障描述 / DGA 数据 / 设备上下文", CARD2),
    ("智能体编排层 (LangGraph)", "Planner → Retriever → Generator → Validator（带质量闭环）", PURPLE_L),
    ("工具能力层 (MCP Client)", "rag_search · fault_attribution · ett_forecast · timeseries_anomaly", CARD2),
    ("基础设施层", "Milvus 向量库 · DashScope LLM/Embedding · ETT 数据集 · 知识库", CARD1),
]
ly = Inches(2.05)
for name, desc, fill in layers:
    rect(s, Inches(0.8), ly, Inches(7.3), Inches(0.92), fill, line=PURPLE if "编排" in name else LINEC, round_=True, lw=1.3 if "编排" in name else 1.0)
    txt(s, Inches(1.0), ly + Inches(0.1), Inches(7.0), Inches(0.78), [
        [Rn(name, 14.5, PURPLE_D if "编排" in name else T1, True)], [Rn(desc, 11.5, T2)]], sa=2)
    ly += Inches(1.02)
# 右侧四智能体卡
chip(s, Inches(8.4), Inches(2.05), Inches(4.1), "四个核心智能体")
agents = [("Planner", "策略规划：意图分析、任务分解、工具决策"),
          ("Retriever", "检索编排：并行调用工具，聚合多源证据"),
          ("Generator", "诊断生成：综合证据撰写诊断与建议"),
          ("Validator", "质量验证：结构化评估，驱动迭代闭环")]
ay = Inches(2.55)
for nm, de in agents:
    rect(s, Inches(8.4), ay, Inches(4.13), Inches(0.92), CARD1, round_=True)
    rect(s, Inches(8.4), ay, Pt(5), Inches(0.92), PURPLE)
    txt(s, Inches(8.62), ay + Inches(0.1), Inches(3.85), Inches(0.78), [
        [Rn(nm, 14, PURPLE, True)], [Rn(de, 11, T2)]], sa=2)
    ay += Inches(1.02)

# ════════ 2. 工作流与质量闭环 ════════
s = new("工作流编排：LangGraph 图式协作与质量闭环", "/ 架构设计")
chip(s, Inches(0.8), Inches(1.4), Inches(11.7), "诊断工作流（带条件路由迭代）")
# 流程节点
nodes = ["Planner", "Retriever", "Generator", "Validator"]
nx = Inches(0.9); ny = Inches(2.15); nw = Inches(2.5); gap = Inches(0.42)
for i, nm in enumerate(nodes):
    rect(s, nx, ny, nw, Inches(1.0), PURPLE_L, line=PURPLE, round_=True, lw=1.2)
    txt(s, nx, ny, nw, Inches(1.0), [[Rn(nm, 16, PURPLE_D, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if i < 3:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, nx + nw + Inches(0.04), ny + Inches(0.32), gap - Inches(0.08), Inches(0.36))
        ar.fill.solid(); ar.fill.fore_color.rgb = PURPLE; ar.line.fill.background(); ar.shadow.inherit = False
    nx = nx + nw + gap
# 闭环回环说明
txt(s, Inches(0.9), Inches(3.3), Inches(11.5), Inches(0.5),
    [[Rn("↑ 当 Validator 判定为 REVISION 时，携带改进建议回流至 Generator 重新生成（最多 3 轮）", 13, ORANGE, True)]])
# 条件路由表
chip(s, Inches(0.8), Inches(4.0), Inches(11.7), "Validator 条件路由决策")
tbl = s.shapes.add_table(4, 4, Inches(0.8), Inches(4.5), Inches(11.73), Inches(2.0)).table
tbl.columns[0].width = Inches(2.5); tbl.columns[1].width = Inches(2.0)
tbl.columns[2].width = Inches(2.8); tbl.columns[3].width = Inches(4.43)
rows = [("验证判定", "评分", "路由", "含义"),
        ("PASS", "≥ 7", "结束", "诊断质量合格，直接输出最终诊断"),
        ("REVISION", "4 – 6", "返回 Generator", "携带改进建议重新生成；达上限则接受并提示"),
        ("FAIL", "< 4", "结束", "质量不达标，输出失败提示并建议补充信息")]
for ri, row in enumerate(rows):
    for ci, t in enumerate(row):
        cell = tbl.cell(ri, ci); cell.fill.solid()
        cell.fill.fore_color.rgb = PURPLE if ri == 0 else (CARD1 if ri % 2 else WHITE)
        tf = cell.text_frame; tf.word_wrap = True; tf.margin_top = Pt(3); tf.margin_bottom = Pt(3)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if ci < 3 else PP_ALIGN.LEFT
        r = p.add_run(); r.text = t
        r.font.size = Pt(12.5 if ri == 0 else 12); r.font.bold = (ri == 0 or ci == 0)
        r.font.color.rgb = WHITE if ri == 0 else (PURPLE if ci == 0 else T1); r.font.name = F

# ════════ 3. 数据体系说明 ════════
s = new("数据体系：四类标准化数据集", "/ 数据说明")
txt(s, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.55),
    [[Rn("按统一 schema 构建自洽数据集用于方法验证，可用公开真实数据集（Kaggle FDD、IEEE DataPort、ETDataset）按相同格式替换。", 13.5, T2)]],
    anchor=MSO_ANCHOR.MIDDLE)
cards = [
    ("DGA 油色谱数据", "3000 条", "含真实故障标签，按三比值法编码区间生成，对接故障归因引擎", CARD2),
    ("在线监测时序", "20 台 × 2000 点", "列对齐 ETT，含日/周周期与热惯性，对接油温预测", CARD1),
    ("故障案例库", "200 例", "征兆→试验→定位→处理闭环，用于知识增强", CARD2),
    ("端到端评测集", "120 条", "问题→标准答案，用于系统整体诊断效果评估", CARD1),
]
cw = Inches(5.75); ch = Inches(2.0); cx0 = Inches(0.8); cy0 = Inches(2.15); gx = Inches(0.23); gy = Inches(0.3)
for i, (nm, scale, desc, fill) in enumerate(cards):
    cx = cx0 + (cw + gx) * (i % 2); cy = cy0 + (ch + gy) * (i // 2)
    rect(s, cx, cy, cw, ch, fill, line=LINEC, round_=True)
    rect(s, cx, cy, Pt(5), ch, PURPLE)
    txt(s, cx + Inches(0.25), cy + Inches(0.18), cw - Inches(0.5), Inches(0.5), [[Rn(nm, 16, PURPLE, True)]])
    txt(s, cx + Inches(0.25), cy + Inches(0.7), cw - Inches(0.5), Inches(0.5), [[Rn(scale, 26, PURPLE_D, True)]])
    txt(s, cx + Inches(0.25), cy + Inches(1.35), cw - Inches(0.5), Inches(0.6), [[Rn(desc, 12, T2)]])

# ════════ 4. 验证方法与评测结果 ════════
s = new("系统验证：分层评测体系与结果", "/ 评测")
chip(s, Inches(0.8), Inches(1.4), Inches(5.7), "两层验证体系")
for i, (nm, de) in enumerate([
    ("① 故障归因引擎评测", "确定性、无 LLM 依赖；用标注 DGA 数据算准确率、F1、分级召回"),
    ("② 端到端系统评测", "全流程跑评测集；评估命中率、安全合规率、迭代有效性")]):
    yy = Inches(1.95) + Inches(1.15) * i
    rect(s, Inches(0.8), yy, Inches(5.7), Inches(1.0), CARD1, round_=True)
    rect(s, Inches(0.8), yy, Pt(5), Inches(1.0), PURPLE)
    txt(s, Inches(1.0), yy + Inches(0.12), Inches(5.4), Inches(0.8), [[Rn(nm, 14.5, PURPLE, True)], [Rn(de, 11.5, T2)]], sa=2)
# 关键指标（大数字）
chip(s, Inches(0.8), Inches(4.4), Inches(5.7), "归因引擎关键指标")
stats = [("88.5%", "Top-3 命中率", GREEN), ("100%", "高危故障召回", GREEN), ("37.9%", "Top-1 准确率", ORANGE)]
for i, (v, lb, c) in enumerate(stats):
    sx = Inches(0.8) + Inches(1.95) * i
    rect(s, sx, Inches(4.95), Inches(1.8), Inches(1.5), CARD2, round_=True)
    txt(s, sx, Inches(5.15), Inches(1.8), Inches(0.7), [[Rn(v, 30, c, True)]], align=PP_ALIGN.CENTER)
    txt(s, sx, Inches(5.95), Inches(1.8), Inches(0.45), [[Rn(lb, 11.5, T2)]], align=PP_ALIGN.CENTER)
# 右栏：结果解读
chip(s, Inches(6.85), Inches(1.4), Inches(5.68), "结果解读与改进方向")
rect(s, Inches(6.85), Inches(1.95), Inches(5.68), Inches(4.5), CARD1, round_=True)
txt(s, Inches(7.1), Inches(2.2), Inches(5.2), Inches(4.1), [
    [Rn("● 数据与故障强相关", 13.5, PURPLE, True)],
    [Rn("Top-3 命中率达 88.5%，归因引擎能将真实故障稳定排进前三。", 12.5, T2)],
    [Rn("", 6, T2)],
    [Rn("● 安全底线达标", 13.5, PURPLE, True)],
    [Rn("critical（高危）故障召回率 100%，无高危漏诊。", 12.5, T2)],
    [Rn("", 6, T2)],
    [Rn("● Top-1 偏低的根因", 13.5, ORANGE, True)],
    [Rn("贝叶斯网络 CPT 为人工标定的近似值，似然连乘对高产气故障存在偏倚。", 12.5, T2)],
    [Rn("", 6, T2)],
    [Rn("● 下一步改进", 13.5, GREEN, True)],
    [Rn("用真实标注数据校准 CPT，提升 Top-1 准确率；接入真实在线监测数据。", 12.5, T2)],
], sa=4, ls=1.12)

# ════════ 5. 混合故障归因·原理 ════════
s = new("故障归因方法：三比值法 × 贝叶斯网络混合推理", "/ 核心方法")
rect(s, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.7), PURPLE_L, round_=True)
txt(s, Inches(1.05), Inches(1.35), Inches(11.3), Inches(0.7),
    [[Rn("核心思想：", 14, PURPLE, True), Rn("以贝叶斯概率推理为主、国标三比值规则为辅，融合二者优势——既可量化故障概率，又符合 DL/T 722 规范、可解释。", 14, T2)]],
    anchor=MSO_ANCHOR.MIDDLE)
LX = Inches(0.8); LW = Inches(5.6)
chip(s, LX, Inches(2.3), LW, "为什么要“混合”？")
for i, (nm, de) in enumerate([("三比值法（DL/T 722 国标）", "✓ 可解释、合规　　✗ 仅判定类型、无法量化概率"),
                              ("贝叶斯网络（概率推理）", "✓ 量化故障概率　　✗ 黑箱、可解释性差")]):
    yy = Inches(2.8) + Inches(1.2) * i
    rect(s, LX, yy, LW, Inches(1.1), CARD1, round_=True)
    txt(s, LX + Inches(0.2), yy + Inches(0.12), LW - Inches(0.4), Inches(0.9), [[Rn(nm, 14, PURPLE, True)], [Rn(de, 12.5, T2)]], sa=3)
chip(s, LX, Inches(5.35), LW, "贝叶斯网络结构（两层有向图）")
rect(s, LX, Inches(5.85), Inches(2.5), Inches(1.0), CARD2, line=PURPLE, round_=True, lw=1.2)
txt(s, LX, Inches(5.85), Inches(2.5), Inches(1.0), [[Rn("故障层 F", 14, PURPLE, True)], [Rn("8 类故障", 11.5, T3)], [Rn("短路/局放/过热…", 10.5, T3)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sa=1)
ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, LX + Inches(2.55), Inches(6.15), Inches(0.5), Inches(0.35))
ar.fill.solid(); ar.fill.fore_color.rgb = PURPLE; ar.line.fill.background(); ar.shadow.inherit = False
rect(s, LX + Inches(3.1), Inches(5.85), Inches(2.5), Inches(1.0), CARD2, line=PURPLE, round_=True, lw=1.2)
txt(s, LX + Inches(3.1), Inches(5.85), Inches(2.5), Inches(1.0), [[Rn("征兆层 S", 14, PURPLE, True)], [Rn("14 类征兆", 11.5, T3)], [Rn("气体超标/油温升高…", 10.5, T3)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sa=1)
RX = Inches(6.75); RW = Inches(5.78)
chip(s, RX, Inches(2.3), RW, "四步推理流程")
steps = [("1", "DGA 三比值法编码", "三个气体比值离散编码为 0/1/2，匹配表7故障判据"),
         ("2", "朴素贝叶斯后验推理", "由先验 P(F) 与条件概率表 CPT 计算后验并归一化"),
         ("3", "规则—概率融合", "0.7×贝叶斯 + 0.3×规则置信，对命中故障加权修正"),
         ("4", "排序 · 分级 · 报告", "按融合概率排序，映射高/中/低风险，输出报告")]
sy = Inches(2.82)
for num, head, desc in steps:
    rect(s, RX, sy, RW, Inches(0.98), CARD1, round_=True)
    oval(s, RX + Inches(0.16), sy + Inches(0.23), Inches(0.52), PURPLE)
    txt(s, RX + Inches(0.16), sy + Inches(0.23), Inches(0.52), Inches(0.52), [[Rn(num, 18, WHITE, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, RX + Inches(0.85), sy + Inches(0.1), RW - Inches(1.0), Inches(0.82), [[Rn(head, 14.5, T1, True)], [Rn(desc, 11.5, T2)]], sa=2)
    sy += Inches(1.06)

# ════════ 6. 混合故障归因·公式 ════════
s = new("故障归因方法：核心公式与诊断算例", "/ 核心方法")
LX = Inches(0.8); LW = Inches(5.85)
chip(s, LX, Inches(1.4), LW, "① 三比值法编码规则（DL/T 722 表6）")
tbl = s.shapes.add_table(4, 2, LX, Inches(1.9), LW, Inches(1.55)).table
tbl.columns[0].width = Inches(2.9); tbl.columns[1].width = Inches(2.95)
data = [("特征比值", "编码规则"), ("C₂H₂ / C₂H₄", "<0.1→0 ; [0.1,3)→1 ; ≥3→2"),
        ("CH₄ / H₂", "<0.1→1 ; [0.1,1)→0 ; ≥1→2"), ("C₂H₄ / C₂H₆", "<1→0 ; [1,3)→1 ; ≥3→2")]
for i, (a, b) in enumerate(data):
    for j, t in enumerate((a, b)):
        cell = tbl.cell(i, j); cell.fill.solid()
        cell.fill.fore_color.rgb = PURPLE if i == 0 else (CARD1 if i % 2 else WHITE)
        tf = cell.text_frame; tf.word_wrap = True; tf.margin_top = Pt(3); tf.margin_bottom = Pt(3)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = t
        r.font.size = Pt(12.5 if i == 0 else 12); r.font.bold = (i == 0)
        r.font.color.rgb = WHITE if i == 0 else T1; r.font.name = F
chip(s, LX, Inches(3.7), LW, "② 朴素贝叶斯后验推理")
rect(s, LX, Inches(4.2), LW, Inches(0.9), PURPLE_L, round_=True)
txt(s, LX, Inches(4.2), LW, Inches(0.9), [[Rn("P(Fᵢ | E) ∝ P(Fᵢ) · ∏ P(eⱼ | Fᵢ)", 19, PURPLE_D, True)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, LX, Inches(5.2), LW, Inches(1.5), [
    [Rn("• P(Fᵢ) ", 12.5, PURPLE, True), Rn("故障先验（常见程度，如绝缘老化 0.18）", 12, T2)],
    [Rn("• P(eⱼ|Fᵢ) ", 12.5, PURPLE, True), Rn("条件概率表 CPT（故障发生时征兆出现概率）", 12, T2)],
    [Rn("• 仅对观测征兆连乘，最后归一化为概率分布", 12, T2)]], sa=5, ls=1.12)
RX = Inches(6.95); RW = Inches(5.6)
chip(s, RX, Inches(1.4), RW, "③ 规则—概率融合")
rect(s, RX, Inches(1.9), RW, Inches(1.2), PURPLE_L, round_=True)
txt(s, RX, Inches(1.98), RW, Inches(1.1), [
    [Rn("P_fused(Fᵢ) =", 15, PURPLE_D, True)],
    [Rn("0.7·P_bayes + 0.3·P_rule ", 16, PURPLE_D, True), Rn("（命中规则）", 11, T3)],
    [Rn("P_bayes ", 15, PURPLE_D, True), Rn("（未命中规则）", 11, T3)]], align=PP_ALIGN.CENTER, sa=2)
txt(s, RX, Inches(3.15), RW, Inches(0.4), [[Rn("以数据驱动概率为主(70%)，国标规则经验修正(30%)", 11.5, T3, False, True)]])
chip(s, RX, Inches(3.7), RW, "诊断算例：典型绕组匝间短路")
rect(s, RX, Inches(4.2), RW, Inches(2.55), CARD2, line=PURPLE, round_=True, lw=1.2)
txt(s, RX + Inches(0.22), Inches(4.35), RW - Inches(0.44), Inches(2.3), [
    [Rn("输入 DGA(μL/L)：", 12.5, PURPLE, True), Rn("H₂=620, CH₄=180,", 12, T1)],
    [Rn("                C₂H₂=240, C₂H₄=300, C₂H₆=85", 12, T1)],
    [Rn("① 编码 [1,0,2] → 命中“电弧放电”规则(0.93)", 12, T2)],
    [Rn("② 多征兆超标，短路 CPT 似然高(0.85~0.90)", 12, T2)],
    [Rn("③ 融合：0.7·P_bayes + 0.3·0.93 → 概率抬升", 12, T2)],
    [Rn("④ 输出：", 12.5, PURPLE, True), Rn("绕组匝间短路 排名第1 【高风险】", 12.5, PURPLE_D, True)]], sa=6, ls=1.06)

prs.save(str(OUT))
print("已生成:", OUT, "共", len(prs.slides._sldIdLst), "页")
