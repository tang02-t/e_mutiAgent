#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制中期论文配图：图3-1 系统总体架构图、图3-2 诊断工作流执行流程图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 中文字体（macOS 常见可用字体）
for cand in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#1F3A5F"
LBLUE = "#DCE6F1"
GREEN = "#2E7D54"
LGREEN = "#D9EAD3"
ORANGE = "#B5651D"
LORANGE = "#FCE5CD"


def box(ax, x, y, w, h, text, fc, ec=BLUE, fs=11, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#1a1a1a")


def arrow(ax, x1, y1, x2, y2, style="-|>", color=BLUE, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw, linestyle=ls))


# ───────────────────────── 图 3-1 系统总体架构图 ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# 三层分区背景
ax.add_patch(FancyBboxPatch((0.3, 8.0), 9.4, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#F2F6FB", ec="#9DB4CE", lw=1.2, ls="--"))
ax.text(0.55, 9.25, "编排层", fontsize=11, color=BLUE, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.3, 4.6), 9.4, 2.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#F4FBF6", ec="#8FC0A2", lw=1.2, ls="--"))
ax.text(0.55, 7.2, "智能体层", fontsize=11, color=GREEN, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.3, 0.5), 9.4, 2.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#FDF7F0", ec="#D2A679", lw=1.2, ls="--"))
ax.text(0.55, 2.45, "工具层", fontsize=11, color=ORANGE, fontweight="bold")

# 编排层
box(ax, 2.0, 8.25, 6.0, 0.95, "LangGraph 图式编排\n有向状态图 / 条件路由 / 共享状态 AgentState", LBLUE, fs=10.5)

# 智能体层（横向四个）
xs = [0.7, 3.0, 5.3, 7.6]
labels = ["Planner\n意图分析+任务分解", "Retriever\n工具调度(并行/串行)",
          "Generator\n诊断结论生成", "Validator\n质量验证+路由"]
for x, lab in zip(xs, labels):
    box(ax, x, 5.4, 1.9, 1.1, lab, LGREEN, ec=GREEN, fs=9.5)
# 智能体顺序箭头
for i in range(3):
    arrow(ax, xs[i] + 1.9, 5.95, xs[i + 1], 5.95, color=GREEN)
# Validator -> Generator 迭代回退（虚线）
arrow(ax, xs[3] + 0.95, 5.4, xs[2] + 0.95, 5.4 - 0.0, color=GREEN, ls=(0, (4, 3)))
ax.annotate("", xy=(xs[2] + 0.95, 5.35), xytext=(xs[3] + 0.95, 5.35),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                            connectionstyle="arc3,rad=-0.4", ls="--"))
ax.text(5.8, 4.75, "REVISION 迭代", fontsize=8.5, color=GREEN, style="italic")

# 编排层 -> Planner
arrow(ax, 5.0, 8.25, 1.65, 6.5, color=BLUE)

# 工具层（横向四个）
txs = [0.7, 3.0, 5.3, 7.6]
tlabels = ["RAG\n知识检索", "贝叶斯\n故障归因", "ETT\n油温预测", "时序\n异常检测"]
for x, lab in zip(txs, tlabels):
    box(ax, x, 0.9, 1.9, 1.0, lab, LORANGE, ec=ORANGE, fs=9.5)
# Retriever -> 工具层
for x in txs:
    arrow(ax, 3.95, 5.4, x + 0.95, 1.9, color=ORANGE, lw=1.2)

ax.set_title("图 3-1  系统总体架构图", fontsize=13, fontweight="bold", color="#1a1a1a", pad=12)
fig.tight_layout()
fig.savefig(OUT / "fig3_1_architecture.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ───────────────────────── 图 3-2 诊断工作流执行流程图 ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 5); ax.axis("off")

# 起点
ax.add_patch(FancyBboxPatch((0.2, 1.9), 2.0, 1.2, boxstyle="round,pad=0.02,rounding_size=0.3",
                            fc="#E8E8E8", ec="#666", lw=1.4))
ax.text(1.2, 2.5, "用户诊断请求", ha="center", va="center", fontsize=10, fontweight="bold")

steps = [("Planner\n规划", 3.0), ("Retriever\n检索", 5.6), ("Generator\n生成", 8.2)]
for lab, x in steps:
    box(ax, x, 1.85, 2.0, 1.3, lab, LGREEN, ec=GREEN, fs=10)

# Validator 菱形（用旋转方框近似）
vx, vy = 11.0, 2.5
diamond = plt.Polygon([(vx, vy + 1.0), (vx + 1.3, vy), (vx, vy - 1.0), (vx - 1.3, vy)],
                      fc=LBLUE, ec=BLUE, lw=1.6)
ax.add_patch(diamond)
ax.text(vx, vy, "Validator\n验证", ha="center", va="center", fontsize=10, fontweight="bold")

# 终点
ax.add_patch(FancyBboxPatch((13.8, 1.9), 2.0, 1.2, boxstyle="round,pad=0.02,rounding_size=0.3",
                            fc="#E8E8E8", ec="#666", lw=1.4))
ax.text(14.8, 2.5, "输出诊断报告", ha="center", va="center", fontsize=10, fontweight="bold")

# 主流程箭头
arrow(ax, 2.2, 2.5, 3.0, 2.5, color=GREEN)
arrow(ax, 5.0, 2.5, 5.6, 2.5, color=GREEN)
arrow(ax, 7.6, 2.5, 8.2, 2.5, color=GREEN)
arrow(ax, 10.2, 2.5, vx - 1.3, 2.5, color=GREEN)
# PASS -> END
arrow(ax, vx + 1.3, 2.5, 13.8, 2.5, color=BLUE)
ax.text(12.8, 2.75, "PASS", fontsize=8.5, color=BLUE, fontweight="bold")
# REVISION 回退 Generator
ax.annotate("", xy=(9.2, 3.15), xytext=(vx, vy + 1.0),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5,
                            connectionstyle="arc3,rad=0.35", ls="--"))
ax.text(9.6, 4.1, "REVISION（未达上限）迭代", fontsize=8.5, color=ORANGE, style="italic")
# FAIL / 达上限 -> END（向下绕）
ax.annotate("", xy=(14.8, 1.9), xytext=(vx, vy - 1.0),
            arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.4,
                            connectionstyle="arc3,rad=-0.3", ls=":"))
ax.text(12.2, 1.05, "FAIL / 达上限", fontsize=8.5, color="#777", style="italic")

ax.set_title("图 3-2  诊断工作流执行流程图", fontsize=13, fontweight="bold", color="#1a1a1a", pad=10)
fig.tight_layout()
fig.savefig(OUT / "fig3_2_workflow.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("图片已生成：")
print(" -", OUT / "fig3_1_architecture.png")
print(" -", OUT / "fig3_2_workflow.png")
