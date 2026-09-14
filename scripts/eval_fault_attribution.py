#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【评测一】故障归因引擎评测（确定性，无 LLM / 无网络依赖）。

针对 src/tools/fault_attribution.py 的贝叶斯归因引擎，用带真实标签的 DGA 数据
计算分类指标，量化"故障归因"这一核心能力的实际效果。

输出指标：
  - Top-1 准确率 / Top-3 命中率
  - 每类故障的 Precision / Recall / F1（宏平均 + 加权平均）
  - 混淆矩阵
  - 平均归因置信度、置信度-正确性相关性（校准性初探）
  - 分严重程度（critical/serious/...）的召回率（安全视角：高危故障是否漏诊）

用法：
  python3 scripts/eval_fault_attribution.py
  python3 scripts/eval_fault_attribution.py --jsonl <path> --out <report.md>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.fault_attribution import fault_attribution  # type: ignore  # noqa: E402

# 故障中文名映射
FAULT_CN = {
    "normal": "正常", "winding_deformation": "绕组变形", "winding_short_circuit": "绕组匝间短路",
    "core_grounding": "铁芯多点接地", "bushing_fault": "套管故障", "olc_fault": "分接开关故障",
    "partial_discharge": "局部放电", "overload_overheating": "过载过热", "insulation_degradation": "绝缘老化",
}


def evaluate(jsonl_path: str):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["fault_label"] == "normal":
                continue  # 归因引擎不输出 normal，只评估故障样本
            records.append(r)

    n = len(records)
    if n == 0:
        raise SystemExit("无可评测样本。")

    labels = sorted({r["fault_label"] for r in records})

    top1 = top3 = 0
    confusion = defaultdict(Counter)          # true -> pred
    tp = Counter(); fp = Counter(); fn = Counter()
    conf_correct = []; conf_wrong = []        # 置信度分桶
    sev_total = Counter(); sev_hit = Counter()  # 分严重程度召回

    for r in records:
        res = fault_attribution(dga_data=r["dga_data"], evidence=r["evidence"], query="")
        true = r["fault_label"]
        pred = res["primary_fault"]
        prob = res.get("primary_probability", 0.0)
        ranking = [x["fault_id"] for x in res["fault_ranking"][:3]]

        confusion[true][pred] += 1
        sev = r.get("severity", "unknown")
        sev_total[sev] += 1

        if pred == true:
            top1 += 1; tp[true] += 1; conf_correct.append(prob)
            sev_hit[sev] += 1
        else:
            fp[pred] += 1; fn[true] += 1; conf_wrong.append(prob)

        if true in ranking:
            top3 += 1

    # Precision/Recall/F1
    per_class = {}
    for c in labels:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        support = sum(confusion[c].values())
        per_class[c] = {"precision": p, "recall": rec, "f1": f1, "support": support}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / n

    avg_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
    avg_conf_wrong = sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0.0

    return {
        "n": n, "labels": labels,
        "top1": top1 / n, "top3": top3 / n,
        "per_class": per_class, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "avg_conf_correct": avg_conf_correct, "avg_conf_wrong": avg_conf_wrong,
        "sev_recall": {s: sev_hit[s] / sev_total[s] for s in sev_total},
        "sev_total": dict(sev_total),
    }


def render_report(m: dict) -> str:
    L = []
    L.append("# 故障归因引擎评测报告\n")
    L.append(f"- 评测样本数（非正常）：**{m['n']}**")
    L.append(f"- **Top-1 准确率**：{m['top1']:.1%}")
    L.append(f"- **Top-3 命中率**：{m['top3']:.1%}")
    L.append(f"- **Macro-F1**：{m['macro_f1']:.3f}　**Weighted-F1**：{m['weighted_f1']:.3f}")
    L.append(f"- 正确预测平均置信度：{m['avg_conf_correct']:.3f}；错误预测平均置信度：{m['avg_conf_wrong']:.3f}")
    L.append("")

    L.append("## 各故障类型 Precision / Recall / F1\n")
    L.append("| 故障类型 | Precision | Recall | F1 | 样本数 |")
    L.append("|----------|-----------|--------|----|--------|")
    for c, v in sorted(m["per_class"].items(), key=lambda x: -x[1]["support"]):
        L.append(f"| {FAULT_CN.get(c, c)} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
    L.append("")

    L.append("## 分严重程度召回率（安全视角：高危故障漏诊检查）\n")
    L.append("| 严重程度 | 召回率 | 样本数 |")
    L.append("|----------|--------|--------|")
    order = ["critical", "serious", "attention", "normal", "unknown"]
    for s in order:
        if s in m["sev_total"]:
            L.append(f"| {s} | {m['sev_recall'][s]:.1%} | {m['sev_total'][s]} |")
    L.append("")

    L.append("## 混淆矩阵（行=真实，列=预测主故障）\n")
    labels = m["labels"]
    header = "| 真实\\预测 | " + " | ".join(FAULT_CN.get(c, c) for c in labels) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(labels) + 1))
    for t in labels:
        row = m["confusion"].get(t, {})
        cells = " | ".join(str(row.get(p, 0)) for p in labels)
        L.append(f"| {FAULT_CN.get(t, t)} | {cells} |")
    L.append("")

    L.append("## 结论解读\n")
    L.append("- **Top-3 命中率**高说明数据特征与故障强相关，归因引擎能把真实故障排进前三。")
    L.append("- **Top-1 偏低 / 某类 Recall 低**通常源于贝叶斯网络 CPT 为人工近似值（likelihood 连乘偏向高产气故障），")
    L.append("  这是用真实标注数据校准 CPT 的直接依据。")
    L.append("- **critical/serious 召回率**是安全底线指标：高危故障一旦漏诊（误判为低危）后果严重，应重点优化。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "data/synthetic/dga/dga_records.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/synthetic/eval/report_fault_attribution.md"))
    args = ap.parse_args()

    m = evaluate(args.jsonl)
    report = render_report(m)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")

    # 控制台摘要
    print(f"样本数: {m['n']}")
    print(f"Top-1: {m['top1']:.1%}  Top-3: {m['top3']:.1%}")
    print(f"Macro-F1: {m['macro_f1']:.3f}  Weighted-F1: {m['weighted_f1']:.3f}")
    print("分严重程度召回:")
    for s in ["critical", "serious", "attention"]:
        if s in m["sev_total"]:
            print(f"  {s:10s} {m['sev_recall'][s]:.1%} (n={m['sev_total'][s]})")
    print(f"\n完整报告已写入: {args.out}")


if __name__ == "__main__":
    main()
