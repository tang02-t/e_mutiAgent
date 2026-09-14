#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA 模拟数据质量校验脚本。

用生成的 DGA 数据调用 src/tools/fault_attribution.py 的贝叶斯归因引擎，
统计 Top-1 / Top-3 命中率与混淆分布，验证数据与故障特征的相关性。

说明：
  - Top-3 命中率高（>85%）说明数据物理自洽，气体特征与标签强相关；
  - Top-1 偏低主要源于贝叶斯网络 CPT 为人工近似值（likelihood 连乘偏向高产气故障），
    这正说明需要用真实标注数据来校准 CPT —— 替换真实数据后用本脚本即可量化改进。

用法：
  python3 scripts/validate_dga_data.py
  python3 scripts/validate_dga_data.py --jsonl data/synthetic/dga/dga_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.fault_attribution import fault_attribution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "data/synthetic/dga/dga_records.jsonl"))
    args = ap.parse_args()

    correct = top3 = total = 0
    mis = defaultdict(Counter)

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["fault_label"] == "normal":
                continue
            res = fault_attribution(dga_data=r["dga_data"], evidence=r["evidence"], query="")
            pred = res["primary_fault"]
            ranking = [x["fault_id"] for x in res["fault_ranking"][:3]]
            total += 1
            if pred == r["fault_label"]:
                correct += 1
            if r["fault_label"] in ranking:
                top3 += 1
            mis[r["fault_label"]][pred] += 1

    if total == 0:
        print("无非正常样本可校验。")
        return

    print(f"非正常样本数 : {total}")
    print(f"Top-1 准确率 : {correct / total:.1%}")
    print(f"Top-3 命中率 : {top3 / total:.1%}")
    print("\n--- 各故障类型归因主结果分布 ---")
    for lbl, c in sorted(mis.items()):
        tot = sum(c.values())
        acc = c.get(lbl, 0) / tot
        print(f"  {lbl:24s} acc={acc:4.0%}  top={c.most_common(2)}")


if __name__ == "__main__":
    main()
