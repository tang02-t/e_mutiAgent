#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯网络参数学习脚本。

用真实标注 DGA 数据（data/real/dga/dga_records.jsonl）统计学习：
  - 先验  P(F)        = 各 fault_label 频率
  - 条件概率 CPT P(S|F):
        p_true  = P(症状=True | 故障=F)   —— 该故障样本中征兆出现的比例
        p_false = P(症状=True | 故障≠F)   —— 非该故障样本中征兆出现的比例
均采用拉普拉斯平滑（Laplace add-k），避免 0 概率导致似然连乘塌缩。

把学到的参数写入 data/real/dga/learned_params.json，供 fault_attribution 加载，
替代原先“专家拍脑袋”的 CPT，实现数据驱动的故障归因。

为客观评估，支持按比例留出测试集（仅用训练集学参数，在测试集上报告准确率）。

用法：
  python3 scripts/learn_cpt.py                       # 全量学习
  python3 scripts/learn_cpt.py --test-ratio 0.3      # 70%学习/30%评估
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.fault_attribution import FAULT_IDS, SYMPTOM_IDS  # noqa: E402

ALPHA = 1.0  # 拉普拉斯平滑系数


def load_records(jsonl: Path) -> list[dict]:
    recs = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def evidence_set(rec: dict) -> set[str]:
    """返回该记录中为 True 的征兆集合。"""
    return {k for k, v in (rec.get("evidence") or {}).items() if v}


def learn(records: list[dict]) -> dict:
    """从记录中统计先验与 CPT（含拉普拉斯平滑）。"""
    n = len(records)
    fault_counter = Counter(r["fault_label"] for r in records)

    # 先验 P(F)，平滑
    prior = {}
    for f in FAULT_IDS:
        prior[f] = (fault_counter.get(f, 0) + ALPHA) / (n + ALPHA * len(FAULT_IDS))

    # 统计每个 (fault, symptom) 的共现
    # pos[f][s] = 故障f样本中 s 出现次数; n_f = 故障f样本数
    pos = defaultdict(lambda: defaultdict(int))
    neg = defaultdict(lambda: defaultdict(int))  # 非f样本中 s 出现次数
    n_f = Counter()
    for r in records:
        f = r["fault_label"]
        ev = evidence_set(r)
        n_f[f] += 1
        for s in SYMPTOM_IDS:
            if s in ev:
                pos[f][s] += 1

    # neg: 对每个 f，非 f 样本中 s 的出现次数 = 全体 s 出现次数 - pos[f][s]
    total_pos = defaultdict(int)
    for r in records:
        ev = evidence_set(r)
        for s in SYMPTOM_IDS:
            if s in ev:
                total_pos[s] += 1

    cpt = {}
    for f in FAULT_IDS:
        nf = n_f.get(f, 0)
        n_not_f = n - nf
        row = {}
        for s in SYMPTOM_IDS:
            p_true = (pos[f][s] + ALPHA) / (nf + 2 * ALPHA)
            neg_count = total_pos[s] - pos[f][s]
            p_false = (neg_count + ALPHA) / (n_not_f + 2 * ALPHA)
            row[s] = {"p_true": round(p_true, 5), "p_false": round(p_false, 5)}
        cpt[f] = row

    return {
        "meta": {
            "n_samples": n,
            "alpha": ALPHA,
            "fault_distribution": dict(fault_counter),
        },
        "prior": {k: round(v, 5) for k, v in prior.items()},
        "cpt": cpt,
    }


def evaluate(records: list[dict], params_path: Path | None) -> tuple[float, float, int]:
    """在给定记录上评估 fault_attribution 的 Top-1/Top-3（可指定参数文件）。"""
    import os
    if params_path:
        os.environ["FAULT_ATTR_PARAMS"] = str(params_path)
    else:
        os.environ.pop("FAULT_ATTR_PARAMS", None)

    # 强制重新加载引擎以应用新参数
    import importlib
    import src.tools.fault_attribution as fa
    importlib.reload(fa)

    correct = top3 = total = 0
    for r in records:
        if r["fault_label"] == "normal":
            continue
        res = fa.fault_attribution(dga_data=r["dga_data"], evidence=r["evidence"], query="")
        pred = res["primary_fault"]
        ranking = [x["fault_id"] for x in res["fault_ranking"][:3]]
        total += 1
        if pred == r["fault_label"]:
            correct += 1
        if r["fault_label"] in ranking:
            top3 += 1
    if total == 0:
        return 0.0, 0.0, 0
    return correct / total, top3 / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "data/real/dga/dga_records.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/real/dga/learned_params.json"))
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_records(Path(args.jsonl))
    print(f"载入记录: {len(records)} 条")

    # 划分训练/测试
    random.seed(args.seed)
    random.shuffle(records)
    n_test = int(len(records) * args.test_ratio)
    test = records[:n_test]
    train = records[n_test:]
    print(f"训练集: {len(train)} 条 | 测试集: {len(test)} 条 (test_ratio={args.test_ratio})")

    # 学习参数（仅用训练集）
    params = learn(train)
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"已写出学习参数: {out_path}")

    # 评估对比：测试集上 学习前(专家CPT) vs 学习后(数据CPT)
    print("\n================ 测试集评估对比 ================")
    a1, a3, n = evaluate(test, params_path=None)
    print(f"[学习前·专家CPT ] Top-1={a1:.1%}  Top-3={a3:.1%}  (n={n})")
    b1, b3, _ = evaluate(test, params_path=out_path)
    print(f"[学习后·数据CPT ] Top-1={b1:.1%}  Top-3={b3:.1%}  (n={n})")
    print(f"\n提升: Top-1 {a1:.1%} -> {b1:.1%} ({(b1 - a1) * 100:+.1f}pt)"
          f" | Top-3 {a3:.1%} -> {b3:.1%} ({(b3 - a3) * 100:+.1f}pt)")

    # 全量再学一份用于线上（覆盖测试用的训练版）
    full_params = learn(records)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_params, f, ensure_ascii=False, indent=2)
    print(f"\n已用全量数据重学并覆盖参数文件，供线上加载: {out_path}")


if __name__ == "__main__":
    main()
