#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 AI 辅助预标注写入 data/kg/eval/precision_sample.jsonl 的 ai_prelabel / ai_note 字段。
人工 label 字段保持为空，供复核后填写；报告脚本会分别统计两者。

判定口径：
  correct : 三元组在变压器领域为真，且证据句直接支持
  unsure  : 领域上可能为真，但证据句不支持 / 为枚举句 / 方向不明
  wrong   : 三元组为假，或证据句明确指向其他标准/实体（抽取错配）
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/kg/eval/precision_sample.jsonl"

PRE = {
    "KGP-001": ("unsure", "证据为综合判断句，未直接说明冷却器部位"),
    "KGP-002": ("correct", ""),
    "KGP-003": ("correct", ""),
    "KGP-004": ("correct", "枚举句，但绕组局部放电为常识事实"),
    "KGP-005": ("correct", ""),
    "KGP-006": ("correct", ""),
    "KGP-007": ("correct", ""),
    "KGP-008": ("wrong", "枚举句误配：红外测温不用于检测局部放电"),
    "KGP-009": ("unsure", "证据句泛化，未指明过热故障"),
    "KGP-010": ("wrong", "DL/T 911 为频响法标准，证据句未提及振动法标准"),
    "KGP-011": ("unsure", "证据讲套管清洁预防放电，非处理放电故障"),
    "KGP-012": ("correct", ""),
    "KGP-013": ("correct", ""),
    "KGP-014": ("correct", ""),
    "KGP-015": ("correct", ""),
    "KGP-016": ("correct", "证据为吊芯补焊，与吊罩检查同义"),
    "KGP-017": ("correct", ""),
    "KGP-018": ("correct", ""),
    "KGP-019": ("correct", ""),
    "KGP-020": ("correct", "高温过热可产生少量乙炔，证据支持"),
    "KGP-021": ("correct", ""),
    "KGP-022": ("correct", ""),
    "KGP-023": ("correct", ""),
    "KGP-024": ("correct", ""),
    "KGP-025": ("correct", "枚举句，但油色谱可检测放电为事实"),
    "KGP-026": ("correct", ""),
    "KGP-027": ("correct", ""),
    "KGP-028": ("wrong", "证据句依据为 GB/T 7252，抽取错配到 DL/T 722"),
    "KGP-029": ("correct", ""),
    "KGP-030": ("correct", ""),
    "KGP-031": ("wrong", "证据讲油色变暗，与微水无关"),
    "KGP-032": ("correct", ""),
    "KGP-033": ("correct", ""),
    "KGP-034": ("unsure", "证据讲真空注油过程可能进水（反向），领域上真空注油确为受潮处理手段"),
    "KGP-035": ("correct", ""),
    "KGP-036": ("correct", ""),
    "KGP-037": ("correct", ""),
    "KGP-038": ("correct", ""),
    "KGP-039": ("unsure", "证据为色谱在线监测未见异常，支持力弱"),
    "KGP-040": ("correct", ""),
    "KGP-041": ("unsure", "因果链绕过杂质累积，直接因果关系存疑"),
    "KGP-042": ("correct", ""),
    "KGP-043": ("correct", ""),
    "KGP-044": ("wrong", "渗漏油的部位为油箱/密封处，'绝缘油'非部位"),
    "KGP-045": ("correct", ""),
    "KGP-046": ("correct", ""),
    "KGP-047": ("unsure", "领域为真（DL/T 911 即频响法），但证据句未提及标准"),
    "KGP-048": ("correct", ""),
    "KGP-049": ("correct", ""),
    "KGP-050": ("wrong", "短路阻抗法对应 DL/T 1093，非 DL/T 911"),
    "KGP-051": ("correct", ""),
    "KGP-052": ("correct", ""),
    "KGP-053": ("unsure", "证据句截断，疑为 GB/T 7252"),
    "KGP-054": ("correct", ""),
    "KGP-055": ("correct", ""),
    "KGP-056": ("correct", ""),
}


def main() -> None:
    rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = 0
    for r in rows:
        if r["sample_id"] in PRE:
            r["ai_prelabel"], r["ai_note"] = PRE[r["sample_id"]]
            n += 1
    P.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"写入 {n}/{len(rows)} 条 AI 预标注 → {P}")


if __name__ == "__main__":
    main()
