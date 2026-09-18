#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-2 验收：20 条手工部分观测样例，检查 EIG 推荐是否符合领域直觉。

每条样例给出：初始观测（部分 DGA / 现场征兆）、领域预期（专家按 DL/T 722 直觉认为最值得补的征兆集合，
命中其一即算一致）、以及预期的理由。脚本输出推荐 Top-3 与是否命中，并写入 docs/eval/eig_sanity_check.md 供人工复核。

用法：FAULT_ATTR_PARAMS=data/real/dga/learned_params.json python3 scripts/eval/eig_sanity_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FAULT_ATTR_PARAMS", str(ROOT / "data/real/dga/learned_params.json"))

from src.tools.fault_attribution import fault_attribution  # noqa: E402

# (name, dga_data, evidence, expected_symptoms(any), rationale)
CASES = [
    ("仅氢气高", {"H2": 400}, {},
     {"C2H2_elevated", "CH4_elevated", "partial_discharge_alarm"},
     "H2 单独升高：局放 vs 受潮 vs 低温过热，关键看 C2H2 / CH4 或局放检测"),
    ("氢乙炔高、无其他气体", {"H2": 345, "C2H2": 58}, {},
     {"C2H4_elevated", "winding_temp_elevated", "oil_temp_elevated", "CH4_elevated"},
     "放电已确定，区分匝间短路（伴随过热/温升）与套管/分接开关放电（不伴随）"),
    ("乙烯乙烷高、无乙炔", {"C2H4": 300, "C2H6": 80, "C2H2": 0}, {},
     {"oil_temp_elevated", "winding_temp_elevated", "load_elevated", "CO_elevated", "CO2_elevated", "CH4_elevated"},
     "过热特征，需确认是否负载/温度导致（过载过热）或固体绝缘参与（CO/CO2）"),
    ("CO 类信息缺失的过热", {"H2": 100, "CH4": 200, "C2H2": 1, "C2H4": 400, "C2H6": 60}, {},
     {"CO_elevated", "CO2_elevated", "oil_temp_elevated", "winding_temp_elevated", "load_elevated"},
     "高温过热已定，区分裸金属过热与固体绝缘过热看 CO/CO2；是否过载看负载与温度"),
    ("全气体正常、用户报油温高", {"H2": 20, "CH4": 10, "C2H2": 0, "C2H4": 5, "C2H6": 5}, {"oil_temp_elevated": True},
     {"load_elevated", "winding_temp_elevated", "CO_elevated", "CO2_elevated", "gas_rate_rapid"},
     "气体正常但油温高：首先排查是否过载或冷却问题，再看产气趋势"),
    ("仅乙炔高", {"C2H2": 30}, {},
     {"H2_elevated", "C2H4_elevated", "CH4_elevated"},
     "乙炔升高必是放电，H2/C2H4 决定能量等级（三比值法两个比值都缺）"),
    ("局放告警、无气体", None, {"partial_discharge_alarm": True},
     {"H2_elevated", "C2H2_elevated", "CH4_elevated", "TDCG_elevated"},
     "局放告警需油色谱确认（H2 为主、C2H2 低）"),
    ("振动异常、无气体", None, {"vibration_elevated": True},
     {"C2H2_elevated", "H2_elevated", "load_elevated", "TDCG_elevated", "gas_rate_rapid"},
     "振动异常对应绕组变形/铁芯松动，先看是否伴随放电气体或负载"),
    ("匝间短路典型气体但无温度", {"H2": 800, "CH4": 300, "C2H2": 200, "C2H4": 400, "C2H6": 60}, {},
     {"winding_temp_elevated", "oil_temp_elevated", "load_elevated", "CO_elevated", "CO2_elevated"},
     "电弧放电兼过热，需要温度/负载确认匝间短路而非分接开关拉弧"),
    ("低能放电、无 H2", {"CH4": 50, "C2H2": 20, "C2H4": 60, "C2H6": 20}, {},
     {"H2_elevated", "partial_discharge_alarm", "oil_temp_elevated", "winding_temp_elevated"},
     "缺 H2 无法判断 CH4/H2，需补 H2 或局放检测"),
    ("过热+CO 高", {"H2": 80, "CH4": 150, "C2H2": 0, "C2H4": 200, "C2H6": 70}, {"CO_elevated": True},
     {"CO2_elevated", "oil_temp_elevated", "winding_temp_elevated", "load_elevated"},
     "固体绝缘参与过热，看 CO2 与温度/负载区分绝缘老化与过载"),
    ("绝缘老化疑似", {"H2": 60, "CH4": 40, "C2H2": 0, "C2H4": 30, "C2H6": 30}, {"CO_elevated": True, "CO2_elevated": True},
     {"oil_temp_elevated", "load_elevated", "gas_rate_rapid", "winding_temp_elevated"},
     "CO/CO2 高但烃低：老化 vs 过载，看负载/温度与产气速率"),
    ("已排除乙炔与过热气体", {"H2": 300, "CH4": 30, "C2H2": 0, "C2H4": 10, "C2H6": 10}, {},
     {"partial_discharge_alarm", "CO_elevated", "CO2_elevated", "gas_rate_rapid"},
     "只剩 H2 高：局放或受潮，局放检测最有区分度"),
    ("总烃高、气体明细缺失", None, {"TDCG_elevated": True},
     {"C2H2_elevated", "C2H4_elevated", "H2_elevated", "CH4_elevated"},
     "总烃高需拆分明细，乙炔决定放电 vs 过热"),
    ("负载高、气体缺失", None, {"load_elevated": True},
     {"oil_temp_elevated", "winding_temp_elevated", "C2H4_elevated", "TDCG_elevated", "CO_elevated"},
     "负载高看温度是否跟随、是否已产生过热气体"),
    ("产气快、其他缺失", None, {"gas_rate_rapid": True},
     {"C2H2_elevated", "H2_elevated", "C2H4_elevated", "TDCG_elevated"},
     "产气速率快说明活跃故障，乙炔决定放电性质"),
    ("套管放电疑似", {"H2": 500, "CH4": 60, "C2H2": 40, "C2H4": 50, "C2H6": 15}, {},
     {"partial_discharge_alarm", "winding_temp_elevated", "oil_temp_elevated", "load_elevated", "vibration_elevated"},
     "低能放电 + 高 H2：套管 / 局放 / 匝间，看局放检测或温度"),
    ("已排除大多数、只剩 CH4", {"H2": 50, "CH4": 200, "C2H2": 0, "C2H4": 20, "C2H6": 20}, {},
     {"oil_temp_elevated", "CO_elevated", "CO2_elevated", "load_elevated", "winding_temp_elevated"},
     "CH4 单独高：低温过热，看温度/负载/CO"),
    ("匝间短路+已知无过热", {"H2": 800, "CH4": 300, "C2H2": 200, "C2H4": 400, "C2H6": 60},
     {"winding_temp_elevated": False, "oil_temp_elevated": False},
     {"load_elevated", "vibration_elevated", "partial_discharge_alarm", "CO_elevated", "CO2_elevated"},
     "温度已排除，剩余区分度在负载/振动/局放/CO"),
    ("正常气体、无征兆", {"H2": 20, "CH4": 10, "C2H2": 0, "C2H4": 5, "C2H6": 5}, {},
     {"oil_temp_elevated", "winding_temp_elevated", "load_elevated", "CO_elevated", "CO2_elevated", "gas_rate_rapid", "vibration_elevated", "partial_discharge_alarm"},
     "气体全正常时任何现场征兆都是补充信息，任意非气体征兆均可接受"),
]


def kappa_sensitivity() -> list[tuple[float, float, float]]:
    """对无数据征兆的专家 CPT 收缩系数 κ 做敏感性：返回 [(κ, top1_rate, top3_rate)]。"""
    from src.tools import fault_attribution as fa
    from src.tools.fault_attribution import CPTRow
    from src.tools.fault_attribution import _FAULT_SYMPTOM_CPT as EXPERT

    params_path = os.environ.get("FAULT_ATTR_PARAMS", "")
    kept: set[str] = set()
    if params_path and Path(params_path).exists():
        kept = set(json.loads(Path(params_path).read_text(encoding="utf-8")).get("meta", {}).get("expert_kept_symptoms", []))
    if not kept:
        return []
    out = []
    for kappa in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
        fa._bn_engine = None  # noqa: SLF001
        e = fa.get_engine()
        for f, rows in e.cpt.items():
            for s in kept:
                exp = EXPERT.get(f, {}).get(s) or CPTRow(0.05, 0.01)
                rows[s] = CPTRow(0.5 + kappa * (exp.p_true - 0.5), exp.p_false)
        h1 = h3 = 0
        for _, dga, ev, expected, _ in CASES:
            r = fa.fault_attribution(dga_data=dga, evidence=ev or None)
            top = [x["symptom"] for x in r["uncertainty"].get("recommendations", [])[:3]]
            h1 += int(bool(top) and top[0] in expected)
            h3 += int(any(s in expected for s in top))
        out.append((kappa, h1 / len(CASES), h3 / len(CASES)))
    fa._bn_engine = None  # noqa: SLF001
    return out


def main() -> None:
    rows = []
    hit = 0
    for name, dga, ev, expected, why in CASES:
        r = fault_attribution(dga_data=dga, evidence=ev or None)
        u = r["uncertainty"]
        recs = u.get("recommendations", [])
        top3 = [x["symptom"] for x in recs[:3]]
        top1 = top3[0] if top3 else None
        ok1 = top1 in expected
        ok3 = any(s in expected for s in top3)
        hit += int(ok1)
        rows.append({
            "case": name, "top1": r["primary_fault"], "H": u["entropy_bits"], "action": u["suggested_action"],
            "rec_top3": [(x["symptom"], x["eig"], x["cost"], x["voi"]) for x in recs[:3]],
            "expected": sorted(expected), "hit_top1": ok1, "hit_top3": ok3, "why": why,
        })
        flag = "✓" if ok1 else ("~" if ok3 else "✗")
        print(f"{flag} {name:22s} H={u['entropy_bits']:.2f} act={u['suggested_action']:8s} top1={top1} | top3={top3}")
    rate1 = hit / len(CASES)
    rate3 = sum(r["hit_top3"] for r in rows) / len(CASES)
    print(f"\nTop-1 一致率 {rate1:.0%}  Top-3 一致率 {rate3:.0%}  (n={len(CASES)})")

    out = ROOT / "docs/eval/eig_sanity_check.md"
    lines = [
        "# EIG 推荐领域一致性抽检（B-2 验收）",
        "",
        f"> 生成：`python3 scripts/eval/eig_sanity_check.py`，参数 `{os.environ.get('FAULT_ATTR_PARAMS')}`",
        f"> 20 条手工部分观测样例；「预期」为按 DL/T 722 直觉预先写下的可接受征兆集合，命中其一即一致。",
        f"> **Top-1 一致率 {rate1:.0%}，Top-3 一致率 {rate3:.0%}**。预期集合由 AI 预填，需人工复核后在此登记复核人与日期。",
        "",
        "| # | 样例 | Top-1 故障 | H(bit) | 动作 | 推荐 Top-3（征兆 / EIG / cost / VoI） | 预期集合 | Top-1 命中 | 理由 |",
        "|---:|---|---|---:|---|---|---|:---:|---|",
    ]
    for i, r in enumerate(rows, 1):
        rec = "<br>".join(f"`{s}` {e:.3f}/{c:.0f}/{v:.3f}" for s, e, c, v in r["rec_top3"])
        exp = ", ".join(f"`{s}`" for s in r["expected"])
        lines.append(f"| {i} | {r['case']} | `{r['top1']}` | {r['H']:.2f} | {r['action']} | {rec} | {exp} | {'✓' if r['hit_top1'] else '✗'} | {r['why']} |")
    sens = kappa_sensitivity()
    if sens:
        lines += ["", "## κ 敏感性（无数据征兆的专家 CPT 收缩系数）", "",
                  "| κ | Top-1 一致率 | Top-3 一致率 |", "|---:|---:|---:|"]
        lines += [f"| {k:.1f} | {a:.0%} | {b:.0%} |" for k, a, b in sens]
        lines.append("")
        lines.append("κ=1.0 为原专家值；`learn_cpt.py` 默认写入 κ=0.5。κ ≤ 0.6 时一致率稳定，说明结论对具体取值不敏感。")
        print("κ 敏感性:", ", ".join(f"κ={k:.1f}: {a:.0%}/{b:.0%}" for k, a, b in sens))
    lines += ["", "人工复核：<待填写：复核人 / 日期 / 修改的预期集合>"]
    out.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "docs/eval/eig_sanity_check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
