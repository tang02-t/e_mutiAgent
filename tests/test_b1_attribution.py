#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-1 单元测试：归因引擎负观测、温度缩放、按类别融合权重、posterior_only、征兆推导。

运行：python3 tests/test_b1_attribution.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("FAULT_ATTR_PARAMS", None)
os.environ.pop("FAULT_ATTR_NEG_EVIDENCE", None)

from src.tools import fault_attribution as fa  # noqa: E402
from src.tools.fault_attribution import (  # noqa: E402
    FAULT_IDS,
    FaultBayesianNetwork,
    derive_gas_evidence,
    fault_attribution,
)

PASSED = FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ── 1. 负观测 ──
print("1. 负观测")
eng = FaultBayesianNetwork(params={})
p_pos = eng._compute_posterior({"C2H2_elevated": True})
p_neg = eng._compute_posterior({"C2H2_elevated": False})
p_none = eng._compute_posterior({"C2H2_elevated": None})
p_empty = eng._compute_posterior({})
check("None 与缺失等价（跳过）", all(close(p_none[f], p_empty[f]) for f in FAULT_IDS))
check("负观测改变分布", any(not close(p_neg[f], p_empty[f], 1e-6) for f in FAULT_IDS))
check("C2H2 排除后匝间短路概率低于先验", p_neg["winding_short_circuit"] < p_empty["winding_short_circuit"])
check("C2H2 出现后匝间短路概率高于先验", p_pos["winding_short_circuit"] > p_empty["winding_short_circuit"])
check("负观测后仍归一化", close(sum(p_neg.values()), 1.0, 1e-9))

eng_off = FaultBayesianNetwork(params={}, use_negative_evidence=False)
p_off = eng_off._compute_posterior({"C2H2_elevated": False})
check("关闭负观测时 False 被跳过（v1 行为）", all(close(p_off[f], p_empty[f]) for f in FAULT_IDS))

# 手算校验：单故障、单征兆 P(F|S=False) ∝ P(F)(1-p_true)
f0 = "partial_discharge"
row = eng._cpt_lookup(f0, "C2H2_elevated")
manual = {f: eng.prior[f] * (1 - eng._cpt_lookup(f, "C2H2_elevated").p_true) for f in FAULT_IDS}
z = sum(manual.values())
check("负观测数值与手算一致", close(p_neg[f0], manual[f0] / z, 1e-9), f"{p_neg[f0]} vs {manual[f0] / z}")

# ── 2. 温度缩放 ──
print("2. 温度缩放")
eng_t = FaultBayesianNetwork(params={"calibration": {"method": "temperature", "T": 2.0}})
check("加载 T", close(eng_t.temperature, 2.0) and eng_t.is_calibrated)
raw = {"a": 0.8, "b": 0.15, "c": 0.05}
sc = eng_t._apply_temperature(raw)
check("T>1 使分布变平", sc["a"] < raw["a"] and sc["c"] > raw["c"])
check("缩放后归一化", close(sum(sc.values()), 1.0, 1e-9))
check("缩放不改变排序", sorted(raw, key=raw.get) == sorted(sc, key=sc.get))
exp = {k: v ** 0.5 for k, v in raw.items()}
zz = sum(exp.values())
check("缩放数值 p^(1/T) 归一化", all(close(sc[k], exp[k] / zz, 1e-12) for k in raw))
check("T=1 恒等", FaultBayesianNetwork(params={})._apply_temperature(raw) == raw)
check("熵随 T 增大", FaultBayesianNetwork.entropy_bits(sc) > FaultBayesianNetwork.entropy_bits(raw))

# ── 3. 融合权重 ──
print("3. 按类别融合权重")
eng_w = FaultBayesianNetwork(params={"fusion_weights": {"overload_overheating": 0.0, "partial_discharge": 1.0}})
check("默认权重 0.7", close(FaultBayesianNetwork(params={}).fusion_weight("olc_fault"), 0.7))
check("按类别读取", close(eng_w.fusion_weight("overload_overheating"), 0.0) and close(eng_w.fusion_weight("partial_discharge"), 1.0))
check("未指定类别回落默认", close(eng_w.fusion_weight("bushing_fault"), 0.7))
check("权重越界裁剪", close(FaultBayesianNetwork(params={"fusion_weights": {"olc_fault": 1.7}}).fusion_weight("olc_fault"), 1.0))
# 高温过热 DGA（C2H4/C2H6>=3，C2H2/C2H4<0.1）→ 命中规则；w=1 应等于纯贝叶斯
ctx = {"H2": 100, "CH4": 200, "C2H2": 1, "C2H4": 400, "C2H6": 60}
eng_w1 = FaultBayesianNetwork(params={"fusion_weights": {f: 1.0 for f in FAULT_IDS}})
dga = eng_w1._analyze_dga(ctx)
check("样例命中三比值规则", len(dga["matched_rules"]) > 0)
bayes = eng_w1._compute_posterior({"C2H4_elevated": True})
fused = {r["fault_id"]: r["probability"] for r in eng_w1._fuse_results(bayes, dga)}
check("w=1 时融合结果等于贝叶斯后验", all(close(fused[f], bayes[f], 1e-9) for f in FAULT_IDS))

# ── 4. posterior_only 与 uncertainty ──
print("4. posterior_only / uncertainty")
po = eng.posterior_only({"H2_elevated": True, "C2H2_elevated": False}, ctx)
check("返回字段齐全", {"probs", "bayes_probs", "entropy_bits", "top1", "top1_prob", "top1_top2_gap", "calibrated"} <= set(po))
check("probs 归一化", close(sum(po["probs"].values()), 1.0, 1e-9))
check("熵在 [0, log2(8)]", 0.0 <= po["entropy_bits"] <= math.log2(len(FAULT_IDS)) + 1e-9)
check("gap = p1 - p2", close(po["top1_top2_gap"], sorted(po["probs"].values(), reverse=True)[0] - sorted(po["probs"].values(), reverse=True)[1]))
check("无观测时熵最大附近（先验分布）", eng.posterior_only({})["entropy_bits"] > 2.5)
# 与 infer 一致
inf = eng.infer({"H2_elevated": True, "C2H2_elevated": False}, ctx)
check("infer 输出 uncertainty 块", "uncertainty" in inf and close(inf["uncertainty"]["entropy_bits"], po["entropy_bits"], 1e-9))
check("infer 与 posterior_only 的 top1 一致", inf["primary_fault"] == po["top1"])
check("infer 记录负观测", inf["evidence_negative"] == ["C2H2_elevated"])
check("fault_ranking 保留 probability_raw", all("probability_raw" in r for r in inf["fault_ranking"]))
check("报告展示已排除征兆", "已排除征兆" in inf["report"])

# ── 5. derive_gas_evidence ──
print("5. 征兆推导")
ev = derive_gas_evidence({"H2": 200, "CH4": 10, "C2H2": 0, "C2H4": 10, "C2H6": 10})
check("超注意值为 True", ev["H2_elevated"] is True)
check("低于注意值为 False", ev["CH4_elevated"] is False and ev["C2H2_elevated"] is False)
check("总烃 30 < 150 → TDCG False", ev["TDCG_elevated"] is False)
check("C2H2 0 → gas_rate_rapid False", ev["gas_rate_rapid"] is False)
ev_pos_only = derive_gas_evidence({"H2": 200, "CH4": 10, "C2H2": 0, "C2H4": 10, "C2H6": 10}, negative=False)
check("negative=False 只返回 True 征兆", set(ev_pos_only) == {"H2_elevated"})
ev_masked = derive_gas_evidence({"H2": 200, "CH4": None, "C2H2": 60, "C2H4": 10, "C2H6": 10})
check("被遮蔽气体不推导对应征兆", "CH4_elevated" not in ev_masked)
check("总烃需全部气体可见", "TDCG_elevated" not in ev_masked)
check("C2H2 60 → gas_rate_rapid True", ev_masked["gas_rate_rapid"] is True)

# ── 6. 入口函数：遮蔽与负观测 ──
print("6. fault_attribution 入口")
fa._bn_engine = None
r_full = fault_attribution(dga_data={"H2": 345, "CH4": 112, "C2H6": 27, "C2H4": 51, "C2H2": 58})
check("入口输出 uncertainty", "uncertainty" in r_full and "entropy_bits" in r_full["uncertainty"])
check("入口自动派生负观测", {"CH4_elevated", "C2H6_elevated"} <= set(r_full["evidence_negative"]))
r_mask = fault_attribution(dga_data={"H2": 345, "C2H2": 58})
check("遮蔽气体不产生负观测", not ({"CH4_elevated", "C2H4_elevated", "C2H6_elevated", "TDCG_elevated"} & set(r_mask["evidence_negative"])))
check("显式 evidence 优先于派生", fault_attribution(dga_data={"H2": 10}, evidence={"H2_elevated": True})["evidence_used"] == ["H2_elevated"])

# ── 7. learned_params.json 校准字段可加载 ──
print("7. 参数文件")
lp = ROOT / "data/real/dga/learned_params.json"
if lp.exists():
    d = json.loads(lp.read_text(encoding="utf-8"))
    check("含 calibration 字段", "calibration" in d and "T" in d["calibration"])
    check("含 fusion_weights 字段", "fusion_weights" in d and set(d["fusion_weights"]) == set(FAULT_IDS))
    eng_lp = FaultBayesianNetwork(params=d)
    check("引擎加载后 is_calibrated", eng_lp.is_calibrated)
    check("ECE 校准后低于校准前", d["calibration"]["ece_after"] < d["calibration"]["ece_before"],
          f"{d['calibration']['ece_after']} vs {d['calibration']['ece_before']}")
    check("Top-1 不下降超过 1 个点", d["calibration"]["top1_after"] >= d["calibration"]["top1_before"] - 0.01)
else:
    print("  [SKIP] learned_params.json 不存在")

print(f"\n结果：{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
