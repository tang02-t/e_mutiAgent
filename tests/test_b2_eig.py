#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-2 单元测试：EIG 模块。

运行：python3 tests/test_b2_eig.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["FAULT_ATTR_PARAMS"] = str(ROOT / "data/real/dga/learned_params.json")
os.environ.pop("FAULT_ATTR_NEG_EVIDENCE", None)

from src.tools import fault_attribution as fa  # noqa: E402
from src.tools.fault_attribution import SYMPTOM_IDS, FaultBayesianNetwork, fault_attribution  # noqa: E402
from src.tools.eig import (  # noqa: E402
    eig_for_symptom,
    kg_evidence_for,
    load_cost_table,
    predictive_symptom_prob,
    recommend,
    render_recommendations,
    symptom_cost,
    unobserved_symptoms,
)

PASSED = FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


fa._bn_engine = None
engine = FaultBayesianNetwork(params={})  # 专家 CPT：14 个征兆都有区分度
learned = fa.get_engine()

print("1. EIG 基本性质（bayes 基，专家 CPT）")
ev0: dict = {}
post0 = engine.posterior_only(ev0)["bayes_probs"]
h0 = FaultBayesianNetwork.entropy_bits(post0)
eigs = {s: eig_for_symptom(engine, ev0, s, basis="bayes")["eig"] for s in SYMPTOM_IDS}
check("EIG 全部非负（互信息）", all(v >= -1e-9 for v in eigs.values()), str({k: round(v, 4) for k, v in eigs.items() if v < 0}))
check("EIG 不超过当前熵", all(v <= h0 + 1e-9 for v in eigs.values()))
check("至少一个征兆 EIG > 0", max(eigs.values()) > 0.01)
r = eig_for_symptom(engine, ev0, "C2H2_elevated", basis="bayes")
check("expected_h_after = p·H_t + (1-p)·H_f", abs(r["expected_h_after"] - (r["p_true"] * r["h_if_true"] + (1 - r["p_true"]) * r["h_if_false"])) < 1e-9)
check("EIG = H_now - expected_h_after", abs(r["eig"] - (h0 - r["expected_h_after"])) < 1e-9)
# 预测概率手算
p_manual = sum(engine._cpt_lookup(f, "C2H2_elevated").p_true * post0[f] for f in post0)
check("P(s=True|E) 与手算一致", abs(predictive_symptom_prob(engine, post0, "C2H2_elevated") - p_manual) < 1e-12)

print("2. 已观测征兆不再推荐 / 全部观测后无候选")
ev_all = {s: True for s in SYMPTOM_IDS}
check("全部观测后 unobserved 为空", unobserved_symptoms(ev_all) == [])
rec_all = recommend(ev_all, engine=engine, with_kg=False)
check("全部观测后无推荐且动作 conclude", rec_all["recommendations"] == [] and rec_all["suggested_action"] == "conclude")
check("stop_reason 为 no_candidates 或 entropy_below_tau", rec_all["stop_reason"] in ("no_candidates", "entropy_below_tau"))
ev_partial = {"H2_elevated": True, "C2H2_elevated": False}
rec_p = recommend(ev_partial, engine=engine, with_kg=False, top_k=20)
check("已观测征兆不在推荐中", not ({"H2_elevated", "C2H2_elevated"} & {x["symptom"] for x in rec_p["recommendations"]}))
check("None 视为未观测", "CH4_elevated" in unobserved_symptoms({"CH4_elevated": None}))

print("3. 成本单调性")
tbl = load_cost_table()
check("成本表加载且覆盖全部征兆", set(tbl["symptoms"]) >= set(SYMPTOM_IDS))
check("气体派生征兆成本 0", all(symptom_cost(s) == 0 for s in ("H2_elevated", "C2H2_elevated", "TDCG_elevated")))
check("局放告警成本最高", symptom_cost("partial_discharge_alarm") >= max(symptom_cost(s) for s in SYMPTOM_IDS if s != "partial_discharge_alarm"))
rec_l0 = recommend(ev0, engine=engine, lam=0.0, with_kg=False, top_k=20)
rec_l1 = recommend(ev0, engine=engine, lam=0.5, with_kg=False, top_k=20)
voi0 = {x["symptom"]: x["voi"] for x in rec_l0["recommendations"]}
voi1 = {x["symptom"]: x["voi"] for x in rec_l1["recommendations"]}
check("λ=0 时 VoI == EIG", all(abs(x["voi"] - x["eig"]) < 1e-9 for x in rec_l0["recommendations"]))
check("λ 增大时有成本征兆的 VoI 单调下降", all(voi1[s] <= voi0[s] + 1e-9 for s in voi0))
check("λ 增大时零成本征兆 VoI 不变", all(abs(voi1[s] - voi0[s]) < 1e-9 for s in voi0 if symptom_cost(s) == 0))
same_eig = [(s, x["eig"]) for s, x in ((x["symptom"], x) for x in rec_l1["recommendations"])]
check("推荐按 VoI 降序", all(rec_l1["recommendations"][i]["voi"] >= rec_l1["recommendations"][i + 1]["voi"] - 1e-9 for i in range(len(rec_l1["recommendations"]) - 1)))

print("4. 停止准则")
rec_tau = recommend(ev0, engine=engine, stop={"tau_h_bits": 10.0}, with_kg=False)
check("τ_H 很大 → conclude / entropy_below_tau", rec_tau["suggested_action"] == "conclude" and rec_tau["stop_reason"] == "entropy_below_tau")
rec_eps = recommend(ev0, engine=engine, stop={"tau_h_bits": 0.0, "eps_voi": 10.0}, with_kg=False)
check("ε 很大 → conclude / voi_below_eps", rec_eps["stop_reason"] == "voi_below_eps")
rec_k = recommend(ev0, engine=engine, stop={"tau_h_bits": 0.0, "eps_voi": -1.0, "max_rounds": 2}, rounds_done=2, with_kg=False)
check("达到最大轮数 → max_rounds", rec_k["stop_reason"] == "max_rounds")
rec_go = recommend(ev0, engine=engine, stop={"tau_h_bits": 0.0, "eps_voi": -1.0, "max_rounds": 9}, with_kg=False)
check("否则给出 ask / call_tool", rec_go["suggested_action"] in ("ask", "call_tool") and rec_go["stop_reason"] is None)
# call_tool 映射
rec_tool = recommend(ev0, engine=engine, candidates=["oil_temp_elevated"], stop={"tau_h_bits": 0.0, "eps_voi": -1.0}, with_kg=False)
check("oil_temp → call_tool:ett_forecast", rec_tool["suggested_action"] == "call_tool" and rec_tool["suggested_tool"] == "ett_forecast")

print("5. 图谱联动与渲染")
kge = kg_evidence_for("C2H2_elevated")
check("乙炔征兆能取到图谱原文", len(kge) >= 1 and kge[0]["sentence"])
check("无 kg_nodes 的征兆返回空", kg_evidence_for("load_elevated") == [])
txt = render_recommendations(rec_go)
check("渲染含建议动作", "建议动作" in txt and "后验熵" in txt)

print("6. 入口集成（学习参数）")
r_full = fault_attribution(dga_data={"H2": 345, "CH4": 112, "C2H6": 27, "C2H4": 51, "C2H2": 58})
u = r_full["uncertainty"]
check("uncertainty 含 recommendations / suggested_action", "recommendations" in u and u["suggested_action"] in ("ask", "call_tool", "conclude"))
check("气体全给出后不再推荐气体征兆", not any(x["symptom"] in fa.GAS_DERIVED_SYMPTOMS for x in u["recommendations"]))
check("报告尾部含不确定性段", "不确定性与下一步建议" in r_full["report"])
r_off = fault_attribution(dga_data={"H2": 345, "C2H2": 58}, with_eig=False)
check("with_eig=False 不输出推荐", "recommendations" not in r_off["uncertainty"])
r_mask = fault_attribution(dga_data={"H2": 345, "C2H2": 58})
recs_mask = r_mask["uncertainty"]["recommendations"]
check("遮蔽气体时零成本气体征兆进入候选", any(x["symptom"] in fa.GAS_DERIVED_SYMPTOMS for x in recs_mask), str([x["symptom"] for x in recs_mask]))
check("遮蔽气体时不做三比值规则匹配", r_mask["dga_analysis"]["ratios_incomplete"] is True and r_mask["dga_analysis"]["matched_rules"] == [])
from src.tools.eig import recommend as _rec  # noqa: E402
ev_mask = {s: True for s in r_mask["evidence_used"]}
ev_mask.update({s: False for s in r_mask["evidence_negative"]})
ctx_mask = {"H2": 345, "C2H2": 58, "CH4": None, "C2H4": None, "C2H6": None}
rec_costly = _rec(ev_mask, ctx_mask, engine=learned, lam=0.5, with_kg=False)
check("λ 增大后零成本气体征兆升至首位", rec_costly["recommendations"][0]["symptom"] in fa.GAS_DERIVED_SYMPTOMS, str(rec_costly["recommendations"][0]))
# 学习参数下 bayes 基 EIG 仍非负
post_l = learned.posterior_only({})["bayes_probs"]
check("学习参数 bayes 基 EIG 非负", all(eig_for_symptom(learned, {}, s, basis="bayes")["eig"] >= -1e-9 for s in SYMPTOM_IDS))

print(f"\n结果：{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
