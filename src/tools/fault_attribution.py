"""
贝叶斯网络故障归因模块。

基于概率图模型建立变压器故障归因推理能力：
- 故障层节点集（Fault Nodes）
- 征兆层节点集（Symptom Nodes）
- 故障与征兆间的因果关系及条件概率（CPT）
- 给定观测征兆时，计算各故障的后验概率

参考：DL/T 722-2014《变压器油中溶解气体分析和判断导则》
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ──────────────────────────────────────────────────────────────
# 1. 节点定义
# ──────────────────────────────────────────────────────────────

# 故障类型枚举（故障层节点）
FAULT_IDS = [
    "winding_deformation",      # 绕组变形/位移
    "winding_short_circuit",    # 绕组匝间短路
    "core_grounding",           # 铁芯多点接地
    "bushing_fault",            # 套管故障
    "olc_fault",                # 分接开关故障
    "partial_discharge",        # 局部放电
    "overload_overheating",     # 过载过热
    "insulation_degradation",   # 绝缘老化
]

# 征兆类型枚举（征兆层节点）
SYMPTOM_IDS = [
    "H2_elevated",      # 氢气升高
    "CH4_elevated",     # 甲烷升高
    "C2H2_elevated",    # 乙炔升高
    "C2H4_elevated",    # 乙烯升高
    "C2H6_elevated",    # 乙烷升高
    "CO_elevated",      # 一氧化碳升高
    "CO2_elevated",     # 二氧化碳升高
    "oil_temp_elevated",    # 油温升高
    "winding_temp_elevated", # 绕组温度升高
    "load_elevated",        # 负载电流升高
    "vibration_elevated",   # 振动异常
    "partial_discharge_alarm",  # 局部放电告警
    "gas_rate_rapid",       # 气体产气速率过快
    "TDCG_elevated",        # 总烃升高
]

# ──────────────────────────────────────────────────────────────
# 2. CPT（条件概率表）定义
#    P(症状 | 故障)  —  每个故障发生时各症状的条件概率
#    使用近似离散概率（高/中/低 影响度）
# ──────────────────────────────────────────────────────────────

@dataclass
class CPTRow:
    """CPT 中的一行，表示某故障发生时该症状的（True/False）概率。"""
    p_true: float   # P(symptom=True | fault=True)
    p_false: float  # P(symptom=True | fault=False)


# CPT：P(symptom | fault)  — 故障发生时，症状出现的概率
# 结构：fault_id -> {symptom_id: CPTRow}
_FAULT_SYMPTOM_CPT: dict[str, dict[str, CPTRow]] = {
    # ── 绕组变形 ──
    "winding_deformation": {
        "H2_elevated":          CPTRow(p_true=0.45, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.30, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.15, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.25, p_false=0.07),
        "C2H6_elevated":         CPTRow(p_true=0.20, p_false=0.06),
        "CO_elevated":           CPTRow(p_true=0.35, p_false=0.10),
        "CO2_elevated":          CPTRow(p_true=0.40, p_false=0.15),
        "oil_temp_elevated":     CPTRow(p_true=0.50, p_false=0.20),
        "winding_temp_elevated": CPTRow(p_true=0.55, p_false=0.18),
        "vibration_elevated":    CPTRow(p_true=0.70, p_false=0.10),
        "TDCG_elevated":         CPTRow(p_true=0.45, p_false=0.12),
        # 未列出的默认为 p_true=0.05, p_false=0.01
    },

    # ── 绕组匝间短路 ──（高能放电，油温高，C2H2显著）
    "winding_short_circuit": {
        "H2_elevated":          CPTRow(p_true=0.85, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.75, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.90, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.80, p_false=0.07),
        "C2H6_elevated":         CPTRow(p_true=0.60, p_false=0.06),
        "CO_elevated":           CPTRow(p_true=0.70, p_false=0.10),
        "CO2_elevated":          CPTRow(p_true=0.65, p_false=0.15),
        "oil_temp_elevated":     CPTRow(p_true=0.85, p_false=0.20),
        "winding_temp_elevated": CPTRow(p_true=0.90, p_false=0.18),
        "load_elevated":         CPTRow(p_true=0.40, p_false=0.25),
        "TDCG_elevated":         CPTRow(p_true=0.88, p_false=0.12),
        "gas_rate_rapid":        CPTRow(p_true=0.80, p_false=0.05),
    },

    # ── 铁芯多点接地 ──（低温过热为主
    "core_grounding": {
        "H2_elevated":          CPTRow(p_true=0.55, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.70, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.10, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.65, p_false=0.07),
        "C2H6_elevated":         CPTRow(p_true=0.50, p_false=0.06),
        "CO_elevated":           CPTRow(p_true=0.30, p_false=0.10),
        "CO2_elevated":          CPTRow(p_true=0.40, p_false=0.15),
        "oil_temp_elevated":     CPTRow(p_true=0.60, p_false=0.20),
        "TDCG_elevated":         CPTRow(p_true=0.65, p_false=0.12),
    },

    # ── 套管故障 ──（放电性，C2H2和H2显著
    "bushing_fault": {
        "H2_elevated":          CPTRow(p_true=0.80, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.45, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.75, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.50, p_false=0.07),
        "C2H6_elevated":         CPTRow(p_true=0.30, p_false=0.06),
        "oil_temp_elevated":     CPTRow(p_true=0.30, p_false=0.20),
        "partial_discharge_alarm": CPTRow(p_true=0.65, p_false=0.05),
        "TDCG_elevated":         CPTRow(p_true=0.60, p_false=0.12),
    },

    # ── 分接开关故障 ──（接触不良打火，高C2H2
    "olc_fault": {
        "H2_elevated":          CPTRow(p_true=0.65, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.55, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.85, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.60, p_false=0.07),
        "C2H6_elevated":         CPTRow(p_true=0.40, p_false=0.06),
        "CO_elevated":           CPTRow(p_true=0.25, p_false=0.10),
        "oil_temp_elevated":     CPTRow(p_true=0.45, p_false=0.20),
        "load_elevated":         CPTRow(p_true=0.35, p_false=0.25),
        "TDCG_elevated":         CPTRow(p_true=0.70, p_false=0.12),
    },

    # ── 局部放电 ──（H2为主，CH4次之，C2H2低或不出现
    "partial_discharge": {
        "H2_elevated":          CPTRow(p_true=0.90, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.50, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.10, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.20, p_false=0.07),
        "CO_elevated":           CPTRow(p_true=0.35, p_false=0.10),
        "partial_discharge_alarm": CPTRow(p_true=0.85, p_false=0.05),
        "gas_rate_rapid":        CPTRow(p_true=0.30, p_false=0.05),
    },

    # ── 过载过热 ──（CO/CO2显著，总烃高，温度高
    "overload_overheating": {
        "H2_elevated":          CPTRow(p_true=0.40, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.60, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.15, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.75, p_false=0.07),
        "C2H6_elevated":         CPTRow(p_true=0.55, p_false=0.06),
        "CO_elevated":           CPTRow(p_true=0.85, p_false=0.10),
        "CO2_elevated":          CPTRow(p_true=0.90, p_false=0.15),
        "oil_temp_elevated":     CPTRow(p_true=0.95, p_false=0.20),
        "winding_temp_elevated": CPTRow(p_true=0.92, p_false=0.18),
        "load_elevated":         CPTRow(p_true=0.80, p_false=0.25),
        "TDCG_elevated":         CPTRow(p_true=0.80, p_false=0.12),
    },

    # ── 绝缘老化 ──（CO/CO2逐步升高，温度略高
    "insulation_degradation": {
        "H2_elevated":          CPTRow(p_true=0.30, p_false=0.10),
        "CH4_elevated":          CPTRow(p_true=0.35, p_false=0.08),
        "C2H2_elevated":         CPTRow(p_true=0.05, p_false=0.02),
        "C2H4_elevated":         CPTRow(p_true=0.40, p_false=0.07),
        "CO_elevated":           CPTRow(p_true=0.80, p_false=0.10),
        "CO2_elevated":          CPTRow(p_true=0.85, p_false=0.15),
        "oil_temp_elevated":     CPTRow(p_true=0.40, p_false=0.20),
        "gas_rate_rapid":        CPTRow(p_true=0.25, p_false=0.05),
        "TDCG_elevated":         CPTRow(p_true=0.55, p_false=0.12),
    },
}

# ──────────────────────────────────────────────────────────────
# 3. 故障先验概率（无任何观测时的基础概率）
# ──────────────────────────────────────────────────────────────

_PRIOR_PROBS: dict[str, float] = {
    "winding_deformation":   0.08,
    "winding_short_circuit": 0.04,
    "core_grounding":       0.06,
    "bushing_fault":        0.07,
    "olc_fault":            0.10,
    "partial_discharge":    0.12,
    "overload_overheating": 0.15,
    "insulation_degradation": 0.18,
}

# ──────────────────────────────────────────────────────────────
# 4. 症状→故障映射（DGA 特征快速判断，辅助初始化）
# ──────────────────────────────────────────────────────────────

# 严格依据 DL/T 722-2014 官方表格的三比值法编码规则
# 比值编码三元组顺序固定为：
#   [C2H2/C2H4, CH4/H2, C2H4/C2H6]
#
# 表1 编码规则：
#                 C2H2/C2H4   CH4/H2   C2H4/C2H6
# - <0.1            0           1          0
# - [0.1, 1)        1           0          0
# - [1, 3)          1           2          1
# - >=3             2           2          2
#
# 表2 故障类型判断方法（编码组合 -> 故障类型）：
#   [0, 0, 1]            低温过热（低于150℃）
#   [0, 2, 0]            低温过热（150~300℃）
#   [0, 2, 1]            中温过热（300~700℃）
#   [0, {0,1,2}, 2]      高温过热（高于700℃）
#   [0, 1, 0]            局部放电
#   [1, {0,1}, {0,1,2}]  低能放电
#   [1, 2, {0,1,2}]      低能放电兼过热
#   [2, {0,1}, {0,1,2}]  电弧放电
#   [2, 2, {0,1,2}]      电弧放电兼过热
_DGA_RULES: list[dict[str, Any]] = [
    # ── C2H2/C2H4 = 0（过热类与局部放电）──
    {
        # 官方表2: [0, 0, 1]
        "name": "低温过热（低于150℃）",
        "conditions": {"code_C2H2_C2H4": [0], "code_CH4_H2": [0], "code_C2H4_C2H6": [1]},
        "related_faults": ["insulation_degradation"],
        "weight": 0.72,
    },
    {
        # 官方表2: [0, 2, 0]
        "name": "低温过热（150℃~300℃）",
        "conditions": {"code_C2H2_C2H4": [0], "code_CH4_H2": [2], "code_C2H4_C2H6": [0]},
        "related_faults": ["core_grounding", "insulation_degradation"],
        "weight": 0.76,
    },
    {
        # 官方表2: [0, 2, 1]
        "name": "中温过热（300℃~700℃）",
        "conditions": {"code_C2H2_C2H4": [0], "code_CH4_H2": [2], "code_C2H4_C2H6": [1]},
        "related_faults": ["core_grounding", "winding_deformation"],
        "weight": 0.82,
    },
    {
        # 官方表2: [0, {0,1,2}, 2]
        "name": "高温过热（高于700℃）",
        "conditions": {"code_C2H2_C2H4": [0], "code_CH4_H2": [0, 1, 2], "code_C2H4_C2H6": [2]},
        "related_faults": ["overload_overheating", "core_grounding"],
        "weight": 0.86,
    },
    {
        # 官方表2: [0, 1, 0]
        "name": "局部放电",
        "conditions": {"code_C2H2_C2H4": [0], "code_CH4_H2": [1], "code_C2H4_C2H6": [0]},
        "related_faults": ["partial_discharge"],
        "weight": 0.84,
    },
    # ── C2H2/C2H4 = 1（低能放电类）──
    {
        # 官方表2: [1, {0,1}, {0,1,2}]
        "name": "低能放电",
        "conditions": {"code_C2H2_C2H4": [1], "code_CH4_H2": [0, 1], "code_C2H4_C2H6": [0, 1, 2]},
        "related_faults": ["partial_discharge", "bushing_fault"],
        "weight": 0.9,
    },
    {
        # 官方表2: [1, 2, {0,1,2}]
        "name": "低能放电兼过热",
        "conditions": {"code_C2H2_C2H4": [1], "code_CH4_H2": [2], "code_C2H4_C2H6": [0, 1, 2]},
        "related_faults": ["partial_discharge", "overload_overheating"],
        "weight": 0.88,
    },
    # ── C2H2/C2H4 = 2（电弧放电类，高能）──
    {
        # 官方表2: [2, {0,1}, {0,1,2}]
        "name": "电弧放电",
        "conditions": {"code_C2H2_C2H4": [2], "code_CH4_H2": [0, 1], "code_C2H4_C2H6": [0, 1, 2]},
        "related_faults": ["winding_short_circuit", "bushing_fault", "olc_fault"],
        "weight": 0.93,
    },
    {
        # 官方表2: [2, 2, {0,1,2}]
        "name": "电弧放电兼过热",
        "conditions": {"code_C2H2_C2H4": [2], "code_CH4_H2": [2], "code_C2H4_C2H6": [0, 1, 2]},
        "related_faults": ["winding_short_circuit", "olc_fault", "overload_overheating"],
        "weight": 0.92,
    },
]


# ──────────────────────────────────────────────────────────────
# 5. 贝叶斯网络推理引擎
# ──────────────────────────────────────────────────────────────

class FaultBayesianNetwork:
    """
    简化的贝叶斯网络故障归因引擎。

    网络结构（有向无环图，朴素独立近似）：
      故障层（F1...Fn） → 征兆层（S1...Sn）

    给定观测到的征兆集合 evidence = {symptom_id: True}，
    计算各故障的后验概率 P(Fi | evidence)。
    """

    def __init__(self) -> None:
        self.faults = list(_PRIOR_PROBS.keys())
        self.symptoms = list(SYMPTOM_IDS)
        self.prior: dict[str, float] = dict(_PRIOR_PROBS)
        self.cpt: dict[str, dict[str, CPTRow]] = dict(_FAULT_SYMPTOM_CPT)

        # 若指定了学习参数文件（环境变量 FAULT_ATTR_PARAMS），用数据驱动的
        # 先验/CPT 覆盖专家默认值；文件不存在则保持专家值（向后兼容）。
        self.params_source = "expert_default"
        self._maybe_load_learned_params()

        # 症状默认概率（无条件）
        self._symptom_base_prob: dict[str, float] = {}
        for s in self.symptoms:
            # 用全故障的加权平均作为先验 P(S)
            total = 0.0
            for f, f_prob in self.prior.items():
                cpt = self.cpt.get(f, {})
                row = cpt.get(s)
                if row:
                    total += f_prob * row.p_true
            self._symptom_base_prob[s] = max(total, 0.01)

    def _maybe_load_learned_params(self) -> None:
        """从 FAULT_ATTR_PARAMS 指定的 JSON 加载数据学习的先验与 CPT。"""
        import json
        import os

        path = os.environ.get("FAULT_ATTR_PARAMS")
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        learned_prior = data.get("prior") or {}
        learned_cpt = data.get("cpt") or {}

        if learned_prior:
            for fid, p in learned_prior.items():
                if fid in self.prior:
                    self.prior[fid] = float(p)

        if learned_cpt:
            for fid, rows in learned_cpt.items():
                row_map: dict[str, CPTRow] = {}
                for sid, vals in rows.items():
                    row_map[sid] = CPTRow(
                        p_true=float(vals.get("p_true", 0.05)),
                        p_false=float(vals.get("p_false", 0.01)),
                    )
                self.cpt[fid] = row_map

        self.params_source = path

    def _cpt_lookup(self, fault: str, symptom: str) -> CPTRow:
        """查询 CPT，若不存在返回极低概率。"""
        row = self.cpt.get(fault, {}).get(symptom)
        if row:
            return row
        return CPTRow(p_true=0.05, p_false=0.01)

    def _normalize(self, probs: dict[str, float]) -> dict[str, float]:
        """归一化概率分布。"""
        total = sum(probs.values())
        if total <= 0:
            return probs
        return {k: v / total for k, v in probs.items()}

    def infer(
        self,
        evidence: dict[str, bool],
        symptom_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        给定观测征兆，执行贝叶斯推理。

        参数:
            evidence: 征兆观测字典，如 {"H2_elevated": True, "C2H2_elevated": True}
            symptom_context: 额外上下文（DGA数值、气体比例等），用于DGA规则匹配

        返回:
            归因分析结果字典
        """
        symptom_context = symptom_context or {}

        # ── Step 1: 先用 DGA 规则快速识别特征 ──
        dga_analysis = self._analyze_dga(symptom_context)

        # ── Step 2: 贝叶斯推理计算后验概率 ──
        posterior = self._compute_posterior(evidence)

        # ── Step 3: 融合规则与贝叶斯结果 ──
        fused = self._fuse_results(posterior, dga_analysis)

        # ── Step 4: 排序并生成诊断报告 ──
        ranked = sorted(fused, key=lambda x: x["probability"], reverse=True)

        for i, item in enumerate(ranked):
            item["rank"] = i + 1
            item["probability_pct"] = f"{item['probability'] * 100:.1f}%"

        report = self._build_report(ranked, dga_analysis, evidence)

        return {
            "status": "ok",
            "primary_fault": ranked[0]["fault_id"] if ranked else None,
            "primary_fault_name": ranked[0]["fault_name"] if ranked else None,
            "primary_probability": ranked[0]["probability"] if ranked else 0.0,
            "fault_ranking": ranked,
            "dga_analysis": dga_analysis,
            "evidence_used": [k for k, v in evidence.items() if v],
            "report": report,
        }

    def _compute_posterior(self, evidence: dict[str, bool]) -> dict[str, float]:
        """朴素贝叶斯推理：P(F|E) ∝ P(E|F) * P(F)"""
        posterior: dict[str, float] = {}

        for fault in self.faults:
            f_prob = self.prior.get(fault, 0.01)

            # P(E|F) = ∏ P(ei | F)，若某症状未观测则跳过
            likelihood = 1.0
            for symptom, observed in evidence.items():
                if not observed:
                    continue
                row = self._cpt_lookup(fault, symptom)
                # P(S=True | F)
                p_s_given_f = row.p_true
                likelihood *= p_s_given_f

            posterior[fault] = likelihood * f_prob

        return self._normalize(posterior)

    def _code_c2h2_c2h4(self, ratio: float) -> int:
        """表6编码：C2H2/C2H4"""
        if ratio < 0.1:
            return 0
        if ratio < 3:
            return 1
        return 2

    def _code_ch4_h2(self, ratio: float) -> int:
        """表6编码：CH4/H2"""
        if ratio < 0.1:
            return 1
        if ratio < 1:
            return 0
        return 2

    def _code_c2h4_c2h6(self, ratio: float) -> int:
        """表6编码：C2H4/C2H6"""
        if ratio < 1:
            return 0
        if ratio < 3:
            return 1
        return 2

    def _analyze_dga(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        基于 DL/T 722-2014 三比值法分析 DGA 数据。
        ctx 支持字段：
            H2, CH4, C2H2, C2H4, C2H6 (单位: μL/L)
            或 ratios: {C2H2_C2H4, CH4_H2, C2H4_C2H6}
        """
        gases = {k: ctx.get(k, 0) for k in ["H2", "CH4", "C2H2", "C2H4", "C2H6"]}
        ratios = ctx.get("ratios", {})

        # 计算气体比例（若未直接提供则从浓度计算）
        total = sum(gases.values()) or 1
        calc_ratios = {
            "C2H2_TDCG": gases["C2H2"] / total,
            "C2H2_C2H4": gases["C2H2"] / (gases["C2H4"] or 1),
            "CH4_H2": gases["CH4"] / (gases["H2"] or 1),
            "C2H4_C2H6": gases["C2H4"] / (gases["C2H6"] or 1),
        }
        calc_ratios.update(ratios)

        # 三比值法编码（表6）
        ratio_codes = {
            "code_C2H2_C2H4": self._code_c2h2_c2h4(calc_ratios.get("C2H2_C2H4", 0)),
            "code_CH4_H2": self._code_ch4_h2(calc_ratios.get("CH4_H2", 0)),
            "code_C2H4_C2H6": self._code_c2h4_c2h6(calc_ratios.get("C2H4_C2H6", 0)),
        }

        # 应用 DGA 规则
        matched_rules: list[dict[str, Any]] = []
        for rule in _DGA_RULES:
            if self._match_rule(rule, ratio_codes):
                matched_rules.append({
                    "rule_name": rule["name"],
                    "related_faults": rule["related_faults"],
                    "confidence": rule["weight"],
                })

        return {
            "gases": gases,
            "ratios": calc_ratios,
            "ratio_codes": ratio_codes,
            "matched_rules": matched_rules,
            "interpretation": self._interpret_gases(gases, calc_ratios),
        }

    def _match_rule(self, rule: dict[str, Any], ratio_codes: dict[str, int]) -> bool:
        conds = rule["conditions"]
        for key in ("code_C2H2_C2H4", "code_CH4_H2", "code_C2H4_C2H6"):
            allowed = conds.get(key)
            if allowed is None:
                continue
            if ratio_codes.get(key) not in allowed:
                return False
        return True

    def _interpret_gases(self, gases: dict[str, float], ratios: dict[str, float]) -> str:
        lines = []
        H2 = gases.get("H2", 0)
        C2H2 = gases.get("C2H2", 0)
        C2H4 = gases.get("C2H4", 0)
        C2H6 = gases.get("C2H6", 0)
        total = sum(gases.values()) or 1

        if H2 > 150:
            lines.append(f"- H₂={H2:.0f}μL/L（超过注意值150μL/L），可能存在放电或低温过热")
        if C2H2 > 5:
            lines.append(f"- C₂H₂={C2H2:.0f}μL/L（超过注意值5μL/L），存在放电性故障风险")
        if C2H2 / total > 0.5:
            lines.append("- C₂H₂占总烃比例>50%，高能放电可能性高")
        elif C2H2 > 0 and C2H2 / total < 0.1:
            lines.append("- C₂H₂占总烃比例<10%，低能放电或局部放电可能性较高")
        if C2H4 / (C2H6 or 1) >= 3:
            lines.append("- C₂H₄/C₂H₆≥3，高温过热特征明显")

        return "\n".join(lines) if lines else "- 气体含量暂无超标，但需持续监测"

    def _fuse_results(
        self,
        posterior: dict[str, float],
        dga: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """融合贝叶斯后验概率和 DGA 规则匹配结果。"""
        FAULT_NAMES = {
            "winding_deformation": "绕组变形/位移",
            "winding_short_circuit": "绕组匝间短路",
            "core_grounding": "铁芯多点接地",
            "bushing_fault": "套管故障",
            "olc_fault": "分接开关接触不良",
            "partial_discharge": "局部放电",
            "overload_overheating": "过载过热",
            "insulation_degradation": "绝缘老化",
        }

        FAULT_DESCRIPTIONS = {
            "winding_deformation": "变压器绕组在短路电动力或运输冲撞下发生变形、扭曲，可能影响电气性能",
            "winding_short_circuit": "绕组匝间绝缘击穿形成短路，故障能量大，发展迅速，需立即停电处理",
            "core_grounding": "铁芯与地之间形成多点连接，产生环流导致局部过热",
            "bushing_fault": "套管内部放电或绝缘击穿，可能发展为对地短路",
            "olc_fault": "分接开关触头接触不良，拉弧产生大量C₂H₂",
            "partial_discharge": "绝缘内部或表面发生局部放电，长期发展会导致绝缘恶化",
            "overload_overheating": "负载超过额定值导致绕组和油温升高，加速绝缘老化",
            "insulation_degradation": "绝缘油/纸/板在热和电场长期作用下老化，机械和电气性能下降",
        }

        results: list[dict[str, Any]] = []
        for fid in self.faults:
            bayes_prob = posterior.get(fid, 0.0)

            # DGA 规则加权
            rule_boost = 0.0
            for rule in dga.get("matched_rules", []):
                if fid in rule["related_faults"]:
                    rule_boost = max(rule_boost, rule["confidence"])

            # 融合：0.7 * 贝叶斯 + 0.3 * 规则（若无规则则全用贝叶斯）
            if rule_boost > 0:
                fused_prob = 0.7 * bayes_prob + 0.3 * rule_boost
            else:
                fused_prob = bayes_prob

            results.append({
                "fault_id": fid,
                "fault_name": FAULT_NAMES.get(fid, fid),
                "fault_description": FAULT_DESCRIPTIONS.get(fid, ""),
                "bayes_probability": bayes_prob,
                "dga_boost": rule_boost,
                "probability": fused_prob,
                "severity": self._severity_level(fused_prob),
            })

        # 归一化融合后的概率
        total = sum(r["probability"] for r in results) or 1
        for r in results:
            r["probability"] /= total

        return results

    def _severity_level(self, prob: float) -> str:
        if prob >= 0.5:
            return "高风险"
        elif prob >= 0.2:
            return "中等风险"
        else:
            return "低风险"

    def _build_report(
        self,
        ranked: list[dict[str, Any]],
        dga: dict[str, Any],
        evidence: dict[str, bool],
    ) -> str:
        lines = ["## 贝叶斯网络故障归因分析报告\n"]

        # 观测征兆
        observed = [k for k, v in evidence.items() if v]
        if observed:
            lines.append(f"**输入征兆**：`{'`, `'.join(observed)}`\n")

        # DGA 解释
        interp = dga.get("interpretation", "")
        if interp:
            lines.append(f"**DGA 特征解释**\n{interp}\n")

        matched = dga.get("matched_rules", [])
        if matched:
            lines.append("**三比值法匹配**")
            for r in matched:
                lines.append(f"- [{r['confidence']:.0%}置信] {r['rule_name']} → 可能故障：{'/'.join([self._fault_name(n) for n in r['related_faults']])}")
            lines.append("")

        # 故障排序
        lines.append("**故障概率排序**（贝叶斯后验 × DGA规则融合）")
        for item in ranked[:5]:
            bar_len = int(item["probability"] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(
                f"  {item['rank']}. [{item['severity']}] {item['fault_name']}"
                f"  {bar}  {item['probability_pct']}"
            )
            if item["fault_description"]:
                lines.append(f"     └─ {item['fault_description'][:60]}")

        return "\n".join(lines)

    def _fault_name(self, fid: str) -> str:
        names = {
            "winding_deformation": "绕组变形", "winding_short_circuit": "匝间短路",
            "core_grounding": "铁芯接地", "bushing_fault": "套管故障",
            "olc_fault": "分接开关", "partial_discharge": "局部放电",
            "overload_overheating": "过载过热", "insulation_degradation": "绝缘老化",
        }
        return names.get(fid, fid)


# ──────────────────────────────────────────────────────────────
# 6. 暴露给 MCP 的入口函数
# ──────────────────────────────────────────────────────────────

_bn_engine: Optional[FaultBayesianNetwork] = None


def get_engine() -> FaultBayesianNetwork:
    global _bn_engine
    if _bn_engine is None:
        _bn_engine = FaultBayesianNetwork()
    return _bn_engine


def fault_attribution(
    dga_data: dict[str, Any] | None = None,
    evidence: dict[str, bool] | None = None,
    query: str = "",
) -> dict[str, Any]:
    """
    故障归因主入口函数，供 MCP 工具调用。

    参数:
        dga_data: DGA 气体数据字典，字段包括 H2/CH4/C2H2/C2H4/C2H6（单位μL/L）
                  以及可选的 ratios 子字典
        evidence: 征兆布尔字典，如 {"H2_elevated": True, "C2H2_elevated": True}
        query: 用户原始问题描述（用于上下文）

    示例:
        fault_attribution(
            dga_data={"H2": 680, "C2H2": 45, "C2H4": 230, "CH4": 150},
            evidence={"H2_elevated": True, "C2H2_elevated": True},
        )
    """
    import re

    engine = get_engine()
    computed_evidence: dict[str, bool] = dict(evidence) if evidence else {}

    def _num(v: Any, default: float = 0.0) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if not isinstance(v, str):
            return default
        m = re.search(r"[-+]?\d*\.?\d+", v.replace(",", ""))
        return float(m.group()) if m else default

    gases: dict[str, float] = {"H2": 0, "CH4": 0, "C2H2": 0, "C2H4": 0, "C2H6": 0}

    if dga_data:
        gases = {
            "H2": _num(dga_data.get("H2")),
            "CH4": _num(dga_data.get("CH4")),
            "C2H2": _num(dga_data.get("C2H2")),
            "C2H4": _num(dga_data.get("C2H4")),
            "C2H6": _num(dga_data.get("C2H6")),
        }

    # 根据注意值（DL/T 722）自动推断征兆
    if gases["H2"] > 150:
        computed_evidence.setdefault("H2_elevated", True)
    if gases["CH4"] > 120:
        computed_evidence.setdefault("CH4_elevated", True)
    if gases["C2H2"] > 5:
        computed_evidence.setdefault("C2H2_elevated", True)
    if gases["C2H4"] > 50:
        computed_evidence.setdefault("C2H4_elevated", True)
    if gases["C2H6"] > 65:
        computed_evidence.setdefault("C2H6_elevated", True)
    if gases["C2H2"] > 50:
        computed_evidence.setdefault("gas_rate_rapid", True)

    if dga_data:
        result = engine.infer(
            computed_evidence,
            symptom_context={
                "H2": gases["H2"],
                "CH4": gases["CH4"],
                "C2H2": gases["C2H2"],
                "C2H4": gases["C2H4"],
                "C2H6": gases["C2H6"],
                "query": query,
            },
        )
    else:
        result = engine.infer(computed_evidence)

    return result
