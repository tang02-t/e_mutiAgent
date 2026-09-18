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
from pathlib import Path
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

    #: 规则 / 贝叶斯融合的默认权重（贝叶斯占比），v1 固定值
    DEFAULT_FUSION_WEIGHT = 0.7

    def __init__(self, params: dict[str, Any] | None = None, *, use_negative_evidence: bool | None = None) -> None:
        import os

        self.faults = list(_PRIOR_PROBS.keys())
        self.symptoms = list(SYMPTOM_IDS)
        self.prior: dict[str, float] = dict(_PRIOR_PROBS)
        self.cpt: dict[str, dict[str, CPTRow]] = dict(_FAULT_SYMPTOM_CPT)

        # B-1 校准字段：按故障类别的融合权重（None → 全部用 DEFAULT_FUSION_WEIGHT）、
        # 温度缩放参数（1.0 → 不缩放）、是否把 evidence 中的 False 作为负观测参与推理
        # （默认开；环境变量 FAULT_ATTR_NEG_EVIDENCE=0 可关闭，用于消融）。
        self.fusion_weights: dict[str, float] | None = None
        self.temperature: float = 1.0
        self.calibration_meta: dict[str, Any] = {}
        if use_negative_evidence is None:
            use_negative_evidence = os.environ.get("FAULT_ATTR_NEG_EVIDENCE", "1") not in ("0", "false", "False")
        self.use_negative_evidence = use_negative_evidence

        # 参数来源优先级：显式传入 params > 环境变量 FAULT_ATTR_PARAMS 指向的文件 > 专家默认值。
        self.params_source = "expert_default"
        if params is not None:
            self._apply_params(params, source="injected")
        else:
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
        self._apply_params(data, source=path)

    def _apply_params(self, data: dict[str, Any], *, source: str) -> None:
        """
        用参数字典覆盖专家默认值。支持字段：
          prior / cpt                  —— learn_cpt.py 学到的先验与 CPT
          fusion_weights: {fault: w}   —— 按类别的贝叶斯占比（B-1）
          calibration: {"method": "temperature", "T": float, ...}（B-1）
        """
        learned_prior = data.get("prior") or {}
        learned_cpt = data.get("cpt") or {}

        if learned_prior:
            for fid, p in learned_prior.items():
                if fid in self.prior:
                    self.prior[fid] = float(p)

        if learned_cpt:
            for fid, rows in learned_cpt.items():
                # 按征兆合并：学习到的征兆覆盖专家值，未学习的征兆保留专家 CPT
                row_map: dict[str, CPTRow] = dict(self.cpt.get(fid, {}))
                for sid, vals in rows.items():
                    row_map[sid] = CPTRow(
                        p_true=float(vals.get("p_true", 0.05)),
                        p_false=float(vals.get("p_false", 0.01)),
                    )
                self.cpt[fid] = row_map

        fw = data.get("fusion_weights")
        if isinstance(fw, dict) and fw:
            self.fusion_weights = {k: min(1.0, max(0.0, float(v))) for k, v in fw.items()}

        cal = data.get("calibration") or {}
        if isinstance(cal, dict) and cal:
            self.calibration_meta = dict(cal)
            t = cal.get("T")
            if t is not None and float(t) > 0:
                self.temperature = float(t)

        self.params_source = source

    # ── 校准相关公开属性 ──
    @property
    def is_calibrated(self) -> bool:
        return abs(self.temperature - 1.0) > 1e-9 or self.fusion_weights is not None

    def fusion_weight(self, fault: str) -> float:
        if self.fusion_weights and fault in self.fusion_weights:
            return self.fusion_weights[fault]
        return self.DEFAULT_FUSION_WEIGHT

    def _apply_temperature(self, probs: dict[str, float]) -> dict[str, float]:
        """温度缩放：p_i^(1/T) 后归一化。T>1 变平（降低过度自信），T<1 变尖。"""
        T = self.temperature
        if abs(T - 1.0) < 1e-9:
            return dict(probs)
        scaled = {k: (max(v, 1e-12) ** (1.0 / T)) for k, v in probs.items()}
        return self._normalize(scaled)

    @staticmethod
    def entropy_bits(probs: dict[str, float]) -> float:
        """离散分布的香农熵（bit）。"""
        h = 0.0
        for p in probs.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def posterior_only(
        self,
        evidence: dict[str, bool],
        symptom_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        只返回（融合 + 校准后的）故障分布及其不确定性度量，不生成报告。
        供 EIG 模块反复调用（假设某征兆取值后重新计算分布）。

        返回:
            {
              "probs": {fault_id: p},           # 归一化后验（已融合规则、已温度缩放）
              "bayes_probs": {fault_id: p},     # 纯贝叶斯后验（未融合、未缩放）
              "entropy_bits": H(F|E),
              "top1": fault_id, "top1_prob": p1,
              "top1_top2_gap": p1 - p2,
              "calibrated": bool,
            }
        """
        symptom_context = symptom_context or {}
        dga_analysis = self._analyze_dga(symptom_context) if symptom_context else {"matched_rules": []}
        bayes = self._compute_posterior(evidence)
        fused = self._fuse_results(bayes, dga_analysis)
        probs = self._apply_temperature({r["fault_id"]: r["probability"] for r in fused})
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        p1 = ranked[0][1] if ranked else 0.0
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        return {
            "probs": probs,
            "bayes_probs": bayes,
            "entropy_bits": self.entropy_bits(probs),
            "top1": ranked[0][0] if ranked else None,
            "top1_prob": p1,
            "top1_top2_gap": p1 - p2,
            "calibrated": self.is_calibrated,
        }

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

        # ── Step 3: 融合规则与贝叶斯结果，并做温度缩放（校准） ──
        fused = self._fuse_results(posterior, dga_analysis)
        scaled = self._apply_temperature({r["fault_id"]: r["probability"] for r in fused})
        for r in fused:
            r["probability_raw"] = r["probability"]
            r["probability"] = scaled[r["fault_id"]]
            r["severity"] = self._severity_level(r["probability"])

        # ── Step 4: 排序并生成诊断报告 ──
        ranked = sorted(fused, key=lambda x: x["probability"], reverse=True)

        for i, item in enumerate(ranked):
            item["rank"] = i + 1
            item["probability_pct"] = f"{item['probability'] * 100:.1f}%"

        report = self._build_report(ranked, dga_analysis, evidence)

        probs = {r["fault_id"]: r["probability"] for r in ranked}
        p1 = ranked[0]["probability"] if ranked else 0.0
        p2 = ranked[1]["probability"] if len(ranked) > 1 else 0.0

        return {
            "status": "ok",
            "primary_fault": ranked[0]["fault_id"] if ranked else None,
            "primary_fault_name": ranked[0]["fault_name"] if ranked else None,
            "primary_probability": p1,
            "fault_ranking": ranked,
            "dga_analysis": dga_analysis,
            "evidence_used": [k for k, v in evidence.items() if v],
            "evidence_negative": [k for k, v in evidence.items() if v is False],
            "uncertainty": {
                "entropy_bits": self.entropy_bits(probs),
                "top1_top2_gap": p1 - p2,
                "calibrated": self.is_calibrated,
                "temperature": self.temperature,
                "params_source": self.params_source,
            },
            "report": report,
        }

    def _compute_posterior(self, evidence: dict[str, bool]) -> dict[str, float]:
        """
        朴素贝叶斯推理：P(F|E) ∝ P(F) · ∏ P(e_i | F)。

        evidence 取值：
          True  → 观测到征兆出现，乘 P(S=True|F)
          False → 明确排除该征兆（负观测），乘 1 - P(S=True|F)（B-1 新增，可用 use_negative_evidence 关闭）
          None / 缺失 → 未观测，跳过
        """
        posterior: dict[str, float] = {}

        for fault in self.faults:
            f_prob = self.prior.get(fault, 0.01)

            likelihood = 1.0
            for symptom, observed in evidence.items():
                if observed is None:
                    continue
                if observed is False and not self.use_negative_evidence:
                    continue
                row = self._cpt_lookup(fault, symptom)
                p_true = min(max(row.p_true, 1e-6), 1 - 1e-6)
                likelihood *= p_true if observed else (1.0 - p_true)

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
            H2, CH4, C2H2, C2H4, C2H6 (单位: μL/L)；值为 None 表示该气体未提供（被遮蔽）
            或 ratios: {C2H2_C2H4, CH4_H2, C2H4_C2H6}
        任一比值所需气体缺失时不做三比值规则匹配（`ratios_incomplete=True`），避免用 0 代入触发伪规则。
        """
        raw = {k: ctx.get(k) for k in ["H2", "CH4", "C2H2", "C2H4", "C2H6"]}
        missing = [k for k, v in raw.items() if v is None]
        gases = {k: (v if v is not None else 0) for k, v in raw.items()}
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

        # 应用 DGA 规则（五种气体齐全或显式给了 ratios 时才匹配）
        matched_rules: list[dict[str, Any]] = []
        ratios_incomplete = bool(missing) and not ratios
        if not ratios_incomplete:
            for rule in _DGA_RULES:
                if self._match_rule(rule, ratio_codes):
                    matched_rules.append({
                        "rule_name": rule["name"],
                        "related_faults": rule["related_faults"],
                        "confidence": rule["weight"],
                    })

        return {
            "gases": gases,
            "missing_gases": missing,
            "ratios": calc_ratios,
            "ratios_incomplete": ratios_incomplete,
            "ratio_codes": ratio_codes,
            "matched_rules": matched_rules,
            "interpretation": self._interpret_gases(gases, calc_ratios)
            + (f"\n- 缺少 {'/'.join(missing)} 浓度，未做三比值法判断" if ratios_incomplete else ""),
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

            # 融合：w · 贝叶斯 + (1-w) · 规则（若无规则则全用贝叶斯）。
            # w 默认 0.7；B-1 校准后可按故障类别从 learned_params.json 的 fusion_weights 读取。
            if rule_boost > 0:
                w = self.fusion_weight(fid)
                fused_prob = w * bayes_prob + (1.0 - w) * rule_boost
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
        excluded = [k for k, v in evidence.items() if v is False]
        if observed:
            lines.append(f"**输入征兆**：`{'`, `'.join(observed)}`\n")
        if excluded:
            lines.append(f"**已排除征兆**（负观测）：`{'`, `'.join(excluded)}`\n")

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

#: DL/T 722 注意值（μL/L），与 scripts/data/convert_real_dga.py 保持一致
GAS_ATTENTION: dict[str, float] = {"H2": 150, "CH4": 120, "C2H2": 5, "C2H4": 50, "C2H6": 65}
TOTAL_HC_ATTENTION = 150.0
GAS_RATE_RAPID_C2H2 = 50.0

#: 由 DGA 数值可完全决定的征兆集合（B-2 EIG 中成本为 0、B-3 模拟器中随气体一起遮蔽）
GAS_DERIVED_SYMPTOMS = ["H2_elevated", "CH4_elevated", "C2H2_elevated", "C2H4_elevated",
                        "C2H6_elevated", "TDCG_elevated", "gas_rate_rapid"]


def derive_gas_evidence(gases: dict[str, float], *, negative: bool = True) -> dict[str, bool]:
    """
    由五种特征气体浓度按 DL/T 722 注意值推导气体类征兆。
    negative=True 时低于注意值的征兆写为 False（负观测），否则只返回 True 的征兆。
    缺失（None）的气体不推导对应征兆。
    """
    ev: dict[str, bool] = {}
    for g, th in GAS_ATTENTION.items():
        v = gases.get(g)
        if v is None:
            continue
        if v > th:
            ev[f"{g}_elevated"] = True
        elif negative:
            ev[f"{g}_elevated"] = False
    hc = [gases.get(g) for g in ("CH4", "C2H2", "C2H4", "C2H6")]
    if all(v is not None for v in hc):
        total_hc = sum(hc)  # type: ignore[arg-type]
        if total_hc > TOTAL_HC_ATTENTION:
            ev["TDCG_elevated"] = True
        elif negative:
            ev["TDCG_elevated"] = False
    c2h2 = gases.get("C2H2")
    if c2h2 is not None:
        if c2h2 > GAS_RATE_RAPID_C2H2:
            ev["gas_rate_rapid"] = True
        elif negative:
            ev["gas_rate_rapid"] = False
    return ev


def get_engine() -> FaultBayesianNetwork:
    global _bn_engine
    if _bn_engine is None:
        _bn_engine = FaultBayesianNetwork()
    return _bn_engine


DEFAULT_LEARNED_PARAMS = Path(__file__).resolve().parents[2] / "data" / "real" / "dga" / "learned_params.json"


def configure_engine(mode: str = "calibrated", params_path: str | Path | None = None) -> FaultBayesianNetwork:
    """
    按配置开关重建全局引擎（B-4 `attribution_mode`）：
      expert      专家默认 CPT，忽略 FAULT_ATTR_PARAMS
      calibrated  加载 learned_params.json（B-1 学习先验 / CPT / 融合权重 / 温度），
                  路径优先级：params_path > FAULT_ATTR_PARAMS > data/real/dga/learned_params.json
    返回新引擎；后续 get_engine() / fault_attribution() 均使用它。
    """
    import json
    import os

    global _bn_engine
    if mode not in ("expert", "calibrated"):
        raise ValueError(f"attribution_mode 只能是 expert|calibrated，得到 {mode!r}")
    if mode == "expert":
        _bn_engine = FaultBayesianNetwork(params={})
        _bn_engine.params_source = "expert_default"
        return _bn_engine
    path = Path(params_path or os.environ.get("FAULT_ATTR_PARAMS") or DEFAULT_LEARNED_PARAMS)
    if not path.exists():
        raise FileNotFoundError(f"attribution_mode=calibrated 需要参数文件，未找到 {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    engine = FaultBayesianNetwork(params=data)
    engine.params_source = str(path)
    _bn_engine = engine
    return engine


def fault_attribution(
    dga_data: dict[str, Any] | None = None,
    evidence: dict[str, bool] | None = None,
    query: str = "",
    *,
    with_eig: bool | None = None,
    rounds_done: int = 0,
) -> dict[str, Any]:
    """
    故障归因主入口函数，供 MCP 工具调用。

    参数:
        dga_data: DGA 气体数据字典，字段包括 H2/CH4/C2H2/C2H4/C2H6（单位μL/L）
                  以及可选的 ratios 子字典；缺失（None）的气体视为未观测（被遮蔽），不派生负观测
        evidence: 征兆布尔字典，如 {"H2_elevated": True, "C2H2_elevated": False}
                  True=出现，False=明确排除（负观测），缺失=未观测
        query: 用户原始问题描述（用于上下文）
        with_eig: 是否附带 EIG 推荐块（默认读环境变量 FAULT_ATTR_EIG，默认开）
        rounds_done: 已进行的追问轮数（供停止准则）

    示例:
        fault_attribution(
            dga_data={"H2": 680, "C2H2": 45, "C2H4": 230, "CH4": 150},
            evidence={"H2_elevated": True, "C2H2_elevated": True},
        )
    """
    import os
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
    gases_obs: dict[str, float | None] = {}  # None 表示该气体未提供（被遮蔽），不派生负观测

    if dga_data:
        for g in gases:
            raw = dga_data.get(g)
            gases_obs[g] = None if raw is None else _num(raw)
            gases[g] = gases_obs[g] or 0.0

    # 根据注意值（DL/T 722）自动推断征兆。
    # 有 DGA 数值时，气体类征兆是「完全可观测」的：低于注意值即为负观测（False），
    # 让贝叶斯推理能利用「已排除的征兆」（B-1）。engine.use_negative_evidence=False 时退回 v1 行为。
    if dga_data:
        for sym, val in derive_gas_evidence(gases_obs, negative=engine.use_negative_evidence).items():
            computed_evidence.setdefault(sym, val)

    symptom_context = None
    if dga_data:
        symptom_context = {
            "H2": gases_obs.get("H2"),
            "CH4": gases_obs.get("CH4"),
            "C2H2": gases_obs.get("C2H2"),
            "C2H4": gases_obs.get("C2H4"),
            "C2H6": gases_obs.get("C2H6"),
            "query": query,
        }
        if isinstance(dga_data.get("ratios"), dict):
            symptom_context["ratios"] = dga_data["ratios"]
        result = engine.infer(computed_evidence, symptom_context=symptom_context)
    else:
        result = engine.infer(computed_evidence)

    # B-2：EIG 推荐块。FAULT_ATTR_EIG=0 或 with_eig=False 时关闭（mode2 工程基座不输出不确定性推荐）。
    if with_eig is None:
        with_eig = os.environ.get("FAULT_ATTR_EIG", "1") not in ("0", "false", "False")
    if with_eig:
        try:
            from src.tools.eig import recommend, render_recommendations
            rec = recommend(computed_evidence, symptom_context, engine=engine, rounds_done=rounds_done)
            result["uncertainty"].update({
                "recommendations": rec["recommendations"],
                "suggested_action": rec["suggested_action"],
                "suggested_tool": rec["suggested_tool"],
                "stop_reason": rec["stop_reason"],
                "lambda": rec["lambda"],
                "stop": rec["stop"],
            })
            result["report"] = result["report"] + "\n\n" + render_recommendations(rec)
        except Exception as exc:  # noqa: BLE001
            result["uncertainty"]["eig_error"] = str(exc)

    return result
