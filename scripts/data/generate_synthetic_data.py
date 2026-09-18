#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变压器故障诊断系统 - 模拟数据生成器（纯标准库，无第三方依赖）。

生成 4 类数据，schema 见 data/synthetic/SCHEMA.md：
  1. DGA 油色谱数据（含真实故障标签，对接 fault_attribution）
  2. 在线监测时序数据（列对齐 ETT，对接 ett_forecast / timeseries_anomaly）
  3. 故障案例库
  4. 端到端评测集

设计原则：
  - 每类故障按其物理机理（参考 DL/T 722-2014 三比值法、产气特征）生成自洽的气体分布；
  - 标签与数据强相关，可用于后续校准贝叶斯网络 / 评估系统准确率；
  - 后续可用真实数据按相同 schema 直接替换。

用法：
  python3 scripts/data/generate_synthetic_data.py            # 默认规模
  python3 scripts/data/generate_synthetic_data.py --seed 7   # 指定随机种子
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 路径
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "synthetic"
DGA_DIR = OUT / "dga"
TS_DIR = OUT / "timeseries"
CASE_DIR = OUT / "cases"
EVAL_DIR = OUT / "eval"
for d in (DGA_DIR, TS_DIR, CASE_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 故障标签元数据（与 src/tools/fault_attribution.py FAULT_IDS 一致）
# ──────────────────────────────────────────────────────────────
FAULT_CN = {
    "normal": "正常",
    "winding_deformation": "绕组变形/位移",
    "winding_short_circuit": "绕组匝间短路",
    "core_grounding": "铁芯多点接地",
    "bushing_fault": "套管故障",
    "olc_fault": "分接开关故障",
    "partial_discharge": "局部放电",
    "overload_overheating": "过载过热",
    "insulation_degradation": "绝缘老化",
}

# 各故障的相对出现频率（先验，正常占比最高）
FAULT_WEIGHTS = {
    "normal": 0.40,
    "insulation_degradation": 0.12,
    "overload_overheating": 0.11,
    "partial_discharge": 0.08,
    "olc_fault": 0.07,
    "core_grounding": 0.05,
    "bushing_fault": 0.05,
    "winding_deformation": 0.05,
    "winding_short_circuit": 0.04,
}

# DGA 气体生成模板：每种故障给出各气体的 (均值, 标准差) μL/L
#
# 关键设计：依据 DL/T 722-2014 三比值法，每类故障的气体严格落在其对应的比值编码区间，
# 使三个比值 [C2H2/C2H4, CH4/H2, C2H4/C2H6] 的编码能够区分故障，而非仅靠绝对值。
# 比值编码（表6）：
#   C2H2/C2H4: <0.1->0, [0.1,3)->1, >=3->2
#   CH4/H2:    <0.1->1, [0.1,1)->0, >=1->2
#   C2H4/C2H6: <1->0,   [1,3)->1,   >=3->2
#
# 各故障对应的目标编码（表7）：
#   局部放电          : C2H2/C2H4=0, CH4/H2=1(即<0.1), C2H4/C2H6=0  -> H2 主导, 烃类低
#   绝缘老化(低温过热) : C2H2/C2H4=0, CH4/H2=0, C2H4/C2H6=0
#   铁芯接地(中温过热) : C2H2/C2H4=0, CH4/H2=2, C2H4/C2H6=1
#   过载过热(高温过热) : C2H2/C2H4=0, CH4/H2=2, C2H4/C2H6=2  -> C2H4 主导
#   套管故障(低能放电) : C2H2/C2H4=2(>=3), 低烃
#   分接开关(电弧放电) : C2H2/C2H4=1([0.1,3)), C2H2 显著
#   绕组短路(电弧兼过热): C2H2/C2H4=1, CH4/H2=2, 高能量, 全气体高
#   绕组变形           : 兼具放电与机械特征, C2H2/C2H4=1, 中等水平
DGA_PROFILE = {
    # 正常：所有气体低
    "normal":                 {"H2": (30, 10),  "CH4": (18, 6),   "C2H2": (0.4, 0.3), "C2H4": (10, 4),  "C2H6": (16, 5),  "CO": (180, 60),  "CO2": (1800, 500)},
    # 局部放电：H2 极高, CH4/H2<0.1, C2H2≈0, C2H4 低于 C2H6
    "partial_discharge":      {"H2": (650, 120),"CH4": (45, 12),  "C2H2": (1.0, 0.6), "C2H4": (18, 6),  "C2H6": (28, 8),  "CO": (240, 70),  "CO2": (2300, 550)},
    # 绝缘老化(低温过热<150℃): C2H2≈0, CH4/H2∈[0.1,1), C2H4<C2H6, CO/CO2 突出
    "insulation_degradation": {"H2": (130, 30), "CH4": (75, 18),  "C2H2": (1.2, 0.7), "C2H4": (32, 9),  "C2H6": (90, 22), "CO": (720, 180), "CO2": (5400, 1000)},
    # 铁芯接地(中温过热): C2H2≈0, CH4/H2>=1, C2H4/C2H6∈[1,3)
    "core_grounding":         {"H2": (90, 20),  "CH4": (220, 45), "C2H2": (1.5, 0.8), "C2H4": (180, 35),"C2H6": (95, 22), "CO": (300, 90),  "CO2": (2600, 600)},
    # 过载过热(高温过热>700℃): C2H2≈0, CH4/H2>=1, C2H4/C2H6>=3, C2H4 主导
    "overload_overheating":   {"H2": (70, 18),  "CH4": (200, 45), "C2H2": (2.0, 1.0), "C2H4": (420, 90),"C2H6": (95, 22), "CO": (520, 140), "CO2": (4200, 900)},
    # 套管故障(低能放电): C2H2/C2H4>=3, 烃类总体低, H2 较高
    "bushing_fault":          {"H2": (320, 80), "CH4": (60, 18),  "C2H2": (160, 40),  "C2H4": (28, 9),  "C2H6": (22, 7),  "CO": (220, 70),  "CO2": (2200, 550)},
    # 分接开关(电弧放电): C2H2/C2H4∈[0.1,3), C2H2 显著, CH4/H2<1
    "olc_fault":              {"H2": (240, 55), "CH4": (90, 22),  "C2H2": (210, 45),  "C2H4": (180, 40),"C2H6": (55, 15), "CO": (240, 70),  "CO2": (2300, 550)},
    # 绕组匝间短路(电弧兼过热): C2H2/C2H4∈[0.1,3), CH4/H2>=1, 能量大全气体高
    "winding_short_circuit":  {"H2": (180, 40), "CH4": (260, 55), "C2H2": (260, 55),  "C2H4": (300, 60),"C2H6": (85, 20), "CO": (360, 110), "CO2": (3000, 700)},
    # 绕组变形(放电+机械): C2H2/C2H4∈[0.1,3) 中等, 整体中等水平
    "winding_deformation":    {"H2": (160, 38), "CH4": (85, 20),  "C2H2": (90, 22),   "C2H4": (150, 32),"C2H6": (62, 16), "CO": (280, 85),  "CO2": (2700, 650)},
}

# 严重程度映射（按故障性质）
FAULT_SEVERITY = {
    "normal": "normal",
    "insulation_degradation": "attention",
    "overload_overheating": "serious",
    "partial_discharge": "attention",
    "olc_fault": "serious",
    "core_grounding": "serious",
    "bushing_fault": "serious",
    "winding_deformation": "serious",
    "winding_short_circuit": "critical",
}

TRANSFORMER_MODELS = [
    ("SFSZ-180000/220", 220, 180000, "ONAN/ONAF"),
    ("SFSZ-240000/220", 220, 240000, "ONAF/OFAF"),
    ("SFP-360000/500",  500, 360000, "ODAF"),
    ("SZ-50000/110",    110, 50000,  "ONAN"),
    ("SFSZ-120000/220", 220, 120000, "ONAN/ONAF"),
    ("SSZ-31500/110",   110, 31500,  "ONAN"),
]


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
def clamp(v, lo=0.0, hi=None):
    v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def gauss_pos(mean, std):
    """正态采样并截断为非负。"""
    return clamp(random.gauss(mean, std), 0.0)


def build_evidence(g: dict) -> dict:
    """按 DL/T 722 注意值推断 evidence 布尔字典（与 fault_attribution 内部逻辑一致）。"""
    ev = {}
    if g["H2"] > 150: ev["H2_elevated"] = True
    if g["CH4"] > 120: ev["CH4_elevated"] = True
    if g["C2H2"] > 5: ev["C2H2_elevated"] = True
    if g["C2H4"] > 50: ev["C2H4_elevated"] = True
    if g["C2H6"] > 65: ev["C2H6_elevated"] = True
    if g.get("CO", 0) > 300: ev["CO_elevated"] = True
    if g.get("CO2", 0) > 4000: ev["CO2_elevated"] = True
    th = g["CH4"] + g["C2H2"] + g["C2H4"] + g["C2H6"]
    if th > 150: ev["TDCG_elevated"] = True
    if g["C2H2"] > 50: ev["gas_rate_rapid"] = True
    return ev


def weighted_choice(weights: dict) -> str:
    r = random.random() * sum(weights.values())
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return list(weights.keys())[-1]


# ──────────────────────────────────────────────────────────────
# 1. 设备台账
# ──────────────────────────────────────────────────────────────
def gen_devices(n: int) -> list:
    devices = []
    this_year = 2026
    for i in range(1, n + 1):
        model, kv, cap, cooling = random.choice(TRANSFORMER_MODELS)
        commission = random.randint(2003, 2023)
        devices.append({
            "device_id": f"TR-{i:04d}",
            "model": model,
            "voltage_level_kv": kv,
            "rated_capacity_kva": cap,
            "commission_year": commission,
            "service_years": this_year - commission,
            "location": f"某{random.choice([110,220,500])}kV变电站#{random.randint(1,9)}",
            "cooling_type": cooling,
        })
    (TS_DIR / "devices.json").write_text(
        json.dumps(devices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return devices


# ──────────────────────────────────────────────────────────────
# 2. DGA 数据
# ──────────────────────────────────────────────────────────────
def gen_dga(n: int, devices: list):
    csv_path = DGA_DIR / "dga_records.csv"
    jsonl_path = DGA_DIR / "dga_records.jsonl"

    headers = [
        "record_id", "device_id", "sample_time",
        "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2",
        "TDCG", "total_hydrocarbon", "gas_rate_ml_per_day",
        "oil_temp", "load_ratio", "fault_label", "fault_label_cn", "severity",
    ]

    base = datetime(2023, 1, 1, 8, 0, 0)
    with open(csv_path, "w", newline="", encoding="utf-8") as cf, \
         open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.writer(cf)
        writer.writerow(headers)

        for i in range(1, n + 1):
            label = weighted_choice(FAULT_WEIGHTS)
            prof = DGA_PROFILE[label]
            g = {gas: round(gauss_pos(m, s), 1) for gas, (m, s) in prof.items()}

            th = round(g["CH4"] + g["C2H2"] + g["C2H4"] + g["C2H6"], 1)
            tdcg = round(th + g["H2"] + g["CO"], 1)

            # 产气速率：故障越严重越快
            sev = FAULT_SEVERITY[label]
            rate_base = {"normal": 2, "attention": 12, "serious": 30, "critical": 55}[sev]
            gas_rate = round(clamp(random.gauss(rate_base, rate_base * 0.3)), 1)

            # 油温/负载率：过热/过载类偏高
            if label in ("overload_overheating", "winding_short_circuit"):
                oil_temp = round(random.gauss(82, 6), 1)
                load_ratio = round(clamp(random.gauss(0.92, 0.06), 0.3, 1.15), 2)
            elif label == "normal":
                oil_temp = round(random.gauss(55, 6), 1)
                load_ratio = round(clamp(random.gauss(0.62, 0.12), 0.2, 1.0), 2)
            else:
                oil_temp = round(random.gauss(65, 8), 1)
                load_ratio = round(clamp(random.gauss(0.72, 0.12), 0.2, 1.05), 2)

            dev = random.choice(devices)
            sample_time = (base + timedelta(hours=random.randint(0, 24 * 365 * 2))).strftime("%Y-%m-%d %H:%M:%S")
            rid = f"DGA-{i:06d}"

            writer.writerow([
                rid, dev["device_id"], sample_time,
                g["H2"], g["CH4"], g["C2H2"], g["C2H4"], g["C2H6"], g["CO"], g["CO2"],
                tdcg, th, gas_rate, oil_temp, load_ratio,
                label, FAULT_CN[label], sev,
            ])

            jf.write(json.dumps({
                "record_id": rid,
                "device_id": dev["device_id"],
                "sample_time": sample_time,
                "dga_data": {k: g[k] for k in ("H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2")},
                "evidence": build_evidence(g),
                "gas_rate_ml_per_day": gas_rate,
                "oil_temp": oil_temp,
                "load_ratio": load_ratio,
                "fault_label": label,
                "fault_label_cn": FAULT_CN[label],
                "severity": sev,
            }, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


# ──────────────────────────────────────────────────────────────
# 3. 时序数据（列对齐 ETT）
# ──────────────────────────────────────────────────────────────
def gen_timeseries(devices: list, n_devices: int, n_points: int):
    """为前 n_devices 台设备各生成 n_points 小时点的负载+油温序列。"""
    sel = devices[:n_devices]
    for dev in sel:
        rows = []
        start = datetime(2024, 1, 1, 0, 0, 0)
        # 设备基线负载水平
        base_load = random.uniform(0.5, 0.8)
        for t in range(n_points):
            ts = start + timedelta(hours=t)
            hour = ts.hour
            # 日周期负载曲线（白天高、夜间低）+ 周周期 + 噪声
            daily = 0.25 * math.sin((hour - 8) / 24 * 2 * math.pi)
            weekly = 0.08 * math.sin(t / (24 * 7) * 2 * math.pi)
            load = clamp(base_load + daily + weekly + random.gauss(0, 0.04), 0.1, 1.2)

            hufl = round(load * 8 + random.gauss(0, 0.3), 3)
            hull = round(load * 2.5 + random.gauss(0, 0.2), 3)
            mufl = round(load * 2.2 + random.gauss(0, 0.15), 3)
            mull = round(load * 0.6 + random.gauss(0, 0.08), 3)
            lufl = round(load * 5.5 + random.gauss(0, 0.25), 3)
            lull = round(load * 1.8 + random.gauss(0, 0.12), 3)

            # 油温：与负载强相关 + 热惯性（与前一刻相关）+ 环境
            ambient = 15 + 12 * math.sin((t / 24 - 6) / 24 * 2 * math.pi)
            ot_target = ambient + load * 45
            if rows:
                prev_ot = rows[-1][-1]
                ot = round(0.7 * prev_ot + 0.3 * ot_target + random.gauss(0, 0.8), 3)
            else:
                ot = round(ot_target + random.gauss(0, 1), 3)

            rows.append([ts.strftime("%Y-%m-%d %H:%M:%S"), hufl, hull, mufl, mull, lufl, lull, ot])

        path = TS_DIR / f"{dev['device_id']}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"])
            w.writerows(rows)
    return len(sel)


# ──────────────────────────────────────────────────────────────
# 4. 故障案例库
# ──────────────────────────────────────────────────────────────
HANDLING = {
    "winding_short_circuit": ("高压绕组匝间短路", "立即停电，吊罩检查，更换故障绕组", "修复后投运正常"),
    "winding_deformation": ("绕组轴向变形", "退出运行，进行频响及短路阻抗测试，必要时返厂", "返厂大修后恢复"),
    "core_grounding": ("铁芯多点接地", "测量铁芯接地电流，查找并消除多余接地点", "处理后接地电流恢复正常"),
    "bushing_fault": ("高压套管内部放电", "停电更换套管", "更换后试验合格"),
    "olc_fault": ("有载分接开关触头烧蚀", "停电检修分接开关，更换触头", "检修后接触电阻合格"),
    "partial_discharge": ("绝缘件局部放电", "局放定位，加强监测，安排停电检查", "监测稳定后计划性检修"),
    "overload_overheating": ("长期过载导致过热", "调整负荷分配，加强冷却，缩短监测周期", "降负荷后油温恢复"),
    "insulation_degradation": ("绝缘老化", "缩短试验周期，评估剩余寿命，安排滤油/换油", "处理后含水量下降"),
}
SYMPTOM_POOL = {
    "winding_short_circuit": ["C2H2持续升高", "油温告警", "轻瓦斯动作", "差动保护动作"],
    "winding_deformation": ["短路阻抗变化", "频响曲线偏移", "异常振动"],
    "core_grounding": ["铁芯接地电流超标", "CH4/C2H4升高", "局部过热"],
    "bushing_fault": ["套管介损增大", "H2显著升高", "套管末屏放电"],
    "olc_fault": ["C2H2突增", "切换异响", "接触电阻增大"],
    "partial_discharge": ["H2升高", "局放量超标", "超声局放信号"],
    "overload_overheating": ["油温持续偏高", "CO/CO2升高", "负载率超100%"],
    "insulation_degradation": ["CO2大幅升高", "油中含水量上升", "绝缘电阻下降"],
}


def gen_cases(n: int, devices: list):
    path = CASE_DIR / "fault_cases.jsonl"
    fault_labels = [k for k in FAULT_CN if k != "normal"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            label = random.choice(fault_labels)
            prof = DGA_PROFILE[label]
            g = {gas: round(gauss_pos(m, s), 1) for gas, (m, s) in prof.items() if gas in ("H2", "CH4", "C2H2", "C2H4", "C2H6")}
            dev = random.choice(devices)
            loc, handling, outcome = HANDLING[label]
            f.write(json.dumps({
                "case_id": f"CASE-{i:04d}",
                "device_id": dev["device_id"],
                "device_info": {
                    "model": dev["model"],
                    "voltage_level_kv": dev["voltage_level_kv"],
                    "service_years": dev["service_years"],
                },
                "symptoms": random.sample(SYMPTOM_POOL[label], k=min(3, len(SYMPTOM_POOL[label]))),
                "dga_snapshot": g,
                "tests": {
                    "dc_resistance_unbalance_pct": round(random.uniform(0.5, 4.0), 2),
                    "insulation_resistance_mohm": round(random.uniform(500, 3000)),
                    "dielectric_loss_pct": round(random.uniform(0.3, 1.5), 2),
                    "winding_fra": random.choice(["正常", "轻微偏移", "明显偏移"]),
                },
                "fault_location": loc,
                "fault_label": label,
                "fault_label_cn": FAULT_CN[label],
                "handling": handling,
                "outcome": outcome,
                "reference": "DL/T 722-2014",
            }, ensure_ascii=False) + "\n")
    return path


# ──────────────────────────────────────────────────────────────
# 5. 评测集
# ──────────────────────────────────────────────────────────────
KEY_POINTS = {
    "winding_short_circuit": ["立即停电", "高能放电", "C2H2显著超标", "吊罩检查"],
    "winding_deformation": ["频响测试", "短路阻抗", "绕组变形"],
    "core_grounding": ["铁芯接地电流", "多点接地", "局部过热"],
    "bushing_fault": ["套管放电", "介损测试", "更换套管"],
    "olc_fault": ["分接开关", "C2H2突增", "触头烧蚀"],
    "partial_discharge": ["局部放电", "H2升高", "加强监测"],
    "overload_overheating": ["过载过热", "降负荷", "油温偏高"],
    "insulation_degradation": ["绝缘老化", "CO2升高", "评估寿命"],
}


def gen_eval(n: int, devices: list):
    path = EVAL_DIR / "eval_set.jsonl"
    fault_labels = list(KEY_POINTS.keys())
    with open(path, "w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            label = random.choice(fault_labels)
            prof = DGA_PROFILE[label]
            g = {gas: round(gauss_pos(m, s), 1) for gas, (m, s) in prof.items() if gas in ("H2", "CH4", "C2H2", "C2H4")}
            dev = random.choice(devices)
            critical = label in ("winding_short_circuit", "bushing_fault", "winding_deformation")
            query = (
                f"{dev['voltage_level_kv']}kV主变（{dev['model']}，运行{dev['service_years']}年）"
                f"油色谱检测：H2={g['H2']}, CH4={g['CH4']}, C2H2={g['C2H2']}, C2H4={g['C2H4']} μL/L，"
                f"请诊断故障类型并给出处理建议。"
            )
            f.write(json.dumps({
                "eval_id": f"EVAL-{i:04d}",
                "user_query": query,
                "context": {"device_id": dev["device_id"], "voltage_level_kv": dev["voltage_level_kv"]},
                "expected": {
                    "primary_fault": label,
                    "primary_fault_cn": FAULT_CN[label],
                    "key_points": KEY_POINTS[label],
                    "must_mention_safety": critical,
                },
            }, ensure_ascii=False) + "\n")
    return path


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="生成变压器故障诊断模拟数据")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-devices", type=int, default=50)
    ap.add_argument("--n-dga", type=int, default=3000)
    ap.add_argument("--ts-devices", type=int, default=20)
    ap.add_argument("--ts-points", type=int, default=2000)
    ap.add_argument("--n-cases", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=120)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[1/5] 生成设备台账 ({args.n_devices} 台) ...")
    devices = gen_devices(args.n_devices)

    print(f"[2/5] 生成 DGA 数据 ({args.n_dga} 条) ...")
    dga_csv, dga_jsonl = gen_dga(args.n_dga, devices)

    print(f"[3/5] 生成时序数据 ({args.ts_devices} 台 × {args.ts_points} 点) ...")
    n_ts = gen_timeseries(devices, args.ts_devices, args.ts_points)

    print(f"[4/5] 生成故障案例 ({args.n_cases} 例) ...")
    case_path = gen_cases(args.n_cases, devices)

    print(f"[5/5] 生成评测集 ({args.n_eval} 条) ...")
    eval_path = gen_eval(args.n_eval, devices)

    print("\n完成。输出目录：", OUT)
    print(f"  - 设备台账   : {TS_DIR / 'devices.json'} ({len(devices)} 台)")
    print(f"  - DGA CSV    : {dga_csv}")
    print(f"  - DGA JSONL  : {dga_jsonl}")
    print(f"  - 时序数据   : {TS_DIR}/TR-*.csv ({n_ts} 个文件)")
    print(f"  - 故障案例   : {case_path}")
    print(f"  - 评测集     : {eval_path}")


if __name__ == "__main__":
    main()
