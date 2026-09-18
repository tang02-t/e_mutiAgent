#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实 DGA 数据合并转换脚本（多源）。

将以下公开真实数据源统一转换、合并为符合 data/synthetic/SCHEMA.md 的
dga_records.csv / dga_records.jsonl，用于替换原合成“假案例”。
下游 fault_attribution / eval_fault_attribution 无需改动。

数据源（放在 data/raw/dga/ 下）：
  1. dga_dataset.csv      —— 来源未核实（非 Kaggle，见 docs/design/data_and_evaluation.md §1.1 R1），IEC 编码标签(NF/PD/D1/D2/T1/T2/T3)，sep=';' decimal=','，4150 条
  2. data.xlsx            —— alan-456，中文故障性质标签，含中国电网数据，2321 条
  3. dataset_(589).xlsx   —— alan-456，中文故障性质标签，论文集，589 条

处理：
  - 两套标签体系（IEC 编码 / 中文性质）统一映射到本项目 9 类 fault_label；
  - 5 种气体统一为 float；近似零值(0.0001)保留；
  - 补全 schema 派生字段(TDCG/total_hydrocarbon/evidence/severity)；
  - 基于 (5气体四舍五入+标签) 去重；
  - 输出合并数据集，并标注每条来源 source。

用法：
  python3 scripts/data/convert_real_dga.py
  python3 scripts/data/convert_real_dga.py --out-dir data/real/dga
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ── 标签映射：两套体系 → 本项目 9 类 fault_label ──
# 说明：源标签是“故障性质/能量等级”，本项目是“故障部位”，取机理最接近的部位；
#       机械类(绕组变形/分接开关)无 DGA 直接对应，故不会出现。
LABEL_MAP: dict[str, tuple[str, str]] = {
    # --- Kaggle IEC 编码 ---
    "NF": ("normal", "正常"),
    "PD": ("partial_discharge", "局部放电"),
    "D1": ("partial_discharge", "局部放电"),            # 低能放电，机理近局放
    "D2": ("winding_short_circuit", "绕组匝间短路"),     # 高能放电
    "T1": ("insulation_degradation", "绝缘老化"),        # 低温过热 <300℃
    "T2": ("overload_overheating", "过载过热"),          # 中温过热 300~700℃
    "T3": ("overload_overheating", "过载过热"),          # 高温过热 >700℃
    # --- 中文故障性质 ---
    "正常": ("normal", "正常"),
    "局部放电": ("partial_discharge", "局部放电"),
    "低能放电": ("partial_discharge", "局部放电"),         # 低能放电，机理近局放
    "高能放电": ("winding_short_circuit", "绕组匝间短路"),  # 电弧放电
    "电弧放电": ("winding_short_circuit", "绕组匝间短路"),
    "低温过热": ("insulation_degradation", "绝缘老化"),
    "中温过热": ("overload_overheating", "过载过热"),
    "高温过热": ("overload_overheating", "过载过热"),
}

# DL/T 722 注意值（与 fault_attribution / generate_synthetic_data 完全一致）
ATTENTION = {"H2": 150, "CH4": 120, "C2H2": 5, "C2H4": 50, "C2H6": 65}

GAS_COLS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


def build_evidence(g: dict) -> dict:
    """按 DL/T 722 注意值推断 evidence 布尔字典（对齐 fault_attribution 内部逻辑）。"""
    ev = {}
    if g["H2"] > ATTENTION["H2"]:
        ev["H2_elevated"] = True
    if g["CH4"] > ATTENTION["CH4"]:
        ev["CH4_elevated"] = True
    if g["C2H2"] > ATTENTION["C2H2"]:
        ev["C2H2_elevated"] = True
    if g["C2H4"] > ATTENTION["C2H4"]:
        ev["C2H4_elevated"] = True
    if g["C2H6"] > ATTENTION["C2H6"]:
        ev["C2H6_elevated"] = True
    th = g["CH4"] + g["C2H2"] + g["C2H4"] + g["C2H6"]
    if th > 150:
        ev["TDCG_elevated"] = True
    if g["C2H2"] > 50:
        ev["gas_rate_rapid"] = True
    return ev


def severity_of(label: str, g: dict) -> str:
    """按故障类型 + 气体超标程度给出严重度。"""
    if label == "normal":
        return "normal"
    c2h2 = g["C2H2"]
    th = g["CH4"] + g["C2H2"] + g["C2H4"] + g["C2H6"]
    if label == "winding_short_circuit":
        return "critical" if c2h2 > 50 else "serious"
    if c2h2 > 50 or th > 1500:
        return "serious"
    return "attention"


def load_source(path: Path) -> pd.DataFrame:
    """读取单个数据源，统一为列：H2,CH4,C2H6,C2H4,C2H2,raw_label,source。"""
    name = path.name
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=";", decimal=",")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Fail": "raw_label"})
    else:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={"故障类型": "raw_label"})

    df["raw_label"] = df["raw_label"].astype(str).str.strip().str.upper()
    for c in GAS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=GAS_COLS)
    df["source"] = name
    return df[GAS_COLS + ["raw_label", "source"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data/raw/dga"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/real/dga"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = ["dga_dataset.csv", "data.xlsx", "dataset_(589).xlsx"]
    frames = []
    for s in sources:
        p = data_dir / s
        if not p.exists():
            print(f"[WARN] 缺少数据源，跳过: {p}")
            continue
        f = load_source(p)
        print(f"[load] {s:24s} -> {len(f)} 行")
        frames.append(f)

    if not frames:
        print("无可用数据源，退出。")
        return

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)

    # 去重：基于 5 气体(四舍五入3位) + 原始标签
    df["_key"] = (
        df[GAS_COLS].round(3).astype(str).agg("|".join, axis=1) + "|" + df["raw_label"]
    )
    df = df.drop_duplicates("_key").reset_index(drop=True)
    n_dedup = len(df)

    csv_path = out_dir / "dga_records.csv"
    jsonl_path = out_dir / "dga_records.jsonl"
    fieldnames = [
        "record_id", "device_id", "sample_time",
        "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2",
        "TDCG", "total_hydrocarbon", "gas_rate_ml_per_day",
        "oil_temp", "load_ratio", "fault_label", "fault_label_cn", "severity", "source",
    ]

    n_ok = 0
    skipped: dict[str, int] = {}
    label_dist: dict[str, int] = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
         open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in df.iterrows():
            raw = row["raw_label"]
            if raw not in LABEL_MAP:
                skipped[raw] = skipped.get(raw, 0) + 1
                continue
            label, label_cn = LABEL_MAP[raw]
            g = {c: round(float(row[c]), 3) for c in GAS_COLS}

            CO = CO2 = 0.0
            tdcg = round(sum(g.values()) + CO, 3)
            total_hc = round(g["CH4"] + g["C2H2"] + g["C2H4"] + g["C2H6"], 3)
            evidence = build_evidence(g)
            severity = severity_of(label, g)
            rid = f"DGA-R{i + 1:06d}"
            did = f"RT-{(i % 50) + 1:04d}"

            writer.writerow({
                "record_id": rid, "device_id": did, "sample_time": "",
                "H2": g["H2"], "CH4": g["CH4"], "C2H2": g["C2H2"],
                "C2H4": g["C2H4"], "C2H6": g["C2H6"], "CO": CO, "CO2": CO2,
                "TDCG": tdcg, "total_hydrocarbon": total_hc,
                "gas_rate_ml_per_day": "", "oil_temp": "", "load_ratio": "",
                "fault_label": label, "fault_label_cn": label_cn,
                "severity": severity, "source": row["source"],
            })
            fjsonl.write(json.dumps({
                "record_id": rid, "device_id": did, "sample_time": "",
                "dga_data": {**g, "CO": CO, "CO2": CO2},
                "evidence": evidence,
                "gas_rate_ml_per_day": None, "oil_temp": None, "load_ratio": None,
                "fault_label": label, "fault_label_cn": label_cn,
                "severity": severity, "source": row["source"],
            }, ensure_ascii=False) + "\n")
            n_ok += 1
            label_dist[label] = label_dist.get(label, 0) + 1

    print(f"\n[OK] 合并 {n_raw} 行 -> 去重 {n_dedup} 行 -> 输出 {n_ok} 条")
    if skipped:
        print(f"     跳过未知标签: {skipped}")
    print(f"     标签分布: {dict(sorted(label_dist.items()))}")
    print(f"     CSV   -> {csv_path}")
    print(f"     JSONL -> {jsonl_path}")


if __name__ == "__main__":
    main()
