#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-3 部分观测诊断模拟器（数据 D12）。

从 5143 条真实 DGA 记录（data/real/dga/dga_records.jsonl，非 normal）抽样，随机遮蔽部分气体，
构造「初始观测 + 隐藏观测 + 真实标签」三元组；提供 `PartialObsEnv` 供主动规划策略逐步揭示征兆，
记录每次揭示的成本。只用真实 DGA 派生，合成数据不入（data_guard 校验）。

遮蔽档位（按遮蔽的气体种类数）：
  light  遮蔽 1 种气体
  medium 遮蔽 2 种气体
  heavy  遮蔽 3 种气体（五种特征气体剩 2 种）
每档 500 条，按故障类别分层（比例与原始 3729 条非 normal 记录一致），固定 seed 可复现。

字段：
  sim_id, source_record_id, level, fault_label,
  full_gases  {H2, CH4, C2H2, C2H4, C2H6}        真实浓度（评测时不可见）
  visible_gases {...}                              初始可见（被遮蔽的键值为 null）
  masked_gases  [gas, ...]
  initial_evidence  {symptom: bool}                由可见气体派生（含负观测）
  hidden_evidence   {symptom: bool}                由被遮蔽气体派生；无法派生的（如 TDCG 需全部烃）不出现
  unavailable_symptoms [symptom, ...]              该记录无字段的征兆（CO / 温度 / 负载 / 振动 / 局放），reveal 返回 None

用法：
  python3 scripts/sim/partial_obs_sim.py --build            # 生成 data/eval/d12/partial_obs.jsonl + DATA_CARD.md
  python3 scripts/sim/partial_obs_sim.py --check            # 校验可复现（重建后 sha256 比对）与标签分布
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tools.fault_attribution import SYMPTOM_IDS, GAS_DERIVED_SYMPTOMS, derive_gas_evidence  # noqa: E402
from src.utils.data_guard import assert_not_synthetic  # noqa: E402

SRC = ROOT / "data/real/dga/dga_records.jsonl"
OUT_DIR = ROOT / "data/eval/d12"
OUT = OUT_DIR / "partial_obs.jsonl"
GASES = ["H2", "CH4", "C2H2", "C2H4", "C2H6"]
LEVELS = {"light": 1, "medium": 2, "heavy": 3}
PER_LEVEL = 500
SEED = 20260915

# 征兆 → 需要的气体（用于判断遮蔽后该征兆是否属于「隐藏可揭示」）
_SYMPTOM_GASES = {
    "H2_elevated": ["H2"], "CH4_elevated": ["CH4"], "C2H2_elevated": ["C2H2"],
    "C2H4_elevated": ["C2H4"], "C2H6_elevated": ["C2H6"],
    "TDCG_elevated": ["CH4", "C2H2", "C2H4", "C2H6"], "gas_rate_rapid": ["C2H2"],
}
NON_GAS_SYMPTOMS = [s for s in SYMPTOM_IDS if s not in GAS_DERIVED_SYMPTOMS]


def load_real(path: Path = SRC) -> list[dict]:
    assert_not_synthetic(path, purpose="d12_source")
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("fault_label") == "normal":
                continue
            g = r.get("dga_data") or {}
            if any(g.get(k) is None for k in GASES):
                continue
            recs.append(r)
    return recs


def make_sample(rec: dict, level: str, masked: list[str], sim_id: str) -> dict[str, Any]:
    full = {k: float(rec["dga_data"][k]) for k in GASES}
    visible = {k: (None if k in masked else full[k]) for k in GASES}
    init_ev = derive_gas_evidence(visible, negative=True)
    full_ev = derive_gas_evidence(full, negative=True)
    hidden_ev = {s: v for s, v in full_ev.items() if s not in init_ev}
    return {
        "sim_id": sim_id,
        "source_record_id": rec["record_id"],
        "source": rec.get("source", ""),
        "level": level,
        "fault_label": rec["fault_label"],
        "severity": rec.get("severity", ""),
        "full_gases": full,
        "visible_gases": visible,
        "masked_gases": masked,
        "initial_evidence": init_ev,
        "hidden_evidence": hidden_ev,
        "unavailable_symptoms": list(NON_GAS_SYMPTOMS),
    }


def stratified_sample(recs: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for r in recs:
        by_label.setdefault(r["fault_label"], []).append(r)
    total = len(recs)
    out: list[dict] = []
    # 按比例分配（最大余数法）
    quota = {k: n * len(v) / total for k, v in by_label.items()}
    base = {k: int(q) for k, q in quota.items()}
    rem = n - sum(base.values())
    for k in sorted(quota, key=lambda k: quota[k] - base[k], reverse=True)[:rem]:
        base[k] += 1
    for k, cnt in base.items():
        pool = list(by_label[k])
        rng.shuffle(pool)
        out.extend(pool[:cnt])
    rng.shuffle(out)
    return out


def build(seed: int = SEED, per_level: int = PER_LEVEL) -> list[dict]:
    recs = load_real()
    rng = random.Random(seed)
    samples: list[dict] = []
    for level, k in LEVELS.items():
        chosen = stratified_sample(recs, per_level, rng)
        for i, rec in enumerate(chosen):
            masked = sorted(rng.sample(GASES, k))
            samples.append(make_sample(rec, level, masked, f"D12-{level[0].upper()}{i:04d}"))
    return samples


def sha256_of(samples: list[dict]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(json.dumps(s, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def write_card(samples: list[dict], recs_n: int, digest: str) -> None:
    labels_all = Counter(s["fault_label"] for s in samples)
    by_level = {lv: Counter(s["fault_label"] for s in samples if s["level"] == lv) for lv in LEVELS}
    src_dist = Counter(r["fault_label"] for r in load_real())
    lines = [
        "# D12 部分观测诊断模拟集数据卡片",
        "",
        f"- 文件：`data/eval/d12/partial_obs.jsonl`",
        f"- 规模：{len(samples)} 条（light / medium / heavy 各 {PER_LEVEL}）；seed {SEED}；sha256 `{digest[:16]}…`",
        f"- 来源：`data/real/dga/dga_records.jsonl` 非 normal 记录 {recs_n} 条（真实文献汇编 DGA，见 `docs/data_inventory.md` R1-R3），"
        "按故障类别分层抽样；同一真实记录可在不同档位重复出现（遮蔽不同），同档内不重复",
        "- 生成：`python3 scripts/sim/partial_obs_sim.py --build`；校验：`--check`",
        "- 真实 / 合成：全部由真实 DGA 派生；`data_guard.assert_not_synthetic` 校验源路径",
        "",
        "## 遮蔽方案",
        "",
        "| 档位 | 遮蔽气体数 | 可见气体数 | 条数 |",
        "|---|---:|---:|---:|",
    ]
    for lv, k in LEVELS.items():
        lines.append(f"| {lv} | {k} | {5 - k} | {sum(by_level[lv].values())} |")
    lines += [
        "",
        "被遮蔽气体在 `visible_gases` 中为 `null`；`initial_evidence` 只含可见气体派生的征兆（含负观测），"
        "`hidden_evidence` 为揭示被遮蔽气体后可获得的征兆。CO / CO2 / 油温 / 绕组温度 / 负载 / 振动 / 局放告警在原始数据中无字段，"
        "`PartialObsEnv.reveal` 返回 `None`（策略应视为「不可获取」并计成本）。",
        "",
        "## 标签分布（与源数据对比）",
        "",
        "| 故障 | 源数据 | 源占比 | D12 全部 | D12 占比 | light | medium | heavy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lab in sorted(src_dist):
        lines.append(f"| `{lab}` | {src_dist[lab]} | {src_dist[lab] / recs_n:.1%} | {labels_all[lab]} | {labels_all[lab] / len(samples):.1%}"
                     f" | {by_level['light'][lab]} | {by_level['medium'][lab]} | {by_level['heavy'][lab]} |")
    lines += [
        "",
        "## 用途",
        "",
        "- B-5 `eval_active_planning.py`：五种问询策略在同一初始观测下逐步揭示征兆，比较停止时准确率与问询次数 / 成本。",
        "- E-3 鲁棒性：mode2 vs mode3 在 heavy 档上的端到端差异。",
        "",
        "## 局限",
        "",
        "- 标签来自文献汇编，不含设备身份与现场征兆；现场征兆（温度 / 负载 / 振动 / 局放）无法揭示，主动规划评测只能在气体征兆空间内进行。",
        "- 遮蔽为均匀随机，不模拟真实工况下「某些气体更常缺失」的偏置。",
    ]
    (OUT_DIR / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# 环境接口
# ──────────────────────────────────────────────────────────────

class PartialObsEnv:
    """
    单条模拟样例的交互环境。

    reveal(symptom) -> True | False | None
        - 征兆可由被遮蔽气体派生：返回真值，并把对应气体加入可见集合、累计成本；
        - 征兆已可见：返回当前值，不计成本；
        - 征兆在该记录中不可获取（现场征兆）：返回 None，仍计一次成本（策略为此付出了代价）。
    reveal_gas(gas) -> float | None  直接揭示某种气体浓度（供 ask_user 追问气体数值的策略）。
    """

    def __init__(self, sample: dict[str, Any], cost_table: Optional[dict[str, Any]] = None) -> None:
        self.sample = sample
        self.visible: dict[str, Optional[float]] = dict(sample["visible_gases"])
        self.full: dict[str, float] = dict(sample["full_gases"])
        self.evidence: dict[str, bool] = dict(sample["initial_evidence"])
        self.label: str = sample["fault_label"]
        self.history: list[dict[str, Any]] = []
        self.total_cost: float = 0.0
        self.n_queries: int = 0
        if cost_table is None:
            from src.tools.eig import load_cost_table
            cost_table = load_cost_table()
        self._cost = {s: float(v.get("cost", 1)) for s, v in cost_table["symptoms"].items()}

    @property
    def masked_remaining(self) -> list[str]:
        return [g for g, v in self.visible.items() if v is None]

    def _refresh_evidence(self) -> None:
        derived = derive_gas_evidence(self.visible, negative=True)
        for s, v in derived.items():
            self.evidence[s] = v

    def reveal_gas(self, gas: str, cost: float = 0.0) -> Optional[float]:
        if gas not in self.full:
            return None
        if self.visible.get(gas) is None:
            self.visible[gas] = self.full[gas]
            self._refresh_evidence()
            self.n_queries += 1
            self.total_cost += cost
            self.history.append({"type": "gas", "target": gas, "value": self.full[gas], "cost": cost})
        return self.visible[gas]

    def reveal(self, symptom: str) -> Optional[bool]:
        cost = self._cost.get(symptom, 1.0)
        if symptom in self.evidence:
            return self.evidence[symptom]
        needed = _SYMPTOM_GASES.get(symptom)
        if needed is None:
            # 现场征兆：本数据集无字段
            self.n_queries += 1
            self.total_cost += cost
            self.history.append({"type": "symptom", "target": symptom, "value": None, "cost": cost})
            return None
        for g in needed:
            if self.visible.get(g) is None:
                self.visible[g] = self.full[g]
        self._refresh_evidence()
        self.n_queries += 1
        self.total_cost += cost
        val = self.evidence.get(symptom)
        self.history.append({"type": "symptom", "target": symptom, "value": val, "cost": cost,
                             "revealed_gases": needed})
        return val

    def context(self) -> dict[str, Any]:
        """当前可见气体（缺失为 None），供 fault_attribution / eig 使用。"""
        return dict(self.visible)


def load_d12(path: Path = OUT, level: Optional[str] = None, limit: int = 0) -> list[dict]:
    assert_not_synthetic(path, purpose="d12")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if level and s["level"] != level:
                continue
            out.append(s)
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.build:
        samples = build(args.seed)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        digest = sha256_of(samples)
        write_card(samples, len(load_real()), digest)
        (OUT_DIR / "build_stats.json").write_text(json.dumps({
            "seed": args.seed, "n": len(samples), "sha256": digest,
            "levels": {lv: sum(1 for s in samples if s["level"] == lv) for lv in LEVELS},
            "labels": dict(Counter(s["fault_label"] for s in samples)),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写出 {OUT}（{len(samples)} 条）sha256={digest[:16]}…")

    if args.check:
        on_disk = load_d12()
        rebuilt = build(args.seed)
        same = sha256_of(on_disk) == sha256_of(rebuilt)
        print(f"可复现: {'OK' if same else 'MISMATCH'}  n={len(on_disk)}")
        src = Counter(r["fault_label"] for r in load_real())
        n_src = sum(src.values())
        got = Counter(s["fault_label"] for s in on_disk)
        max_dev = max(abs(got[k] / len(on_disk) - src[k] / n_src) for k in src)
        print(f"标签分布最大偏差: {max_dev:.3%}")
        for lv in LEVELS:
            sub = [s for s in on_disk if s["level"] == lv]
            print(f"  {lv}: n={len(sub)} 平均隐藏征兆数={sum(len(s['hidden_evidence']) for s in sub) / len(sub):.2f}")
        # 环境自检
        env = PartialObsEnv(on_disk[0])
        g0 = env.masked_remaining[0]
        sym = f"{g0}_elevated"
        v = env.reveal(sym)
        assert v is not None and env.n_queries == 1 and env.total_cost == 0.0, "reveal 气体征兆应返回真值且成本 0"
        assert env.reveal("oil_temp_elevated") is None and env.total_cost == 1.0, "现场征兆应返回 None 并计成本 1"
        print("PartialObsEnv 自检 OK")
        if not same or max_dev > 0.02:
            sys.exit(1)


if __name__ == "__main__":
    main()
