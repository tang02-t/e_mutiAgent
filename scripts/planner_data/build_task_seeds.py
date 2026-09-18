#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4 规划模块数据构建（离线部分）：任务种子 + 金标动作。

不依赖 LLM。产出的是「结构化种子」：每条含模板化查询、五类标签、金标动作（工具名+参数 / direct_answer / ask_user）、
seed_source（用于分组切分）与 expected_hint（金标块 / 金标实体 / 原始标签）。自然语言扩写（口语化、模拟用户）留给 LLM 阶段。

数据来源（均为真实数据或由真实文献派生）：
- 事实类      ← data/kb/eval/retrieval_seed.jsonl（金标块）
- 推理类      ← data/kg/eval/kg_qa_seed.jsonl（1 跳）+ data/kg/graph.json（2 跳链路）
- 数值工具类  ← data/real/dga/dga_records.jsonl（R4 原始标签）、data/raw/ETT-small/*.csv（预测 + 异常检测窗口）
- 无需工具类  ← 手写常识/闲聊模板
- 信息不足类  ← 手写缺参数模板（缺 DGA / 缺序列 / 缺设备语境）
- 复合类      ← DGA+文献 / DGA+图谱 / 预测+异常 的多步金标（供 D7/D8 多工具样本）

运行：
  python3 scripts/planner_data/build_task_seeds.py            # 生成 + 参数校验
  python3 scripts/planner_data/build_task_seeds.py --execute  # 额外在真实工具环境执行金标动作并剔除失败样本
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tools.tool_registry import validate_arguments_strict  # noqa: E402

OUT_DIR = ROOT / "data/planner/seeds"
KB_SEED = ROOT / "data/kb/eval/retrieval_seed.jsonl"
KB_SPLIT = ROOT / "data/kb/eval/split.json"
KG_SEED = ROOT / "data/kg/eval/kg_qa_seed.jsonl"
KG_GRAPH = ROOT / "data/kg/graph.json"
DGA_REAL = ROOT / "data/real/dga/dga_records.jsonl"
ETT_DIR = ROOT / "data/raw/ETT-small"

CATEGORY_TARGET = {"fact": 0.35, "reasoning": 0.15, "numeric_tool": 0.30, "no_tool": 0.10, "insufficient": 0.10}
DEVICE_POOL = ["220kV 主变 #1", "110kV 主变 #2", "500kV 1 号主变", "35kV 站用变", "某 220kV 变电站 2 号主变",
               "SFSZ-240000/220 型主变", "SSZ-31500/110 型主变", "某厂 10000kVA 电力变压器"]


def _jl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _sid(*parts: Any) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]


def _seed(category: str, query: str, actions: List[Dict[str, Any]], seed_source: str, group: str,
          expected_hint: Dict[str, Any], sub_type: str, note: str = "") -> Dict[str, Any]:
    return {
        "seed_id": f"S-{category[:3].upper()}-{_sid(category, query, seed_source)}",
        "category": category,
        "sub_type": sub_type,
        "query": query,
        "gold_actions": actions,          # 按顺序执行的动作列表；多步表示多工具
        "seed_source": seed_source,
        "group_key": group,               # 分组切分键：同组只进一个 split
        "expected_hint": expected_hint,
        "needs_llm_rewrite": True,        # 所有种子均需 LLM 口语化/多样化扩写
        "note": note,
    }


def tool_action(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "tool_call", "tool": tool, "arguments": arguments}


def direct_answer(reason: str) -> Dict[str, Any]:
    return {"type": "direct_answer", "reason": reason}


def ask_user(missing: List[str], reason: str) -> Dict[str, Any]:
    return {"type": "ask_user", "missing": missing, "reason": reason}


# ──────────────────────────────────────────────────────────────
# 1. 事实类（rag_search）
# ──────────────────────────────────────────────────────────────
FACT_TEMPLATES = {
    "A_title": ["{q}的相关规定和做法是什么？", "请介绍一下{q}。", "关于{q}，文献中有哪些要点？", "{q}应如何进行？"],
    "B_caption": ["请解释{q}所反映的内容。", "{q}说明了什么？"],
    "C_definition": ["{q}？", "{q}，请给出定义和说明。"],
}


def build_fact(rng: random.Random) -> List[Dict[str, Any]]:
    rows = _jl(KB_SEED)
    out = []
    for r in rows:
        tmpls = FACT_TEMPLATES.get(r["type"], ["{q}"])
        q0 = r["query"].strip()
        # B 类图注本身像标题，去掉"图N"前缀噪声
        q = rng.choice(tmpls).format(q=q0)
        out.append(_seed(
            "fact", q, [tool_action("rag_search", {"query": q0})],
            seed_source=f"chunk:{r['gold_chunk_ids'][0]}", group=f"doc:{r['gold_doc_id']}",
            expected_hint={"gold_chunk_ids": r["gold_chunk_ids"], "gold_doc_id": r["gold_doc_id"],
                           "kb_split": r.get("split")},
            sub_type=r["type"], note="rag_search.query 为规范化短语，扩写后应改为口语问题但保留检索意图"))
    return out


# ──────────────────────────────────────────────────────────────
# 2. 推理类（kg_search，1 跳 + 2 跳）
# ──────────────────────────────────────────────────────────────
TWO_HOP_TEMPLATES = {
    ("CAUSES", "TREATED_BY"): "{x}引起的故障应如何处理？",
    ("CAUSES", "LOCATED_IN"): "{x}引起的故障通常发生在变压器的哪个部位？",
    ("CAUSES", "DETECTED_BY"): "{x}导致的故障可以通过什么方法检测出来？",
    ("CAUSES", "PRODUCES"): "{x}导致的故障会使油中哪些气体升高？",
    ("INDICATES", "TREATED_BY"): "出现{x}时，对应故障该怎样处理？",
    ("INDICATES", "LOCATED_IN"): "出现{x}，问题一般出在哪个部件？",
    ("INDICATES", "CAUSES"): "出现{x}，其背后的故障还可能进一步引发什么？",
}


def build_reasoning(rng: random.Random) -> List[Dict[str, Any]]:
    out = []
    for r in _jl(KG_SEED):
        out.append(_seed(
            "reasoning", r["question"],
            [tool_action("kg_search", {"query": r["anchor_surface"], "relations": [r["relation"]],
                                       "direction": r["direction"], "hops": 1})],
            seed_source=f"kgqa:{r['qa_id']}", group=f"entity:{r['anchor_entity']}",
            expected_hint={"expected_entities": r["expected_entities"], "relation": r["relation"],
                           "kg_split": r.get("split")},
            sub_type=f"kg_1hop_{r['surface_kind']}"))

    g = json.load(open(KG_GRAPH, encoding="utf-8"))
    name = {n["id"]: n["name"] for n in g["nodes"]}
    out_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in g["edges"]:
        out_edges[e["source"]].append(e)
    seen = set()
    for a, es in out_edges.items():
        for e1 in es:
            for e2 in out_edges.get(e1["target"], []):
                key = (e1["relation"], e2["relation"])
                if key not in TWO_HOP_TEMPLATES or e2["target"] == a:
                    continue
                if (a, key) in seen:
                    continue
                seen.add((a, key))
                x = name[a]
                out.append(_seed(
                    "reasoning", TWO_HOP_TEMPLATES[key].format(x=x),
                    [tool_action("kg_search", {"query": x, "relations": list(key), "hops": 2, "direction": "out"})],
                    seed_source=f"kg2hop:{a}:{key[0]}:{key[1]}", group=f"entity:{a}",
                    expected_hint={"path_prefix": [a, key[0], e1["target"], key[1]],
                                   "support": e1["support_count"] + e2["support_count"]},
                    sub_type="kg_2hop"))
    return out


# ──────────────────────────────────────────────────────────────
# 3. 数值工具类
# ──────────────────────────────────────────────────────────────
DGA_TEMPLATES = [
    "{dev}油色谱检测：{gas}，请诊断故障类型并给出处理建议。",
    "{dev}最近一次油中溶解气体分析结果为 {gas}（μL/L），判断一下可能是什么故障？",
    "油色谱数据 {gas}，这台变压器哪种故障可能性最大？",
    "{dev}色谱：{gas}。是否需要停运检查？",
]
GAS_KEYS = ["H2", "CH4", "C2H2", "C2H4", "C2H6"]


def _fmt_gas(d: Dict[str, float], rng: random.Random) -> Tuple[str, Dict[str, float]]:
    keys = [k for k in GAS_KEYS if k in d]
    if rng.random() < 0.2:
        keys = [k for k in keys if k != "C2H6"]  # 部分样本省略 C2H6，训练缺字段容错
    vals = {k: round(float(d[k]), 2) for k in keys}
    return ", ".join(f"{k}={v}" for k, v in vals.items()), vals


def build_dga(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    rows = _jl(DGA_REAL)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[r["fault_label"]].append(r)
    per = max(1, n // len(by_label))
    out = []
    for lab, rs in by_label.items():
        rng.shuffle(rs)
        for r in rs[:per]:
            gas_str, vals = _fmt_gas(r["dga_data"], rng)
            dev = rng.choice(DEVICE_POOL)
            q = rng.choice(DGA_TEMPLATES).format(dev=dev, gas=gas_str)
            out.append(_seed(
                "numeric_tool", q, [tool_action("fault_attribution", {"dga_data": vals, "query": q})],
                seed_source=f"dga:{r['record_id']}", group=f"device:{r['device_id']}",
                expected_hint={"raw_label": r["fault_label"], "raw_label_cn": r["fault_label_cn"],
                               "severity": r["severity"], "source": r["source"]},
                sub_type="dga_attribution",
                note="raw_label 为数据集原始标签（R4），不是 fault_attribution 的输出"))
    return out


ETT_RANGE = ("2016-07-01", "2018-06-26")
HORIZON_NL = [  # (自然语言, 小时数)
    ("未来 6 小时", 6), ("未来 12 小时", 12), ("未来 24 小时", 24), ("未来一天", 24), ("未来 48 小时", 48), ("未来 3 天", 72),
]
ETT_TEMPLATES = [
    "请基于 {ds} 数据集预测该变压器{hz}的油温走势，并判断是否存在过热风险。",
    "用 {ds} 的历史负载与油温数据，预测{hz}油温。",
    "{ds} 数据：以 {end} 为截止时间，预测{hz}的顶层油温。",
    "基于 {ds}（{start} 至 {end}）的数据，油温{hz}会不会继续上升？",
]


def _rand_range(rng: random.Random) -> Tuple[str, str]:
    y = rng.choice([2016, 2017, 2018])
    m = rng.randint(7, 12) if y == 2016 else (rng.randint(1, 5) if y == 2018 else rng.randint(1, 12))
    d = rng.randint(1, 20)
    start = f"{y}-{m:02d}-{d:02d}"
    end = f"{y}-{m:02d}-{min(d + rng.randint(5, 8), 28):02d}"
    return start, end


def build_ett(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    out = []
    for _ in range(n):
        ds = rng.choice(["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
        hz_nl, hours = rng.choice(HORIZON_NL)
        horizon = hours if ds.startswith("ETTh") else hours * 4  # 15 分钟级换算
        tmpl = rng.choice(ETT_TEMPLATES)
        start, end = _rand_range(rng)
        args: Dict[str, Any] = {"dataset": ds, "horizon": horizon}
        if "{end}" in tmpl:
            args["end_time"] = end
        if "{start}" in tmpl:
            args["start_time"] = start
        q = tmpl.format(ds=ds, hz=hz_nl, start=start, end=end)
        out.append(_seed(
            "numeric_tool", q, [tool_action("ett_forecast", args)],
            seed_source=f"ett:{ds}:{end if 'end_time' in args else 'latest'}:{horizon}",
            group=f"ett:{ds}:{(args.get('end_time') or 'latest')[:7]}",
            expected_hint={"hours": hours, "freq": "1h" if ds.startswith("ETTh") else "15min"},
            sub_type="ett_forecast",
            note="horizon 需按数据集采样间隔换算：ETTh 小时级、ETTm 15 分钟级（6 小时 → 6 / 24）"))
    return out


ANOM_TEMPLATES = [
    "以下是{dev}最近 {n} 个采样点的{sig}读数：{seq}。请检查其中是否有异常点。",
    "{sig}序列 {seq}，帮我做一次 3σ 异常筛查。",
    "这段{sig}数据正常吗？{seq}",
]
SIGNAL_NAMES = [("OT", "顶层油温"), ("HUFL", "高压侧有功负载"), ("MUFL", "中压侧有功负载"), ("LUFL", "低压侧有功负载")]


def _ett_column(ds: str, col: str) -> List[float]:
    with open(ETT_DIR / f"{ds}.csv", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return [float(r[col]) for r in rd]


def build_anomaly(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    out = []
    cols = {ds: {c: _ett_column(ds, c) for c, _ in SIGNAL_NAMES} for ds in ["ETTh1", "ETTh2"]}
    for _ in range(n):
        ds = rng.choice(["ETTh1", "ETTh2"])
        col, cn = rng.choice(SIGNAL_NAMES)
        series = cols[ds][col]
        L = rng.choice([12, 16, 20, 24, 30])
        s = rng.randint(0, len(series) - L - 1)
        window = [round(v, 2) for v in series[s:s + L]]
        if rng.random() < 0.4:  # 注入一个明显离群点，保证部分样本确有异常
            k = rng.randrange(L)
            window[k] = round(window[k] * rng.choice([2.5, 3.0, 0.2]), 2)
        seq = "[" + ", ".join(str(v) for v in window) + "]"
        dev = rng.choice(DEVICE_POOL)
        q = rng.choice(ANOM_TEMPLATES).format(dev=dev, n=L, sig=cn, seq=seq)
        out.append(_seed(
            "numeric_tool", q, [tool_action("timeseries_anomaly", {"signal": window})],
            seed_source=f"ettwin:{ds}:{col}:{s}:{L}", group=f"ettwin:{ds}:{s // 2000}",
            expected_hint={"signal_name": cn, "n_points": L},
            sub_type="timeseries_anomaly",
            note="signal 必须逐字来自用户给出的序列；对比样本：无序列 → 不可调用（见 insufficient）"))
    return out


# ──────────────────────────────────────────────────────────────
# 4. 无需工具类 / 5. 信息不足类（手写模板）
# ──────────────────────────────────────────────────────────────
NO_TOOL = [
    ("你好", "greeting"), ("你是谁？能做什么？", "capability"), ("谢谢，辛苦了", "greeting"),
    ("你能帮我诊断变压器故障吗？", "capability"), ("你支持哪些数据输入？", "capability"),
    ("μL/L 和 ppm 是一样的单位吗？", "common_sense"), ("1 mL/L 等于多少 μL/L？", "common_sense"),
    ("变压器的基本工作原理是什么？", "common_sense"), ("什么是油浸式变压器？", "common_sense"),
    ("ETTh1 和 ETTm1 数据集有什么区别？", "capability"), ("三比值法大概是什么思路？一句话说明。", "common_sense"),
    ("DGA 是什么的缩写？", "common_sense"), ("TDCG 指的是什么？", "common_sense"),
    ("你刚才的结论是基于哪些工具得出的？", "meta"), ("请把上面的建议整理成三条要点。", "meta"),
    ("变压器铭牌上的 SFSZ 代表什么含义？", "common_sense"), ("kV 与 V 如何换算？", "common_sense"),
    ("今天天气怎么样？", "out_of_scope"), ("帮我写一封请假邮件。", "out_of_scope"), ("你的回答可靠吗？", "meta"),
    ("变压器油的主要作用是什么？", "common_sense"), ("什么叫三比值法中的编码？", "common_sense"),
    ("氢气在油中升高一般意味着什么？一句话。", "common_sense"), ("什么是局部放电？简单解释。", "common_sense"),
    ("变压器分接开关是干什么用的？", "common_sense"), ("轻瓦斯和重瓦斯有什么区别？", "common_sense"),
    ("你能读取 Excel 里的色谱数据吗？", "capability"), ("上面的诊断结果置信度是多少？", "meta"),
    ("变压器的额定容量单位为什么是 kVA 不是 kW？", "common_sense"), ("能不能用英文回答？", "meta"),
]
INSUFFICIENT = [
    ("请判断{dev}的故障类型。", ["dga_data 或征兆描述"], "fault_attribution 需要 DGA 数据或征兆，问题未提供且上下文为空"),
    ("{dev}油色谱有点异常，是什么故障？", ["具体气体浓度"], "仅说“异常”无数值，无法归因"),
    ("帮我看看这段电流波形有没有异常。", ["signal 数值序列"], "timeseries_anomaly 需要序列，用户未给出"),
    ("对下面的振动数据做异常检测。", ["signal 数值序列"], "提到“下面的数据”但实际未附带"),
    ("乙炔涨了，严重吗？", ["C2H2 浓度及其他气体数值", "产气速率"], "缺少数值无法判定严重程度"),
    ("{dev}温度偏高，能预测一下吗？", ["历史油温/负载数据或指定 ETT 数据集"], "无历史数据来源且未指定数据集时应追问；若用户接受演示数据可用 ETTh1"),
    ("这台变压器要不要停运？", ["故障类型或检测数据"], "决策类问题缺任何数据支撑"),
    ("请诊断。", ["设备信息", "检测数据"], "空指令"),
    ("色谱数据如下，请分析。", ["气体浓度数值"], "声称附带数据但为空"),
    ("{dev}的负载率高吗？", ["负载数据或额定容量"], "缺数据"),
    ("{dev}总烃超标了，算哪类故障？", ["各组分气体浓度"], "仅有总烃无法用三比值法"),
    ("这组数据 H2 偏高，其他正常，什么问题？", ["H2 及其他气体的具体数值"], "定性描述无数值"),
    ("帮我预测一下油温。", ["数据集或历史数据", "预测时长"], "缺数据来源与 horizon"),
    ("{dev}有异响，是什么故障？", ["色谱数据、电气试验结果或更多现象描述"], "仅一个症状，需追问；可选提示做 kg_search 但不足以归因"),
]
# 对比样本：与 insufficient 同问法，但上下文里带了 DGA → 必须调用 fault_attribution 并填入 dga_data
INSUFFICIENT_DGA_IDX = [0, 1, 4, 7, 8, 10]


OPENERS = ["", "请问", "麻烦问一下，", "想确认一下：", "师傅，", "你好，", "快速问个问题：", "打扰了，"]
CLOSERS = ["", "谢谢。", "尽量简短。", "急，在线等。", "麻烦了。"]


def _vary(rng: random.Random, q: str) -> str:
    o, c = rng.choice(OPENERS), rng.choice(CLOSERS)
    return f"{o}{q}{c}".strip()


def build_no_tool(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        i = rng.randrange(len(NO_TOOL))
        q0, st = NO_TOOL[i]
        q = _vary(rng, q0) if len(out) >= len(NO_TOOL) else q0
        if q in seen:
            continue
        seen.add(q)
        out.append(_seed("no_tool", q, [direct_answer(f"{st}：无需工具，直接回答或说明能力范围")],
                         seed_source=f"manual:no_tool:{i}", group=f"manual:no_tool:{i}",
                         expected_hint={"sub_type": st, "base_query": q0}, sub_type=st,
                         note="扩写时保持无需工具属性"))
    return out


def build_insufficient(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        i = rng.randrange(len(INSUFFICIENT))
        t, missing, reason = INSUFFICIENT[i]
        q0 = t.format(dev=rng.choice(DEVICE_POOL))
        q = _vary(rng, q0) if len(out) >= len(INSUFFICIENT) else q0
        if q in seen:
            continue
        seen.add(q)
        out.append(_seed("insufficient", q, [ask_user(missing, reason)],
                         seed_source=f"manual:insufficient:{i}", group=f"manual:insufficient:{i}",
                         expected_hint={"missing": missing, "base_template": t}, sub_type="ask_missing_params",
                         note="上下文 context 应为空；对比样本为同问题 + 上下文含 DGA 时应调用工具"))
    return out


# ──────────────────────────────────────────────────────────────
# 6. 复合类（多工具，供 D7/D8）
# ──────────────────────────────────────────────────────────────
def build_context_contrast(rng: random.Random, dga_seeds: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    对比样本：问法与 insufficient 相同，但结构化上下文 context.dga 含数据 →
    金标为 fault_attribution 且 dga_data 取自上下文（对应 Planner 系统提示“上下文若已给出 DGA 应填入”）。
    """
    out = []
    pool = dga_seeds[:]
    rng.shuffle(pool)
    for i, s in enumerate(pool[:n]):
        t, _, _ = INSUFFICIENT[INSUFFICIENT_DGA_IDX[i % len(INSUFFICIENT_DGA_IDX)]]
        q = t.format(dev=rng.choice(DEVICE_POOL))
        dga_vals = s["gold_actions"][0]["arguments"]["dga_data"]
        sd = _seed("numeric_tool", q, [tool_action("fault_attribution", {"dga_data": dga_vals, "query": q})],
                   seed_source=s["seed_source"], group=s["group_key"],
                   expected_hint=dict(s["expected_hint"], contrast_of="insufficient"),
                   sub_type="dga_from_context",
                   note="问题本身无数值，dga_data 必须从 context.dga 填入；与 insufficient 类同问法形成对比")
        sd["context"] = {"device_id": s["group_key"].split(":", 1)[1], "dga": dga_vals}
        out.append(sd)
    return out


def build_composite(rng: random.Random, dga_seeds: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out = []
    pool = [s for s in dga_seeds if s["expected_hint"].get("raw_label") != "normal"]
    rng.shuffle(pool)
    kinds = ["dga_rag", "dga_kg", "ett_anomaly"]
    for i, s in enumerate(pool[:n]):
        kind = kinds[i % 3]
        dga_args = s["gold_actions"][0]["arguments"]
        gas = ", ".join(f"{k}={v}" for k, v in dga_args["dga_data"].items())
        dev = rng.choice(DEVICE_POOL)
        if kind == "dga_rag":
            q = f"{dev}油色谱 {gas}，请判断故障类型，并结合规程给出检修处理建议。"
            acts = [tool_action("fault_attribution", {"dga_data": dga_args["dga_data"], "query": q}),
                    tool_action("rag_search", {"query": f"{s['expected_hint']['raw_label_cn']} 检修处理 规程"})]
            hint = {"raw_label_cn": s["expected_hint"]["raw_label_cn"],
                    "note": "第 2 步 query 应替换为第 1 步返回的 primary_fault_name（此处用原始标签占位）"}
        elif kind == "dga_kg":
            q = f"{dev}油色谱 {gas}，最可能是什么故障？该故障一般出在哪个部件、怎么处理？"
            acts = [tool_action("fault_attribution", {"dga_data": dga_args["dga_data"], "query": q}),
                    tool_action("kg_search", {"query": s["expected_hint"]["raw_label_cn"],
                                              "relations": ["LOCATED_IN", "TREATED_BY"], "direction": "out"})]
            hint = {"raw_label_cn": s["expected_hint"]["raw_label_cn"],
                    "note": "第 2 步 query 应替换为第 1 步返回的 primary_fault_name"}
        else:
            ds = rng.choice(["ETTh1", "ETTh2"])
            hz_nl, hours = rng.choice(HORIZON_NL[:4])
            q = f"基于 {ds} 数据集预测{hz_nl}油温，并检查历史油温中是否存在异常点。"
            acts = [tool_action("ett_forecast", {"dataset": ds, "horizon": hours})]
            hint = {"note": "ett_forecast 自带 3σ 异常检测，无需再调 timeseries_anomaly；对比样本用于学习“不多调工具”"}
        out.append(_seed("composite", q, acts, seed_source=s["seed_source"], group=s["group_key"],
                         expected_hint=hint, sub_type=kind))
    return out


# ──────────────────────────────────────────────────────────────
# 校验 / 执行 / 切分 / 报告
# ──────────────────────────────────────────────────────────────
def validate_all(seeds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ok, bad = [], []
    for s in seeds:
        errs = []
        for a in s["gold_actions"]:
            if a["type"] != "tool_call":
                continue
            v = validate_arguments_strict(a["tool"], a["arguments"])
            if not v["valid"]:
                errs.append({"tool": a["tool"], "errors": v["errors"]})
        if errs:
            s["_invalid"] = errs
            bad.append(s)
        else:
            ok.append(s)
    return ok, bad


def execute_all(seeds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """在真实工具环境执行金标动作；业务失败（error/no_data/no_match/no_relation/空检索）的样本剔除。"""
    from src.tools.fault_attribution import fault_attribution
    from src.tools.kg_search import kg_search
    from src.tools.ett_forecasting import ett_forecast
    from src.tools.local_kb import local_kb_search
    from src.tools.timeseries import timeseries_anomaly

    fns = {"fault_attribution": fault_attribution, "kg_search": kg_search, "ett_forecast": ett_forecast,
           "rag_search": local_kb_search, "timeseries_anomaly": timeseries_anomaly}
    bad_status = {"error", "no_data", "failed", "no_match", "no_relation"}
    ok, bad = [], []
    cache: Dict[str, Any] = {}
    for s in seeds:
        fail = None
        for a in s["gold_actions"]:
            if a["type"] != "tool_call":
                continue
            key = a["tool"] + json.dumps(a["arguments"], sort_keys=True, ensure_ascii=False)
            if key not in cache:
                try:
                    cache[key] = fns[a["tool"]](**a["arguments"]) if a["tool"] != "fault_attribution" \
                        else fns[a["tool"]](dga_data=a["arguments"].get("dga_data"), query=a["arguments"].get("query", ""))
                except Exception as exc:  # noqa: BLE001
                    cache[key] = {"status": "error", "message": str(exc)}
            r = cache[key]
            if isinstance(r, dict):
                st = r.get("status", "ok")
                if st in bad_status:
                    fail = {"tool": a["tool"], "status": st, "message": str(r.get("message") or r.get("summary"))[:120]}
            elif isinstance(r, list):
                if not r:
                    fail = {"tool": a["tool"], "status": "empty"}
            if fail:
                break
        if fail:
            s["_exec_fail"] = fail
            bad.append(s)
        else:
            s["exec_verified"] = True
            ok.append(s)
    return ok, bad


def assign_split(seeds: List[Dict[str, Any]], rng: random.Random) -> None:
    """按 group_key 分组切分 70/10/20。事实/推理类沿用知识库、图谱既有 split（保持 doc 级一致）。"""
    kb_split = json.load(open(KB_SPLIT, encoding="utf-8"))
    doc2split = {d: sp for sp, ds in kb_split.items() for d in ds}
    groups = sorted({s["group_key"] for s in seeds})
    rng.shuffle(groups)
    fixed: Dict[str, str] = {}
    for s in seeds:
        g = s["group_key"]
        if g.startswith("doc:") and g[4:] in doc2split:
            fixed[g] = doc2split[g[4:]]
        elif s["category"] == "reasoning" and s["expected_hint"].get("kg_split"):
            fixed.setdefault(g, s["expected_hint"]["kg_split"])
    rest = [g for g in groups if g not in fixed]
    n = len(rest)
    plan = {}
    for i, g in enumerate(rest):
        plan[g] = "train" if i < n * 0.7 else ("dev" if i < n * 0.8 else "test")
    plan.update(fixed)
    for s in seeds:
        s["split"] = plan[s["group_key"]]


def _tok(q: str) -> set:
    return set(q.replace(" ", "")) if len(q) < 8 else {q[i:i + 2] for i in range(len(q) - 1)}


def leakage_check(seeds: List[Dict[str, Any]], thr: float = 0.9) -> List[Tuple[str, str, float]]:
    """训练集 vs 测试集查询 bigram-Jaccard 相似度 > thr 的跨集对。数值类因数字不同天然不重复，仅比对非数值类。"""
    tr = [s for s in seeds if s["split"] == "train" and s["category"] in ("fact", "reasoning", "no_tool", "insufficient")]
    te = [s for s in seeds if s["split"] == "test" and s["category"] in ("fact", "reasoning", "no_tool", "insufficient")]
    te_tok = [(s, _tok(s["query"])) for s in te]
    pairs = []
    for a in tr:
        ta = _tok(a["query"])
        for b, tb in te_tok:
            j = len(ta & tb) / max(1, len(ta | tb))
            if j > thr:
                pairs.append((a["seed_id"], b["seed_id"], round(j, 3)))
    return pairs


def report(seeds: List[Dict[str, Any]], bad_v: List, bad_e: List, leaks: List, executed: bool) -> str:
    n = len(seeds)
    cat = Counter(s["category"] for s in seeds)
    sub = Counter((s["category"], s["sub_type"]) for s in seeds)
    sp = Counter(s["split"] for s in seeds)
    tools = Counter(a["tool"] for s in seeds for a in s["gold_actions"] if a["type"] == "tool_call")
    acts = Counter("multi_tool" if sum(a["type"] == "tool_call" for a in s["gold_actions"]) > 1
                   else s["gold_actions"][0]["type"] for s in seeds)
    L = ["# P4 任务种子（D6 种子 + 金标动作）构建报告\n"]
    L.append(f"- 种子总数：{n}（参数校验剔除 {len(bad_v)}，真实执行剔除 {len(bad_e) if executed else '未执行'}）")
    L.append("- 五类占比（目标 fact/reasoning/numeric/no_tool/insufficient ≈ 35/15/30/10/10，composite 额外供 D7/D8）：")
    core = sum(v for k, v in cat.items() if k != "composite")
    for k in ["fact", "reasoning", "numeric_tool", "no_tool", "insufficient", "composite"]:
        if cat.get(k):
            pct = cat[k] / core if k != "composite" else 0
            L.append(f"  - {k}: {cat[k]}" + (f"（{pct:.1%}，目标 {CATEGORY_TARGET[k]:.0%}）" if k != "composite" else ""))
    L.append(f"- 动作类型分布：{dict(acts)}")
    L.append(f"- 金标工具分布：{dict(tools)}")
    L.append(f"- 切分（按 group_key 分组，事实类沿用文献 split、推理类沿用图谱 split）：{dict(sp)}")
    L.append(f"- 泄漏检查（train vs test 查询 bigram-Jaccard > 0.9）：{len(leaks)} 对" + ("（已剔除测试侧）" if leaks else ""))
    L.append("")
    L.append("## 子类分布")
    L.append("| 类别 | 子类 | 数量 |")
    L.append("|---|---|---|")
    for (c, st), v in sorted(sub.items()):
        L.append(f"| {c} | {st} | {v} |")
    if bad_v:
        L.append("\n## 参数校验失败样例（前 5）")
        for s in bad_v[:5]:
            L.append(f"- {s['seed_id']} {s['_invalid']}")
    if executed and bad_e:
        L.append("\n## 真实执行失败样例（前 10）")
        for s in bad_e[:10]:
            L.append(f"- {s['seed_id']} [{s['category']}/{s['sub_type']}] {s['_exec_fail']}")
    L.append("\n## 说明")
    L.append("- 所有 query 为模板化文本，`needs_llm_rewrite=true`：需 LLM 做口语化、省略主语、术语/俗称混用等“模拟用户”改写（P4-2），"
             "改写后金标动作不变。")
    L.append("- `gold_actions` 中 `tool_call` 的 arguments 已通过 `validate_arguments_strict`；`--execute` 模式下还在真实工具环境执行并剔除业务失败样本。")
    L.append("- composite 类第 2 步的 kg_search/rag_search query 以原始标签占位，构造 D8 多轮样本时应替换为第 1 步真实返回的 primary_fault_name。")
    L.append("- 事实类 expected_hint.gold_chunk_ids、推理类 expected_entities 供后续端到端评测使用，不进入 Planner 训练标签。")
    L.append("- 数值类 raw_label 来自数据集原始标签（R4），与 fault_attribution 输出可能不一致，只作回答评估参考。")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-dga", type=int, default=520)
    ap.add_argument("--n-ett", type=int, default=200)
    ap.add_argument("--n-anom", type=int, default=160)
    ap.add_argument("--n-context", type=int, default=120, help="上下文含 DGA 的对比样本数")
    ap.add_argument("--n-no-tool", type=int, default=340)
    ap.add_argument("--n-insufficient", type=int, default=340)
    ap.add_argument("--n-composite", type=int, default=240)
    ap.add_argument("--execute", action="store_true", help="在真实工具环境执行金标动作并剔除失败样本")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    fact = build_fact(rng)
    reasoning = build_reasoning(rng)
    dga = build_dga(rng, args.n_dga)
    ett = build_ett(rng, args.n_ett)
    anom = build_anomaly(rng, args.n_anom)
    ctx = build_context_contrast(rng, dga, args.n_context)
    no_tool = build_no_tool(rng, args.n_no_tool)
    insuff = build_insufficient(rng, args.n_insufficient)
    comp = build_composite(rng, dga, args.n_composite)
    seeds = fact + reasoning + dga + ett + anom + ctx + no_tool + insuff + comp

    # 去重（同一 seed_id）
    uniq: Dict[str, Dict[str, Any]] = {}
    for s in seeds:
        uniq.setdefault(s["seed_id"], s)
    seeds = list(uniq.values())

    seeds, bad_v = validate_all(seeds)
    bad_e: List[Dict[str, Any]] = []
    if args.execute:
        seeds, bad_e = execute_all(seeds)

    assign_split(seeds, rng)
    leaks = leakage_check(seeds)
    if leaks:
        drop = {b for _, b, _ in leaks}
        seeds = [s for s in seeds if s["seed_id"] not in drop]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "task_seeds.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "rejected.jsonl", "w", encoding="utf-8") as f:
        for s in bad_v + bad_e:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    rep = report(seeds, bad_v, bad_e, leaks, args.execute)
    (OUT_DIR / "report.md").write_text(rep, encoding="utf-8")
    print(rep)
    print(f"\n已写入 {len(seeds)} 条 → {OUT_DIR / 'task_seeds.jsonl'}")


if __name__ == "__main__":
    main()
