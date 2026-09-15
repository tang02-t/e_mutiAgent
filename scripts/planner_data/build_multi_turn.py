#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4-3 工具调用训练集（D8）多轮与错误恢复样本构建（离线，无需 LLM）。

在 task_seeds.jsonl（单轮金标）基础上，用真实工具返回构造第二轮决策：
  A. multi_turn      composite 种子：第 1 步 fault_attribution 的真实返回 primary_fault_name 替换第 2 步占位 query，
                     形成「user → assistant(tool_call 1) → tool(result 1) → assistant(tool_call 2)」轨迹。
  B. multi_turn_end  单工具种子：工具返回后 assistant 给出「finish」决策（不再调用），学习何时停止。
  C. error_recovery  故意注入可恢复错误并用真实工具返回作为 tool 消息：
       - ett_forecast：dataset 写成 ETTh3 / 时间范围超出 → error → 改参数重试（真实可恢复）
       - kg_search：限定关系下 no_relation → 放宽 relations 重试；no_match（俗称）→ 换标准实体名重试
       - fault_attribution：dga_data 缺组分 → 工具仍 ok，但 assistant 需按 evidence 提示补齐；此处不注入
       - 不可恢复：rag_search 空结果 / timeseries_anomaly 序列过短 → 如实告知（direct_answer 说明缺数据）

轨迹中 tool 消息内容均来自真实执行（截断到 600 字），避免臆造工具返回。
输出：data/planner/seeds/multi_turn_seeds.jsonl 与 report_multi_turn.md；export_sft.py 会读取并导出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tools.tool_registry import validate_arguments_strict  # noqa: E402

SEEDS = ROOT / "data/planner/seeds/task_seeds.jsonl"
OUT = ROOT / "data/planner/seeds/multi_turn_seeds.jsonl"
REPORT = ROOT / "data/planner/seeds/report_multi_turn.md"

# 俗称 → 图谱标准实体名（用真实 kg_search 验证 no_match → ok）
COLLOQUIAL_TO_STD = [
    ("有响声", "噪声异常"), ("变压器有异响", "噪声异常"), ("冒油", "喷油"), ("油位不对", "油位异常"),
    ("油温偏高", "油温升高"), ("差动跳了", "差动保护动作"), ("直阻三相差得多", "直流电阻不平衡"),
    ("重瓦斯动了", "轻瓦斯动作"), ("导线", "引线"), ("超载", "过负荷"),
]
STD_QUESTION = {
    "噪声异常": "变压器{c}，可能是什么原因，怎么查？", "喷油": "{dev}{c}了，一般是什么故障引起的？",
    "油位异常": "{dev}{c}，是哪里出问题了？", "油温升高": "{dev}{c}，会是什么故障？",
    "差动保护动作": "{dev}{c}，可能的故障有哪些？", "直流电阻不平衡": "{dev}{c}，说明什么问题？",
    "轻瓦斯动作": "{dev}{c}，什么原因会导致？", "引线": "{c}过热一般会导致什么故障？", "过负荷": "{dev}长期{c}会造成什么后果？",
}
DEVICE_POOL = ["220kV 主变 #1", "110kV 主变 #2", "500kV 1 号主变", "35kV 站用变", "某 220kV 变电站 2 号主变", "厂用变 #3"]


def _sid(*parts: Any) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:10]


def _trunc(obj: Any, n: int = 600) -> str:
    s = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[:n] + "…"


def _tool_fns() -> Dict[str, Any]:
    from src.tools.fault_attribution import fault_attribution
    from src.tools.kg_search import kg_search
    from src.tools.ett_forecasting import ett_forecast
    from src.tools.local_kb import local_kb_search
    from scripts.eval_end2end import mock_timeseries_anomaly
    return {"fault_attribution": lambda **kw: fault_attribution(dga_data=kw.get("dga_data"), query=kw.get("query", ""),
                                                                 **({"evidence": kw["evidence"]} if kw.get("evidence") else {})),
            "kg_search": kg_search, "ett_forecast": ett_forecast, "rag_search": local_kb_search,
            "timeseries_anomaly": mock_timeseries_anomaly}


def _compact_result(tool: str, r: Any) -> Dict[str, Any]:
    """把真实工具返回压缩为进入对话的 tool 消息（保留决策所需字段）。"""
    if tool == "fault_attribution" and isinstance(r, dict):
        return {"status": r.get("status"), "primary_fault": r.get("primary_fault"), "primary_fault_name": r.get("primary_fault_name"),
                "primary_probability": round(float(r.get("primary_probability", 0)), 3),
                "top3": [{"name": x.get("fault_name") or x.get("name"), "p": round(float(x.get("probability", x.get("p", 0))), 3)}
                         for x in (r.get("fault_ranking") or [])[:3]]}
    if tool == "kg_search" and isinstance(r, dict):
        return {"status": r.get("status"), "message": r.get("message"), "matched_entities": [m.get("name") for m in r.get("matched_entities", [])[:3]],
                "n_paths": len(r.get("paths", [])), "summary": _trunc(r.get("summary", ""), 300)}
    if tool == "ett_forecast" and isinstance(r, dict):
        out = {"status": r.get("status"), "message": r.get("message")}
        if r.get("status") == "ok":
            fc = r.get("forecast") or []
            vals = [x.get("predicted_OT", x) if isinstance(x, dict) else x for x in fc]
            out.update({"dataset": r.get("dataset"), "horizon": r.get("horizon"), "n_forecast": len(fc),
                        "pred_range": [round(min(vals), 2), round(max(vals), 2)] if vals and all(isinstance(v, (int, float)) for v in vals) else None,
                        "anomalies": len(r.get("anomalies") or [])})
        return out
    if tool == "rag_search":
        items = r if isinstance(r, list) else []
        return {"status": "ok" if items else "empty_result", "n": len(items),
                "hits": [{"title": x.get("metadata", {}).get("title", ""), "score": round(float(x.get("score", 0)), 3)} for x in items[:3]]}
    if tool == "timeseries_anomaly" and isinstance(r, dict):
        return {k: r.get(k) for k in ("status", "message", "n", "anomaly_indices") if k in r}
    return {"status": "ok", "raw": _trunc(r, 300)}


def _turn(seed_id: str, category: str, sub_type: str, query: str, trajectory: List[Dict[str, Any]],
          group: str, seed_source: str, note: str, context: Optional[Dict[str, Any]] = None, split: str = "") -> Dict[str, Any]:
    rec = {"seed_id": seed_id, "category": category, "sub_type": sub_type, "query": query, "trajectory": trajectory,
           "group_key": group, "seed_source": seed_source, "note": note, "needs_llm_rewrite": True, "split": split, "exec_verified": True}
    if context:
        rec["context"] = context
    return rec


def build(seeds: List[Dict[str, Any]], rng: random.Random, n_end: int, n_err: int) -> List[Dict[str, Any]]:
    fns = _tool_fns()
    out: List[Dict[str, Any]] = []
    stats = Counter()

    # ── A. composite → 真实两轮 ──
    for s in [x for x in seeds if x["category"] == "composite" and len(x["gold_actions"]) >= 2]:
        a1, a2 = s["gold_actions"][0], s["gold_actions"][1]
        r1 = fns[a1["tool"]](**a1["arguments"])
        c1 = _compact_result(a1["tool"], r1)
        if c1.get("status") != "ok":
            stats["A_skip_step1_fail"] += 1
            continue
        args2 = dict(a2["arguments"])
        if a1["tool"] == "fault_attribution":
            fname = c1["primary_fault_name"]
            if a2["tool"] == "kg_search":
                args2["query"] = fname
            elif a2["tool"] == "rag_search":
                args2["query"] = f"{fname} 检修处理 规程"
        r2 = fns[a2["tool"]](**args2)
        c2 = _compact_result(a2["tool"], r2)
        traj = [{"role": "assistant", "tool_call": {"tool": a1["tool"], "arguments": a1["arguments"]},
                 "thought": "先用数值工具做定量分析，再根据其返回结果补充知识/关系解释。"},
                {"role": "tool", "tool": a1["tool"], "content": c1},
                {"role": "assistant", "tool_call": {"tool": a2["tool"], "arguments": args2},
                 "thought": f"第 1 步判定主故障为「{c1.get('primary_fault_name', '')}」，据此进行第 2 步检索。" if a1["tool"] == "fault_attribution"
                 else "基于第 1 步结果进行第 2 步。"}]
        if c2.get("status") in ("ok",):
            traj += [{"role": "tool", "tool": a2["tool"], "content": c2},
                     {"role": "assistant", "finish": True, "thought": "两步结果均有效，信息充分，结束规划交给生成器。"}]
            sub = "composite_2step"
        elif a2["tool"] == "kg_search" and c2.get("status") == "no_relation":
            # 真实可恢复：放宽关系
            args3 = {k: v for k, v in args2.items() if k != "relations"}
            args3["direction"] = "both"
            r3 = fns["kg_search"](**args3)
            c3 = _compact_result("kg_search", r3)
            traj += [{"role": "tool", "tool": a2["tool"], "content": c2},
                     {"role": "assistant", "tool_call": {"tool": "kg_search", "arguments": args3},
                      "thought": "实体已定位但限定关系下无边，放宽关系类型与方向重试。"},
                     {"role": "tool", "tool": "kg_search", "content": c3},
                     {"role": "assistant", "finish": True,
                      "thought": "重试后已取得关系路径，结束规划。" if c3.get("status") == "ok" else "重试仍无关系，如实告知图谱无该实体的关系信息。"}]
            sub = "composite_recover_kg"
        else:
            traj += [{"role": "tool", "tool": a2["tool"], "content": c2},
                     {"role": "assistant", "finish": True, "thought": "第 2 步无有效返回，不再重试，如实说明并以第 1 步结果作答。"}]
            sub = "composite_2step_partial"
        for t in traj:
            if "tool_call" in t:
                v = validate_arguments_strict(t["tool_call"]["tool"], t["tool_call"]["arguments"])
                assert v["valid"], (s["seed_id"], v)
        out.append(_turn(f"M-{_sid('A', s['seed_id'])}", "multi_turn", sub, s["query"], traj, s["group_key"], s["seed_source"],
                         "由 composite 种子 + 真实工具返回构造", split=s["split"]))
        stats[sub] += 1

    # ── B. 单工具 → finish ──
    pool = [x for x in seeds if x["category"] == "numeric_tool" and len(x["gold_actions"]) == 1]
    rng.shuffle(pool)
    for s in pool[:n_end]:
        a = s["gold_actions"][0]
        r = fns[a["tool"]](**a["arguments"])
        c = _compact_result(a["tool"], r)
        if c.get("status") != "ok":
            continue
        traj = [{"role": "assistant", "tool_call": {"tool": a["tool"], "arguments": a["arguments"]}, "thought": "问题给出了可计算数据，直接调用数值工具。"},
                {"role": "tool", "tool": a["tool"], "content": c},
                {"role": "assistant", "finish": True, "thought": "工具已返回有效结果且问题只有一个目标，不再调用其他工具。"}]
        out.append(_turn(f"M-{_sid('B', s['seed_id'])}", "multi_turn", "single_then_finish", s["query"], traj, s["group_key"],
                         s["seed_source"], "学习「够了就停」", context=s.get("context"), split=s["split"]))
        stats["single_then_finish"] += 1

    # ── C. 错误恢复 ──
    ett_pool = [x for x in seeds if x["sub_type"] == "ett_forecast"]
    rng.shuffle(ett_pool)
    k = 0
    for s in ett_pool:
        if k >= n_err // 3:
            break
        a = s["gold_actions"][0]
        good = dict(a["arguments"])
        kind = rng.choice(["bad_dataset", "bad_range"])
        bad = dict(good)
        if kind == "bad_dataset":
            bad["dataset"] = "ETTh3" if good.get("dataset", "ETTh1").startswith("ETTh") else "ETTm3"
            v = validate_arguments_strict("ett_forecast", bad)
            # ETTh3 不在 enum → 严格校验会拒绝，这类错误在 Retriever 层就被拦截，轨迹里以 validation_failed 呈现
            c_bad = {"status": "validation_failed", "errors": v["errors"]}
            thought_fix = "参数校验失败：dataset 不在枚举中，改为合法数据集名重试。"
        else:
            bad["start_time"], bad["end_time"] = "2019-01-01", "2019-01-07"
            c_bad = _compact_result("ett_forecast", fns["ett_forecast"](**bad))
            if c_bad.get("status") != "error":
                continue
            thought_fix = "工具返回无数据：时间范围超出数据集覆盖（2016-07 ~ 2018-06），改用数据集内的时间范围重试。"
        r_good = fns["ett_forecast"](**good)
        c_good = _compact_result("ett_forecast", r_good)
        if c_good.get("status") != "ok":
            continue
        traj = [{"role": "assistant", "tool_call": {"tool": "ett_forecast", "arguments": bad}, "thought": "调用油温预测工具。"},
                {"role": "tool", "tool": "ett_forecast", "content": c_bad},
                {"role": "assistant", "tool_call": {"tool": "ett_forecast", "arguments": good}, "thought": thought_fix},
                {"role": "tool", "tool": "ett_forecast", "content": c_good},
                {"role": "assistant", "finish": True, "thought": "重试成功，结束。"}]
        out.append(_turn(f"M-{_sid('C1', s['seed_id'], kind)}", "error_recovery", f"ett_{kind}", s["query"], traj, s["group_key"],
                         s["seed_source"], "首次调用参数错误（真实返回），第二次修正", split=s["split"]))
        stats[f"ett_{kind}"] += 1
        k += 1

    # kg：俗称 no_match → 标准名重试（真实验证）
    k = 0
    for c_word, std in COLLOQUIAL_TO_STD:
        if k >= n_err // 3:
            break
        r_bad = fns["kg_search"](query=c_word)
        r_good = fns["kg_search"](query=std)
        cb, cg = _compact_result("kg_search", r_bad), _compact_result("kg_search", r_good)
        if cb.get("status") != "no_match" or cg.get("status") != "ok":
            continue
        for _ in range(3):
            dev = rng.choice(DEVICE_POOL)
            q = STD_QUESTION[std].format(c=c_word, dev=dev)
            traj = [{"role": "assistant", "tool_call": {"tool": "kg_search", "arguments": {"query": c_word}}, "thought": "用户描述的是现象，先到图谱定位实体。"},
                    {"role": "tool", "tool": "kg_search", "content": cb},
                    {"role": "assistant", "tool_call": {"tool": "kg_search", "arguments": {"query": std}},
                     "thought": f"图谱未匹配口语「{c_word}」，改用标准术语「{std}」重试。"},
                    {"role": "tool", "tool": "kg_search", "content": cg},
                    {"role": "assistant", "finish": True, "thought": "已取得关系路径，结束。"}]
            out.append(_turn(f"M-{_sid('C2', c_word, std, dev)}", "error_recovery", "kg_colloquial_to_std", q, traj,
                             f"kgalias:{std}", f"kg:{std}", "俗称→标准术语的规范化恢复；训练侧同时学到术语映射", split=""))
            stats["kg_colloquial_to_std"] += 1
        k += 1

    # 不可恢复：timeseries 序列过短 → 如实告知
    short_signals = [[1.0, 2.0], [23.5], [12.1, 12.3]]
    for i, sig in enumerate(short_signals):
        for dev in rng.sample(DEVICE_POOL, 3):
            q = f"{dev}油温最近两次读数 {sig}，做个异常检测。"
            cb = _compact_result("timeseries_anomaly", fns["timeseries_anomaly"](signal=sig))
            traj = [{"role": "assistant", "tool_call": {"tool": "timeseries_anomaly", "arguments": {"signal": sig}}, "thought": "用户给了序列，调用异常检测。"},
                    {"role": "tool", "tool": "timeseries_anomaly", "content": cb},
                    {"role": "assistant", "finish": True, "ask_user": ["至少 3 个（建议 24 个以上）连续采样点"],
                     "thought": "工具因数据点不足报错，该错误无法通过改参数恢复，如实告知并请用户补充更多数据。"}]
            out.append(_turn(f"M-{_sid('C3', i, dev)}", "error_recovery", "ts_unrecoverable_ask", q, traj, f"tsshort:{i}", "synthetic_signal",
                             "不可恢复错误 → 如实告知 + 追问，不得臆造结果", split=""))
            stats["ts_unrecoverable_ask"] += 1

    # 分配尚未有 split 的样本（按 group_key）
    groups = sorted({x["group_key"] for x in out if not x["split"]})
    rng.shuffle(groups)
    plan = {g: ("train" if i < len(groups) * 0.7 else "dev" if i < len(groups) * 0.8 else "test") for i, g in enumerate(groups)}
    for x in out:
        if not x["split"]:
            x["split"] = plan[x["group_key"]]
    print(dict(stats))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-end", type=int, default=300, help="single_then_finish 样本数")
    ap.add_argument("--n-err", type=int, default=90, help="错误恢复目标数（ett/kg 各约 1/3）")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seeds = [json.loads(l) for l in open(SEEDS, encoding="utf-8") if l.strip()]
    out = build(seeds, rng, a.n_end, a.n_err)
    with open(OUT, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    cat = Counter((x["category"], x["sub_type"]) for x in out)
    sp = Counter(x["split"] for x in out)
    turns = Counter(sum(1 for t in x["trajectory"] if "tool_call" in t) for x in out)
    L = ["# P4-3 多轮 / 错误恢复样本（D8 补充）构建报告\n",
         f"- 样本总数：{len(out)}；切分：{dict(sp)}", f"- 每样本工具调用次数分布：{dict(sorted(turns.items()))}",
         "- 所有 tool 消息内容来自真实工具执行（压缩字段），assistant 的第二轮决策由规则依据真实返回生成；不含任何臆造工具输出。", "",
         "| 类别 | 子类 | 数量 | 说明 |", "|---|---|---:|---|"]
    desc = {"composite_2step": "两步均成功，finish", "composite_recover_kg": "第 2 步 no_relation → 放宽关系重试",
            "composite_2step_partial": "第 2 步无有效返回 → 不重试如实说明", "single_then_finish": "单工具成功后停止",
            "ett_bad_dataset": "枚举外数据集 → 校验失败 → 修正", "ett_bad_range": "时间范围超出 → 工具 error → 修正",
            "kg_colloquial_to_std": "俗称 no_match → 标准术语重试", "ts_unrecoverable_ask": "序列过短 → 不可恢复 → 追问"}
    for (c, s), v in sorted(cat.items()):
        L.append(f"| {c} | {s} | {v} | {desc.get(s, '')} |")
    L += ["", "## 说明",
          "- `trajectory` 为统一轨迹：assistant(tool_call|finish[, ask_user]) / tool(content)。导出 SFT 时每个 assistant 决策点生成一条训练样本（前缀含此前全部消息）。",
          "- `composite_*` 沿用原 composite 种子的 group_key/split；kg 俗称类按标准实体分组；ts 类为合成短序列，仅用于训练侧"
          "「不可恢复→追问」行为，不进入评测。",
          "- query 仍为模板问法（needs_llm_rewrite=true）。"]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"写入 {len(out)} 条 → {OUT}\n" + "\n".join(L[:4]))


if __name__ == "__main__":
    main()
