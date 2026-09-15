#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5-4：工具调用 Planner 离线评测（读 training/planner_sft/predict.py 产出的预测文件）。

指标（PLAN P5-4，均在封存 test 集上计算）：
  格式合法率      pred 非空（输出可解析为 {content, tool_calls} 且每个调用的 arguments 为合法 JSON 对象）
  工具选择正确率  预测调用的工具名多重集合 == 金标工具名多重集合（无调用样本：预测也无调用）
  参数正确率      对有金标调用的样本：每个金标调用都能在预测中找到同名调用，且关键参数逐键匹配
                  （KEY_PARAMS 与 P6 评测一致：query 词法重合 ≥ 0.5；dga_data 数值容差；relations 集合；其余相等）
  完整调用率      格式合法 ∧ 工具正确 ∧ 参数正确 ∧ 参数通过 Schema 严格校验（validate_arguments_strict）
  不必要调用率    金标无调用（no_tool / insufficient / single_then_finish 收尾）而预测发起了调用
  追问正确率      insufficient / ts_unrecoverable_ask 样本：预测无调用 ∧ content 含追问/缺少措辞
  错误恢复成功率  error_recovery 且 decision_index ≥ 1（已看到工具报错）的样本：预测调用工具与金标一致且参数正确
分类别报告：fact / reasoning / numeric_tool / no_tool / insufficient / composite / multi_turn / error_recovery。

用法：
  python3 scripts/eval/eval_planner_offline.py --pred-dir data/planner/predictions --write-report
  python3 scripts/eval/eval_planner_offline.py --pred data/planner/predictions/M3_seed42.jsonl
  python3 scripts/eval/eval_planner_offline.py --self-test    # 用金标作为预测做自检（应全部 100%）
输出：docs/planner_eval.md（同名 run 的两个 seed 自动取均值，列出 seed 数）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.tools.tool_registry import validate_arguments_strict  # noqa: E402

REPORT = ROOT / "docs/planner_eval.md"
KEY_PARAMS = {
    "rag_search": ["query"],
    "kg_search": ["query", "relations", "hops", "direction"],
    "fault_attribution": ["dga_data"],
    "ett_forecast": ["dataset", "horizon", "start_time", "end_time"],
    "timeseries_anomaly": ["signal"],
}
ASK_WORDS = ("缺少", "补充", "提供", "请给出", "请提供", "需要您", "无法", "不足", "请告知", "请说明")
CATEGORY_ORDER = ["fact", "reasoning", "numeric_tool", "no_tool", "insufficient", "composite",
                  "multi_turn", "error_recovery"]
METRICS = ["format_valid", "tool_correct", "param_correct", "complete_call", "unnecessary_call",
           "ask_correct", "recovery_success"]
METRIC_CN = {
    "format_valid": "格式合法率", "tool_correct": "工具选择正确率", "param_correct": "参数正确率",
    "complete_call": "完整调用率", "unnecessary_call": "不必要调用率", "ask_correct": "追问正确率",
    "recovery_success": "错误恢复成功率",
}


def _tokens(s: str) -> set:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(s or ""))
    toks = set()
    for w in s.split():
        if re.fullmatch(r"[A-Za-z0-9_.-]+", w):
            toks.add(w.lower())
        else:
            toks.update(w[i:i + 2] for i in range(max(len(w) - 1, 1)))
    return toks


def _query_match(got: str, expect: str) -> bool:
    ta, tb = _tokens(got), _tokens(expect)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(tb) >= 0.5 or expect in got or got in expect


def _param_ok(key: str, expect: Any, got: Any) -> bool:
    if key == "query":
        return _query_match(str(got or ""), str(expect or ""))
    if key == "dga_data":
        if not isinstance(got, dict) or not isinstance(expect, dict):
            return False
        try:
            return all(k in got and abs(float(got[k]) - float(v)) < 1e-6 for k, v in expect.items())
        except (TypeError, ValueError):
            return False
    if key == "relations":
        return set(got or []) == set(expect or [])
    if key == "signal":
        try:
            return isinstance(got, list) and len(got) == len(expect or []) and all(
                abs(float(x) - float(y)) < 1e-6 for x, y in zip(got, expect))
        except (TypeError, ValueError):
            return False
    if key in ("hops", "horizon"):
        if got is None and expect is None:
            return True
        try:
            return got is not None and expect is not None and int(got) == int(expect)
        except (TypeError, ValueError):
            return False
    return got == expect


def _call_params_ok(gold_call: Dict[str, Any], pred_call: Dict[str, Any]) -> bool:
    ga, pa = gold_call.get("arguments") or {}, pred_call.get("arguments") or {}
    if not isinstance(pa, dict):
        return False
    for k in KEY_PARAMS.get(gold_call.get("name"), []):
        if k in ga and not _param_ok(k, ga.get(k), pa.get(k)):
            return False
    return True


def _schema_ok(calls: List[Dict[str, Any]]) -> bool:
    for c in calls:
        args = c.get("arguments")
        if not isinstance(args, dict):
            return False
        if not validate_arguments_strict(c.get("name", ""), args).get("valid"):
            return False
    return True


def _is_ask_sample(row: Dict[str, Any]) -> bool:
    return row.get("category") == "insufficient" or row.get("sub_type") == "ts_unrecoverable_ask" and not row["gold"]["tool_calls"]


def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
    gold, pred = row["gold"], row.get("pred")
    gcalls = gold.get("tool_calls") or []
    out: Dict[str, Any] = {"format_valid": pred is not None}
    pcalls = (pred or {}).get("tool_calls") or []
    out["tool_correct"] = pred is not None and sorted(c.get("name") or "" for c in pcalls) == sorted(c.get("name") for c in gcalls)
    if gcalls:
        used = set()
        ok = pred is not None
        for g in gcalls:
            hit = None
            for i, p in enumerate(pcalls):
                if i in used or p.get("name") != g.get("name"):
                    continue
                if _call_params_ok(g, p):
                    hit = i
                    break
            if hit is None:
                ok = False
                break
            used.add(hit)
        out["param_correct"] = ok
    else:
        out["param_correct"] = None  # 无调用样本不计参数正确率
    out["complete_call"] = (out["format_valid"] and out["tool_correct"] and (out["param_correct"] is not False)
                            and _schema_ok(pcalls)) if gcalls else None
    out["unnecessary_call"] = (len(pcalls) > 0) if not gcalls else None
    if _is_ask_sample(row):
        content = (pred or {}).get("content") or ""
        out["ask_correct"] = pred is not None and not pcalls and any(w in content for w in ASK_WORDS)
    else:
        out["ask_correct"] = None
    if row.get("category") == "error_recovery" and (row.get("decision_index") or 0) >= 1 and gcalls:
        out["recovery_success"] = bool(out["tool_correct"] and out["param_correct"])
    else:
        out["recovery_success"] = None
    return out


def _rate(vals: List[Optional[bool]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return (sum(1 for x in v if x) / len(v)) if v else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [(r, score_row(r)) for r in rows]
    res: Dict[str, Any] = {"n": len(rows), "overall": {}, "by_category": {}}
    for m in METRICS:
        res["overall"][m] = _rate([s[m] for _, s in scored])
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, s in scored:
        by_cat[r.get("category") or "unknown"].append(s)
    for cat, ss in by_cat.items():
        res["by_category"][cat] = {"n": len(ss), **{m: _rate([s[m] for s in ss]) for m in METRICS}}
    res["mean_latency_ms"] = (sum(r.get("latency_ms") or 0 for r in rows) / len(rows)) if rows else None
    return res


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _run_base(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def aggregate_runs(per_run: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """同一变体不同 seed 取均值。"""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for name, s in per_run.items():
        groups[_run_base(name)].append(s)
    agg: Dict[str, Dict[str, Any]] = {}
    for base, lst in groups.items():
        def mean_of(getter):
            vals = [getter(s) for s in lst]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None
        a: Dict[str, Any] = {"n_seeds": len(lst), "n": lst[0]["n"], "overall": {}, "by_category": {}}
        for m in METRICS:
            a["overall"][m] = mean_of(lambda s, m=m: s["overall"].get(m))
        cats = sorted({c for s in lst for c in s["by_category"]}, key=lambda c: (CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99, c))
        for c in cats:
            a["by_category"][c] = {"n": next((s["by_category"][c]["n"] for s in lst if c in s["by_category"]), 0),
                                   **{m: mean_of(lambda s, m=m, c=c: (s["by_category"].get(c) or {}).get(m)) for m in METRICS}}
        a["mean_latency_ms"] = mean_of(lambda s: s.get("mean_latency_ms"))
        agg[base] = a
    return agg


def write_report(agg: Dict[str, Dict[str, Any]], pred_dir: Path) -> None:
    order = sorted(agg, key=lambda k: (0 if k.startswith("M0") else 1, k))
    lines = ["# Planner 离线评测（P5-4，工具调用智能体）", "",
             f"- 预测目录：`{pred_dir.relative_to(ROOT) if pred_dir.is_relative_to(ROOT) else pred_dir}`；"
             "评测集：`data/planner/sft/swift_test.jsonl` + `swift_multiturn_test.jsonl`（封存 test，合成数据不参与）",
             "- 同一变体多个 seed 取均值（n_seeds 列）。指标定义见 `scripts/eval/eval_planner_offline.py` 文件头。",
             "- M0 未微调基座；M1 普通 LoRA-SFT；M2 结构 Token 加权；M3 结构 + 领域 Token 加权（`training/planner_sft/`）。", "",
             "## 总表", "",
             "| 变体 | seeds | n | " + " | ".join(METRIC_CN[m] for m in METRICS) + " | 平均延迟 ms |",
             "|---|---|---|" + "---|" * len(METRICS) + "---|"]
    for k in order:
        a = agg[k]
        lat = "-" if a.get("mean_latency_ms") is None else f"{a['mean_latency_ms']:.0f}"
        lines.append(f"| {k} | {a['n_seeds']} | {a['n']} | " + " | ".join(_fmt(a['overall'][m]) for m in METRICS) + f" | {lat} |")
    lines += ["", "## 分类别", ""]
    for k in order:
        a = agg[k]
        lines += [f"### {k}", "", "| 类别 | n | " + " | ".join(METRIC_CN[m] for m in METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        for c, s in a["by_category"].items():
            lines.append(f"| {c} | {s['n']} | " + " | ".join(_fmt(s[m]) for m in METRICS) + " |")
        lines.append("")
    lines += ["## 说明", "",
              "- 参数正确率 / 完整调用率只在有金标调用的样本上计算；不必要调用率只在金标无调用的样本上计算；"
              "追问正确率只在 insufficient / ts_unrecoverable_ask 样本上计算；错误恢复成功率只在已看到工具报错的决策点上计算。",
              "- query 参数采用词法重合 ≥ 0.5 的宽松匹配（与 P6 端到端评测一致），其余参数精确比对。",
              "- 查询分解智能体（Q0/Q1）的评测（子查询数量准确率、LLM 裁判胜率）需调用 LLM，尚未纳入本表。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告写入 {REPORT.relative_to(ROOT)}")


def _load(p: Path) -> List[Dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def self_test() -> None:
    """用金标当预测：所有比率应为 100%（不必要调用率 0%）。"""
    sys.path.insert(0, str(ROOT / "training/planner_sft"))
    from predict import gold_of, load_test  # type: ignore
    rows = []
    for s in load_test():
        g = gold_of(s)
        meta = s.get("meta", {})
        rows.append({"seed_id": meta.get("seed_id"), "category": meta.get("category"), "sub_type": meta.get("sub_type"),
                     "decision_index": meta.get("decision_index"), "raw": "", "pred": g, "gold": g, "latency_ms": 0})
    s = summarize(rows)
    print(json.dumps({"n": s["n"], "overall": s["overall"]}, ensure_ascii=False, indent=1))
    for c, v in s["by_category"].items():
        print(c, {m: _fmt(v[m]) for m in METRICS})
    bad = [m for m in ("format_valid", "tool_correct", "param_correct", "complete_call", "ask_correct", "recovery_success")
           if s["overall"][m] is not None and s["overall"][m] < 1.0]
    if bad or (s["overall"]["unnecessary_call"] or 0) > 0:
        print("自检未通过：", bad)
        sys.exit(1)
    print("自检通过")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default=str(ROOT / "data/planner/predictions"))
    ap.add_argument("--pred", nargs="*", default=None, help="指定预测文件（默认目录内全部 *.jsonl）")
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    pred_dir = Path(args.pred_dir)
    files = [Path(p) for p in args.pred] if args.pred else sorted(pred_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"未找到预测文件：{pred_dir}")
    per_run = {f.stem: summarize(_load(f)) for f in files}
    agg = aggregate_runs(per_run)
    for k, a in sorted(agg.items()):
        print(k, {METRIC_CN[m]: _fmt(a["overall"][m]) for m in METRICS})
    if args.write_report:
        write_report(agg, pred_dir)


if __name__ == "__main__":
    main()
