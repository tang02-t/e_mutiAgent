#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P6-1：构建端到端评测集 D10（data/eval/d10/end2end_eval.jsonl）。

来源：D7/D8 封存 test 切分（task_seeds.jsonl + multi_turn_seeds.jsonl，split == test）。
本脚本只做程序化分层抽样与字段整理，不改写任何查询、不查看训练集。

分层配额（目标 ~196 条，覆盖 PLAN P6-1 五类场景 + 错误恢复）：
  single_tool_fact      40  fact：A_title / B_caption / C_definition
  single_tool_numeric   40  numeric_tool：dga_attribution 20 / ett_forecast 10 / timeseries_anomaly 10
  kg_reasoning          30  reasoning：kg_1hop_canonical 10 / kg_1hop_alias 10 / kg_2hop 10
  multi_tool            30  composite（dga_rag / dga_kg / ett_anomaly 各 5）+ multi_turn.composite_2step 15
  ask_user              17  insufficient 全部
  context_contrast       8  numeric_tool.dga_from_context（与 ask_user 同问法、上下文含 DGA）
  no_tool               15  no_tool 四个子类按比例
  error_recovery        16  kg_colloquial_to_std 9 / ett_bad_dataset 5 / ett_bad_range 2

每条字段：
  eval_id, scenario, category, sub_type, user_query, context,
  required_actions      必要动作（有序；tool_call / ask_user / direct_answer）
  key_params            关键参数（评分时只比对这些键，其余参数不计）
  evidence_source       期望证据来源（kb chunk / kg path / dga label / ett dataset / none）
  reference_points      参考答案要点（离线自动生成，annotation.status = auto，待人工复核）
  annotation            {status, source, notes}
  seed_id, seed_source, group_key

用法：
  python3 scripts/eval/build_d10.py [--seed 20260915] [--out data/eval/d10/end2end_eval.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TASK_SEEDS = ROOT / "data/planner/seeds/task_seeds.jsonl"
MULTI_SEEDS = ROOT / "data/planner/seeds/multi_turn_seeds.jsonl"
CHUNKS = ROOT / "data/kb/chunks.jsonl"
SUMMARIES = ROOT / "data/kb/index/chunk_summaries.jsonl"
OUT_DIR = ROOT / "data/eval/d10"

QUOTA: Dict[str, Dict[str, int]] = {
    "single_tool_fact": {"A_title": 16, "B_caption": 14, "C_definition": 10},
    "single_tool_numeric": {"dga_attribution": 20, "ett_forecast": 10, "timeseries_anomaly": 10},
    "kg_reasoning": {"kg_1hop_canonical": 10, "kg_1hop_alias": 10, "kg_2hop": 10},
    "multi_tool": {"dga_rag": 5, "dga_kg": 5, "ett_anomaly": 5, "composite_2step": 15},
    "ask_user": {"ask_missing_params": 17},
    "context_contrast": {"dga_from_context": 8},
    "no_tool": {"common_sense": 8, "greeting": 2, "capability": 2, "meta": 2, "out_of_scope": 1},
    "error_recovery": {"kg_colloquial_to_std": 9, "ett_bad_dataset": 5, "ett_bad_range": 2},
}

# 关键参数：评分时只比对这些键
KEY_PARAMS = {
    "rag_search": ["query"],
    "kg_search": ["query", "relations", "hops", "direction"],
    "fault_attribution": ["dga_data"],
    "ett_forecast": ["dataset", "horizon", "start_time", "end_time"],
    "timeseries_anomaly": ["signal"],
}


def _jl(p: Path) -> List[Dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_kb() -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    chunks = {r["chunk_id"]: r for r in _jl(CHUNKS)} if CHUNKS.exists() else {}
    summ = {r["chunk_id"]: r["summary"] for r in _jl(SUMMARIES)} if SUMMARIES.exists() else {}
    return chunks, summ


def _actions_from_gold(gold: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acts = []
    for g in gold:
        if g.get("type") == "tool_call":
            tool = g["tool"]
            args = g.get("arguments") or {}
            acts.append({"type": "tool_call", "tool": tool,
                         "key_params": {k: args[k] for k in KEY_PARAMS.get(tool, []) if k in args}})
        elif g.get("type") == "ask_user":
            acts.append({"type": "ask_user", "missing": g.get("missing", [])})
        else:
            acts.append({"type": "direct_answer"})
    return acts


def _actions_from_trajectory(traj: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """多轮轨迹 → 必要动作序列 + 期望的恢复轨迹（首个错误调用不计入必要动作）。"""
    acts, recovery = [], []
    pending_error = None
    for i, step in enumerate(traj):
        if step.get("role") != "assistant":
            continue
        if step.get("finish"):
            if step.get("ask_user"):
                acts.append({"type": "ask_user", "missing": []})
            continue
        tc = step.get("tool_call") or {}
        tool, args = tc.get("tool"), tc.get("arguments") or {}
        # 看下一条 tool 消息是否为业务错误
        nxt = traj[i + 1] if i + 1 < len(traj) else {}
        content = nxt.get("content") if nxt.get("role") == "tool" else None
        is_err = isinstance(content, dict) and content.get("status") in (
            "error", "no_match", "no_data", "failed", "no_relation", "validation_failed")
        entry = {"type": "tool_call", "tool": tool,
                 "key_params": {k: args[k] for k in KEY_PARAMS.get(tool, []) if k in args}}
        if is_err:
            pending_error = {"first_call": entry, "error": content}
            continue
        if pending_error is not None:
            recovery.append({**pending_error, "recovered_call": entry})
            pending_error = None
        acts.append(entry)
    return acts, recovery


def _evidence_and_points(rec: Dict[str, Any], chunks: Dict[str, Dict[str, Any]],
                         summ: Dict[str, str]) -> tuple[Dict[str, Any], List[str]]:
    cat, st, hint = rec["category"], rec["sub_type"], rec.get("expected_hint") or {}
    if cat == "fact":
        cids = hint.get("gold_chunk_ids", [])
        c = chunks.get(cids[0]) if cids else None
        ev = {"type": "kb_chunk", "chunk_ids": cids, "doc_id": hint.get("gold_doc_id"),
              "doc_title": c.get("doc_title") if c else None,
              "section_path": c.get("section_path") if c else None}
        pts = [summ.get(cids[0], "")] if cids and summ.get(cids[0]) else []
        pts.append("答案应引用该文献片段的内容，不得凭空编造标准条款或数值")
        return ev, pts
    if cat == "reasoning":
        if st == "kg_2hop":
            ev = {"type": "kg_path", "path_prefix": hint.get("path_prefix", [])}
            pts = [f"应给出两跳链路：{' → '.join(hint.get('path_prefix', []))} → （部位/部件）"]
        else:
            ev = {"type": "kg_entities", "expected_entities": hint.get("expected_entities", []),
                  "relation": hint.get("relation")}
            names = [e.split(":", 1)[-1] for e in hint.get("expected_entities", [])]
            pts = [f"应提及图谱中的后果/关联实体：{'、'.join(names)}" if names else "应基于图谱关系回答"]
        pts.append("每条关系应能追溯到文献原文（kg_search 返回 evidence）")
        return ev, pts
    if cat == "numeric_tool":
        if st in ("dga_attribution", "dga_from_context"):
            ev = {"type": "dga_label", "raw_label": hint.get("raw_label"), "raw_label_cn": hint.get("raw_label_cn"),
                  "severity": hint.get("severity"), "source": hint.get("source")}
            pts = [f"数据集原始标签：{hint.get('raw_label_cn')}（严重度 {hint.get('severity')}）；"
                   "fault_attribution 输出可与之不一致，评分以“是否调用工具并如实引用其结果”为准",
                   "应给出主故障、概率排序与处理建议；高危时提示安全措施"]
            return ev, pts
        if st == "ett_forecast":
            ev = {"type": "ett_dataset", "hours": hint.get("hours"), "freq": hint.get("freq")}
            pts = ["应引用 ett_forecast 返回的预测均值/趋势与历史统计",
                   f"horizon 需按采样间隔换算（{hint.get('freq')}，{hint.get('hours')} 小时）"]
            return ev, pts
        if st == "timeseries_anomaly":
            ev = {"type": "signal_inline"}
            pts = ["应调用 timeseries_anomaly 并报告均值、标准差与异常点索引", "不得凭肉眼判断“正常”"]
            return ev, pts
    if cat == "composite":
        ev = {"type": "multi", "raw_label_cn": hint.get("raw_label_cn"), "note": hint.get("note")}
        pts = ["两步结果都应体现在答案中；第二步参数应基于第一步真实返回",
               hint.get("note") or ""]
        return ev, [p for p in pts if p]
    if cat == "insufficient":
        ev = {"type": "none"}
        pts = [f"应追问缺失信息：{'、'.join(hint.get('missing', []))}", "不得调用任何工具、不得凭空诊断"]
        return ev, pts
    if cat == "no_tool":
        ev = {"type": "none"}
        pts = ["无需调用工具，直接回答或说明能力范围", "回答简洁、不编造设备数据"]
        return ev, pts
    if cat in ("multi_turn", "error_recovery"):
        ev = {"type": "trajectory", "note": rec.get("note")}
        pts = ["按期望轨迹调用工具；工具报错时应修正参数重试或如实告知，不得编造结果"]
        return ev, pts
    return {"type": "unknown"}, []


def build(seed: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    chunks, summ = _load_kb()
    task = [r for r in _jl(TASK_SEEDS) if r.get("split") == "test"]
    multi = [r for r in _jl(MULTI_SEEDS) if r.get("split") == "test"]
    pool = collections.defaultdict(list)
    for r in task + multi:
        pool[r["sub_type"]].append(r)
    for v in pool.values():
        rng.shuffle(v)

    out: List[Dict[str, Any]] = []
    shortage: Dict[str, int] = {}
    for scenario, quota in QUOTA.items():
        for st, n in quota.items():
            cands = pool.get(st, [])
            take = cands[:n]
            if len(take) < n:
                shortage[f"{scenario}/{st}"] = n - len(take)
            for rec in take:
                if "trajectory" in rec:
                    acts, recovery = _actions_from_trajectory(rec["trajectory"])
                else:
                    acts, recovery = _actions_from_gold(rec.get("gold_actions", [])), []
                ev, pts = _evidence_and_points(rec, chunks, summ)
                out.append({
                    "eval_id": f"D10-{len(out) + 1:04d}",
                    "scenario": scenario,
                    "category": rec["category"],
                    "sub_type": st,
                    "user_query": rec["query"],
                    "context": rec.get("context") or {},
                    "required_actions": acts,
                    "expected_recovery": recovery,
                    "n_required_tool_calls": sum(1 for a in acts if a["type"] == "tool_call"),
                    "evidence_source": ev,
                    "reference_points": pts,
                    "annotation": {"status": "auto", "source": "seed_gold+kb_summary",
                                   "notes": "参考要点由脚本从金标/摘要生成，需人工复核后改为 reviewed"},
                    "seed_id": rec["seed_id"],
                    "seed_source": rec.get("seed_source"),
                    "group_key": rec.get("group_key"),
                })
    rng.shuffle(out)
    for i, r in enumerate(out, 1):
        r["eval_id"] = f"D10-{i:04d}"

    dist = collections.Counter(r["scenario"] for r in out)
    sub = collections.Counter(f"{r['scenario']}/{r['sub_type']}" for r in out)
    stats = {"n": len(out), "seed": seed, "by_scenario": dict(dist), "by_sub_type": dict(sub),
             "shortage": shortage,
             "n_unique_group_key": len({r["group_key"] for r in out}),
             "source_splits": "task_seeds.test + multi_turn_seeds.test"}
    return out, stats


def write_card(stats: Dict[str, Any], out_path: Path) -> None:
    L = ["# D10 端到端评测集数据卡片\n",
         f"- 文件：`{out_path.relative_to(ROOT)}`",
         f"- 规模：{stats['n']} 条；随机种子 {stats['seed']}",
         f"- 来源：D7/D8 封存 test 切分（{stats['source_splits']}），按 sub_type 程序化分层抽样，未做任何改写",
         f"- 覆盖 group_key：{stats['n_unique_group_key']} 个（同一来源派生的样本不会同时出现在训练集）",
         "- 真实/合成：查询与金标均派生自真实文献块、真实图谱、真实 DGA/ETT 记录；不含 `data/synthetic`",
         "",
         "## 场景分布", "",
         "| 场景 | 条数 |", "|---|---:|"]
    for k, v in sorted(stats["by_scenario"].items()):
        L.append(f"| {k} | {v} |")
    L += ["", "## 子类分布", "", "| 场景/子类 | 条数 |", "|---|---:|"]
    for k, v in sorted(stats["by_sub_type"].items()):
        L.append(f"| {k} | {v} |")
    if stats["shortage"]:
        L += ["", "## 配额缺口", ""] + [f"- {k}: 缺 {v} 条" for k, v in stats["shortage"].items()]
    L += ["", "## 标注字段说明", "",
          "- `required_actions`：必要动作序列；`tool_call.key_params` 只列关键参数，其余参数不计分。",
          "- `expected_recovery`：错误恢复样本的“首次错误调用 → 修正调用”对，评测 `错误恢复成功率`。",
          "- `evidence_source`：期望证据来源（kb_chunk / kg_path / kg_entities / dga_label / ett_dataset / none）。",
          "- `reference_points`：参考答案要点，`annotation.status=auto` 为脚本生成，人工复核后改为 `reviewed`。",
          "",
          "## 已知偏差", "",
          "- 查询仍为模板化措辞（P4-2 口语化改写需 LLM，尚未执行），对 Planner 偏乐观。",
          "- DGA 场景的 `raw_label_cn` 是数据集标签，不代表真实故障部位，不能直接作为“正确答案”。",
          "- ETT 负载特征单位未知，预测数值只做趋势判断。",
          ]
    (out_path.parent / "DATA_CARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", default=str(OUT_DIR / "end2end_eval.jsonl"))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, stats = build(args.seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_path.parent / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_card(stats, out_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"写入 {out_path}（{len(rows)} 条）")


if __name__ == "__main__":
    main()
