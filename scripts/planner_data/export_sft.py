#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4-3 / P5 数据格式导出：把 task_seeds.jsonl（金标动作）转换为 Planner 微调样本。

两种输出格式：
1. ms-swift messages + tools（agent 模板，assistant 以 tool_calls 输出）  → data/planner/sft/swift_{split}.jsonl
2. LLaMA-Factory function-calling（conversations + tools 字符串）           → data/planner/sft/lf_{split}.json
3. 同时给出「JSON 文本规划」版本（与线上 json_text 回退路径一致）        → data/planner/sft/jsontext_{split}.jsonl

系统提示与线上 Planner 完全一致（PLANNER_SYSTEM_PROMPT + render_planner_user），保证训练/推理分布对齐。
所有样本 `needs_llm_rewrite=true` 仍为模板问法；本脚本为格式管线，LLM 扩写完成后重新运行即可。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.prompts import PLANNER_SYSTEM_PROMPT, render_planner_user  # noqa: E402
from src.tools.tool_registry import to_openai_tools                       # noqa: E402
from src.agents.planner import PlannerAgent                               # noqa: E402

SEEDS = ROOT / "data/planner/seeds/task_seeds.jsonl"
OUT = ROOT / "data/planner/sft"

STAGE_CN = {"rag_search": "知识检索", "kg_search": "图谱关系检索", "fault_attribution": "故障归因",
            "ett_forecast": "油温预测", "timeseries_anomaly": "信号异常检测"}

INTENT_TMPL = {
    "fact": "用户询问变压器领域的知识/规程/做法，属于知识查询，需要检索文献支撑。",
    "reasoning": "用户询问故障之间的因果、部位、检测或处理等结构化关系，属于关系推理，需在故障关系图谱中检索。",
    "numeric_tool": "用户提供了可计算的数据（油色谱浓度 / 时序数据 / 指定 ETT 数据集），需调用数值工具。",
    "composite": "用户问题包含多个子目标，需先做数据分析再补充知识或关系解释。",
    "no_tool": "用户问题为常识、能力询问或闲聊，可直接回答，无需调用工具。",
    "insufficient": "用户问题缺少调用工具所必需的数据，不应臆造参数，需向用户补充信息。",
}


def _intent(seed: Dict[str, Any]) -> str:
    base = INTENT_TMPL[seed["category"]]
    a0 = seed["gold_actions"][0]
    if a0["type"] == "ask_user":
        return f"{base} 缺少：{'、'.join(a0['missing'])}。{a0['reason']}。"
    if a0["type"] == "direct_answer":
        return f"{base}（{a0['reason']}）"
    if seed.get("sub_type") == "dga_from_context":
        return base + " 问题本身未给数值，但结构化上下文中已有前端填写的 DGA 数据，应填入 fault_attribution 的 dga_data。"
    if seed.get("sub_type") == "ett_forecast":
        h = seed["expected_hint"]
        return base + f" 数据集采样间隔为 {h['freq']}，{h['hours']} 小时应换算为 horizon={a0['arguments']['horizon']}。"
    return base


def _plan_json(seed: Dict[str, Any]) -> Dict[str, Any]:
    steps = []
    for i, a in enumerate([x for x in seed["gold_actions"] if x["type"] == "tool_call"], 1):
        steps.append({"id": i, "stage": STAGE_CN[a["tool"]],
                      "description": f"调用 {a['tool']} 完成{STAGE_CN[a['tool']]}",
                      "tool": a["tool"], "arguments": a["arguments"]})
    return {"intent_analysis": _intent(seed), "steps": steps}


def _messages(seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    system = PLANNER_SYSTEM_PROMPT()
    user = render_planner_user(query=seed["query"], context=PlannerAgent._render_context(seed.get("context") or {}))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def to_swift(seed: Dict[str, Any], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ms-swift agent 格式：assistant 用 tool_calls；无工具时 assistant 直接给 JSON 文本（intent + 空 steps）。"""
    msgs = _messages(seed)
    plan = _plan_json(seed)
    calls = [a for a in seed["gold_actions"] if a["type"] == "tool_call"]
    if calls:
        msgs.append({"role": "assistant", "content": plan["intent_analysis"],
                     "tool_calls": [{"type": "function",
                                     "function": {"name": a["tool"],
                                                  "arguments": json.dumps(a["arguments"], ensure_ascii=False)}}
                                    for a in calls]})
    else:
        msgs.append({"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)})
    return {"messages": msgs, "tools": tools, "meta": {"seed_id": seed["seed_id"], "category": seed["category"],
                                                        "sub_type": seed["sub_type"], "split": seed["split"]}}


def to_jsontext(seed: Dict[str, Any]) -> Dict[str, Any]:
    """与线上 json_text 回退路径一致：assistant 输出完整 plan JSON 字符串。"""
    msgs = _messages(seed)
    msgs.append({"role": "assistant", "content": json.dumps(_plan_json(seed), ensure_ascii=False)})
    return {"messages": msgs, "meta": {"seed_id": seed["seed_id"], "category": seed["category"], "split": seed["split"]}}


def to_llama_factory(seed: Dict[str, Any], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    msgs = _messages(seed)
    plan = _plan_json(seed)
    calls = [a for a in seed["gold_actions"] if a["type"] == "tool_call"]
    conv = [{"from": "human", "value": msgs[1]["content"]}]
    if calls:
        conv.append({"from": "function_call",
                     "value": json.dumps([{"name": a["tool"], "arguments": a["arguments"]} for a in calls],
                                         ensure_ascii=False) if len(calls) > 1
                     else json.dumps({"name": calls[0]["tool"], "arguments": calls[0]["arguments"]}, ensure_ascii=False)})
    else:
        conv.append({"from": "gpt", "value": json.dumps(plan, ensure_ascii=False)})
    return {"conversations": conv, "system": msgs[0]["content"],
            "tools": json.dumps([t["function"] for t in tools], ensure_ascii=False)}


# ── 多轮 / 错误恢复轨迹（build_multi_turn.py 产物）──
MULTI = ROOT / "data/planner/seeds/multi_turn_seeds.jsonl"


def _finish_content(step: Dict[str, Any]) -> str:
    """finish 决策的 assistant 文本：与线上 json_text 结构一致（intent + 空 steps），并带 ask_user 字段。"""
    obj: Dict[str, Any] = {"intent_analysis": step.get("thought", ""), "steps": []}
    if step.get("ask_user"):
        obj["ask_user"] = step["ask_user"]
    return json.dumps(obj, ensure_ascii=False)


def multi_turn_samples(rec: Dict[str, Any], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """每个 assistant 决策点展开为一条样本：messages 前缀 = system + user + 之前的 assistant/tool 消息。"""
    base = _messages(rec)
    out: List[Dict[str, Any]] = []
    hist: List[Dict[str, Any]] = []
    call_idx = 0
    for k, step in enumerate(rec["trajectory"]):
        if step["role"] == "tool":
            hist.append({"role": "tool", "tool_call_id": f"call_{call_idx}", "name": step["tool"],
                         "content": json.dumps(step["content"], ensure_ascii=False)})
            continue
        # assistant 决策点
        if "tool_call" in step:
            call_idx += 1
            target = {"role": "assistant", "content": step.get("thought", ""),
                      "tool_calls": [{"id": f"call_{call_idx}", "type": "function",
                                      "function": {"name": step["tool_call"]["tool"],
                                                   "arguments": json.dumps(step["tool_call"]["arguments"], ensure_ascii=False)}}]}
        else:
            target = {"role": "assistant", "content": _finish_content(step)}
        out.append({"messages": base + hist + [target], "tools": tools,
                    "meta": {"seed_id": rec["seed_id"], "category": rec["category"], "sub_type": rec["sub_type"],
                             "split": rec["split"], "decision_index": len(out), "n_prior_calls": call_idx - (1 if "tool_call" in step else 0)}})
        hist.append(target)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(SEEDS))
    ap.add_argument("--include-composite", action="store_true", default=True)
    ap.add_argument("--no-test", action="store_true", help="不导出 test（封存，训练前不查看）")
    ap.add_argument("--no-multi-turn", action="store_true", help="不导出多轮/错误恢复样本")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    tools = to_openai_tools()
    OUT.mkdir(parents=True, exist_ok=True)
    splits = ["train", "dev"] + ([] if args.no_test else ["test"])
    stats = Counter()
    for sp in splits:
        rows = [s for s in seeds if s["split"] == sp]
        with open(OUT / f"swift_{sp}.jsonl", "w", encoding="utf-8") as f1, \
             open(OUT / f"jsontext_{sp}.jsonl", "w", encoding="utf-8") as f2:
            lf = []
            for s in rows:
                f1.write(json.dumps(to_swift(s, tools), ensure_ascii=False) + "\n")
                f2.write(json.dumps(to_jsontext(s), ensure_ascii=False) + "\n")
                lf.append(to_llama_factory(s, tools))
                stats[(sp, s["category"])] += 1
        json.dump(lf, open(OUT / f"lf_{sp}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{sp}: {len(rows)} 条 → swift_{sp}.jsonl / jsontext_{sp}.jsonl / lf_{sp}.json")

    # 多轮 / 错误恢复（ms-swift messages 格式，含 tool 角色）
    mt_stats = Counter()
    if MULTI.exists() and not args.no_multi_turn:
        recs = [json.loads(l) for l in open(MULTI, encoding="utf-8") if l.strip()]
        for sp in splits:
            n = 0
            with open(OUT / f"swift_multiturn_{sp}.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    if r["split"] != sp:
                        continue
                    for smp in multi_turn_samples(r, tools):
                        f.write(json.dumps(smp, ensure_ascii=False) + "\n")
                        n += 1
                        mt_stats[(sp, r["category"], r["sub_type"])] += 1
            print(f"{sp}: 多轮决策点样本 {n} 条 → swift_multiturn_{sp}.jsonl")

    # 数据卡片
    card = ["# Planner SFT 数据卡片（模板阶段，LLM 扩写前）\n",
            f"- 来源：`{Path(args.seeds).relative_to(ROOT)}`（P4 任务种子，金标动作已参数校验 + 真实执行）"
            + (f"；多轮/错误恢复：`{MULTI.relative_to(ROOT)}`（工具返回全部来自真实执行）" if mt_stats else ""),
            "- 系统提示：与线上 `PLANNER_SYSTEM_PROMPT()` 一致（含工具清单）；user 为 `render_planner_user` 渲染，context 走 `PlannerAgent._render_context`",
            "- 格式：ms-swift messages+tools（`swift_*.jsonl`）、LLaMA-Factory function-calling（`lf_*.json`）、JSON 文本规划（`jsontext_*.jsonl`）；"
            "多轮为 ms-swift messages（含 `tool` 角色，`swift_multiturn_*.jsonl`），每个 assistant 决策点一条样本",
            "- 切分：按 `group_key` 分组，train/dev/test 互不共享来源；test 封存",
            "- 已知偏差：问法为模板生成，多样性不足（待 P4-2 LLM 口语化扩写）；数值类 DGA 记录来自 3 个公开数据集，标签分布不均（过载过热/正常偏多）；"
            "图谱推理类受规则抽取图谱覆盖限制（90 节点/190 边）；多轮样本中 assistant 的 thought 为规则模板文本",
            "- 许可：文献数据仅用于内部研究；ETT 数据集 CC BY 4.0；DGA 数据集见 data/real/dga 来源说明",
            "", "## 规模（单轮）", "| split | category | n |", "|---|---|---|"]
    for (sp, c), v in sorted(stats.items()):
        card.append(f"| {sp} | {c} | {v} |")
    if mt_stats:
        card += ["", "## 规模（多轮决策点样本）", "| split | category | sub_type | n |", "|---|---|---|---|"]
        for (sp, c, st), v in sorted(mt_stats.items()):
            card.append(f"| {sp} | {c} | {st} | {v} |")
    (OUT / "DATA_CARD.md").write_text("\n".join(card), encoding="utf-8")
    print(f"数据卡片 → {OUT / 'DATA_CARD.md'}")


if __name__ == "__main__":
    main()
