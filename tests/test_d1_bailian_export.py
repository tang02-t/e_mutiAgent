#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D-1 / A-2：百炼 SFT 格式导出自检。

覆盖：
1. to_bailian_record 剥离 meta / tool.name、补 id、arguments 串化；
2. validate_bailian_record 能捕获典型错误（多余字段、tool_call_id 不对应、arguments 非 JSON、末条非 assistant）；
3. 真实种子 → to_swift / multi_turn_samples → 百炼记录 全部通过校验，tool_calls 能被 PlannerAgent._build_plan_from_tool_calls 解析；
4. 若 data/planner/sft/bailian_{train,dev}.jsonl 已导出，则逐行校验、无 test 文件、大小 < 200 MB。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.planner_data.bailian_format import (  # noqa: E402
    to_bailian_record, validate_bailian_record, validate_bailian_file, MAX_FILE_BYTES)
from scripts.planner_data import export_sft as ex  # noqa: E402
from src.tools.tool_registry import to_openai_tools  # noqa: E402
from src.agents.planner import PlannerAgent  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _tools():
    return to_openai_tools()


# ── 1. 转换 ─────────────────────────────────────────────
def test_convert_strips_and_normalizes():
    print("1. to_bailian_record 转换")
    sample = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "think",
             "tool_calls": [{"type": "function", "function": {"name": "rag_search", "arguments": {"query": "q"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "rag_search", "content": {"hits": []}},
            {"role": "assistant", "content": "{\"intent_analysis\":\"x\",\"steps\":[]}"},
        ],
        "tools": _tools(),
        "meta": {"seed_id": "x"},
    }
    rec = to_bailian_record(sample)
    check("剥离顶层 meta", set(rec.keys()) == {"messages", "tools"}, str(rec.keys()))
    tc = rec["messages"][2]["tool_calls"][0]
    check("补 id=call_1", tc["id"] == "call_1", tc.get("id"))
    check("arguments 串化", isinstance(tc["function"]["arguments"], str)
          and json.loads(tc["function"]["arguments"]) == {"query": "q"})
    tm = rec["messages"][3]
    check("tool 消息剥离 name、content 串化", set(tm.keys()) == {"role", "tool_call_id", "content"}
          and isinstance(tm["content"], str), str(tm))
    check("转换结果通过校验", validate_bailian_record(rec) == [], str(validate_bailian_record(rec)))


# ── 2. 校验器负例 ───────────────────────────────────────
def test_validator_catches_errors():
    print("2. validate_bailian_record 负例")
    base = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                         {"role": "assistant", "content": "A"}]}
    check("合法最小样本", validate_bailian_record(base) == [])

    bad = json.loads(json.dumps(base)); bad["meta"] = {}
    check("顶层多余字段", any("顶层多余" in e for e in validate_bailian_record(bad)))

    bad = json.loads(json.dumps(base)); bad["messages"][2]["weight"] = 1
    check("assistant 多余字段 weight", any("多余字段" in e for e in validate_bailian_record(bad)))

    bad = json.loads(json.dumps(base))
    bad["messages"][2] = {"role": "assistant", "content": "",
                          "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "f", "arguments": "{bad json"}}]}
    bad["messages"].append({"role": "tool", "tool_call_id": "c9", "content": "{}"})
    bad["messages"].append({"role": "assistant", "content": "done"})
    errs = validate_bailian_record(bad)
    check("arguments 非 JSON", any("不是合法 JSON" in e for e in errs), str(errs))
    check("tool_call_id 不对应", any("未对应" in e for e in errs), str(errs))

    bad = json.loads(json.dumps(base)); bad["messages"].append({"role": "user", "content": "again"})
    check("末条非 assistant", any("末条" in e for e in validate_bailian_record(bad)))

    bad = json.loads(json.dumps(base)); bad["messages"][2]["tool_calls"] = [
        {"id": "c1", "type": "function", "function": {"name": "not_a_tool", "arguments": "{}"}}]
    bad["tools"] = _tools()
    check("函数名不在 tools", any("不在 tools" in e for e in validate_bailian_record(bad)))


# ── 3. 真实种子端到端 ───────────────────────────────────
def test_real_seeds_roundtrip():
    print("3. 真实种子 → 百炼记录 → 校验 / 线上解析")
    tools = _tools()
    seeds = [json.loads(l) for l in open(ex.SEEDS, encoding="utf-8") if l.strip()]
    recs = [json.loads(l) for l in open(ex.MULTI, encoding="utf-8") if l.strip()] if ex.MULTI.exists() else []
    seeds_by_cat = {}
    for s in seeds:
        if s["split"] == "train":
            seeds_by_cat.setdefault(s["category"], s)
    n_ok = 0
    n_parse_ok = 0
    n_with_calls = 0
    planner = object.__new__(PlannerAgent)          # 不触发 LLM 客户端初始化
    planner.allowed_tools = None
    for s in seeds_by_cat.values():
        rec = to_bailian_record(ex.to_swift(s, tools))
        errs = validate_bailian_record(rec)
        n_ok += not errs
        last = rec["messages"][-1]
        if last.get("tool_calls"):
            n_with_calls += 1
            calls = [{"id": c["id"], "name": c["function"]["name"],
                      "arguments": json.loads(c["function"]["arguments"])} for c in last["tool_calls"]]
            plan = planner._build_plan_from_tool_calls(calls, last["content"])
            steps = plan.get("steps", [])
            n_parse_ok += (len(steps) == len(calls) and plan["plan_status"] == "ok"
                           and all(st["validation"]["valid"] for st in steps)
                           and all(st["call_id"] for st in steps))
    check(f"每类单轮种子各 1 条全部合法（{len(seeds_by_cat)} 类）", n_ok == len(seeds_by_cat))
    check(f"tool_calls 可被 _build_plan_from_tool_calls 解析且参数校验通过、call_id 保留（{n_with_calls} 条）",
          n_with_calls > 0 and n_parse_ok == n_with_calls, f"{n_parse_ok}/{n_with_calls}")

    mt_ok = mt_total = 0
    mt_with_tool_msg = 0
    seen = {}
    for r in recs:
        if r["split"] != "train":
            continue
        key = (r["category"], r["sub_type"])
        if key in seen:
            continue
        seen[key] = 1
        for smp in ex.multi_turn_samples(r, tools):
            rec = to_bailian_record(smp)
            mt_total += 1
            mt_ok += validate_bailian_record(rec) == []
            mt_with_tool_msg += any(m["role"] == "tool" for m in rec["messages"])
    check(f"多轮决策点样本全部合法（{len(seen)} 类，{mt_total} 条）", mt_total > 0 and mt_ok == mt_total,
          f"{mt_ok}/{mt_total}")
    check("多轮样本中存在 tool 角色消息（id 对应关系已校验）", mt_with_tool_msg > 0)


# ── 4. 已导出文件 ───────────────────────────────────────
def test_exported_files():
    print("4. 已导出的 bailian_{train,dev}.jsonl")
    out = ex.OUT
    for sp in ("train", "dev"):
        p = out / f"bailian_{sp}.jsonl"
        if not p.exists():
            check(f"{p.name} 存在（先运行 export_sft.py）", False)
            continue
        n, errs = validate_bailian_file(p)
        check(f"{p.name} {n} 行全部通过校验", not errs, "; ".join(errs[:3]))
        check(f"{p.name} 大小 < 200 MB", p.stat().st_size < MAX_FILE_BYTES)
    check("不存在 bailian_test.jsonl（test 封存）", not (out / "bailian_test.jsonl").exists())
    p = out / "bailian_train.jsonl"
    if p.exists():
        swift_n = sum(1 for l in open(out / "swift_train.jsonl", encoding="utf-8") if l.strip())
        mt_p = out / "swift_multiturn_train.jsonl"
        mt_n = sum(1 for l in open(mt_p, encoding="utf-8") if l.strip()) if mt_p.exists() else 0
        bl_n = sum(1 for l in open(p, encoding="utf-8") if l.strip())
        check(f"bailian_train 行数 = swift_train + swift_multiturn_train（{swift_n}+{mt_n}）", bl_n == swift_n + mt_n, str(bl_n))


if __name__ == "__main__":
    test_convert_strips_and_normalizes()
    test_validator_catches_errors()
    test_real_seeds_roundtrip()
    test_exported_files()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
