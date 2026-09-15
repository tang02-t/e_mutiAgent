#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D-2 偏好对构造管线自检（无 LLM）。

覆盖：
1. parse_candidate：tool_calls / json_text / 坏格式 三条路径；
2. score_candidate：六分项与成本惩罚方向正确（金标 = 1.0；换错工具 / Schema 错 / 多余调用 / 该问不问 / 格式坏 各自扣对应项）；
3. make_pairs：分差阈值、每 prompt ≤ 1 对、可复现；
4. to_dpo_record / validate_dpo_record：百炼 DPO 结构（messages 以 user 结尾、chosen ≠ rejected、tool_calls 合法）；
5. leakage_check：train 通过、test / D10 命中被识别；
6. 端到端：金标扰动 demo 小样本（真实工具执行）→ 双子集偏好对 ≥ prompt 数 × 0.9、非法 0、泄漏 0。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.planner_data import score_candidates as SC  # noqa: E402
from src.tools.tool_registry import to_openai_tools  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name} {detail}")


SEEDS = [json.loads(l) for l in open(SC.SEEDS, encoding="utf-8") if l.strip()]
BY_ID = {s["seed_id"]: s for s in SEEDS}


def pick(category, split="train"):
    return next(s for s in SEEDS if s["category"] == category and s["split"] == split)


def test_parse():
    print("1. parse_candidate")
    p = SC.parse_candidate({"content": "x", "tool_calls": [{"name": "rag_search", "arguments": "{\"query\": \"q\"}"}]})
    check("tool_calls 字符串参数", p.format_ok and p.steps == [{"tool": "rag_search", "arguments": {"query": "q"}}])
    p = SC.parse_candidate({"content": json.dumps({"intent_analysis": "i", "steps": [{"id": 1, "tool": "kg_search", "arguments": {"query": "k"}}], "ask_user": ["a"]})})
    check("json_text 计划 + ask_user", p.format_ok and p.steps[0]["tool"] == "kg_search" and p.ask_user == ["a"])
    p = SC.parse_candidate({"content": "```json\n{\"intent_analysis\": \"i\", \"steps\": []}\n```"})
    check("代码围栏 json", p.format_ok and p.steps == [])
    check("坏格式", not SC.parse_candidate({"content": "{\"intent"}).format_ok)
    check("tool_calls 参数非 JSON", not SC.parse_candidate({"tool_calls": [{"name": "rag_search", "arguments": "{bad"}]}).format_ok)


def test_score_offline():
    print("2. score_candidate（不执行工具，exec_res 手工给定）")
    s = pick("numeric_tool")
    gold = SC.parse_candidate(SC._gold_raw(s))
    ok = {"n_calls": 1, "business_all": True, "schema_all": True, "n_claims": 5, "unsupported_ratio": 0.0, "claim_verdict": "PASS"}
    r = SC.score_candidate(s, gold, ok)
    check("金标 full=1.0 exec=1.0", r["score_full"] == 1.0 and r["score_exec"] == 1.0, str(r))
    r2 = SC.score_candidate(s, gold, {**ok, "unsupported_ratio": 0.4})
    check("faith = 1 − unsupported_ratio 只影响 full", abs(r2["parts"]["faith"] - 0.6) < 1e-9 and r2["score_exec"] == 1.0 and r2["score_full"] < 1.0)
    bad_tool = SC.ParsedPlan(True, steps=[{"tool": "kg_search", "arguments": {"query": "x"}}])
    r3 = SC.score_candidate(s, bad_tool, ok)
    check("换错工具 tool=0", r3["parts"]["tool"] == 0.0 and r3["parts"]["format"] == 1.0)
    bad_schema = SC.ParsedPlan(True, steps=[{"tool": "fault_attribution", "arguments": {"dga_data": {"H2": 1}, "top_k": 3}}])
    r4 = SC.score_candidate(s, bad_schema, None)
    check("Schema 错 schema=0 且未执行 business=0", r4["parts"]["schema"] == 0.0 and r4["parts"]["business"] == 0.0)
    extra = SC.ParsedPlan(True, steps=gold.steps + [{"tool": "rag_search", "arguments": {"query": "a"}}, {"tool": "kg_search", "arguments": {"query": "b"}}])
    r5 = SC.score_candidate(s, extra, ok)
    check("多余 2 次调用：extra_calls=2、full 扣 0.2、exec 不扣成本", r5["extra_calls"] == 2
          and abs(r5["score_full"] - (5 / 6 - 0.2)) < 1e-3 and r5["score_exec"] == 0.75, str(r5))
    check("格式坏 全 0", SC.score_candidate(s, SC.ParsedPlan(False), None)["score_full"] == 0.0)

    a = pick("insufficient")
    gold_a = SC.parse_candidate(SC._gold_raw(a))
    ra = SC.score_candidate(a, gold_a, None)
    check("ask_user 金标：追问命中 → ask=1，无工具 schema/business/faith=1", ra["score_full"] == 1.0, str(ra))
    no_ask = SC.ParsedPlan(True, steps=[], ask_user=[])
    check("该问不问 ask=0", SC.score_candidate(a, no_ask, None)["parts"]["ask"] == 0.0)
    wrong_ask = SC.ParsedPlan(True, steps=[], ask_user=["请提供设备型号"])
    check("追问未命中缺失项 ask=0", SC.score_candidate(a, wrong_ask, None)["parts"]["ask"] == 0.0)
    n = pick("no_tool")
    check("no_tool 金标：多问一句 ask=0", SC.score_candidate(n, SC.ParsedPlan(True, steps=[], ask_user=["x"]), None)["parts"]["ask"] == 0.0)
    check("no_tool 金标：多调一次工具 tool=0 且扣成本", SC.score_candidate(
        n, SC.ParsedPlan(True, steps=[{"tool": "rag_search", "arguments": {"query": "q"}}]), ok)["extra_calls"] == 1)


def test_pairs():
    print("3. make_pairs")
    rows = [{"seed_id": "A", "candidate_id": "A#1", "score_full": 1.0}, {"seed_id": "A", "candidate_id": "A#2", "score_full": 0.5},
            {"seed_id": "A", "candidate_id": "A#3", "score_full": 0.9}, {"seed_id": "B", "candidate_id": "B#1", "score_full": 0.8},
            {"seed_id": "B", "candidate_id": "B#2", "score_full": 0.6}]
    p = SC.make_pairs(rows, "score_full")
    check("A 配对最高 vs 最低、B 分差 0.2 丢弃", len(p) == 1 and p[0]["chosen"]["candidate_id"] == "A#1" and p[0]["rejected"]["candidate_id"] == "A#2")
    check("可复现（两次结果一致）", SC.make_pairs(rows, "score_full") == p)


def test_dpo_record():
    print("4. to_dpo_record / validate_dpo_record")
    s = pick("composite")
    tools = to_openai_tools()
    gold = {"candidate_id": "g", "raw": SC._gold_raw(s)}
    bad = {"candidate_id": "b", "raw": {"content": "{\"intent"}}
    pair = {"seed_id": s["seed_id"], "chosen": gold, "rejected": bad}
    rec = SC.to_dpo_record(s, pair, tools, "tool_calls")
    errs = SC.validate_dpo_record(rec)
    check("tool_calls 风格记录合法", not errs, str(errs))
    check("messages 以 user 结尾且 chosen 含 2 个 tool_calls 带 id", rec["messages"][-1]["role"] == "user"
          and len(rec["chosen"]["tool_calls"]) == 2 and all(c["id"] for c in rec["chosen"]["tool_calls"]))
    rec2 = SC.to_dpo_record(s, pair, tools, "jsontext")
    check("jsontext 风格：chosen 为 JSON 计划文本且无 tools", "tool_calls" not in rec2["chosen"]
          and json.loads(rec2["chosen"]["content"])["steps"][0]["tool"] == "fault_attribution" and "tools" not in rec2)
    same = {"seed_id": s["seed_id"], "chosen": gold, "rejected": gold}
    check("chosen == rejected 被拒", any("相同" in e for e in SC.validate_dpo_record(SC.to_dpo_record(s, same, tools, "tool_calls"))))


def test_leakage():
    print("5. leakage_check")
    tr = pick("fact", "train")["seed_id"]; te = pick("fact", "test")["seed_id"]
    r = SC.leakage_check([tr, te], BY_ID)
    check("train 通过、test 命中", r["n_leaks"] == 1 and r["leaks"][0]["seed_id"] == te and "D8-test" in r["leaks"][0]["hit"])


def test_end2end_demo():
    print("6. 端到端 demo（金标扰动 + 真实工具）")
    rng = random.Random(7)
    prompts = SC.stratified_train_seeds(SEEDS, 12, rng)
    cands = [c for s in prompts for c in SC.synthetic_candidates(s, rng)]
    summary = SC.run(cands, BY_ID, execute=True, dpo_style="tool_calls", tag="unittest")
    check(f"gold 平均 full 分 = 1.0（{summary['source_mean_full'].get('gold')}）", summary["source_mean_full"].get("gold") == 1.0)
    check("每类扰动平均分均 < gold", all(v < 1.0 for k, v in summary["source_mean_full"].items() if k != "gold"), str(summary["source_mean_full"]))
    n_ask = sum(1 for s in prompts if s["category"] == "insufficient")
    v = summary["subsets"]["exec"]
    check(f"D13-exec：偏好对 {v['n_pairs']} ≥ 非追问类 {len(prompts) - n_ask} × 0.9、非法 0、泄漏 0（exec 分不含 ask 项，追问类可无分差）",
          v["n_pairs"] >= 0.9 * (len(prompts) - n_ask) and v["n_invalid_records"] == 0 and v["leakage"]["n_leaks"] == 0)
    v = summary["subsets"]["full"]
    check(f"D13-full：偏好对 {v['n_pairs']} ≥ {len(prompts)} × 0.9、非法 0、泄漏 0",
          v["n_pairs"] >= 0.9 * len(prompts) and v["n_invalid_records"] == 0 and v["leakage"]["n_leaks"] == 0)
    check("full 子集偏好对数 ≥ exec 子集（验证信号带来额外可分性）", summary["subsets"]["full"]["n_pairs"] >= summary["subsets"]["exec"]["n_pairs"])
    p = SC.OUT / "d13_full_unittest.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    check("导出文件每条 validate_dpo_record 通过", all(not SC.validate_dpo_record(r) for r in rows))
    for f in SC.OUT.glob("*_unittest*"):
        f.unlink()


if __name__ == "__main__":
    test_parse(); test_score_offline(); test_pairs(); test_dpo_record(); test_leakage(); test_end2end_demo()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
