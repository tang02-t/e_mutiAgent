#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-2 偏好对构造（数据 D13）：候选打分 → 配对 → D13-exec / D13-full 双导出 → 百炼 DPO 格式 → 泄漏检查 → 数据卡片。

输入（候选文件，每行一条；由 M1 / M0 采样产生，后置；本脚本亦可 `--synthetic-demo` 用金标扰动生成候选以验证管线）：
    {"seed_id": "...", "candidate_id": "...", "source": "m1_t0.7 | m1_t1.0 | m0 | gold | perturb:*",
     "raw": {"content": "<assistant 文本>", "tool_calls": [{"name": ..., "arguments": {...} | "<json>"}]}}
`raw` 与线上 Planner 的两条解析路径一致：有 tool_calls 走 function_calling；否则 content 按 json_text 计划解析。

打分（每项 0~1，权重可配；总分 = 加权和 / 权重和 − 0.1 × 多余调用数，下限 0）：
    format      候选可解析为合法计划（steps 列表、tool 为字符串、arguments 为对象）
    tool        工具多重集与金标一致（金标 ask_user / direct_answer → 候选不调用工具）
    schema      所有步骤通过 validate_arguments_strict（金标要求工具而候选为空 → 0）
    business    真实执行后所有调用 business_success（同上）
    faith       声明忠实度 = 1 − unsupported_ratio（候选轨迹 → Generator 规则声明 → ClaimChecker）
    ask         金标 ask_user 时候选是否追问且命中缺失项；金标非 ask_user 时候选不追问
    D13-exec 只用 format / tool / schema / business；D13-full 用全部六项 + 成本惩罚。

配对：同一 prompt 内 full 分（或 exec 分）最高与最低候选分差 ≥ 0.3 → (chosen, rejected)，每 prompt 最多 1 对。

用法：
    python3 scripts/planner_data/score_candidates.py --candidates data/planner/dpo/candidates.jsonl
    python3 scripts/planner_data/score_candidates.py --synthetic-demo --n 60 --seed 20260915
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                 # noqa: E402
from src.graph.workflow import run_diagnosis_workflow                  # noqa: E402
from src.graph.system_modes import tool_stack_spec, build_agents         # noqa: E402
from src.agents.generator import GeneratorAgent                        # noqa: E402
from src.agents.claim_checker import ClaimChecker                      # noqa: E402
from src.tools.tool_registry import validate_arguments_strict, get_tool_names, to_openai_tools  # noqa: E402
from src.tools.fault_attribution import configure_engine               # noqa: E402
from scripts.planner_data.export_sft import _messages, SEEDS, MULTI    # noqa: E402
from scripts.planner_data.bailian_format import validate_bailian_record  # noqa: E402

D10 = ROOT / "data/eval/d10/end2end_eval.jsonl"
OUT = ROOT / "data/planner/dpo"

NO_LLM_CFG: Dict[str, Any] = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 1, "planner_strategy": "free", "attribution_mode": "calibrated",
                 "generator_output_mode": "claims"},
}
WEIGHTS_EXEC = {"format": 1.0, "tool": 1.0, "schema": 1.0, "business": 1.0}
WEIGHTS_FULL = {"format": 1.0, "tool": 1.0, "schema": 1.0, "business": 1.0, "faith": 1.0, "ask": 1.0}
COST_PENALTY = 0.1
PAIR_MIN_GAP = 0.3


# ──────────────────────────────────────────────────────────────
# 候选解析
# ──────────────────────────────────────────────────────────────
@dataclass
class ParsedPlan:
    format_ok: bool
    steps: List[Dict[str, Any]] = field(default_factory=list)      # {tool, arguments}
    ask_user: List[str] = field(default_factory=list)
    intent: str = ""
    error: str = ""


def parse_candidate(raw: Dict[str, Any]) -> ParsedPlan:
    """与线上 Planner 一致：tool_calls 优先；否则 content 视为 json_text 计划。"""
    calls = raw.get("tool_calls")
    content = raw.get("content") or ""
    if calls:
        steps = []
        for c in calls:
            fn = c.get("function", c)
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return ParsedPlan(False, error="tool_calls.arguments 不是合法 JSON")
            if not isinstance(name, str) or not isinstance(args, dict):
                return ParsedPlan(False, error="tool_calls 结构非法")
            steps.append({"tool": name, "arguments": args})
        return ParsedPlan(True, steps=steps, intent=content)
    txt = content.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return ParsedPlan(False, error="content 不是合法 JSON 计划")
    if not isinstance(obj, dict) or not isinstance(obj.get("steps", []), list):
        return ParsedPlan(False, error="计划缺少 steps 列表")
    steps = []
    for s in obj.get("steps", []):
        if not isinstance(s, dict) or not isinstance(s.get("tool"), str) or not isinstance(s.get("arguments", {}), dict):
            return ParsedPlan(False, error="step 结构非法")
        steps.append({"tool": s["tool"], "arguments": s.get("arguments") or {}})
    ask = obj.get("ask_user") or []
    if not isinstance(ask, list):
        ask = [str(ask)]
    return ParsedPlan(True, steps=steps, ask_user=[str(a) for a in ask], intent=str(obj.get("intent_analysis", "")))


# ──────────────────────────────────────────────────────────────
# 执行：候选计划 → 真实工具 → 规则声明 → ClaimChecker
# ──────────────────────────────────────────────────────────────
class CandidatePlanner:
    """把已解析的候选计划注入工作流（不调用 LLM）。"""
    planner_mode = "candidate"
    model_name = "candidate"

    def __init__(self, allowed_tools: Optional[List[str]] = None) -> None:
        self.allowed = set(allowed_tools) if allowed_tools is not None else None
        self._plan = ParsedPlan(False)

    def bind(self, plan: ParsedPlan) -> None:
        self._plan = plan

    def run(self, state: AgentState) -> AgentState:
        valid = set(get_tool_names())
        steps = []
        for i, s in enumerate(self._plan.steps, 1):
            step = {"id": i, "stage": "candidate", "description": "候选动作", "tool": s["tool"], "arguments": s["arguments"]}
            if s["tool"] not in valid:
                step["validation"] = {"valid": False, "errors": [{"field": None, "code": "unknown_tool", "message": "未知工具"}]}
            elif self.allowed is not None and s["tool"] not in self.allowed:
                step["validation"] = {"valid": False, "errors": [{"field": None, "code": "tool_disabled", "message": "模式外"}]}
            else:
                step["validation"] = validate_arguments_strict(s["tool"], s["arguments"])
            steps.append(step)
        plan = {"intent_analysis": self._plan.intent, "steps": steps, "plan_status": "ok" if steps else "no_tool",
                "planner_mode": "candidate"}
        state.plan_status = plan["plan_status"]
        state.plan = self._plan.intent
        state.reasoning_trace.append({"agent": "planner", "type": "llm_plan", "content": plan})
        return state


class Executor:
    def __init__(self) -> None:
        logging.disable(logging.WARNING)
        from eval_system_modes import build_mcp
        configure_engine("calibrated")
        spec = tool_stack_spec(NO_LLM_CFG)
        mcp, self.kb_desc = build_mcp(spec)
        _, self.retriever, _, self.validator, self.reflector = build_agents(NO_LLM_CFG, mcp, spec)
        self.generator = GeneratorAgent(NO_LLM_CFG, output_mode="claims")
        self.planner = CandidatePlanner(spec.allowed_tools)
        self.checker = ClaimChecker(llm=None)

    def run(self, seed: Dict[str, Any], plan: ParsedPlan) -> Dict[str, Any]:
        self.planner.bind(plan)
        st = AgentState(user_query=seed["query"], context=dict(seed.get("context") or {}), max_iterations=1)
        final = run_diagnosis_workflow(st, self.planner, self.retriever, self.generator, self.validator,
                                       reflector=self.reflector)
        traj = final.trajectory
        r = self.checker.check(final)
        return {"n_calls": len(traj), "business_all": all(t["business_success"] for t in traj),
                "schema_all": all((t.get("validation") or {}).get("valid") for t in traj),
                "n_claims": r.n_claims, "unsupported_ratio": r.unsupported_ratio, "claim_verdict": r.verdict}


# ──────────────────────────────────────────────────────────────
# 打分
# ──────────────────────────────────────────────────────────────
def _gold_info(seed: Dict[str, Any]) -> Dict[str, Any]:
    acts = seed["gold_actions"]
    tools = sorted(a["tool"] for a in acts if a["type"] == "tool_call")
    ask = [m for a in acts if a["type"] == "ask_user" for m in a.get("missing", [])]
    return {"tools": tools, "n_calls": len(tools), "ask": ask, "kind": acts[0]["type"]}


def _bigram(s: str) -> set:
    s = "".join(ch for ch in s.lower() if not ch.isspace())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def _ask_hit(cand: List[str], gold_missing: List[str]) -> bool:
    for c in cand:
        for g in gold_missing:
            if g in c or c in g:
                return True
            a, b = _bigram(c), _bigram(g)
            if len(a & b) / max(1, len(a | b)) >= 0.3:
                return True
    return False


def score_candidate(seed: Dict[str, Any], plan: ParsedPlan, exec_res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """返回分项与两种总分。exec_res=None 表示未执行（format 失败时）。"""
    g = _gold_info(seed)
    parts: Dict[str, float] = {}
    parts["format"] = 1.0 if plan.format_ok else 0.0
    if not plan.format_ok:
        parts.update({"tool": 0.0, "schema": 0.0, "business": 0.0, "faith": 0.0, "ask": 0.0})
        n_calls = 0
    else:
        cand_tools = sorted(s["tool"] for s in plan.steps)
        n_calls = len(cand_tools)
        parts["tool"] = 1.0 if cand_tools == g["tools"] else 0.0
        need_tools = g["n_calls"] > 0
        if n_calls == 0:
            parts["schema"] = 0.0 if need_tools else 1.0
            parts["business"] = 0.0 if need_tools else 1.0
            parts["faith"] = 0.0 if need_tools else 1.0
        else:
            parts["schema"] = 1.0 if all(validate_arguments_strict(s["tool"], s["arguments"])["valid"] for s in plan.steps) else 0.0
            if exec_res is None:
                parts["business"] = 0.0
                parts["faith"] = 0.0
            else:
                parts["business"] = 1.0 if exec_res["business_all"] else 0.0
                parts["faith"] = (1.0 - exec_res["unsupported_ratio"]) if exec_res["n_claims"] > 0 else 0.0
        if g["kind"] == "ask_user":
            parts["ask"] = 1.0 if (plan.ask_user and _ask_hit(plan.ask_user, g["ask"])) else 0.0
        else:
            parts["ask"] = 0.0 if plan.ask_user else 1.0
    extra = max(0, n_calls - g["n_calls"])
    penalty = COST_PENALTY * extra

    def total(w: Dict[str, float], with_cost: bool) -> float:
        s = sum(w[k] * parts[k] for k in w) / sum(w.values())
        return round(max(0.0, s - (penalty if with_cost else 0.0)), 4)

    return {"parts": parts, "extra_calls": extra, "score_exec": total(WEIGHTS_EXEC, False),
            "score_full": total(WEIGHTS_FULL, True)}


# ──────────────────────────────────────────────────────────────
# 配对与导出
# ──────────────────────────────────────────────────────────────
def make_pairs(scored: List[Dict[str, Any]], key: str, min_gap: float = PAIR_MIN_GAP) -> List[Dict[str, Any]]:
    """每 prompt 最多 1 对：最高分 vs 最低分，分差 ≥ min_gap；同分并列取 candidate_id 字典序保证可复现。"""
    by_seed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_seed[r["seed_id"]].append(r)
    pairs = []
    for sid in sorted(by_seed):
        rows = sorted(by_seed[sid], key=lambda r: (r[key], r["candidate_id"]))
        lo, hi = rows[0], rows[-1]
        if hi[key] - lo[key] >= min_gap:
            pairs.append({"seed_id": sid, "chosen": hi, "rejected": lo, "gap": round(hi[key] - lo[key], 4), "score_key": key})
    return pairs


def _assistant_msg(raw: Dict[str, Any], style: str) -> Dict[str, Any]:
    """chosen / rejected 的 assistant 消息。tool_calls 风格与 SFT 训练分布一致；jsontext 为备选。"""
    plan = parse_candidate(raw)
    if style == "tool_calls" and raw.get("tool_calls"):
        calls = []
        for k, c in enumerate(raw["tool_calls"], 1):
            fn = c.get("function", c)
            args = fn.get("arguments", {})
            calls.append({"id": c.get("id") or f"call_{k}", "type": "function",
                          "function": {"name": fn.get("name"), "arguments": args if isinstance(args, str)
                                       else json.dumps(args, ensure_ascii=False)}})
        return {"role": "assistant", "content": raw.get("content") or "", "tool_calls": calls}
    if raw.get("tool_calls"):      # jsontext 风格：把 tool_calls 折叠成 JSON 计划文本
        obj = {"intent_analysis": raw.get("content") or "", "steps": [
            {"id": i, "tool": s["tool"], "arguments": s["arguments"]} for i, s in enumerate(plan.steps, 1)]}
        return {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}
    return {"role": "assistant", "content": raw.get("content") or ""}


def to_dpo_record(seed: Dict[str, Any], pair: Dict[str, Any], tools: List[Dict[str, Any]], style: str) -> Dict[str, Any]:
    """百炼 DPO：messages（以 user 结尾）+ chosen + rejected（assistant 消息）+ tools。"""
    rec = {"messages": _messages(seed), "chosen": _assistant_msg(pair["chosen"]["raw"], style),
           "rejected": _assistant_msg(pair["rejected"]["raw"], style)}
    if style == "tool_calls":
        rec["tools"] = tools
    return rec


def validate_dpo_record(rec: Dict[str, Any]) -> List[str]:
    errs = []
    if rec["messages"][-1]["role"] != "user":
        errs.append("messages 须以 user 结尾")
    if rec["chosen"] == rec["rejected"]:
        errs.append("chosen 与 rejected 相同")
    for k in ("chosen", "rejected"):
        m = rec[k]
        if m.get("role") != "assistant":
            errs.append(f"{k}.role 须为 assistant")
        probe = {"messages": rec["messages"] + [m]}
        if "tools" in rec:
            probe["tools"] = rec["tools"]
        errs += [f"{k}: {e}" for e in validate_bailian_record(probe)]
    return errs


def leakage_check(pair_seed_ids: List[str], seeds_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """prompt 的 seed_source / group_key 不得出现在 D8 test 与 D10。"""
    test_src = {s["seed_source"] for s in seeds_by_id.values() if s["split"] == "test"}
    test_grp = {s["group_key"] for s in seeds_by_id.values() if s["split"] == "test"}
    d10_src, d10_grp = set(), set()
    if D10.exists():
        for l in open(D10, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                d10_src.add(r.get("seed_source")); d10_grp.add(r.get("group_key"))
    leaks = []
    for sid in pair_seed_ids:
        s = seeds_by_id[sid]
        hit = []
        if s["split"] != "train":
            hit.append(f"split={s['split']}")
        if s["seed_source"] in test_src or s["group_key"] in test_grp:
            hit.append("D8-test")
        if s["seed_source"] in d10_src or s["group_key"] in d10_grp:
            hit.append("D10")
        if hit:
            leaks.append({"seed_id": sid, "hit": hit})
    return {"n_checked": len(pair_seed_ids), "n_leaks": len(leaks), "leaks": leaks[:20]}


# ──────────────────────────────────────────────────────────────
# 金标扰动候选（管线验证；正式候选来自 M1 / M0 采样）
# ──────────────────────────────────────────────────────────────
def _gold_raw(seed: Dict[str, Any]) -> Dict[str, Any]:
    calls = [a for a in seed["gold_actions"] if a["type"] == "tool_call"]
    if calls:
        return {"content": "按金标调用工具。", "tool_calls": [{"name": a["tool"], "arguments": a["arguments"]} for a in calls]}
    a0 = seed["gold_actions"][0]
    obj: Dict[str, Any] = {"intent_analysis": a0.get("reason", ""), "steps": []}
    if a0["type"] == "ask_user":
        obj["ask_user"] = list(a0["missing"])
    return {"content": json.dumps(obj, ensure_ascii=False)}


def synthetic_candidates(seed: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    gold = _gold_raw(seed)
    out = [{"source": "gold", "raw": gold}]
    tools = get_tool_names()
    calls = gold.get("tool_calls") or []
    # 1) 换错工具 / 无中生有调用
    if calls:
        c = copy.deepcopy(calls)
        c[0]["name"] = rng.choice([t for t in tools if t != c[0]["name"]])
        out.append({"source": "perturb:wrong_tool", "raw": {"content": gold["content"], "tool_calls": c}})
    else:
        out.append({"source": "perturb:spurious_call", "raw": {
            "content": "先检索。", "tool_calls": [{"name": "rag_search", "arguments": {"query": seed["query"][:30]}}]}})
    # 2) 参数 Schema 错误（未知字段 / 枚举错误）
    if calls:
        c = copy.deepcopy(calls)
        if c[0]["name"] == "ett_forecast":
            c[0]["arguments"] = {**c[0]["arguments"], "dataset": "ETTh3"}
        else:
            c[0]["arguments"] = {**c[0]["arguments"], "top_k": 5}
        out.append({"source": "perturb:bad_schema", "raw": {"content": gold["content"], "tool_calls": c}})
    # 3) 多余调用（成本惩罚；查询选图谱 / 知识库中确有结果的词，避免与业务失败混淆）
    c = copy.deepcopy(calls) + [{"name": "kg_search", "arguments": {"query": "局部放电"}},
                                {"name": "rag_search", "arguments": {"query": "变压器 检修"}}]
    out.append({"source": "perturb:extra_calls", "raw": {"content": gold.get("content", ""), "tool_calls": c}})
    # 4) 格式坏 / 该问不问
    if seed["gold_actions"][0]["type"] == "ask_user":
        out.append({"source": "perturb:no_ask", "raw": {"content": json.dumps({"intent_analysis": "直接回答。", "steps": []}, ensure_ascii=False)}})
    else:
        out.append({"source": "perturb:bad_format", "raw": {"content": "{\"intent_analysis\": \"..."}})
    for k, o in enumerate(out):
        o["seed_id"] = seed["seed_id"]
        o["candidate_id"] = f"{seed['seed_id']}#{k}:{o['source']}"
    return out


def stratified_train_seeds(seeds: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in seeds:
        if s["split"] == "train":
            by_cat[s["category"]].append(s)
    per = max(1, n // len(by_cat))
    out = []
    for c in sorted(by_cat):
        rows = by_cat[c]; rng.shuffle(rows)
        out += rows[:per]
    return out


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def run(candidates: List[Dict[str, Any]], seeds_by_id: Dict[str, Dict[str, Any]], *, execute: bool = True,
        dpo_style: str = "tool_calls", tag: str = "") -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = Executor() if execute else None
    tools = to_openai_tools()
    scored: List[Dict[str, Any]] = []
    t0 = time.time()
    for c in candidates:
        seed = seeds_by_id[c["seed_id"]]
        plan = parse_candidate(c["raw"])
        res = ex.run(seed, plan) if (ex and plan.format_ok and plan.steps) else None
        sc = score_candidate(seed, plan, res)
        scored.append({**c, "parse_error": plan.error, "exec": res, **sc})
    elapsed = round(time.time() - t0, 2)
    sfx = f"_{tag}" if tag else ""
    with open(OUT / f"scored{sfx}.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {"n_candidates": len(scored), "n_prompts": len({r['seed_id'] for r in scored}),
                               "elapsed": elapsed, "rag": ex.kb_desc if ex else None, "dpo_style": dpo_style,
                               "source_mean_full": {}, "subsets": {}}
    by_src: Dict[str, List[float]] = defaultdict(list)
    for r in scored:
        by_src[r["source"]].append(r["score_full"])
    summary["source_mean_full"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_src.items())}

    for subset, key in (("exec", "score_exec"), ("full", "score_full")):
        pairs = make_pairs(scored, key)
        n_bad = 0
        with open(OUT / f"d13_{subset}{sfx}.jsonl", "w", encoding="utf-8") as f:
            for p in pairs:
                rec = to_dpo_record(seeds_by_id[p["seed_id"]], p, tools, dpo_style)
                errs = validate_dpo_record(rec)
                n_bad += bool(errs)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(OUT / f"d13_{subset}{sfx}_pairs_meta.jsonl", "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps({"seed_id": p["seed_id"], "gap": p["gap"], "score_key": key,
                                    "chosen": p["chosen"]["candidate_id"], "rejected": p["rejected"]["candidate_id"],
                                    "chosen_parts": p["chosen"]["parts"], "rejected_parts": p["rejected"]["parts"]},
                                   ensure_ascii=False) + "\n")
        leak = leakage_check([p["seed_id"] for p in pairs], seeds_by_id)
        summary["subsets"][subset] = {"n_pairs": len(pairs), "n_invalid_records": n_bad,
                                      "mean_gap": round(sum(p["gap"] for p in pairs) / len(pairs), 4) if pairs else 0.0,
                                      "leakage": leak}
    (OUT / f"summary{sfx}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_data_card(summary: Dict[str, Any], demo: bool) -> None:
    lines = ["# D13 偏好对数据卡片", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/planner_data/score_candidates.py`",
             f"- 候选来源：{'金标扰动（synthetic-demo，仅用于验证打分 / 配对 / 导出管线，**不用于训练**）' if demo else 'M1（temperature 0.7 / 1.0 各 2）+ M0 基座 1，共 5 候选 / prompt'}",
             f"- prompt 数 {summary['n_prompts']}，候选数 {summary['n_candidates']}，真实工具执行（rag={summary['rag']}），耗时 {summary['elapsed']} s，Token 0",
             "- 打分：format / tool / schema / business / faith(1 − unsupported_ratio，Generator 规则声明 + ClaimChecker) / ask；总分 = 加权和 / 权重和 − 0.1 × 多余调用（下限 0）",
             f"- 配对：同 prompt 最高 vs 最低，分差 ≥ {PAIR_MIN_GAP}，每 prompt ≤ 1 对；D13-exec 按 4 项 exec 分，D13-full 按 6 项 + 成本",
             f"- 导出：百炼 DPO `messages`（system + user，以 user 结尾）+ `chosen` / `rejected`（assistant，风格 `{summary['dpo_style']}`）+ `tools`；每条经 `validate_bailian_record` 复核",
             "- 泄漏检查：prompt 的 `seed_source` / `group_key` 不出现在 D8 test 与 D10，且 split=train",
             "", "| 子集 | 偏好对 | 平均分差 | 非法记录 | 泄漏 |", "|---|---|---|---|---|"]
    for k, v in summary["subsets"].items():
        lines.append(f"| D13-{k} | {v['n_pairs']} | {v['mean_gap']} | {v['n_invalid_records']} | {v['leakage']['n_leaks']} / {v['leakage']['n_checked']} |")
    lines += ["", "候选来源平均 full 分：", ""] + [f"- `{k}`：{v}" for k, v in summary["source_mean_full"].items()]
    lines += ["", "后置：候选生成需 M1（D-1）与 M0 采样；人工抽 100 对复核偏好方向正确率 ≥ 90%（PLAN D-2）。"]
    (OUT / "DATA_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="", help="候选 jsonl（M1 / M0 采样产物）")
    ap.add_argument("--synthetic-demo", action="store_true", help="用金标扰动生成候选验证管线")
    ap.add_argument("--n", type=int, default=60, help="synthetic-demo 抽取的 train prompt 数（按类别分层）")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--dpo-style", choices=["tool_calls", "jsontext"], default="tool_calls")
    ap.add_argument("--no-execute", action="store_true", help="不执行工具（business / faith 记 0），仅调试")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in open(SEEDS, encoding="utf-8") if l.strip()]
    seeds_by_id = {s["seed_id"]: s for s in seeds}
    if args.synthetic_demo:
        rng = random.Random(args.seed)
        cands = [c for s in stratified_train_seeds(seeds, args.n, rng) for c in synthetic_candidates(s, rng)]
        tag = args.tag or "demo"
    elif args.candidates:
        cands = [json.loads(l) for l in open(args.candidates, encoding="utf-8") if l.strip()]
        tag = args.tag
    else:
        sys.exit("需要 --candidates 或 --synthetic-demo")
    summary = run(cands, seeds_by_id, execute=not args.no_execute, dpo_style=args.dpo_style, tag=tag)
    write_data_card(summary, demo=args.synthetic_demo)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
