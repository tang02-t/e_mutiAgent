#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三条主线的「从输入到答案」逐步执行轨迹（离线、零 LLM、零费用），供快速理解系统如何运转。

  python3 scripts/demo_traces.py                 # 打印三条轨迹
  python3 scripts/demo_traces.py --write         # 同时写入 docs/walkthrough_traces.md

三段分别复用现有测试 / 评测的构造方式，保证与正式实验同一条代码路径：
  主线一 mode3：D12 部分观测样例 + PartialObsEnv 代替用户回答（同 tests/test_b4_active.py）
  主线二 mode4：D11 故障注入样例，snapshot 重建证据后走 Validator / supplement / Retriever（同 scripts/eval/eval_validator.py）
  主线三 D13  ：D8 train 种子 + 金标扰动候选，真实工具执行后打分配对（同 scripts/planner_data/score_candidates.py --synthetic-demo）
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

import partial_obs_sim as sim                                                # noqa: E402
import build_d11_fault_injection as d11                                      # noqa: E402
from scripts.eval import eval_validator as ev                                # noqa: E402
from scripts.planner_data import score_candidates as sc                      # noqa: E402
from src.graph import workflow as wf                                         # noqa: E402
from src.graph.state import AgentState                                       # noqa: E402
from src.graph.system_modes import resolve_mode, tool_stack_spec, build_agents  # noqa: E402
from src.agents.validator import ValidatorAgent                              # noqa: E402
from src.agents.claim_checker import ClaimChecker                            # noqa: E402
from src.tools.fault_attribution import configure_engine                     # noqa: E402
from eval_system_modes import build_mcp                                      # noqa: E402

OUT_MD = ROOT / "docs/walkthrough_traces.md"
LINES: List[str] = []


def p(s: str = "") -> None:
    print(s)
    LINES.append(s)


def h(title: str) -> None:
    p()
    p(f"## {title}")
    p()


def code(obj: Any, lang: str = "json") -> None:
    txt = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2)
    p(f"```{lang}")
    p(txt)
    p("```")


def _short(v: Any, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


# ──────────────────────────────────────────────────────────────
# 主线一：mode3 主动追问
# ──────────────────────────────────────────────────────────────
def trace_mainline1(level: str = "heavy", seed: int = 7) -> None:
    h("主线一 · mode3 active_plan：信息不足时先问什么")
    p("样例来自 D12 部分观测模拟集：一条真实 DGA 记录被遮蔽 3 种气体（重档），`PartialObsEnv` 扮演用户，")
    p("被问到某征兆时按真值回答并计费。整条链路不调用 LLM：Planner 走 `decision=eig` 路径，Generator 走模板。")
    p()
    recs = sim.load_d12(level=level)
    rng = random.Random(seed)
    rec = rng.choice(recs)
    env = sim.PartialObsEnv(rec)

    p(f"**输入**（`sim_id={rec['sim_id']}`，真实标签 `{rec['fault_label']}`，评测时对系统隐藏）")
    code({"user_query": "这台变压器油色谱有部分数据缺失，请判断故障类型。",
          "context.dga（None = 未检测）": env.context(),
          "被遮蔽的气体": rec["masked_gases"]})

    spec = resolve_mode("mode3")
    cfg = {"llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                                "base_url": "http://127.0.0.1:1"}},
           "workflow": {"max_iterations": 1, "planner_strategy": "active", "planner_decision": "eig",
                        "max_inquiry_rounds": 3, "attribution_mode": "calibrated"}}
    mcp, _ = build_mcp(spec)
    planner, retriever, generator, validator, reflector = build_agents(cfg, mcp, spec, planner_decision="eig")
    state = AgentState(user_query="这台变压器油色谱有部分数据缺失，请判断故障类型。",
                       context={"dga": env.context(), "device_id": rec["sim_id"]},
                       max_iterations=1, max_inquiry_rounds=3)
    final = wf.run_diagnosis_workflow(state, planner, retriever, generator, validator, reflector=reflector,
                                      answer_fn=env.answer)

    p("**逐轮轨迹**（`inquiry_log`：每轮 Planner 的动作、依据的熵与 Top-1、用户回答、累计成本）")
    p()
    p("| 轮 | 动作 | 决策来源 | 追问征兆 | 用户回答 | 累计成本 | 决策时后验熵 bit |")
    p("|---|---|---|---|---|---|---|")
    for e in final.inquiry_log:
        ent = e.get("entropy_bits")
        p(f"| {e['round']} | {e['action']} | {e.get('decision_source')} | {e.get('symptom') or '-'} | {e.get('answer')} "
          f"| {e.get('cost', 0):.2f} | {ent:.3f} |" if ent is not None else
          f"| {e['round']} | {e['action']} | {e.get('decision_source')} | {e.get('symptom') or '-'} | {e.get('answer')} "
          f"| {e.get('cost', 0):.2f} | - |")
    p()
    attrs = [c for c in final.tool_calls if c.get("tool") == "fault_attribution" and c.get("success")]
    p("**每次归因后的 Top-1 变化**（同一条记录，随追问答案更新）")
    p()
    for i, c in enumerate(attrs):
        r = c.get("result") or {}
        u = r.get("uncertainty") or {}
        p(f"- 第 {i} 次归因：Top-1 `{r.get('primary_fault')}`（{(r.get('primary_probability') or 0):.3f}），"
          f"熵 {u.get('entropy_bits', 0):.3f} bit，负观测 {r.get('evidence_negative')}")
    p()
    unc = (final.context or {}).get("uncertainty") or {}
    recs_ = unc.get("recommendations") or []
    last = (attrs[-1].get("result") if attrs else {}) or {}
    p("**归因工具最后一次返回的 `uncertainty` 块**（Planner 就是看这个决定问不问、问什么）")
    code({"entropy_bits": unc.get("entropy_bits"), "top1_top2_gap": unc.get("top1_top2_gap"),
          "calibrated": unc.get("calibrated"), "suggested_action": unc.get("suggested_action"),
          "recommendations（前 3）": [{k: r.get(k) for k in ("symptom", "eig", "cost", "voi", "how_to_obtain")}
                                    for r in recs_[:3]]})
    p("**工具调用序列**（`trajectory`）")
    p()
    for i, t in enumerate(final.trajectory):
        p(f"- call:{i} `{t['tool']}` 参数 `{_short(t.get('raw_arguments'), 120)}` → business_success={t.get('business_success')}")
    p()
    p(f"**最终答案**（模板生成，LLM 关闭）：最终 Top-1 = `{last.get('primary_fault')}`，真实标签 = `{rec['fault_label']}`，"
      f"问询 {final.inquiry_rounds} 轮，成本 {final.inquiry_cost:.2f}，验证 `{final.validation_verdict}`")
    code((final.final_answer or "")[:600], "text")
    p("**怎么读**：第 0 轮先归因，熵高于阈值且 VoI 最大的征兆是「问用户」能拿到的，于是 `ask`；用户回答后把新气体写回 `context.dga` 重新归因，")
    p("熵降到 τ_H=0.8 以下或最大 VoI 低于 0.02 时 `conclude`。基线 mode2 没有这一环，会直接用缺测数据出结论。")


# ──────────────────────────────────────────────────────────────
# 主线二：mode4 声明核查 + 补证
# ──────────────────────────────────────────────────────────────
def trace_mainline2(subtype: str = "remove_dga_premise") -> None:
    h("主线二 · mode4 claim_verify：一条没有依据的「立即停电」建议是怎么被拦下的")
    p("样例来自 D11 故障注入集：在一份干净诊断草案上删掉支撑「高风险处置建议」的观测声明（SAFETY 类注入），")
    p("并额外追加 2 条需要尚未调用工具才能落证的声明（E2 设定）。`snapshot` 离线重建同一份工具结果，")
    p("Generator 用 `SimGenerator` 模拟（只修 Validator 标记的声明），Validator / supplement / Retriever 是真实节点。")
    p()
    recs = ev.load_d11()
    rec = next(r for r in recs if r.get("subtype") == subtype)
    body = rec["draft_answer"].split(d11.CLAIMS_HEAD)[0].rstrip()

    p(f"**输入**（`eval_id={rec['eval_id']}`，底稿 `{rec['base_eval_id']}`，场景 `{rec['scenario']}`，注入 `{rec['type']}/{rec['subtype']}`）")
    code({"user_query": rec["snapshot"]["user_query"],
          "context": rec["snapshot"].get("context"),
          "已有工具调用": [f"call:{i} {c['tool']}" for i, c in enumerate(rec["snapshot"]["tool_calls"])]})
    p("**Generator 首轮输出的声明清单**（注入后；`c7` 的处置建议失去了它本该引用的观测 `c1`）")
    p()
    p("| id | type | 声明 | evidence 引用 |")
    p("|---|---|---|---|")
    for c in rec["claims"]:
        refs = ", ".join(f"{e.get('source')}:{e.get('ref')}" for e in c.get("evidence") or []) or "（无）"
        p(f"| {c['id']} | {c['type']} | {_short(c['text'], 60)} | {refs} |")
    p()
    p(f"被删除的观测声明：`{_short(rec['original']['removed_claims'][0]['text'], 90)}`")
    p()

    cfg = json.loads(json.dumps(ev.NO_LLM_CFG))
    cfg["workflow"]["claim_check"] = "route"
    configure_engine("calibrated")
    spec = tool_stack_spec(cfg)
    mcp, _ = build_mcp(spec)
    _, retriever, _, _, _ = build_agents(cfg, mcp, spec)
    validator = ValidatorAgent(cfg, claim_check="route")

    state = d11.state_from_snapshot(rec["snapshot"], [], "")
    state.iteration, state.max_iterations = 0, 3
    extras = ev.make_extras(rec)
    gen = ev.SimGenerator(rec["claims"], body, extras)
    logging.disable(logging.WARNING)
    final = wf._finish_generation(state, retriever, gen, validator, None)

    extras_desc = [f"{c['id']}→需 {c['_need_tool']}" for c in extras]
    p(f"追加的缺证声明（E2）：{extras_desc}")
    p()
    p("**Validator 逐轮判定**（`reasoning_trace` 中 agent=validator 的记录）")
    p()
    rnd = 0
    for tr in final.reasoning_trace:
        if tr.get("agent") != "validator" or tr.get("type") != "evaluation":
            continue
        rnd += 1
        c = tr.get("content") or {}
        p(f"第 {rnd} 轮：verdict=`{c.get('verdict')}` claim_check=`{c.get('claim_check_verdict')}` "
          f"unsupported_ratio={c.get('unsupported_ratio')}")
        p()
        p("| claim | verdict | 违反约束 | 说明 |")
        p("|---|---|---|---|")
        for v in c.get("claim_verdicts") or []:
            if v.get("verdict") == "pass" and not v.get("violated_constraints"):
                continue
            p(f"| {v['claim_id']} | {v['verdict']} | {', '.join(v.get('violated_constraints') or []) or '-'} | {_short('；'.join(v.get('detail') or []), 90)} |")
        p()
    p("**路由与补证**（`route_log`）")
    p()
    for r in final.route_log:
        p(f"- 目标 `{r.get('target')}`，原因：{_short(r.get('reason', ''), 100)}"
          + (f"，补证工具：{r.get('tools')}" if r.get("tools") else ""))
    p()
    extra_calls = final.trajectory[len(rec["snapshot"]["tool_calls"]):] if len(final.trajectory) > len(rec["snapshot"]["tool_calls"]) else []
    if extra_calls:
        p("补证新增的工具调用：")
        for t in extra_calls:
            p(f"- `{t['tool']}` 参数 `{_short(t.get('raw_arguments'), 100)}` → business_success={t.get('business_success')}")
        p()
    p(f"**结局**：最终 verdict=`{final.validation_verdict}`，修订 {final.iteration} 轮，补证 {final.evidence_rounds} 轮，"
      f"SimGenerator 修正 {gen.n_fixed} 条 / 再落证 {gen.n_regrounded} 条 / 删除 {gen.n_dropped} 条 / 保留 {gen.n_kept} 条，"
      f"最终声明 {len(final.draft_claims)} 条")
    p()
    p("**怎么读**：v1 Validator 只给整份草案打一个分，这类「建议成立但前提被删」的错误 100% 放行；mode4 把每条声明的 evidence 解析出来，")
    p("SAFETY 类建议必须有 `observation` 支撑，否则直接 REVISION 并指出是哪一条；缺证但可补的声明触发 supplement 节点去调工具，而不是反复修订到弃答。")


# ──────────────────────────────────────────────────────────────
# 主线三：D13 偏好对
# ──────────────────────────────────────────────────────────────
def trace_mainline3(seed: int = 20260915) -> None:
    h("主线三 · D13 偏好对：一条 prompt 的 6 个候选计划如何被打分、配对")
    p("样例来自 D8 train 种子。正式流程中候选由 M1 / M0 采样产生；这里用金标的六类扰动代替（与 `--synthetic-demo` 相同），")
    p("每个候选都在真实工具环境中执行，再送 Generator（规则声明）+ ClaimChecker 得到忠实度分。零 LLM。")
    p()
    seeds = [json.loads(l) for l in open(sc.SEEDS, encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    pool = [s for s in seeds if s.get("split") == "train" and s["gold_actions"][0]["type"] == "ask_user"]
    seed_rec = rng.choice(pool)
    seeds_by_id = {seed_rec["seed_id"]: seed_rec}
    cands = sc.synthetic_candidates(seed_rec, rng)
    for i, c in enumerate(cands):
        c.setdefault("seed_id", seed_rec["seed_id"])
        c.setdefault("candidate_id", f"{seed_rec['seed_id']}#{i}")

    p(f"**prompt**（`seed_id={seed_rec['seed_id']}`，类别 `{seed_rec.get('category')}`）")
    code({"query": seed_rec["query"], "context": seed_rec.get("context"),
          "gold_actions": seed_rec["gold_actions"]})

    ex = sc.Executor()
    rows = []
    for c in cands:
        plan = sc.parse_candidate(c["raw"])
        res = ex.run(seed_rec, plan) if (plan.format_ok and plan.steps) else None
        s = sc.score_candidate(seed_rec, plan, res)
        rows.append({"source": c["source"], "plan": plan, "exec": res, **s})

    p("**候选与打分**（六分项：format / tool / schema / business / faith / ask；`score_exec` 只看前四项，`score_full` 六项加成本惩罚）")
    p()
    p("| 候选来源 | 计划摘要 | format | tool | schema | business | faith | ask | 多余调用 | score_exec | score_full |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pl = r["plan"]
        if not pl.format_ok:
            summ = f"解析失败：{_short(pl.error or '', 40)}"
        elif pl.ask_user:
            summ = f"ask_user {pl.ask_user}"
        elif pl.steps:
            summ = " → ".join(f"{s['tool']}({_short(s['arguments'], 40)})" for s in pl.steps)
        else:
            summ = "直接回答（无调用）"
        pt = r["parts"]
        p(f"| `{r['source']}` | {_short(summ, 90)} | {pt['format']:.0f} | {pt['tool']:.0f} | {pt['schema']:.0f} | {pt['business']:.0f} "
          f"| {pt['faith']:.2f} | {pt['ask']:.0f} | {r['extra_calls']} | {r['score_exec']} | {r['score_full']} |")
    p()
    scored = [{"seed_id": seed_rec["seed_id"], "candidate_id": f"c{i}", "source": r["source"], "parts": r["parts"],
               "score_exec": r["score_exec"], "score_full": r["score_full"]} for i, r in enumerate(rows)]
    for subset, key in (("D13-exec", "score_exec"), ("D13-full", "score_full")):
        pairs = sc.make_pairs(scored, key)
        if pairs:
            pr = pairs[0]
            p(f"- {subset}（按 `{key}`）：chosen=`{pr['chosen']['source']}`（{pr['chosen'][key]}） vs "
              f"rejected=`{pr['rejected']['source']}`（{pr['rejected'][key]}），分差 {pr['gap']}")
        else:
            p(f"- {subset}（按 `{key}`）：最高-最低分差不足 {sc.PAIR_MIN_GAP}，该 prompt 不产生偏好对")
    p()
    p("**怎么读**：这是一条「应追问」的 prompt。只看工具执行成败（exec）时，「不问直接答」与金标都没有调用工具，得分相同，无法配对；")
    p("加入主线一的追问一致性与主线二的声明忠实度后（full），金标明显高于 `no_ask`，才能形成 chosen / rejected。")
    p("这就是 D13-exec 与 D13-full 两份子集要做的消融：验证信号是否比单纯的工具成败信号更能教会 Planner「该问就问、该停就停」。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写入 docs/walkthrough_traces.md")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    p("# 三条主线执行轨迹速览（离线，零 LLM）")
    p()
    p("> 由 `python3 scripts/demo_traces.py --write` 生成。每段轨迹都走与正式实验相同的代码路径，")
    p("> 只是 LLM 关闭（Planner 用 EIG 决策 / 金标扰动，Generator 用模板或规则声明）。看懂这三段就看懂了系统 80% 的数据流。")
    trace_mainline1(seed=args.seed)
    trace_mainline2()
    trace_mainline3()
    if args.write:
        OUT_MD.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        print(f"\n[written] {OUT_MD}")


if __name__ == "__main__":
    main()
