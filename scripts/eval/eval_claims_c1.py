"""
C-1 验收脚本：Generator 声明级输出（claims 模式）在 D10 上的 JSON 合法率与证据覆盖。

流程：D10 分层抽样 → OraclePlanner（金标动作，不调 LLM）→ 真实工具（fault_attribution / kg_search /
ett_forecast / timeseries_anomaly / local_kb BM25）→ GeneratorAgent(output_mode=claims) → 统计。

默认不启用 LLM：Generator 走模板回退路径，声明由 build_rule_claims 规则拼装（claims_source=rule）。
`--llm` 时读取 config.yaml 真实 LLM 配置走 system_claims.txt 提示词路径（claims_source=llm），
用于补跑 LLM 路径合法率；报告中会分别标注。

用法：
    python3 scripts/eval/eval_claims_c1.py                     # 30 条，无 LLM
    python3 scripts/eval/eval_claims_c1.py --n 30 --llm        # 启用 LLM
    python3 scripts/eval/eval_claims_c1.py --n 5 --no-report   # 快速自检，不落盘
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                # noqa: E402
from src.graph.workflow import run_diagnosis_workflow                 # noqa: E402
from src.graph.system_modes import tool_stack_spec, build_agents        # noqa: E402
from src.agents.generator import GeneratorAgent                       # noqa: E402
from src.agents.claims import validate_claims, evidence_catalog, CLAIM_TYPES   # noqa: E402
from src.tools.fault_attribution import configure_engine              # noqa: E402
from eval_system_modes import build_mcp, OraclePlanner                # noqa: E402

D10 = ROOT / "data/eval/d10/end2end_eval.jsonl"
REPORT_MD = ROOT / "docs/eval/claims_acceptance.md"
REPORT_JSON = ROOT / "docs/eval/claims_acceptance.json"

NO_LLM_CFG: Dict[str, Any] = {
    "llms": {"default": {"provider": "openai", "model_name": "x", "api_key": "", "api_key_env": "___DISABLED___",
                         "base_url": "http://127.0.0.1:1"}},
    "workflow": {"max_iterations": 1, "planner_strategy": "free", "attribution_mode": "calibrated",
                 "generator_output_mode": "claims"},
}


def load_d10() -> List[Dict[str, Any]]:
    rows = []
    with D10.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """按 scenario 分层，每层至少 1 条，其余按比例分配。"""
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["scenario"]].append(r)
    keys = sorted(groups)
    if n >= len(rows):
        return list(rows)
    quota = {k: 1 for k in keys}
    rest = n - len(keys)
    total = sum(len(groups[k]) for k in keys)
    frac = {k: rest * len(groups[k]) / total for k in keys}
    for k in keys:
        quota[k] += int(frac[k])
    leftover = n - sum(quota.values())
    for k in sorted(keys, key=lambda k: frac[k] - int(frac[k]), reverse=True)[:max(leftover, 0)]:
        quota[k] += 1
    out = []
    for k in keys:
        pool = list(groups[k])
        rng.shuffle(pool)
        out.extend(pool[:min(quota[k], len(pool))])
    rng.shuffle(out)
    return out[:n]


def _safety_supported(claims: List[Dict[str, Any]]) -> bool:
    """safety 声明必须引用 tool 证据（检测结果支撑）。"""
    for c in claims:
        if c.get("type") == "safety":
            if not any(e.get("source") == "tool" for e in c.get("evidence", [])):
                return False
    return True


def run(samples: List[Dict[str, Any]], cfg: Dict[str, Any], *, use_llm: bool) -> List[Dict[str, Any]]:
    logging.disable(logging.WARNING)
    configure_engine(cfg.get("workflow", {}).get("attribution_mode", "calibrated"))
    spec = tool_stack_spec(cfg)
    mcp, kb_desc = build_mcp(spec)
    _, retriever, _, validator, reflector = build_agents(cfg, mcp, spec)
    generator = GeneratorAgent(cfg, output_mode="claims")
    planner = OraclePlanner(spec.allowed_tools)
    print(f"=== C-1 claims acceptance | n={len(samples)} rag={kb_desc} llm={'on' if use_llm else 'off'} "
          f"generator.llm_enabled={generator.llm.enabled} ===")

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        planner.bind(s)
        state = AgentState(user_query=s["user_query"], context=dict(s.get("context") or {}), max_iterations=1)
        t0 = time.time()
        err = None
        try:
            final = run_diagnosis_workflow(state, planner, retriever, generator, validator, reflector=reflector)
        except Exception as exc:  # noqa: BLE001
            final, err = state, str(exc)
        elapsed = time.time() - t0

        claims = list(final.draft_claims or [])
        cat = evidence_catalog(final)
        ok, errors, _ = validate_claims(claims, cat["refs"])
        hard_errors = [e for e in errors if not e.startswith("[warning]")]
        warn = [e for e in errors if e.startswith("[warning]")]
        n_ev = sum(len(c.get("evidence", [])) for c in claims)
        all_have_ev = bool(claims) and all(len(c.get("evidence", [])) >= 1 for c in claims)
        # JSON 可序列化 + 反序列化 round-trip
        try:
            json_rt = json.loads(json.dumps({"claims": claims}, ensure_ascii=False))
            json_ok = isinstance(json_rt.get("claims"), list)
        except Exception:  # noqa: BLE001
            json_ok = False
        rec = {
            "eval_id": s["eval_id"], "scenario": s["scenario"], "sub_type": s.get("sub_type"),
            "error": err, "elapsed": round(elapsed, 3),
            "claims_source": final.claims_source, "claims_json_valid": final.claims_json_valid,
            "n_claims": len(claims), "n_evidence": n_ev,
            "schema_ok": ok and json_ok, "all_have_evidence": all_have_ev,
            "ref_unknown": len(warn), "hard_errors": hard_errors[:5],
            "safety_supported": _safety_supported(claims),
            "type_counts": dict(Counter(c.get("type") for c in claims)),
            "source_counts": dict(Counter(e.get("source") for c in claims for e in c.get("evidence", []))),
            "n_tool_calls": len(final.tool_calls), "n_kb": len(final.retrieved_knowledge),
            "draft_len": len(final.draft_answer or ""),
            "claims": claims,
        }
        out.append(rec)
        print(f"  [{i}/{len(samples)}] {s['eval_id']} {s['scenario']:<20} src={rec['claims_source']:<4} "
              f"claims={rec['n_claims']:<2} ev={n_ev:<2} schema={'Y' if rec['schema_ok'] else 'N'} "
              f"allev={'Y' if all_have_ev else 'N'} unk_ref={rec['ref_unknown']} err={err or '-'}")
    return out


def summarize(rows: List[Dict[str, Any]], *, use_llm: bool) -> Dict[str, Any]:
    n = len(rows)
    ok_rows = [r for r in rows if not r["error"]]
    schema_ok = sum(1 for r in rows if r["schema_ok"])
    all_ev = sum(1 for r in rows if r["all_have_evidence"])
    total_claims = sum(r["n_claims"] for r in rows)
    total_ev = sum(r["n_evidence"] for r in rows)
    claims_with_ev = sum(
        1 for r in rows for c in r["claims"] if len(c.get("evidence", [])) >= 1)
    type_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for r in rows:
        type_counts.update(r["type_counts"])
        source_counts.update(r["source_counts"])
    per_scn: Dict[str, Dict[str, Any]] = {}
    for scn in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == scn]
        per_scn[scn] = {
            "n": len(sub),
            "schema_ok": sum(1 for r in sub if r["schema_ok"]),
            "avg_claims": round(sum(r["n_claims"] for r in sub) / len(sub), 2),
            "avg_evidence": round(sum(r["n_evidence"] for r in sub) / len(sub), 2),
            "ref_unknown": sum(r["ref_unknown"] for r in sub),
        }
    return {
        "n": n, "n_errors": n - len(ok_rows), "use_llm": use_llm,
        "claims_source": dict(Counter(r["claims_source"] for r in rows)),
        "json_valid_rate": schema_ok / n if n else 0.0,
        "all_evidence_rate": all_ev / n if n else 0.0,
        "claim_evidence_coverage": claims_with_ev / total_claims if total_claims else 0.0,
        "total_claims": total_claims, "total_evidence": total_ev,
        "avg_claims_per_sample": total_claims / n if n else 0.0,
        "avg_evidence_per_claim": total_ev / total_claims if total_claims else 0.0,
        "ref_unknown_total": sum(r["ref_unknown"] for r in rows),
        "safety_supported_rate": sum(1 for r in rows if r["safety_supported"]) / n if n else 0.0,
        "type_counts": dict(type_counts), "source_counts": dict(source_counts),
        "per_scenario": per_scn,
        "avg_elapsed": sum(r["elapsed"] for r in rows) / n if n else 0.0,
        "pass": (schema_ok / n >= 0.95 if n else False) and (claims_with_ev == total_claims) and total_claims > 0,
    }


def write_report(summary: Dict[str, Any], rows: List[Dict[str, Any]], seed: int) -> None:
    s = summary
    src_txt = "、".join(f"{k or 'n/a'}={v}" for k, v in s["claims_source"].items())
    lines = [
        "# C-1 声明级输出验收报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/eval/eval_claims_c1.py`；seed={seed}。",
        "",
        "## 1. 设置",
        "",
        f"- 样本：D10 `data/eval/d10/end2end_eval.jsonl` 按 scenario 分层抽 {s['n']} 条",
        "- Planner：OraclePlanner（金标动作，不调 LLM）；工具：fault_attribution / kg_search / ett_forecast / timeseries_anomaly 真实实现，rag_search 走本地 BM25 两路检索",
        f"- Generator：`output_mode=claims`；LLM {'启用' if s['use_llm'] else '未启用（走模板回退 + 规则拼装路径）'}；声明来源分布：{src_txt}",
        "- 校验：`validate_claims`（schema + 每条 claim ≥1 evidence）+ JSON round-trip；`ref` 与本轮证据目录比对",
        "",
        "## 2. 主结果",
        "",
        "| 指标 | 值 | 验收线 |",
        "|---|---|---|",
        f"| JSON 合法率（schema_ok） | {s['json_valid_rate'] * 100:.1f}%（{int(round(s['json_valid_rate'] * s['n']))}/{s['n']}） | ≥ 95% |",
        f"| 每条 claim ≥1 evidence 的声明占比 | {s['claim_evidence_coverage'] * 100:.1f}%（{s['total_claims']} 条声明） | 100% |",
        f"| 样本级全部声明有证据 | {s['all_evidence_rate'] * 100:.1f}% | - |",
        f"| 平均声明数 / 样本 | {s['avg_claims_per_sample']:.2f} | - |",
        f"| 平均证据数 / 声明 | {s['avg_evidence_per_claim']:.2f} | - |",
        f"| ref 不在本轮证据目录的证据数 | {s['ref_unknown_total']} | 0 |",
        f"| safety 声明均有 tool 证据支撑的样本占比 | {s['safety_supported_rate'] * 100:.1f}% | 100% |",
        f"| 工作流异常样本 | {s['n_errors']} | 0 |",
        f"| 平均耗时 / 样本 | {s['avg_elapsed']:.2f} s | - |",
        "",
        f"**验收判定：{'通过' if s['pass'] else '未通过'}**（JSON 合法率 ≥95% 且每条声明至少一条证据）。",
        "",
        "## 3. 声明类型与证据来源分布",
        "",
        "| type | 数量 |", "|---|---|",
    ]
    for t in CLAIM_TYPES:
        lines.append(f"| {t} | {s['type_counts'].get(t, 0)} |")
    lines += ["", "| evidence.source | 数量 |", "|---|---|"]
    for k in ("tool", "kb", "kg", "user"):
        lines.append(f"| {k} | {s['source_counts'].get(k, 0)} |")
    lines += ["", "## 4. 按场景", "", "| scenario | n | schema_ok | 平均声明数 | 平均证据数 | 未知 ref |", "|---|---|---|---|---|---|"]
    for scn, v in s["per_scenario"].items():
        lines.append(f"| {scn} | {v['n']} | {v['schema_ok']} | {v['avg_claims']} | {v['avg_evidence']} | {v['ref_unknown']} |")
    lines += ["", "## 5. 样例（前 3 条）", ""]
    for r in rows[:3]:
        lines.append(f"### {r['eval_id']}（{r['scenario']}，source={r['claims_source']}）")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({"claims": r["claims"][:4]}, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines += [
        "## 6. 说明与局限",
        "",
        "- 本报告默认在 LLM 未启用条件下产出，验证的是规则拼装路径（`build_rule_claims`）的 schema 合法性与证据覆盖，以及 claims 模式下工作流链路不断。",
        "- LLM 路径（`system_claims.txt` 提示词 → JSON 解析 → `validate_claims`）的合法率需用 `--llm` 补跑，届时 `claims_source=llm` 占比与 `claims_json_valid` 即为该路径指标；解析失败会自动回退规则声明并计入 `state.errors`。",
        "- OraclePlanner 结果只用于校验 Generator 输出规范，不代表任何系统模式的正式成绩。",
        "- 规则声明的 `span` 为工具结果的 JSON 片段而非自然语言原文，C-2 确定性层按 JSON 子串 / 数值比对核查。",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k != "claims"} | {"claims": r["claims"]} for r in rows]
    REPORT_JSON.write_text(json.dumps({"summary": s, "rows": slim}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--llm", action="store_true", help="读取 config.yaml 真实 LLM 配置（走 LLM claims 路径）")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    if args.llm:
        from src.utils.config import load_config
        cfg = load_config()
        cfg.setdefault("workflow", {})["generator_output_mode"] = "claims"
        cfg["workflow"]["max_iterations"] = 1
    else:
        cfg = NO_LLM_CFG

    rows = stratified_sample(load_d10(), args.n, args.seed)
    t0 = time.time()
    res = run(rows, cfg, use_llm=args.llm)
    summary = summarize(res, use_llm=args.llm)
    summary["elapsed_total"] = round(time.time() - t0, 2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_scenario",)}, ensure_ascii=False, indent=2))
    if not args.no_report:
        write_report(summary, res, args.seed)
        print(f"report -> {REPORT_MD}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
