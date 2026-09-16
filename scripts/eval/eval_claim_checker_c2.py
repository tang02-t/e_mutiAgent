"""
C-2 验收脚本：声明级约束核查器（确定性层）。

两部分：
(a) 构造样例：复用 tests/test_c2_claim_checker.py 的四类约束 × (5 违反 + 5 合规) 共 40 条，统计检出率与误报；
(b) D10 全量 196 条：OraclePlanner + 真实工具 + 规则声明（Generator 无 LLM），核查器对「干净」声明的误报率
    （规则声明由工具结果直接拼装，理论上应全部 pass，任何 non-pass 即误报）。
语义层（LLM NLI）不在本脚本内跑：需 30 条人工标注声明，后置到 C-5 / 人工项。

用法：python3 scripts/eval/eval_claim_checker_c2.py [--no-report]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                # noqa: E402
from src.graph.workflow import run_diagnosis_workflow                 # noqa: E402
from src.graph.system_modes import tool_stack_spec, build_agents        # noqa: E402
from src.agents.generator import GeneratorAgent                       # noqa: E402
from src.agents.claim_checker import ClaimChecker                     # noqa: E402
from src.agents.claims import CONSTRAINT_TYPES                        # noqa: E402
from src.tools.fault_attribution import configure_engine              # noqa: E402
from eval_system_modes import build_mcp, OraclePlanner                # noqa: E402
import eval_claims_c1 as C1                                           # noqa: E402

REPORT_MD = ROOT / "docs/claim_checker_acceptance.md"
REPORT_JSON = ROOT / "docs/claim_checker_acceptance.json"


def _load_cases():
    """从测试文件里取构造样例（避免重复维护）。"""
    spec = importlib.util.spec_from_file_location("_c2cases", ROOT / "tests" / "test_c2_claim_checker.py")
    src = (ROOT / "tests" / "test_c2_claim_checker.py").read_text(encoding="utf-8")
    # 只执行样例定义部分：截到第一处 "for i, c in enumerate(data_bad" 之前会遗漏后续组，故改为执行整个文件但屏蔽 check/exit
    ns: Dict[str, Any] = {"__file__": str(ROOT / "tests" / "test_c2_claim_checker.py"), "__name__": "_c2cases"}
    code = src.replace("sys.exit(1 if FAILED else 0)", "pass")
    code = code.replace("def check(cond: bool, msg: str) -> None:", "def check(cond: bool, msg: str) -> None:\n    return\n\ndef _unused(cond, msg):")
    code = code.replace('print(f"  ok   {msg}")', "pass").replace('print(f"  FAIL {msg}")', "pass")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, str(ROOT / "tests" / "test_c2_claim_checker.py"), "exec"), ns)  # noqa: S102
    return ns


def part_a(ns) -> Dict[str, Any]:
    ck = ClaimChecker(llm=None)
    groups = {
        "DATA": (ns["data_bad"], ns["data_good"], None),
        "EVIDENCE": (ns["ev_bad"], ns["ev_good"], None),
        "APPLICABILITY": (ns["ap_bad"], ns["ap_good"], None),
        "SAFETY": (ns["sa_bad"], ns["sa_good"], True),
    }
    out: Dict[str, Any] = {}
    for ctype, (bad, good, paired) in groups.items():
        det = fp = 0
        for item in bad:
            claim, st = (item if paired else (item, ns["base_state"]()))
            st.draft_claims = [claim]
            r = ck.check(st)
            if any(ctype in v.violated_constraints for v in r.claim_verdicts):
                det += 1
        for item in good:
            claim, st = (item if paired else (item, ns["base_state"]()))
            st.draft_claims = [claim]
            r = ck.check(st)
            if any(v.verdict != "pass" or v.violated_constraints for v in r.claim_verdicts):
                fp += 1
        out[ctype] = {"n_bad": len(bad), "detected": det, "n_good": len(good), "false_positive": fp}
    return out


def part_b() -> Dict[str, Any]:
    logging.disable(logging.WARNING)
    cfg = C1.NO_LLM_CFG
    configure_engine("calibrated")
    spec = tool_stack_spec(cfg)
    mcp, kb_desc = build_mcp(spec)
    _, retriever, _, validator, reflector = build_agents(cfg, mcp, spec)
    generator = GeneratorAgent(cfg, output_mode="claims")
    planner = OraclePlanner(spec.allowed_tools)
    ck = ClaimChecker(llm=None)
    rows = C1.load_d10()
    n_claims = n_bad = n_samples_bad = 0
    vc: Counter = Counter()
    verdicts: Counter = Counter()
    bad_examples: List[Dict[str, Any]] = []
    t0 = time.time()
    for s in rows:
        planner.bind(s)
        st = AgentState(user_query=s["user_query"], context=dict(s.get("context") or {}), max_iterations=1)
        final = run_diagnosis_workflow(st, planner, retriever, generator, validator, reflector=reflector)
        r = ck.check(final)
        n_claims += r.n_claims
        verdicts[r.verdict] += 1
        vc.update(r.violation_counts)
        bad = [v for v in r.claim_verdicts if v.verdict != "pass"]
        n_bad += len(bad)
        if bad:
            n_samples_bad += 1
            for v in bad[:2]:
                txt = next((c["text"] for c in final.draft_claims if c["id"] == v.claim_id), "")
                bad_examples.append({"eval_id": s["eval_id"], **v.to_dict(), "text": txt[:100]})
    return {"n_samples": len(rows), "n_claims": n_claims, "n_nonpass_claims": n_bad, "n_samples_with_nonpass": n_samples_bad,
            "claim_fp_rate": n_bad / n_claims if n_claims else 0.0, "sample_fp_rate": n_samples_bad / len(rows),
            "violation_counts": dict(vc), "verdicts": dict(verdicts), "rag": kb_desc,
            "elapsed": round(time.time() - t0, 2), "bad_examples": bad_examples[:10]}


def write_report(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    tot_bad = sum(v["n_bad"] for v in a.values()); tot_det = sum(v["detected"] for v in a.values())
    tot_good = sum(v["n_good"] for v in a.values()); tot_fp = sum(v["false_positive"] for v in a.values())
    pass_a = tot_det == tot_bad and tot_fp == 0
    lines = [
        "# C-2 声明级约束核查器验收报告（确定性层）",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；脚本 `scripts/eval/eval_claim_checker_c2.py`。",
        "",
        "## 1. 构造样例（四类约束 × 5 违反 + 5 合规）",
        "",
        "| 约束 | 违反样例 | 检出 | 合规样例 | 误报 |", "|---|---|---|---|---|",
    ]
    for k in CONSTRAINT_TYPES:
        v = a[k]
        lines.append(f"| {k} | {v['n_bad']} | {v['detected']} | {v['n_good']} | {v['false_positive']} |")
    lines += [
        f"| 合计 | {tot_bad} | {tot_det}（{tot_det / tot_bad * 100:.0f}%） | {tot_good} | {tot_fp} |",
        "",
        f"**验收判定（确定性层）：{'通过' if pass_a else '未通过'}**（要求检出率 100%、误报 0）。样例定义见 [tests/test_c2_claim_checker.py](/Users/ts/Desktop/thu/multi_Agent/tests/test_c2_claim_checker.py)，正反例与 `docs/claim_schema.md` §4 对应。",
        "",
        "## 2. D10 全量干净声明误报率",
        "",
        f"- 设置：D10 {b['n_samples']} 条，OraclePlanner + 真实工具（rag={b['rag']}），Generator 无 LLM → `build_rule_claims` 规则声明（视为干净样本）",
        f"- 声明总数 {b['n_claims']}，未通过 {b['n_nonpass_claims']}（声明级误报率 {b['claim_fp_rate'] * 100:.2f}%），涉及样本 {b['n_samples_with_nonpass']}（样本级 {b['sample_fp_rate'] * 100:.2f}%）",
        f"- 约束违反计数：{b['violation_counts']}；核查结论分布：{b['verdicts']}",
        f"- 耗时 {b['elapsed']} s",
        "",
    ]
    if b["bad_examples"]:
        lines.append("未通过样例（前 10）：")
        lines.append("")
        for e in b["bad_examples"]:
            lines.append(f"- {e['eval_id']} [{e['claim_id']}] {'/'.join(e['violated_constraints']) or e['verdict']}：{'；'.join(e['detail'])} ← {e['text']}")
        lines.append("")
    lines += [
        "## 3. 判定规则实现",
        "",
        "- 任一 `SAFETY` 或 `DATA` 违反 → `REVISION`；`unsupported_ratio`（无依据 / 矛盾 / EVIDENCE 违反声明占比）> 0.3 → `REVISION`；无声明 → `REVISION`",
        "- `iteration ≥ claim_abstain_after`（默认 3，即两轮修订后）仍不通过 → `ABSTAIN`，Validator 输出「证据不足，建议补充 X」并结束",
        "- `ValidationResult` v2 新增 `claim_verdicts / unsupported_ratio / violation_counts / claim_check_verdict / missing_evidence`，旧字段保留；`claim_check=off` 时行为与 v1 完全一致",
        "",
        "## 4. 语义层（后置）",
        "",
        "- `ClaimChecker._semantic_layer` 对通过确定性层的 `inference` 声明做 NLI（entail / contradict / unsupported），提示词仅含该声明与其引用片段；LLM 未启用时自动跳过。",
        "- 验收「30 条人工标注声明一致率 ≥ 85%」需启用 LLM 并人工标注，后置到 C-5 阶段与人工项一并完成。",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({"constructed": a, "d10_clean": b, "pass_deterministic": pass_a}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    ns = _load_cases()
    a = part_a(ns)
    print("constructed:", json.dumps(a, ensure_ascii=False))
    b = part_b()
    print("d10_clean:", json.dumps({k: v for k, v in b.items() if k != "bad_examples"}, ensure_ascii=False))
    for e in b["bad_examples"]:
        print("  ", e)
    tot_bad = sum(v["n_bad"] for v in a.values()); tot_det = sum(v["detected"] for v in a.values())
    tot_fp = sum(v["false_positive"] for v in a.values())
    ok = tot_det == tot_bad and tot_fp == 0
    if not args.no_report:
        write_report(a, b)
        print(f"report -> {REPORT_MD}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
