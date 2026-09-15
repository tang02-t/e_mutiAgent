"""
C-4 故障注入评测集 D11 自检测试（无 LLM、无 GPU）。

运行：python3 tests/test_c4_d11.py

覆盖：
[1] 文件规模：300 条 = 200 注入（四类各 50）+ 100 干净；eval_id 唯一且格式 D11-XXXX；build_stats 与文件一致
[2] 字段完整性：注入条含 injected/type/subtype/location/original/injection_detail/expected_constraints；干净条 injected=False
[3] 声明 schema：全部 claims 通过 validate_claims；注入条 claims 与底稿声明不同（location 指向的声明确实被改动 / 新增）
[4] snapshot 可重建 AgentState 并可被 ClaimChecker 离线复跑；确定性层对注入声明的标记与 checker_preview 一致（抽样 60 条）
[5] 干净负样本确定性层零误报（全量 100 条）；注入条的期望约束与 EXPECTED 映射一致
[6] 小规模重建可复现：同 seed 生成 --per-type 2 --clean 4 两次结果一致
[7] 文档产物存在：DATA_CARD.md / build_stats.json / manual_review_sample.md（40 行样本，每类 10）
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.agents.claims import validate_claims                          # noqa: E402
from src.agents.claim_checker import ClaimChecker                      # noqa: E402
import build_d11_fault_injection as d11                                # noqa: E402

logging.disable(logging.WARNING)

PASS = 0
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


D11_PATH = ROOT / "data" / "eval" / "d11" / "fault_injection_eval.jsonl"
STATS_PATH = ROOT / "data" / "eval" / "d11" / "build_stats.json"
CARD_PATH = ROOT / "data" / "eval" / "d11" / "DATA_CARD.md"
REVIEW_PATH = ROOT / "data" / "eval" / "d11" / "manual_review_sample.md"

REQUIRED_KEYS = {"eval_id", "base_eval_id", "scenario", "sub_type", "injected", "type", "subtype",
                 "expected_constraints", "location", "original", "injection_detail", "claims",
                 "draft_answer", "snapshot", "checker_preview", "annotation"}
SNAP_KEYS = {"user_query", "context", "tool_calls", "retrieved_knowledge", "inquiry_log"}


def load() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in D11_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _flagged_ids(report: Any) -> set:
    return {v.claim_id for v in report.claim_verdicts if v.verdict != "pass" or v.violated_constraints}


def test_1_scale(records: List[Dict[str, Any]]) -> None:
    print("[1] 规模与 eval_id")
    check(len(records) == 300, f"共 300 条（实际 {len(records)}）")
    inj = [r for r in records if r["injected"]]
    clean = [r for r in records if not r["injected"]]
    check(len(inj) == 200 and len(clean) == 100, f"注入 200 / 干净 100（实际 {len(inj)} / {len(clean)}）")
    by_type = Counter(r["type"] for r in inj)
    check(set(by_type) == set(d11.TYPES) and all(v == 50 for v in by_type.values()), f"四类各 50：{dict(by_type)}")
    ids = [r["eval_id"] for r in records]
    check(len(set(ids)) == 300, "eval_id 唯一")
    check(all(re.fullmatch(r"D11-\d{4}", i) for i in ids), "eval_id 格式 D11-XXXX")
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    check(stats["n_records"] == 300 and stats["n_injected"] == 200 and stats["n_clean"] == 100,
          "build_stats.json 与文件一致")
    check(all(stats["per_type"][t]["n"] == 50 for t in d11.TYPES), "build_stats 四类各 50")
    check(len({r["base_eval_id"] for r in clean}) == 100, "干净样本 100 个不同底稿")


def test_2_fields(records: List[Dict[str, Any]]) -> None:
    print("[2] 字段完整性")
    check(all(REQUIRED_KEYS <= set(r) for r in records), "所有记录含必需字段")
    check(all(SNAP_KEYS <= set(r["snapshot"]) for r in records), "snapshot 含五个键")
    inj = [r for r in records if r["injected"]]
    check(all(isinstance(r["location"], dict) and r["location"].get("claim_id") and r["location"].get("field")
              for r in inj), "注入条 location{claim_id, field} 完整")
    check(all(r["injection_detail"] and r["subtype"] for r in inj), "注入条 subtype / injection_detail 非空")
    check(all(r["expected_constraints"] and set(r["expected_constraints"]) <= {"DATA", "EVIDENCE", "APPLICABILITY", "SAFETY"}
              for r in inj), "expected_constraints 取值合法")
    clean = [r for r in records if not r["injected"]]
    check(all(r["type"] is None and r["location"] is None and r["original"] is None and not r["expected_constraints"]
              for r in clean), "干净条 type/location/original 为空")
    check(all(r["annotation"]["status"] == "auto" and r["annotation"]["manual_review"] is None for r in records),
          "annotation.status=auto（人工抽检后置）")
    check(all("【声明与证据清单】" in r["draft_answer"] for r in records), "draft_answer 含声明清单段")


def test_3_claims(records: List[Dict[str, Any]]) -> None:
    print("[3] 声明 schema 与注入位置")
    bad = []
    for r in records:
        ok, errs, _ = validate_claims(r["claims"])
        if not ok:
            bad.append((r["eval_id"], errs[:2]))
    check(not bad, f"全部 claims 通过 validate_claims（失败 {len(bad)}：{bad[:2]}）")
    inj = [r for r in records if r["injected"]]
    loc_ok = 0
    for r in inj:
        cid = r["location"]["claim_id"]
        cur = {c["id"]: c for c in r["claims"]}
        field = r["location"]["field"]
        if field == "new_claim":
            loc_ok += cid in cur and r["original"] is None
        elif field == "text":
            loc_ok += cid in cur and cur[cid]["text"] != r["original"]
        elif field == "evidence[0].ref":
            loc_ok += cid in cur and cur[cid]["evidence"][0]["ref"] != r["original"]
        elif field == "evidence[0].span":
            loc_ok += cid in cur and cur[cid]["evidence"][0]["span"] != r["original"]
        elif field == "evidence,removed_observations":
            orig = r["original"]
            loc_ok += (cid in cur and cur[cid]["evidence"] != orig["safety_evidence"]
                       and not any(c["id"] in {o["id"] for o in orig["removed_claims"]} for c in r["claims"]))
        elif field == "text,type":
            loc_ok += cid in cur and cur[cid]["type"] == "observation" and cur[cid]["text"].startswith("本次检测结果显示")
        else:
            loc_ok += 0
    check(loc_ok == len(inj), f"注入位置的声明确实被改动 / 新增（{loc_ok}/{len(inj)}）")
    check(all(len(c["evidence"][0].get("span", "")) <= 300 for r in records for c in r["claims"]),
          "span ≤ 300 字符")


def test_4_rebuild(records: List[Dict[str, Any]]) -> None:
    print("[4] snapshot 重建 AgentState 与 ClaimChecker 复跑一致性")
    rng = random.Random(7)
    inj = [r for r in records if r["injected"]]
    sample = rng.sample(inj, 60)
    checker = ClaimChecker()
    agree = 0
    verdict_agree = 0
    for r in sample:
        st = d11.state_from_snapshot(r["snapshot"], r["claims"], r["draft_answer"])
        rep = checker.check(st, use_llm=False)
        flagged = _flagged_ids(rep)
        tgt = r["location"]["claim_id"]
        agree += (tgt in flagged) == bool(r["checker_preview"]["target_flagged"])
        verdict_agree += rep.verdict == r["checker_preview"]["verdict"]
    check(agree == 60, f"注入声明标记与 checker_preview 一致（{agree}/60）")
    check(verdict_agree == 60, f"整体 verdict 与 checker_preview 一致（{verdict_agree}/60）")
    st = d11.state_from_snapshot(sample[0]["snapshot"], sample[0]["claims"], sample[0]["draft_answer"])
    check(st.user_query == sample[0]["snapshot"]["user_query"] and st.draft_claims == sample[0]["claims"],
          "重建的 AgentState 携带 user_query / draft_claims")


def test_5_clean_and_expected(records: List[Dict[str, Any]]) -> None:
    print("[5] 干净零误报与期望约束")
    checker = ClaimChecker()
    fp = 0
    for r in records:
        if r["injected"]:
            continue
        st = d11.state_from_snapshot(r["snapshot"], r["claims"], r["draft_answer"])
        rep = checker.check(st, use_llm=False)
        if rep.verdict != "PASS" or _flagged_ids(rep):
            fp += 1
    check(fp == 0, f"干净负样本确定性层误报 {fp}/100")
    inj = [r for r in records if r["injected"]]
    ok = 0
    for r in inj:
        exp = d11.EXPECTED[r["type"]]
        ok += set(exp) <= set(r["expected_constraints"])
    check(ok == len(inj), f"expected_constraints 含类型主约束（{ok}/{len(inj)}）")
    hit = sum(1 for r in inj if r["checker_preview"]["target_flagged"])
    check(hit >= 0.9 * len(inj), f"确定性层预览标记注入声明 {hit}/{len(inj)}（≥90%）")
    subtypes = Counter((r["type"], r["subtype"]) for r in inj)
    per_type = {t: Counter(r["subtype"] for r in inj if r["type"] == t) for t in d11.TYPES}
    check(all(n == 10 for n in per_type["numeric_tamper"].values()) and all(n == 10 for n in per_type["fake_reference"].values()),
          f"numeric_tamper / fake_reference 子类各 10：{dict(per_type['numeric_tamper'])} {dict(per_type['fake_reference'])}")
    check(len(per_type["applicability_swap"]) == 3 and len(per_type["safety_premise_removed"]) == 2,
          f"applicability 3 子类 / safety 2 子类（low_risk_shutdown 候选为 0）：{dict(per_type['applicability_swap'])} {dict(per_type['safety_premise_removed'])}")
    check(len(subtypes) == 15, f"共 15 个子类（实际 {len(subtypes)}）")


def test_6_reproducible() -> None:
    print("[6] 小规模重建可复现")
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = Path(td) / "a.jsonl", Path(td) / "b.jsonl"
        d11.build(2, 4, 20260915, p1, write_docs=False)
        d11.build(2, 4, 20260915, p2, write_docs=False)
        a = p1.read_text(encoding="utf-8")
        b = p2.read_text(encoding="utf-8")
        strip = lambda s: re.sub(r'"latency_ms": [0-9.]+', '"latency_ms": 0', s)  # noqa: E731
        check(strip(a) == strip(b), "同 seed 两次构建输出完全一致（忽略 latency_ms）")
        n = len([l for l in a.splitlines() if l.strip()])
        check(n == 12, f"--per-type 2 --clean 4 → 12 条（实际 {n}）")


def test_7_docs() -> None:
    print("[7] 文档产物")
    check(CARD_PATH.exists() and "## 注入类型" in CARD_PATH.read_text(encoding="utf-8"), "DATA_CARD.md 存在且含注入类型表")
    card = CARD_PATH.read_text(encoding="utf-8")
    check("## 已知偏差" in card and "low_risk_shutdown" in card, "DATA_CARD 说明已知偏差（含 low_risk_shutdown 候选为 0）")
    check(REVIEW_PATH.exists(), "manual_review_sample.md 存在")
    rows = [l for l in REVIEW_PATH.read_text(encoding="utf-8").splitlines() if re.match(r"\|\s*\d+\s*\|", l)]
    check(len(rows) == 40, f"人工抽检样本 40 行（实际 {len(rows)}）")
    per = Counter(l.split("|")[3].strip().split("/")[0] for l in rows)
    check(all(per[t] == 10 for t in d11.TYPES), f"每类 10 条：{dict(per)}")


def main() -> int:
    records = load()
    test_1_scale(records)
    test_2_fields(records)
    test_3_claims(records)
    test_4_rebuild(records)
    test_5_clean_and_expected(records)
    test_6_reproducible()
    test_7_docs()
    print(f"\n通过 {PASS} / 失败 {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
