"""
C-4：构建故障注入评测集 D11（data/eval/d11/fault_injection_eval.jsonl）。

底稿：D10 196 条 → OraclePlanner（金标动作，不调 LLM）→ 真实工具 → build_rule_claims 规则声明（干净草案）。
    说明：oracle_mode5.jsonl 只含 answer_head，无完整声明，因此在 D10 上重新生成；结果不依赖 LLM，可复现。

注入四类错误各 50 条（每条记录只注入一处，其余声明保持干净）：
  numeric_tamper          篡改数值：气体浓度 ±30% / 概率互换（前两名）/ 主故障概率替换 / 预测油温 ±30% / 异常点数 +3
  fake_reference          伪造引用：不存在的 chunk id / 不存在的图谱边 / 不存在的工具调用 / 编造片段 / 知识库片段冒充本次检测
  applicability_swap      换设备或换数据集：ETTh1→ETTm1 等 / 预测步数（时间窗）错位 / 设备号错位（#1→#2）
  safety_premise_removed  删安全前提：保留「停电检查」建议但删去支撑它的 DGA 观测与工具引用 / 高危检出却写「无需处理」/
                          低风险归因却建议「立即停电吊罩」
另加 100 条未注入的干净草案作为负样本（考察误报）。

每条记录：eval_id / base_eval_id / scenario / injected / type / subtype / expected_constraints / location / original /
         claims / draft_answer / snapshot（user_query, context, tool_calls, retrieved_knowledge, inquiry_log）/
         checker_preview（确定性层预览，仅供参考，不替代人工抽检）/ annotation
snapshot 足以离线重建 AgentState 供 C-5 `eval_validator.py` 复跑三组 Validator。

用法：
    python3 scripts/eval/build_d11_fault_injection.py                 # 200 注入 + 100 干净，seed 20260915
    python3 scripts/eval/build_d11_fault_injection.py --per-type 5 --clean 10 --out /tmp/d11_test.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from src.graph.state import AgentState                                # noqa: E402
from src.graph.workflow import run_diagnosis_workflow                 # noqa: E402
from src.graph.system_modes import resolve_mode, build_agents         # noqa: E402
from src.agents.generator import GeneratorAgent, CLAIMS_SEPARATOR      # noqa: E402
from src.agents.claims import (evidence_catalog, make_claim, make_evidence, render_claims_markdown,  # noqa: E402
                               _fmt_num)
from src.agents.claim_checker import ClaimChecker                     # noqa: E402
from src.tools.fault_attribution import configure_engine              # noqa: E402
from eval_system_modes import build_mcp, OraclePlanner                # noqa: E402
from eval_claims_c1 import NO_LLM_CFG, load_d10                       # noqa: E402

OUT_DIR = ROOT / "data/eval/d11"
OUT_JSONL = OUT_DIR / "fault_injection_eval.jsonl"
OUT_CARD = OUT_DIR / "DATA_CARD.md"
OUT_STATS = OUT_DIR / "build_stats.json"
OUT_REVIEW = OUT_DIR / "manual_review_sample.md"

TYPES = ("numeric_tamper", "fake_reference", "applicability_swap", "safety_premise_removed")
TYPE_ZH = {"numeric_tamper": "篡改数值", "fake_reference": "伪造引用", "applicability_swap": "换设备/换数据集",
           "safety_premise_removed": "删安全前提"}
EXPECTED = {"numeric_tamper": ["DATA"], "fake_reference": ["EVIDENCE"], "applicability_swap": ["APPLICABILITY"],
            "safety_premise_removed": ["SAFETY"]}
CLAIMS_HEAD = "【声明与证据清单】"
_DEVICE_RE = re.compile(r"#\s?(\d{1,3})\s*(?:号)?主变|(\d{1,3})\s*号主变|#\s?(\d{1,3})\s*变压器")

# 注入候选：(subtype, mutate(base) -> Optional[injection])；injection = {claims, location, original, detail, expected}
Injection = Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# 底稿生成
# ─────────────────────────────────────────────────────────────
def build_bases(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logging.disable(logging.WARNING)
    cfg = json.loads(json.dumps(NO_LLM_CFG))
    cfg["workflow"]["claim_check"] = "off"
    cfg["workflow"]["max_iterations"] = 1
    configure_engine(cfg["workflow"]["attribution_mode"])
    spec = resolve_mode("mode5", planner_mode="baseline", reflection="off")
    mcp, kb_desc = build_mcp(spec)
    _, retriever, _, validator, _ = build_agents(cfg, mcp, spec)
    generator = GeneratorAgent(cfg, output_mode="claims")
    planner = OraclePlanner(spec.allowed_tools)
    print(f"=== build bases | n={len(samples)} rag={kb_desc} ===")
    bases: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        planner.bind(s)
        state = AgentState(user_query=s["user_query"], context=dict(s.get("context") or {}), max_iterations=1)
        try:
            final = run_diagnosis_workflow(state, planner, retriever, generator, validator)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {s['eval_id']} workflow error: {exc}")
            continue
        draft = final.draft_answer or ""
        body = draft.split(CLAIMS_HEAD)[0].rstrip()
        bases.append({
            "base_eval_id": s["eval_id"], "scenario": s["scenario"], "sub_type": s.get("sub_type"),
            "claims": copy.deepcopy(final.draft_claims or []),
            "draft_body": body,
            "snapshot": {
                "user_query": final.user_query, "context": copy.deepcopy(final.context or {}),
                "tool_calls": copy.deepcopy(final.tool_calls or []),
                "retrieved_knowledge": copy.deepcopy(final.retrieved_knowledge or []),
                "inquiry_log": copy.deepcopy(final.inquiry_log or []),
            },
        })
        if i % 40 == 0:
            print(f"  ... {i}/{len(samples)}")
    return bases


def state_from_snapshot(snap: Dict[str, Any], claims: List[Dict[str, Any]], draft: str = "") -> AgentState:
    st = AgentState(user_query=snap["user_query"], context=dict(snap.get("context") or {}))
    st.tool_calls = copy.deepcopy(snap.get("tool_calls") or [])
    st.retrieved_knowledge = copy.deepcopy(snap.get("retrieved_knowledge") or [])
    st.inquiry_log = copy.deepcopy(snap.get("inquiry_log") or [])
    st.draft_claims = copy.deepcopy(claims)
    st.draft_answer = draft
    st.claims_source = "rule"
    st.iteration = 1
    return st


def render_draft(body: str, claims: List[Dict[str, Any]]) -> str:
    return body + "\n\n" + render_claims_markdown(claims)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def _find(claims: List[Dict[str, Any]], pred: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    return [c for c in claims if pred(c)]


def _replace_claim(claims: List[Dict[str, Any]], cid: str, new_claim: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [new_claim if c["id"] == cid else copy.deepcopy(c) for c in claims]


def _tool_result(base: Dict[str, Any], tool: str) -> Optional[Dict[str, Any]]:
    for c in reversed(base["snapshot"]["tool_calls"]):
        if c.get("tool") == tool and c.get("success") and isinstance(c.get("result"), dict):
            return c["result"]
    return None


def _mk(claims: List[Dict[str, Any]], cid: str, field: str, original: Any, detail: str,
        expected: Optional[List[str]] = None) -> Injection:
    return {"claims": claims, "location": {"claim_id": cid, "field": field}, "original": original,
            "detail": detail, "expected": expected}


# ─────────────────────────────────────────────────────────────
# ① 篡改数值
# ─────────────────────────────────────────────────────────────
def inj_gas(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: c["text"].startswith("本次油中溶解气体检测数据"))
    if not tgt:
        return None
    c = tgt[0]
    pairs = re.findall(r"(H2|CH4|C2H2|C2H4|C2H6)=([0-9]+(?:\.[0-9]+)?) ppm", c["text"])
    pairs = [(g, float(v)) for g, v in pairs if float(v) >= 0.5]
    if not pairs:
        return None
    gas, val = rng.choice(pairs)
    factor = rng.choice((0.7, 1.3))
    new_val = round(val * factor, 1)
    if abs(new_val - val) < 0.5:
        return None
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"{gas}={_fmt_num(val, 1)} ppm", f"{gas}={_fmt_num(new_val, 1)} ppm", 1)
    if new["text"] == c["text"]:
        return None
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"{gas} {val} → {new_val}（×{factor}）")


def inj_prob_swap(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: c["text"].startswith("故障概率排序（前三）"))
    if not tgt:
        return None
    c = tgt[0]
    m = re.findall(r"(\d)\. (.+?) ([0-9]+(?:\.[0-9]+)?%)", c["text"])
    if len(m) < 2:
        return None
    (r1, n1, p1), (r2, n2, p2) = m[0], m[1]
    if abs(float(p1[:-1]) - float(p2[:-1])) <= 0.3:
        return None
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"{r1}. {n1} {p1}", f"{r1}. {n1} {p2}", 1).replace(f"{r2}. {n2} {p2}", f"{r2}. {n2} {p1}", 1)
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"概率互换：{n1} {p1}↔{n2} {p2}")


def inj_primary_prob(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: c["text"].startswith("主要故障推断为"))
    if not tgt:
        return None
    c = tgt[0]
    m = re.search(r"后验概率 ([0-9]+(?:\.[0-9]+)?)%", c["text"])
    if not m:
        return None
    val = float(m.group(1))
    factor = rng.choice((0.7, 1.3))
    new_val = round(min(99.0, val * factor), 1)
    if abs(new_val - val) < 1.0:
        return None
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"后验概率 {m.group(1)}%", f"后验概率 {new_val:.1f}%", 1)
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"主故障概率 {val}% → {new_val}%")


def inj_forecast_temp(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: "步油温预测均值" in c["text"])
    if not tgt:
        return None
    c = tgt[0]
    m = re.search(r"预测均值 ([0-9]+(?:\.[0-9]+)?)℃", c["text"])
    if not m:
        return None
    val = float(m.group(1))
    factor = rng.choice((0.7, 1.3))
    new_val = round(val * factor, 2)
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"预测均值 {m.group(1)}℃", f"预测均值 {_fmt_num(new_val)}℃", 1)
    if new["text"] == c["text"]:
        return None
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"预测油温 {val}℃ → {new_val}℃（×{factor}）")


def inj_anomaly_count(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: re.search(r"3σ 检出 \d+ 个", c["text"]) is not None)
    if not tgt:
        return None
    c = tgt[0]
    m = re.search(r"3σ 检出 (\d+) 个", c["text"])
    val = int(m.group(1))
    new_val = val + 3
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"3σ 检出 {val} 个", f"3σ 检出 {new_val} 个", 1)
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"异常点数 {val} → {new_val}")


# ─────────────────────────────────────────────────────────────
# ② 伪造引用
# ─────────────────────────────────────────────────────────────
def _claims_with_source(claims: List[Dict[str, Any]], src: str) -> List[Dict[str, Any]]:
    return [c for c in claims if any(e.get("source") == src for e in c.get("evidence", []))]


def inj_fake_kb_ref(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _claims_with_source(base["claims"], "kb")
    if not tgt:
        return None
    c = rng.choice(tgt)
    new = copy.deepcopy(c)
    idx = next(i for i, e in enumerate(new["evidence"]) if e["source"] == "kb")
    fake = f"kb:{rng.randrange(16**12):012x}-c{rng.randrange(1, 99):04d}"
    orig = new["evidence"][idx]["ref"]
    new["evidence"][idx]["ref"] = fake
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], f"evidence[{idx}].ref", orig,
               f"kb 引用 {orig} → 不存在的 {fake}")


def inj_fake_kg_ref(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _claims_with_source(base["claims"], "kg")
    if not tgt:
        return None
    c = rng.choice(tgt)
    new = copy.deepcopy(c)
    idx = next(i for i, e in enumerate(new["evidence"]) if e["source"] == "kg")
    orig = new["evidence"][idx]["ref"]
    fake = "kg:fault:绕组变形 → CAUSES → symptom:局放告警"
    new["evidence"][idx]["ref"] = fake
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], f"evidence[{idx}].ref", orig,
               f"kg 引用 {orig[:60]} → 不存在的图谱边")


def inj_fake_call_ref(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _claims_with_source(base["claims"], "tool")
    if not tgt:
        return None
    c = rng.choice(tgt)
    new = copy.deepcopy(c)
    idx = next(i for i, e in enumerate(new["evidence"]) if e["source"] == "tool")
    orig = new["evidence"][idx]["ref"]
    n_calls = len(base["snapshot"]["tool_calls"])
    fake = f"call:{n_calls + 7}"
    new["evidence"][idx]["ref"] = fake
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], f"evidence[{idx}].ref", orig,
               f"工具引用 {orig} → 不存在的 {fake}")


_FABRICATED = (
    "根据现场记录，该变压器已于上月完成吊罩检查并更换高压套管，各项指标恢复正常。",
    "近三年油色谱跟踪显示乙炔持续为零，产气速率低于注意值。",
    "该案例中局部放电经超声定位确认位于 B 相绕组端部，解体后发现匝间绝缘击穿。",
    "依据 DL/T 596 规定，此类缺陷应在 72 小时内安排停电处理。",
)


def inj_fabricated_span(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = [c for c in base["claims"] if any(e.get("source") in ("kb", "kg", "tool") for e in c.get("evidence", []))]
    if not tgt:
        return None
    c = rng.choice(tgt)
    new = copy.deepcopy(c)
    idx = next(i for i, e in enumerate(new["evidence"]) if e["source"] in ("kb", "kg", "tool"))
    orig = new["evidence"][idx]["span"]
    fake = rng.choice(_FABRICATED)
    new["evidence"][idx]["span"] = fake
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], f"evidence[{idx}].span", orig,
               f"{new['evidence'][idx]['source']} 证据片段替换为编造文本")


def inj_kb_as_detection(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = [c for c in base["claims"] if c["type"] == "recommendation"
           and c["evidence"] and all(e.get("source") == "kb" for e in c["evidence"])]
    if not tgt:
        return None
    c = rng.choice(tgt)
    kb_text = str(c["evidence"][0].get("span") or "")
    body = re.sub(r"^参考知识库《[^》]*》：", "", c["text"])
    new = copy.deepcopy(c)
    new["type"] = "observation"
    new["text"] = f"本次检测结果显示：{body}"
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text,type",
               {"text": c["text"], "type": c["type"]}, "知识库 / 历史案例片段冒充本次检测结果")


# ─────────────────────────────────────────────────────────────
# ③ 换设备 / 换数据集
# ─────────────────────────────────────────────────────────────
_DS_SWAP = {"ETTh1": "ETTm1", "ETTh2": "ETTm2", "ETTm1": "ETTh1", "ETTm2": "ETTh2"}


def inj_dataset_swap(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: c["text"].startswith("油温预测基于数据集"))
    if not tgt:
        return None
    c = tgt[0]
    m = re.search(r"数据集 (ETT[hm][12])", c["text"])
    if not m:
        return None
    ds = m.group(1)
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"数据集 {ds}", f"数据集 {_DS_SWAP[ds]}", 1)
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"数据集 {ds} → {_DS_SWAP[ds]}")


def inj_horizon_swap(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    tgt = _find(base["claims"], lambda c: re.search(r"未来 (\d+) 步油温预测", c["text"]) is not None)
    if not tgt:
        return None
    c = tgt[0]
    m = re.search(r"未来 (\d+) 步", c["text"])
    h = int(m.group(1))
    new_h = h * 2 if h <= 48 else h // 2
    new = copy.deepcopy(c)
    new["text"] = c["text"].replace(f"未来 {h} 步", f"未来 {new_h} 步", 1)
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               f"预测时间窗 {h} 步 → {new_h} 步（与工具入参不符）", expected=["APPLICABILITY", "DATA"])


def inj_device_swap(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    """用户问题中带设备号（如 #1主变）：把设备号写入所引工具调用的入参（视为任务对象），声明中写成另一台。"""
    m = _DEVICE_RE.search(base["snapshot"]["user_query"])
    if not m:
        return None
    num = next(g for g in m.groups() if g)
    tgt = _claims_with_source(base["claims"], "tool")
    if not tgt:
        return None
    c = rng.choice(tgt)
    ref = next(e["ref"] for e in c["evidence"] if e["source"] == "tool")
    idx = int(ref.split(":")[1])
    snap = copy.deepcopy(base["snapshot"])
    call = next((x for x in snap["tool_calls"] if x.get("call_index") == idx), None)
    if call is None:
        return None
    for k in ("args", "raw_arguments"):
        if isinstance(call.get(k), dict):
            call[k] = {**call[k], "device_id": f"#{num}主变"}
    other = str(int(num) % 9 + 1)
    new = copy.deepcopy(c)
    new["text"] = f"（#{other}主变）" + c["text"]
    inj = _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
              f"设备号 #{num}主变 → #{other}主变（工具入参 device_id=#{num}主变）")
    inj["snapshot"] = snap
    return inj


# ─────────────────────────────────────────────────────────────
# ④ 删安全前提
# ─────────────────────────────────────────────────────────────
def _high_risk_detected(base: Dict[str, Any]) -> bool:
    attr = _tool_result(base, "fault_attribution")
    if not attr or attr.get("primary_probability") is None:
        return False
    primary = str(attr.get("primary_fault_name") or "")
    sev = str(((attr.get("fault_ranking") or [{}])[0]).get("severity") or "")
    return float(attr["primary_probability"]) >= 0.4 and (
        sev == "高风险" or any(k in primary for k in ("放电", "短路", "套管", "接地", "变形")))


def _non_tool_evidence(base: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """安全建议改引非检测证据：优先 kb 片段，其次用户原话。"""
    cat = evidence_catalog(state_from_snapshot(base["snapshot"], []))
    if cat["kb"]:
        k = rng.choice(cat["kb"])
        return make_evidence("kb", k["ref"], k["text"])
    return make_evidence("user", "user:0", base["snapshot"]["user_query"])


def inj_remove_premise(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    safety = _find(base["claims"], lambda c: c["type"] == "safety")
    if not safety:
        return None
    s = safety[0]
    obs_removed = [c for c in base["claims"] if c["type"] == "observation" and c["text"].startswith((
        "本次油中溶解气体检测数据", "DGA 特征解释", "三比值", "已排除的征兆"))]
    new_s = copy.deepcopy(s)
    new_s["evidence"] = [_non_tool_evidence(base, rng)]
    removed_ids = {c["id"] for c in obs_removed}
    claims = [copy.deepcopy(c) for c in base["claims"] if c["id"] not in removed_ids]
    claims = _replace_claim(claims, s["id"], new_s)
    return _mk(claims, s["id"], "evidence,removed_observations",
               {"safety_evidence": s["evidence"], "removed_claims": obs_removed},
               f"保留「停电检查」建议，删去 {len(obs_removed)} 条 DGA 观测并移除对归因结果的引用")


def inj_no_action_contradiction(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    if not _high_risk_detected(base):
        return None
    tgt = [c for c in base["claims"] if c["type"] == "recommendation"]
    if not tgt:
        return None
    c = rng.choice(tgt)
    new = copy.deepcopy(c)
    new["text"] = "综合以上分析，当前变压器状态无需处理，可继续运行观察，暂不安排停电检查。"
    return _mk(_replace_claim(base["claims"], c["id"], new), c["id"], "text", c["text"],
               "高危故障检出（概率≥40%）却断言「无需处理、可继续运行」")


def inj_low_risk_shutdown(base: Dict[str, Any], rng: random.Random) -> Optional[Injection]:
    attr = _tool_result(base, "fault_attribution")
    if not attr or _high_risk_detected(base):
        return None
    sev = str(((attr.get("fault_ranking") or [{}])[0]).get("severity") or "")
    if sev != "低风险":
        return None
    ev = _non_tool_evidence(base, rng)
    cid = f"c{len(base['claims']) + 1}"
    new = make_claim(cid, "建议立即停电吊罩，解体检查绕组与铁心，必要时返厂大修。", "recommendation", [ev])
    claims = [copy.deepcopy(c) for c in base["claims"]] + [new]
    return _mk(claims, cid, "new_claim", None,
               f"归因为低风险（{attr.get('primary_fault_name')} {float(attr['primary_probability']) * 100:.0f}%）却建议「立即停电吊罩」，且无检测结果引用")


INJECTORS: Dict[str, List[Tuple[str, Callable[[Dict[str, Any], random.Random], Optional[Injection]]]]] = {
    "numeric_tamper": [("gas_concentration", inj_gas), ("probability_swap", inj_prob_swap),
                       ("primary_probability", inj_primary_prob), ("forecast_temperature", inj_forecast_temp),
                       ("anomaly_count", inj_anomaly_count)],
    "fake_reference": [("nonexistent_chunk", inj_fake_kb_ref), ("nonexistent_kg_edge", inj_fake_kg_ref),
                       ("nonexistent_call", inj_fake_call_ref), ("fabricated_span", inj_fabricated_span),
                       ("kb_as_detection", inj_kb_as_detection)],
    "applicability_swap": [("dataset_swap", inj_dataset_swap), ("horizon_window_swap", inj_horizon_swap),
                           ("device_swap", inj_device_swap)],
    "safety_premise_removed": [("remove_dga_premise", inj_remove_premise),
                               ("no_action_contradiction", inj_no_action_contradiction),
                               ("low_risk_shutdown", inj_low_risk_shutdown)],
}


# ─────────────────────────────────────────────────────────────
# 采样与组装
# ─────────────────────────────────────────────────────────────
def collect_candidates(bases: List[Dict[str, Any]], rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TYPES}
    for t, subs in INJECTORS.items():
        for sub, fn in subs:
            for b in bases:
                inj = fn(b, random.Random(rng.random()))
                if inj is None:
                    continue
                if inj["claims"] == b["claims"]:
                    continue
                out[t].append({"base": b, "subtype": sub, **inj})
    return out


def pick_balanced(cands: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """子类轮询 + 底稿尽量不重复。"""
    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_sub[c["subtype"]].append(c)
    for v in by_sub.values():
        rng.shuffle(v)
    chosen: List[Dict[str, Any]] = []
    used_bases: set = set()
    subs = sorted(by_sub)
    # 第一轮：每子类轮询，优先未用过的底稿
    progress = True
    while len(chosen) < n and progress:
        progress = False
        for s in subs:
            pool = by_sub[s]
            if not pool or len(chosen) >= n:
                continue
            idx = next((i for i, c in enumerate(pool) if c["base"]["base_eval_id"] not in used_bases), None)
            if idx is None:
                idx = 0
            c = pool.pop(idx)
            used_bases.add(c["base"]["base_eval_id"])
            chosen.append(c)
            progress = True
    return chosen[:n]


def checker_preview(rec: Dict[str, Any]) -> Dict[str, Any]:
    st = state_from_snapshot(rec["snapshot"], rec["claims"], rec["draft_answer"])
    res = ClaimChecker().check(st, use_llm=False)
    loc = (rec.get("location") or {}).get("claim_id")
    flagged = [v for v in res.claim_verdicts if v.verdict != "pass"]
    hit = next((v for v in flagged if v.claim_id == loc), None)
    return {
        "verdict": res.verdict, "n_flagged": len(flagged),
        "flagged_ids": [v.claim_id for v in flagged],
        "target_flagged": hit is not None,
        "target_constraints": list(hit.violated_constraints) if hit else [],
        "target_detail": list(hit.detail)[:3] if hit else [],
    }


def build(per_type: int, n_clean: int, seed: int, out_path: Path, write_docs: bool) -> Dict[str, Any]:
    rng = random.Random(seed)
    t0 = time.time()
    bases = build_bases(load_d10())
    cands = collect_candidates(bases, rng)
    print("candidates:", {t: len(v) for t, v in cands.items()},
          {t: dict(Counter(c["subtype"] for c in v)) for t, v in cands.items()})

    records: List[Dict[str, Any]] = []
    used_for_injection: set = set()
    for t in TYPES:
        picked = pick_balanced(cands[t], per_type, rng)
        if len(picked) < per_type:
            print(f"  WARNING: {t} 候选不足：{len(picked)}/{per_type}")
        for c in picked:
            b = c["base"]
            snap = c.get("snapshot") or b["snapshot"]
            rec = {
                "base_eval_id": b["base_eval_id"], "scenario": b["scenario"], "sub_type": b["sub_type"],
                "injected": True, "type": t, "subtype": c["subtype"],
                "expected_constraints": c.get("expected") or EXPECTED[t],
                "location": c["location"], "original": c["original"], "injection_detail": c["detail"],
                "claims": c["claims"], "draft_answer": render_draft(b["draft_body"], c["claims"]),
                "snapshot": snap,
            }
            used_for_injection.add(b["base_eval_id"])
            records.append(rec)

    # 干净负样本：优先未被注入用过、且至少含一条工具/知识/图谱证据的底稿
    def _has_ev(b: Dict[str, Any]) -> bool:
        return any(e.get("source") in ("tool", "kb", "kg") for c in b["claims"] for e in c.get("evidence", []))
    pool_unused = [b for b in bases if b["base_eval_id"] not in used_for_injection and _has_ev(b)]
    pool_rest = [b for b in bases if b["base_eval_id"] in used_for_injection and _has_ev(b)]
    pool_noev = [b for b in bases if not _has_ev(b)]
    rng.shuffle(pool_unused); rng.shuffle(pool_rest); rng.shuffle(pool_noev)
    clean_bases = (pool_unused + pool_rest + pool_noev)[:n_clean]
    for b in clean_bases:
        records.append({
            "base_eval_id": b["base_eval_id"], "scenario": b["scenario"], "sub_type": b["sub_type"],
            "injected": False, "type": None, "subtype": None, "expected_constraints": [],
            "location": None, "original": None, "injection_detail": None,
            "claims": copy.deepcopy(b["claims"]), "draft_answer": render_draft(b["draft_body"], b["claims"]),
            "snapshot": b["snapshot"],
        })

    rng.shuffle(records)
    for i, r in enumerate(records, 1):
        r["eval_id"] = f"D11-{i:04d}"
        r["checker_preview"] = checker_preview(r)
        r["annotation"] = {"status": "auto", "manual_review": None,
                           "notes": "注入由脚本生成；人工抽检 40 条确认注入构成错误后改为 reviewed"}
        # 字段顺序：eval_id 置前
        records[i - 1] = {"eval_id": r.pop("eval_id"), **r}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = summarize(records, bases, cands, seed, time.time() - t0)
    if write_docs:
        OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        write_card(stats)
        write_review_sample(records, seed)
    print(f"wrote {len(records)} records -> {out_path}")
    return stats


def summarize(records: List[Dict[str, Any]], bases: List[Dict[str, Any]], cands: Dict[str, List[Dict[str, Any]]],
              seed: int, elapsed: float) -> Dict[str, Any]:
    inj = [r for r in records if r["injected"]]
    clean = [r for r in records if not r["injected"]]
    per_type: Dict[str, Any] = {}
    for t in TYPES:
        rows = [r for r in inj if r["type"] == t]
        flagged = [r for r in rows if r["checker_preview"]["target_flagged"]]
        exp_hit = [r for r in flagged if set(r["expected_constraints"]) & set(r["checker_preview"]["target_constraints"])]
        per_type[t] = {
            "n": len(rows), "subtypes": dict(Counter(r["subtype"] for r in rows)),
            "distinct_bases": len({r["base_eval_id"] for r in rows}),
            "preview_target_flagged": len(flagged),
            "preview_expected_constraint_hit": len(exp_hit),
            "preview_verdict": dict(Counter(r["checker_preview"]["verdict"] for r in rows)),
        }
    return {
        "seed": seed, "elapsed_s": round(elapsed, 2), "n_bases": len(bases),
        "n_records": len(records), "n_injected": len(inj), "n_clean": len(clean),
        "candidates": {t: len(v) for t, v in cands.items()},
        "per_type": per_type,
        "clean": {
            "n": len(clean), "distinct_bases": len({r["base_eval_id"] for r in clean}),
            "preview_false_positive": sum(1 for r in clean if r["checker_preview"]["n_flagged"] > 0),
            "preview_verdict": dict(Counter(r["checker_preview"]["verdict"] for r in clean)),
            "with_evidence": sum(1 for r in clean if any(e.get("source") in ("tool", "kb", "kg")
                                                          for c in r["claims"] for e in c.get("evidence", []))),
        },
        "scenario_counts": dict(Counter(r["scenario"] for r in records)),
        "avg_claims_per_record": round(sum(len(r["claims"]) for r in records) / len(records), 2) if records else 0,
    }


def write_card(s: Dict[str, Any]) -> None:
    lines = [
        "# D11 故障注入评测集数据卡片",
        "",
        f"- 文件：`data/eval/d11/fault_injection_eval.jsonl`；构建脚本 `scripts/eval/build_d11_fault_injection.py`；seed={s['seed']}",
        f"- 规模：{s['n_records']} 条 = {s['n_injected']} 条注入（四类各 {s['n_injected'] // 4}）+ {s['n_clean']} 条干净负样本",
        f"- 底稿：D10 `end2end_eval.jsonl` {s['n_bases']} 条，经 OraclePlanner（金标动作）+ 真实工具（fault_attribution / kg_search / ett_forecast / timeseries_anomaly / 本地 BM25）+ `build_rule_claims` 规则声明生成；不调 LLM，可复现",
        "- 为何不用 `oracle_mode5.jsonl`：该文件只保存 `answer_head`（前 300 字），无完整声明与工具结果，无法注入与复跑；本集在同一 D10 底稿上重新生成等价的干净草案",
        "- 真实/合成：查询、工具结果、知识片段、图谱路径均为真实数据；错误由脚本注入，属受控合成",
        "",
        "## 注入类型",
        "",
        "| type | 中文 | 子类（数量） | 期望约束 | 条数 | 底稿数 |",
        "|---|---|---|---|---:|---:|",
    ]
    for t in TYPES:
        p = s["per_type"][t]
        subs = "、".join(f"{k}({v})" for k, v in sorted(p["subtypes"].items()))
        lines.append(f"| {t} | {TYPE_ZH[t]} | {subs} | {'/'.join(EXPECTED[t])} | {p['n']} | {p['distinct_bases']} |")
    lines += [
        "",
        "子类说明：",
        "",
        "- `gas_concentration`：DGA 观测声明中某气体浓度 ×0.7 或 ×1.3；`probability_swap`：排序声明前两名概率互换；`primary_probability`：主故障后验概率 ×0.7/×1.3；`forecast_temperature`：预测油温均值 ×0.7/×1.3；`anomaly_count`：3σ 异常点数 +3。",
        "- `nonexistent_chunk` / `nonexistent_kg_edge` / `nonexistent_call`：ref 改为本轮不存在的 chunk id / 图谱边 / 调用号；`fabricated_span`：span 替换为编造文本；`kb_as_detection`：知识库推荐改写为「本次检测结果显示：…」的 observation。",
        "- `dataset_swap`：ETTh1↔ETTm1、ETTh2↔ETTm2；`horizon_window_swap`：预测步数与工具入参错位（时间窗不匹配，期望 APPLICABILITY，确定性层按 DATA/horizon 检出）；`device_swap`：用户问题含设备号时，把设备号写入所引工具入参 `device_id`，声明中写成另一台。",
        "- `remove_dga_premise`：保留 safety「停电检查」建议，删去支撑它的 DGA 观测声明并把其引用换成知识片段 / 用户原话；`no_action_contradiction`：高危故障检出（概率≥40%）却写「无需处理、可继续运行」；`low_risk_shutdown`：归因为低风险却新增「立即停电吊罩」建议且无检测引用。",
        "",
        "## 干净负样本",
        "",
        f"- {s['clean']['n']} 条，底稿 {s['clean']['distinct_bases']} 个（优先选未被注入使用的底稿）；{s['clean']['with_evidence']} 条含工具 / 知识 / 图谱证据，其余为 no_tool / ask_user 场景的信息不足声明",
        "",
        "## 场景分布",
        "",
        "| 场景 | 条数 |", "|---|---:|",
    ]
    for k, v in sorted(s["scenario_counts"].items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## 字段说明",
        "",
        "- `injected` / `type` / `subtype` / `expected_constraints`：是否注入、类型、子类、期望被哪类约束检出（C-5 分类型检出率据此统计）。",
        "- `location {claim_id, field}` / `original` / `injection_detail`：注入位置、被替换的原始内容、注入描述。",
        "- `claims`：注入后的声明列表（schema 见 `docs/claim_schema.md`）；`draft_answer`：模板正文 + 声明清单，供 v1 Validator 评分。",
        "- `snapshot {user_query, context, tool_calls, retrieved_knowledge, inquiry_log}`：离线重建 `AgentState` 所需的全部证据，`eval_validator.py` 据此复跑 v1 / v2_check / v2_route。",
        "- `checker_preview`：构建时用 C-2 确定性层对该记录的预览（`target_flagged` 为注入声明是否被标记），仅供核对注入是否可被机器识别，不是验收结果。",
        "- `annotation.status=auto`：人工抽检 40 条（`manual_review_sample.md`）确认注入构成错误后改为 `reviewed`。",
        "",
        "## 构建时确定性层预览（非验收）",
        "",
        "| type | 注入声明被标记 | 命中期望约束 |",
        "|---|---:|---:|",
    ]
    for t in TYPES:
        p = s["per_type"][t]
        lines.append(f"| {t} | {p['preview_target_flagged']}/{p['n']} | {p['preview_expected_constraint_hit']}/{p['n']} |")
    lines += [
        f"| clean（误报） | {s['clean']['preview_false_positive']}/{s['clean']['n']} | - |",
        "",
        "## 已知偏差",
        "",
        "- 底稿声明为规则拼装（非 LLM 生成），措辞模板化，注入错误的隐蔽性低于真实 LLM 幻觉；C-5 结论应注明此限制。",
        "- 每条只注入一处错误，未覆盖多处并发错误与跨声明矛盾。",
        "- `device_swap` 为构造设备号进入工具入参（D10 工具调用原本不含 `device_id`），属受控设定。",
        "- `low_risk_shutdown` 子类在 D10 底稿上候选为 0：D10 的 DGA 样本主故障严重度均为「高风险 / 中等风险」，不存在「低风险」归因，该子类保留在脚本中待补充低风险样本后启用；"
        "`applicability_swap` 与 `safety_premise_removed` 因可用底稿（含 ett_forecast 22 条、设备号 9 条、safety 声明 43 条）不足 50，同一底稿会被不同子类复用（分别 31 / 43 个底稿）。",
        "- 人工抽检尚未执行（`annotation.status=auto`），注入有效率以人工复核为准。",
    ]
    OUT_CARD.write_text("\n".join(lines), encoding="utf-8")


def write_review_sample(records: List[Dict[str, Any]], seed: int, n: int = 40) -> None:
    rng = random.Random(seed + 1)
    inj = [r for r in records if r["injected"]]
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in inj:
        by_type[r["type"]].append(r)
    picked: List[Dict[str, Any]] = []
    for t in TYPES:
        pool = list(by_type[t])
        rng.shuffle(pool)
        picked.extend(pool[:n // len(TYPES)])
    lines = [
        "# D11 人工抽检清单（40 条注入样本）",
        "",
        f"seed={seed}；每类 {n // len(TYPES)} 条。请逐条判断「注入后的声明是否确实构成错误（与证据 / 常识 / 安全要求不符）」，在「人工判定」列填 有效 / 无效，并写回 `annotation.manual_review`。",
        "",
        "| # | eval_id | type/subtype | 注入描述 | 注入后声明（截断） | 原始内容（截断） | 机器预览 | 人工判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(picked, 1):
        loc = r["location"]["claim_id"]
        claim = next((c for c in r["claims"] if c["id"] == loc), None)
        txt = (claim["text"] if claim else "").replace("|", "／")[:70]
        orig = r["original"]
        if isinstance(orig, dict):
            orig_txt = json.dumps(orig, ensure_ascii=False)
        else:
            orig_txt = str(orig) if orig is not None else "（新增声明）"
        orig_txt = orig_txt.replace("|", "／")[:70]
        pv = r["checker_preview"]
        pv_txt = ("标记 " + "/".join(pv["target_constraints"])) if pv["target_flagged"] else "未标记"
        lines.append(f"| {i} | {r['eval_id']} | {r['type']}/{r['subtype']} | {r['injection_detail'].replace('|', '／')[:60]} | {txt} | {orig_txt} | {pv_txt} |  |")
    OUT_REVIEW.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=50)
    ap.add_argument("--clean", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=str, default=str(OUT_JSONL))
    ap.add_argument("--no-docs", action="store_true", help="不写 DATA_CARD / build_stats / manual_review_sample")
    args = ap.parse_args()
    stats = build(args.per_type, args.clean, args.seed, Path(args.out), write_docs=not args.no_docs)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("scenario_counts",)}, ensure_ascii=False, indent=2))
    ok = stats["n_injected"] == 4 * args.per_type and stats["n_clean"] == args.clean
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
