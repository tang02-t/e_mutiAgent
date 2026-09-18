"""
C-1 声明与证据（claims）公共模块。

规范见 docs/design/claim_schema.md。本模块提供：
- CLAIM_TYPES / EVIDENCE_SOURCES / CONSTRAINT_TYPES 常量
- evidence_catalog(state)：枚举本轮可被引用的证据（工具调用 / 知识片段 / 图谱关系 / 用户轮次）
- render_evidence_catalog(catalog)：把证据目录渲染成提示词文本（claims 模式下附给 LLM）
- validate_claims(obj)：对 LLM 或规则产出的 claims 做 schema 校验，返回 (ok, errors, normalized_claims)
- build_rule_claims(state)：不依赖 LLM，按工具结果规则拼装 claims（模板回退与 LLM 解析失败时使用）
- render_claims_markdown(claims)：把 claims 渲染为可读文本（附在草案末尾）

引用（ref）格式：
- tool → "call:<call_index>"（Retriever 记录的连续序号，追问循环中跨轮唯一）
- kb   → "kb:<chunk_id>"
- kg   → "kg:<path>"（path 为 kg_search 返回的关系链字符串）
- user → "user:<round>"（0 为原始提问，n 为第 n 轮追问回答）
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


CLAIM_TYPES = ("observation", "inference", "recommendation", "safety")
EVIDENCE_SOURCES = ("tool", "kb", "kg", "user")
CONSTRAINT_TYPES = ("DATA", "EVIDENCE", "APPLICABILITY", "SAFETY")

SPAN_MAX = 300
_REF_PATTERN = re.compile(r"^(call|kb|kg|user):.+$")

# 触发 safety 类声明的高危故障（与 fault_attribution 的 FAULT_IDS 对齐：涉及放电 / 短路 / 套管 / 铁芯接地）
_HIGH_RISK_FAULTS = {
    "winding_short_circuit", "partial_discharge", "bushing_fault", "core_grounding", "winding_deformation",
}
_HIGH_RISK_SEVERITY = {"高风险", "高", "严重", "high", "critical"}


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
def _compact(obj: Any, limit: int = SPAN_MAX) -> str:
    """把对象压成单行 JSON / 字符串作为 span，超长截断。"""
    if isinstance(obj, str):
        text = obj.strip()
    else:
        try:
            text = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            text = str(obj)
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _fmt_num(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def tool_ref(call: Dict[str, Any]) -> str:
    idx = call.get("call_index")
    if idx is None:
        idx = call.get("call_id") or "?"
    return f"call:{idx}"


def kb_ref(item: Dict[str, Any], fallback_idx: int) -> str:
    cid = item.get("chunk_id") or item.get("id")
    if not cid and isinstance(item.get("child"), dict):
        cid = item["child"].get("id")
    if not cid:
        cid = item.get("path") or f"#{fallback_idx}"
    return f"kb:{cid}"


def kg_ref(path: Dict[str, Any]) -> str:
    p = path.get("path") if isinstance(path, dict) else path
    if isinstance(p, (list, tuple)):
        p = " → ".join(str(x) for x in p)
    return f"kg:{p}"


def make_evidence(source: str, ref: str, span: Any, **extra: Any) -> Dict[str, Any]:
    ev: Dict[str, Any] = {"source": source, "ref": ref, "span": _compact(span)}
    for k, v in extra.items():
        if v is not None:
            ev[k] = v
    return ev


def make_claim(cid: str, text: str, ctype: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": cid, "text": text.strip(), "type": ctype, "evidence": list(evidence)}


# ──────────────────────────────────────────────────────────────
# 证据目录
# ──────────────────────────────────────────────────────────────
def _ok_calls(state) -> List[Dict[str, Any]]:
    return [c for c in (state.tool_calls or []) if c.get("success") and c.get("result")]


def evidence_catalog(state) -> Dict[str, Any]:
    """
    枚举本轮可引用的证据。返回：
    {
      "tool": [{ref, tool, call_index, call_id, inquiry_round, args, result_summary}],
      "kb":   [{ref, chunk_id, source, title, text}],
      "kg":   [{ref, path, confidence, support_count, evidence}],
      "user": [{ref, round, text}],
      "refs": set(所有合法 ref)
    }
    """
    cat: Dict[str, Any] = {"tool": [], "kb": [], "kg": [], "user": [], "refs": set()}

    for c in _ok_calls(state):
        ref = tool_ref(c)
        res = c.get("result")
        summary: Dict[str, Any] = {}
        if isinstance(res, dict):
            for k in ("status", "primary_fault", "primary_fault_name", "primary_probability",
                      "mean", "std", "anomaly_indices", "dataset", "horizon", "forecast", "summary"):
                if k in res:
                    summary[k] = res[k]
        cat["tool"].append({
            "ref": ref, "tool": c.get("tool"), "call_index": c.get("call_index"),
            "call_id": c.get("call_id"), "inquiry_round": c.get("inquiry_round", 0),
            "args": c.get("raw_arguments") or c.get("args") or {},
            "result_summary": summary,
        })
        cat["refs"].add(ref)

    for i, item in enumerate((state.retrieved_knowledge or [])[:10], start=1):
        if not isinstance(item, dict):
            continue
        ref = kb_ref(item, i)
        meta = item.get("metadata") or {}
        text = item.get("snippet") or item.get("text") or ""
        if isinstance(item.get("child"), dict) and item["child"].get("text"):
            text = item["child"]["text"]
        cat["kb"].append({
            "ref": ref, "chunk_id": ref[3:],
            "source": meta.get("doc_name") or meta.get("source") or item.get("path") or "",
            "title": meta.get("title") or "",
            "text": str(text),
        })
        cat["refs"].add(ref)

    for c in _ok_calls(state):
        if c.get("tool") != "kg_search":
            continue
        res = c.get("result") or {}
        for p in (res.get("paths") or [])[:10]:
            if not isinstance(p, dict):
                continue
            ref = kg_ref(p)
            cat["kg"].append({
                "ref": ref, "path": p.get("path"), "confidence": p.get("confidence"),
                "support_count": p.get("support_count"), "evidence": p.get("evidence") or "",
            })
            cat["refs"].add(ref)

    cat["user"].append({"ref": "user:0", "round": 0, "text": state.user_query or ""})
    cat["refs"].add("user:0")
    for e in (getattr(state, "inquiry_log", None) or []):
        if e.get("action") != "ask" or e.get("symptom") is None:
            continue
        rnd = int(e.get("round", 0) or 0)
        ans = e.get("answer")
        ans_txt = "有" if ans is True else ("无" if ans is False else "未知")
        ref = f"user:{rnd}"
        cat["user"].append({"ref": ref, "round": rnd, "symptom": e.get("symptom"),
                            "answer": ans, "text": f"{e.get('symptom')}={ans_txt}"})
        cat["refs"].add(ref)
    return cat


def render_evidence_catalog(cat: Dict[str, Any]) -> str:
    """把证据目录渲染为提示词文本，供 claims 模式下 LLM 选择引用。"""
    lines: List[str] = ["【可引用证据目录】（evidence.ref 必须从下列 ref 中选取，span 必须是对应内容的原文片段）"]
    for t in cat.get("tool", []):
        lines.append(f"- {t['ref']}  工具={t['tool']}  入参={_compact(t['args'], 160)}  结果摘要={_compact(t['result_summary'], 220)}")
    for k in cat.get("kb", []):
        head = k.get("title") or k.get("source") or ""
        lines.append(f"- {k['ref']}  知识片段[{head}]：{_compact(k['text'], 200)}")
    for g in cat.get("kg", []):
        lines.append(f"- {g['ref']}  置信度={g.get('confidence')}  支持文献={g.get('support_count')}  证据：{_compact(g.get('evidence'), 160)}")
    for u in cat.get("user", []):
        lines.append(f"- {u['ref']}  用户输入：{_compact(u['text'], 160)}")
    if not (cat.get("tool") or cat.get("kb") or cat.get("kg")):
        lines.append("- （本轮无任何可引用证据（工具 / 知识库 / 图谱）；请只输出说明信息不足的 observation 声明并引用 user:0）")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# schema 校验
# ──────────────────────────────────────────────────────────────
def validate_claims(obj: Any, known_refs: Optional[set] = None) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    校验 claims 列表（或含 claims 键的 dict）。
    返回 (ok, errors, normalized)。ok 为 True 当且仅当结构合法且每条 claim 至少一条 evidence。
    known_refs 非空时额外检查 ref 是否在本轮证据目录中（结构合法但引用未知记为 warning 级错误，不影响 ok）。
    """
    errors: List[str] = []
    claims = obj.get("claims") if isinstance(obj, dict) else obj
    if not isinstance(claims, list):
        return False, ["claims 不是列表"], []

    normalized: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            errors.append(f"claims[{i}] 不是对象")
            continue
        cid = str(c.get("id") or f"c{i + 1}")
        if cid in seen_ids:
            errors.append(f"claims[{i}] id 重复：{cid}")
        seen_ids.add(cid)
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{cid}: text 为空")
        ctype = c.get("type")
        if ctype not in CLAIM_TYPES:
            errors.append(f"{cid}: type 非法：{ctype}")
        evs = c.get("evidence")
        if not isinstance(evs, list) or not evs:
            errors.append(f"{cid}: evidence 缺失或为空")
            evs = []
        norm_evs: List[Dict[str, Any]] = []
        for j, e in enumerate(evs):
            if not isinstance(e, dict):
                errors.append(f"{cid}.evidence[{j}] 不是对象")
                continue
            src = e.get("source")
            ref = e.get("ref")
            span = e.get("span")
            if src not in EVIDENCE_SOURCES:
                errors.append(f"{cid}.evidence[{j}] source 非法：{src}")
            if not isinstance(ref, str) or not _REF_PATTERN.match(ref):
                errors.append(f"{cid}.evidence[{j}] ref 格式非法：{ref}")
            elif src in EVIDENCE_SOURCES and not ref.startswith({"tool": "call", "kb": "kb", "kg": "kg", "user": "user"}[src] + ":"):
                errors.append(f"{cid}.evidence[{j}] ref 前缀与 source 不匹配：{src} / {ref}")
            if not isinstance(span, str) or not span.strip():
                errors.append(f"{cid}.evidence[{j}] span 为空")
            ne = {"source": src, "ref": ref, "span": _compact(span if isinstance(span, str) else "")}
            for k in ("tool", "call_id"):
                if k in e:
                    ne[k] = e[k]
            norm_evs.append(ne)
        normalized.append({"id": cid, "text": (text or "").strip() if isinstance(text, str) else "",
                           "type": ctype, "evidence": norm_evs})

    ok = not errors
    if known_refs:
        for c in normalized:
            for e in c["evidence"]:
                if e["ref"] not in known_refs:
                    errors.append(f"[warning] {c['id']}: ref 不在本轮证据目录：{e['ref']}")
    return ok, errors, normalized


# ──────────────────────────────────────────────────────────────
# 规则拼装（无 LLM）
# ──────────────────────────────────────────────────────────────
class _IdGen:
    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        self.n += 1
        return f"c{self.n}"


def _latest_ok_call(state, tool: str) -> Optional[Dict[str, Any]]:
    calls = [c for c in _ok_calls(state) if c.get("tool") == tool]
    return calls[-1] if calls else None


def _attribution_claims(state, ids: _IdGen) -> List[Dict[str, Any]]:
    call = _latest_ok_call(state, "fault_attribution")
    if not call:
        return []
    res = call.get("result") or {}
    if res.get("status") != "ok":
        return []
    ref = tool_ref(call)
    tool = "fault_attribution"
    args = call.get("raw_arguments") or call.get("args") or {}
    out: List[Dict[str, Any]] = []

    dga = args.get("dga_data") if isinstance(args, dict) else None
    if isinstance(dga, dict) and dga:
        gases = "、".join(f"{k}={_fmt_num(v, 1)} ppm" for k, v in dga.items())
        out.append(make_claim(ids.next(), f"本次油中溶解气体检测数据：{gases}。", "observation",
                              [make_evidence("tool", ref, {"dga_data": dga}, tool=tool)]))

    dga_an = res.get("dga_analysis") or {}
    interp = (dga_an.get("interpretation") or "").strip()
    if interp:
        out.append(make_claim(ids.next(), f"DGA 特征解释：{interp}", "observation",
                              [make_evidence("tool", ref, {"dga_analysis": {"interpretation": interp}}, tool=tool)]))

    matched = dga_an.get("matched_rules") or []
    codes = dga_an.get("ratio_codes") or {}
    if codes and not dga_an.get("ratios_incomplete"):
        code_txt = "-".join(str(codes.get(k, "?")) for k in ("code_C2H2_C2H4", "code_CH4_H2", "code_C2H4_C2H6"))
        out.append(make_claim(ids.next(), f"三比值法编码（C2H2/C2H4、CH4/H2、C2H4/C2H6）：{code_txt}。", "observation",
                              [make_evidence("tool", ref, {"ratio_codes": codes}, tool=tool)]))
    if matched:
        names = "、".join(
            f"{m.get('rule_name')}（置信度 {m.get('confidence')}）" if isinstance(m, dict) else str(m)
            for m in matched[:3])
        slim = [{"rule_name": m.get("rule_name"), "confidence": m.get("confidence")} if isinstance(m, dict) else m
                for m in matched[:3]]
        out.append(make_claim(ids.next(), f"三比值 / 特征气体规则命中：{names}。", "observation",
                              [make_evidence("tool", ref, {"matched_rules": slim}, tool=tool)]))

    neg = res.get("evidence_negative") or []
    if neg:
        out.append(make_claim(ids.next(), f"已排除的征兆（负观测）：{', '.join(map(str, neg))}。", "observation",
                              [make_evidence("tool", ref, {"evidence_negative": neg}, tool=tool)]))

    primary = res.get("primary_fault_name") or res.get("primary_fault")
    prob = res.get("primary_probability")
    unc = res.get("uncertainty") or {}
    if primary and prob is not None:
        gap = unc.get("top1_top2_gap")
        conf = "确定" if (isinstance(prob, (int, float)) and prob >= 0.7) else ("可能" if (isinstance(prob, (int, float)) and prob >= 0.4) else "待确认")
        text = f"主要故障推断为「{primary}」，后验概率 {float(prob) * 100:.1f}%（置信等级：{conf}）。"
        span_obj = {"primary_fault": res.get("primary_fault"), "primary_fault_name": primary,
                    "primary_probability": prob}
        if gap is not None:
            span_obj["top1_top2_gap"] = gap
        out.append(make_claim(ids.next(), text, "inference",
                              [make_evidence("tool", ref, span_obj, tool=tool)]))

    ranking = res.get("fault_ranking") or []
    if len(ranking) >= 2:
        top = "；".join(f"{r.get('rank')}. {r.get('fault_name')} {r.get('probability_pct')}" for r in ranking[:3])
        slim_rank = [{"rank": r.get("rank"), "fault_id": r.get("fault_id"), "fault_name": r.get("fault_name"),
                      "probability": r.get("probability"), "probability_pct": r.get("probability_pct")}
                     for r in ranking[:3]]
        out.append(make_claim(ids.next(), f"故障概率排序（前三）：{top}。", "inference",
                              [make_evidence("tool", ref, {"fault_ranking": slim_rank}, tool=tool)]))

    if unc and unc.get("entropy_bits") is not None:
        slim_unc = {k: unc.get(k) for k in ("entropy_bits", "top1_top2_gap", "calibrated", "top1", "top1_prob") if k in unc}
        out.append(make_claim(
            ids.next(),
            f"归因不确定性：后验熵 {_fmt_num(unc.get('entropy_bits'))} bit"
            + (f"，Top-1 与 Top-2 概率差 {float(unc.get('top1_top2_gap', 0)) * 100:.1f}%" if unc.get("top1_top2_gap") is not None else "")
            + ("（参数已校准）" if unc.get("calibrated") else "") + "。",
            "inference",
            [make_evidence("tool", ref, {"uncertainty": slim_unc}, tool=tool)]))

    # 安全类：仅当主要故障高危且有概率支撑时产出，并引用归因结果 + 支撑其的 DGA 观测
    sev = ""
    for r in ranking[:1]:
        sev = str(r.get("severity") or "")
    fault_id = str(res.get("primary_fault") or "")
    is_high = (fault_id in _HIGH_RISK_FAULTS) or (sev in _HIGH_RISK_SEVERITY) or (
        any(k in str(primary or "") for k in ("放电", "短路", "套管", "接地", "变形")))
    if primary and prob is not None and is_high and float(prob) >= 0.4:
        evs = [make_evidence("tool", ref, {"primary_fault_name": primary, "primary_probability": prob,
                                           "severity": sev}, tool=tool)]
        if isinstance(dga, dict) and dga:
            evs.append(make_evidence("tool", ref, {"dga_data": dga}, tool=tool))
        out.append(make_claim(
            ids.next(),
            f"鉴于「{primary}」概率 {float(prob) * 100:.1f}% 且属高危类型，建议在确认差动 / 瓦斯保护正常、"
            "复测 DGA 趋势后申请停电检查；停电检修须执行工作票、验电、接地等安全措施。",
            "safety", evs))
    return out


def _inquiry_claims(state, ids: _IdGen) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in (getattr(state, "inquiry_log", None) or []):
        if e.get("action") != "ask" or e.get("symptom") is None:
            continue
        rnd = int(e.get("round", 0) or 0)
        ans = e.get("answer")
        ans_txt = "存在" if ans is True else ("不存在" if ans is False else "无法提供")
        out.append(make_claim(ids.next(), f"追问第 {rnd} 轮：用户确认征兆「{e.get('symptom')}」{ans_txt}。",
                              "observation",
                              [make_evidence("user", f"user:{rnd}", f"{e.get('symptom')}={ans_txt}")]))
    return out


def _timeseries_claims(state, ids: _IdGen) -> List[Dict[str, Any]]:
    call = _latest_ok_call(state, "timeseries_anomaly")
    if not call:
        return []
    res = call.get("result") or {}
    if res.get("status") != "ok":
        return []
    ref = tool_ref(call)
    idxs = res.get("anomaly_indices") or []
    out = [make_claim(
        ids.next(),
        f"时序信号均值 {_fmt_num(res.get('mean'))}，标准差 {_fmt_num(res.get('std'))}，3σ 检出 {len(idxs)} 个异常点。",
        "observation",
        [make_evidence("tool", ref, {"mean": res.get("mean"), "std": res.get("std"),
                                     "anomaly_indices": idxs[:20], "n": res.get("n")}, tool="timeseries_anomaly")])]
    if idxs:
        out.append(make_claim(ids.next(), f"存在 {len(idxs)} 个超 3σ 的异常点，可能对应短时冲击或工况突变，建议核对对应时刻负载与操作记录。",
                              "inference",
                              [make_evidence("tool", ref, {"anomaly_indices": idxs[:20]}, tool="timeseries_anomaly")]))
    return out


def _forecast_claims(state, ids: _IdGen) -> List[Dict[str, Any]]:
    call = _latest_ok_call(state, "ett_forecast")
    if not call:
        return []
    res = call.get("result") or {}
    if res.get("status") != "ok":
        return []
    ref = tool_ref(call)
    tool = "ett_forecast"
    fc = res.get("forecast") or {}
    hist = res.get("history_summary") or {}
    anom = res.get("anomalies") or {}
    ds, horizon, lookback = res.get("dataset"), res.get("horizon"), res.get("lookback")
    out = [make_claim(
        ids.next(),
        f"油温预测基于数据集 {ds}（{res.get('freq', '')} 采样），回看 {lookback} 点、预测 {horizon} 步；"
        f"历史油温均值 {_fmt_num(hist.get('ot_mean'))}℃，标准差 {_fmt_num(hist.get('ot_std'))}℃。",
        "observation",
        [make_evidence("tool", ref, {"dataset": ds, "freq": res.get("freq"), "lookback": lookback, "horizon": horizon,
                                     "history_summary": {"ot_mean": hist.get("ot_mean"), "ot_std": hist.get("ot_std")}}, tool=tool)])]
    if fc.get("mean_predicted_ot") is not None:
        slim_fc = {k: fc.get(k) for k in ("mean_predicted_ot", "min_predicted_ot", "max_predicted_ot") if k in fc}
        out.append(make_claim(
            ids.next(),
            f"未来 {horizon} 步油温预测均值 {_fmt_num(fc.get('mean_predicted_ot'))}℃，"
            f"范围 {_fmt_num(fc.get('min_predicted_ot'))}℃ ~ {_fmt_num(fc.get('max_predicted_ot'))}℃。",
            "inference",
            [make_evidence("tool", ref, {"horizon": horizon, "forecast": slim_fc}, tool=tool)]))
    cnt = anom.get("count")
    if cnt is not None:
        text = (f"历史窗口 3σ 检出 {cnt} 个油温异常点，建议关注油温突变风险并核查冷却系统。" if cnt
                else "历史窗口未检出油温 3σ 异常，油温运行在正常区间。")
        out.append(make_claim(ids.next(), text, "recommendation" if cnt else "observation",
                              [make_evidence("tool", ref, {"anomalies": {"count": cnt}}, tool=tool)]))
    return out


def _kg_claims(state, ids: _IdGen) -> List[Dict[str, Any]]:
    call = _latest_ok_call(state, "kg_search")
    if not call:
        return []
    res = call.get("result") or {}
    if res.get("status") != "ok":
        return []
    out: List[Dict[str, Any]] = []
    for p in (res.get("paths") or [])[:3]:
        if not isinstance(p, dict) or not p.get("path"):
            continue
        path_txt = p["path"] if isinstance(p["path"], str) else " → ".join(map(str, p["path"]))
        ev_txt = (p.get("evidence") or "").strip() or path_txt
        out.append(make_claim(
            ids.next(),
            f"图谱关系链「{path_txt}」（支持文献 {p.get('support_count', '?')}，置信度 {p.get('confidence', '?')}）"
            "可用于解释当前征兆的因果机理，需与实测数据交叉验证。",
            "inference",
            [make_evidence("kg", kg_ref(p), ev_txt)]))
    return out


def _kb_claims(state, ids: _IdGen) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate((state.retrieved_knowledge or [])[:3], start=1):
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        text = item.get("snippet") or item.get("text") or ""
        if isinstance(item.get("child"), dict) and item["child"].get("text"):
            text = item["child"]["text"]
        text = str(text).strip()
        if not text:
            continue
        head = meta.get("title") or meta.get("doc_name") or item.get("path") or "知识库片段"
        out.append(make_claim(
            ids.next(),
            f"参考知识库《{head}》：{text[:80]}{'…' if len(text) > 80 else ''}",
            "recommendation",
            [make_evidence("kb", kb_ref(item, i), text)]))
    return out


def build_rule_claims(state) -> List[Dict[str, Any]]:
    """按工具结果规则拼装 claims；无任何工具结果时产出一条引用 user:0 的信息不足声明。"""
    ids = _IdGen()
    claims: List[Dict[str, Any]] = []
    claims += _attribution_claims(state, ids)
    claims += _inquiry_claims(state, ids)
    claims += _timeseries_claims(state, ids)
    claims += _forecast_claims(state, ids)
    claims += _kg_claims(state, ids)
    claims += _kb_claims(state, ids)
    if not claims:
        pend = getattr(state, "pending_questions", None) or []
        if pend:
            need = "、".join(str(q.get("symptom") or q.get("question") or "") for q in pend[:3])
            text = f"当前信息不足以给出诊断结论，需要用户补充：{need}。"
        else:
            text = "当前未获得任何工具结果或知识片段，无法给出有依据的诊断结论，建议补充 DGA 数据或设备信息。"
        claims.append(make_claim(ids.next(), text, "observation",
                                 [make_evidence("user", "user:0", state.user_query or "")]))
    return claims


# ──────────────────────────────────────────────────────────────
# 渲染
# ──────────────────────────────────────────────────────────────
_TYPE_LABEL = {"observation": "观测", "inference": "推断", "recommendation": "建议", "safety": "安全"}


def render_claims_markdown(claims: List[Dict[str, Any]]) -> str:
    if not claims:
        return "（无声明）"
    lines = ["【声明与证据清单】"]
    for c in claims:
        refs = "，".join(e.get("ref", "") for e in c.get("evidence", []))
        lines.append(f"- [{c.get('id')}][{_TYPE_LABEL.get(c.get('type'), c.get('type'))}] {c.get('text')}  ⟨证据：{refs}⟩")
    return "\n".join(lines)


def claims_to_json(claims: List[Dict[str, Any]]) -> str:
    return json.dumps({"claims": claims}, ensure_ascii=False, indent=2)
