"""
C-2 声明级约束核查器（ClaimChecker）。

两层核查：
- 确定性层（无 LLM）：对 state.draft_claims 逐条核对
    DATA          声明中的数值（气体浓度 / 概率 / 预测值 / horizon / 三比值编码）与被引用工具结果逐项比对，容差可配
    EVIDENCE      ref 在本轮证据目录中存在；span 是被引用证据的子串（工具证据按 JSON 片段解析后逐键比对）
    APPLICABILITY 声明中的数据集名 / 设备号 / 时间窗与工具入参一致
    SAFETY        涉及停电 / 吊罩 / 更换 / 退出运行的 safety / recommendation 声明必须存在 observation 支撑，
                  且不能与检测结果矛盾（主要故障低风险 / 高危故障却说无需处理）
- 语义层（LLM，可选）：对通过确定性层的 inference 声明做 NLI 判定 entail | contradict | unsupported，
  提示词只给该声明与其引用的证据片段。LLM 不可用时跳过（verdict 保持确定性层结论）。

输出 ClaimCheckResult：claim_verdicts / unsupported_ratio / violation_counts / verdict(PASS | REVISION | ABSTAIN)。
判定规则（PLAN C-2）：任一 SAFETY 或 DATA 违反 → REVISION；unsupported_ratio > 0.3 → REVISION；
两轮修订仍不通过（iteration ≥ abstain_after）→ ABSTAIN。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agents.claims import CONSTRAINT_TYPES, evidence_catalog
from src.utils.logging import get_logger

logger = get_logger(__name__)

NLI_LABELS = ("entail", "contradict", "unsupported")

# 高风险处置关键词（SAFETY 约束触发）
HIGH_RISK_ACTIONS = ("停电", "吊罩", "更换", "退出运行", "停运", "解体", "返厂", "大修")
# 「无需处理」类断言（与高危检测结果矛盾时触发 SAFETY）
NO_ACTION_PHRASES = ("无需处理", "可继续运行", "无需停电", "不需要处理", "正常运行，无需", "无需采取")
# 「低风险」故障：说停电吊罩则视为与检测矛盾（用于 SAFETY）
LOW_RISK_FAULT_NAMES = ("正常", "无故障")

# 工具入参中承载适用性的键
_APPLIC_KEYS = ("dataset", "device_id", "transformer_id", "equipment_id", "horizon", "lookback", "freq")
_DATASET_PATTERN = re.compile(r"ETT[hm][12]", re.IGNORECASE)
_DEVICE_PATTERN = re.compile(r"#\s?(\d{1,3})\s*(?:号)?主变|(\d{1,3})\s*号主变|#\s?(\d{1,3})\s*变压器")
_GAS_KEYS = ("H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "TDCG")
_GAS_PATTERN = re.compile(r"(H2|CH4|C2H2|C2H4|C2H6|CO2|CO|TDCG)\s*[=＝:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(ppm|μL/L|uL/L|µL/L)?", re.IGNORECASE)
_PCT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_TEMP_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:℃|°C)")
_HORIZON_PATTERN = re.compile(r"(?:预测|未来)\s*([0-9]+)\s*步")
_LOOKBACK_PATTERN = re.compile(r"回看\s*([0-9]+)\s*点")
_ENTROPY_PATTERN = re.compile(r"后验熵\s*([0-9]+(?:\.[0-9]+)?)\s*bit")
_RATIO_CODE_PATTERN = re.compile(r"三比值.*?([0-2])\s*-\s*([0-2])\s*-\s*([0-2])")
_ANOMALY_PATTERN = re.compile(r"检出\s*([0-9]+)\s*个")


@dataclass
class ClaimVerdict:
    claim_id: str
    verdict: str                                   # pass | violated | unsupported | contradict
    violated_constraints: List[str] = field(default_factory=list)
    detail: List[str] = field(default_factory=list)
    nli: Optional[str] = None                      # entail | contradict | unsupported | None

    def to_dict(self) -> Dict[str, Any]:
        return {"claim_id": self.claim_id, "verdict": self.verdict,
                "violated_constraints": list(self.violated_constraints), "detail": list(self.detail), "nli": self.nli}


@dataclass
class ClaimCheckResult:
    verdict: str                                   # PASS | REVISION | ABSTAIN
    claim_verdicts: List[ClaimVerdict]
    unsupported_ratio: float
    violation_counts: Dict[str, int]
    n_claims: int
    n_checked_llm: int = 0
    missing_evidence: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict, "n_claims": self.n_claims,
            "unsupported_ratio": round(self.unsupported_ratio, 4),
            "violation_counts": dict(self.violation_counts),
            "claim_verdicts": [v.to_dict() for v in self.claim_verdicts],
            "n_checked_llm": self.n_checked_llm,
            "missing_evidence": list(self.missing_evidence), "reasons": list(self.reasons),
        }

    def feedback_lines(self) -> List[str]:
        out: List[str] = []
        for v in self.claim_verdicts:
            if v.verdict != "pass":
                out.append(f"[{v.claim_id}] {'/'.join(v.violated_constraints) or v.verdict}：{'；'.join(v.detail) or '证据不足'}")
        return out


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
def _num_close(a: Any, b: Any, rel: float, abs_tol: float) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) <= max(abs_tol, rel * max(abs(fa), abs(fb)))


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """把嵌套 dict/list 展平为 {路径: 叶值}。"""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _try_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def _leaf_matches(span_leaves: Dict[str, Any], source_leaves: Dict[str, Any], rel: float, abs_tol: float) -> List[str]:
    """span 的每个叶值必须能在源对象中找到同名末级键且值一致。返回不一致项。"""
    bad: List[str] = []
    src_by_tail: Dict[str, List[Any]] = {}
    for k, v in source_leaves.items():
        tail = re.sub(r"\[\d+\]", "", k).split(".")[-1]
        src_by_tail.setdefault(tail, []).append(v)
    for k, v in span_leaves.items():
        tail = re.sub(r"\[\d+\]", "", k).split(".")[-1]
        cands = src_by_tail.get(tail)
        if cands is None:
            # 允许 span 直接是源的一个子路径
            if k in source_leaves:
                cands = [source_leaves[k]]
            else:
                bad.append(f"{k} 不在证据中")
                continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if not any(_num_close(v, c, rel, abs_tol) for c in cands if isinstance(c, (int, float))):
                bad.append(f"{k}={v} 与证据不一致（证据 {cands[:3]}）")
        else:
            if not any(str(v) == str(c) for c in cands):
                bad.append(f"{k}={v!s} 与证据不一致")
    return bad


# ──────────────────────────────────────────────────────────────
# 核查器
# ──────────────────────────────────────────────────────────────
class ClaimChecker:
    def __init__(self, *, rel_tol: float = 0.02, abs_tol: float = 0.05, pct_abs_tol: float = 0.15,
                 unsupported_threshold: float = 0.3, abstain_after: int = 3, llm=None, nli_max_claims: int = 12) -> None:
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.pct_abs_tol = pct_abs_tol           # 百分比声明允许 ±0.15 个百分点（四舍五入误差）
        self.unsupported_threshold = unsupported_threshold
        self.abstain_after = abstain_after       # iteration ≥ 此值且仍不通过 → ABSTAIN（两轮修订 = 第 3 次评估）
        self.llm = llm
        self.nli_max_claims = nli_max_claims

    # ── 入口 ────────────────────────────────────────────────
    def check(self, state, *, use_llm: Optional[bool] = None) -> ClaimCheckResult:
        claims = list(getattr(state, "draft_claims", None) or [])
        cat = evidence_catalog(state)
        by_ref = self._index_evidence(state, cat)
        obs_texts = [c.get("text", "") for c in claims if c.get("type") == "observation"]
        attr = self._latest_result(state, "fault_attribution")

        verdicts: List[ClaimVerdict] = []
        counts = {k: 0 for k in CONSTRAINT_TYPES}
        for c in claims:
            v = self._check_one(c, by_ref, cat["refs"], obs_texts, attr)
            for k in v.violated_constraints:
                counts[k] += 1
            verdicts.append(v)

        # 语义层
        n_llm = 0
        want_llm = (self.llm is not None and getattr(self.llm, "enabled", False)) if use_llm is None else bool(use_llm)
        if want_llm and self.llm is not None:
            n_llm = self._semantic_layer(claims, verdicts, by_ref)

        n = len(claims)
        n_unsupported = sum(1 for v in verdicts if v.verdict in ("unsupported", "contradict") or "EVIDENCE" in v.violated_constraints)
        ratio = (n_unsupported / n) if n else 1.0
        reasons: List[str] = []
        if n == 0:
            reasons.append("草案未产出任何声明")
        if counts["SAFETY"]:
            reasons.append(f"SAFETY 违反 {counts['SAFETY']} 条")
        if counts["DATA"]:
            reasons.append(f"DATA 违反 {counts['DATA']} 条")
        if ratio > self.unsupported_threshold:
            reasons.append(f"无依据 / 矛盾声明占比 {ratio:.0%} > {self.unsupported_threshold:.0%}")
        if any(v.nli == "contradict" for v in verdicts):
            reasons.append("语义层判定存在 contradict 声明")

        if not reasons:
            verdict = "PASS"
        elif getattr(state, "iteration", 1) >= self.abstain_after:
            verdict = "ABSTAIN"
        else:
            verdict = "REVISION"

        missing = self._missing_evidence(claims, verdicts)
        return ClaimCheckResult(verdict=verdict, claim_verdicts=verdicts, unsupported_ratio=ratio,
                                violation_counts=counts, n_claims=n, n_checked_llm=n_llm,
                                missing_evidence=missing, reasons=reasons)

    # ── 证据索引 ────────────────────────────────────────────
    @staticmethod
    def _latest_result(state, tool: str) -> Optional[Dict[str, Any]]:
        for c in reversed(state.tool_calls or []):
            if c.get("tool") == tool and c.get("success") and isinstance(c.get("result"), dict):
                return c["result"]
        return None

    @staticmethod
    def _index_evidence(state, cat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """ref → {kind, text, obj, args}"""
        idx: Dict[str, Dict[str, Any]] = {}
        calls_by_idx = {c.get("call_index"): c for c in (state.tool_calls or [])}
        calls_by_id = {c.get("call_id"): c for c in (state.tool_calls or []) if c.get("call_id")}
        for t in cat["tool"]:
            call = calls_by_idx.get(t["call_index"]) or calls_by_id.get(t["call_id"])
            if not call:
                continue
            obj = {"result": call.get("result"), "args": call.get("raw_arguments") or call.get("args") or {}}
            # 展平时同时包含 args 与 result 叶子（dga_data 在 args 中）
            idx[t["ref"]] = {"kind": "tool", "tool": t["tool"], "obj": obj, "args": obj["args"],
                             "result": obj["result"], "text": json.dumps(obj, ensure_ascii=False, default=str)}
        for k in cat["kb"]:
            idx[k["ref"]] = {"kind": "kb", "text": k["text"], "obj": None, "args": {}}
        for g in cat["kg"]:
            path_txt = g["path"] if isinstance(g["path"], str) else " → ".join(map(str, g["path"] or []))
            idx[g["ref"]] = {"kind": "kg", "text": (g.get("evidence") or "") + "\n" + path_txt, "obj": None, "args": {}}
        for u in cat["user"]:
            idx[u["ref"]] = {"kind": "user", "text": u["text"], "obj": None, "args": {}}
        return idx

    # ── 单条核查 ────────────────────────────────────────────
    def _check_one(self, claim: Dict[str, Any], by_ref: Dict[str, Dict[str, Any]], known_refs: set,
                   obs_texts: List[str], attr: Optional[Dict[str, Any]]) -> ClaimVerdict:
        cid = str(claim.get("id"))
        text = str(claim.get("text") or "")
        ctype = claim.get("type")
        v = ClaimVerdict(claim_id=cid, verdict="pass")
        evs = claim.get("evidence") or []
        if not evs:
            v.verdict = "unsupported"
            v.violated_constraints.append("EVIDENCE")
            v.detail.append("无证据引用")
            return v

        valid_refs: List[Dict[str, Any]] = []
        valid_spans: List[str] = []
        for e in evs:
            ref = e.get("ref")
            span = str(e.get("span") or "")
            if ref not in known_refs or ref not in by_ref:
                self._add(v, "EVIDENCE", f"引用 {ref} 不存在于本轮证据")
                continue
            src = by_ref[ref]
            if not self._span_supported(span, src):
                self._add(v, "EVIDENCE", f"引用 {ref} 的片段与证据内容不符")
                continue
            valid_refs.append(src)
            valid_spans.append(span)

        # DATA / APPLICABILITY 仅对工具证据有意义
        tool_srcs = [s for s in valid_refs if s["kind"] == "tool"]
        if tool_srcs:
            for d in self._check_data(text, tool_srcs):
                self._add(v, "DATA", d)
            for d in self._check_applicability(text, tool_srcs):
                self._add(v, "APPLICABILITY", d)
        elif ctype == "observation" and re.search(r"本次|检测(数据|结果)", text) and any(s["kind"] == "kb" for s in valid_refs):
            self._add(v, "EVIDENCE", "以知识库 / 历史案例片段作为本次检测结果的依据")

        # SAFETY
        for d in self._check_safety(text, ctype, valid_refs, valid_spans, obs_texts, attr):
            self._add(v, "SAFETY", d)

        if not valid_refs and v.verdict == "pass":
            v.verdict = "unsupported"
        return v

    @staticmethod
    def _add(v: ClaimVerdict, ctype: str, detail: str) -> None:
        if ctype not in v.violated_constraints:
            v.violated_constraints.append(ctype)
        v.detail.append(detail)
        v.verdict = "violated"

    # ── EVIDENCE ──────────────────────────────────────────
    def _span_supported(self, span: str, src: Dict[str, Any]) -> bool:
        if not span.strip():
            return False
        if src["kind"] != "tool":
            hay = re.sub(r"\s+", "", src["text"])
            needle = re.sub(r"\s+", "", span)
            if needle in hay:
                return True
            # 允许截断（span 末尾带省略号）
            needle = needle.rstrip("…...")
            return len(needle) >= 6 and needle[: min(len(needle), 40)] in hay
        # 工具证据：span 为 JSON 片段 → 逐键比对；否则退化为子串
        obj = _try_json(span)
        if obj is None:
            # 可能被截断（_compact 300 字符），尝试补齐再解析
            for tail in ("}", "}}", "]}", "]}}", "\"}", "\"}}"):
                obj = _try_json(span + tail)
                if obj is not None:
                    break
        if obj is None:
            # 截断且无法补齐：退化为逐个 "key": scalar 对比对（键序无关）
            pairs = re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*("([^"\\]|\\.)*"|-?[0-9]+(?:\.[0-9]+)?|true|false|null)', span)
            if not pairs:
                return re.sub(r"\s+", "", span)[:40] in re.sub(r"\s+", "", src["text"])
            src_leaves = _flatten(src["obj"])
            by_tail: Dict[str, List[Any]] = {}
            for k, v in src_leaves.items():
                by_tail.setdefault(re.sub(r"\[\d+\]", "", k).split(".")[-1], []).append(v)
            for key, raw, _ in pairs:
                cands = by_tail.get(key)
                if cands is None:
                    return False
                val = _try_json(raw)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if not any(isinstance(c, (int, float)) and _num_close(val, c, self.rel_tol, self.abs_tol) for c in cands):
                        return False
                elif not any(str(val) == str(c) for c in cands):
                    return False
            return True
        span_leaves = _flatten(obj)
        src_leaves = _flatten(src["obj"])
        bad = _leaf_matches(span_leaves, src_leaves, self.rel_tol, self.abs_tol)
        return not bad

    # ── DATA ──────────────────────────────────────────────
    @staticmethod
    def _quoted(text: str, m: "re.Match", str_leaves: List[str], ctx: int = 6) -> bool:
        """声明中该数字所在的短语是否原样出自工具的文本型字段（如 interpretation），是则视为转述而非数值断言。"""
        frag = text[max(0, m.start() - ctx): m.end() + ctx]
        frag = re.sub(r"\s+", "", frag)
        core = re.sub(r"\s+", "", m.group(0))
        for s in str_leaves:
            s2 = re.sub(r"\s+", "", s)
            if core in s2 and (frag in s2 or any(text[max(0, m.start() - k): m.end()].replace(" ", "") in s2 for k in (2, 3, 4))):
                return True
        return False

    def _check_data(self, text: str, tool_srcs: List[Dict[str, Any]]) -> List[str]:
        bad: List[str] = []
        leaves: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []
        for s in tool_srcs:
            leaves.update(_flatten(s["obj"]))
            if isinstance(s.get("result"), dict):
                results.append(s["result"])
        nums = [v for v in leaves.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        str_leaves = [v for v in leaves.values() if isinstance(v, str) and len(v) >= 4]

        # 气体浓度
        for m in _GAS_PATTERN.finditer(text):
            gas, val = m.group(1).upper(), float(m.group(2))
            if self._quoted(text, m, str_leaves):
                continue
            cands = [v for k, v in leaves.items() if k.split(".")[-1].upper() == gas and isinstance(v, (int, float))]
            if cands and not any(_num_close(val, c, self.rel_tol, self.abs_tol) for c in cands):
                bad.append(f"{gas}={val} 与工具值 {cands[:2]} 不一致")

        # 百分比（概率）
        probs = [v for k, v in leaves.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                 and ("prob" in k.lower() or "gap" in k.lower() or k.endswith("confidence"))]
        pct_strings = [str(v) for k, v in leaves.items() if isinstance(v, str) and v.endswith("%")]
        for m in _PCT_PATTERN.finditer(text):
            val = float(m.group(1))
            if self._quoted(text, m, str_leaves):
                continue
            ok = any(abs(val - float(p) * 100) <= self.pct_abs_tol for p in probs) or \
                any(abs(val - float(ps.rstrip("%"))) <= self.pct_abs_tol for ps in pct_strings if ps.rstrip("%").replace(".", "", 1).isdigit()) or \
                any(_num_close(val, n, self.rel_tol, self.abs_tol) for n in nums)   # 例如「注意值 5%」类阈值不在工具中时不判
            if not ok and probs:
                bad.append(f"百分比 {val}% 与工具概率 {[round(p * 100, 1) for p in probs[:4]]} 不一致")

        # 温度（油温预测 / 历史均值：仅与 ett_forecast 的 *ot* 字段比对，不与归因引擎的温度缩放参数混淆）
        temps = [v for k, v in leaves.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                 and re.search(r"(^|_)ot($|_)|predicted_ot|oil_temp", k.lower().split(".")[-1])]
        for m in _TEMP_PATTERN.finditer(text):
            val = float(m.group(1))
            if self._quoted(text, m, str_leaves):
                continue
            if temps and not any(_num_close(val, t, self.rel_tol, self.abs_tol) for t in temps):
                bad.append(f"温度 {val}℃ 与工具值 {temps[:4]} 不一致")

        # horizon / lookback
        for pat, key, label in ((_HORIZON_PATTERN, "horizon", "预测步数"), (_LOOKBACK_PATTERN, "lookback", "回看点数")):
            for m in pat.finditer(text):
                val = int(m.group(1))
                cands = [v for k, v in leaves.items() if k.split(".")[-1] == key and isinstance(v, (int, float))]
                if cands and not any(int(c) == val for c in cands):
                    bad.append(f"{label} {val} 与工具值 {cands[:2]} 不一致")

        # 后验熵
        for m in _ENTROPY_PATTERN.finditer(text):
            val = float(m.group(1))
            cands = [v for k, v in leaves.items() if k.endswith("entropy_bits")]
            if cands and not any(abs(val - float(c)) <= 0.011 for c in cands):
                bad.append(f"后验熵 {val} 与工具值 {cands[:1]} 不一致")

        # 三比值编码
        m = _RATIO_CODE_PATTERN.search(text)
        if m:
            codes = None
            for r in results:
                codes = ((r.get("dga_analysis") or {}).get("ratio_codes")) or codes
            if codes:
                expect = (int(codes.get("code_C2H2_C2H4", -1)), int(codes.get("code_CH4_H2", -1)), int(codes.get("code_C2H4_C2H6", -1)))
                got = tuple(int(x) for x in m.groups())
                if got != expect:
                    bad.append(f"三比值编码 {'-'.join(map(str, got))} 与工具 {'-'.join(map(str, expect))} 不一致")

        # 异常点数
        for m in _ANOMALY_PATTERN.finditer(text):
            val = int(m.group(1))
            cands = []
            for r in results:
                if isinstance(r.get("anomaly_indices"), list):
                    cands.append(len(r["anomaly_indices"]))
                if isinstance(r.get("anomalies"), dict) and r["anomalies"].get("count") is not None:
                    cands.append(int(r["anomalies"]["count"]))
            if cands and val not in cands:
                bad.append(f"异常点数 {val} 与工具 {cands} 不一致")

        # 主要故障名互换（声明写了某故障 + 百分比，但该百分比对应另一个故障）
        for r in results:
            ranking = r.get("fault_ranking") or []
            if not ranking:
                continue
            for item in ranking:
                name = str(item.get("fault_name") or "")
                if name and name in text:
                    p = item.get("probability")
                    near = re.search(re.escape(name) + r"[^%]{0,12}?([0-9]+(?:\.[0-9]+)?)\s*%", text)
                    if near and p is not None and abs(float(near.group(1)) - float(p) * 100) > self.pct_abs_tol:
                        bad.append(f"「{name}」概率 {near.group(1)}% 与工具 {float(p) * 100:.1f}% 不一致")
        return bad

    # ── APPLICABILITY ─────────────────────────────────────
    def _check_applicability(self, text: str, tool_srcs: List[Dict[str, Any]]) -> List[str]:
        bad: List[str] = []
        args_all: Dict[str, Any] = {}
        for s in tool_srcs:
            for k, v in (s.get("args") or {}).items():
                args_all.setdefault(k, v)
            r = s.get("result")
            if isinstance(r, dict):
                for k in ("dataset", "freq"):
                    if k in r:
                        args_all.setdefault(k, r[k])
        # 数据集
        ds_claim = {m.group(0).upper() for m in _DATASET_PATTERN.finditer(text)}
        ds_tool = str(args_all.get("dataset") or "").upper()
        if ds_claim and ds_tool and ds_claim != {ds_tool}:
            bad.append(f"数据集 {sorted(ds_claim)} 与工具入参 {ds_tool} 不一致")
        # 设备号
        dev_tool = None
        for k in ("device_id", "transformer_id", "equipment_id"):
            if args_all.get(k):
                dev_tool = str(args_all[k])
                break
        if dev_tool:
            m = _DEVICE_PATTERN.search(text)
            if m:
                num = next(g for g in m.groups() if g)
                if num not in dev_tool:
                    bad.append(f"设备号 #{num} 与工具入参 {dev_tool} 不一致")
        return bad

    # ── SAFETY ────────────────────────────────────────────
    def _check_safety(self, text: str, ctype: Optional[str], valid_refs: List[Dict[str, Any]], valid_spans: List[str],
                      obs_texts: List[str], attr: Optional[Dict[str, Any]]) -> List[str]:
        bad: List[str] = []
        primary = str((attr or {}).get("primary_fault_name") or "")
        prob = (attr or {}).get("primary_probability")
        sev = ""
        for r in ((attr or {}).get("fault_ranking") or [])[:1]:
            sev = str(r.get("severity") or "")
        high_risk_detected = attr is not None and prob is not None and float(prob) >= 0.4 and (
            sev == "高风险" or any(k in primary for k in ("放电", "短路", "套管", "接地", "变形")))

        actions = [k for k in HIGH_RISK_ACTIONS if k in text]
        mentions_high_action = bool(actions) and ctype in ("safety", "recommendation")
        # recommendation 类若高风险词均出自所引用的知识库 / 图谱原文（文献转述，非系统自身处置建议），不触发前提检查
        if mentions_high_action and ctype == "recommendation":
            lit_spans = [sp for sp, s in zip(valid_spans, valid_refs) if s["kind"] in ("kb", "kg")]
            if lit_spans and all(any(a in sp for sp in lit_spans) for a in actions):
                mentions_high_action = False
        if mentions_high_action:
            # 前提：必须引用 tool 证据，或存在 observation 声明支撑
            has_tool = any(s["kind"] == "tool" for s in valid_refs)
            if not has_tool and not obs_texts:
                bad.append("高风险处置建议缺少检测结果（observation）支撑")
            elif not has_tool:
                bad.append("高风险处置建议未引用工具检测结果作为前提")
            # 矛盾：检测结果为正常 / 低风险却建议停电吊罩
            if attr is not None and prob is not None:
                if any(k in primary for k in LOW_RISK_FAULT_NAMES) and float(prob) >= 0.5:
                    bad.append(f"建议高风险处置，但归因结果为「{primary}」（{float(prob) * 100:.0f}%）")
                elif sev == "低风险" and not high_risk_detected and "建议" in text and "立即" in text:
                    bad.append(f"归因为低风险（{primary} {float(prob) * 100:.0f}%），「立即」高风险处置缺乏前提")

        if any(p in text for p in NO_ACTION_PHRASES) and high_risk_detected:
            bad.append(f"声明「无需处理」但归因显示高危故障「{primary}」概率 {float(prob) * 100:.0f}%")
        return bad

    # ── 语义层 ────────────────────────────────────────────
    def _semantic_layer(self, claims: List[Dict[str, Any]], verdicts: List[ClaimVerdict],
                        by_ref: Dict[str, Dict[str, Any]]) -> int:
        n = 0
        for c, v in zip(claims, verdicts):
            if c.get("type") != "inference" or v.verdict != "pass":
                continue
            if n >= self.nli_max_claims:
                break
            spans = []
            for e in c.get("evidence") or []:
                spans.append(f"[{e.get('ref')}] {e.get('span')}")
            label = self._nli(c.get("text", ""), spans)
            n += 1
            v.nli = label
            if label == "contradict":
                v.verdict = "contradict"
                v.detail.append("语义层：声明与证据矛盾")
            elif label == "unsupported":
                v.verdict = "unsupported"
                v.detail.append("语义层：证据不足以支持该声明")
        return n

    def _nli(self, claim_text: str, spans: List[str]) -> str:
        prompt = (
            "判断下列声明是否被给出的证据片段支持。只输出一个词：entail（证据支持）、contradict（证据与声明矛盾）"
            "或 unsupported（证据不足以支持）。\n\n【声明】\n" + claim_text + "\n\n【证据片段】\n" + "\n".join(spans)
        )
        try:
            choice = self.llm.chat([{"role": "system", "content": "你是严格的证据核查员。"},
                                    {"role": "user", "content": prompt}])
            content = (choice.get("content", "") if isinstance(choice, dict) else "").strip().lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLI LLM call failed: %s", exc)
            return "entail"
        for lab in NLI_LABELS:
            if lab in content:
                return lab
        return "unsupported"

    # ── 补证建议 ──────────────────────────────────────────
    @staticmethod
    def _missing_evidence(claims: List[Dict[str, Any]], verdicts: List[ClaimVerdict]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c, v in zip(claims, verdicts):
            if v.verdict not in ("unsupported",) and "EVIDENCE" not in v.violated_constraints:
                continue
            text = str(c.get("text") or "")
            tool, query = None, text[:60]
            if re.search(r"ppm|μL/L|气体|乙炔|氢气|三比值|故障概率|后验", text):
                tool = "fault_attribution"
            elif re.search(r"油温|预测|℃", text):
                tool = "ett_forecast"
            elif re.search(r"关系|机理|产生|导致|图谱", text):
                tool = "kg_search"
            elif c.get("type") in ("recommendation", "safety"):
                tool = "rag_search"
            out.append({"claim_id": c.get("id"), "suggested_tool": tool, "suggested_query": query, "reason": "；".join(v.detail) or v.verdict})
        return out
