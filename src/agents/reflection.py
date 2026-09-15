"""
反思模块（P3）：对 Retriever 返回的知识片段做质量评分 + 上下文补召回。

设计目标（与 PLAN P3-1 / P3-2 对齐）：
- 评分标准：0 无关 / 1 弱相关 / 2 部分回答 / 3 直接回答（论文表 2.2）。
- 评分 < 2 的块丢弃；全部丢弃时触发一次查询改写重检索。
- 对保留块做上下文补召回（同章节相邻块），补召回后重新评分，最多迭代 2 轮。
- 记录每次反思的输入、决策、耗时，写入 state.reasoning_trace 与 state.reflection_log。

评分器是可插拔的：
- `LLMScorer`     ：调用 LLM（需要接口，默认关闭）。
- `LexicalScorer` ：离线词法评分（jieba 分词 + 领域词覆盖 + 章节标题匹配），作为无接口时的基线，
                    同时也是 P3-3 中与人工评分对比的"弱基线"。
接口不可用时自动回退到 LexicalScorer，并在日志中标注 scorer 名称，便于评测时区分。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.graph.state import AgentState
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCORE_RUBRIC = {
    0: "无关：文本块与查询主题无关",
    1: "弱相关：同属变压器故障主题，但不涉及查询要点",
    2: "部分回答：涉及查询要点，但只能回答一部分或需要结合其他块",
    3: "直接回答：文本块本身即可回答查询的核心问题",
}

_DEFAULT_MIN_KEEP_SCORE = 2
_DEFAULT_MAX_ROUNDS = 2

try:
    from scripts.kb.build_index import tokenize as _tokenize  # 复用知识库分词（含领域词典）
except Exception:  # noqa: BLE001
    _tokenize = None

_STOP = {"变压器", "电力变压器", "故障", "问题", "情况", "如何", "什么", "哪些", "怎么", "是否", "进行", "可能", "一般"}


def _tok(text: str) -> List[str]:
    if _tokenize is not None:
        try:
            return [t for t in _tokenize(text) if t not in _STOP]
        except Exception:  # noqa: BLE001
            pass
    return [t for t in re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z0-9]{2,}", text.lower()) if t not in _STOP]


class Scorer:
    name = "base"

    def score(self, query: str, chunk_text: str, section_title: str = "") -> int:  # pragma: no cover
        raise NotImplementedError

    def score_batch(self, query: str, items: Sequence[Tuple[str, str]]) -> List[int]:
        return [self.score(query, t, s) for t, s in items]


class LexicalScorer(Scorer):
    """
    离线词法评分：
    - 查询关键词按 IDF 加权后在块正文中的覆盖率 cov（IDF 取自 BM25 索引，缺失时退化为 1），
    - 章节标题命中加分，
    - 覆盖率阈值映射到 0-3。
    校准（retrieval_seed 随机 400 条，2026-09-15）：金标块得 3 分占 97%，随机块得 0 分占 79%；
    对 BM25 检回的非金标块约 11% 被判 <2（其余多为 2-3 分），因此它只能剔除明显无关块，
    正式实验应使用 LLMScorer 并在 D9 上校验一致率。
    """

    name = "lexical"

    def __init__(self, t1: float = 0.2, t2: float = 0.5, t3: float = 0.8, idf: Optional[Dict[str, float]] = None) -> None:
        self.t1, self.t2, self.t3 = t1, t2, t3
        self._idf = idf
        self._idf_loaded = idf is not None

    def _load_idf(self) -> Dict[str, float]:
        if self._idf_loaded:
            return self._idf or {}
        self._idf_loaded = True
        try:
            from src.tools.local_kb import get_local_kb
            bm25 = get_local_kb().bm25
            self._idf = dict(getattr(bm25, "idf", {}) or {})
        except Exception:  # noqa: BLE001
            self._idf = {}
        return self._idf

    def score(self, query: str, chunk_text: str, section_title: str = "") -> int:
        q = list(dict.fromkeys(_tok(query)))
        if not q:
            return 1
        idf = self._load_idf()
        w = {t: max(0.1, float(idf.get(t, 1.0))) for t in q}
        total = sum(w.values()) or 1.0
        body = set(_tok(chunk_text))
        cov = sum(w[t] for t in q if t in body) / total
        if section_title and set(q) & set(_tok(section_title)):
            cov = min(1.0, cov + 0.15)
        if cov >= self.t3:
            return 3
        if cov >= self.t2:
            return 2
        if cov >= self.t1:
            return 1
        return 0


class LLMScorer(Scorer):
    """LLM 评分器：一次调用对多个块打分，返回 JSON 数组。接口不可用时抛异常，由上层回退。"""

    name = "llm"

    _SYSTEM = (
        "你是电力变压器领域的检索质量评审员。给定用户查询与若干文本块，按以下标准为每个块打 0-3 分：\n"
        "0 无关；1 弱相关（同主题但不涉及查询要点）；2 部分回答；3 直接回答。\n"
        "只输出 JSON 数组，如 [3,1,0]，数组长度与块数一致，不要输出其他内容。"
    )

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    def score_batch(self, query: str, items: Sequence[Tuple[str, str]]) -> List[int]:
        blocks = []
        for i, (text, sec) in enumerate(items, 1):
            blocks.append(f"[块{i}] 章节：{sec or '（无）'}\n{text[:600]}")
        user = f"【查询】{query}\n\n" + "\n\n".join(blocks)
        choice = self.llm.chat([{"role": "system", "content": self._SYSTEM}, {"role": "user", "content": user}])
        content = choice.get("content", "") if isinstance(choice, dict) else str(choice)
        m = re.search(r"\[[\d,\s]*\]", content)
        if not m:
            raise ValueError(f"LLM 评分输出无法解析：{content[:100]}")
        arr = json.loads(m.group(0))
        if len(arr) != len(items):
            raise ValueError(f"LLM 评分数量不匹配：期望 {len(items)}，得到 {len(arr)}")
        return [max(0, min(3, int(x))) for x in arr]

    def score(self, query: str, chunk_text: str, section_title: str = "") -> int:
        return self.score_batch(query, [(chunk_text, section_title)])[0]


def _section_title(item: Dict[str, Any]) -> str:
    md = item.get("metadata") or {}
    sp = md.get("section_path") or []
    if isinstance(sp, list) and sp:
        return " / ".join(str(x) for x in sp[-2:])
    return str(md.get("title") or "")


class ReflectionModule:
    """
    用法：
        reflector = ReflectionModule(scorer=None, kb=None)   # 自动选择离线评分器与 local_kb
        state = reflector.run(state)                          # 就地更新 state.retrieved_knowledge

    参数：
    - scorer      : Scorer 实例；None 时若提供 llm_client 则用 LLMScorer，否则 LexicalScorer
    - kb          : 提供 neighbors(chunk_id) / format_chunk(c) 的知识库对象；None 时尝试 local_kb 单例
    - research_fn : 查询改写后重检索的函数 f(query) -> List[item]；None 时用 kb.search
    - rewrite_fn  : 查询改写函数 f(query) -> str；None 时用离线规则（去停用词 + 领域词保留）
    """

    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        kb: Any = None,
        llm_client: Any = None,
        research_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        rewrite_fn: Optional[Callable[[str], str]] = None,
        min_keep_score: int = _DEFAULT_MIN_KEEP_SCORE,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
        neighbor_window: int = 1,
        max_total_chunks: int = 8,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.min_keep_score = min_keep_score
        self.max_rounds = max_rounds
        self.neighbor_window = neighbor_window
        self.max_total_chunks = max_total_chunks
        self.lexical = LexicalScorer()
        self.scorer: Scorer = scorer or (LLMScorer(llm_client) if llm_client is not None and getattr(llm_client, "enabled", False) else self.lexical)
        self.kb = kb
        if self.kb is None:
            try:
                from src.tools.local_kb import get_local_kb
                self.kb = get_local_kb()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reflection: local_kb 不可用，补召回关闭：%s", exc)
                self.kb = None
        self.research_fn = research_fn or (self.kb.search if self.kb is not None else None)
        self.rewrite_fn = rewrite_fn or self._rule_rewrite

    # ── 评分（带回退）──
    def _score_items(self, query: str, items: List[Dict[str, Any]]) -> Tuple[List[int], str]:
        pairs = [(it.get("text") or "", _section_title(it)) for it in items]
        if not pairs:
            return [], self.scorer.name
        try:
            return self.scorer.score_batch(query, pairs), self.scorer.name
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reflection: %s 评分失败（%s），回退 lexical", self.scorer.name, exc)
            return self.lexical.score_batch(query, pairs), f"lexical(fallback:{type(exc).__name__})"

    # ── 查询改写（离线规则）──
    @staticmethod
    def _rule_rewrite(query: str) -> str:
        toks = _tok(query)
        return " ".join(dict.fromkeys(toks)) if toks else query

    # ── 补召回 ──
    def _expand(self, kept: List[Dict[str, Any]], seen: set) -> List[Dict[str, Any]]:
        if self.kb is None or not hasattr(self.kb, "neighbors"):
            return []
        added: List[Dict[str, Any]] = []
        for it in kept:
            cid = it.get("chunk_id")
            if not cid:
                continue
            for c in self.kb.neighbors(cid, window=self.neighbor_window, same_section=True):
                if c["chunk_id"] in seen:
                    continue
                seen.add(c["chunk_id"])
                nb = self.kb.format_chunk(c, score=0.0)
                nb["expanded_from"] = cid
                added.append(nb)
        return added

    # ── 入口 ──
    def run(self, state: AgentState) -> AgentState:
        if not self.enabled:
            return state
        query = state.user_query
        items = list(state.retrieved_knowledge or [])
        t0 = time.perf_counter()
        log: Dict[str, Any] = {"scorer": self.scorer.name, "rounds": [], "input_count": len(items),
                               "rewrite_triggered": False}
        if not items:
            log["decision"] = "no_input"
            log["output_count"] = 0
            log["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self._commit(state, items, log)
            return state

        seen = {it.get("chunk_id") for it in items if it.get("chunk_id")}
        kept: List[Dict[str, Any]] = []
        for rnd in range(1, self.max_rounds + 1):
            scores, scorer_name = self._score_items(query, items)
            for it, s in zip(items, scores):
                it["reflection_score"] = s
            kept = [it for it in items if it.get("reflection_score", 0) >= self.min_keep_score]
            dropped = len(items) - len(kept)
            rnd_log = {"round": rnd, "scorer": scorer_name, "scored": len(items), "kept": len(kept),
                       "dropped": dropped, "scores": scores}

            # 全部丢弃 → 改写重检索（仅第一轮触发一次）
            if not kept and rnd == 1 and self.research_fn is not None and not log["rewrite_triggered"]:
                new_q = self.rewrite_fn(query)
                log["rewrite_triggered"] = True
                log["rewritten_query"] = new_q
                try:
                    re_items = self.research_fn(new_q) if new_q != query else []
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Reflection: 重检索失败 %s", exc)
                    re_items = []
                re_items = [x for x in re_items if x.get("chunk_id") not in seen]
                for x in re_items:
                    seen.add(x.get("chunk_id"))
                    x["from_rewrite"] = True
                rnd_log["research_added"] = len(re_items)
                if re_items:
                    rs, _ = self._score_items(query, re_items)
                    for it, s in zip(re_items, rs):
                        it["reflection_score"] = s
                    kept = [it for it in re_items if it["reflection_score"] >= self.min_keep_score]
                    rnd_log["kept_after_research"] = len(kept)
            log["rounds"].append(rnd_log)

            if not kept:
                break
            # 补召回：只在非最后一轮做，让下一轮对新块评分
            if rnd < self.max_rounds and len(kept) < self.max_total_chunks:
                expanded = self._expand(kept, seen)
                rnd_log["expanded"] = len(expanded)
                if not expanded:
                    break
                items = kept + expanded
            else:
                break

        kept.sort(key=lambda it: (-int(it.get("reflection_score", 0)), -float(it.get("score") or 0.0)))
        kept = kept[: self.max_total_chunks]
        log["output_count"] = len(kept)
        log["decision"] = "kept" if kept else "all_dropped"
        log["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self._commit(state, kept, log)
        return state

    @staticmethod
    def _commit(state: AgentState, kept: List[Dict[str, Any]], log: Dict[str, Any]) -> None:
        state.retrieved_knowledge = kept
        if not hasattr(state, "reflection_log") or state.reflection_log is None:
            try:
                state.reflection_log = []
            except Exception:  # noqa: BLE001
                pass
        if isinstance(getattr(state, "reflection_log", None), list):
            state.reflection_log.append(log)
        state.reasoning_trace.append({"agent": "reflection", "type": "reflection", "content": log})
        logger.info("Reflection[%s]: in=%d out=%d rounds=%d rewrite=%s %.1fms",
                    log.get("scorer"), log.get("input_count", 0), log.get("output_count", 0),
                    len(log.get("rounds", [])), log.get("rewrite_triggered"), log.get("latency_ms", 0))
