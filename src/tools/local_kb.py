"""
本地知识库检索器（P1-6：两路检索 + 融合 + 重排接口）。

数据来源：data/kb/（由 scripts/kb/build_units.py → build_chunks.py → build_index.py 生成）

检索通道：
  ① BM25 关键词通道（jieba 分词，领域词典）— 始终可用，不依赖网络
  ② 稠密向量通道 — 可选；需要 data/kb/index/chunk_embeddings.npy（由 scripts/kb/embed_chunks.py 在接口恢复后生成）
  ③ 锚点单元通道 — 对表/图/公式单元做 BM25，命中后映射回所在块，用于「表 N / 图 N / 公式」类精确定位
融合：Reciprocal Rank Fusion（RRF）。
重排：默认关闭；`reranker` 参数可传入任意 callable(query, texts) -> scores。

返回结构与 RAGEngine.search 的 flat 模式兼容：
  [{"score": float, "text": str, "metadata": {...}, "chunk_id": str, "channels": {...}}, ...]
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "data/kb"
IDX = KB / "index"

sys.path.insert(0, str(ROOT / "scripts/kb"))
try:
    from build_index import tokenize  # noqa: E402  复用同一套分词与领域词典
except Exception:  # pragma: no cover
    tokenize = None  # type: ignore


class LocalKBRetriever:
    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 30,
        use_dense: bool = True,
        use_anchor: bool = True,
        rrf_k: int = 60,
        reranker: Optional[Callable[[str, List[str]], List[float]]] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        mode: str = "two_way",
    ) -> None:
        """
        mode（P1-6 消融口径）：
          naive          仅块级 BM25（+ 锚点通道）
          two_way        块级 BM25 + 子块层（映射回父块）+ 摘要层，RRF 融合
          two_way_rerank two_way 基础上启用 reranker（未提供 reranker 时退化为词法重排 LexicalReranker）
        """
        if tokenize is None:
            raise RuntimeError("无法导入 scripts/kb/build_index.tokenize，请先确认 jieba/rank_bm25 已安装")
        if mode not in ("naive", "two_way", "two_way_rerank"):
            raise ValueError(f"未知 mode: {mode}")
        self.mode = mode
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.rrf_weights: Dict[str, float] = {"bm25": 1.0, "dense": 1.0, "anchor": 1.2, "sub": 1.2, "summary": 0.3}  # dev 网格最优（scripts/kb/tune_rrf_weights.py）
        self.reranker = reranker
        if mode == "two_way_rerank" and self.reranker is None:
            self.reranker = lexical_rerank
        self.embed_fn = embed_fn

        self.chunks: List[Dict[str, Any]] = [json.loads(l) for l in open(KB / "chunks.jsonl", encoding="utf-8")]
        self.chunk_pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        with open(IDX / "bm25_chunks.pkl", "rb") as f:
            d = pickle.load(f)
        self.bm25, self.bm25_ids = d["bm25"], d["ids"]

        self.bm25_sub = self.bm25_sum = None
        if mode != "naive":
            if (IDX / "bm25_subchunks.pkl").exists():
                with open(IDX / "bm25_subchunks.pkl", "rb") as f:
                    ds = pickle.load(f)
                self.bm25_sub, self.sub_parent = ds["bm25"], ds["parent"]
            if (IDX / "bm25_summaries.pkl").exists():
                with open(IDX / "bm25_summaries.pkl", "rb") as f:
                    dm = pickle.load(f)
                self.bm25_sum, self.sum_ids = dm["bm25"], dm["ids"]

        self.use_anchor = use_anchor and (IDX / "bm25_units.pkl").exists()
        if self.use_anchor:
            with open(IDX / "bm25_units.pkl", "rb") as f:
                du = pickle.load(f)
            self.bm25_units, self.unit_ids = du["bm25"], du["ids"]
            self.unit_to_chunk: Dict[str, str] = {}
            for c in self.chunks:
                for uid in c["unit_ids"]:
                    self.unit_to_chunk[uid.split("#")[0]] = c["chunk_id"]

        self.dense_matrix = None
        emb_path = IDX / "chunk_embeddings.npy"
        if use_dense and emb_path.exists() and embed_fn is not None:
            import numpy as np
            self.dense_matrix = np.load(emb_path)
            ids_path = IDX / "chunk_embeddings_ids.json"
            self.dense_ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else self.bm25_ids

    # ── 通道 ──
    def _bm25_channel(self, query: str) -> List[str]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.candidate_k]
        return [self.bm25_ids[i] for i in order if scores[i] > 0]

    def _anchor_channel(self, query: str) -> List[str]:
        if not self.use_anchor:
            return []
        if not re.search(r"(表|图|公式|式|table|fig)\s*\d|(表格|图片|曲线|示意图|结构图|流程图|计算公式)", query, re.I):
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25_units.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:10]
        out: List[str] = []
        for i in order:
            if scores[i] <= 0:
                continue
            cid = self.unit_to_chunk.get(self.unit_ids[i])
            if cid and cid not in out:
                out.append(cid)
        return out

    def _dense_channel(self, query: str) -> List[str]:
        if self.dense_matrix is None or self.embed_fn is None:
            return []
        import numpy as np
        q = np.asarray(self.embed_fn([query])[0], dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.dense_matrix @ q
        order = np.argsort(-sims)[: self.candidate_k]
        return [self.dense_ids[i] for i in order]

    def _subchunk_channel(self, query: str) -> List[str]:
        """路一：子文本块召回 top-20，映射回父块（去重保序）。"""
        if self.bm25_sub is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25_sub.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.candidate_k]
        out: List[str] = []
        for i in order:
            if scores[i] <= 0:
                break
            p = self.sub_parent[i]
            if p not in out:
                out.append(p)
        return out

    def _summary_channel(self, query: str) -> List[str]:
        """路二：摘要层召回 top-20，映射回块。"""
        if self.bm25_sum is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25_sum.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.candidate_k]
        return [self.sum_ids[i] for i in order if scores[i] > 0]

    # ── 融合 ──
    def _rrf(self, channels: Dict[str, List[str]]) -> List[tuple]:
        agg: Dict[str, float] = {}
        hit: Dict[str, Dict[str, int]] = {}
        weights = self.rrf_weights
        for name, ids in channels.items():
            for rank, cid in enumerate(ids):
                agg[cid] = agg.get(cid, 0.0) + weights.get(name, 1.0) / (self.rrf_k + rank + 1)
                hit.setdefault(cid, {})[name] = rank + 1
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])
        return [(cid, s, hit[cid]) for cid, s in ranked]

    # ── 入口 ──
    def search(self, query: str, top_k: Optional[int] = None, doc_filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        channels = {
            "bm25": self._bm25_channel(query),
            "dense": self._dense_channel(query),
            "anchor": self._anchor_channel(query),
        }
        if self.mode != "naive":
            channels["sub"] = self._subchunk_channel(query)
            channels["summary"] = self._summary_channel(query)
        fused = self._rrf({n: ids for n, ids in channels.items() if ids})
        # 低价值章节（参考文献 / 致谢 / 作者简介）降权
        low = re.compile(r"(参考文献|致谢|作者简介|References|目录)")
        fused = sorted(
            ((cid, s * (0.4 if any(low.search(p) for p in self.chunks[self.chunk_pos[cid]]["section_path"]) else 1.0), h)
             for cid, s, h in fused),
            key=lambda x: -x[1],
        )
        cands: List[Dict[str, Any]] = []
        for cid, score, hits in fused:
            c = self.chunks[self.chunk_pos[cid]]
            if doc_filter and not doc_filter(c):
                continue
            cands.append({"chunk": c, "fused_score": score, "channels": hits})
            if len(cands) >= max(k * 3, self.candidate_k):
                break

        if self.reranker and cands:
            rs = self.reranker(query, [x["chunk"]["text"] for x in cands])
            for x, s in zip(cands, rs):
                x["rerank_score"] = float(s)
            cands.sort(key=lambda x: -x["rerank_score"])

        out: List[Dict[str, Any]] = []
        for x in cands[:k]:
            c = x["chunk"]
            item = self.format_chunk(c, score=x.get("rerank_score", x["fused_score"]))
            item["channels"] = x["channels"]
            out.append(item)
        return out

    # ── 上下文扩展（P3 补召回用）──
    def format_chunk(self, c: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
        return {
            "score": score,
            "text": c["text"],
            "chunk_id": c["chunk_id"],
            "channels": {},
            "metadata": {
                "source": "local_kb",
                "doc_id": c["doc_id"],
                "doc_name": c["doc_title"],
                "title": c["doc_title"],
                "section_path": c["section_path"],
                "unit_types": c["unit_types"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "published_date": "",
                "image_paths": [],
            },
        }

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        i = self.chunk_pos.get(chunk_id)
        return self.chunks[i] if i is not None else None

    @staticmethod
    def _same_section(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """chunks.jsonl 的 section_path 是块跨越的章节标题列表；两块共享任一章节即视为同章节。"""
        sa, sb = set(a.get("section_path") or []), set(b.get("section_path") or [])
        if not sa and not sb:
            return True
        return bool(sa & sb)

    def neighbors(self, chunk_id: str, window: int = 1, same_section: bool = True) -> List[Dict[str, Any]]:
        """
        返回 chunk_id 在同一文献内前后 window 个相邻块（chunks.jsonl 按文献内顺序写入）。
        same_section=True 时只保留与目标块共享章节的邻块。
        """
        i = self.chunk_pos.get(chunk_id)
        if i is None:
            return []
        base = self.chunks[i]
        out: List[Dict[str, Any]] = []
        for j in range(i - window, i + window + 1):
            if j == i or j < 0 or j >= len(self.chunks):
                continue
            c = self.chunks[j]
            if c["doc_id"] != base["doc_id"]:
                continue
            if same_section and not self._same_section(base, c):
                continue
            out.append(c)
        return out

    def section_chunks(self, chunk_id: str, max_chunks: int = 6) -> List[Dict[str, Any]]:
        """返回与 chunk_id 同文献且章节连通（相邻块章节集合有交集，链式向两侧延伸）的块，不含自身。"""
        i = self.chunk_pos.get(chunk_id)
        if i is None:
            return []
        base = self.chunks[i]
        left: List[Dict[str, Any]] = []
        j, prev = i - 1, base
        while j >= 0 and self.chunks[j]["doc_id"] == base["doc_id"] and self._same_section(prev, self.chunks[j]):
            left.append(self.chunks[j]); prev = self.chunks[j]; j -= 1
        right: List[Dict[str, Any]] = []
        j, prev = i + 1, base
        while j < len(self.chunks) and self.chunks[j]["doc_id"] == base["doc_id"] and self._same_section(prev, self.chunks[j]):
            right.append(self.chunks[j]); prev = self.chunks[j]; j += 1
        # 以目标块为中心交错取，直到 max_chunks
        out: List[Dict[str, Any]] = []
        li = ri = 0
        while len(out) < max_chunks and (li < len(left) or ri < len(right)):
            if li < len(left):
                out.append(left[li]); li += 1
            if len(out) < max_chunks and ri < len(right):
                out.append(right[ri]); ri += 1
        out.sort(key=lambda c: self.chunk_pos[c["chunk_id"]])
        return out


_singleton: Optional[LocalKBRetriever] = None
_singleton_mode: Optional[str] = None


def lexical_rerank(query: str, texts: List[str]) -> List[float]:
    """离线词法重排：查询词覆盖率 × 词序邻近奖励。Qwen3-Reranker-0.6B 不可本地运行时的占位实现，
    保留 callable(query, texts)->scores 接口，接入真实重排器时直接替换。"""
    qt = [t for t in dict.fromkeys(tokenize(query))]
    if not qt:
        return [0.0] * len(texts)
    out: List[float] = []
    for txt in texts:
        toks = tokenize(txt)
        pos: Dict[str, int] = {}
        for i, t in enumerate(toks):
            pos.setdefault(t, i)
        hit = [t for t in qt if t in pos]
        cov = len(hit) / len(qt)
        prox = 0.0
        if len(hit) >= 2:
            span = max(pos[t] for t in hit) - min(pos[t] for t in hit) + 1
            prox = len(hit) / span
        out.append(cov + 0.3 * prox)
    return out


def get_local_kb(top_k: int = 5, mode: str = "two_way") -> LocalKBRetriever:
    global _singleton, _singleton_mode
    if _singleton is None or _singleton_mode != mode:
        _singleton = LocalKBRetriever(top_k=top_k, mode=mode)
        _singleton_mode = mode
    return _singleton


def local_kb_search(query: str, top_k: int = 5, mode: str = "two_way") -> List[Dict[str, Any]]:
    """可直接注册为 rag_search 工具的函数。mode 见 LocalKBRetriever。"""
    return get_local_kb(top_k, mode=mode).search(query, top_k=top_k)
