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
    ) -> None:
        if tokenize is None:
            raise RuntimeError("无法导入 scripts/kb/build_index.tokenize，请先确认 jieba/rank_bm25 已安装")
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.embed_fn = embed_fn

        self.chunks: List[Dict[str, Any]] = [json.loads(l) for l in open(KB / "chunks.jsonl", encoding="utf-8")]
        self.chunk_pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        with open(IDX / "bm25_chunks.pkl", "rb") as f:
            d = pickle.load(f)
        self.bm25, self.bm25_ids = d["bm25"], d["ids"]

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

    # ── 融合 ──
    def _rrf(self, channels: Dict[str, List[str]]) -> List[tuple]:
        agg: Dict[str, float] = {}
        hit: Dict[str, Dict[str, int]] = {}
        weights = {"bm25": 1.0, "dense": 1.0, "anchor": 1.2}
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
            out.append({
                "score": x.get("rerank_score", x["fused_score"]),
                "text": c["text"],
                "chunk_id": c["chunk_id"],
                "channels": x["channels"],
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
            })
        return out


_singleton: Optional[LocalKBRetriever] = None


def local_kb_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """可直接注册为 rag_search 工具的函数。"""
    global _singleton
    if _singleton is None:
        _singleton = LocalKBRetriever(top_k=top_k)
    return _singleton.search(query, top_k=top_k)
