from typing import Dict, List, Any

from pymilvus import Collection, connections  # type: ignore

from src.utils.logging import get_logger
from src.utils.embedding import LocalEmbeddingModel


logger = get_logger(__name__)


class RAGEngine:
    """
    基于 Milvus + 本地 Embedding 模型的 RAG 引擎。

    支持两种检索模式：
    - 父子分块模式（默认）：先命中子块 → 通过 parent_id 展开父块完整上下文
    - 整篇模式：直接检索整篇文档
    """

    def __init__(
        self,
        uri: str | None,
        token: str | None,
        host: str | None,
        port: int | None,
        collection_name: str,
        text_field: str,
        vector_field: str,
        top_k: int = 5,
        parent_child: bool = True,
        embedding_model: LocalEmbeddingModel | None = None,
        metric_type: str = "L2",
        nprobe: int = 10,
    ) -> None:
        self.top_k = top_k
        self.text_field = text_field
        self.vector_field = vector_field
        self.parent_child = parent_child
        self.metric_type = metric_type
        self.nprobe = nprobe

        if uri and token:
            if "://" not in uri:
                raise ValueError(
                    f"Zilliz Cloud uri must be a full URL (https:// or tcp://), got: {uri!r}"
                )
            connections.connect(alias="default", uri=uri, token=token)
        elif host and port is not None:
            connections.connect(alias="default", host=host, port=str(port))
        else:
            raise ValueError("Provide (uri, token) or (host, port) to connect to Milvus.")
        self.collection = Collection(collection_name)

        # 元数据字段
        self.meta_fields = [
            "chunk_id", "chunk_type", "parent_id", "chunk_level", "chunk_category",
            "source", "doc_name", "title", "published_date", "image_paths",
        ]

        self.embedding_model = embedding_model or LocalEmbeddingModel()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        在 Milvus 中进行相似度检索，返回结构化结果：

        parent_child=True 时（父子分块模式）：
        [
          {
            "score": 0.12,          # 子块相似度分数
            "child": {              # 子块（检索命中单元）
              "text": "具体段落内容...",
              "parent_heading": "1.2.1 外部检查",
              "image_paths": ["E:/path/to/img.jpg"],
            },
            "parent": {             # 父块（完整章节上下文）
              "text": "## 1.2.1 外部检查\\n\\n完整章节所有段落...",
              "heading": "1.2.1 外部检查",
              "chunk_level": 3,
            },
            "metadata": {
              "source": "...", "doc_name": "...", "title": "...",
              "published_date": "2021-08-09", "image_paths": [...],
            }
          },
          ...
        ]
        """
        import json

        query_vec = self.embedding_model.embed([query])[0]
        search_params = {"metric_type": self.metric_type, "params": {"nprobe": self.nprobe}}

        if self.parent_child:
            return self._search_parent_child(query_vec, search_params)
        return self._search_flat(query_vec, search_params)

    def _search_parent_child(
        self,
        query_vec: List[float],
        search_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        父子分块检索：
        1. 只在 child 块中做向量检索（精确定位到段落）
        2. 通过 parent_id 展开父块完整章节内容
        3. 返回每个命中的子块 + 对应父块上下文
        """
        import json

        # ① 只检索子块
        results = self.collection.search(
            data=[query_vec],
            anns_field=self.vector_field,
            param=search_params,
            limit=self.top_k,
            output_fields=self.meta_fields + [self.text_field],
            expr='chunk_type == "child"',
        )

        hits = results[0] if results else []
        if not hits:
            logger.info("No child chunk hits for query.")
            return []

        # ② 收集所有命中的 parent_id，一次性批量查询父块
        parent_ids = list({h.entity.get("parent_id") for h in hits if h.entity.get("parent_id")})

        parent_map: Dict[str, Dict[str, Any]] = {}
        if parent_ids:
            # 用 json.dumps 安全构造 Milvus 表达式，避免 id 含引号/特殊字符导致的语法错误或注入
            id_list_expr = json.dumps(parent_ids, ensure_ascii=False)
            parent_results = self.collection.query(
                expr=f"chunk_id in {id_list_expr}",
                output_fields=self.meta_fields + [self.text_field],
            )
            for row in parent_results:
                parent_map[row["chunk_id"]] = row

        # ③ 组装结果
        candidates: List[Dict[str, Any]] = []
        for h in hits:
            parent_id = h.entity.get("parent_id") or ""
            parent_row = parent_map.get(parent_id, {})

            child_images_raw = h.entity.get("image_paths") or ""
            try:
                child_images: List[str] = json.loads(child_images_raw)
            except Exception:
                child_images = []

            parent_images_raw = parent_row.get("image_paths") or ""
            try:
                parent_images: List[str] = json.loads(parent_images_raw)
            except Exception:
                parent_images = []

            entry = {
                "score": float(h.distance),
                "child": {
                    "text": h.entity.get(self.text_field) or "",
                    "parent_heading": h.entity.get("parent_id") or "",
                    "chunk_category": h.entity.get("chunk_category") or "paragraph",
                    "image_paths": child_images,
                },
                "parent": {
                    "text": parent_row.get(self.text_field) or "",
                    "heading": parent_row.get("parent_id") or "",
                    "chunk_level": parent_row.get("chunk_level") or 0,
                    "image_paths": parent_images,
                },
                "metadata": {
                    "source": h.entity.get("source") or "",
                    "doc_name": h.entity.get("doc_name") or "",
                    "title": h.entity.get("title") or "",
                    "published_date": h.entity.get("published_date") or "",
                    "all_image_paths": child_images + parent_images,
                },
            }
            candidates.append(entry)

        logger.info("Parent-child search found %d child hits, expanded to %d parents.",
                    len(hits), len(parent_map))
        return candidates

    def _search_flat(
        self,
        query_vec: List[float],
        search_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """整篇文档检索（无父子分块）。"""
        import json

        results = self.collection.search(
            data=[query_vec],
            anns_field=self.vector_field,
            param=search_params,
            limit=self.top_k,
            output_fields=self.meta_fields + [self.text_field],
        )

        hits = results[0] if results else []
        candidates: List[Dict[str, Any]] = []

        for h in hits:
            images_raw = h.entity.get("image_paths") or ""
            try:
                images = json.loads(images_raw)
            except Exception:
                images = []

            candidates.append({
                "score": float(h.distance),
                "text": h.entity.get(self.text_field) or "",
                "metadata": {
                    "source": h.entity.get("source") or "",
                    "doc_name": h.entity.get("doc_name") or "",
                    "title": h.entity.get("title") or "",
                    "published_date": h.entity.get("published_date") or "",
                    "image_paths": images,
                },
            })

        logger.info("Flat search found %d hits.", len(candidates))
        return candidates
