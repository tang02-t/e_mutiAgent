from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from src.utils.logging import get_logger


logger = get_logger(__name__)


try:  # optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class LocalEmbeddingModel:
    """
    Embedding 模型封装（支持通过 API 部署的"本地/私有"Embedding 服务）。

    当前默认实现对接阿里云百炼 DashScope：

    - base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    - model: text-embedding-v4
    - encoding_format: float

    注意：不要把 API Key 写死在代码或 config 文件里，推荐用环境变量传入。
    """

    model_path: str = ""  # 保留字段，兼容旧配置（不再强依赖）
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "text-embedding-v4"
    api_key: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"

    def __post_init__(self) -> None:
        # 延迟初始化的客户端缓存，避免每次 embed() 都重建连接
        self._client = None

    def _get_client(self):
        """惰性创建并复用 OpenAI 兼容客户端。"""
        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise RuntimeError(
                "未安装 openai 包，无法调用 embedding API。请先 pip install -r requirements.txt"
            )

        # 优先使用环境变量，再 fallback 到 api_key 字段（仅用于开发调试）
        api_key = os.getenv(self.api_key_env) or self.api_key
        if not api_key:
            raise RuntimeError(
                f"未设置环境变量 {self.api_key_env}，无法调用 embedding API。"
            )

        self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将一批文本编码为向量。

        通过 OpenAI 兼容 SDK 调用 embeddings 接口，支持 batch 输入。
        """
        if not texts:
            return []

        client = self._get_client()
        resp = client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )

        # openai>=1.0: resp.data[i].embedding
        vectors: List[List[float]] = [d.embedding for d in resp.data]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding API 返回数量与输入不一致。")

        return vectors

