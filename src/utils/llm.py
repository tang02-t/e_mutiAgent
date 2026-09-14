from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


try:  # 允许在没有安装 openai 时退化
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


@dataclass
class LLMConfig:
    provider: str = "openai"
    model_name: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    # 兼容 OpenAI SDK 的第三方网关（ModelScope / vLLM 等）
    base_url: str = "https://api.openai.com/v1"
    # 直接写死的 api_key（推荐开发调试用），优先级高于 api_key_env
    api_key: str = ""
    # 环境变量方式（优先级低于 api_key）
    api_key_env: str = "OPENAI_API_KEY"
    # Qwen3 系列模型：是否启用思考模式（默认关闭，直接返回结果）
    enable_thinking: bool = False
    # 单次请求超时时间（秒）
    timeout: float = 60.0
    # 失败后的最大重试次数（指数退避）
    max_retries: int = 2


class LLMClient:
    """
    统一封装大模型调用接口，当前示例以 OpenAI 风格为主：

    - 从 config.yaml 读取模型配置
    - 从环境变量读取 API Key（如 OPENAI_API_KEY）
    - 若未配置或未安装 openai 包，则自动退回为“禁用状态”，上层可自行采用规则逻辑
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client: Any | None = None

        # 如果 model_name 为空，认为该智能体不需要 LLM
        if not cfg.model_name:
            self._enabled = False
            return

        if cfg.provider.lower() == "openai" and OpenAI is not None:
            # 优先使用 config 中的 api_key，再 fallback 到环境变量
            api_key = cfg.api_key or os.getenv(cfg.api_key_env)
            if not api_key:
                logger.warning("LLM API Key 未设置（config.api_key 和 %s 环境变量都未配置），LLM 调用将被禁用。", cfg.api_key_env)
                self._enabled = False
            else:
                self._client = OpenAI(
                    base_url=cfg.base_url,
                    api_key=api_key,
                    timeout=cfg.timeout,
                    max_retries=0,  # 重试由 chat() 自行控制，避免双重退避
                )
                self._enabled = True
        else:
            if cfg.provider.lower() == "openai":
                logger.warning("未安装 openai 包，LLM 调用将被禁用。")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        统一的对话式调用接口。

        返回值结构（兼容 OpenAI ChatCompletion message）：
            {
              "role": "assistant",
              "content": "...",
              "tool_calls": [   # 仅当传入 tools 且模型触发函数调用时存在
                {"name": "rag_search", "arguments": {"query": "..."}},
                ...
              ],
            }
        其中 arguments 已解析为 dict（解析失败则为空 dict）。
        """
        if not self._enabled:
            raise RuntimeError("LLMClient 未启用，无法实际调用大模型。")

        if self.cfg.provider.lower() == "openai":
            if self._client is None:
                raise RuntimeError("LLMClient client 未初始化。")

            # 构建请求参数
            create_kwargs: Dict[str, Any] = {
                "model": self.cfg.model_name,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
                "messages": messages,
            }

            # Qwen3 系列模型：禁用思考模式，直接返回结果
            if self.cfg.enable_thinking is False:
                create_kwargs["extra_body"] = {"enable_thinking": False}

            if tools:
                create_kwargs["tools"] = tools

            # openai>=1.0: client.chat.completions.create
            # 有限重试 + 指数退避：网络抖动 / 限流时提升鲁棒性
            attempts = max(1, self.cfg.max_retries + 1)
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    resp = self._client.chat.completions.create(**create_kwargs)  # type: ignore[attr-defined]
                    msg = resp.choices[0].message
                    return self._normalize_message(msg)
                except Exception as exc:  # noqa: BLE001 - 网络/限流等异常统一重试
                    last_exc = exc
                    if attempt < attempts - 1:
                        backoff = 2 ** attempt
                        logger.warning(
                            "LLM 调用失败（第 %d/%d 次），%ds 后重试: %s",
                            attempt + 1, attempts, backoff, exc,
                        )
                        time.sleep(backoff)

            # 所有重试均失败，向上抛出由调用方降级处理
            raise RuntimeError(f"LLM 调用在 {attempts} 次尝试后仍失败: {last_exc}") from last_exc

        raise NotImplementedError(f"暂不支持的 provider: {self.cfg.provider}")

    @staticmethod
    def _normalize_message(msg: Any) -> Dict[str, Any]:
        """将 OpenAI message 对象规范化为统一 dict，并解析 tool_calls 的 arguments。"""
        import json

        result: Dict[str, Any] = {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", "") or "",
        }

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            parsed_calls: List[Dict[str, Any]] = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                raw_args = getattr(fn, "arguments", "") or ""
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except (json.JSONDecodeError, ValueError, TypeError):
                    logger.warning("tool_call arguments 解析失败: %s", raw_args[:200])
                    args = {}
                parsed_calls.append({"name": getattr(fn, "name", ""), "arguments": args})
            if parsed_calls:
                result["tool_calls"] = parsed_calls

        return result

