from typing import Any, Callable, Dict

from src.utils.logging import get_logger


logger = get_logger(__name__)


ToolFn = Callable[..., Any]


class MCPClient:
    """
    极简版 MCP 风格客户端：
    - 支持通过字符串名称注册工具函数
    - 统一的 call_tool 接口，便于在智能体中调用
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register_tool(self, name: str, fn: ToolFn) -> None:
        if name in self._tools:
            logger.warning("Tool %s already registered, overriding.", name)
        self._tools[name] = fn
        logger.info("Registered tool [bold]%s[/bold].", name)

    def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool {name!r} not registered.")
        logger.info("Calling tool [bold]%s[/bold] ...", name)
        fn = self._tools[name]
        return fn(*args, **kwargs)

