import logging
import os
from typing import Optional

from rich.logging import RichHandler


_CONFIGURED = False


def _configure_root_logger(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    # 批量脚本可用 LOG_LEVEL=WARNING 静音智能体的 INFO 日志
    level = os.environ.get("LOG_LEVEL", level)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    获取项目统一的 logger。
    """
    _configure_root_logger(level=level)
    return logging.getLogger(name if name is not None else "metallurgy")

