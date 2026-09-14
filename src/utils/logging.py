import logging
from typing import Optional

from rich.logging import RichHandler


_CONFIGURED = False


def _configure_root_logger(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

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

