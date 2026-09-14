from pathlib import Path
from typing import Optional, Dict, Any

import yaml

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _default_config_path() -> Path:
    # src/utils/config.py -> project root
    return Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件，带简单缓存。
    """
    global _CONFIG_CACHE

    if path is None and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    cfg_path = Path(path) if path is not None else _default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if path is None:
        _CONFIG_CACHE = config

    return config


def get_llm_config(config: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    获取指定智能体的 LLM 配置。

    优先级：llms.{agent_name} > llms.default
    若整个 llms 节点不存在（旧配置兼容），则回退到顶层的 model 节点。
    """
    llms_cfg = config.get("llms", {})

    if not llms_cfg:
        # 兼容旧配置格式：顶层 model 节点
        return config.get("model", {})

    # 优先使用指定智能体的配置，否则用 default
    agent_cfg = llms_cfg.get(agent_name)
    default_cfg = llms_cfg.get("default", {})

    if agent_cfg:
        result = dict(default_cfg)
        result.update(agent_cfg)
        return result
    return default_cfg

