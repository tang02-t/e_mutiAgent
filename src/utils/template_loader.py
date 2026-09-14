"""
模板加载器：从 templates/ 目录加载智能体提示词模板。

目录结构：
    templates/
    ├── planner/
    │   ├── system.txt   # Planner 系统提示词
    │   └── user.txt     # Planner 用户提示词模板（支持 {{变量名}} 占位符）
    ├── generator/
    │   ├── system.txt   # Generator 系统提示词
    │   └── user.txt     # Generator 用户提示词模板
    └── validator/
        ├── system.txt   # Validator 系统提示词
        └── user.txt     # Validator 用户提示词模板

使用方式：
    loader = TemplateLoader()
    system_prompt = loader.load_system("planner")
    user_prompt = loader.render_user("planner", query="球磨机故障")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateLoader:
    """
    轻量级模板加载器，支持：
    - 从文件加载系统提示词
    - 渲染用户提示词模板（替换 {{变量名}} 占位符）
    - 若文件不存在则返回空字符串，并记录警告
    - 缓存已加载的模板内容（按路径）
    """

    def __init__(self, templates_dir: str | None = None) -> None:
        if templates_dir:
            self._base = Path(templates_dir)
        else:
            # templates/ 位于项目根目录
            self._base = Path(__file__).resolve().parents[2] / "templates"
        self._cache: Dict[str, str] = {}

    def _load_file(self, relative_path: str) -> str:
        """加载单个模板文件，带缓存。"""
        if relative_path in self._cache:
            return self._cache[relative_path]

        full_path = self._base / relative_path
        if not full_path.exists():
            logger.warning("模板文件不存在: %s", full_path)
            self._cache[relative_path] = ""
            return ""

        try:
            content = full_path.read_text(encoding="utf-8")
            self._cache[relative_path] = content
            return content
        except Exception as exc:
            logger.error("读取模板文件失败: %s, 错误: %s", full_path, exc)
            self._cache[relative_path] = ""
            return ""

    def load_system(self, agent_name: str) -> str:
        """
        加载指定智能体的系统提示词。
        例如：load_system("planner") → templates/planner/system.txt
        """
        return self._load_file(f"{agent_name}/system.txt")

    def load_user_template(self, agent_name: str) -> str:
        """
        加载指定智能体的用户提示词模板。
        例如：load_user_template("planner") → templates/planner/user.txt
        """
        return self._load_file(f"{agent_name}/user.txt")

    def render_user(self, agent_name: str, **kwargs: Any) -> str:
        """
        渲染指定智能体的用户提示词模板，将 {{变量名}} 替换为实际值。
        支持的变量由各模板文件定义。

        示例：
            render_user("planner", query="球磨机故障")
            render_user("generator", query="...", kb_text="...", ts_text="...")
        """
        template = self.load_user_template(agent_name)
        if not template:
            return ""

        def replacer(match: re.Match) -> str:
            var_name = match.group(1).strip()
            return str(kwargs.get(var_name, match.group(0)))

        return re.sub(r"\{\{(\w+)\}\}", replacer, template)
