"""
数据守卫：防止合成数据混入评测集 / 训练集。

用法：
    from src.utils.data_guard import assert_not_synthetic, is_synthetic_path

    assert_not_synthetic("data/synthetic/eval/eval_set.jsonl", purpose="eval_set")
    # -> 抛出 SyntheticDataError

判定规则：
1. 路径位于任何含 SYNTHETIC_MARKER.json 的目录（或其子目录）之下；
2. 或路径中包含 "synthetic" 目录段。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

_MARKER_NAME = "SYNTHETIC_MARKER.json"


class SyntheticDataError(RuntimeError):
    pass


def _find_marker(path: Path) -> Optional[Path]:
    p = path if path.is_dir() else path.parent
    for parent in [p, *p.parents]:
        m = parent / _MARKER_NAME
        if m.exists():
            return m
    return None


def is_synthetic_path(path: str | Path) -> bool:
    p = Path(path).resolve()
    if any(seg.lower() == "synthetic" for seg in p.parts):
        return True
    return _find_marker(p) is not None


def marker_info(path: str | Path) -> Optional[dict]:
    m = _find_marker(Path(path).resolve())
    if m is None:
        return None
    try:
        return json.loads(m.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"synthetic": True}


def assert_not_synthetic(path: str | Path, purpose: str) -> None:
    """
    purpose 取值：eval_set / train_set / test_set / baseline_report / ...
    若路径为合成数据且 purpose 在其 forbidden_roles 中，抛 SyntheticDataError。
    """
    if not is_synthetic_path(path):
        return
    info = marker_info(path) or {}
    forbidden = set(info.get("forbidden_roles") or
                    ["eval_set", "train_set", "test_set", "baseline_report"])
    if purpose in forbidden:
        raise SyntheticDataError(
            f"拒绝将合成数据用于 {purpose}：{path}\n"
            f"原因：{info.get('note', '该数据为程序合成')}"
        )


def assert_all_not_synthetic(paths: Iterable[str | Path], purpose: str) -> None:
    for p in paths:
        assert_not_synthetic(p, purpose)
