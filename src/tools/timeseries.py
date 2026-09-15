"""
通用时序异常检测工具（timeseries_anomaly）。

这是 app.py / scripts/eval/eval_system_modes.py / 测试共用的唯一实现，
避免同一份 3σ 逻辑在多处复制。

约束（P0 口径）：
- signal 必须由模型 / 用户显式传入；缺失或过短直接返回业务错误，不再使用任何示例序列。
- 返回值遵循工具统一约定：status=ok|error，error 时附 message。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

_MIN_POINTS = 3


def timeseries_anomaly(signal: Optional[Iterable[float]] = None, **_: Any) -> Dict[str, Any]:
    """对一段数值序列做 3σ 异常检测。"""
    if signal is None or not isinstance(signal, (list, tuple)) or len(signal) == 0:
        return {"status": "error", "message": "timeseries_anomaly 需要非空 signal 序列"}
    try:
        values: List[float] = [float(v) for v in signal]
    except (TypeError, ValueError) as exc:
        return {"status": "error", "message": f"signal 含非数值元素：{exc}"}
    if len(values) < _MIN_POINTS:
        return {"status": "error", "message": f"signal 长度 {len(values)} 过短，至少需要 {_MIN_POINTS} 个点"}

    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    std = math.sqrt(var)
    anomalies = [i for i, v in enumerate(values) if std and abs(v - mean) > 3 * std]
    return {
        "status": "ok",
        "n": len(values),
        "mean": mean,
        "std": std,
        "anomaly_indices": anomalies,
        "series": values,
    }
