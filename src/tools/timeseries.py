from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

import math

from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TimeSeriesTool:
    """
    时序数据分析工具（简化版）。

    真实场景下可以在这里接入 DCS/历史数据库，或调用专用时序分析服务。
    """

    dcs_endpoint: str

    def detect_anomalies(self, signal: Iterable[float]) -> Dict[str, Any]:
        """
        使用简单的 3σ 规则做异常检测示例。
        """
        values: List[float] = [float(v) for v in signal]
        if not values:
            return {"status": "no_data", "message": "没有提供时序数据"}

        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        std = math.sqrt(var)

        if std == 0:
            anomalies: List[int] = []
        else:
            anomalies = [
                idx
                for idx, v in enumerate(values)
                if abs(v - mean) > 3 * std
            ]

        logger.info(
            "Time series anomaly detection: n=%d, mean=%.3f, std=%.3f, anomalies=%d",
            len(values),
            mean,
            std,
            len(anomalies),
        )

        return {
            "status": "ok",
            "mean": mean,
            "std": std,
            "anomaly_indices": anomalies,
        }

