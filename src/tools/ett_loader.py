"""
ETT（Electric Transformer Temperature）数据集加载器。

数据来源：中国国家电网，某省两个不同县变压器，时间跨度 2016-07 至 2018-07（约2年）。

数据变体：
  - ETTh1 / ETTh2：小时级采样（hourly），共约 17420 条/文件
  - ETTm1 / ETTm2：15分钟级采样（minutely），共约 69680 条/文件

字段说明：
  - date         ：时间戳
  - HUFL / HULL  ：高位负载特征（High/Uncertain Load Feature）
  - MUFL / MULL  ：中位负载特征（Middle Load Feature）
  - LUFL / LULL  ：低位负载特征（Low Load Feature）
  - OT           ：Oil Temperature，油温（预测目标）

典型任务：
  - 输入前 6 个负载特征（HUFL ~ LULL）的历史窗口
  - 预测未来油温（OT）值
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import pandas as pd

from src.utils.logging import get_logger


logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# 数据集根目录
# ──────────────────────────────────────────────────────────────

_DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "ETT-small"

_DATASET_FILES: dict[str, dict[str, Any]] = {
    "ETTh1": {"path": _DATA_ROOT / "ETTh1.csv", "freq": "H"},
    "ETTh2": {"path": _DATA_ROOT / "ETTh2.csv", "freq": "H"},
    "ETTm1": {"path": _DATA_ROOT / "ETTm1.csv", "freq": "15T"},
    "ETTm2": {"path": _DATA_ROOT / "ETTm2.csv", "freq": "15T"},
}


# ──────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────

@dataclass
class ETTSample:
    """单条 ETT 记录。"""
    date: datetime
    hufl: float  # 高位负载特征（High Utility Factor Load）
    hull: float
    mufl: float  # 中位负载特征
    mull: float
    lufl: float  # 低位负载特征
    lull: float
    ot: float    # 油温（预测目标）

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "HUFL": self.hufl,
            "HULL": self.hull,
            "MUFL": self.mufl,
            "MULL": self.mull,
            "LUFL": self.lufl,
            "LULL": self.lull,
            "OT": self.ot,
        }


@dataclass
class ETTWindow:
    """一个时间窗口：输入特征序列 + 对应 OT 序列。"""
    start_time: datetime
    end_time: datetime
    load_features: list[dict[str, float]]   # 每条记录的 HUFL~LULL
    oil_temps: list[float]                  # 每条记录的 OT
    freq: str                                # 采样频率，H / 15T

    def to_input_features(self) -> list[dict[str, float]]:
        """返回 6 个负载特征列表（不含 OT），供模型输入。"""
        return self.load_features

    def last_oil_temp(self) -> float | None:
        return self.oil_temps[-1] if self.oil_temps else None

    def summary(self) -> dict[str, Any]:
        def _stats(values: list[float]) -> dict[str, float]:
            n = len(values)
            if n == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            m = sum(values) / n
            var = sum((v - m) ** 2 for v in values) / max(n - 1, 1)
            return {"mean": m, "std": math.sqrt(var), "min": min(values), "max": max(values)}
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "num_points": len(self.load_features),
            "freq": self.freq,
            "load_summary": {k: _stats([r[k] for r in self.load_features])
                            for k in ("hufl", "hull", "mufl", "mull", "lufl", "lull")},
            "ot_summary": _stats(self.oil_temps),
        }


# ──────────────────────────────────────────────────────────────
# 数据集缓存
# ──────────────────────────────────────────────────────────────

class ETTDatasetCache:
    """
    全局数据集缓存，只在首次访问时加载 CSV 到内存。
    支持按时间范围查询和滑动窗口切分。
    """

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, name: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]) -> pd.DataFrame:
        """加载指定数据集（带缓存）。"""
        if name in self._cache:
            return self._cache[name]

        info = _DATASET_FILES.get(name)
        if not info:
            raise ValueError(f"Unknown dataset {name!r}. Available: {list(_DATASET_FILES)}")

        path = info["path"]
        if not path.exists():
            raise FileNotFoundError(f"ETT dataset not found: {path}")

        # 注意：pandas 新版本中 "15T" 已弃用，等价写法为 "15min"；
        # 旧代码把 15T 误映射成 "min"（1 分钟），会把 15 分钟数据扩成 1 分钟并插入大量 NaN。
        freq_map = {"H": "h", "15T": "15min"}
        freq = freq_map[info["freq"]]

        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        n_raw = len(df)
        df = df.asfreq(freq)
        n_missing = int(df["OT"].isna().sum()) if "OT" in df.columns else 0
        if len(df) != n_raw or n_missing:
            logger.warning(
                "ETT %s asfreq(%s): 原始 %d 行 → 对齐后 %d 行，OT 缺失 %d 个（原始数据存在时间间隙）",
                name, freq, n_raw, len(df), n_missing,
            )

        self._cache[name] = df
        logger.info("Loaded ETT dataset [bold]%s[/bold]: %d rows, freq=%s, range=%s ~ %s",
                    name, len(df), info["freq"],
                    df.index[0].isoformat() if len(df) else "N/A",
                    df.index[-1].isoformat() if len(df) else "N/A")
        return df

    def get_window(
        self,
        name: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        start: datetime | str,
        end: datetime | str,
    ) -> ETTWindow:
        """
        获取指定时间窗口的数据。
        start / end 可以是 datetime 或 ISO 字符串。
        """
        df = self.load(name)
        freq = _DATASET_FILES[name]["freq"]

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        mask = (df.index >= start) & (df.index <= end)
        sliced = df[mask]

        if sliced.empty:
            raise ValueError(f"No data found in {name} for range [{start}, {end}]")

        records = []
        for ts, row in sliced.iterrows():
            records.append({
                "date": ts,
                "hufl": float(row["HUFL"]),
                "hull": float(row["HULL"]),
                "mufl": float(row["MUFL"]),
                "mull": float(row["MULL"]),
                "lufl": float(row["LUFL"]),
                "lull": float(row["LULL"]),
                "ot": float(row["OT"]),
            })

        load_features = [{"hufl": r["hufl"], "hull": r["hull"],
                          "mufl": r["mufl"], "mull": r["mull"],
                          "lufl": r["lufl"], "lull": r["lull"]} for r in records]
        oil_temps = [r["ot"] for r in records]

        return ETTWindow(
            start_time=records[0]["date"],
            end_time=records[-1]["date"],
            load_features=load_features,
            oil_temps=oil_temps,
            freq=freq,
        )

    def iter_windows(
        self,
        name: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        window_size: int,
        step: int = 1,
    ) -> Iterator[ETTWindow]:
        """
        滑动窗口迭代器。

        参数:
            name        : 数据集名称
            window_size: 窗口大小（数据点数）
            step        : 滑动步长（默认1）
        """
        df = self.load(name)
        freq = _DATASET_FILES[name]["freq"]

        for start_idx in range(0, len(df) - window_size + 1, step):
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx]
            records = []
            for ts, row in window_df.iterrows():
                records.append({
                    "date": ts,
                    "hufl": float(row["HUFL"]),
                    "hull": float(row["HULL"]),
                    "mufl": float(row["MUFL"]),
                    "mull": float(row["MULL"]),
                    "lufl": float(row["LUFL"]),
                    "lull": float(row["LULL"]),
                    "ot": float(row["OT"]),
                })
            load_features = [{"hufl": r["hufl"], "hull": r["hull"],
                              "mufl": r["mufl"], "mull": r["mull"],
                              "lufl": r["lufl"], "lull": r["lull"]} for r in records]
            oil_temps = [r["ot"] for r in records]
            yield ETTWindow(
                start_time=records[0]["date"],
                end_time=records[-1]["date"],
                load_features=load_features,
                oil_temps=oil_temps,
                freq=freq,
            )

    def get_latest(
        self,
        name: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        n: int = 24,
    ) -> ETTWindow:
        """获取最近 n 个点的窗口。"""
        df = self.load(name)
        freq = _DATASET_FILES[name]["freq"]
        sliced = df.tail(n)

        records = []
        for ts, row in sliced.iterrows():
            records.append({
                "date": ts,
                "hufl": float(row["HUFL"]),
                "hull": float(row["HULL"]),
                "mufl": float(row["MUFL"]),
                "mull": float(row["MULL"]),
                "lufl": float(row["LUFL"]),
                "lull": float(row["LULL"]),
                "ot": float(row["OT"]),
            })
        load_features = [{"hufl": r["hufl"], "hull": r["hull"],
                          "mufl": r["mufl"], "mull": r["mull"],
                          "lufl": r["lufl"], "lull": r["lull"]} for r in records]
        oil_temps = [r["ot"] for r in records]
        return ETTWindow(
            start_time=records[0]["date"],
            end_time=records[-1]["date"],
            load_features=load_features,
            oil_temps=oil_temps,
            freq=freq,
        )

    def dataset_info(self, name: str | None = None) -> dict[str, Any]:
        """返回数据集概览信息。"""
        if name:
            names = [name] if name in _DATASET_FILES else []
        else:
            names = list(_DATASET_FILES)
        result = {}
        for n in names:
            df = self.load(n)
            info = _DATASET_FILES[n]
            result[n] = {
                "path": str(info["path"]),
                "freq": info["freq"],
                "num_rows": len(df),
                "start": df.index[0].isoformat() if len(df) else None,
                "end": df.index[-1].isoformat() if len(df) else None,
                "columns": list(df.columns),
            }
        return result


# 全局单例
_ett_cache: ETTDatasetCache | None = None


def get_ett_cache() -> ETTDatasetCache:
    global _ett_cache
    if _ett_cache is None:
        _ett_cache = ETTDatasetCache()
    return _ett_cache
