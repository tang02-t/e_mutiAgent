"""
ETT 油温预测工具。

基于 ETT 数据集（电力变压器历史时序数据），提供：
  1. 油温预测（线性回归模型拟合负载特征与油温的线性关系）
  2. 负载-油温统计分析（均值、标准差、趋势）
  3. 油温异常检测（3σ 规则）

该工具供 MCP 工具调用，接收参数后返回预测结果和诊断文本。

数据说明（ETT 数据集）：
  - ETTh1/ETTh2：小时级采样，数据范围 2016-07 ~ 2018-07
  - ETTm1/ETTm2：15分钟级采样，数据范围同上
  - 特征列：HUFL, HULL, MUFL, MULL, LUFL, LULL（负载相关）
  - 目标列：OT（Oil Temperature，油温）

典型任务（参考 Informer/ETT 论文）：
  - 输入：前 lookback 个时间点的负载特征序列
  - 输出：未来 horizon 个时间点的油温预测值
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.tools.ett_loader import ETTDatasetCache, ETTWindow, get_ett_cache
from src.utils.logging import get_logger


logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. 线性回归预测模型（最小二乘法）
# ──────────────────────────────────────────────────────────────

@dataclass
class LinearRegressionOT:
    """
    基于最小二乘法的线性回归模型。
    输入：6维负载特征向量 [HUFL, HULL, MUFL, MULL, LUFL, LULL]
    输出：标量油温 OT
    """
    weights: np.ndarray = field(default_factory=lambda: np.zeros(6))
    bias: float = 0.0
    feature_names: list[str] = field(
        default_factory=lambda: ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    )
    trained: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionOT":
        """
        用最小二乘法拟合线性回归参数。
        X shape: (n_samples, 6), y shape: (n_samples,)
        """
        n = X.shape[0]
        if n < 2:
            raise ValueError(f"Need at least 2 samples to fit, got {n}")

        # 增广矩阵 [X | 1]
        ones = np.ones((n, 1))
        A = np.hstack([X, ones])   # (n, 7)
        try:
            # 正规方程: θ = (A^T A)^(-1) A^T y
            theta = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ y)
            self.weights = theta[:6]
            self.bias = theta[6]
            self.trained = True
        except np.linalg.LinAlgError:
            # 奇异矩阵，改用伪逆
            pseudo_inv = np.linalg.pinv(A)
            theta = pseudo_inv @ y
            self.weights = theta[:6]
            self.bias = theta[6]
            self.trained = True

        logger.info(
            "LinearRegressionOT fitted: weights=%s, bias=%.3f",
            self.weights.round(3).tolist(),
            self.bias,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测油温。
        X shape: (n_samples, 6) 或 (6,) 单样本
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        X = np.atleast_2d(X)
        return X @ self.weights + self.bias

    def coefficients_summary(self) -> list[dict[str, Any]]:
        """返回各特征的系数摘要（绝对值排序）。"""
        result = []
        for name, w in zip(self.feature_names, self.weights):
            result.append({"feature": name, "coefficient": float(w)})
        result.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        return result


# ──────────────────────────────────────────────────────────────
# 2. 滑动窗口预测（参考 ETT 基准任务）
# ──────────────────────────────────────────────────────────────

@dataclass
class ETTSlidingForecaster:
    """
    ETT 滑动窗口预测器。

    典型配置（参考 Informer 论文）：
      - ETTh1/ETTh2：lookback=24*4=96（4天小时数据），horizon=24（1天预测）
      - ETTm1/ETTm2：lookback=24*4*4=384（4天分钟数据），horizon=24*4=96（1天预测）
    """

    dataset_name: str
    lookback: int = 96
    horizon: int = 24
    _model: LinearRegressionOT | None = None

    def __post_init__(self) -> None:
        self._cache: ETTDatasetCache | None = None

    @property
    def cache(self) -> ETTDatasetCache:
        if self._cache is None:
            self._cache = get_ett_cache()
        return self._cache

    def _prepare_data(self, window: ETTWindow) -> tuple[np.ndarray, np.ndarray]:
        """
        将 ETTWindow 转为训练数据格式。
        对每个时间点，用当前时刻的负载特征预测当前时刻的油温。
        """
        X_list: list[list[float]] = []
        y_list: list[float] = []

        for record, ot in zip(window.load_features, window.oil_temps):
            x = [record["hufl"], record["hull"], record["mufl"],
                 record["mull"], record["lufl"], record["lull"]]
            X_list.append(x)
            y_list.append(ot)

        return np.array(X_list), np.array(y_list)

    def _step_ahead_predict(
        self,
        win: ETTWindow,
        horizon: int,
        model: LinearRegressionOT,
    ) -> list[float]:
        """
        用最新窗口的最后一个状态做滚动 horizon 步预测。
        简化版：假设未来负载特征等于窗口最后一个值（基准预测）。
        """
        last_features = win.load_features[-1]
        last_x = np.array([[
            last_features["hufl"], last_features["hull"],
            last_features["mufl"], last_features["mull"],
            last_features["lufl"], last_features["lull"],
        ]])
        last_ot = win.oil_temps[-1]

        predictions: list[float] = []
        current_x = last_x.copy()
        current_ot = last_ot

        for _ in range(horizon):
            pred_ot = model.predict(current_x)[0]
            # 加入轻微的均值回归，避免温度无限漂移
            pred_ot = 0.9 * pred_ot + 0.1 * current_ot
            predictions.append(float(pred_ot))
            # 保持特征不变（简化假设）
            current_ot = pred_ot

        return predictions

    def run_forecast(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        train_window_size: int = 168,  # 默认用最近 168 个点训练
    ) -> dict[str, Any]:
        """
        执行 ETT 油温预测。

        参数:
            start_time       : ISO 格式字符串，查询数据起始时间（None=取最早数据）
            end_time         : ISO 格式字符串，查询数据截止时间（None=取最晚数据）
            train_window_size: 训练窗口大小（数据点数），默认 168（约7天小时数据）

        返回:
            预测结果字典，包含模型参数、预测值、统计摘要
        """
        # ── 加载数据窗口 ──
        try:
            if start_time and end_time:
                window = self.cache.get_window(self.dataset_name, start_time, end_time)
            else:
                window = self.cache.get_latest(self.dataset_name, n=train_window_size)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if len(window.load_features) < 10:
            return {
                "status": "error",
                "message": f"数据点不足（需要>=10，当前{len(window.load_features)}）",
            }

        # ── 训练线性回归模型 ──
        X, y = self._prepare_data(window)
        model = LinearRegressionOT()
        model.fit(X, y)

        # ── 预测未来 horizon 个点 ──
        predictions = self._step_ahead_predict(window, self.horizon, model)

        # ── 历史统计摘要 ──
        summary = self._compute_summary(window, model)

        # ── 异常检测 ──
        anomalies = self._detect_anomalies(window)

        # ── 组装结果 ──
        result = {
            "status": "ok",
            "dataset": self.dataset_name,
            "freq": window.freq,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "history_window": {
                "start": window.start_time.isoformat(),
                "end": window.end_time.isoformat(),
                "num_points": len(window.load_features),
            },
            "model": {
                "type": "LinearRegressionOT",
                "num_params": len(model.weights) + 1,
                "coefficients": model.coefficients_summary(),
                "bias": float(model.bias),
            },
            "forecast": {
                "num_steps": len(predictions),
                "values": [round(v, 3) for v in predictions],
                "mean_predicted_ot": round(sum(predictions) / len(predictions), 3),
                "min_predicted_ot": round(min(predictions), 3),
                "max_predicted_ot": round(max(predictions), 3),
            },
            "history_summary": summary,
            "anomalies": anomalies,
            "text_report": self._build_text_report(window, predictions, summary, anomalies),
        }

        logger.info(
            "ETT forecast [%s]: lookback=%d, horizon=%d, "
            "predicted_OT range=[%.1f, %.1f], anomalies=%d",
            self.dataset_name,
            self.lookback,
            self.horizon,
            min(predictions),
            max(predictions),
            anomalies["count"],
        )
        return result

    def _compute_summary(
        self,
        window: ETTWindow,
        model: LinearRegressionOT,
    ) -> dict[str, Any]:
        """计算历史窗口的统计摘要。"""
        ot_vals = window.oil_temps
        n = len(ot_vals)
        if n == 0:
            return {}
        m = sum(ot_vals) / n
        var = sum((v - m) ** 2 for v in ot_vals) / max(n - 1, 1)
        std = math.sqrt(var)

        # 计算 R²（拟合优度）
        X, y = self._prepare_data(window)
        y_pred = model.predict(X)
        ss_res = sum((a - b) ** 2 for a, b in zip(y, y_pred))
        ss_tot = sum((v - m) ** 2 for v in ot_vals)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        # 各特征均值
        feature_means = {}
        for feat in ("hufl", "hull", "mufl", "mull", "lufl", "lull"):
            vals = [r[feat] for r in window.load_features]
            feature_means[feat.upper()] = round(sum(vals) / len(vals), 3) if vals else 0.0

        return {
            "num_points": n,
            "ot_mean": round(m, 3),
            "ot_std": round(std, 3),
            "ot_min": round(min(ot_vals), 3),
            "ot_max": round(max(ot_vals), 3),
            "r_squared": round(r2, 4),
            "feature_means": feature_means,
        }

    def _detect_anomalies(self, window: ETTWindow) -> dict[str, Any]:
        """基于 3σ 规则检测油温异常。"""
        ot_vals = window.oil_temps
        n = len(ot_vals)
        if n < 3:
            return {"count": 0, "anomaly_indices": []}

        m = sum(ot_vals) / n
        var = sum((v - m) ** 2 for v in ot_vals) / max(n - 1, 1)
        std = math.sqrt(var)

        if std == 0:
            return {"count": 0, "anomaly_indices": [], "threshold_3sigma": 0.0}

        anomaly_indices = []
        for idx, v in enumerate(ot_vals):
            if abs(v - m) > 3 * std:
                anomaly_indices.append(idx)

        return {
            "count": len(anomaly_indices),
            "anomaly_indices": anomaly_indices,
            "threshold_3sigma": round(3 * std, 3),
            "upper_bound": round(m + 3 * std, 3),
            "lower_bound": round(m - 3 * std, 3),
        }

    def _build_text_report(
        self,
        window: ETTWindow,
        predictions: list[float],
        summary: dict[str, Any],
        anomalies: dict[str, Any],
    ) -> str:
        """生成人类可读的文字报告。"""
        lines = ["**ETT 油温预测分析报告**\n"]

        lines.append(f"- 数据集：{self.dataset_name}（{window.freq}）")
        lines.append(f"- 历史窗口：{window.start_time.strftime('%Y-%m-%d %H:%M')} ~ {window.end_time.strftime('%Y-%m-%d %H:%M')}（{summary.get('num_points', 0)} 个点）")
        lines.append("")

        # 统计摘要
        lines.append("**历史油温统计**")
        lines.append(f"- 均值：{summary.get('ot_mean', 'N/A')}℃，标准差：{summary.get('ot_std', 'N/A')}℃")
        lines.append(f"- 范围：{summary.get('ot_min', 'N/A')}℃ ~ {summary.get('ot_max', 'N/A')}℃")
        lines.append(f"- 模型拟合优度 R²={summary.get('r_squared', 'N/A')}（1.0为完美拟合）")
        lines.append("")

        # 负载特征均值
        feat_means = summary.get("feature_means", {})
        if feat_means:
            lines.append("**负载特征均值**（最近窗口）")
            for k, v in feat_means.items():
                lines.append(f"  - {k}：{v}")
            lines.append("")

        # 预测
        lines.append(f"**未来 {self.horizon} 步油温预测**（{window.freq} 间隔）")
        lines.append(f"- 预测均值：{summary.get('forecast', {}).get('mean_predicted_ot', 'N/A')}℃")
        lines.append(f"- 预测范围：{min(predictions):.1f}℃ ~ {max(predictions):.1f}℃")
        # 展示前5步预测
        for i, p in enumerate(predictions[:5]):
            lines.append(f"  - 第 {i+1} 步预测：{p:.2f}℃")
        if len(predictions) > 5:
            lines.append(f"  - ...（共 {len(predictions)} 步，后略）")
        lines.append("")

        # 异常
        anom_count = anomalies.get("count", 0)
        if anom_count > 0:
            lines.append(f"**油温异常告警**（3σ 规则）：发现 {anom_count} 个异常点")
            lines.append(f"  - 告警阈值：±{anomalies.get('threshold_3sigma', 'N/A')}℃")
            lines.append(f"  - 正常范围：[{anomalies.get('lower_bound', 'N/A')}℃, {anomalies.get('upper_bound', 'N/A')}℃]")
            lines.append("  - 建议：结合 DGA 数据和负载情况进一步诊断")
        else:
            lines.append("**油温异常告警**：历史窗口内未检测到 3σ 异常，当前油温运行在正常区间")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 3. MCP 入口函数
# ──────────────────────────────────────────────────────────────

def ett_forecast(
    dataset: str = "ETTh1",
    lookback: int = 96,
    horizon: int = 24,
    start_time: str | None = None,
    end_time: str | None = None,
    train_window_size: int = 168,
) -> dict[str, Any]:
    """
    ETT 油温预测 MCP 工具入口。

    参数:
        dataset          : 数据集名称，可选 ETTh1 / ETTh2 / ETTm1 / ETTm2（默认 ETTh1）
        lookback         : 回看窗口大小（数据点数），默认 96
        horizon          : 预测步数，默认 24
        start_time       : ISO 格式字符串，查询数据起始时间（可选）
        end_time         : ISO 格式字符串，查询数据截止时间（可选）
        train_window_size: 训练数据窗口大小，默认 168

    示例:
        ett_forecast(dataset="ETTh1", horizon=24)
        ett_forecast(dataset="ETTm1", horizon=96, start_time="2018-06-01", end_time="2018-06-07")

    返回:
        {
          "status": "ok",
          "dataset": "ETTh1",
          "forecast": { "values": [...], "mean_predicted_ot": 45.2, ... },
          "history_summary": { "ot_mean": 43.1, "ot_std": 5.3, ... },
          "anomalies": { "count": 2, ... },
          "text_report": "**ETT 油温预测分析报告**\n..."
        }
    """
    valid_datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    if dataset not in valid_datasets:
        return {
            "status": "error",
            "message": f"dataset must be one of {valid_datasets}, got {dataset!r}",
        }

    if lookback < 1 or horizon < 1:
        return {"status": "error", "message": "lookback and horizon must be >= 1"}

    try:
        forecaster = ETTSlidingForecaster(
            dataset_name=dataset,
            lookback=lookback,
            horizon=horizon,
        )
        return forecaster.run_forecast(
            start_time=start_time,
            end_time=end_time,
            train_window_size=train_window_size,
        )
    except Exception as e:
        logger.error("ett_forecast failed: %s", e)
        return {"status": "error", "message": str(e)}
