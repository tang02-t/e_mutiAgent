# 变压器故障诊断系统 - 标准数据模板规范

本文档定义 4 类核心数据的标准字段、格式与取值规范。所有模拟数据均按此规范生成，
后续可用**真实场景数据**按相同 schema 直接替换，无需改动下游工具代码。

目录：
```
data/synthetic/
├── SCHEMA.md                 # 本文档
├── dga/                      # 1. DGA 油色谱数据（对接 fault_attribution）
│   ├── dga_records.csv       #    扁平记录（含真实故障标签）
│   └── dga_records.jsonl     #    同内容 JSONL（含贝叶斯入参 evidence）
├── timeseries/               # 2. 在线监测时序数据（对接 ett_forecast / timeseries_anomaly）
│   ├── TR-0001.csv ...       #    每台设备一个 CSV，列对齐 ETT
│   └── devices.json          #    设备台账（型号/电压等级/投运年限）
├── cases/                    # 3. 故障案例库
│   └── fault_cases.jsonl     #    结构化案例（征兆→试验→定位→处理）
└── eval/                     # 4. 端到端评测集
    └── eval_set.jsonl        #    问题 → 标准答案
```

---

## 1. DGA 油色谱数据 (`dga/`)

变压器油中溶解气体分析，是故障归因（`src/tools/fault_attribution.py`）的核心输入。
气体浓度单位均为 **μL/L (ppm)**。

### CSV 字段 (`dga_records.csv`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | str | 记录唯一 ID，如 `DGA-000001` |
| `device_id` | str | 设备 ID，关联 `timeseries/devices.json`，如 `TR-0001` |
| `sample_time` | datetime | 取样时间 `YYYY-MM-DD HH:MM:SS` |
| `H2` | float | 氢气浓度 (μL/L)，注意值 150 |
| `CH4` | float | 甲烷浓度 (μL/L)，注意值 120 |
| `C2H2` | float | 乙炔浓度 (μL/L)，注意值 5 |
| `C2H4` | float | 乙烯浓度 (μL/L)，注意值 50 |
| `C2H6` | float | 乙烷浓度 (μL/L)，注意值 65 |
| `CO` | float | 一氧化碳浓度 (μL/L) |
| `CO2` | float | 二氧化碳浓度 (μL/L) |
| `TDCG` | float | 总可燃气体 = H2+CH4+C2H2+C2H4+C2H6+CO |
| `total_hydrocarbon` | float | 总烃 = CH4+C2H2+C2H4+C2H6 |
| `gas_rate_ml_per_day` | float | 相对产气速率 (mL/天)，>较高则产气快 |
| `oil_temp` | float | 取样时上层油温 (℃) |
| `load_ratio` | float | 取样时负载率 (0~1) |
| `fault_label` | str | **真实故障标签**（停电检修后结论），见下方枚举 |
| `fault_label_cn` | str | 故障标签中文名 |
| `severity` | str | 严重程度：normal / attention / serious / critical |

### `fault_label` 枚举（与 `fault_attribution.FAULT_IDS` 完全一致）

| label | 中文 |
|-------|------|
| `normal` | 正常 |
| `winding_deformation` | 绕组变形/位移 |
| `winding_short_circuit` | 绕组匝间短路 |
| `core_grounding` | 铁芯多点接地 |
| `bushing_fault` | 套管故障 |
| `olc_fault` | 分接开关故障 |
| `partial_discharge` | 局部放电 |
| `overload_overheating` | 过载过热 |
| `insulation_degradation` | 绝缘老化 |

### JSONL 行结构 (`dga_records.jsonl`)
在 CSV 字段基础上额外提供可直接喂给 `fault_attribution()` 的入参：
```json
{
  "record_id": "DGA-000001",
  "device_id": "TR-0001",
  "sample_time": "2024-03-12 09:30:00",
  "dga_data": {"H2": 680, "CH4": 150, "C2H2": 45, "C2H4": 230, "C2H6": 60, "CO": 320, "CO2": 2800},
  "evidence": {"H2_elevated": true, "C2H2_elevated": true, "C2H4_elevated": true},
  "gas_rate_ml_per_day": 38.5,
  "oil_temp": 78.2,
  "load_ratio": 0.86,
  "fault_label": "winding_short_circuit",
  "fault_label_cn": "绕组匝间短路",
  "severity": "critical"
}
```
> `dga_data` 可直接传入 `fault_attribution(dga_data=...)`；`fault_label` 用于评估归因准确率。

---

## 2. 在线监测时序数据 (`timeseries/`)

每台设备一个 CSV，**列严格对齐 ETT 数据集**，可直接复用 `src/tools/ett_loader.py`
与 `ett_forecasting.py`。采样频率默认小时级（freq=H）。

### CSV 字段 (`TR-XXXX.csv`)

| 字段 | 类型 | 说明 | ETT 对应 |
|------|------|------|----------|
| `date` | datetime | 时间戳 `YYYY-MM-DD HH:MM:SS` | date |
| `HUFL` | float | 高压侧有功负载特征 | HUFL |
| `HULL` | float | 高压侧无功负载特征 | HULL |
| `MUFL` | float | 中压侧有功负载特征 | MUFL |
| `MULL` | float | 中压侧无功负载特征 | MULL |
| `LUFL` | float | 低压侧有功负载特征 | LUFL |
| `LULL` | float | 低压侧无功负载特征 | LULL |
| `OT` | float | 上层油温 (℃)，预测目标 | OT |

> 扩展列（真实场景可附加，下游可选用）：`winding_temp`(绕组温度)、`load_current`(负载电流A)、
> `partial_discharge_pc`(局放量pC)、`core_ground_current`(铁芯接地电流mA)、`vibration`(振动mm/s)。
> 默认生成文件仅含 8 个 ETT 标准列以保证兼容；扩展列写入 `TR-XXXX_ext.csv`。

### 设备台账 (`devices.json`)
```json
[
  {
    "device_id": "TR-0001",
    "model": "SFSZ-180000/220",
    "voltage_level_kv": 220,
    "rated_capacity_kva": 180000,
    "commission_year": 2009,
    "service_years": 17,
    "location": "某500kV变电站",
    "cooling_type": "ONAN/ONAF"
  }
]
```

---

## 3. 故障案例库 (`cases/fault_cases.jsonl`)

结构化真实故障案例，用于补充 RAG 知识库 + 构建端到端评测。

```json
{
  "case_id": "CASE-0001",
  "device_id": "TR-0007",
  "device_info": {"model": "SFSZ-180000/220", "voltage_level_kv": 220, "service_years": 15},
  "symptoms": ["油温告警", "C2H2持续升高", "轻瓦斯动作"],
  "dga_snapshot": {"H2": 520, "CH4": 130, "C2H2": 68, "C2H4": 210, "C2H6": 55},
  "tests": {
    "dc_resistance_unbalance_pct": 3.2,
    "insulation_resistance_mohm": 1200,
    "dielectric_loss_pct": 0.8,
    "winding_fra": "明显偏移"
  },
  "fault_location": "高压绕组A相匝间短路",
  "fault_label": "winding_short_circuit",
  "handling": "立即停电，吊罩检查，更换高压绕组",
  "outcome": "修复后投运正常",
  "reference": "DL/T 722-2014"
}
```

---

## 4. 端到端评测集 (`eval/eval_set.jsonl`)

用于评估 Planner→Retriever→Generator→Validator 整条链路。

```json
{
  "eval_id": "EVAL-0001",
  "user_query": "220kV主变油色谱C2H2达到68μL/L，H2 520，伴随油温告警，请诊断并给出处理建议",
  "context": {"device_id": "TR-0007", "voltage_level_kv": 220},
  "expected": {
    "primary_fault": "winding_short_circuit",
    "primary_fault_cn": "绕组匝间短路",
    "key_points": ["立即停电", "高能放电", "C2H2显著超标", "吊罩检查"],
    "must_mention_safety": true
  }
}
```

---

## 数据规模（默认生成）

| 数据集 | 规模 |
|--------|------|
| 设备台账 | 50 台 |
| DGA 记录 | 3000 条（覆盖 9 类标签，含正常） |
| 时序数据 | 20 台 × 约 2000 小时点 |
| 故障案例 | 200 例 |
| 评测集 | 120 条 |
