# 变压器故障关系链路图谱 — Schema v0.1（P2-1）

> 生成日期：2026-09-15
> 依据：`docs/design/data_and_evaluation.md` §1.1 R9 文献库；第二篇论文的"关系链路图谱"思想，但关系类型按变压器物理因果收紧。

## 1. 设计原则

- 关系必须能在文献原句中找到直接依据，抽取时保留 `evidence`（句子 + unit_id）。
- `CAUSES` 只允许「故障/工况 → 故障/症状」方向，禁止「症状 → 故障」（那是 `INDICATES`），避免论文中出现的"逻辑合理性下降"问题。
- 实体先按规范名归一（同义词表），再进图；无法归一的实体只保留原文名并标记 `canonical=false`。
- 每条边带 `confidence`（规则抽取 0.6 / LLM 抽取 0.75 / 人工确认 1.0）与 `support_count`（多少篇文献独立支持）。

## 2. 实体类型（8 类）

| 类型 | 说明 | 示例 |
|---|---|---|
| Component | 变压器部件 | 绕组、铁心、套管、有载分接开关、冷却器、油箱、夹件、引线、压板 |
| Fault | 故障（有明确物理性质） | 匝间短路、铁心多点接地、绕组变形、绝缘老化、绝缘受潮、局部放电、高温过热、低能放电、高能放电、渗漏油 |
| Symptom | 可观测现象/征兆 | 乙炔升高、总烃超标、轻瓦斯动作、油温升高、绝缘电阻下降、直流电阻不平衡、噪声异常、油位异常 |
| Indicator | 可测量指标（含单位） | H2、CH4、C2H2、C2H4、C2H6、CO、CO2、总烃、产气速率、介质损耗因数、吸收比、极化指数、短路阻抗、糠醛、聚合度 |
| Method | 诊断/检测方法 | 三比值法、大卫三角形、特征气体法、频响法、低电压短路阻抗法、局放测量、红外测温、振动分析、油色谱在线监测 |
| Condition | 运行工况/外部诱因 | 突发短路、过负荷、雷击、过电压、冷却失效、进水受潮、长期运行、制造缺陷、安装不当 |
| Action | 处理/检修措施 | 吊罩检查、滤油、真空注油、更换绕组、紧固压钉、串接电阻、停运、缩短色谱周期 |
| Standard | 标准/导则/规程 | DL/T 722、GB/T 7252、IEC 60599、DL/T 596、GB 1094 |

## 3. 关系类型（8 类，含方向约束）

| 关系 | 头 → 尾 | 语义 | 触发词（规则抽取） |
|---|---|---|---|
| CAUSES | Condition/Fault → Fault/Symptom | 物理因果 | 导致、引起、引发、造成、使得、会使 |
| PRODUCES | Fault → Indicator | 故障产生特征气体/量 | 产生、析出、生成、分解出、以…为主 |
| INDICATES | Symptom/Indicator → Fault | 症状指示故障（证据方向） | 表明、说明、反映、意味着、征兆、判断为、诊断为 |
| LOCATED_IN | Fault → Component | 故障部位 | X（部件）的 Y（故障）、发生在 |
| DETECTED_BY | Fault/Symptom → Method | 检测手段 | 采用/通过/利用 … 法/试验 … 检测/判断/诊断 |
| MEASURES | Method → Indicator | 方法测量的量 | 测量、测得、检测 … 值 |
| TREATED_BY | Fault → Action | 处理措施 | 应/需/必须/建议 … 更换/处理/检修/吊罩/滤油 |
| SPECIFIED_IN | Indicator/Method → Standard | 判据来源 | 按照/依据/根据 … 导则/标准/规程 |

约束检查（入库时执行）：
- 任何 `CAUSES` 边若尾实体是 Condition，或头实体是 Symptom/Indicator → 拒绝。
- `INDICATES` 头必须是 Symptom 或 Indicator，尾必须是 Fault。
- `PRODUCES` 尾必须是 Indicator。
- 自环、重复边合并（累加 `support_count`）。

## 4. 实体规范名与同义词表（初版，`data/kg/synonyms.json`）

规范名规则：中文、无空格、优先采用 DL/T 722 / GB/T 7252 用语。

示例：
- 铁心多点接地 ← 铁芯多点接地、铁心多点接地故障、铁心两点接地、铁芯接地故障
- 匝间短路 ← 匝间短路故障、绕组匝间短路、线圈匝间短路、匝间绝缘击穿
- 局部放电 ← 局放、PD、局部放电故障
- 高温过热 ← 高温过热故障、T3、>700℃过热
- 中温过热 ← T2、300~700℃过热
- 低温过热 ← T1、<300℃过热
- 高能放电 ← D2、电弧放电、高能量放电
- 低能放电 ← D1、火花放电、低能量放电
- 乙炔 ← C2H2
- 氢气 ← H2、氢
- 甲烷 ← CH4
- 乙烯 ← C2H4
- 乙烷 ← C2H6
- 一氧化碳 ← CO
- 二氧化碳 ← CO2
- 总烃 ← C1+C2、总烃含量
- 三比值法 ← IEC三比值、改良三比值法、三比值
- 大卫三角形 ← Duval三角形、杜瓦尔三角、大卫三角
- 频响法 ← 频率响应法、FRA、频率响应分析法
- 有载分接开关 ← OLTC、有载开关、有载调压开关
- 突发短路 ← 出口短路、近区短路、外部短路冲击

## 5. 三元组记录格式（`data/kg/triples.jsonl`）

```json
{"head": "突发短路", "head_type": "Condition",
 "relation": "CAUSES",
 "tail": "绕组变形", "tail_type": "Fault",
 "evidence": "变压器在遭受突发短路冲击后，绕组会发生辐向或轴向变形。",
 "unit_id": "2abb73a935dd-0021", "doc_id": "2abb73a935dd",
 "extractor": "rule|llm|human", "confidence": 0.6, "support_count": 1}
```

## 6. 抽取分两阶段

- 阶段 1（无 LLM，现在可做）：规则抽取。以触发词分句，用实体词典做双端匹配，只在「头尾实体都命中词典且类型满足约束」时产出三元组。预期召回低、精度可控。
- 阶段 2（接口恢复后）：LLM 抽取。对含触发词但规则未命中的句子（约 3400 句）批量抽取，输出同格式，`extractor=llm`。
- 阶段 3：两路结果合并、同义归一、约束检查，输出 `graph.json`（节点 + 边）并挂 `kg_search` 工具。

## 7. 与 DGA 诊断的闭环

`fault_attribution` 输出 primary_fault 后，`kg_search` 可按 `Fault → (LOCATED_IN) Component`、`Fault → (TREATED_BY) Action`、`Fault → (DETECTED_BY) Method` 三跳给出部位 / 处理 / 复核手段，Generator 在回答中引用。这是 P2-4 的接入目标。
