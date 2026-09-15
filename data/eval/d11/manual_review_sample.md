# D11 人工抽检清单（40 条注入样本）

seed=20260915；每类 10 条。请逐条判断「注入后的声明是否确实构成错误（与证据 / 常识 / 安全要求不符）」，在「人工判定」列填 有效 / 无效，并写回 `annotation.manual_review`。

| # | eval_id | type/subtype | 注入描述 | 注入后声明（截断） | 原始内容（截断） | 机器预览 | 人工判定 |
|---|---|---|---|---|---|---|---|
| 1 | D11-0162 | numeric_tamper/probability_swap | 概率互换：过载过热 50.0%↔绕组匝间短路 46.3% | 故障概率排序（前三）：1. 过载过热 46.3%；2. 绕组匝间短路 50.0%；3. 局部放电 2.7%。 | 故障概率排序（前三）：1. 过载过热 50.0%；2. 绕组匝间短路 46.3%；3. 局部放电 2.7%。 | 标记 DATA |  |
| 2 | D11-0188 | numeric_tamper/primary_probability | 主故障概率 57.6% → 74.9% | 主要故障推断为「过载过热」，后验概率 74.9%（置信等级：可能）。 | 主要故障推断为「过载过热」，后验概率 57.6%（置信等级：可能）。 | 标记 DATA |  |
| 3 | D11-0298 | numeric_tamper/anomaly_count | 异常点数 0 → 3 | 时序信号均值 3.70，标准差 1.18，3σ 检出 3 个异常点。 | 时序信号均值 3.70，标准差 1.18，3σ 检出 0 个异常点。 | 标记 DATA |  |
| 4 | D11-0158 | numeric_tamper/forecast_temperature | 预测油温 43.78℃ → 56.91℃（×1.3） | 未来 24 步油温预测均值 56.91℃，范围 43.77℃ ~ 43.99℃。 | 未来 24 步油温预测均值 43.78℃，范围 43.77℃ ~ 43.99℃。 | 标记 DATA |  |
| 5 | D11-0294 | numeric_tamper/primary_probability | 主故障概率 85.2% → 99.0% | 主要故障推断为「局部放电」，后验概率 99.0%（置信等级：确定）。 | 主要故障推断为「局部放电」，后验概率 85.2%（置信等级：确定）。 | 标记 DATA |  |
| 6 | D11-0295 | numeric_tamper/forecast_temperature | 预测油温 7.91℃ → 5.54℃（×0.7） | 未来 12 步油温预测均值 5.54℃，范围 7.89℃ ~ 8.06℃。 | 未来 12 步油温预测均值 7.91℃，范围 7.89℃ ~ 8.06℃。 | 标记 DATA |  |
| 7 | D11-0215 | numeric_tamper/forecast_temperature | 预测油温 43.79℃ → 56.93℃（×1.3） | 未来 12 步油温预测均值 56.93℃，范围 43.77℃ ~ 43.99℃。 | 未来 12 步油温预测均值 43.79℃，范围 43.77℃ ~ 43.99℃。 | 标记 DATA |  |
| 8 | D11-0264 | numeric_tamper/forecast_temperature | 预测油温 47.22℃ → 61.39℃（×1.3） | 未来 48 步油温预测均值 61.39℃，范围 47.03℃ ~ 47.22℃。 | 未来 48 步油温预测均值 47.22℃，范围 47.03℃ ~ 47.22℃。 | 标记 DATA |  |
| 9 | D11-0003 | numeric_tamper/probability_swap | 概率互换：过载过热 87.5%↔绕组匝间短路 6.4% | 故障概率排序（前三）：1. 过载过热 6.4%；2. 绕组匝间短路 87.5%；3. 绝缘老化 4.6%。 | 故障概率排序（前三）：1. 过载过热 87.5%；2. 绕组匝间短路 6.4%；3. 绝缘老化 4.6%。 | 标记 DATA |  |
| 10 | D11-0024 | numeric_tamper/anomaly_count | 异常点数 0 → 3 | 时序信号均值 7.29，标准差 5.13，3σ 检出 3 个异常点。 | 时序信号均值 7.29，标准差 5.13，3σ 检出 0 个异常点。 | 标记 DATA |  |
| 11 | D11-0015 | fake_reference/fabricated_span | kg 证据片段替换为编造文本 | 图谱关系链「Indicator:总烃 → SPECIFIED_IN → Standard:GB/T 7252」（支持文献 1，置信度 0.6 | 按《变压器油中溶解气体分析和判断导则》（GB7252-2001）中，运行中220kV及以下变压器油中溶解气体含量超过下列任一项值时应引起注意 | 标记 EVIDENCE |  |
| 12 | D11-0127 | fake_reference/nonexistent_kg_edge | kg 引用 kg:Indicator:总烃 → INDICATES → Fault:过热故障 → TREATED_BY  | 图谱关系链「Indicator:总烃 → INDICATES → Fault:过热故障 → TREATED_BY → Action:清洗」（ | kg:Indicator:总烃 → INDICATES → Fault:过热故障 → TREATED_BY → Action:清洗 | 标记 EVIDENCE |  |
| 13 | D11-0074 | fake_reference/nonexistent_kg_edge | kg 引用 kg:Fault:过热故障 → LOCATED_IN → Component:绕组 → 不存在的图谱边 | 图谱关系链「Fault:过热故障 → LOCATED_IN → Component:绕组」（支持文献 2，置信度 0.6）可用于解释当前征兆 | kg:Fault:过热故障 → LOCATED_IN → Component:绕组 | 标记 EVIDENCE |  |
| 14 | D11-0238 | fake_reference/nonexistent_chunk | kb 引用 kb:28e9dec0a50d-c0006 → 不存在的 kb:f205dadaef85-c0069 | 参考知识库《电力变压器常见故障分析及预防》：《电力变压器常见故障分析及预防》 2 变压器故障类型 > 2.3 套管的短路故障 > 2.3.1 | kb:28e9dec0a50d-c0006 | 标记 EVIDENCE |  |
| 15 | D11-0033 | fake_reference/fabricated_span | kg 证据片段替换为编造文本 | 图谱关系链「Symptom:噪声异常 → ←CAUSES → Fault:接触不良」（支持文献 1，置信度 0.6）可用于解释当前征兆的因果 | $\textcircled { 1 } 1 0 \mathrm { k V }$线路电气原件接触不良，引发变压器电压波动、变压器声音异常。 | 标记 EVIDENCE |  |
| 16 | D11-0141 | fake_reference/kb_as_detection | 知识库 / 历史案例片段冒充本次检测结果 | 本次检测结果显示：《电力变压器直流电阻不平衡故障分析》 3 变压器直阻不平衡案例分析 > 3.2 案例2
[表] 表2 直流电阻测试数据
／ | {"text": "参考知识库《电力变压器直流电阻不平衡故障分析》：《电力变压器直流电阻不平衡故障分析》 3 变压器直阻不平衡案例分析 >  | 标记 EVIDENCE |  |
| 17 | D11-0274 | fake_reference/kb_as_detection | 知识库 / 历史案例片段冒充本次检测结果 | 本次检测结果显示：《电力变压器常见故障分析》 二、电力变压器的常见故障分析 > ( 一) 变压器温度过高
在变压器的故障中，变压器温度过高是 | {"text": "参考知识库《电力变压器常见故障分析》：《电力变压器常见故障分析》 二、电力变压器的常见故障分析 > ( 一) 变压器温度 | 标记 EVIDENCE |  |
| 18 | D11-0189 | fake_reference/nonexistent_kg_edge | kg 引用 kg:Indicator:产气速率 → SPECIFIED_IN → Standard:GB/T 7252  | 图谱关系链「Indicator:产气速率 → SPECIFIED_IN → Standard:GB/T 7252」（支持文献 1，置信度 0 | kg:Indicator:产气速率 → SPECIFIED_IN → Standard:GB/T 7252 | 标记 EVIDENCE |  |
| 19 | D11-0089 | fake_reference/fabricated_span | tool 证据片段替换为编造文本 | 本次油中溶解气体检测数据：H2=228.2 ppm、CH4=312.3 ppm、C2H2=10.2 ppm、C2H4=72.8 ppm。 | {"dga_data": {"C2H2": 10.21, "C2H4": 72.85, "CH4": 312.28, "H2": 228.2 | 标记 EVIDENCE |  |
| 20 | D11-0296 | fake_reference/nonexistent_call | 工具引用 call:0 → 不存在的 call:9 | 三比值法编码（C2H2/C2H4、CH4/H2、C2H4/C2H6）：2-0-0。 | call:0 | 标记 EVIDENCE |  |
| 21 | D11-0059 | applicability_swap/dataset_swap | 数据集 ETTh2 → ETTm2 | 油温预测基于数据集 ETTm2（H 采样），回看 96 点、预测 12 步；历史油温均值 36.23℃，标准差 5.82℃。 | 油温预测基于数据集 ETTh2（H 采样），回看 96 点、预测 12 步；历史油温均值 36.23℃，标准差 5.82℃。 | 标记 APPLICABILITY |  |
| 22 | D11-0186 | applicability_swap/horizon_window_swap | 预测时间窗 6 步 → 12 步（与工具入参不符） | 未来 12 步油温预测均值 7.92℃，范围 7.89℃ ~ 8.06℃。 | 未来 6 步油温预测均值 7.92℃，范围 7.89℃ ~ 8.06℃。 | 标记 DATA |  |
| 23 | D11-0116 | applicability_swap/horizon_window_swap | 预测时间窗 24 步 → 48 步（与工具入参不符） | 未来 48 步油温预测均值 43.78℃，范围 43.77℃ ~ 43.99℃。 | 未来 24 步油温预测均值 43.78℃，范围 43.77℃ ~ 43.99℃。 | 标记 DATA |  |
| 24 | D11-0049 | applicability_swap/horizon_window_swap | 预测时间窗 6 步 → 12 步（与工具入参不符） | 未来 12 步油温预测均值 15.26℃，范围 15.12℃ ~ 15.29℃。 | 未来 6 步油温预测均值 15.26℃，范围 15.12℃ ~ 15.29℃。 | 标记 DATA |  |
| 25 | D11-0207 | applicability_swap/dataset_swap | 数据集 ETTm1 → ETTh1 | 油温预测基于数据集 ETTh1（15T 采样），回看 96 点、预测 96 步；历史油温均值 10.01℃，标准差 0.87℃。 | 油温预测基于数据集 ETTm1（15T 采样），回看 96 点、预测 96 步；历史油温均值 10.01℃，标准差 0.87℃。 | 标记 APPLICABILITY |  |
| 26 | D11-0261 | applicability_swap/horizon_window_swap | 预测时间窗 12 步 → 24 步（与工具入参不符） | 未来 24 步油温预测均值 39.10℃，范围 39.09℃ ~ 39.15℃。 | 未来 12 步油温预测均值 39.10℃，范围 39.09℃ ~ 39.15℃。 | 标记 DATA |  |
| 27 | D11-0151 | applicability_swap/device_swap | 设备号 #2主变 → #3主变（工具入参 device_id=#2主变） | （#3主变）三比值 / 特征气体规则命中：电弧放电（置信度 0.93）。 | 三比值 / 特征气体规则命中：电弧放电（置信度 0.93）。 | 标记 APPLICABILITY |  |
| 28 | D11-0212 | applicability_swap/dataset_swap | 数据集 ETTh2 → ETTm2 | 油温预测基于数据集 ETTm2（H 采样），回看 96 点、预测 12 步；历史油温均值 36.23℃，标准差 5.82℃。 | 油温预测基于数据集 ETTh2（H 采样），回看 96 点、预测 12 步；历史油温均值 36.23℃，标准差 5.82℃。 | 标记 APPLICABILITY |  |
| 29 | D11-0066 | applicability_swap/dataset_swap | 数据集 ETTm2 → ETTh2 | 油温预测基于数据集 ETTh2（15T 采样），回看 96 点、预测 48 步；历史油温均值 43.10℃，标准差 4.84℃。 | 油温预测基于数据集 ETTm2（15T 采样），回看 96 点、预测 48 步；历史油温均值 43.10℃，标准差 4.84℃。 | 标记 APPLICABILITY |  |
| 30 | D11-0180 | applicability_swap/horizon_window_swap | 预测时间窗 24 步 → 48 步（与工具入参不符） | 未来 48 步油温预测均值 40.38℃，范围 40.37℃ ~ 40.69℃。 | 未来 24 步油温预测均值 40.38℃，范围 40.37℃ ~ 40.69℃。 | 标记 DATA |  |
| 31 | D11-0076 | safety_premise_removed/no_action_contradiction | 高危故障检出（概率≥40%）却断言「无需处理、可继续运行」 | 综合以上分析，当前变压器状态无需处理，可继续运行观察，暂不安排停电检查。 | 参考知识库《一起大容量电力变压器局部放电故障分析与处理》：《一起大容量电力变压器局部放电故障分析与处理》
DOI：10． 3969 ／ j． | 标记 SAFETY |  |
| 32 | D11-0047 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 5 条 DGA 观测并移除对归因结果的引用 | 鉴于「局部放电」概率 79.4% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票、验 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 33 | D11-0026 | safety_premise_removed/no_action_contradiction | 高危故障检出（概率≥40%）却断言「无需处理、可继续运行」 | 综合以上分析，当前变压器状态无需处理，可继续运行观察，暂不安排停电检查。 | 参考知识库《500kV电力变压器内部过热故障分析及处理》：《500kV电力变压器内部过热故障分析及处理》 6 检修效果
大修完毕后对变压器油 | 标记 SAFETY |  |
| 34 | D11-0211 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 2 条 DGA 观测并移除对归因结果的引用 | 鉴于「绕组匝间短路」概率 79.3% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 35 | D11-0105 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 5 条 DGA 观测并移除对归因结果的引用 | 鉴于「局部放电」概率 48.6% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票、验 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 36 | D11-0007 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 5 条 DGA 观测并移除对归因结果的引用 | 鉴于「局部放电」概率 49.6% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票、验 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 37 | D11-0119 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 5 条 DGA 观测并移除对归因结果的引用 | 鉴于「局部放电」概率 68.6% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票、验 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 38 | D11-0078 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 5 条 DGA 观测并移除对归因结果的引用 | 鉴于「绕组匝间短路」概率 60.4% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 39 | D11-0193 | safety_premise_removed/remove_dga_premise | 保留「停电检查」建议，删去 5 条 DGA 观测并移除对归因结果的引用 | 鉴于「过载过热」概率 87.5% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；停电检修须执行工作票、验 | {"safety_evidence": [{"source": "tool", "ref": "call:0", "span": "{\"p | 标记 SAFETY |  |
| 40 | D11-0216 | safety_premise_removed/no_action_contradiction | 高危故障检出（概率≥40%）却断言「无需处理、可继续运行」 | 综合以上分析，当前变压器状态无需处理，可继续运行观察，暂不安排停电检查。 | 参考知识库《500kV电力变压器内部过热故障分析及处理》：《500kV电力变压器内部过热故障分析及处理》 6 检修效果
大修完毕后对变压器油 | 标记 SAFETY |  |