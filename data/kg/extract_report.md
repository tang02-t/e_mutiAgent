# P2-2 规则三元组抽取报告

- 句子总数：18905；含触发词：10958；规则命中：317
- 唯一三元组：190（≥2 篇文献支持：70）
- 节点：90  按类型：{'Method': 14, 'Component': 11, 'Condition': 11, 'Standard': 5, 'Indicator': 14, 'Fault': 20, 'Symptom': 8, 'Action': 7}
- 关系分布（去重前计数）：{'INDICATES': 13, 'LOCATED_IN': 105, 'DETECTED_BY': 69, 'CAUSES': 113, 'SPECIFIED_IN': 38, 'TREATED_BY': 27, 'PRODUCES': 12}
- 待 LLM 抽取候选句：1660

## 支持度最高的 25 条

| head | relation | tail | support |
|---|---|---|---:|
| 过热故障 | LOCATED_IN | 铁心 | 12 |
| 绝缘受潮 | LOCATED_IN | 绕组 | 10 |
| 铁心多点接地 | CAUSES | 过热故障 | 9 |
| 渗漏油 | TREATED_BY | 补焊 | 9 |
| 分接开关故障 | LOCATED_IN | 有载分接开关 | 6 |
| 渗漏油 | LOCATED_IN | 套管 | 6 |
| 局部放电 | DETECTED_BY | 在线监测 | 5 |
| 绝缘受潮 | LOCATED_IN | 绝缘油 | 5 |
| 过热故障 | DETECTED_BY | 油色谱分析 | 5 |
| 相间短路 | LOCATED_IN | 绕组 | 5 |
| 高能放电 | DETECTED_BY | 三比值法 | 4 |
| 过热故障 | DETECTED_BY | 三比值法 | 4 |
| 渗漏油 | TREATED_BY | 停运 | 4 |
| 过热故障 | LOCATED_IN | 固体绝缘 | 4 |
| 进水受潮 | CAUSES | 绝缘受潮 | 4 |
| 绝缘受潮 | CAUSES | 过热故障 | 4 |
| 匝间短路 | LOCATED_IN | 绕组 | 4 |
| 渗漏油 | CAUSES | 绝缘受潮 | 4 |
| 接触不良 | CAUSES | 过热故障 | 4 |
| 悬浮放电 | CAUSES | 低能放电 | 4 |
| 铁心多点接地 | LOCATED_IN | 铁心 | 3 |
| 三比值法 | SPECIFIED_IN | GB/T 7252 | 3 |
| 局部放电 | DETECTED_BY | 油色谱分析 | 3 |
| 局部放电 | DETECTED_BY | 局放测量 | 3 |
| 直流电阻 | SPECIFIED_IN | DL/T 596 | 3 |

## 实体命中 Top 30

| type | entity | count |
|---|---|---:|
| Component | 绕组 | 1327 |
| Component | 铁心 | 735 |
| Component | 绝缘油 | 551 |
| Component | 套管 | 428 |
| Method | 油色谱分析 | 400 |
| Fault | 过热故障 | 358 |
| Fault | 渗漏油 | 342 |
| Component | 引线 | 302 |
| Fault | 放电故障 | 295 |
| Fault | 局部放电 | 253 |
| Component | 油箱 | 242 |
| Component | 有载分接开关 | 239 |
| Method | 故障树分析 | 227 |
| Method | 在线监测 | 223 |
| Action | 停运 | 208 |
| Method | 神经网络 | 185 |
| Method | 支持向量机 | 184 |
| Component | 冷却器 | 150 |
| Component | 固体绝缘 | 138 |
| Indicator | 微水 | 125 |
| Method | 三比值法 | 121 |
| Fault | 绝缘受潮 | 116 |
| Fault | 铁心多点接地 | 115 |
| Indicator | 直流电阻 | 106 |
| Fault | 过负荷 | 106 |
| Fault | 接触不良 | 104 |
| Indicator | 绝缘电阻 | 102 |
| Fault | 匝间短路 | 101 |
| Condition | 过负荷 | 94 |
| Indicator | 氢气 | 92 |