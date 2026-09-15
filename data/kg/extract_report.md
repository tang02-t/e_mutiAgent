# P2-2 规则三元组抽取报告

- 句子总数：18905；含触发词：10958；规则命中：318
- 唯一三元组：206（≥2 篇文献支持：77）
- 节点：95  按类型：{'Condition': 12, 'Fault': 20, 'Method': 15, 'Symptom': 9, 'Component': 11, 'Action': 7, 'Indicator': 16, 'Standard': 5}
- 关系分布（去重前计数）：{'CAUSES': 118, 'INDICATES': 14, 'LOCATED_IN': 106, 'DETECTED_BY': 67, 'PRODUCES': 12, 'SPECIFIED_IN': 60, 'TREATED_BY': 27}
- 待 LLM 抽取候选句：1649

## 支持度最高的 25 条

| head | relation | tail | support |
|---|---|---|---:|
| 过热故障 | LOCATED_IN | 铁心 | 12 |
| 绝缘受潮 | LOCATED_IN | 绕组 | 10 |
| 渗漏油 | TREATED_BY | 补焊 | 9 |
| 铁心多点接地 | CAUSES | 过热故障 | 8 |
| 分接开关故障 | LOCATED_IN | 有载分接开关 | 6 |
| 渗漏油 | LOCATED_IN | 套管 | 6 |
| 局部放电 | DETECTED_BY | 在线监测 | 5 |
| 绝缘受潮 | LOCATED_IN | 绝缘油 | 5 |
| 相间短路 | LOCATED_IN | 绕组 | 5 |
| 高能放电 | DETECTED_BY | 三比值法 | 4 |
| 过热故障 | DETECTED_BY | 三比值法 | 4 |
| 三比值法 | SPECIFIED_IN | DL/T 722 | 4 |
| 局部放电 | DETECTED_BY | 油色谱分析 | 4 |
| 渗漏油 | TREATED_BY | 停运 | 4 |
| 过热故障 | LOCATED_IN | 固体绝缘 | 4 |
| 进水受潮 | CAUSES | 绝缘受潮 | 4 |
| 绝缘受潮 | CAUSES | 过热故障 | 4 |
| 匝间短路 | LOCATED_IN | 绕组 | 4 |
| 渗漏油 | CAUSES | 绝缘受潮 | 4 |
| 悬浮放电 | CAUSES | 低能放电 | 4 |
| 铁心多点接地 | LOCATED_IN | 铁心 | 3 |
| 三比值法 | SPECIFIED_IN | GB/T 7252 | 3 |
| 局部放电 | DETECTED_BY | 局放测量 | 3 |
| 直流电阻 | SPECIFIED_IN | DL/T 596 | 3 |
| 渗漏油 | CAUSES | 油位异常 | 3 |

## 实体命中 Top 30

| type | entity | count |
|---|---|---:|
| Component | 绕组 | 1327 |
| Component | 铁心 | 735 |
| Component | 绝缘油 | 551 |
| Component | 套管 | 428 |
| Fault | 过热故障 | 358 |
| Fault | 渗漏油 | 342 |
| Component | 引线 | 302 |
| Fault | 放电故障 | 295 |
| Method | 油色谱分析 | 263 |
| Fault | 局部放电 | 249 |
| Component | 油箱 | 242 |
| Component | 有载分接开关 | 238 |
| Method | 在线监测 | 223 |
| Action | 停运 | 208 |
| Method | 故障树分析 | 207 |
| Method | 神经网络 | 185 |
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
| Method | 支持向量机 | 102 |
| Fault | 匝间短路 | 101 |
| Condition | 过负荷 | 94 |
| Condition | 突发短路 | 91 |