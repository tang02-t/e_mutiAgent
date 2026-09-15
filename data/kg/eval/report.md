# 图谱评测报告（P2-5 离线部分）

- 图谱规模：90 节点 / 190 边；min_support=1
- 关系问答种子：304 条（split 分布：{'train': 116, 'dev': 48, 'test': 140}；方向：{'out': 190, 'in': 114}）
- 精度抽检样本：56 条（已标注 0 条）

## kg_search 自动评测（工具对图谱的忠实度）
- 实体定位率：100.0%
- 一跳命中率 HitAny：100.0%（至少一个期望实体被检回）
- 一跳命中率 HitAll：100.0%（全部期望实体被检回）

| 关系 | n | HitAny | HitAll |
|---|---|---|---|
| CAUSES | 98 | 100.0% | 100.0% |
| DETECTED_BY | 52 | 100.0% | 100.0% |
| INDICATES | 18 | 100.0% | 100.0% |
| LOCATED_IN | 50 | 100.0% | 100.0% |
| PRODUCES | 22 | 100.0% | 100.0% |
| SPECIFIED_IN | 38 | 100.0% | 100.0% |
| TREATED_BY | 26 | 100.0% | 100.0% |

| split | n | HitAny | HitAll |
|---|---|---|---|
| dev | 48 | 100.0% | 100.0% |
| test | 140 | 100.0% | 100.0% |
| train | 116 | 100.0% | 100.0% |

| 问法 | n | HitAny | HitAll |
|---|---|---|---|
| alias | 152 | 100.0% | 100.0% |
| canonical | 152 | 100.0% | 100.0% |

## 三元组精度（人工抽检）
- 人工尚未标注。请在 `data/kg/eval/precision_sample.jsonl` 的 `label` 字段填 correct/wrong/unsure 后重跑本脚本。

### AI 预标注（待人工复核，不作为正式结论）
- AI 预标注（抽样时全部样本）：56 条，correct=42 wrong=6 unsure=8
- **精度（correct / 已标注）：75.0%**；严格精度（unsure 计为错）同上；宽松精度（unsure 计为对）：89.3%

| 关系 | 标注数 | correct | wrong | 精度 |
|---|---|---|---|---|
| CAUSES | 8 | 7 | 0 | 87.5% |
| DETECTED_BY | 8 | 5 | 1 | 62.5% |
| INDICATES | 8 | 7 | 1 | 87.5% |
| LOCATED_IN | 8 | 6 | 1 | 75.0% |
| PRODUCES | 8 | 8 | 0 | 100.0% |
| SPECIFIED_IN | 8 | 3 | 3 | 37.5% |
| TREATED_BY | 8 | 6 | 0 | 75.0% |

| 支持度分桶 | 标注数 | 精度 |
|---|---|---|
| support=1 | 32 | 68.8% |
| support>=2 | 24 | 83.3% |

规则修订后已从图谱剔除的抽样边：9 条（correct=3 wrong=4 unsure=2）

- AI 预标注（仅当前图谱仍保留的边）：47 条，correct=39 wrong=2 unsure=6
- **精度（correct / 已标注）：83.0%**；严格精度（unsure 计为错）同上；宽松精度（unsure 计为对）：95.7%

| 关系 | 标注数 | correct | wrong | 精度 |
|---|---|---|---|---|
| CAUSES | 8 | 7 | 0 | 87.5% |
| DETECTED_BY | 8 | 5 | 1 | 62.5% |
| INDICATES | 6 | 6 | 0 | 100.0% |
| LOCATED_IN | 7 | 6 | 0 | 85.7% |
| PRODUCES | 6 | 6 | 0 | 100.0% |
| SPECIFIED_IN | 4 | 3 | 1 | 75.0% |
| TREATED_BY | 8 | 6 | 0 | 75.0% |

| 支持度分桶 | 标注数 | 精度 |
|---|---|---|
| support=1 | 24 | 79.2% |
| support>=2 | 23 | 87.0% |

预标注为 wrong 的样例（提示规则抽取的典型错误模式）：
- KGP-008 局部放电 --DETECTED_BY(检测手段)--> 红外测温（s=2）：枚举句误配：红外测温不用于检测局部放电
- KGP-010 振动分析 --SPECIFIED_IN(依据标准)--> DL/T 911（s=1）：DL/T 911 为频响法标准，证据句未提及振动法标准
- KGP-028 氢气 --SPECIFIED_IN(依据标准)--> DL/T 722（s=3）：证据句依据为 GB/T 7252，抽取错配到 DL/T 722
- KGP-031 微水 --INDICATES(指示)--> 绝缘老化（s=1）：证据讲油色变暗，与微水无关
- KGP-044 渗漏油 --LOCATED_IN(部位)--> 绝缘油（s=1）：渗漏油的部位为油箱/密封处，'绝缘油'非部位
- KGP-050 低电压短路阻抗法 --SPECIFIED_IN(依据标准)--> DL/T 911（s=1）：短路阻抗法对应 DL/T 1093，非 DL/T 911


## 说明
- 自动评测的期望答案来自图谱本身，仅衡量 kg_search 的实体定位与检索忠实度，不衡量图谱正确性。
- 图谱正确性由人工抽检精度衡量；LLM 抽取阶段（P2 阶段 2）完成后应重跑并对比 rule-only 与 rule+llm。