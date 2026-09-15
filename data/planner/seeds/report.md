# P4 任务种子（D6 种子 + 金标动作）构建报告

- 种子总数：3482（参数校验剔除 0，真实执行剔除 1）
- 五类占比（目标 fact/reasoning/numeric/no_tool/insufficient ≈ 35/15/30/10/10，composite 额外供 D7/D8）：
  - fact: 1207（37.2%，目标 35%）
  - reasoning: 403（12.4%，目标 15%）
  - numeric_tool: 952（29.4%，目标 30%）
  - no_tool: 340（10.5%，目标 10%）
  - insufficient: 340（10.5%，目标 10%）
  - composite: 240
- 动作类型分布：{'tool_call': 2642, 'direct_answer': 340, 'ask_user': 340, 'multi_tool': 160}
- 金标工具分布：{'rag_search': 1287, 'kg_search': 483, 'fault_attribution': 800, 'ett_forecast': 232, 'timeseries_anomaly': 160}
- 切分（按 group_key 分组，事实类沿用文献 split、推理类沿用图谱 split）：{'train': 2311, 'dev': 439, 'test': 732}
- 泄漏检查（train vs test 查询 bigram-Jaccard > 0.9）：2 对（已剔除测试侧）

## 子类分布
| 类别 | 子类 | 数量 |
|---|---|---|
| composite | dga_kg | 80 |
| composite | dga_rag | 80 |
| composite | ett_anomaly | 80 |
| fact | A_title | 698 |
| fact | B_caption | 400 |
| fact | C_definition | 109 |
| insufficient | ask_missing_params | 340 |
| no_tool | capability | 56 |
| no_tool | common_sense | 187 |
| no_tool | greeting | 25 |
| no_tool | meta | 55 |
| no_tool | out_of_scope | 17 |
| numeric_tool | dga_attribution | 520 |
| numeric_tool | dga_from_context | 120 |
| numeric_tool | ett_forecast | 152 |
| numeric_tool | timeseries_anomaly | 160 |
| reasoning | kg_1hop_alias | 152 |
| reasoning | kg_1hop_canonical | 152 |
| reasoning | kg_2hop | 99 |

## 真实执行失败样例（前 10）
- S-FAC-a3ef916d1b [fact/A_title] {'tool': 'rag_search', 'status': 'empty'}

## 说明
- 所有 query 为模板化文本，`needs_llm_rewrite=true`：需 LLM 做口语化、省略主语、术语/俗称混用等“模拟用户”改写（P4-2），改写后金标动作不变。
- `gold_actions` 中 `tool_call` 的 arguments 已通过 `validate_arguments_strict`；`--execute` 模式下还在真实工具环境执行并剔除业务失败样本。
- composite 类第 2 步的 kg_search/rag_search query 以原始标签占位，构造 D8 多轮样本时应替换为第 1 步真实返回的 primary_fault_name。
- 事实类 expected_hint.gold_chunk_ids、推理类 expected_entities 供后续端到端评测使用，不进入 Planner 训练标签。
- 数值类 raw_label 来自数据集原始标签（R4），与 fault_attribution 输出可能不一致，只作回答评估参考。