# D10 端到端评测集数据卡片

- 文件：`data/eval/d10/end2end_eval.jsonl`
- 规模：196 条；随机种子 20260915
- 来源：D7/D8 封存 test 切分（task_seeds.test + multi_turn_seeds.test），按 sub_type 程序化分层抽样，未做任何改写
- 覆盖 group_key：71 个（同一来源派生的样本不会同时出现在训练集）
- 真实/合成：查询与金标均派生自真实文献块、真实图谱、真实 DGA/ETT 记录；不含 `data/synthetic`

## 场景分布

| 场景 | 条数 |
|---|---:|
| ask_user | 17 |
| context_contrast | 8 |
| error_recovery | 16 |
| kg_reasoning | 30 |
| multi_tool | 30 |
| no_tool | 15 |
| single_tool_fact | 40 |
| single_tool_numeric | 40 |

## 子类分布

| 场景/子类 | 条数 |
|---|---:|
| ask_user/ask_missing_params | 17 |
| context_contrast/dga_from_context | 8 |
| error_recovery/ett_bad_dataset | 5 |
| error_recovery/ett_bad_range | 2 |
| error_recovery/kg_colloquial_to_std | 9 |
| kg_reasoning/kg_1hop_alias | 10 |
| kg_reasoning/kg_1hop_canonical | 10 |
| kg_reasoning/kg_2hop | 10 |
| multi_tool/composite_2step | 15 |
| multi_tool/dga_kg | 5 |
| multi_tool/dga_rag | 5 |
| multi_tool/ett_anomaly | 5 |
| no_tool/capability | 2 |
| no_tool/common_sense | 8 |
| no_tool/greeting | 2 |
| no_tool/meta | 2 |
| no_tool/out_of_scope | 1 |
| single_tool_fact/A_title | 16 |
| single_tool_fact/B_caption | 14 |
| single_tool_fact/C_definition | 10 |
| single_tool_numeric/dga_attribution | 20 |
| single_tool_numeric/ett_forecast | 10 |
| single_tool_numeric/timeseries_anomaly | 10 |

## 标注字段说明

- `required_actions`：必要动作序列；`tool_call.key_params` 只列关键参数，其余参数不计分。
- `expected_recovery`：错误恢复样本的“首次错误调用 → 修正调用”对，评测 `错误恢复成功率`。
- `evidence_source`：期望证据来源（kb_chunk / kg_path / kg_entities / dga_label / ett_dataset / none）。
- `reference_points`：参考答案要点，`annotation.status=auto` 为脚本生成，人工复核后改为 `reviewed`。

## 已知偏差

- 查询仍为模板化措辞（P4-2 口语化改写需 LLM，尚未执行），对 Planner 偏乐观。
- DGA 场景的 `raw_label_cn` 是数据集标签，不代表真实故障部位，不能直接作为“正确答案”。
- ETT 负载特征单位未知，预测数值只做趋势判断。
