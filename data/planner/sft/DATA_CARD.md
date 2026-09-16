# Planner SFT 数据卡片（模板阶段，LLM 扩写前）

- 来源：`data/planner/seeds/task_seeds.jsonl`（P4 任务种子，金标动作已参数校验 + 真实执行）；多轮/错误恢复：`data/planner/seeds/multi_turn_seeds.jsonl`（工具返回全部来自真实执行）
- 系统提示：与线上 `PLANNER_SYSTEM_PROMPT()` 一致（含工具清单）；user 为 `render_planner_user` 渲染，context 走 `PlannerAgent._render_context`
- 格式：ms-swift messages+tools（`swift_*.jsonl`）、LLaMA-Factory function-calling（`lf_*.json`）、JSON 文本规划（`jsontext_*.jsonl`）；多轮为 ms-swift messages（含 `tool` 角色，`swift_multiturn_*.jsonl`），每个 assistant 决策点一条样本；百炼模型调优 SFT（`bailian_{train,dev}.jsonl`）= 单轮 + 多轮合并，剥离 `meta` / tool 消息 `name`，`tool_calls[].id` 与 `tool_call_id` 一一对应，`arguments` 为 JSON 字符串，导出时已通过 `bailian_format.validate_bailian_file`；test 不导出百炼文件
- 切分：按 `group_key` 分组，train/dev/test 互不共享来源；test 封存
- 已知偏差：问法为模板生成，多样性不足（待 P4-2 LLM 口语化扩写）；数值类 DGA 记录来自 3 个公开数据集，标签分布不均（过载过热/正常偏多）；图谱推理类受规则抽取图谱覆盖限制（90 节点/190 边）；多轮样本中 assistant 的 thought 为规则模板文本
- 错误恢复样本：刻意错误的首次调用只保留在上下文中，不导出为训练目标（decision_index 从 1 起）；tool 返回为真实执行结果
- 许可：文献数据仅用于内部研究；ETT 数据集 CC BY-ND 4.0（以原仓库为准）；DGA 数据集见 data/real/dga 来源说明

## 规模（单轮）
| split | category | n |
|---|---|---|
| dev | composite | 27 |
| dev | fact | 144 |
| dev | insufficient | 33 |
| dev | no_tool | 57 |
| dev | numeric_tool | 109 |
| dev | reasoning | 69 |
| test | composite | 26 |
| test | fact | 263 |
| test | insufficient | 17 |
| test | no_tool | 96 |
| test | numeric_tool | 164 |
| test | reasoning | 166 |
| train | composite | 187 |
| train | fact | 800 |
| train | insufficient | 290 |
| train | no_tool | 187 |
| train | numeric_tool | 679 |
| train | reasoning | 168 |

## 规模（多轮决策点样本）
| split | category | sub_type | n |
|---|---|---|---|
| dev | error_recovery | ett_bad_dataset | 6 |
| dev | error_recovery | kg_colloquial_to_std | 6 |
| dev | multi_turn | composite_2step | 42 |
| dev | multi_turn | composite_recover_kg | 4 |
| dev | multi_turn | single_then_finish | 86 |
| test | error_recovery | ett_bad_dataset | 10 |
| test | error_recovery | ett_bad_range | 4 |
| test | error_recovery | kg_colloquial_to_std | 18 |
| test | multi_turn | composite_2step | 57 |
| test | multi_turn | single_then_finish | 114 |
| train | error_recovery | ett_bad_dataset | 20 |
| train | error_recovery | ett_bad_range | 20 |
| train | error_recovery | kg_colloquial_to_std | 36 |
| train | error_recovery | ts_unrecoverable_ask | 9 |
| train | multi_turn | composite_2step | 357 |
| train | multi_turn | composite_recover_kg | 28 |
| train | multi_turn | single_then_finish | 400 |

## 规模（百炼 SFT 上传文件，单轮 + 多轮合并）
| split | 单轮 | 多轮决策点 | 合计 |
|---|---|---|---|
| train | 2311 | 870 | 3181 |
| dev | 439 | 144 | 583 |