# Planner SFT 数据卡片（模板阶段，LLM 扩写前）

- 来源：`data/planner/seeds/task_seeds.jsonl`（P4 任务种子，金标动作已参数校验 + 真实执行）
- 系统提示：与线上 `PLANNER_SYSTEM_PROMPT()` 一致（含工具清单）；user 为 `render_planner_user` 渲染，context 走 `PlannerAgent._render_context`
- 格式：ms-swift messages+tools（`swift_*.jsonl`）、LLaMA-Factory function-calling（`lf_*.json`）、JSON 文本规划（`jsontext_*.jsonl`）
- 切分：按 `group_key` 分组，train/dev/test 互不共享来源；test 封存
- 已知偏差：问法为模板生成，多样性不足（待 P4-2 LLM 口语化扩写）；数值类 DGA 记录来自 3 个公开数据集，标签分布不均（过载过热/正常偏多）；图谱推理类受规则抽取图谱覆盖限制（90 节点/190 边）
- 许可：文献数据仅用于内部研究；ETT 数据集 CC BY 4.0；DGA 数据集见 data/real/dga 来源说明

## 规模
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