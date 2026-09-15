# C-3 补证与重规划路由验收报告

生成时间：2026-09-15 23:25:05；脚本 `scripts/eval/eval_route_c3.py`；seed=20260915。

## 1. 设置

- 样本：D10 `data/eval/d10/end2end_eval.jsonl` 按 scenario 分层抽 30 条；Planner 为 OraclePlanner（金标动作，不调 LLM）
- Generator：`output_mode=claims` 模板路径（规则声明）；首轮草案额外注入 4 条无 evidence 的声明，文本分别命中 `fault_attribution / ett_forecast / kg_search / rag_search` 四条补证建议规则
- Validator：`claim_check=route`，`claim_supplement_rounds=1`，`max_iterations=3`；对照组 `claim_check=check`（核查但不路由）
- 去重签名：工具名 + 规范化参数（`ett_forecast` 补默认 dataset/lookback/horizon，`fault_attribution` 忽略 `query`，剔除 None，键排序）

## 2. 主结果（route 模式）

| 指标 | 值 | 验收线 |
|---|---|---|
| 补证路由触发样本 | 28/30（93%） | > 0 |
| 补证调用总数 / 平均每样本 | 95 / 3.17 | - |
| 补证调用与首轮调用重复（同工具同参数） | 0 | 0 |
| 补证调用内部重复 | 0 | 0 |
| 全部工具调用重复数 | 0 | 0 |
| 补证调用成功数 | 95/95 | - |
| 最终 verdict 分布 | {'PASS': 30} | - |
| 首轮声明级核查 verdict | {'REVISION': 28, 'PASS': 2} | - |
| 平均 Generator 轮次 / 平均工具调用数 | 1.93 / 4.07 | - |
| 补证轮额外 Token / LLM 调用 | 0 / 0 | 无 LLM 时为 0 |
| 补证轮平均耗时（触发样本） | 29 ms | - |
| 工作流异常样本 | 0 | 0 |
| 平均总耗时 / 样本 | 0.04 s | - |

**验收判定：通过**（补证路由触发的调用中同工具同参数重复数为 0，且至少一条样本触发补证，无工作流异常）。

## 3. 补证工具分布与去重跳过原因

| 补证工具 | 调用数 |
|---|---|
| ett_forecast | 28 |
| fault_attribution | 26 |
| kg_search | 13 |
| rag_search | 28 |

| 跳过原因 | 次数 |
|---|---|
| duplicate_of_existing_call | 2 |
| no_entity_in_kg | 15 |

路径分布（Validator 路由目标序列）：

- `supplement>END`：28
- `END`：2

## 4. 对照：claim_check=check（核查不路由）

| 指标 | route | check |
|---|---|---|
| 补证触发样本 | 28 | 0 |
| 平均工具调用数 | 4.07 | 0.90 |
| 平均 Generator 轮次 | 1.93 | 1.93 |
| 最终 verdict | {'PASS': 30} | {'PASS': 30} |
| 最终声明证据来源 | {'tool': 228, 'kg': 42, 'kb': 87} | {'tool': 78, 'user': 6, 'kg': 16, 'kb': 21} |
| 平均耗时 | 0.04 s | 0.01 s |

说明：注入声明只出现在首轮，修订轮由规则重建，因此两种模式最终都会 PASS；差别在 route 模式为缺证声明补充了新的证据来源（kb / kg / ett 工具结果进入证据目录），check 模式仅删掉无据声明。

## 5. 样例（前 3 条）

### D10-0074（context_contrast）

- 首轮工具：fault_attribution
- 补证工具：ett_forecast, kg_search, rag_search；跳过：{'fault_attribution:duplicate_of_existing_call': 1}
- 路由：supplement → END；原因：['unsupported:4 claims need evidence', 'pass']
- 声明级核查：['REVISION', 'PASS']；最终 verdict=PASS，iteration=2

### D10-0107（single_tool_numeric）

- 首轮工具：fault_attribution
- 补证工具：fault_attribution, ett_forecast, kg_search, rag_search；跳过：无
- 路由：supplement → END；原因：['unsupported:4 claims need evidence', 'pass']
- 声明级核查：['REVISION', 'PASS']；最终 verdict=PASS，iteration=2

### D10-0112（no_tool）

- 首轮工具：无
- 补证工具：fault_attribution, ett_forecast, rag_search；跳过：{'kg_search:no_entity_in_kg': 1}
- 路由：supplement → END；原因：['unsupported:4 claims need evidence', 'pass']
- 声明级核查：['REVISION', 'PASS']；最终 verdict=PASS，iteration=2

## 6. 说明与局限

- 补证计划由规则构造（`workflow._build_supplement_plan`，不调 LLM）：按 `missing_evidence.suggested_tool` 选工具，`kg_search` 以最新归因的主故障名为 query，`fault_attribution` 复用上下文 DGA / evidence，`ett_forecast` 取上下文 dataset/horizon，`rag_search` 用 `suggested_query`。
- 去重覆盖两层：与本轮已执行调用同签名 → `duplicate_of_existing_call`；计划内部同签名 → `duplicate_in_plan`；无建议工具 → `no_tool`；模式外工具 → `tool_disabled`；`kg_search` 候选查询词（主故障名 / 建议词 / 用户问题）均无法在图谱中定位实体 → `no_entity_in_kg`（避免必然 no_match 的无效调用）。
- 未触发补证的样本为首轮规则声明较多（multi_tool）的场景：4 条注入声明占比未超过 `claim_unsupported_threshold=0.3`，声明级核查直接 PASS，属阈值设计内的行为。
- Planner `active` 模板已加入 `missing_evidence` 处理规则（LLM 重规划路径），但本报告验证的是规则补证路径；LLM 重规划的去重效果需启用 LLM 后在 C-5 一并评测。
- 每次 Validator 路由写入 `state.route_log`（触发原因 / 目标 / 补证工具 / 跳过 / 额外 Token / LLM 调用 / 耗时）。