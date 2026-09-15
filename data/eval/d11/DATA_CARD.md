# D11 故障注入评测集数据卡片

- 文件：`data/eval/d11/fault_injection_eval.jsonl`；构建脚本 `scripts/eval/build_d11_fault_injection.py`；seed=20260915
- 规模：300 条 = 200 条注入（四类各 50）+ 100 条干净负样本
- 底稿：D10 `end2end_eval.jsonl` 196 条，经 OraclePlanner（金标动作）+ 真实工具（fault_attribution / kg_search / ett_forecast / timeseries_anomaly / 本地 BM25）+ `build_rule_claims` 规则声明生成；不调 LLM，可复现
- 为何不用 `oracle_mode5.jsonl`：该文件只保存 `answer_head`（前 300 字），无完整声明与工具结果，无法注入与复跑；本集在同一 D10 底稿上重新生成等价的干净草案
- 真实/合成：查询、工具结果、知识片段、图谱路径均为真实数据；错误由脚本注入，属受控合成

## 注入类型

| type | 中文 | 子类（数量） | 期望约束 | 条数 | 底稿数 |
|---|---|---|---|---:|---:|
| numeric_tamper | 篡改数值 | anomaly_count(10)、forecast_temperature(10)、gas_concentration(10)、primary_probability(10)、probability_swap(10) | DATA | 50 | 50 |
| fake_reference | 伪造引用 | fabricated_span(10)、kb_as_detection(10)、nonexistent_call(10)、nonexistent_chunk(10)、nonexistent_kg_edge(10) | EVIDENCE | 50 | 50 |
| applicability_swap | 换设备/换数据集 | dataset_swap(21)、device_swap(9)、horizon_window_swap(20) | APPLICABILITY | 50 | 31 |
| safety_premise_removed | 删安全前提 | no_action_contradiction(12)、remove_dga_premise(38) | SAFETY | 50 | 43 |

子类说明：

- `gas_concentration`：DGA 观测声明中某气体浓度 ×0.7 或 ×1.3；`probability_swap`：排序声明前两名概率互换；`primary_probability`：主故障后验概率 ×0.7/×1.3；`forecast_temperature`：预测油温均值 ×0.7/×1.3；`anomaly_count`：3σ 异常点数 +3。
- `nonexistent_chunk` / `nonexistent_kg_edge` / `nonexistent_call`：ref 改为本轮不存在的 chunk id / 图谱边 / 调用号；`fabricated_span`：span 替换为编造文本；`kb_as_detection`：知识库推荐改写为「本次检测结果显示：…」的 observation。
- `dataset_swap`：ETTh1↔ETTm1、ETTh2↔ETTm2；`horizon_window_swap`：预测步数与工具入参错位（时间窗不匹配，期望 APPLICABILITY，确定性层按 DATA/horizon 检出）；`device_swap`：用户问题含设备号时，把设备号写入所引工具入参 `device_id`，声明中写成另一台。
- `remove_dga_premise`：保留 safety「停电检查」建议，删去支撑它的 DGA 观测声明并把其引用换成知识片段 / 用户原话；`no_action_contradiction`：高危故障检出（概率≥40%）却写「无需处理、可继续运行」；`low_risk_shutdown`：归因为低风险却新增「立即停电吊罩」建议且无检测引用。

## 干净负样本

- 100 条，底稿 100 个（优先选未被注入使用的底稿）；100 条含工具 / 知识 / 图谱证据，其余为 no_tool / ask_user 场景的信息不足声明

## 场景分布

| 场景 | 条数 |
|---|---:|
| context_contrast | 19 |
| error_recovery | 28 |
| kg_reasoning | 32 |
| multi_tool | 87 |
| single_tool_fact | 46 |
| single_tool_numeric | 88 |

## 字段说明

- `injected` / `type` / `subtype` / `expected_constraints`：是否注入、类型、子类、期望被哪类约束检出（C-5 分类型检出率据此统计）。
- `location {claim_id, field}` / `original` / `injection_detail`：注入位置、被替换的原始内容、注入描述。
- `claims`：注入后的声明列表（schema 见 `docs/claim_schema.md`）；`draft_answer`：模板正文 + 声明清单，供 v1 Validator 评分。
- `snapshot {user_query, context, tool_calls, retrieved_knowledge, inquiry_log}`：离线重建 `AgentState` 所需的全部证据，`eval_validator.py` 据此复跑 v1 / v2_check / v2_route。
- `checker_preview`：构建时用 C-2 确定性层对该记录的预览（`target_flagged` 为注入声明是否被标记），仅供核对注入是否可被机器识别，不是验收结果。
- `annotation.status=auto`：人工抽检 40 条（`manual_review_sample.md`）确认注入构成错误后改为 `reviewed`。

## 构建时确定性层预览（非验收）

| type | 注入声明被标记 | 命中期望约束 |
|---|---:|---:|
| numeric_tamper | 50/50 | 50/50 |
| fake_reference | 50/50 | 50/50 |
| applicability_swap | 50/50 | 50/50 |
| safety_premise_removed | 50/50 | 50/50 |
| clean（误报） | 0/100 | - |

## 已知偏差

- 底稿声明为规则拼装（非 LLM 生成），措辞模板化，注入错误的隐蔽性低于真实 LLM 幻觉；C-5 结论应注明此限制。
- 每条只注入一处错误，未覆盖多处并发错误与跨声明矛盾。
- `device_swap` 为构造设备号进入工具入参（D10 工具调用原本不含 `device_id`），属受控设定。
- `low_risk_shutdown` 子类在 D10 底稿上候选为 0：D10 的 DGA 样本主故障严重度均为「高风险 / 中等风险」，不存在「低风险」归因，该子类保留在脚本中待补充低风险样本后启用；`applicability_swap` 与 `safety_premise_removed` 因可用底稿（含 ett_forecast 22 条、设备号 9 条、safety 声明 43 条）不足 50，同一底稿会被不同子类复用（分别 31 / 43 个底稿）。
- 人工抽检尚未执行（`annotation.status=auto`），注入有效率以人工复核为准。