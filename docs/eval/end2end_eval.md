# 端到端评测：五级系统模式对比（E-1 / E-2）

- 评测集：`data/eval/d10/end2end_eval.jsonl`（D10，5 条，--limit 截断）
- 运行时间：2026-09-16 10:13；LLM：禁用（仅流程连通性）
- 显式覆盖：无（按模式默认；mode5 的 finetuned Planner 未配置时自动回退 baseline）
- 忠实度：离线证据锚点代理（非 LLM 裁判，见指标说明）

## 模式开关

| 模式 | 名称 | 工具 | 检索 | 反思 | 归因 | EIG | Planner | 策略 | Generator | Validator |
|---|---|---|---|---|---|---|---|---|---|---|
| mode1 | `llm_only` | 无 | — | off | expert | 关 | baseline | free | text | off |
| mode2 | `tool_base` | 五工具 | two_way | lexical | expert | 关 | baseline | free | text | off |
| mode3 | `active_plan` | 五工具 | two_way | lexical | calibrated | 开 | baseline | active | text | off |
| mode4 | `claim_verify` | 五工具 | two_way | lexical | calibrated | 开 | baseline | active | claims | route |
| mode5 | `dpo_planner` | 五工具 | two_way | lexical | calibrated | 开 | finetuned | active | claims | route |

## 主表

| 指标 | 模式 1：LLM 直答 | 模式 2：工具基座 | 模式 3：+ 主动规划 | 模式 4：+ 声明级验证 | 模式 5：+ DPO Planner（完整系统） |
|---|---:|---:|---:|---:|---:|
| 任务成功率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 动作正确率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 关键参数正确率 | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| 工具结果有效率 | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| 证据忠实度（0-1） | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 追问正确率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 错误恢复成功率 | — | — | — | — | — |
| 无依据结论率（首轮声明核查） | — | — | — | 0.0% | 0.0% |
| 平均问询次数 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 平均补证轮数 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 平均工具调用次数 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 不必要调用率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 平均延迟 (s) | 0.04 | 0.01 | 0.01 | 0.01 | 0.01 |
| 平均 Token | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 运行异常条数 | 0 | 0 | 0 | 0 | 0 |

## 分场景任务成功率

| 场景 | 模式 1：LLM 直答 | 模式 2：工具基座 | 模式 3：+ 主动规划 | 模式 4：+ 声明级验证 | 模式 5：+ DPO Planner（完整系统） |
|---|---:|---:|---:|---:|---:|
| ask_user | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| kg_reasoning | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| single_tool_fact | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Planner 计划状态分布

- 模式 1：LLM 直答：{'llm_disabled': 5}
- 模式 2：工具基座：{'llm_disabled': 5}
- 模式 3：+ 主动规划：{'no_tool': 5}
- 模式 4：+ 声明级验证：{'no_tool': 5}
- 模式 5：+ DPO Planner（完整系统）：{'no_tool': 5}

## 指标说明

- 任务成功率 = 动作正确 ∧ 关键参数正确 ∧ 工具结果有效 ∧ 答案忠实（四项同时满足）。
- 动作正确：必要工具全部被调用；ask_user/direct_answer 样本不得调用工具；ask_user 还需答案含追问措辞。
- 关键参数：只比对 D10 `key_params`；`query` 用词法重合 ≥ 0.5 判定，`dga_data` 数值容差 1e-6，`relations` 集合相等。
- 证据忠实度（离线代理）：kb 样本看金标文档标题/章节词与检索命中；kg 样本看期望实体出现；dga 样本看工具返回的主故障名是否被答案引用；ett 看预测数值/趋势词。该代理偏向词面匹配，正式报告需 LLM 裁判 + 10% 人工复核。
- 无依据结论率：Validator 声明级核查（mode4/5，或 `--validator-mode check|route`）首轮 `unsupported_ratio` 均值；v1 Validator 不产出该值（—）。
- 平均问询次数：active 策略（mode3+）的 `inquiry_rounds` 均值；D10 无隐藏征兆真值，追问一律按「无法提供」回答。
- 不必要调用率 = 非必要调用次数 / 总调用次数；模式外工具被 Retriever 拒绝执行（tool_disabled）也计入。
- Token 为 `src.utils.llm.USAGE` 累计的 prompt+completion；LLM 禁用时为 0。

## 已知局限

- LLM 禁用时 free 策略的 Planner 不产出计划，需要工具的样本均失败；active 策略回退 EIG 规则，仅在有 DGA 的样本上调用归因。此时只验证流程连通性与 no_tool/ask_user 的“不调用”口径。
- mode5 依赖 `llms.planner_finetuned`（百炼部署的 DPO 模型）；未部署时自动回退 baseline，mode5 与 mode4 等价。
- D10 查询未经口语化改写（A-2 rewrite 需 LLM），对 Planner 偏乐观。
