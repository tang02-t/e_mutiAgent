# C-1 声明级输出验收报告

生成时间：2026-09-15 22:33:47；脚本 `scripts/eval/eval_claims_c1.py`；seed=20260915。

## 1. 设置

- 样本：D10 `data/eval/d10/end2end_eval.jsonl` 按 scenario 分层抽 30 条
- Planner：OraclePlanner（金标动作，不调 LLM）；工具：fault_attribution / kg_search / ett_forecast / timeseries_anomaly 真实实现，rag_search 走本地 BM25 两路检索
- Generator：`output_mode=claims`；LLM 未启用（走模板回退 + 规则拼装路径）；声明来源分布：rule=30
- 校验：`validate_claims`（schema + 每条 claim ≥1 evidence）+ JSON round-trip；`ref` 与本轮证据目录比对

## 2. 主结果

| 指标 | 值 | 验收线 |
|---|---|---|
| JSON 合法率（schema_ok） | 100.0%（30/30） | ≥ 95% |
| 每条 claim ≥1 evidence 的声明占比 | 100.0%（110 条声明） | 100% |
| 样本级全部声明有证据 | 100.0% | - |
| 平均声明数 / 样本 | 3.67 | - |
| 平均证据数 / 声明 | 1.05 | - |
| ref 不在本轮证据目录的证据数 | 0 | 0 |
| safety 声明均有 tool 证据支撑的样本占比 | 100.0% | 100% |
| 工作流异常样本 | 0 | 0 |
| 平均耗时 / 样本 | 0.01 s | - |

**验收判定：通过**（JSON 合法率 ≥95% 且每条声明至少一条证据）。

## 3. 声明类型与证据来源分布

| type | 数量 |
|---|---|
| observation | 40 |
| inference | 43 |
| recommendation | 21 |
| safety | 6 |

| evidence.source | 数量 |
|---|---|
| tool | 73 |
| kb | 21 |
| kg | 16 |
| user | 6 |

## 4. 按场景

| scenario | n | schema_ok | 平均声明数 | 平均证据数 | 未知 ref |
|---|---|---|---|---|---|
| ask_user | 3 | 3 | 1.0 | 1.0 | 0 |
| context_contrast | 2 | 2 | 7.5 | 8.5 | 0 |
| error_recovery | 3 | 3 | 2.33 | 2.33 | 0 |
| kg_reasoning | 4 | 4 | 2.0 | 2.0 | 0 |
| multi_tool | 4 | 4 | 7.75 | 8.25 | 0 |
| no_tool | 3 | 3 | 1.0 | 1.0 | 0 |
| single_tool_fact | 6 | 6 | 3.0 | 3.0 | 0 |
| single_tool_numeric | 5 | 5 | 5.0 | 5.4 | 0 |

## 5. 样例（前 3 条）

### D10-0074（context_contrast，source=rule）

```json
{
  "claims": [
    {
      "id": "c1",
      "text": "本次油中溶解气体检测数据：H2=95.4 ppm、CH4=108.1 ppm、C2H2=0.0 ppm、C2H4=29.2 ppm、C2H6=193.4 ppm。",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"dga_data\": {\"C2H2\": 0.0, \"C2H4\": 29.2, \"C2H6\": 193.42, \"CH4\": 108.13, \"H2\": 95.42}}",
          "tool": "fault_attribution"
        }
      ]
    },
    {
      "id": "c2",
      "text": "DGA 特征解释：- 气体含量暂无超标，但需持续监测",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"dga_analysis\": {\"interpretation\": \"- 气体含量暂无超标，但需持续监测\"}}",
          "tool": "fault_attribution"
        }
      ]
    },
    {
      "id": "c3",
      "text": "三比值 / 特征气体规则命中：低温过热（150℃~300℃）（置信度 0.76）。",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"matched_rules\": [{\"confidence\": 0.76, \"rule_name\": \"低温过热（150℃~300℃）\"}]}",
          "tool": "fault_attribution"
        }
      ]
    },
    {
      "id": "c4",
      "text": "已排除的征兆（负观测）：H2_elevated, CH4_elevated, C2H2_elevated, C2H4_elevated, gas_rate_rapid。",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"evidence_negative\": [\"H2_elevated\", \"CH4_elevated\", \"C2H2_elevated\", \"C2H4_elevated\", \"gas_rate_rapid\"]}",
          "tool": "fault_attribution"
        }
      ]
    }
  ]
}
```

### D10-0107（single_tool_numeric，source=rule）

```json
{
  "claims": [
    {
      "id": "c1",
      "text": "本次油中溶解气体检测数据：H2=14.2 ppm、CH4=17.2 ppm、C2H2=0.0 ppm、C2H4=0.8 ppm、C2H6=4.9 ppm。",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"dga_data\": {\"C2H2\": 0.0, \"C2H4\": 0.83, \"C2H6\": 4.93, \"CH4\": 17.2, \"H2\": 14.2}}",
          "tool": "fault_attribution"
        }
      ]
    },
    {
      "id": "c2",
      "text": "DGA 特征解释：- 气体含量暂无超标，但需持续监测",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"dga_analysis\": {\"interpretation\": \"- 气体含量暂无超标，但需持续监测\"}}",
          "tool": "fault_attribution"
        }
      ]
    },
    {
      "id": "c3",
      "text": "三比值 / 特征气体规则命中：低温过热（150℃~300℃）（置信度 0.76）。",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"matched_rules\": [{\"confidence\": 0.76, \"rule_name\": \"低温过热（150℃~300℃）\"}]}",
          "tool": "fault_attribution"
        }
      ]
    },
    {
      "id": "c4",
      "text": "已排除的征兆（负观测）：H2_elevated, CH4_elevated, C2H2_elevated, C2H4_elevated, C2H6_elevated, TDCG_elevated, gas_rate_rapid。",
      "type": "observation",
      "evidence": [
        {
          "source": "tool",
          "ref": "call:0",
          "span": "{\"evidence_negative\": [\"H2_elevated\", \"CH4_elevated\", \"C2H2_elevated\", \"C2H4_elevated\", \"C2H6_elevated\", \"TDCG_elevated\", \"gas_rate_rapid\"]}",
          "tool": "fault_attribution"
        }
      ]
    }
  ]
}
```

### D10-0112（no_tool，source=rule）

```json
{
  "claims": [
    {
      "id": "c1",
      "text": "当前未获得任何工具结果或知识片段，无法给出有依据的诊断结论，建议补充 DGA 数据或设备信息。",
      "type": "observation",
      "evidence": [
        {
          "source": "user",
          "ref": "user:0",
          "span": "快速问个问题：你好"
        }
      ]
    }
  ]
}
```

## 6. 说明与局限

- 本报告默认在 LLM 未启用条件下产出，验证的是规则拼装路径（`build_rule_claims`）的 schema 合法性与证据覆盖，以及 claims 模式下工作流链路不断。
- LLM 路径（`system_claims.txt` 提示词 → JSON 解析 → `validate_claims`）的合法率需用 `--llm` 补跑，届时 `claims_source=llm` 占比与 `claims_json_valid` 即为该路径指标；解析失败会自动回退规则声明并计入 `state.errors`。
- OraclePlanner 结果只用于校验 Generator 输出规范，不代表任何系统模式的正式成绩。
- 规则声明的 `span` 为工具结果的 JSON 片段而非自然语言原文，C-2 确定性层按 JSON 子串 / 数值比对核查。