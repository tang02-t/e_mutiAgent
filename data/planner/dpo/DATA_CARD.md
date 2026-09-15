# D13 偏好对数据卡片

- 生成时间：2026-09-16 01:06:59；脚本 `scripts/planner_data/score_candidates.py`
- 候选来源：金标扰动（synthetic-demo，仅用于验证打分 / 配对 / 导出管线，**不用于训练**）
- prompt 数 60，候选数 280，真实工具执行（rag=local_kb:two_way），耗时 2.37 s，Token 0
- 打分：format / tool / schema / business / faith(1 − unsupported_ratio，Generator 规则声明 + ClaimChecker) / ask；总分 = 加权和 / 权重和 − 0.1 × 多余调用（下限 0）
- 配对：同 prompt 最高 vs 最低，分差 ≥ 0.3，每 prompt ≤ 1 对；D13-exec 按 4 项 exec 分，D13-full 按 6 项 + 成本
- 导出：百炼 DPO `messages`（system + user，以 user 结尾）+ `chosen` / `rejected`（assistant，风格 `tool_calls`）+ `tools`；每条经 `validate_bailian_record` 复核
- 泄漏检查：prompt 的 `seed_source` / `group_key` 不出现在 D8 test 与 D10，且 split=train

| 子集 | 偏好对 | 平均分差 | 非法记录 | 泄漏 |
|---|---|---|---|---|
| D13-exec | 50 | 1.0 | 0 | 0 / 50 |
| D13-full | 60 | 0.9222 | 0 | 0 / 60 |

候选来源平均 full 分：

- `gold`：1.0
- `perturb:bad_format`：0.0
- `perturb:bad_schema`：0.6667
- `perturb:extra_calls`：0.6055
- `perturb:no_ask`：0.8333
- `perturb:spurious_call`：0.6417
- `perturb:wrong_tool`：0.5333

后置：候选生成需 M1（D-1）与 M0 采样；人工抽 100 对复核偏好方向正确率 ≥ 90%（PLAN D-2）。
