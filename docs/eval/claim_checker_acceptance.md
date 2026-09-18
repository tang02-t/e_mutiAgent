# C-2 声明级约束核查器验收报告（确定性层）

生成时间：2026-09-15 22:50:07；脚本 `scripts/eval/eval_claim_checker_c2.py`。

## 1. 构造样例（四类约束 × 5 违反 + 5 合规）

| 约束 | 违反样例 | 检出 | 合规样例 | 误报 |
|---|---|---|---|---|
| DATA | 5 | 5 | 5 | 0 |
| EVIDENCE | 5 | 5 | 5 | 0 |
| APPLICABILITY | 5 | 5 | 5 | 0 |
| SAFETY | 5 | 5 | 5 | 0 |
| 合计 | 20 | 20（100%） | 20 | 0 |

**验收判定（确定性层）：通过**（要求检出率 100%、误报 0）。样例定义见 [tests/test_c2_claim_checker.py](../../tests/test_c2_claim_checker.py)，正反例与 `docs/design/claim_schema.md` §4 对应。

## 2. D10 全量干净声明误报率

- 设置：D10 196 条，OraclePlanner + 真实工具（rag=local_kb:two_way），Generator 无 LLM → `build_rule_claims` 规则声明（视为干净样本）
- 声明总数 817，未通过 0（声明级误报率 0.00%），涉及样本 0（样本级 0.00%）
- 约束违反计数：{'DATA': 0, 'EVIDENCE': 0, 'APPLICABILITY': 0, 'SAFETY': 0}；核查结论分布：{'PASS': 196}
- 耗时 1.54 s

## 3. 判定规则实现

- 任一 `SAFETY` 或 `DATA` 违反 → `REVISION`；`unsupported_ratio`（无依据 / 矛盾 / EVIDENCE 违反声明占比）> 0.3 → `REVISION`；无声明 → `REVISION`
- `iteration ≥ claim_abstain_after`（默认 3，即两轮修订后）仍不通过 → `ABSTAIN`，Validator 输出「证据不足，建议补充 X」并结束
- `ValidationResult` v2 新增 `claim_verdicts / unsupported_ratio / violation_counts / claim_check_verdict / missing_evidence`，旧字段保留；`claim_check=off` 时行为与 v1 完全一致

## 4. 语义层（后置）

- `ClaimChecker._semantic_layer` 对通过确定性层的 `inference` 声明做 NLI（entail / contradict / unsupported），提示词仅含该声明与其引用片段；LLM 未启用时自动跳过。
- 验收「30 条人工标注声明一致率 ≥ 85%」需启用 LLM 并人工标注，后置到 C-5 阶段与人工项一并完成。