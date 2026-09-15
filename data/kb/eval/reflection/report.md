# 反思评分校验报告（P3-3 / D9）

- 样本对：298；桶分布：{'retrieved_nongold': 105, 'same_doc_random': 43, 'global_random': 45, 'gold': 105}
- 已人工标注：0

## 评分器输出分布（按采样桶）
| 桶 | n | lexical=0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| gold | 105 | 6 | 4 | 14 | 81 |
| retrieved_nongold | 105 | 3 | 10 | 21 | 71 |
| same_doc_random | 43 | 20 | 8 | 8 | 7 |
| global_random | 45 | 40 | 5 | 0 | 0 |

## 说明
- 尚无人工标注。请在 `data/kb/eval/reflection/d9_pairs.jsonl` 中填写 `human_score`（0-3，标准见 src/agents/reflection.py SCORE_RUBRIC），两人标注时第二人填 `human_score_2`，争议由第三人填 `adjudicated`。
- `expected_hint` 只是采样桶的先验提示，不是答案；标注时请以 0-3 标准独立判断。