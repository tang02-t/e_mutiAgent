# D8 test 切分封存记录（SEALED）

封存日期：2026-09-15。封存对象为 Planner 训练数据 D8 的 test 切分。在 D-1 / D-3 训练与 D-4 评测完成前，任何人不得读取、抽样或据此调参；评测脚本（`eval_planner_offline.py`）只能在训练结束后一次性读取。

规则：
- 百炼上传文件只导出 `bailian_train.jsonl` / `bailian_dev.jsonl`，`export_sft.py` 不产出 `bailian_test.jsonl`（`tests/test_d1_bailian_export.py` 断言）。
- 切分按 `group_key` 分组，test 与 train / dev 互不共享来源；泄漏检查见 `build_task_seeds.py`。
- 种子级哈希以 test 切分的原始行为准（与系统提示版本无关）；导出文件哈希会随 `PLANNER_SYSTEM_PROMPT` 变更而变化，重导出后仅需核对种子级哈希不变。
- 若种子级哈希发生变化，必须在本文件追加一条记录说明原因（如 A-2 口语化改写重导出），并保证 test 的 `seed_id` 集合与 `gold_actions` 不变。

| 对象 | sha256 |
|---|---|
| `seeds/task_seeds.jsonl（test 切分 732 条，按行拼接）` | `7a2773e4a637b4b8a5d91e490830e58f1c127de033a54bae172bca691dd48457` |
| `seeds/multi_turn_seeds.jsonl（test 切分 92 条，按行拼接）` | `fc6eb1d1c77a6371e17779eff7248a79335c550fa42f6a62aee97c2616ef7cac` |
| `sft/swift_test.jsonl（732 条）` | `1ef11d9e516512f562a8361761fcc264c7cb5b0c87de2e9fcfca889238ad80a4` |
| `sft/jsontext_test.jsonl（732 条）` | `24da633c66e49c5f9e36c84dd0f01db74e32ba96f218b275100d6fe300fd2eaf` |
| `sft/lf_test.json（732 条）` | `cce8551c2f5dde545511b2af8b0e32a268fdcdc79c9e642210ccab34a981bc61` |
| `sft/swift_multiturn_test.jsonl（203 条）` | `5ab2a535b9c2cb9cf9ec2e4d700a2c5300d3dc4d6d209ed261c1622ae7592257` |

校验命令：`python3 scripts/planner_data/check_sealed.py`（比对上表种子级哈希，任何不一致即退出码 1）。

## 变更记录

- 2026-09-16 A-2 口语化改写后重导出（`export_sft.py --seeds task_seeds_rewritten.jsonl`）：test 切分未改写，种子级 sha256 两项不变（`check_sealed.py` 通过）；`swift_test.jsonl` / `jsontext_test.jsonl` / `lf_test.json` / `swift_multiturn_test.jsonl` 字节级与上表一致。train / dev 由 2311 / 439 扩至 6131 / 1157 单轮样本。
