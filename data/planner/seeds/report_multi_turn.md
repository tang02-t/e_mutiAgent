# P4-3 多轮 / 错误恢复样本（D8 补充）构建报告

- 样本总数：529；切分：{'train': 373, 'dev': 64, 'test': 92}
- 每样本工具调用次数分布：{1: 309, 2: 212, 3: 8}
- 所有 tool 消息内容来自真实工具执行（压缩字段），assistant 的第二轮决策由规则依据真实返回生成；不含任何臆造工具输出。

| 类别 | 子类 | 数量 | 说明 |
|---|---|---:|---|
| error_recovery | ett_bad_dataset | 18 | 枚举外数据集 → 校验失败 → 修正 |
| error_recovery | ett_bad_range | 12 | 时间范围超出 → 工具 error → 修正 |
| error_recovery | kg_colloquial_to_std | 30 | 俗称 no_match → 标准术语重试 |
| error_recovery | ts_unrecoverable_ask | 9 | 序列过短 → 不可恢复 → 追问 |
| multi_turn | composite_2step | 152 | 两步均成功，finish |
| multi_turn | composite_recover_kg | 8 | 第 2 步 no_relation → 放宽关系重试 |
| multi_turn | single_then_finish | 300 | 单工具成功后停止 |

## 说明
- `trajectory` 为统一轨迹：assistant(tool_call|finish[, ask_user]) / tool(content)。导出 SFT 时每个 assistant 决策点生成一条训练样本（前缀含此前全部消息）。
- `composite_*` 沿用原 composite 种子的 group_key/split；kg 俗称类按标准实体分组；ts 类为合成短序列，仅用于训练侧「不可恢复→追问」行为，不进入评测。
- query 仍为模板问法（needs_llm_rewrite=true）。