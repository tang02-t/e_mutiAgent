# D12 部分观测诊断模拟集数据卡片

- 文件：`data/eval/d12/partial_obs.jsonl`
- 规模：1500 条（light / medium / heavy 各 500）；seed 20260915；sha256 `9e2a35b3e0758e9b…`
- 来源：`data/real/dga/dga_records.jsonl` 非 normal 记录 3729 条（真实文献汇编 DGA，见 `docs/data_inventory.md` R1-R3），按故障类别分层抽样；同一真实记录可在不同档位重复出现（遮蔽不同），同档内不重复
- 生成：`python3 scripts/sim/partial_obs_sim.py --build`；校验：`--check`
- 真实 / 合成：全部由真实 DGA 派生；`data_guard.assert_not_synthetic` 校验源路径

## 遮蔽方案

| 档位 | 遮蔽气体数 | 可见气体数 | 条数 |
|---|---:|---:|---:|
| light | 1 | 4 | 500 |
| medium | 2 | 3 | 500 |
| heavy | 3 | 2 | 500 |

被遮蔽气体在 `visible_gases` 中为 `null`；`initial_evidence` 只含可见气体派生的征兆（含负观测），`hidden_evidence` 为揭示被遮蔽气体后可获得的征兆。CO / CO2 / 油温 / 绕组温度 / 负载 / 振动 / 局放告警在原始数据中无字段，`PartialObsEnv.reveal` 返回 `None`（策略应视为「不可获取」并计成本）。

## 标签分布（与源数据对比）

| 故障 | 源数据 | 源占比 | D12 全部 | D12 占比 | light | medium | heavy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `insulation_degradation` | 511 | 13.7% | 207 | 13.8% | 69 | 69 | 69 |
| `overload_overheating` | 1495 | 40.1% | 600 | 40.0% | 200 | 200 | 200 |
| `partial_discharge` | 1070 | 28.7% | 429 | 28.6% | 143 | 143 | 143 |
| `winding_short_circuit` | 653 | 17.5% | 264 | 17.6% | 88 | 88 | 88 |

## 用途

- B-5 `eval_active_planning.py`：五种问询策略在同一初始观测下逐步揭示征兆，比较停止时准确率与问询次数 / 成本。
- E-3 鲁棒性：mode2 vs mode3 在 heavy 档上的端到端差异。

## 局限

- 标签来自文献汇编，不含设备身份与现场征兆；现场征兆（温度 / 负载 / 振动 / 局放）无法揭示，主动规划评测只能在气体征兆空间内进行。
- 遮蔽为均匀随机，不模拟真实工况下「某些气体更常缺失」的偏置。