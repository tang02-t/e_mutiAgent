# Planner 百炼调优（D-1 SFT 基线 M1 / D-3 DPO M4）

本目录是 PLAN D 阶段在百炼平台（阿里云 Model Studio，北京地域）训练 Planner 的操作记录与脚本。
训练与部署都发生在平台侧，本地只负责：数据导出与校验 → 上传 → 建任务 → 轮询 → 部署 → 用部署端点评测。

## 1. 路线与约束

| 项 | 取值 | 说明 |
|---|---|---|
| 基座 | `qwen3-8b` | 文本 8B 级，平台同时支持 `efficient_sft` 与 `dpo_lora`；Planner 无视觉输入 |
| SFT（M1） | `training_type=efficient_sft` | `n_epochs=3`、`batch_size=16`、`max_length=4096`、`learning_rate` 平台默认 |
| DPO（M4） | `training_type=dpo_lora`，`model=<M1 finetuned_output>` | `n_epochs=2`、`batch_size=16`、`max_length=4096`、`dpo_beta=0.1`（若不可配则记录） |
| 数据 | `data/planner/sft/bailian_{train,dev}.jsonl`（D8）；`data/planner/dpo/*.jsonl`（D13，D-2 产出） | 单文件 ≤ 200 MB，总配额 5 GB / 100 文件 |
| 鉴权 | 环境变量 `DASHSCOPE_API_KEY` | 不写入仓库；`config.yaml` 现有密钥待 A-4 撤销 |
| 计费 | API 创建任务按训练 Token 计费；部署按时长计费 | 评测结束立即 `undeploy` |
| 平台限制 | 参数快照不可下载 | 评测只能走部署端点；LoRA 权重无法迁移到魔搭 / 本地，备选路线见 PLAN D-3 |

## 2. 数据格式（百炼 SFT，ChatML + 工具调用）

由 `scripts/planner_data/export_sft.py`（默认或 `--format bailian`）导出，`scripts/planner_data/bailian_format.py` 负责转换与校验：

- 顶层仅 `messages` + `tools`；无 `meta`。
- `assistant.tool_calls[] = {id, type:"function", function:{name, arguments:<JSON 字符串>}}`。
- `tool` 消息仅 `role / tool_call_id / content`，`tool_call_id` 与前一条 assistant 的 `tool_calls[].id` 一一对应；不带 `name`。
- 末条消息为 assistant（训练目标）。错误恢复轨迹的「刻意错误首调」只在上下文中出现，不做训练目标（平台对所有 assistant 输出都计算 loss，不支持逐条权重）。
- 单轮 + 多轮决策点合并（含 A-2 口语化改写，2026-09-16 重导出）：train 7001 条（6131 + 870，91.8 MB），dev 1301 条（1157 + 144，17.0 MB）。test 封存不导出（`data/planner/sft/SEALED.md`）。百炼单文件上限请以控制台为准；如超限可用 `--no-multi-turn` 或按行切分上传。

校验：`python3 scripts/planner_data/bailian_format.py data/planner/sft/bailian_train.jsonl data/planner/sft/bailian_dev.jsonl`
自检：`python3 tests/test_d1_bailian_export.py`

## 3. 操作步骤（`submit_job.py`，默认 dry-run）

```bash
export DASHSCOPE_API_KEY=sk-...
# 上传（purpose=fine-tune）
python3 training/planner_bailian/submit_job.py --execute upload data/planner/sft/bailian_train.jsonl data/planner/sft/bailian_dev.jsonl
# 创建 SFT 任务
python3 training/planner_bailian/submit_job.py --execute create --training-file <train_file_id> --validation-file <dev_file_id> --job-name planner_sft_m1 --model-name planner-sft-m1
# 轮询直到终态（每 2 min）
python3 training/planner_bailian/submit_job.py --execute status --job-id <job_id> --wait
# 部署
python3 training/planner_bailian/submit_job.py --execute deploy --model-name <finetuned_output> --capacity 1
```

去掉 `--execute` 即只打印请求体（无网络、无费用），用于核对参数。每次真实调用的响应追加写入 `jobs.jsonl`。

部署完成后：
1. `config.yaml` 新增 `llms.planner_finetuned`（`base_url` 为 DashScope 兼容模式，`model_name=<deployed_model>`），保持 `enable_thinking: false`。
2. `python3 scripts/dev/probe_llm_endpoint.py` 验证返回 `tool_calls` 可被 `PlannerAgent._build_plan_from_tool_calls` 解析（`id / name / arguments` 三字段）。
3. 在 D8 test 上运行 `eval_planner_offline.py`，与 M0 一并写入 `docs/planner_dpo_eval.md` 第一节。

## 4. 任务记录（实际提交后填写）

| 阶段 | 日期 | file_id（train / dev） | job_id | finetuned_output | deployed_model | 备注 |
|---|---|---|---|---|---|---|
| M1 SFT | 待提交 | – | – | – | – | 超参见 §1 |
| M4-exec DPO | 待提交 | – | – | – | – | 训练集 D13-exec |
| M4-full DPO | 待提交 | – | – | – | – | 训练集 D13-full |

## 5. 已知风险

- 若控制台 `qwen3-8b` 的 `dpo_lora` 不可用或 `dpo_beta` 不可配，按 PLAN D-3 备选（魔搭 A10 `swift rlhf --rlhf_type dpo`）并在此记录原因。
- 部署实例空闲也计费；评测完成后执行 `submit_job.py --execute undeploy --deployed-model <id>`。
- 训练后系统提示若变更，需重新导出并重新训练（训练 / 推理分布一致性由 `PLANNER_SYSTEM_PROMPT()` 保证）。
