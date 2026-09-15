# 进展报告：P5 / P6 / P7 离线部分（2026-09-15）

承接 [progress_P2-5_P3_P4_offline.md](progress_P2-5_P3_P4_offline.md)。本轮按用户要求「非必要不调用模型」推进，所有工作在无 LLM、无 GPU 条件下完成并通过回归测试（87/87）。

## 1. 本轮完成

### P5-5 / P6 系统接入与五级模式（提交 `9cc5c2e`、`cbd8732`）
- `workflow.planner_mode: baseline | finetuned`，finetuned 读 `llms.planner_finetuned`，缺失自动回退。
- `src/graph/system_modes.py`：mode1 无 RAG → mode2 朴素 RAG → mode3 + 微调 Planner → mode4 + 图谱 → mode5 + 反思；`allowed_tools` 同时作用于 Planner 提示词 / function calling 与 Retriever 执行守卫（越权记 `tool_disabled`）。
- D10 端到端评测集 196 条（8 场景，71 group_key，来自封存 test），`scripts/eval/eval_system_modes.py` 计算任务成功率四要素 / 追问 / 恢复 / 不必要调用 / 调用次数 / 延迟 / Token。OraclePlanner 校验评分链路：task_success 88.8%，param_acc 100%，tool_result_valid 100%。
- 前端五级模式下拉、工具调用轨迹表、图谱链路 graphviz、反思评分逐轮展示、5 个演示案例。
- `src/utils/llm.py` 新增 `UsageTracker`（Token 成本）。

### P5 训练侧（提交 `0c63fb9`）
- 路线定案：路线 A（魔搭 ms-swift LoRA → OSS → 百炼导入 → DashScope 调用），理由与百炼约束核对表见 [planner_training.md](planner_training.md)。
- `training/planner_sft/train.sh`：LoRA rank 8 / alpha 32 / dropout 0.05、lr 5e-5、AdamW、warmup 20、cosine、bf16、seq 2048、有效 batch 16、epoch ≤ 5 按 dev loss 早停、冻结 ViT；`PILOT=1` 为 50 条小闭环。
- 加权损失：`weighting.py`（纯 Python 切分：普通文本 1 / 结构 2 / 领域 3）、`plugin_loss_scale.py`（注册 `planner_struct` / `planner_domain` loss_scale 与 `normalized_weighted_ce`，兼容 ms-swift 3.x / 4.x）、`domain_terms.json`（609 项，由工具 Schema、故障引擎、图谱同义词与节点自动生成）。多轮 train 集响应字符分布：文本 42.2% / 结构 37.6% / 领域 20.2%。
- `predict.py`（swift / openai 两种后端，统一预测格式，断点续跑）、`run_matrix.sh`（M0 + M1–M3 × seeds 42/2026 → 推理 → 评测）。
- `scripts/eval/eval_planner_offline.py`：格式合法率、工具选择正确率、参数正确率、完整调用率（含 Schema 严格校验）、不必要调用率、追问正确率、错误恢复成功率，分 8 类别，多 seed 取均值写 `docs/planner_eval.md`；金标自检 951 条全部 100%。
- `LLMClient` 新增 `provider: dashscope`（百炼导入模型不支持 OpenAI 兼容模式）。

### 数据修正
- 错误恢复轨迹中刻意错误的首调（ETTh3 / 超范围时间 / 口语词）不再导出为训练目标，只保留在上下文；多轮样本 1017 → 870/144/203。自检时发现该问题：金标含 ETTh3 使完整调用率 84.4%，修正后 100%。

### P7
- `README.md`（环境、一键复现、脚本表、报告索引、已知局限）、`reproduce.sh`（8 阶段可选，`LOG_LEVEL=WARNING` 静音）、`requirements.txt`。
- `PLAN_优化计划清单.md` 勾选 37 项（累计 83 项）、进度表更新。
- `.gitignore` 增加 `output/`、`data/planner/predictions/`。

## 2. 检查结论
- `python3 tests/test_p0_fixes.py`：87 passed（新增 P5-2 / P5-4 16 项）。
- `bash reproduce.sh eval test`：Planner 离线评测自检通过；mode5 OraclePlanner 50 条 task_success 92%。
- `plugin_loss_scale.py` 本地导入不报错（无 swift 时只加载词典，不注册）。
- `bash -n` 通过：`train.sh`、`run_matrix.sh`、`reproduce.sh`。

## 3. 剩余事项（全部需要 LLM 批量调用或 GPU 环境）

| 事项 | 需要 | 入口 |
|---|---|---|
| P0-3 基线数字 | LLM | `scripts/eval_end2end.py --purpose eval_set` |
| P4-2 口语化改写 / 复合连贯性 / D7 分解集 | LLM | `scripts/planner_data/rewrite_queries.py`（已就位，dry-run 估算 2750 条 / 约 1.2 元）→ `export_sft.py --seeds task_seeds_rewritten.jsonl` |
| P3 LLMScorer 与人工 Kappa | LLM + 人工 D9 | `build_reflection_evalset.py --eval-only` |
| P2 LLM 抽取补图谱 / LightRAG 对比 | LLM | 1677 候选句 |
| P1 图片描述 / 摘要 / 稠密向量 / 重排 | LLM + embedding | — |
| P5 50 条小闭环 → 正式矩阵 → 百炼导入 | 魔搭 A10 + OSS + 百炼 | `PILOT=1 train.sh M3 42` → `run_matrix.sh` |
| P6 五级模式正式评测 + LLM 裁判 | LLM（含微调 Planner） | `eval_system_modes.py --modes all --judge llm` |
| P7 `v1-final` tag | 上述完成后 | — |

人工待办：D9 `human_score`；图谱 `precision_sample.jsonl` label；`power_transformer_fault.csv` 来源；D10 `reference_points` 复核；轮换已泄露的 AccessKey `REDACTED_AK_ID`。
