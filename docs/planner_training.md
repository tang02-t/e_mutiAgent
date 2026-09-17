# Planner 微调路线与训练流程（P5，v1 路线）

> **v2 说明**：本文件记录的是 v1「魔搭 ms-swift 加权 LoRA → OSS → 百炼导入」路线，现已降级为备选。当前主路线为百炼平台 SFT + DPO，见 [PLAN_优化计划清单.md](../PLAN_优化计划清单.md) D 阶段。

> 状态：离线部分完成（脚本 / 插件 / 评测 / 配置全部就绪，本地 dry-run 通过）；训练与部署需魔搭 A10 环境与百炼账号，按下文「执行清单」进行。

## 1. 路线定案（P5-1）

**路线 A：魔搭 ms-swift 本地 LoRA 训练 → 导出 LoRA 权重 → OSS → 百炼「自定义模型导入」→ DashScope 调用。**

选择理由：
- 论文对照：与叶金涛论文实验平台一致（魔搭 A10 24GB、Qwen3-VL-8B-Instruct、ms-swift LoRA rank 8 / alpha 32 / lr 5e-5），结论可直接比较。
- 损失加权必须自己控制训练循环：百炼平台内置 SFT 不暴露 token 级 loss 权重，无法实现 M2 / M3；ms-swift 的 `loss_scale` 插件机制可以。
- 百炼只承担推理部署：本地多智能体编排、工具、知识库、图谱全部保留在项目内，与百炼解耦，只把 Planner 换成微调模型（`workflow.planner_mode: finetuned`）。

放弃的备选：
- 路线 B（百炼平台内 SFT）：无法加权损失、训练数据格式受平台限制、参数快照不可下载，仅可作为 M1 的旁证。
- 路线 C（本地 vLLM 部署）：无 GPU 服务器长期在线，答辩演示不稳定。

## 2. 百炼 LoRA 导入约束核对

| 约束 | 本项目对应 | 状态 |
|---|---|---|
| 基座须为百炼支持导入的开源模型 | Qwen3-VL-8B-Instruct（与训练基座一致） | 需在控制台「自定义模型导入」页面核对当前支持列表 |
| LoRA rank ∈ {8, 16, 32, 64} | rank 8 | 满足 |
| 不能修改词表、tokenizer、chat_template | 训练不加特殊 token，agent_template 用 hermes（Qwen 原生 `<tool_call>` 格式） | 满足 |
| 视觉模型须冻结 ViT | `--freeze_vit true --freeze_aligner true` | 满足 |
| LoRA 权重放 OSS Bucket 子目录（adapter_config.json + adapter_model.safetensors） | ms-swift checkpoint 目录直接上传 | 满足；上传前删除 optimizer.pt 等训练态文件 |
| OSS 与百炼同地域（华北 2 北京） | 需新建 Bucket 时选择 cn-beijing | 待执行 |
| 导入模型只能 DashScope 原生方式调用，不支持 OpenAI 兼容模式 | `src/utils/llm.py` 新增 `provider: dashscope`；`config.example.yaml` 的 `planner_finetuned` 示例已更新 | 满足 |
| 费用 | 导入模型按调用计费 + 部署实例费；魔搭 A10 免费额度按小时 | 训练结束即释放实例 |

## 3. 数据

- 训练集：`data/planner/sft/swift_train.jsonl`（2311 单轮）+ `swift_multiturn_train.jsonl`（870 多轮决策点）
- 验证集：`swift_dev.jsonl`（439）+ `swift_multiturn_dev.jsonl`（144）
- 测试集（封存）：`swift_test.jsonl`（732）+ `swift_multiturn_test.jsonl`（203），只在最终评测读取
- 错误恢复样本中刻意错误的首调只作上下文，不做训练目标（`decision_index` 从 1 起）
- 合成数据（`data/synthetic/`）不参与；数据卡片：`data/planner/sft/DATA_CARD.md`
- 口语化改写（A-2）已于 2026-09-16 完成并重导出：单轮 train / dev 6131 / 1157，百炼 ChatML train 7001 / dev 1301；本路线的 ms-swift / LLaMA-Factory 格式如需使用请重跑 `export_sft.py --seeds data/planner/seeds/task_seeds_rewritten.jsonl --format swift`

## 4. 训练配置（P5-2）

`training/planner_sft/train.sh`：

| 项 | 值 |
|---|---|
| LoRA | rank 8 / alpha 32 / dropout 0.05 / all-linear |
| 优化 | AdamW，lr 5e-5，warmup 20 步，cosine |
| 精度 / 长度 | bf16，max_length 2048（超长样本 delete） |
| batch | per_device 2 × grad_accum 8 = 有效 16 |
| epoch | 上限 5，按 dev loss 早停（`early_stop_interval 2`，`load_best_model_at_end`）；不照抄论文 20 epoch |
| Agent 模板 | hermes |
| 冻结 | ViT + aligner |

### 加权损失（M2 / M3）

`training/planner_sft/plugin_loss_scale.py` 通过 `--external_plugins` 加载，注册：

- `loss_scale planner_struct`：结构 token 权重 2（`<tool_call>` 标签、JSON 键、括号、非自由文本的参数标量值），其余响应文本 1
- `loss_scale planner_domain`：在上面基础上，领域关键标识权重 3（词典 `domain_terms.json`，609 项：工具名、参数名、枚举值、故障 / 征兆 ID 与中文名、DGA 气体、ETT 数据集、图谱实体及同义词，由 `build_domain_terms.py` 从项目资产自动生成）
- `loss_type normalized_weighted_ce`：L = Σ wᵢ·CEᵢ / Σ wᵢ，避免加权后 loss 量级随方案变化
- 输入（system / user）与工具返回权重 0 由 ms-swift 的上下文类型机制处理

切分算法在 `weighting.py`，纯 Python，本地可测：多轮 train 集响应字符分布为普通文本 42.2% / 结构 37.6% / 领域 20.2%。

## 5. 对照矩阵（P5-3）

| 变体 | 说明 | 命令 |
|---|---|---|
| M0 | 未微调基座 | `predict.py --model Qwen/Qwen3-VL-8B-Instruct --run-name M0` |
| M1 | 普通 LoRA-SFT | `train.sh M1 <seed>` |
| M2 | 结构 token 加权 | `train.sh M2 <seed>` |
| M3 | 结构 + 领域 token 加权 | `train.sh M3 <seed>` |

每个变体 2 个 seed（42 / 2026），`run_matrix.sh` 一键完成训练 → 推理 → 评测；报告 `docs/planner_eval.md` 对同变体多 seed 取均值。Q0 / Q1（查询分解智能体有无微调）在 v2 中不做。

## 6. 评测（P5-4）

`scripts/eval/eval_planner_offline.py` 读取 `training/planner_sft/predict.py` 的预测文件，输出 7 项指标并分 8 个类别：格式合法率、工具选择正确率、参数正确率、完整调用率（含 Schema 严格校验）、不必要调用率、追问正确率、错误恢复成功率。已用金标做自检（`--self-test`，951 条全部 100% / 不必要调用 0%）。

## 7. 执行清单（需环境时）

**50 条小闭环（P5-1 验证流程）**
1. 魔搭创建 A10 实例，`pip install ms-swift[llm] -U`，`git clone` 项目或只上传 `training/planner_sft/` 与 `data/planner/sft/`
2. `PILOT=1 bash training/planner_sft/train.sh M3 42`（50 条、1 epoch，约 10 分钟）
3. 导出：`ls output/planner_sft/pilot_M3_seed42/checkpoint-*/`，保留 `adapter_config.json`、`adapter_model.safetensors`、`README.md`
4. `ossutil cp -r checkpoint-x oss://<bucket>/planner-lora/pilot/`
5. 百炼控制台 → 模型 → 自定义模型导入 → 选择基座 Qwen3-VL-8B-Instruct → 填 OSS 路径 → 部署
6. 本地 `config.yaml` 填 `llms.planner_finetuned`（provider dashscope，model_name 为部署 ID），`python3 scripts/probe_llm_endpoint.py`
7. `python3 scripts/eval_end2end.py --limit 3 --planner-mode finetuned` 确认能跑通

**正式矩阵**
1. `bash training/planner_sft/run_matrix.sh`（6 次训练，每次约 1–2 小时）
2. 挑选 dev 上最优的 M3 checkpoint 上传 OSS 并导入百炼，替换 `planner_finetuned.model_name`
3. `python3 scripts/eval/eval_system_modes.py --eval --modes all --judge llm`（P6 五级模式正式数字）

## 8. 已知风险
- ms-swift 版本迭代快，`plugin_loss_scale.py` 同时兼容 3.x / 4.x 导入路径；若基类签名再变，以当版 `LossScale` 为准修改 `get_loss_scale`
- 百炼支持导入的基座列表会变化；若 Qwen3-VL-8B 不在列表，退回 Qwen2.5-VL-7B-Instruct 或 Qwen3-8B（纯文本），并同步更新论文对照说明
- 领域词典采用最长匹配，可能把普通句中的「铁心」「油温」等升权；M3 的收益需与 M2 对照确认，而不是默认更好
