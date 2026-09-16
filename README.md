# 电力变压器故障诊断多智能体系统

Planner → Retriever → [Reflection] → Generator → Validator 五智能体（LangGraph 编排），工具层包含 DGA 贝叶斯归因、ETT 油温预测、时序异常检测、离线知识库两路检索（`rag_search`）、故障关系图谱多跳检索（`kg_search`）。规划模块支持用微调后的 Planner 替换（`planner_mode: finetuned`，v2 路线为百炼 `qwen3-8b` SFT + DPO），并提供五级系统模式用于增量归因。研究主线与任务清单见 [PLAN_优化计划清单.md](PLAN_优化计划清单.md)。

- 项目结构说明：[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- 完整计划与进度：[PLAN_优化计划清单.md](PLAN_优化计划清单.md)（顶部「进度状态」表）
- 阶段报告：`docs/`（见下文「报告索引」）

## 环境

```bash
python3 --version          # 3.9+（3.9 环境下运行时注解不使用 X | None）
pip3 install -r requirements.txt   # openai langgraph streamlit rank-bm25 jieba pyyaml numpy pandas rich graphviz
cp config.example.yaml config.yaml # 填 llms.*.api_key / base_url / model_name（config.yaml 已 gitignore）
python3 scripts/probe_llm_endpoint.py   # 探测对话与 function calling 是否可用
```

训练：v2 主路线走百炼平台 SFT / DPO（无需本地 GPU）；v1 魔搭 ms-swift 路线见 [docs/planner_training.md](docs/planner_training.md)（备选）。

## 一键复现（离线，不调用 LLM）

```bash
bash reproduce.sh                # 全部：DGA → 知识库 → 图谱 → 反思 → 规划数据 → D10 → 评测自检 → 回归测试
bash reproduce.sh kb kg          # 只跑指定阶段
SKIP_EXISTING=1 bash reproduce.sh
for t in tests/test_*.py; do python3 "$t" | tail -1; done   # 回归测试：p0 87 项 + B-1 47 项 + B-2 34 项
```

各阶段对应脚本：

| 阶段 | 脚本 | 产物 |
|---|---|---|
| DGA 合并 / 参数学习 | `scripts/convert_real_dga.py`、`scripts/learn_cpt.py` | `data/real/dga/dga_records.jsonl`（5143）、`learned_params.json` |
| P1 知识库 | `scripts/kb/build_units.py` → `build_chunks.py` → `build_index.py`；`build_retrieval_evalset.py`；`eval_retrieval.py --ablation`；`ablation_chunking.py` | `data/kb/{units,chunks}.jsonl`、`data/kb/index/`、`data/kb/eval/retrieval_seed.jsonl`（1209）、`docs/kb_ablation_test.md` |
| P2 图谱 | `scripts/kg/extract_rules.py` → `build_graph.py` → `build_kg_eval.py` | `data/kg/graph.json`（90 节点 / 190 边）、`data/kg/eval/`（304 条问答 + 56 条精度抽检） |
| P3 反思 | `scripts/kb/build_reflection_evalset.py --n 300`；`eval_reflection_recall.py` | `data/kb/eval/reflection/d9_pairs.jsonl`（298 对）、`docs/reflection_eval.md` |
| P4 规划数据 | `scripts/planner_data/build_task_seeds.py --execute` → `build_multi_turn.py` → `export_sft.py`；`training/planner_sft/build_domain_terms.py` | `data/planner/seeds/`、`data/planner/sft/`（ms-swift 单轮 2311/439/732 + 多轮 870/144/203，LLaMA-Factory，JSON 文本）、`DATA_CARD.md` |
| P5 训练（v1 备选路线） | `training/planner_sft/train.sh`、`run_matrix.sh`、`predict.py`；`scripts/eval/eval_planner_offline.py --write-report` | `output/planner_sft/`、`data/planner/predictions/`、`docs/planner_eval.md` |
| P6 评测 | `scripts/eval/build_d10.py`；`scripts/eval/eval_system_modes.py --modes all [--judge llm]` | `data/eval/d10/end2end_eval.jsonl`（196）、`docs/end2end_eval.md` |
| 端到端（旧接口，合成 eval_set 回归） | `scripts/eval_end2end.py --limit N --reflection lexical --kb-mode local_kb|mock` | `data/synthetic/eval/report_end2end_dev*.md` |

## 运行前端

```bash
bash run_frontend.sh                      # 默认加载学习到的贝叶斯参数
USE_LEARNED_CPT=0 bash run_frontend.sh    # 专家默认 CPT 对照
```

侧边栏「系统模式（五级增量）」：mode1 `llm_only`（无工具）/ mode2 `tool_base`（全工具 + 知识库 + 图谱 + 词法反思，专家 CPT，自由规划）/ mode3 `active_plan`（+ 校准归因与 EIG 主动追问）/ mode4 `claim_verify`（+ 声明级输出、约束核查与补证路由）/ mode5 `dpo_planner`（+ DPO 微调 Planner）；「高级」可逐项覆盖 Planner 模型、归因参数、规划策略、Generator 输出、Validator 模式与反思开关。标签页展示规划计划、工具调用轨迹、检索块、图谱链路、反思评分、追问过程。

## 报告索引

| 报告 | 内容 | 状态 |
|---|---|---|
| `docs/技术报告_v2.md` | 阶段技术报告：背景、三条主线设计与已验证数字、完成状态、预期目标与量化目标、创新点、已知局限（v1 `技术报告.md` 已被取代） | 2026-09-16 |
| `docs/data_and_evaluation.md` | 数据来源、许可与已知问题（R1–R11）、派生谱系与文件格式（D1–D13）、合成数据守卫、角色隔离规则、四层评测体系与指标口径 | 完成 |
| `docs/kb_ablation_test.md` | 分块 α 网格、切分方式、检索通道消融（Recall@k） | 离线通道版本；稠密向量 + 重排待接口 |
| `data/kg/eval/report.md`、`docs/kg_schema.md` | 图谱 schema、304 条问答评测、精度抽检 | 规则抽取版本；LLM 抽取待接口 |
| `docs/reflection_eval.md` | 有 / 无反思检索对比、D9 评分分布 | LexicalScorer；LLMScorer Kappa 待人工分 |
| `docs/planner_training.md` | v1 微调路线（魔搭 LoRA → 百炼导入）、百炼约束 | 已降级为备选；v2 走百炼 SFT + DPO |
| `docs/baseline.md` | A-1 三组基线：检索 Recall@k、M0 Planner 7 指标（100 条）、mode2 端到端（50 条）与失败模式归因 | 已产出（2026-09-16） |
| `docs/planner_eval.md` | Planner 七项指标分类别（v2 矩阵：M0 / M1 / M4-exec / M4-full） | M0 列已产出；M1 / M4 待百炼训练 |
| `docs/end2end_eval.md` | 五级模式任务成功率 / 忠实度 / 调用次数 / 延迟 / Token | 框架就位，正式数字待 Planner 真实调用 |
| `docs/chapter3_active_planning_material.md`、`docs/chapter4_claim_verification_material.md`、`docs/chapter5_planner_dpo_material.md` | 论文第 3-5 章素材：定义表、流程图、消融设计、相关工作差异、可引用数字 | 第 3-4 章数字齐全；第 5 章数字待百炼训练 |
| `docs/progress_*.md` | 阶段进展报告 | 持续更新 |

## 数据与安全约束

- `data/synthetic/` 下的合成数据只用于流程回归（`--purpose dev_regression`），`assert_not_synthetic` 强制阻止其进入评测集与训练集。
- 训练 / 验证 / 测试按 `group_key`（来源块 / 实体对 / DGA 记录 / ETT 片段）分组切分，test 封存，跨集泄漏 0。
- `config.yaml` 含密钥不入库；训练产物 `output/`、模型预测 `data/planner/predictions/` 不入库。

## 已知局限

- DGA 数据集（5143 条）由三份文献汇编来源合并去重而来：`dga_dataset.csv`（4150 行，来源未核实，非 Kaggle）与 `alan-456/transformer-fault-dataset` 的两份文件（2321 + 589 行，未声明许可）。IEC 故障标签为文献汇编标签，无法逐条追溯原始检修记录；`fault_label` 是 IEC 标签到本系统故障类型的工程映射，评测使用 `raw_label`；CO / CO2 为 0 填充，不是真实测量；`device_id` 为循环生成，不能作为设备维度分析依据。
- ETT 数据集（ETDataset, CC BY-ND 4.0，以原仓库 LICENSE 为准）是电力变压器油温公开数据，负载特征（HUFL 等）单位未知；ETTh 与 ETTm 疑似同源，切分时按同组处理；预测模型为线性回归，仅用于工具调用行为研究，不代表预测 SOTA。
- `power_transformer_fault.csv`（1866 行）来源尚未核实，公开数据站点检索未匹配，未进入任何训练 / 评测集。
- 知识库文献存在重复与 MinerU 解析噪声，图片仅有路径、表格无标题，多模态转文本依赖 LLM 尚未完成；当前检索为 BM25 子块 + 锚点两路 RRF，无稠密向量与重排器。
- 图谱由规则抽取（显式因果触发词），90 节点 / 190 边，覆盖有限，精度抽检 83%（严格）/ 95.7%（宽松）。
- 反思评分器目前是词法基线，LLMScorer 与人工评分的一致性尚未测量；D9 人工 `human_score` 未填写。
- 规划训练数据问法为模板生成，口语化改写（P4-2）待 LLM；多轮样本的 assistant thought 为规则模板文本。
- 端到端评测的「答案忠实度」在离线模式下使用证据锚点代理，LLM 裁判与 10% 人工复核未执行；D10 标注 `status=auto`。
- 所有依赖 LLM 的正式数字（基线、五级模式 E-2、M0 / M1 / M4 对照）尚未产出；DPO 训练需在百炼平台执行（`submit_job.py`）。
