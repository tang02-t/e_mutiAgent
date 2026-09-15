# 电力变压器故障诊断多智能体系统

Planner → Retriever → [Reflection] → Generator → Validator 五智能体（LangGraph 编排），工具层包含 DGA 贝叶斯归因、ETT 油温预测、时序异常检测、离线知识库两路检索（`rag_search`）、故障关系图谱多跳检索（`kg_search`）。规划模块支持用 LoRA 微调后的 Qwen3-VL-8B 替换（`planner_mode: finetuned`），并提供五级系统模式用于消融对比。

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

训练环境（可选，魔搭 A10 24GB）：`pip install ms-swift[llm] -U`，见 [docs/planner_training.md](docs/planner_training.md)。

## 一键复现（离线，不调用 LLM）

```bash
bash reproduce.sh                # 全部：DGA → 知识库 → 图谱 → 反思 → 规划数据 → D10 → 评测自检 → 回归测试
bash reproduce.sh kb kg          # 只跑指定阶段
SKIP_EXISTING=1 bash reproduce.sh
python3 tests/test_p0_fixes.py   # 87 项回归测试
```

各阶段对应脚本：

| 阶段 | 脚本 | 产物 |
|---|---|---|
| DGA 合并 / 参数学习 | `scripts/convert_real_dga.py`、`scripts/learn_cpt.py` | `data/real/dga/dga_records.jsonl`（5143）、`learned_params.json` |
| P1 知识库 | `scripts/kb/build_units.py` → `build_chunks.py` → `build_index.py`；`build_retrieval_evalset.py`；`eval_retrieval.py --ablation`；`ablation_chunking.py` | `data/kb/{units,chunks}.jsonl`、`data/kb/index/`、`data/kb/eval/retrieval_seed.jsonl`（1209）、`docs/kb_ablation_test.md` |
| P2 图谱 | `scripts/kg/extract_rules.py` → `build_graph.py` → `build_kg_eval.py` | `data/kg/graph.json`（90 节点 / 190 边）、`data/kg/eval/`（304 条问答 + 56 条精度抽检） |
| P3 反思 | `scripts/kb/build_reflection_evalset.py --n 300`；`eval_reflection_recall.py` | `data/kb/eval/reflection/d9_pairs.jsonl`（298 对）、`docs/reflection_eval.md` |
| P4 规划数据 | `scripts/planner_data/build_task_seeds.py --execute` → `build_multi_turn.py` → `export_sft.py`；`training/planner_sft/build_domain_terms.py` | `data/planner/seeds/`、`data/planner/sft/`（ms-swift 单轮 2311/439/732 + 多轮 870/144/203，LLaMA-Factory，JSON 文本）、`DATA_CARD.md` |
| P5 训练（需 GPU） | `training/planner_sft/train.sh M1|M2|M3 <seed>`、`run_matrix.sh`、`predict.py`；`scripts/eval/eval_planner_offline.py --write-report` | `output/planner_sft/`、`data/planner/predictions/`、`docs/planner_eval.md` |
| P6 评测 | `scripts/eval/build_d10.py`；`scripts/eval/eval_system_modes.py --modes all [--judge llm]` | `data/eval/d10/end2end_eval.jsonl`（196）、`docs/end2end_eval.md` |
| 端到端（旧接口） | `scripts/eval_end2end.py --limit N --reflection lexical --planner-mode baseline|finetuned` | `docs/eval_*.md` |

## 运行前端

```bash
bash run_frontend.sh                      # 默认加载学习到的贝叶斯参数
USE_LEARNED_CPT=0 bash run_frontend.sh    # 专家默认 CPT 对照
```

侧边栏「系统模式（P6 五级对比）」：mode1 无 RAG / mode2 朴素 RAG / mode3 + 微调 Planner / mode4 + 图谱 / mode5 完整系统（+ 反思）；「高级」可覆盖 Planner 模型与反思开关。标签页展示规划计划、工具调用轨迹、检索块、图谱链路、反思评分。

## 报告索引

| 报告 | 内容 | 状态 |
|---|---|---|
| `docs/data_inventory.md` | 数据清单、角色标注、合成数据守卫 | 完成 |
| `docs/kb_ablation_test.md` | 分块 α 网格、切分方式、检索通道消融（Recall@k） | 离线通道版本；稠密向量 + 重排待接口 |
| `data/kg/eval/report.md`、`docs/kg_schema.md` | 图谱 schema、304 条问答评测、精度抽检 | 规则抽取版本；LLM 抽取待接口 |
| `docs/reflection_eval.md` | 有 / 无反思检索对比、D9 评分分布 | LexicalScorer；LLMScorer Kappa 待人工分 |
| `docs/planner_training.md` | 微调路线定案、百炼约束、训练配置、执行清单 | 完成（训练待 GPU） |
| `docs/planner_eval.md` | M0–M3 七项指标分类别 | 待模型预测 |
| `docs/end2end_eval.md` | 五级模式任务成功率 / 忠实度 / 调用次数 / 延迟 / Token | 框架就位，正式数字待 Planner 真实调用 |
| `docs/progress_*.md` | 阶段进展报告 | 持续更新 |

## 数据与安全约束

- `data/synthetic/` 下的合成数据只用于流程回归（`--purpose dev_regression`），`assert_not_synthetic` 强制阻止其进入评测集与训练集。
- 训练 / 验证 / 测试按 `group_key`（来源块 / 实体对 / DGA 记录 / ETT 片段）分组切分，test 封存，跨集泄漏 0。
- `config.yaml` 含密钥不入库；训练产物 `output/`、模型预测 `data/planner/predictions/` 不入库。

## 已知局限

- DGA 数据集的 `fault_label` 是 IEC 标签到本系统故障类型的工程映射，评测使用 `raw_label`；CO / CO2 为 0 填充，不是真实测量；`device_id` 为循环生成，不能作为设备维度分析依据。
- ETT 数据集是电力变压器油温公开数据，负载特征（HUFL 等）单位未知；ETTh 与 ETTm 疑似同源，切分时按同组处理；预测模型为线性回归，仅用于工具调用行为研究，不代表预测 SOTA。
- `power_transformer_fault.csv`（1866 行）来源尚未核实，未进入任何训练 / 评测集。
- 知识库文献存在重复与 MinerU 解析噪声，图片仅有路径、表格无标题，多模态转文本依赖 LLM 尚未完成；当前检索为 BM25 子块 + 锚点两路 RRF，无稠密向量与重排器。
- 图谱由规则抽取（显式因果触发词），90 节点 / 190 边，覆盖有限，精度抽检 83%（严格）/ 95.7%（宽松）。
- 反思评分器目前是词法基线，LLMScorer 与人工评分的一致性尚未测量；D9 人工 `human_score` 未填写。
- 规划训练数据问法为模板生成，口语化改写（P4-2）待 LLM；多轮样本的 assistant thought 为规则模板文本。
- 端到端评测的「答案忠实度」在离线模式下使用证据锚点代理，LLM 裁判与 10% 人工复核未执行；D10 标注 `status=auto`。
- 所有依赖 LLM 的正式数字（基线、五级模式、M0–M3 对照）尚未产出；论文对照实验的 GPU 训练需在魔搭环境执行。
