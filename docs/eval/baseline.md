# A-1 基线数字（baseline-v0 对照）

> 用途：v2 三条主线（主动规划 / 声明级验证 / DPO Planner）所有增益对比的起点。三组数字均在**未微调** `qwen3.7-flash`（阿里云百炼专属网关，`enable_thinking=false`）与 v1 工程基座（mode2）上产出。
> 代码基线：tag `baseline-v0`（`9be86da`，2026-09-14）定义评测口径；本报告在 `76883a9` 之后运行（2026-09-16），评分脚本与 `baseline-v0` 的差别仅为 E-1 五级模式重定义，评测集与指标定义不变。
> 复现：见文末命令；三组运行合计 LLM 调用约 250 次、约 0.5M Token。

## 1. 知识库检索 Recall@k（D3 test 切分，264 条）

零 LLM 调用。`naive` = 块级 BM25 + 锚点通道（无子块/摘要层）；`two_way` = 块级 BM25 + 子块层 + 摘要层 RRF 融合（RRF 权重取 dev 网格最优）。

| 配置 | 类型 | n | R@1 | R@3 | R@5 | R@10 | MRR | Doc@5 | 延迟 ms |
|---|---|---|---|---|---|---|---|---|---|
| naive | ALL | 264 | 0.360 | 0.576 | 0.674 | 0.780 | 0.494 | 0.788 | 2.5 |
| naive | A_title | 143 | 0.329 | 0.559 | 0.685 | 0.804 | 0.477 | 0.797 | |
| naive | B_caption | 92 | 0.500 | 0.696 | 0.772 | 0.848 | 0.616 | 0.880 | |
| naive | C_definition | 29 | 0.069 | 0.276 | 0.310 | 0.448 | 0.186 | 0.448 | |
| two_way | ALL | 264 | **0.402** | **0.629** | **0.742** | **0.811** | **0.534** | **0.830** | 12.8 |
| two_way | A_title | 143 | 0.455 | 0.671 | 0.783 | 0.853 | 0.584 | 0.832 | |
| two_way | B_caption | 92 | 0.402 | 0.663 | 0.783 | 0.848 | 0.550 | 0.924 | |
| two_way | C_definition | 29 | 0.138 | 0.310 | 0.414 | 0.483 | 0.241 | 0.517 | |

结论：two_way 相对 naive 在 ALL 上 R@5 +6.8 pt、MRR +4.0 pt，但 B_caption 的 R@1 反而下降 9.8 pt（子块层稀释了图注短文本的精确匹配），C_definition 两档都很低（定义类问句与正文措辞差异大）。这两处是 D3 自然问句版与稠密通道（`kb_ablation_test.md` 待产出项）的改进空间。数字与 `data/kb/eval/retrieval_result_bm25+anchor_test.json`、`retrieval_result_two_way_test.json` 一致。

## 2. Planner 离线 7 指标（D8 test 分层抽 100 条，M0 = qwen3.7-flash）

- 抽样：`predict.py --stratified 100 --seed 42`，按 `meta.category` 每类保底 5 条、余额按比例，确定性可复现。抽样分布：fact 25 / reasoning 15 / numeric_tool 15 / multi_turn 15 / no_tool 11 / error_recovery 7 / composite 6 / insufficient 6。
- 推理：OpenAI 兼容接口，`temperature=0`，传 `tools`。100 条 0 次调用失败，平均延迟 3330 ms（中位 3259 ms）。
- 输出形态：97 条走系统提示词要求的 JSON 规划文本（`{"intent_analysis", "steps":[...]}`），3 条走原生 `tool_calls`；两种形态 `parse_response` 均可解析，故格式合法率 100%。
- 预测文件：`data/planner/predictions/M0_qwen3.7-flash_s42.jsonl`（`.gitignore` 排除，260 KB，本机保留）；报告 `docs/eval/planner_eval.md`。

| 变体 | n | 格式合法率 | 工具选择正确率 | 参数正确率 | 完整调用率 | 不必要调用率 | 追问正确率 | 错误恢复成功率 |
|---|---|---|---|---|---|---|---|---|
| M0_qwen3.7-flash_s42 | 100 | 100.0% | 55.0% | 61.1% | 51.4% | 53.6% | 66.7% | 0.0% |

分类别：

| 类别 | n | 工具选择 | 参数正确 | 完整调用 | 不必要调用 | 追问正确 | 错误恢复 |
|---|---|---|---|---|---|---|---|
| fact | 25 | 84.0% | 84.0% | 84.0% | - | - | - |
| reasoning | 15 | 20.0% | 0.0% | 0.0% | - | - | - |
| numeric_tool | 15 | 73.3% | 93.3% | 66.7% | - | - | - |
| no_tool | 11 | 81.8% | - | - | 18.2% | - | - |
| insufficient | 6 | 66.7% | - | - | 33.3% | 66.7% | - |
| composite | 6 | 16.7% | 0.0% | 0.0% | - | - | - |
| multi_turn | 15 | 40.0% | 100.0% | 66.7% | 100.0% | - | - |
| error_recovery | 7 | 0.0% | 0.0% | 0.0% | 100.0% | - | 0.0% |

失败模式（供 D-1 SFT / D-2 偏好对构造针对性覆盖）：

1. reasoning / composite：M0 倾向把 `kg_search` 与 `rag_search` 并列调用（工具选择多重集合不匹配），且 12 次 `kg_search` 中 11 次省略 `hops`、5 次省略 `direction`，与金标「显式 `hops=1`、`direction=in/out`」不一致；即便按缺省容忍 `hops`，15 条中也只有 4 条 `relations` 匹配。composite 场景 M0 用 `rag_search` 代替第二步 `kg_search`（6 条中 4 条），另 2 条在金标之外多调 `rag_search`。
2. multi_turn 决策点 ≥1 与 error_recovery 决策点 2：金标为「工具已返回有效结果，结束规划」（无调用），M0 一律再次发起调用，不必要调用率 100%。这是 M0 缺少「何时停」的判断，也是 D13 偏好对里 `extra_calls` 扰动对应的真实失败。
3. error_recovery 决策点 1（看到工具报错后）：M0 重复原调用或改用 `rag_search`，未按金标改写为标准术语再调 `kg_search`，恢复成功率 0/2。
4. insufficient：6 条中 2 条对缺参数问题直接猜测默认值调用，追问正确率 66.7%。

## 3. mode2 端到端（D10 前 50 条，工程基座）

- 命令：`eval_system_modes.py --modes mode2 --limit 50 --tag a1_baseline`（旁路结果 `data/eval/d10/results/mode2_a1_baseline.jsonl`，不进 `docs/eval/end2end_eval.md`）。
- mode2 开关：五工具、rag two_way、lexical 反思、expert 归因、Planner baseline/free、Generator text、Validator off（声明级核查关；v1 LLM 打分验证仍在，6 条被判 FAIL 走 END）。
- 前 50 条场景分布：single_tool_numeric 14 / single_tool_fact 12 / kg_reasoning 8 / multi_tool 5 / no_tool 4 / ask_user 3 / error_recovery 3 / context_contrast 1（与 196 条全集比例接近）。
- 忠实度为离线代理（证据锚点覆盖），LLM 裁判未接入；38 条可判。

| 指标 | 值 |
|---|---|
| 任务成功率（四要素） | **46.0%**（23/50） |
| 动作正确率 | 82.0% |
| 关键参数正确率 | 58.0% |
| 工具结果有效率 | 78.0% |
| 忠实度代理均值 | 0.795（n=38） |
| 追问正确率（ask_user） | 0.0%（0/3） |
| 错误恢复率 | 66.7%（2/3） |
| 平均工具调用次数 | 1.22 |
| 不必要调用率 | 29.5%（18/61 次调用） |
| 平均 LLM 调用次数 / 样本 | 3.04（Planner + Generator + Validator，1 条二轮） |
| 平均延迟 | 24.4 s |
| 平均 Token / 样本 | 6790（合计 339,475） |
| 工具业务成功 / 调用 | 55 / 61 |

分场景成功率：

| 场景 | n | 成功率 | 动作 | 参数 | 结果有效 | 忠实度 | 平均调用 |
|---|---|---|---|---|---|---|---|
| no_tool | 4 | 75.0% | 75.0% | 100.0% | 100.0% | - | 0.25 |
| single_tool_fact | 12 | 66.7% | 83.3% | 75.0% | 83.3% | 0.75 | 0.92 |
| single_tool_numeric | 14 | 78.6% | 100.0% | 85.7% | 100.0% | 0.93 | 1.14 |
| kg_reasoning | 8 | 0.0% | 75.0% | 0.0% | 50.0% | 0.53 | 1.62 |
| multi_tool | 5 | 0.0% | 80.0% | 0.0% | 20.0% | 1.00 | 2.60 |
| ask_user | 3 | 0.0% | 0.0% | 100.0% | 100.0% | - | 0.67 |
| error_recovery | 3 | 0.0% | 100.0% | 0.0% | 66.7% | - | 1.33 |
| context_contrast | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 1.00 | 1.00 |

27 条失败归因：关键参数不匹配 16（主要是 `kg_search` 缺 `hops`/`direction`、`ett_forecast` 缺 `dataset`/`end_time`，与第 2 节失败模式 1 同源）；缺必需调用或 no_tool 却调用 6；ask_user 场景 3 条全部直接用默认参数调 `ett_forecast` 而不追问（对应主线一「主动追问」的动机）；忠实度 <0.5 有 2 条。

Oracle 上界对照：同一评分链路下金标 Planner（`oracle_mode2.jsonl`，196 条）成功率 97.4%、参数正确率 100%。46.0% 与 97.4% 之间的差距全部来自 Planner 决策与 Generator 忠实度，是三条主线要填的空间。

## 4. 与后续对比的口径约定

- 第 1 节 = 主线二不涉及检索层改动，作为 `kb_ablation_test.md` 的固定参照。
- 第 2 节 = D-4 矩阵的 M0 列（`docs/planner_dpo_eval.md`）；M1 / M4 在同一 100 条（`--stratified 100 --seed 42`）上跑即可直接对比。正式矩阵仍应补全 935 条 test 与 2 seed。
- 第 3 节 = E-2 五级模式表的 mode2 列在前 50 条上的先行值；E-2 正式跑 196 条 × 2 次后以正式值为准，本节仅作 A-1 基线登记。
- 忠实度、追问正确率等依赖答案文本的指标均为离线代理，E-2 接入 LLM 裁判后重报。

## 5. 复现命令

```bash
# 1 检索（零 LLM）
python3 scripts/kb/eval_retrieval.py --split test --mode naive --tag naive
python3 scripts/kb/eval_retrieval.py --split test --mode two_way
# 2 Planner 离线（100 次 LLM 调用）
export DASHSCOPE_API_KEY=<config.yaml llms.planner.api_key>
python3 training/planner_sft/predict.py --backend openai \
  --base-url https://llm-8hwhlldwix34tnyh.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --model-name qwen3.7-flash --stratified 100 --seed 42 --run-name M0_qwen3.7-flash_s42
python3 scripts/eval/eval_planner_offline.py --pred data/planner/predictions/M0_qwen3.7-flash_s42.jsonl --write-report
# 3 mode2 端到端（约 150 次 LLM 调用）
python3 scripts/eval/eval_system_modes.py --modes mode2 --limit 50 --tag a1_baseline
```
