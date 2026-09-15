# 证据可信的变压器故障诊断多智能体系统 优化计划清单（v2）

> 制定日期：2026-09-15（v2 重写；v1 为 2026-09-14「复现两篇论文并组合」版本，已完成项归档于附录 D）
> 主叙事：**证据可信的诊断智能体**——不确定性驱动的主动规划（主线一）、声明级证据约束验证（主线二）、以验证信号为偏好的 Planner 优化（主线三）。
> 使用方式：每项任务前的 `[ ]` 用于勾选进度；「验收标准」未满足不进入下一阶段。任务编号 `A-1`、`B-2` 等在论文、commit message、docs 报告中统一引用。

---

## 0. 决策记录与总览

### 0.1 已拍板的四项决策（2026-09-15）

| 编号 | 决策 | 对计划的影响 |
|---|---|---|
| D-1 | 主叙事由「复现叶金涛 + 梁亨源两篇论文并组合」改为「证据可信的诊断智能体」，三章分别对应主动规划、声明验证、偏好优化 | 知识库 / 图谱 / 反思 / 两路检索降级为工程基座，不再作为论文贡献；五级模式按三条主线重定义 |
| D-2 | 主线三采用 DPO（偏好对来自 Validator 判定 + 真实工具执行结果），不做 GRPO | 不需要在线 rollout 环境；偏好对离线构造；训练可全部在百炼平台完成 |
| D-3 | 训练与部署仍用百炼平台 | 优先走百炼内置「模型调优」（SFT + DPO）→ 直接部署；魔搭 ms-swift + OSS + 自定义模型导入降为备选路线；基座由 Qwen3-VL-8B 改为纯文本 `qwen3-8b`（Planner 无视觉输入） |
| D-4 | 不复现林金山三智能体流程作为对照基线 | 相关工作中定性对比即可；对照组改为本系统内部的能力开关（五级模式）与通用方法基线（LLM 自由追问、固定顺序问询、现有 Validator） |

### 0.2 论文章节 ↔ 主线 ↔ 阶段映射

| 论文章节 | 研究问题 | 主线 | 计划阶段 | 核心对照 |
|---|---|---|---|---|
| 第 3 章 | 后验不确定时，下一步该问什么 / 调什么工具 | 主线一：不确定性驱动的主动诊断规划 | B | LLM 自由追问 / 固定顺序 / 随机 vs 贝叶斯 EIG 贪心 |
| 第 4 章 | 诊断结论中的每条声明是否有证据支撑、是否违反工程约束 | 主线二：声明级证据约束验证 | C | 现有整体评分 Validator vs 声明级核查 vs 核查 + 补证重规划 |
| 第 5 章 | 用验证信号优化 Planner，是否比 SFT 更能减少无依据结论与无效调用 | 主线三：偏好优化 Planner | D | M0 基座 / M1 SFT / M4 SFT+DPO，以及偏好来源消融 |
| 第 6 章 | 三项能力各贡献多少 | 系统级增量评测 | E | 五级模式（重定义） |

### 0.3 阶段总览

| 阶段 | 名称 | 目标 | 前置 | 核心产出 | 预估工时 | 资源 |
|---|---|---|---|---|---:|---|
| A | 基础设施收口与基线 | 关闭 v1 遗留、跑出基线数字、封存数据 | 无 | `docs/baseline.md`、改写后的训练集、人工标注 | 2 周（与 B 并行） | 少量 LLM 调用 |
| B | 主线一：主动诊断规划 | 校准的贝叶斯归因 + EIG 推荐 + Planner 接入 + 部分观测模拟评测 | A-1 | `eig.py`、模拟器、`docs/active_planning_eval.md` | 4-5 周 | 纯 CPU + 少量 LLM |
| C | 主线二：声明级验证 | claim-evidence 输出、约束核查器、故障注入评测集 D11 | A-1 | `claim_checker.py`、D11、`docs/validator_eval.md` | 4 周 | 少量 LLM 调用 |
| D | 主线三：DPO Planner | 百炼 SFT 基线 → 偏好对构造 → 百炼 DPO → 部署 | A-2、C-3 | D13 偏好对、部署模型 ID、`docs/planner_dpo_eval.md` | 5-6 周 | 百炼训练 + 调用 |
| E | 系统级增量评测 | 五级模式重定义、正式跑数、LLM 裁判 + 人工复核 | B、C、D | `docs/system_modes_eval.md` | 3 周 | LLM 调用 |
| F | 成果固化 | 报告、复现脚本、tag、论文实验章节初稿 | E | `v2-final` tag、六份报告 | 2-3 周 | 否 |

合计约 20-23 周。B 与 C 可并行（不同文件、不同数据）；D 的 D-1（百炼 SFT 基线）可在 A 结束后立即启动，不必等 B、C。

### 0.4 时间线（截至 2027-07 毕业）

| 时间窗 | 里程碑 |
|---|---|
| 2026-09-15 ~ 09-26 | A 收口；D-1 百炼 SFT 基线提交训练 |
| 2026-09-22 ~ 10-30 | B 主线一全部数字 |
| 2026-10-13 ~ 11-13 | C 主线二全部数字 |
| 2026-11-09 ~ 12-18 | D 主线三：偏好对、DPO 训练、部署、离线评测 |
| 2026-12-14 ~ 2027-01-15 | E 五级模式正式评测 |
| 2027-01-18 ~ 02-05 | F 固化，`v2-final` tag，实验章节初稿 |
| 2027-02 ~ 04 | 论文写作、补实验 |
| 2027-05 | 预答辩 |

### 0.5 五级系统模式（重定义，替换 [system_modes.py](/Users/ts/Desktop/thu/multi_Agent/src/graph/system_modes.py) 现有定义）

每级只在上一级基础上打开一项能力。知识库两路检索、`kg_search`、词法反思属于工程基座，从模式 2 起全部开启且不再单独归因。

| 模式 | 名称 | 新增能力 | 对应主线 |
|---|---|---|---|
| mode1 | `llm_only` | 无任何工具，LLM 直接回答 | 参照 |
| mode2 | `tool_base` | 全部五个工具 + 两路检索 + 图谱 + 词法反思；归因引擎用专家默认参数、不输出不确定性；Validator 为现有整体评分版；Planner 为通用基座 | 工程基座 |
| mode3 | `active_plan` | 归因引擎切换为校准参数并输出 EIG 推荐；Planner 提示词启用主动问询 / 补证策略 | 主线一 |
| mode4 | `claim_verify` | Generator 输出 claim-evidence 结构；Validator 切换为声明级约束核查 + 补证重规划路由 | 主线二 |
| mode5 | `dpo_planner` | Planner 切换为百炼部署的 DPO 模型 | 主线三 |

`resolve_mode` 的显式覆盖参数保留（`--planner-mode`、`--validator-mode`、`--attribution-mode`），用于交叉对照（例如 mode4 + baseline planner）。

---

## A 基础设施收口与基线

目标：关闭 v1 遗留的低成本事项，产出所有后续对照都要用的基线数字，把评测集彻底封存。

### A-1 基线数字（原 P0-3）

- [ ] 用 `qwen3.7-flash`（未微调）在 [retrieval_seed.jsonl](/Users/ts/Desktop/thu/multi_Agent/data/kb/eval/retrieval_seed.jsonl) test 切分上记录 Recall@1/3/5（two_way 与 naive 两档）。
- [ ] 用未微调 Planner 在 D8 test 切分抽 100 条（分层覆盖 8 类）跑 [eval_planner_offline.py](/Users/ts/Desktop/thu/multi_Agent/scripts/eval/eval_planner_offline.py) 7 项指标。
- [ ] 用 mode2（工程基座）在 D10 抽 50 条跑 [eval_system_modes.py](/Users/ts/Desktop/thu/multi_Agent/scripts/eval/eval_system_modes.py)，记录任务成功率、平均工具调用次数、平均轮次、Token 成本。
- [ ] 写入 `docs/baseline.md`，与 `baseline-v0` tag 对应。

验收标准：三组数字齐全，任何人按 README 可复现。

### A-2 训练问法口语化改写（原 P4-2 保留项）

- [ ] 执行 [rewrite_queries.py](/Users/ts/Desktop/thu/multi_Agent/scripts/planner_data/rewrite_queries.py) 去掉 `--dry-run`（估算约 1.2 元），产出 `task_seeds_rewritten.jsonl`。
- [ ] 人工抽 50 条确认数字 / 实体守卫生效（气体浓度、设备号、数据集名未被改写）。
- [ ] `export_sft.py --seeds task_seeds_rewritten.jsonl` 重导出；同时新增 `--format bailian` 导出百炼 ChatML（messages 多轮，含 tool 角色消息按百炼模板处理）。
- [ ] test 切分封存：写入 `data/planner/sft/SEALED.md` 记录 sha256，训练结束前不读取。

验收标准：改写后训练 / 验证集重导出完成，泄漏检查仍为 0，百炼格式文件通过控制台数据校验。

### A-3 人工标注（与 B、C 并行，不阻塞代码）

- [ ] D9 反思校验集 298 对填 `human_score`（两人标注，争议第三人裁定），计算 LexicalScorer 与人工的加权 Kappa。
- [ ] 图谱 [precision_sample.jsonl](/Users/ts/Desktop/thu/multi_Agent/data/kg/eval/precision_sample.jsonl) 56 条精度抽检人工确认（当前 AI 预标注 83%）。
- [ ] D10 196 条 `reference_points` 人工复核，`status` 由 `auto` 改为 `reviewed`。
- [x] 2026-09-15 核实数据来源与许可：R1 非 Kaggle（字段不符，改写为「来源未核实的 IEC 60599 标签 DGA 汇编」）；R3 589 条为公开基准并已知含重复/冲突；ETT 许可统一为 CC BY-ND 4.0（原仓库为准，脚注说明 HF 镜像差异）；R8 公开站点检索未匹配。已同步 `docs/data_inventory.md`、`docs/data_and_evaluation.md`。
- [ ] R8 `power_transformer_fault.csv` 若在 F 阶段前仍无法核实来源，从仓库移除并登记附录 B。
- [x] B-1 校准实验增加「R3 589 公开基准子集」单列结果（见 `docs/attribution_calibration.md` 附录，n=438，Top-1 0.662 → 0.669，ECE 0.113 → 0.091；注意该子集参与了全量学习，非严格外部检验）。（2026-09-15）
- [ ] 视时间补 IEC TC 10 案例库（IEEE DataPort, DOI 10.21227/h8g0-8z59）外部检验。

验收标准：四项标注 / 核实结果写入对应数据卡片。

### A-4 安全与仓库

- [ ] 阿里云 AccessKey `LTAI5t7f…` 到控制台撤销重建；GitHub 推送 token 撤销重建。
- [x] `.gitignore` 增加 `config.yaml`、`*.key`、`output/`、`data/planner/dpo/candidates/`。（2026-09-15）
- [ ] 87 项回归测试保持通过；新增测试文件命名 `tests/test_<阶段>_<模块>.py`。

### A-5 砍掉项落地

- [x] `run_matrix.sh` 顶部标注 v1 备选路线（保留脚本）。（2026-09-15）
- [ ] `system_modes.py` 旧五级定义由 E-1 替换。
- [x] 仓库清理（2026-09-15）：删除无引用的 Milvus 路线（`rag_engine.py`、`milvus_setup.py`、`utils/embedding.py`）、`remove_references.py`、`validate_dga_data.py`、两份早期 PPT 脚本、`data/rag.txt`、旧合成端到端报告；`timeseries.py` 收敛为唯一 3σ 实现，`app.py` / 两份评测脚本改为导入；`tool_registry.py` 移除已弃用的宽松 `validate_arguments`；`config.example.yaml` 移除 `knowledge_base` / `embedding` / `retriever` LLM / 未被读取的 `tools` 段；前端下线 Milvus 选项；`PROJECT_OVERVIEW.md` 按当前架构重写。
- [x] `docs/planner_training.md` 顶部加「v1 路线，见 v2 计划 D 阶段」提示，不删除。（2026-09-15）
- [ ] 在本文件附录 B 登记全部砍掉 / 降级项及理由。

---

## B 主线一：不确定性驱动的主动诊断规划

目标：让归因引擎不只给出后验，还给出「下一个最值得观测的征兆」；Planner 据此决定追问用户、调用工具补证，还是直接给结论。核心指标是同等准确率下更少的问询与调用。

涉及文件：[fault_attribution.py](/Users/ts/Desktop/thu/multi_Agent/src/tools/fault_attribution.py)（`FaultBayesianNetwork`）、[learn_cpt.py](/Users/ts/Desktop/thu/multi_Agent/scripts/learn_cpt.py)、[learned_params.json](/Users/ts/Desktop/thu/multi_Agent/data/real/dga/learned_params.json)、[planner.py](/Users/ts/Desktop/thu/multi_Agent/src/agents/planner.py)、[templates/planner/system.txt](/Users/ts/Desktop/thu/multi_Agent/templates/planner/system.txt)、[state.py](/Users/ts/Desktop/thu/multi_Agent/src/graph/state.py)。

### B-1 归因引擎补全与校准

- [x] `_compute_posterior` 支持负观测：`evidence[s] == False` 时乘 `1 - p_true`；`None` 仍跳过。入口 `fault_attribution` 在有 DGA 数值时把低于注意值的气体征兆自动写为 `False`（被遮蔽 / 缺失的气体不派生）；`FAULT_ATTR_NEG_EVIDENCE=0` 关闭以做消融。（2026-09-15）
- [x] 新增 `posterior_only(evidence, symptom_context)` 公开方法，返回归一化后验、熵 `H(F|E)`、Top-1 与 Top-2 概率差；`infer` 输出新增 `uncertainty` 块与 `evidence_negative`。（2026-09-15）
- [x] `learn_cpt.py --kfold 5` 分层 5 折交叉验证，逐折报告 Top-1 / Top-3 / NLL / Brier / ECE。（2026-09-15）
- [x] 置信度校准：验证折二分（前半拟合、后半 held-out 评估），温度 T 网格 0.5~4.5 目标 NLL；ECE 15 桶、Brier、可靠性图；校准参数写入 `learned_params.json.calibration`。（2026-09-15）
- [x] `_fuse_results` 融合权重改为按故障类别可学习（坐标下降网格，目标 NLL），写入 `learned_params.json.fusion_weights`；与固定 0.7 对照。（2026-09-15）
- [x] 输出 [attribution_calibration.md](/Users/ts/Desktop/thu/multi_Agent/docs/attribution_calibration.md) 与 `docs/figures/fig_b1_reliability.png`。（2026-09-15）

验收结果（2026-09-15，held-out 半折 5 折平均，n=3729 非 normal）：ECE 0.101 → 0.055（下降 46%），NLL 1.061 → 0.822，Top-1 0.635 → 0.665（+3.0pt，未下降），Top-3 0.915 → 0.975。负观测消融：关闭时 Top-1 0.596 / ECE 0.117，开启时 0.635 / 0.101。单元测试 [test_b1_attribution.py](/Users/ts/Desktop/thu/multi_Agent/tests/test_b1_attribution.py) 47 项通过。注意：`overload_overheating` 学到 w_f=0，即该类完全依赖三比值规则，报告需说明。

验收标准：校准后 ECE 相对校准前下降可量化；Top-1 准确率不下降超过 1 个点；单元测试覆盖负观测与温度缩放。

### B-2 期望信息增益（EIG）模块

- [x] 新建 [eig.py](/Users/ts/Desktop/thu/multi_Agent/src/tools/eig.py)：`EIG(s) = H(F|E) - Σ_v P(s=v|E)·H(F|E, s=v)`，`P(s=v|E) = Σ_F P(s=v|F)·P(F|E)`；两种计算基 `bayes`（严格互信息，非负）/ `fused`（系统实际置信分布，裁剪到 0）。（2026-09-15）
- [x] 成本表 [symptom_cost.json](/Users/ts/Desktop/thu/multi_Agent/data/real/dga/symptom_cost.json)：气体派生征兆 0、现场征兆 / 时序工具 1、局放检测 3；`VoI = EIG - λ·cost`，λ 默认 0.05。（2026-09-15）
- [x] 停止准则 `H < τ_H(0.8) | max VoI < ε(0.02) | rounds ≥ K(3)`，成本表 `stop` 段可配，`recommend(..., stop=...)` 可覆盖。（2026-09-15）
- [x] `fault_attribution` 返回 `uncertainty` 块新增 `recommendations[{symptom, eig, cost, voi, p_true, how_to_obtain, ask_hint, kg_evidence}]`、`suggested_action`、`suggested_tool`、`stop_reason`；`how_to_obtain` 映射 `oil_temp_elevated → call_tool:ett_forecast`、`load_elevated → call_tool:timeseries_anomaly`，其余 `ask_user`。`FAULT_ATTR_EIG=0` 或 `with_eig=False` 关闭（mode2 基座用）。（2026-09-15）
- [x] 与 `kg_search` 联动：成本表 `kg_nodes` → 图谱 `INDICATES / DETECTED_BY / PRODUCES / CAUSES` 边原文（最多 2 条）附在推荐上。（2026-09-15）
- [x] 单元测试 [test_b2_eig.py](/Users/ts/Desktop/thu/multi_Agent/tests/test_b2_eig.py) 34 项：EIG 非负、不超过当前熵、全部观测后无候选、λ 单调性、三种停止原因、call_tool 映射、遮蔽气体不触发伪三比值规则。（2026-09-15）

附带修复（2026-09-15）：三比值法在任一气体缺失时不再用 0 代入触发伪规则（`dga_analysis.ratios_incomplete`）；数据集不含 CO / 温度 / 负载 / 振动 / 局放 7 个征兆，`learn_cpt.py` 对其写入向 0.5 收缩的专家 CPT（κ=`EXPERT_SHRINK`=0.5，条目标记 `source: expert_shrunk`），否则原专家值在 EIG 中会压过数据学到的气体征兆。

验收结果（2026-09-15）：[eig_sanity_check.md](/Users/ts/Desktop/thu/multi_Agent/docs/eig_sanity_check.md) 20 条手工部分观测样例，Top-1 一致率 90%、Top-3 一致率 95%（κ=0.5；κ=1.0 时仅 60%/70%，κ≤0.6 稳定在 90%/95%）。预期集合为 AI 预填，**需人工复核后登记**。

验收标准：对 20 条手工部分观测样例，推荐征兆与领域专家直觉一致率不低于 80%（人工判定）。

### B-3 部分观测诊断模拟器（数据 D12）

- [x] 新建 `scripts/sim/partial_obs_sim.py`：从 5143 条真实 DGA 记录抽样，随机遮蔽 1-3 种气体或全部现场征兆，构造「初始观测 + 隐藏观测 + 真实标签」三元组。（2026-09-15；源为 3729 条非 normal 记录，现场征兆在源数据中无字段，统一记入 `unavailable_symptoms`）
- [x] 模拟器接口：`reveal(symptom) -> bool | None`（None 表示该记录无此字段）；记录每次揭示的成本。（`PartialObsEnv`：`reveal / reveal_gas / context / masked_remaining / history / total_cost / n_queries`；成本读 `symptom_cost.json`，不可获取征兆仍计费）
- [x] 按遮蔽比例分三档（轻 / 中 / 重）各 500 条，按故障类别分层；只用真实 DGA 派生，合成数据不入。（light/medium/heavy 各遮蔽 1/2/3 种气体；`data_guard.assert_not_synthetic` 校验源路径）
- [x] 数据卡片 `data/eval/d12/DATA_CARD.md`。（含遮蔽方案、标签分布对照、用途、局限）

验收标准：1500 条模拟样例可复现（固定 seed），标签分布与原始数据一致。

验收结果（2026-09-15）：`--build` 生成 1500 条，seed 20260915，sha256 `9e2a35b3…`；`--check` 重建比对一致，标签分布最大偏差 0.097%（insulation 13.8% / overheating 40.0% / PD 28.6% / short 17.6%，与源一致）；平均隐藏征兆数 light 2.02 / medium 3.47 / heavy 4.60；`tests/test_b3_sim.py` 33 项通过。局限：现场征兆（温度 / 负载 / 振动 / 局放）无法揭示，B-5 主动规划评测只能在 7 个气体征兆空间内进行，需在论文中说明。

### B-4 Planner 接入主动策略

- [ ] `state.py` 增加 `pending_questions`、`asked_symptoms`、`user_answers` 字段；`workflow.py` 增加「追问 → 用户回答 → 重新归因」的有限循环（最多 K 轮），评测时用模拟器代替用户。
- [ ] `templates/planner/system.txt` 新增 `active` 变体：说明 `uncertainty` 块含义、三种动作的选择规则、追问话术要求（一次只问一个征兆、给出为什么问）。
- [ ] Planner 输出增加 `action: ask | call_tool | conclude` 与 `rationale`；`_validate_steps` 校验 `ask` 时必须携带 `symptom` 字段且属于推荐列表。
- [ ] 配置开关 `attribution_mode: expert | calibrated` 与 `planner_strategy: free | active`。
- [ ] 前端 [app.py](/Users/ts/Desktop/thu/multi_Agent/app.py)：展示后验分布、熵、推荐征兆与追问话术；演示案例新增 2 个「信息不足 → 追问 → 确诊」流程。

验收标准：mode3 在 D12 抽 30 条上能完整走通追问循环，轨迹日志记录每轮 EIG 与动作。

### B-5 对照实验与评测

- [ ] 新建 `scripts/eval/eval_active_planning.py`，在 D12 上运行以下策略：`random`（随机问）、`fixed`（按 IEC 60599 常规顺序问）、`llm_free`（当前 Planner 自由追问）、`eig_greedy`（本文）、`eig_cost`（EIG 减成本）；可选 `llm_sampled_eig`（让 LLM 采样估计后验再算 EIG，对标 BED-LLM 思路）。
- [ ] 指标：停止时 Top-1 / Top-3 准确率、平均问询次数、平均获取成本、追问命中率（问的是否在最高 VoI 前 2）、每轮后验熵曲线、校准 ECE。
- [ ] 消融：有 / 无校准（B-1）、有 / 无成本项、不同 τ_H。
- [ ] LLM 相关策略每条固定 seed 跑 2 次取均值；EIG 策略确定性。
- [ ] 结果写入 `docs/active_planning_eval.md`，图表：准确率-问询次数曲线（`fig_b5_acc_vs_queries.png`）、熵下降曲线。

验收标准：`eig_greedy` 在同等 Top-1 下平均问询次数低于 `llm_free` 与 `fixed`，差异有配对检验 p 值；结论写入报告。

### B-6 章节素材

- [ ] 整理算法伪代码（EIG 计算、停止准则、动作选择）与符号表。
- [ ] 与 Active Task Disambiguation、BED-LLM、When Should AI Ask、InfoGatherer 的差异说明：本文用领域概率模型精确计算 EIG，而非 LLM 采样估计；引入工程获取成本。

---

## C 主线二：声明级证据约束验证

目标：Validator 从「给草案打整体分」升级为「逐条核对声明与证据、判定违反哪类工程约束」，并驱动补证或重规划。

涉及文件：[generator.py](/Users/ts/Desktop/thu/multi_Agent/src/agents/generator.py)、[validator.py](/Users/ts/Desktop/thu/multi_Agent/src/agents/validator.py)、[templates/generator/](/Users/ts/Desktop/thu/multi_Agent/templates/generator/)、[prompts.py](/Users/ts/Desktop/thu/multi_Agent/src/utils/prompts.py)、[workflow.py](/Users/ts/Desktop/thu/multi_Agent/src/graph/workflow.py)。

### C-1 声明与证据的输出规范

- [ ] 定义 `docs/claim_schema.md`：`claims: [{id, text, type: observation | inference | recommendation | safety, evidence: [{source: tool | kb | kg | user, ref, span}]}]`，`ref` 为工具调用 id / 文本块 uuid / 图谱边 id / 用户轮次号；`span` 为证据原文片段。
- [ ] 约束类型定义：`DATA`（数值与工具输出不一致、单位错误、比值编码错误）、`EVIDENCE`（引用不存在、片段不含该内容、历史案例冒充当前检测）、`APPLICABILITY`（证据设备 / 数据集 / 时间窗与当前任务不匹配）、`SAFETY`（处置建议缺少前提条件、与检测结果矛盾）。每类给正例、反例各 3 条。
- [ ] Generator 新增 `templates/generator/system_claims.txt`，要求先输出 JSON 声明列表再渲染自然语言答案；`_llm_generate` 增加 `output_mode: text | claims`。
- [ ] 模板回退 `_template_generate` 同步产出声明（规则拼装，保证 LLM 不可用时链路不断）。

验收标准：D10 抽 30 条，Generator 在 `claims` 模式下 JSON 合法率不低于 95%，每条 claim 至少一条 evidence 引用。

### C-2 约束核查器

- [ ] 新建 `src/agents/claim_checker.py`，两层核查：
- [ ] 确定性层（无 LLM）：数值一致性（声明中的气体浓度、概率、预测值、horizon 与 `state.tool_results` 逐项比对，容差可配）；引用存在性（`ref` 在本轮工具返回中可找到，`span` 为其子串）；适用性（声明中的设备号、数据集名、时间范围与工具入参一致）；安全前提（推荐类声明若涉及停电 / 吊罩 / 更换，必须存在对应 `observation` 声明支撑）。
- [ ] 语义层（LLM）：对通过确定性层的 `inference` 声明做 NLI 判定 `entail | contradict | unsupported`，提示词只给该声明与其引用的证据片段，不给全文。
- [ ] 输出 `ValidationResult` v2：新增 `claim_verdicts: [{claim_id, verdict, violated_constraints: [...], detail}]`、`unsupported_ratio`、`violation_counts`；保留旧字段兼容前端。
- [ ] 判定规则：任一 `SAFETY` 或 `DATA` 违反 → `REVISION`；`unsupported_ratio > 0.3` → `REVISION`；两轮修订仍不通过 → `ABSTAIN`（输出「证据不足，建议补充 X」而非强行结论）。
- [ ] 单元测试：四类约束各 5 个构造样例。

验收标准：确定性层对构造样例检出率 100%，误报 0；语义层在 30 条人工标注声明上与人工一致率不低于 85%。

### C-3 补证与重规划路由

- [ ] `workflow.py` 路由新增：`unsupported` 且证据可补 → 回 Planner 并携带 `missing_evidence: [{claim_id, suggested_tool, suggested_query}]`；`contradict` → 回 Generator 修订；`ABSTAIN` → 结束。
- [ ] Planner `active` 模板增加对 `missing_evidence` 的处理规则（优先调用建议工具，最多补证 1 轮）。
- [ ] 每次路由记录到轨迹日志：触发原因、补证工具、额外 Token 与耗时。

验收标准：D10 抽 30 条，补证路由触发的调用中不出现重复调用同一工具同一参数。

### C-4 故障注入评测集（数据 D11）

- [ ] 新建 `scripts/eval/build_d11_fault_injection.py`：以 [oracle_mode5.jsonl](/Users/ts/Desktop/thu/multi_Agent/data/eval/d10/results/oracle_mode5.jsonl) 中 196 条正确草案为底，自动注入四类错误各 50 条：篡改数值（气体浓度 ±30%、概率互换）、伪造引用（不存在的 chunk uuid / 编造片段）、换设备或换数据集（ETTh1 → ETTm1、设备号错位）、删安全前提（保留「立即停电吊罩」删去支撑它的高乙炔观测）。
- [ ] 加 100 条未注入的干净草案作为负样本（考察误报）。
- [ ] 每条记录 `injected: bool, type, location, original`；人工抽 40 条确认注入确实构成错误。
- [ ] 数据卡片 `data/eval/d11/DATA_CARD.md`。

验收标准：300 条（200 注入 + 100 干净），人工抽检注入有效率不低于 95%。

### C-5 对照实验与评测

- [ ] 新建 `scripts/eval/eval_validator.py`，三组对照：`v1`（现有整体评分 Validator）、`v2_check`（声明级核查，不路由）、`v2_route`（核查 + 补证重规划）。
- [ ] 指标：错误通过率（注入样本被判 PASS 的比例）、分类型检出率、干净样本误报率、无依据结论率（最终答案中 `unsupported` 声明占比）、约束违反率、弃答率、额外调用次数与 Token 成本。
- [ ] 语义层 LLM 判定抽 10% 人工复核一致率。
- [ ] 结果写入 `docs/validator_eval.md`，图表：分类型检出率柱状图、通过率-成本散点。

验收标准：`v2_check` 相对 `v1` 错误通过率显著下降且干净样本误报率不高于 10%；`v2_route` 的无依据结论率进一步下降。

### C-6 章节素材

- [ ] 整理约束类型定义表、核查流程图（Mermaid）、路由状态机。
- [ ] 与 MAST「验证失效」分类、RT4CHART 声明级核查的差异说明：本文证据源为异构工具输出 + 文献 + 图谱，约束类型面向电力工程。

---

## D 主线三：以验证信号为偏好的 Planner 优化（百炼 SFT + DPO）

目标：把主线二 Validator 的声明判定与真实工具执行结果转成偏好对，在百炼平台对 Planner 做 SFT → DPO 两阶段训练，验证「验证信号驯化 Planner」是否比 SFT 更能减少无效调用与无依据结论。

基座：`qwen3-8b`（百炼支持 `efficient_sft` 与 `dpo_lora`；Planner 无视觉输入，不再用 Qwen3-VL）。若控制台列表变化，按「文本 8B 级、同时支持 SFT 与 DPO LoRA」原则替换并记录。

### D-1 百炼 SFT 基线（M1）

- [ ] A-2 产出的百炼 ChatML 训练集 + 验证集上传（`purpose=fine-tune`），记录 `file_id`。
- [ ] 控制台或 API 创建任务：`training_type=efficient_sft`，`n_epochs=3`、`batch_size=16`、`max_length=4096`、`learning_rate` 取平台默认，`split` 不用（已自带验证集）。
- [ ] 训练完成后部署，记录模型 ID 到 `config.yaml` 的 `llms.planner_finetuned`；`scripts/probe_llm_endpoint.py` 验证 `tool_calls` 能被 `_build_plan_from_tool_calls` 解析。
- [ ] 在 D8 test 切分上跑 `eval_planner_offline.py`，与 A-1 的 M0 数字并列写入 `docs/planner_dpo_eval.md` 第一节。
- [ ] 训练脚本、超参与 job_id 记录到 `training/planner_bailian/README.md`；新建 `training/planner_bailian/submit_job.py`（封装文件上传、任务创建、状态轮询、部署）。

验收标准：M1 部署可调用；离线 7 项指标相对 M0 有提升；50 条端到端任务跑通。

### D-2 偏好对构造（数据 D13）

- [ ] 采样 prompt：从 D8 train 切分抽 1500 条任务（分层覆盖单工具 / 多工具 / 追问 / 错误恢复 / 无需工具），加 B-3 模拟器生成的 500 条部分观测任务。
- [ ] 候选生成：对每条 prompt 用 M1 模型（temperature 0.7 / 1.0）各采样 2 个 Planner 输出，加 M0 基座 1 个，共 5 个候选；在真实工具环境执行（复用 [retriever.py](/Users/ts/Desktop/thu/multi_Agent/src/agents/retriever.py) 的执行与 `exec/business_success` 记录）。
- [ ] 候选打分（`scripts/planner_data/score_candidates.py`）：分项打分并加权求和，权重可配：格式合法（0/1）、工具与金标一致（0/1）、参数 Schema 通过（0/1）、工具业务成功（0/1）、下游声明忠实度（把候选轨迹送 Generator + C-2 核查器，取 `1 - unsupported_ratio`）、追问是否在推荐列表（B-2，0/1）、调用成本惩罚（`-0.1 × 多余调用数`）。
- [ ] 配对规则：同一 prompt 内分差 ≥ 0.3 的最高与最低候选组成 `(chosen, rejected)`；分差不足则丢弃该 prompt；每 prompt 最多 1 对。
- [ ] 偏好来源消融准备：额外导出两份子集，`D13-exec`（只用格式 + 工具 + 参数 + 业务成功打分）与 `D13-full`（加声明忠实度 + 推荐一致 + 成本），用于 D-4。
- [ ] 人工抽 100 对复核偏好方向正确率；泄漏检查：prompt 的 `seed_source` 不得出现在 D8 test 与 D10。
- [ ] 导出百炼 DPO 格式（`prompt / chosen / rejected` jsonl，prompt 含 system + 历史轮次）；数据卡片 `data/planner/dpo/DATA_CARD.md`。

验收标准：有效偏好对不少于 1200 条；人工复核偏好方向正确率不低于 90%；泄漏 0。

### D-3 百炼 DPO 训练（M4）

- [ ] 以 D-1 产出的 M1 模型 ID 为 `model`，`training_type=dpo_lora`，`n_epochs=2`、`batch_size=16`、`max_length=4096`；`dpo_beta` 若控制台可配取 0.1。
- [ ] 训练曲线：导出 reward margin / eval loss 截图归档到 `docs/figures/fig_d3_dpo_curve.png`。
- [ ] 部署 M4，记录模型 ID；`probe_llm_endpoint.py` 验证。
- [ ] 若百炼 DPO 对 `qwen3-8b` 不可用：备选为魔搭 A10 上 `swift rlhf --rlhf_type dpo --train_type lora --lora_rank 8`，LoRA 导出后走 OSS → 自定义模型导入（v1 路线 §7 步骤）。备选触发即在此处记录原因。

验收标准：M4 部署可调用，dev 上 reward margin 为正且稳定。

### D-4 对照矩阵与离线评测

- [ ] 矩阵：M0 基座、M1 SFT、M4-exec（D13-exec 训练）、M4-full（D13-full 训练）；M1 与两个 M4 各跑 2 个 seed（百炼任务重复提交）取均值。
- [ ] 离线指标：`eval_planner_offline.py` 7 项 × 8 类别；新增「追问推荐一致率」（B-2）与「平均调用次数」。
- [ ] 端到端指标：在 D10 test 上以 mode4 配置替换 Planner，记录任务成功率、无依据结论率（C-2 核查器）、约束违反率、平均调用次数。
- [ ] 结果写入 `docs/planner_dpo_eval.md`，关键结论：M4-full 相对 M1 是否在无依据结论率与调用次数上有增益；M4-full 相对 M4-exec 是否证明「声明忠实度信号」有效。

验收标准：四组数字齐全；M4-full 相对 M1 至少在两项核心指标上显著改善，否则在报告中如实记录并分析。

### D-5 章节素材

- [ ] 偏好构造流程图、打分函数定义表、消融设计说明。
- [ ] 与 ToolRL / ARTIST / Deep-DxSearch 的差异说明：本文奖励信号来自下游声明级验证器而非仅工具执行结果；采用 DPO 离线优化以适配平台约束。

---

## E 系统级增量评测

### E-1 五级模式重定义

- [ ] 重写 `system_modes.py` 为 §0.5 定义；`SystemModeSpec` 新增 `attribution_mode`、`planner_strategy`、`validator_mode`、`generator_output` 字段。
- [ ] `build_agents` 按字段构建；`eval_system_modes.py` 与 `app.py` 下拉框同步更新。
- [ ] OraclePlanner 校验在新 mode2 上重跑一次，确认 `task_success` 校验逻辑仍成立。

验收标准：五级模式各跑 D10 抽 5 条冒烟通过。

### E-2 正式评测

- [ ] D10（196 条，A-3 复核后）× 五级模式，LLM 相关模式各 2 次取均值。
- [ ] 指标：任务成功率四要素、无依据结论率、约束违反率、弃答率、平均问询次数、平均工具调用次数、平均延迟、Token 成本。
- [ ] 交叉对照：mode4 + baseline planner、mode5 + v1 validator，用于分离主线二与主线三的贡献。
- [ ] LLM 裁判 + 10% 人工复核一致率。
- [ ] 结果写入 `docs/system_modes_eval.md`；图表：五级模式增量柱状图、成功率-成本散点。

验收标准：增量表完成，每一级新增能力的贡献可量化并附置信区间。

### E-3 鲁棒性附加实验（时间允许时）

- [ ] 在 D11 注入样本上跑 mode2 与 mode5 端到端，报告错误最终进入答案的比例。
- [ ] 在 D12 重遮蔽档上跑 mode2 与 mode3，报告准确率与问询次数差异。

---

## F 成果固化

- [ ] 六份报告齐全：`baseline.md`、`attribution_calibration.md`、`active_planning_eval.md`、`validator_eval.md`、`planner_dpo_eval.md`、`system_modes_eval.md`。
- [ ] 数据卡片：D11、D12、D13 新增；D8、D10 更新复核状态。
- [ ] `reproduce.sh` 更新为 A → B → C → E 一键（D 需百炼账号，提供 `submit_job.py` 与说明）。
- [ ] README 更新架构图与三条主线说明；`docs/figures/fig3_1_architecture.png` 重绘加入 EIG 模块与声明核查器。
- [ ] 代码打 tag `v2-final`，与 `baseline-v0` 对照。
- [ ] 已知局限：DGA 标签映射不代表真实故障部位；征兆成本表为专家设定；D11 为自动注入而非真实错误；评测依赖 LLM 裁判；DPO 偏好对由自动打分构造。
- [ ] 论文第 3-6 章实验小节初稿，每章引用对应报告的表与图。

---

## 附录 A 数据清单（v2）

| 编号 | 数据 | 规模 | 阶段 | 用途 | 状态 |
|---|---|---|---|---|---|
| D3 | 检索评测集 | 1209 组 | 基座 | Recall@k 基线 | 已有 |
| D4 | 故障图谱 | 90 节点 / 190 边 | 基座 | `kg_search`、追问话术 | 已有 |
| D8 | 工具调用训练集 | 3482 单轮 + 多轮 | A-2 / D-1 | SFT、偏好 prompt 来源 | 待改写重导出 |
| D9 | 反思评分校验集 | 298 对 | A-3 | Kappa | 待人工分 |
| D10 | 端到端评测集 | 196 条 | E | 五级模式 | 待人工复核 |
| D11 | 故障注入评测集 | 300 条 | C-4 | 验证器对照 | 新建 |
| D12 | 部分观测模拟集 | 1500 条 | B-3 | 主动规划对照 | 新建 |
| D13 | Planner 偏好对 | ≥1200 对 | D-2 | DPO | 新建 |

## 附录 B 砍掉与降级项登记

| 项 | 处置 | 理由 |
|---|---|---|
| D7 查询分解集、Q0/Q1 双 Planner | 砍掉 | 与三条主线无关；论文范围收窄到工具调用 Planner |
| M2 / M3 结构 / 领域 token 加权 SFT | 砍掉 | 百炼内置 SFT 不暴露 token 级权重；v2 对照改为 SFT vs DPO；`plugin_loss_scale.py` 保留在仓库作为备选路线附件 |
| M3-random 对照 | 随 M3 砍掉 | 同上 |
| 稠密向量 + Qwen3-Reranker、Milvus 四层 | 不做 | 检索质量非论文贡献，两路 BM25 + RRF 已满足工程基座 |
| 图片描述 / 表格改写 / LLM 摘要 | 不做 | 同上 |
| LLM 图谱抽取补图、模糊消歧、LightRAG 对比 | 不做 | 图谱只作追问话术与证据来源，规模够用 |
| 林金山三智能体流程复现 | 不做（D-4） | 相关工作定性对比 |
| 魔搭 A10 + OSS + 自定义模型导入 | 降为备选 | D-3 百炼 DPO 不可用时启用 |
| LLMScorer 反思评分 | 降级 | 词法评分器作为基座固定配置；Kappa 仅作数据卡片附注 |

## 附录 C 关键风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 朴素贝叶斯独立假设使 EIG 高估 | 推荐征兆偏差 | B-1 交叉验证 + 校准；报告中说明假设；可选加征兆间相关性修正 |
| 征兆成本表主观 | 成本项结论可争议 | 报告 λ 敏感性分析；`eig_greedy`（无成本）作为主结果 |
| 声明 JSON 输出降低答案流畅度 | 用户体验 | 先 JSON 后渲染，前端只展示自然语言 + 可展开证据 |
| 确定性核查容差设置不当 | 误报或漏报 | D11 干净样本误报率作为硬指标，容差在 dev 上调 |
| 自动打分构造的偏好对方向错误 | DPO 学偏 | D-2 人工抽 100 对复核；分差阈值 0.3 过滤模糊对 |
| 百炼 DPO 对目标基座不可用 / 超参不可配 | 训练受阻 | D-3 备选路线；提前在控制台核对支持列表 |
| DPO 后模型「变平庸」（过度保守、少调用） | 端到端成功率下降 | 监控不必要调用率与追问正确率双向指标；`dpo_beta` 0.1 起步 |
| LLM 裁判偏差 | 结论不可信 | 每项裁判抽 10% 人工复核并报告一致率 |
| 训练 / 测试泄漏 | 评测失真 | D13 prompt 按 `seed_source` 与 D8 test、D10 隔离 |

## 附录 D v1 已完成资产（归档，直接复用）

| v1 阶段 | 复用资产 |
|---|---|
| P0 | 7 项缺陷修复、`exec/business_success` 分离、轨迹日志、`assert_not_synthetic`、数据清单、`baseline-v0` tag、87 项回归测试 |
| P1 | MinerU 单元清洗、DP 分块 5404 块、子块 / 锚点两路 + RRF、`local_kb mode`、D3 1209 组、`docs/kb_ablation_test.md` |
| P2 | 8 实体 / 9 关系 schema、规则抽取 90 节点 / 190 边、`kg_search` 与三智能体接入、忠实度评测 HitAll 100% |
| P3 | `ReflectionModule`、`LexicalScorer`、D9 298 对、`docs/reflection_eval.md` |
| P4 | 3482 单轮 + 多轮 / 错误恢复轨迹、分组切分 2311/439/732、三格式导出、数据卡片、`rewrite_queries.py` |
| P5 | `eval_planner_offline.py`（7 指标 × 8 类别）、`planner_mode` 开关、`provider: dashscope`、`domain_terms.json`（609 项，可复用于 C-2 实体一致性核查） |
| P6 | D10 196 条、`eval_system_modes.py`（任务成功率四要素、Token 成本）、前端轨迹 / 图谱 / 反思展示 |
| P7 | README、`reproduce.sh`、requirements、四份数据卡片 |

