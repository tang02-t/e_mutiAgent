# 离线推进进展报告（P2-5 / P3 / P4）

> 日期：2026-09-15
> 背景：百炼 LLM / embedding 接口返回 `403 AllocationQuota.FreeTierOnly`，按指令跳过所有依赖接口的任务，先推进可离线完成事项。
> 对应提交：`91bc0bc`（P2）、`a322ff3`（P2-5 + P3）、本次提交（P4 离线种子 + 导出）。

## 1. P2-5 图谱评测与规则修订

- 自动评测（`scripts/kg/build_kg_eval.py`）：304 条问答种子（canonical + alias 两种问法），衡量 `kg_search` 对图谱的忠实度，实体定位率 / HitAny / HitAll 均 100%。
- 三元组精度抽检：56 条分层抽样，AI 预标注 correct=42 / wrong=6 / unsure=8。根因修复后（英文缩写边界 bug、分句边界、并列枚举误配、DL/T 911 别名过泛、伪 LOCATED_IN），当前图谱保留的 47 条中 correct=39 / wrong=2 / unsure=6（严格 83.0%，宽松 95.7%）。
- 图谱由 95 节点 / 206 边收敛为 90 节点 / 190 边（≥2 篇支持 70 条）。
- 人工标注入口：`data/kg/eval/precision_sample.jsonl` 的 `label` 字段（`ai_prelabel` 仅供复核）。

## 2. P3 反思模块

- `src/agents/reflection.py`：0-3 评分（论文表 2.2）、<2 丢弃、全丢触发一次改写重检索、同章节邻块补召回、最多 2 轮；评分器可插拔（`LexicalScorer` IDF 加权基线 / `LLMScorer`，后者接口失败时自动回退）。
- `src/tools/local_kb.py` 新增 `get_chunk / neighbors / section_chunks / get_local_kb`，"同章节"定义为 `section_path` 集合有交集。
- 接入：`run_diagnosis_workflow(..., reflector=...)` 在 `retriever → reflection → generator` 插入节点；`app.py` 侧栏"反思模块"开关；`scripts/eval/eval_end2end.py --reflection {off,lexical,llm}`，报告新增反思决策分布、保留率、改写触发数。
- LexicalScorer 校准：金标块 3 分占 97%，随机块 0 分占 79%，BM25 检回非金标约 11% 判 <2。
- D9 校验集：`scripts/kb/build_reflection_evalset.py` 生成 298 对（gold 105 / retrieved_nongold 105 / same_doc_random 43 / global_random 45），待人工填 `human_score`，脚本 `--eval-only` 计算一致率与二次加权 Kappa。
- 测试：`tests/test_p0_fixes.py` 第 10 组 17 项（丢弃、补召回、全丢改写、no_input、all_dropped、禁用、评分器异常回退、工作流位置），全部 58 项通过。

## 3. P4 规划模块数据（离线种子）

`scripts/planner_data/build_task_seeds.py --execute` 产出 `data/planner/seeds/task_seeds.jsonl`：

| 类别 | 数量 | 占比（不含 composite） | 来源 | 金标动作 |
|---|---|---|---|---|
| fact | 1207 | 37.2% | retrieval_seed（金标块） | `rag_search` |
| reasoning | 403 | 12.4% | kg_qa_seed 1 跳 + graph.json 2 跳链路 | `kg_search`（relations/direction/hops） |
| numeric_tool | 952 | 29.4% | 5143 条真实 DGA、ETT 四数据集、ETT 序列窗口 | `fault_attribution` / `ett_forecast` / `timeseries_anomaly` |
| no_tool | 340 | 10.5% | 33 条手写模板 × 语气变体 | `direct_answer` |
| insufficient | 340 | 10.5% | 14 条缺参数模板 × 语气变体 | `ask_user(missing)` |
| composite | 240 | — | DGA+文献 / DGA+图谱 / 预测（不重复调异常） | 多步 |

关键设计：

- 对比样本：`dga_from_context` 120 条与 insufficient 同问法，但 `context.dga` 有数据 → 必须调 `fault_attribution` 并从上下文填 `dga_data`；ETTh/ETTm horizon 按采样间隔换算（6 小时 → 6 / 24）；`ett_anomaly` 复合类只调 `ett_forecast`（自带 3σ），用于学"不多调工具"。
- 校验：全部 tool_call 通过 `validate_arguments_strict`，并在真实工具环境执行，剔除业务失败 1 条（空检索）。
- 切分：按 `group_key`（文献 doc / 图谱实体 / DGA 设备 / ETT 月份 / 模板 id）分组，事实类沿用知识库 split，推理类沿用图谱 split；train 2311 / dev 439 / test 732，跨集 bigram-Jaccard>0.9 泄漏 0 对。
- 导出（`scripts/planner_data/export_sft.py`）：系统提示与线上 `PLANNER_SYSTEM_PROMPT()` 完全一致，user 走 `render_planner_user + _render_context`；三种格式 `swift_*.jsonl`（messages+tools+tool_calls）、`lf_*.json`（LLaMA-Factory function-calling）、`jsontext_*.jsonl`（与 json_text 回退路径一致）；数据卡片 `data/planner/sft/DATA_CARD.md`。

已知限制：所有查询为模板问法（`needs_llm_rewrite=true`），多样性不足；`composite` 第 2 步 query 以原始标签占位，需在有 LLM 轨迹后替换为第 1 步真实返回。

## 4. 剩余事项均需百炼接口

| 阶段 | 阻塞任务 | 所需能力 |
|---|---|---|
| P0-3 | 三组基线数字（Recall / Planner 正确率 / 端到端成功率） | LLM |
| P1 | 图片描述与分类、LLM 摘要、评测集 LLM 改写与"新手用户"过滤、稠密向量 + Qwen3-Reranker | VL + embedding + LLM |
| P2 | 阶段 2 LLM 抽取（1677 候选句）、实体模糊合并、LLM 裁判胜率对比 | LLM |
| P3 | `LLMScorer` 评分、与人工 D9 的 Kappa、有/无反思端到端对比 | LLM |
| P4 | P4-2 口语化"模拟用户"改写与复合问题连贯性判断、D8 多轮 / 错误恢复样本（需真实 LLM 轨迹） | LLM |
| P5 | 魔搭 ms-swift LoRA-SFT（Qwen3-VL-8B-Instruct）、百炼部署 | 训练 + 部署配额 |
| P6/P7 | 五级模式对比、消融、成果固化 | LLM |

恢复接口后的第一批动作（按依赖顺序）：P0-3 基线 → P4-2 改写（可直接对 `task_seeds.jsonl` 批量改写并重跑 `export_sft.py`）→ P3 LLMScorer 一致性 → P2 LLM 抽取补图谱 → P5 训练。
