# 数据清单与角色标注（P0-2）

> 生成日期：2026-09-14
> 目的：在开始构建训练数据与评测集之前，明确每份数据的来源、规模、许可状态、已知问题，以及它在实验中允许扮演的角色。
> 角色定义：
> - 训练监督：可作为模型学习目标（如工具调用轨迹）的原始材料
> - 执行环境：工具运行时读取的数据（如 ETT 历史序列、贝叶斯参数）
> - 评测标签：用于判断业务结果是否正确的金标
> - 演示数据：仅用于前端演示或流程连通性检查，禁止进入任何评测集或训练集
> - 知识源：RAG / 图谱的入库文档

---

## 1. 真实来源数据

| 编号 | 路径 | 规模 | 来源 | 许可 | 允许角色 | 禁止角色 |
|---|---|---:|---|---|---|---|
| R1 | `data/dga/dga_dataset.csv` | 4150 行；分号分隔、逗号小数 | 来源未核实。早期项目说明写作「Kaggle」，2026-09-15 核对后不成立：Kaggle「Power Transformers FDD and RUL」为 4 气体时序、4 类标签，与本文件（H2/CH4/C2H6/C2H4/C2H2 五气体、IEC 60599 六类标签 PD/D1/D2/T1/T2/T3）不符；未找到匹配的发布页 | 未核实；论文与报告中不得写作 Kaggle，应写「来源未核实的 IEC 60599 标签 DGA 汇编」 | 执行环境（学习 CPT）、评测标签（气体→IEC 故障性质） | 直接作为工具调用监督 |
| R2 | `data/dga/data.xlsx` | 2321 行；Sheet1 | GitHub `alan-456/transformer-fault-dataset`，README 说明为数据库 + 8 篇以上中国高校学位论文（大连理工、上海交大、东南大学等）+ 部分中国电网数据汇集，`data.xlsx` 为国内数据 | 仓库未声明许可，仅可在学术引用范围内使用 | 同 R1 | 同 R1 |
| R3 | `data/dga/dataset_(589).xlsx` | 589 行 | 同 R2 仓库，README 标注为国外文献汇集。2024–2026 年多篇论文（KGRAT / Electronics 2026、GraphSmin、GCDGA 等）以此 589 条为公开基准；KGRAT 审计报告其中含 7 组完全重复、5 组跨标签冲突，且无设备、来源、采样时间字段 | 同 R2 | 同 R1；另可单列为「589 公开基准子集」做与文献可比的校准报告 | 同 R1 |
| R4 | `data/real/dga/dga_records.jsonl` / `.csv` | 5143 条（R1-R3 合并去重后） | 由 `scripts/convert_real_dga.py` 生成 | 继承 R1-R3 | 执行环境、评测标签、任务种子 | 见 §3 已知问题 |
| R5 | `data/real/dga/learned_params.json` | 8 类故障先验 + CPT | 由 `scripts/learn_cpt.py` 从 R4 学习 | 派生 | 执行环境（`FAULT_ATTR_PARAMS`） | 评测标签 |
| R6 | `data/ETT-small/ETTh1.csv`、`ETTh2.csv` | 各 17420 行，小时级 | ETDataset（Zhou et al., Informer, AAAI 2021），北航团队与北京国网富达合作采集，两站 2016-07 至 2018-07 油温与负载 | CC BY-ND 4.0（以 GitHub 原仓库 `zhouhaoyi/ETDataset` 的 LICENSE 为准；HuggingFace 镜像 `ETDataset/ett` 标注为 CC BY 4.0，与原仓库不一致，项目统一采用原仓库写法并在论文脚注说明） | 执行环境（油温预测）、评测标签（未来真实 OT）、任务种子 | 不能与 DGA 拼成同一设备；ND 条款下不得分发修改后的数据文件，仓库只保留原始 CSV |
| R7 | `data/ETT-small/ETTm1.csv`、`ETTm2.csv` | 各 69680 行，15 分钟级 | 同 R6 | 同 R6 | 同 R6 | 同 R6；ETTh 与 ETTm 疑似同源序列，不能视为独立设备做泛化测试 |
| R8 | `data/external_transformer/power_transformer_fault.csv` | 1866 行，106 台变压器，含 9 种气体、油质、糠醛及 4 种经典比值法诊断列 | 来源未记录。2026-09-15 按字段结构（`Transformer;Measurement Number`、糠醛、油质、Duval/Dornenburg/Rogers/IEC 诊断列）检索 Kaggle、IEEE DataPort、Mendeley Data 均未匹配到发布页 | 未核实 | 候选：评测标签（Duval / Rogers 等诊断列作为规则法参考）、任务种子（含设备编号与多次测量，可构造趋势类任务） | 未核实来源前不入训练集与评测集；F 阶段固化前若仍无法核实，从仓库移除并在附录 B 登记 |
| R9 | `data/fast_md/*/` | 182 个 MinerU 解析目录（每个含 `full.md`、`content_list_v2.json`、`layout.json`、`images/`） | 中文期刊/学位论文 PDF 解析产物 | 学术引用范围 | 知识源（RAG、图谱）、任务种子（事实类查询） | 测试问题的标准答案不能进知识库 |
| R10 | `data/ele_trans/*.pdf` | 31 篇 PDF | 与 R9 部分重叠的原始 PDF | 同 R9 | 知识源备份 | — |
| R11 | `data/manuals/ABB_*.pdf`、`Siemens_*.pdf` | 2 份厂商手册 | 公开厂商资料 | 厂商版权 | 知识源 | — |

## 2. 合成 / 派生数据（禁止进入评测集与正式训练集）

| 编号 | 路径 | 规模 | 生成方式 | 允许角色 | 备注 |
|---|---|---:|---|---|---|
| S1 | `data/synthetic/dga/dga_records.jsonl` / `.csv` | 3000 条 | `scripts/generate_synthetic_data.py` 按规则采样 | 演示数据、单元测试 | 已加 `synthetic` 标记（见 §4） |
| S2 | `data/synthetic/timeseries/TR-0001..0020.csv` + `devices.json` | 20 台 × 2000 点 | 同上 | 演示数据、单元测试 | 前端异常检测已不再默认读取 TR-0001 |
| S3 | `data/synthetic/cases/fault_cases.jsonl` | 200 条 | 同上 | 演示数据（mock RAG）、单元测试 | 不代表真实案例 |
| S4 | `data/synthetic/eval/eval_set.jsonl` | 120 条 | 同上；`expected.primary_fault` 由生成规则给出 | 开发回归（仅检查流程连通与格式） | 不作为论文/报告中的评测集；其 `expected` 与 S1 同源，存在循环验证风险 |
| S5 | `data/synthetic/eval/report_*.md` | 2 份 | 旧评测输出 | 历史记录 | 旧报告中的 3 样本 100% 完成率不具统计意义 |

## 3. 已知问题（影响角色判定）

R4（合并 DGA）：
- `device_id` 为程序循环生成，不是真实设备；不能用于分组切分或"同一设备多次测量"任务。
- `fault_type` 是从原始 IEC 标签（PD/D1/D2/T1/T2/T3/NF）到项目 8 类故障的工程映射（如 D2→绕组匝间短路、T1→绝缘老化），不代表确诊部位。评测标签应回退到 `raw_label`（原始 IEC 性质）。
- `CO`、`CO2` 原始数据缺失，被填为 0.0，不是真实测量。
- `evidence`、`severity`、`gas_rate_rapid` 由单次浓度规则派生，不是原始标注；产气速率需要时间序列，单次样本不能给出。
- 三份来源之间存在重复/近重复记录（7060 → 5143，约 27% 重复），切分前需按气体向量聚类去重；R3 内部另有已报告的跨标签冲突记录。
- 标签可信度：R1–R3 的 IEC 故障标签均来自文献汇编，多数由文献作者据事后检修给出，本研究无法逐条追溯到原始检修记录。论文「实验设置」须写明：标签为文献汇编标签而非现场确认标签；无设备身份字段，不能做设备维度泛化测试。
- 外部验证建议：补一份来源清晰的第三方基准做校准结果的外部检验，候选为 IEEE DataPort「Dissolved Gas Analysis dataset」（Enwen Li，DOI 10.21227/h8g0-8z59，含 IEC TC 10 数据库 117 例，经拆检确认）或 IEEE DataPort Dissanayake 2026 数据集（含 49 条 IEC TC 10 基准）。IEC TC 10 案例库是 IEC 60599 修订时使用的官方案例集，是 DGA 领域可信度最高的小样本。

R6/R7（ETT）：
- 负载特征 HUFL/HULL/MUFL/MULL/LUFL/LULL 官方未给单位与测点定义。
- ETTh1 与 ETTm1（及 h2/m2）可能为同一序列的不同采样，防泄漏时应视为同组。
- 修复前的加载代码把 15 分钟数据按 1 分钟对齐并插入 NaN（已在 P0-1 修复）。
- 数据本身存在少量时间间隙，`asfreq` 后会有 NaN，工具层需处理。

R8（external_transformer）：
- 来源、许可未记录，公开数据站点检索未匹配；`Year` 有空值；四种诊断列大量为 Undefined/Unidentifiable。
- 优点：有真实设备编号与多次测量号，是目前唯一能构造"同一设备趋势"任务的真实数据。
- 处置：来源未核实前不使用；F 阶段固化前仍无法核实则移除，不带入论文任何数字。

R9（文献解析）：
- 182 目录中存在明显重复（同名 PDF 带 "(1)" 后缀等），文档级去重预计剩约 150 篇。
- 图片 956 个仅有路径，表格 627 个 `table_caption` 为空，公式 1034 个未与解释合并，页眉页脚等噪声约 5300 个（P1-1 处理）。

## 4. 合成数据标记方式

在 `data/synthetic/SYNTHETIC_MARKER.json` 写入声明，并要求所有读取 `data/synthetic/` 的评测/造数脚本在启动时检查路径前缀并拒绝写入评测集。实现见 `src/utils/data_guard.py`。

## 5. 结论：各阶段可用数据

| 阶段 | 可用 | 说明 |
|---|---|---|
| P0-3 基线 | R4（回退 raw_label）、R6/R7、R9 | S4 仅做流程连通检查，不进基线报告 |
| P1 知识库 | R9、R10、R11 | 先去重 |
| P2 图谱 | R9 | — |
| P4 造数 | R4、R6/R7、R9、（R8 待核实） | 合成数据不进入 |
| P6 端到端评测 | 从 P4 封存测试集抽取 | — |
