# 多智能体系统验证方案

本文档说明如何用 `data/synthetic/` 下的数据，系统性地评价变压器故障诊断多智能体系统的实际效果。

验证分为**两个层次**，对应系统的两类能力，可独立运行、互不依赖。

---

## 评测一：故障归因引擎评测（确定性，无外部依赖）

**目的**：单独量化 `src/tools/fault_attribution.py` 贝叶斯归因引擎的诊断准确性。
**数据**：`data/synthetic/dga/dga_records.jsonl`（3000 条 DGA 记录，含真实故障标签）。
**特点**：不调用 LLM、不连 Milvus，秒级出结果，适合反复迭代调参 / CI 回归。

### 运行
```bash
python3 scripts/eval/eval_fault_attribution.py
# 报告输出：data/synthetic/eval/report_fault_attribution.md
```

### 指标体系
| 指标 | 含义 | 解读 |
|------|------|------|
| Top-1 准确率 | 主故障预测正确比例 | 直接反映归因精度 |
| Top-3 命中率 | 真实故障排进前三的比例 | 反映数据-特征相关性 / 召回潜力 |
| Precision/Recall/F1（每类） | 各故障类型分类质量 | 定位薄弱故障类型 |
| Macro-F1 / Weighted-F1 | 整体分类质量 | 综合评分 |
| **分严重程度召回率** | critical/serious 故障是否漏诊 | **安全底线指标** |
| 置信度-正确性对比 | 正确/错误预测的平均置信度 | 校准性初探 |

### 当前基线结果（模拟数据）
- Top-1：**37.9%**，Top-3：**88.5%**，Macro-F1：0.229
- critical 故障召回 **100%**，serious 召回仅 **32%**

**解读**：Top-3 高说明数据物理自洽、特征与故障强相关；Top-1 低是因为贝叶斯网络 CPT
为人工近似值（likelihood 连乘偏向高产气故障）。**这正是用真实标注数据校准 CPT 的直接依据**——
替换真实 DGA 数据后重跑本脚本，即可量化 CPT 校准带来的提升。

---

## 评测二：多智能体系统端到端评测

**目的**：评价 Planner→Retriever→Generator→Validator 完整链路的诊断效果。
**数据**：`data/synthetic/eval/eval_set.jsonl`（120 条问题→标准答案）。
**设计**：采用可注入工具，保证离线也能跑流程；LLM 用 `config.yaml` 真实配置。

### 运行
```bash
# 1) 离线流程连通性测试（不调 LLM，仅验证流程不崩 + 规则能力下限）
python3 scripts/eval/eval_end2end.py --no-llm --limit 10

# 2) 真实诊断能力评测（需 config.yaml 中 LLM 可用）
python3 scripts/eval/eval_end2end.py --limit 20

# 3) 用合成案例 mock 替代本地知识库（仅流程调试）
python3 scripts/eval/eval_end2end.py --kb-mode mock --limit 20

# 全量评测
python3 scripts/eval/eval_end2end.py
# 报告输出：data/synthetic/eval/report_end2end_dev.md（合成数据只允许 dev_regression）
```

### 指标体系
| 指标 | 含义 | 健康值参考 |
|------|------|-----------|
| 流程完成率 | 成功跑完工作流的比例 | 应 100% |
| **主故障命中率** | 诊断文本是否命中真实故障 | 越高越好 |
| 关键要点覆盖率 | expected.key_points 平均覆盖比例 | 越高越好 |
| **安全合规率** | 高危样本是否含安全/停电/保护提示 | 应接近 100% |
| Validator 平均得分 | 最终验证评分均值 | 反映质检严格度 |
| 平均迭代轮次 | Generator 重生成次数 | 反映自我修正活跃度 |
| **LLM 降级率** | 触发模板/规则降级比例 | 应为 0 |

### 当前发现（重要）
运行 `--limit 3` 真实评测时，`config.yaml` 中的 **DashScope API Key 免费额度已耗尽**
（`403 AllocationQuota.FreeTierOnly`），导致 Planner/Generator/Validator 全部降级为规则模式，
主故障命中率为 0、LLM 降级率 100%。

> **结论**：评测框架本身工作正常（完成率 100%，正确捕获并报告了降级）。
> 但要评估系统的**真实诊断能力**，必须先解决 LLM 调用问题：
> 1. 在 `config.yaml` 更换为有额度的 API Key，或关闭"仅免费层"模式；
> 2. 重新运行 `python3 scripts/eval/eval_end2end.py --limit 20`。
> 在 LLM 不可用前，端到端命中率仅代表模板/规则的能力下限，不代表系统真实水平。

---

## 推荐的验证流程（分步）

1. **先跑评测一**（无依赖，秒级）：确认归因引擎基线，记录 Top-1/F1。
2. **跑评测二 `--no-llm`**：确认全流程连通性（完成率应 100%）。
3. **修复 LLM 配额** → 跑评测二真实模式：得到系统真实诊断指标。
4. **用真实数据替换 `data/synthetic/` 下对应文件（保持 schema 不变），重跑全部脚本，
   对比模拟 vs 真实的指标差异，并据此校准贝叶斯 CPT、优化提示词。

## 脚本清单
| 脚本 | 作用 |
|------|------|
| `scripts/data/generate_synthetic_data.py` | 生成 4 类模拟数据 |
| `scripts/eval/eval_fault_attribution.py` | 评测一：归因引擎 |
| `scripts/eval/eval_end2end.py` | 评测二：端到端多智能体 |
