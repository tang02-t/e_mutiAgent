# 三条主线执行轨迹速览（离线，零 LLM）

> 由 `python3 scripts/report/demo_traces.py --write` 生成。每段轨迹都走与正式实验相同的代码路径，
> 只是 LLM 关闭（Planner 用 EIG 决策 / 金标扰动，Generator 用模板或规则声明）。看懂这三段就看懂了系统 80% 的数据流。

## 主线一 · mode3 active_plan：信息不足时先问什么

样例来自 D12 部分观测模拟集：一条真实 DGA 记录被遮蔽 3 种气体（重档），`PartialObsEnv` 扮演用户，
被问到某征兆时按真值回答并计费。整条链路不调用 LLM：Planner 走 `decision=eig` 路径，Generator 走模板。

**输入**（`sim_id=D12-H0165`，真实标签 `overload_overheating`，评测时对系统隐藏）
```json
{
  "user_query": "这台变压器油色谱有部分数据缺失，请判断故障类型。",
  "context.dga（None = 未检测）": {
    "H2": null,
    "CH4": 189.0,
    "C2H2": null,
    "C2H4": null,
    "C2H6": 25.0
  },
  "被遮蔽的气体": [
    "C2H2",
    "C2H4",
    "H2"
  ]
}
```
**逐轮轨迹**（`inquiry_log`：每轮 Planner 的动作、依据的熵与 Top-1、用户回答、累计成本）

| 轮 | 动作 | 决策来源 | 追问征兆 | 用户回答 | 累计成本 | 决策时后验熵 bit |
|---|---|---|---|---|---|---|
| 0 | call_tool | eig | - | None | 0.00 | 1.883 |
| 1 | ask | eig | C2H2_elevated | False | 0.00 | 1.883 |
| 1 | call_tool | reattribution | - | None | 0.00 | 1.403 |
| 2 | ask | eig | C2H4_elevated | True | 0.00 | 1.403 |
| 2 | call_tool | reattribution | - | None | 0.00 | 0.753 |
| 2 | conclude | eig | - | None | 0.00 | 0.753 |

**每次归因后的 Top-1 变化**（同一条记录，随追问答案更新）

- 第 0 次归因：Top-1 `overload_overheating`（0.426），熵 1.883 bit，负观测 ['C2H6_elevated']
- 第 1 次归因：Top-1 `overload_overheating`（0.637），熵 1.403 bit，负观测 ['C2H6_elevated', 'C2H2_elevated', 'gas_rate_rapid']
- 第 2 次归因：Top-1 `overload_overheating`（0.861），熵 0.753 bit，负观测 ['C2H6_elevated', 'C2H2_elevated', 'gas_rate_rapid']

**归因工具最后一次返回的 `uncertainty` 块**（Planner 就是看这个决定问不问、问什么）
```json
{
  "entropy_bits": 0.7530724837803491,
  "top1_top2_gap": 0.7634812651992213,
  "calibrated": true,
  "suggested_action": "conclude",
  "recommendations（前 3）": [
    {
      "symptom": "H2_elevated",
      "eig": 0.01975,
      "cost": 0.0,
      "voi": 0.01975,
      "how_to_obtain": "ask_user"
    },
    {
      "symptom": "winding_temp_elevated",
      "eig": 0.05947,
      "cost": 1.0,
      "voi": 0.00947,
      "how_to_obtain": "ask_user"
    },
    {
      "symptom": "load_elevated",
      "eig": 0.04471,
      "cost": 1.0,
      "voi": -0.00529,
      "how_to_obtain": "call_tool:timeseries_anomaly"
    }
  ]
}
```
**工具调用序列**（`trajectory`）

- call:0 `fault_attribution` 参数 `{"query": "这台变压器油色谱有部分数据缺失，请判断故障类型。", "dga_data": {"CH4": 189.0, "C2H6": 25.0}}` → business_success=True
- call:1 `fault_attribution` 参数 `{"query": "这台变压器油色谱有部分数据缺失，请判断故障类型。", "dga_data": {"CH4": 189.0, "C2H2": 5.0, "C2H6": 25.0}, "evidence": {"CH4_elevated"…` → business_success=True
- call:2 `fault_attribution` 参数 `{"query": "这台变压器油色谱有部分数据缺失，请判断故障类型。", "dga_data": {"CH4": 189.0, "C2H2": 5.0, "C2H4": 295.0, "C2H6": 25.0}, "evidence": …` → business_success=True

**最终答案**（模板生成，LLM 关闭）：最终 Top-1 = `overload_overheating`，真实标签 = `overload_overheating`，问询 2 轮，成本 0.00，验证 `PASS`
```text
【模板生成模式：LLM 未启用或调用失败，以下为基于工具结果的结构化摘要】

用户问题：这台变压器油色谱有部分数据缺失，请判断故障类型。

一、相关案例与知识片段：
（知识库中未检索到高度相关的案例，以下结论基于通用经验。）

二、故障关系图谱：
（未调用故障关系图谱）

三、时序信号分析要点：
- 当前未获得有效时序分析结果。

四、故障归因分析：
- 主要故障推断：过载过热（后验概率 86.1%）
- 故障概率排序（贝叶斯网络 × DGA规则融合）：
  1. 过载过热（86.1%，高风险）
  2. 绝缘老化（9.8%，低风险）
  3. 局部放电（2.8%，低风险）
- DGA特征解释：- C₂H₂占总烃比例<10%，低能放电或局部放电可能性较高
- C₂H₄/C₂H₆≥3，高温过热特征明显
- 缺少 H2 浓度，未做三比值法判断
- 不确定性：后验熵 0.75 bit，Top-1 与 Top-2 概率差 76.3%（参数已校准）
- 已排除的征兆（负观测）：C2H6_elevated, C2H2_elevated, gas_rate_rapid
- 追问获得的补充证据（2 轮）：C2H2_elevated=无；C2H4_elevated=有

五、油温预测：
- 当前无有效油温预测分析结果。

六、诊断结论与处理建议（通用）：
1. 【安全优先】在进行任何检修前，确认相关
```
**怎么读**：第 0 轮先归因，熵高于阈值且 VoI 最大的征兆是「问用户」能拿到的，于是 `ask`；用户回答后把新气体写回 `context.dga` 重新归因，
熵降到 τ_H=0.8 以下或最大 VoI 低于 0.02 时 `conclude`。基线 mode2 没有这一环，会直接用缺测数据出结论。

## 主线二 · mode4 claim_verify：一条没有依据的「立即停电」建议是怎么被拦下的

样例来自 D11 故障注入集：在一份干净诊断草案上删掉支撑「高风险处置建议」的观测声明（SAFETY 类注入），
并额外追加 2 条需要尚未调用工具才能落证的声明（E2 设定）。`snapshot` 离线重建同一份工具结果，
Generator 用 `SimGenerator` 模拟（只修 Validator 标记的声明），Validator / supplement / Retriever 是真实节点。

**输入**（`eval_id=D11-0005`，底稿 `D10-0177`，场景 `context_contrast`，注入 `safety_premise_removed/remove_dga_premise`）
```json
{
  "user_query": "请判断SFSZ-240000/220 型主变的故障类型。",
  "context": {
    "device_id": "RT-0046",
    "dga": {
      "H2": 66.1,
      "CH4": 21.1,
      "C2H2": 95.3,
      "C2H4": 51.8
    }
  },
  "已有工具调用": [
    "call:0 fault_attribution"
  ]
}
```
**Generator 首轮输出的声明清单**（注入后；`c7` 的处置建议失去了它本该引用的观测 `c1`）

| id | type | 声明 | evidence 引用 |
|---|---|---|---|
| c4 | inference | 主要故障推断为「绕组匝间短路」，后验概率 72.6%（置信等级：确定）。 | tool:call:0 |
| c5 | inference | 故障概率排序（前三）：1. 绕组匝间短路 72.6%；2. 局部放电 15.8%；3. 过载过热 10.5%。 | tool:call:0 |
| c6 | inference | 归因不确定性：后验熵 1.19 bit，Top-1 与 Top-2 概率差 56.8%（参数已校准）。 | tool:call:0 |
| c7 | safety | 鉴于「绕组匝间短路」概率 72.6% 且属高危类型，建议在确认差动 / 瓦斯保护正常、复测 DGA 趋势后申请停电检查；… | user:user:0 |

被删除的观测声明：`本次油中溶解气体检测数据：H2=66.1 ppm、CH4=21.1 ppm、C2H2=95.3 ppm、C2H4=51.8 ppm。`

追加的缺证声明（E2）：['x_1→需 ett_forecast', 'x_2→需 kg_search']

**Validator 逐轮判定**（`reasoning_trace` 中 agent=validator 的记录）

第 1 轮：verdict=`REVISION` claim_check=`REVISION` unsupported_ratio=0.3333333333333333

| claim | verdict | 违反约束 | 说明 |
|---|---|---|---|
| c7 | violated | SAFETY | 高风险处置建议缺少检测结果（observation）支撑 |
| x_1 | unsupported | EVIDENCE | 无证据引用 |
| x_2 | unsupported | EVIDENCE | 无证据引用 |

第 2 轮：verdict=`PASS` claim_check=`PASS` unsupported_ratio=0.0

| claim | verdict | 违反约束 | 说明 |
|---|---|---|---|

**路由与补证**（`route_log`）

- 目标 `supplement`，原因：unsupported:2 claims need evidence，补证工具：[{'tool': 'ett_forecast', 'arguments': {}, 'for_claims': ['x_1']}, {'tool': 'kg_search', 'arguments': {'query': '绕组匝间短路', 'hops': 1}, 'for_claims': ['x_2']}]
- 目标 `END`，原因：pass

补证新增的工具调用：
- `kg_search` 参数 `{"query": "绕组匝间短路", "hops": 1}` → business_success=True

**结局**：最终 verdict=`PASS`，修订 2 轮，补证 1 轮，SimGenerator 修正 1 条 / 再落证 2 条 / 删除 0 条 / 保留 0 条，最终声明 6 条

**怎么读**：v1 Validator 只给整份草案打一个分，这类「建议成立但前提被删」的错误 100% 放行；mode4 把每条声明的 evidence 解析出来，
SAFETY 类建议必须有 `observation` 支撑，否则直接 REVISION 并指出是哪一条；缺证但可补的声明触发 supplement 节点去调工具，而不是反复修订到弃答。

## 主线三 · D13 偏好对：一条 prompt 的 6 个候选计划如何被打分、配对

样例来自 D8 train 种子。正式流程中候选由 M1 / M0 采样产生；这里用金标的六类扰动代替（与 `--synthetic-demo` 相同），
每个候选都在真实工具环境中执行，再送 Generator（规则声明）+ ClaimChecker 得到忠实度分。零 LLM。

**prompt**（`seed_id=S-INS-0d2b2ede5b`，类别 `insufficient`）
```json
{
  "query": "师傅，对下面的振动数据做异常检测。谢谢。",
  "context": null,
  "gold_actions": [
    {
      "type": "ask_user",
      "missing": [
        "signal 数值序列"
      ],
      "reason": "提到“下面的数据”但实际未附带"
    }
  ]
}
```
**候选与打分**（六分项：format / tool / schema / business / faith / ask；`score_exec` 只看前四项，`score_full` 六项加成本惩罚）

| 候选来源 | 计划摘要 | format | tool | schema | business | faith | ask | 多余调用 | score_exec | score_full |
|---|---|---|---|---|---|---|---|---|---|---|
| `gold` | ask_user ['signal 数值序列'] | 1 | 1 | 1 | 1 | 1.00 | 1 | 0 | 1.0 | 1.0 |
| `perturb:spurious_call` | rag_search({"query": "师傅，对下面的振动数据做异常检测。谢谢。"}) | 1 | 0 | 1 | 1 | 1.00 | 0 | 1 | 0.75 | 0.5667 |
| `perturb:extra_calls` | kg_search({"query": "局部放电"}) → rag_search({"query": "变压器 检修"}) | 1 | 0 | 1 | 1 | 1.00 | 0 | 2 | 0.75 | 0.4667 |
| `perturb:no_ask` | 直接回答（无调用） | 1 | 1 | 1 | 1 | 1.00 | 0 | 0 | 1.0 | 0.8333 |

- D13-exec（按 `score_exec`）：最高-最低分差不足 0.3，该 prompt 不产生偏好对
- D13-full（按 `score_full`）：chosen=`gold`（1.0） vs rejected=`perturb:extra_calls`（0.4667），分差 0.5333

**怎么读**：这是一条「应追问」的 prompt。只看工具执行成败（exec）时，「不问直接答」与金标都没有调用工具，得分相同，无法配对；
加入主线一的追问一致性与主线二的声明忠实度后（full），金标明显高于 `no_ask`，才能形成 chosen / rejected。
这就是 D13-exec 与 D13-full 两份子集要做的消融：验证信号是否比单纯的工具成败信号更能教会 Planner「该问就问、该停就停」。
