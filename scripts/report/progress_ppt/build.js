const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..", "..");
const FIG = (n) => path.join(ROOT, "docs/figures", n);

// 配色：深海蓝主色 + 青色辅色 + 琥珀强调（未完成）+ 青绿（已完成）
const C = {
  navy: "0B2545", deep: "13315C", teal: "1C7293", ice: "E6F0F5",
  bg: "F7F9FB", text: "1E293B", muted: "64748B", white: "FFFFFF",
  done: "2A9D8F", todo: "E76F51", amber: "F4A261", line: "D5DEE6",
};
const HF = "Microsoft YaHei", BF = "Microsoft YaHei";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.author = "进度汇报";
pres.title = "证据可信的变压器故障诊断多智能体系统 进度汇报";

const shadow = () => ({ type: "outer", color: "000000", blur: 4, offset: 1.5, angle: 135, opacity: 0.10 });

function header(slide, title, tag, idx) {
  slide.background = { color: C.bg };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.42, w: 0.14, h: 0.5, fill: { color: C.teal }, line: { color: C.teal } });
  slide.addText(title, { x: 0.78, y: 0.33, w: 6.7, h: 0.68, fontFace: HF, fontSize: 24, bold: true, color: C.navy, margin: 0, valign: "middle" });
  if (tag) slide.addText(tag, { x: 7.6, y: 0.42, w: 1.9, h: 0.5, fontFace: BF, fontSize: 11, color: C.white, fill: { color: tag.startsWith("未") ? C.todo : C.teal }, align: "center", valign: "middle", margin: 0 });
  slide.addText(`${idx}`, { x: 9.3, y: 5.2, w: 0.4, h: 0.3, fontFace: BF, fontSize: 9, color: C.muted, align: "right", margin: 0 });
  slide.addText("变压器故障诊断多智能体系统 · 进度汇报 · 2026-09", { x: 0.5, y: 5.2, w: 6, h: 0.3, fontFace: BF, fontSize: 9, color: C.muted, margin: 0 });
}

function card(slide, x, y, w, h, color, title, lines, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.white }, line: { color: C.line, width: 0.75 }, shadow: shadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h, fill: { color }, line: { color } });
  slide.addText(title, { x: x + 0.2, y: y + 0.1, w: w - 0.35, h: 0.36, fontFace: HF, fontSize: opts.titleSize || 13, bold: true, color: C.navy, margin: 0, valign: "middle" });
  const runs = lines.map((t, i) => ({ text: t, options: { bullet: opts.bullet !== false ? { code: "25AA" } : false, breakLine: i < lines.length - 1, paraSpaceAfter: 3 } }));
  slide.addText(runs, { x: x + 0.2, y: y + 0.5, w: w - 0.35, h: h - 0.6, fontFace: BF, fontSize: opts.fontSize || 10.5, color: C.text, margin: 0, valign: "top" });
}

function stat(slide, x, y, w, big, label, color, size) {
  slide.addText(big, { x, y, w, h: 0.62, fontFace: HF, fontSize: size || 30, bold: true, color: color || C.teal, align: "center", margin: 0, valign: "bottom" });
  slide.addText(label, { x, y: y + 0.64, w, h: 0.5, fontFace: BF, fontSize: 10, color: C.muted, align: "center", margin: 0, valign: "top" });
}

// ───────── 1 封面 ─────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("硕士学位论文 · 阶段进度汇报", { x: 0.9, y: 1.05, w: 8, h: 0.4, fontFace: BF, fontSize: 14, color: "9FC3D6", margin: 0 });
  s.addText("证据可信的变压器故障诊断多智能体系统", { x: 0.9, y: 1.5, w: 8.6, h: 0.9, fontFace: HF, fontSize: 32, bold: true, color: C.white, margin: 0 });
  s.addText("中期答辩之后：主动规划 · 声明级验证 · 偏好优化 Planner", { x: 0.9, y: 2.45, w: 8.6, h: 0.5, fontFace: BF, fontSize: 16, color: "CADCFC", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 3.35, w: 8.2, h: 0.9, fill: { color: C.deep }, line: { color: C.deep } });
  s.addText([
    { text: "计划完成度  ", options: { color: "9FC3D6", fontSize: 12 } },
    { text: "79 / 113", options: { color: C.white, fontSize: 22, bold: true } },
    { text: "      三条主线  ", options: { color: "9FC3D6", fontSize: 12 } },
    { text: "2 条已验收 · 1 条待训练", options: { color: C.white, fontSize: 16, bold: true } },
    { text: "      回归测试  ", options: { color: "9FC3D6", fontSize: 12 } },
    { text: "608 项通过", options: { color: C.white, fontSize: 16, bold: true } },
  ], { x: 1.1, y: 3.35, w: 7.9, h: 0.9, fontFace: HF, margin: 0, valign: "middle" });
  s.addText("2026 年 9 月 · 中期答辩（2026-06）后第一次进度汇报", { x: 0.9, y: 4.7, w: 8, h: 0.4, fontFace: BF, fontSize: 11, color: "9FC3D6", margin: 0 });
}

// ───────── 2 中期 vs 现在 ─────────
{
  const s = pres.addSlide();
  header(s, "从中期答辩到现在：叙事升级", "总览", 2);
  const colW = 4.3, y0 = 1.15, h = 3.75;
  // 左：中期
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: y0, w: colW, h, fill: { color: C.white }, line: { color: C.line, width: 0.75 }, shadow: shadow() });
  s.addText("中期答辩（2026-06）", { x: 0.5, y: y0, w: colW, h: 0.5, fontFace: HF, fontSize: 14, bold: true, color: C.white, fill: { color: C.muted }, align: "center", valign: "middle", margin: 0 });
  s.addText([
    { text: "主题：多智能体协作 + 质量闭环的诊断范式", options: { bold: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "五智能体 LangGraph 编排，工具三件套（归因 / 检索 / 时序）", options: { bullet: true, breakLine: true } },
    { text: "归因 Top-1 37.9%，CPT 为人工近似", options: { bullet: true, breakLine: true } },
    { text: "检索用合成语料，评测用模拟数据", options: { bullet: true, breakLine: true } },
    { text: "端到端只验证了连通率 100%，无真实诊断数字", options: { bullet: true, breakLine: true } },
    { text: "Validator 整体打分，结论逐句无法核对", options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "承诺创新点：范式 / 规则-概率融合 / 三位一体证据融合", options: { italic: true, color: C.muted } },
  ], { x: 0.7, y: y0 + 0.6, w: colW - 0.4, h: h - 0.7, fontFace: BF, fontSize: 11, color: C.text, margin: 0, valign: "top" });
  // 箭头
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.85, y: 2.75, w: 0.6, h: 0.5, fill: { color: C.teal }, line: { color: C.teal } });
  // 右：现在
  const x2 = 5.5;
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: y0, w: colW, h, fill: { color: C.white }, line: { color: C.line, width: 0.75 }, shadow: shadow() });
  s.addText("现在（2026-09，v2）", { x: x2, y: y0, w: colW, h: 0.5, fontFace: HF, fontSize: 14, bold: true, color: C.white, fill: { color: C.teal }, align: "center", valign: "middle", margin: 0 });
  s.addText([
    { text: "主题：证据可信的诊断智能体", options: { bold: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "问题不再是「能不能诊断」，而是「过程和结论可信吗」", options: { bullet: true, breakLine: true } },
    { text: "主线一：不确定时该问什么（EIG 主动规划）", options: { bullet: true, breakLine: true } },
    { text: "主线二：每句结论有没有证据（声明级核查 + 补证）", options: { bullet: true, breakLine: true } },
    { text: "主线三：用验证信号训 Planner（DPO 偏好优化）", options: { bullet: true, breakLine: true } },
    { text: "评测改为五级增量模式，逐级分离每条主线贡献", options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "工程基座（知识库 / 图谱 / 反思）降为支撑，不作论文贡献", options: { italic: true, color: C.muted } },
  ], { x: x2 + 0.2, y: y0 + 0.6, w: colW - 0.4, h: h - 0.7, fontFace: BF, fontSize: 11, color: C.text, margin: 0, valign: "top" });
}

// ───────── 3 中期遗留问题的回应 ─────────
{
  const s = pres.addSlide();
  header(s, "中期提出的不足与计划，逐项回应", "总览", 3);
  const rows = [
    [{ text: "中期指出的不足 / 计划", options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "本阶段做了什么", options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "状态", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } }],
    ["CPT 为人工近似，Top-1 37.9% 待提升", "引入 5143 条真实 DGA，极大似然 + 5 折校准；Top-1 0.635 → 0.665，ECE 0.101 → 0.055；支持「未检测」负观测", { text: "完成", options: { color: C.done, bold: true, align: "center" } }],
    ["时序模块简化，未接真实数据", "接入 ETT 公开油温数据集（CC BY-ND 4.0），滑窗回归 + 3σ；定位为工具调用行为研究，不追求预测 SOTA", { text: "完成（降级）", options: { color: C.done, bold: true, align: "center" } }],
    ["端到端受 LLM 配额限制未评测", "建 196 条端到端评测集 D10 + Oracle 上界 97.4%；跑出 mode2 基线 46.0% 与 Planner 7 指标；正式五级评测待 LLM 费用", { text: "部分完成", options: { color: C.amber, bold: true, align: "center" } }],
    ["计划：Milvus 向量知识库", "改为离线 BM25 两路 RRF（182 篇 / 5404 块），删除 Milvus 路线；稠密向量明确不做", { text: "调整", options: { color: C.muted, bold: true, align: "center" } }],
    ["计划：与传统方法 / 单一 LLM 对比", "对照改为系统内部能力开关（五级模式）+ 通用方法基线（固定顺序问询、v1 Validator、M0 基座）", { text: "已定义", options: { color: C.done, bold: true, align: "center" } }],
    ["计划：LSTM / Transformer 油温预测", "砍掉：与「证据可信」主叙事无关，登记于计划附录 B", { text: "砍掉", options: { color: C.todo, bold: true, align: "center" } }],
  ];
  s.addTable(rows, { x: 0.5, y: 1.15, w: 9.0, colW: [2.6, 5.3, 1.1], fontFace: BF, fontSize: 10, color: C.text, border: { type: "solid", pt: 0.5, color: C.line }, valign: "middle", rowH: [0.38, 0.52, 0.52, 0.52, 0.5, 0.52, 0.42], margin: 0.06 });
}

// ───────── 4 总体进展 ─────────
{
  const s = pres.addSlide();
  header(s, "总体进展：计划 113 项，完成 79 项", "总览", 4);
  s.addChart(pres.charts.BAR, [
    { name: "已完成", labels: ["A 基础设施与基线", "B 主线一 主动规划", "C 主线二 声明级验证", "D 主线三 DPO Planner", "E 系统级增量评测", "F 成果固化"], values: [17, 28, 22, 7, 3, 2] },
    { name: "未完成", labels: ["A 基础设施与基线", "B 主线一 主动规划", "C 主线二 声明级验证", "D 主线三 DPO Planner", "E 系统级增量评测", "F 成果固化"], values: [6, 0, 1, 15, 7, 5] },
  ], {
    x: 0.5, y: 1.1, w: 5.6, h: 3.9, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
    chartColors: [C.done, C.amber], chartArea: { fill: { color: C.white }, roundedCorners: false },
    catAxisLabelColor: C.text, catAxisLabelFontSize: 10, catAxisLabelFontFace: BF, catAxisOrientation: "maxMin",
    valAxisLabelColor: C.muted, valAxisLabelFontSize: 9, valAxisMaxVal: 30, valGridLine: { color: "EDF2F7", size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelColor: C.white, dataLabelFontSize: 9, dataLabelFontBold: true,
    showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: BF,
  });
  card(s, 6.4, 1.1, 3.1, 1.2, C.done, "B、C 两条主线提前完成", ["计划 B 至 10-30、C 至 11-13 结束", "实际 09-16 全部验收，提前约 8 周"], { fontSize: 10 });
  card(s, 6.4, 2.45, 3.1, 1.2, C.amber, "D 主线三是当前瓶颈", ["脚本、数据、请求体全部就绪", "15 项训练 / 部署 / 评测等百炼账号"], { fontSize: 10 });
  card(s, 6.4, 3.8, 3.1, 1.2, C.teal, "未完成 34 项按资源分", ["百炼 15 · LLM 费用 8 · 人工 7 · 收尾 4", "无一项被代码阻塞"], { fontSize: 10 });
}

// ───────── 5 已完成：工程基座与数据 ─────────
{
  const s = pres.addSlide();
  header(s, "已完成 · 工程基座与数据（全部换真）", "已完成", 5);
  const y = 1.2;
  stat(s, 0.5, y, 1.8, "5143", "真实 DGA 记录\n三源合并，8 类故障", C.teal);
  stat(s, 2.35, y, 1.8, "182 / 5404", "领域文献 / 分块\nBM25 两路 RRF", C.teal, 24);
  stat(s, 4.2, y, 1.8, "90 / 190", "故障图谱节点 / 边\n带文献支持数", C.teal, 24);
  stat(s, 6.05, y, 1.8, "7001", "Planner 训练样本\n口语化改写后（百炼格式）", C.teal);
  stat(s, 7.9, y, 1.6, "608", "回归测试项\n13 个文件全过", C.teal);
  card(s, 0.5, 2.6, 4.4, 2.4, C.done, "四份评测集（中期时只有模拟数据）", [
    "D10 端到端 196 条：8 类场景，金标必要动作 + 关键参数 + 证据来源",
    "D11 故障注入 300 条：四类工程约束 × 15 子类错误 + 100 干净",
    "D12 部分观测 1500 条：从真实 DGA 遮蔽 1-3 种气体，轻 / 中 / 重",
    "D13 偏好对：候选打分配对管线，demo 50 / 60 对，正式 ≥1200 待训练",
  ]);
  card(s, 5.1, 2.6, 4.4, 2.4, C.done, "数据治理", [
    "合成数据只用于流程回归，assert_not_synthetic 硬拦进评测 / 训练",
    "test 切分 sha256 封存（SEALED.md），train vs test 泄漏 0 对",
    "每份数据有 DATA_CARD；来源、许可、格式、字段统一在 data_and_evaluation.md",
    "reproduce.sh 一键离线复现：dga → kb → kg → 反思 → 训练数据 → B / C / D / E → test",
  ]);
}

// ───────── 6 已完成：基线揭示的三个失败模式 ─────────
{
  const s = pres.addSlide();
  header(s, "已完成 · 基线：通用 Planner 为何不够", "已完成", 6);
  s.addText("qwen3.7-flash 作 Planner、全工具打开（mode2），D10 前 50 条端到端 + 分层 100 条离线 7 指标。基线低不是坏事：它把三条主线要解决的问题量化出来了。", { x: 0.5, y: 1.1, w: 9, h: 0.5, fontFace: BF, fontSize: 11, color: C.muted, margin: 0 });
  stat(s, 0.5, 1.75, 2.9, "46.0%", "任务成功率（mode2 基线）", C.todo);
  s.addText("vs", { x: 3.4, y: 1.85, w: 0.5, h: 0.5, fontFace: HF, fontSize: 14, color: C.muted, align: "center", margin: 0 });
  stat(s, 3.9, 1.75, 2.9, "97.4%", "Oracle 金标 Planner 上界（196 条）", C.done);
  s.addText("51 pt 差距 = 三条主线要填的空间", { x: 6.9, y: 1.85, w: 2.6, h: 0.9, fontFace: HF, fontSize: 13, bold: true, color: C.navy, valign: "middle", margin: 0, fill: { color: C.ice } });
  const y = 3.15, w = 2.9, h = 1.85;
  card(s, 0.5, y, w, h, C.todo, "不会问 → 主线一", ["ask_user 场景追问正确率 0%", "3 条应追问样本全部直接用默认参数调 ett_forecast", "追问正确率（离线）66.7%"], { fontSize: 10 });
  card(s, 3.55, y, w, h, C.todo, "不会停、不会改 → 主线三", ["多轮决策点该结束时 100% 再发调用", "工具报错后错误恢复成功率 0%", "kg_search 12 次中 11 次漏 hops 参数"], { fontSize: 10 });
  card(s, 6.6, y, w, h, C.todo, "结论没法核对 → 主线二", ["忠实度只能用证据锚点代理 0.795", "v1 Validator 对注入错误通过率 100%", "无法判断哪一句结论没有依据"], { fontSize: 10 });
}

// ───────── 7 已完成：主线一 ─────────
{
  const s = pres.addSlide();
  header(s, "已完成 · 主线一：EIG 主动诊断规划", "已完成 · 已验收", 7);
  s.addImage({ path: FIG("fig_b1_reliability.png"), x: 0.5, y: 1.1, w: 4.5, h: 2.0 });
  s.addImage({ path: FIG("fig_b5_acc_vs_queries.png"), x: 5.2, y: 1.1, w: 3.05, h: 2.1 });
  s.addText("左：5 折校准前后可靠性图　右：四种问询策略 Top-1 随问询次数变化（D12 1500 条）", { x: 0.5, y: 3.15, w: 8.5, h: 0.3, fontFace: BF, fontSize: 9, color: C.muted, margin: 0 });
  card(s, 0.5, 3.55, 2.9, 1.45, C.done, "归因引擎校准（B-1）", ["ECE 0.101 → 0.055，Top-1 +3.0 pt", "按类别可学习融合权重 + 温度"], { fontSize: 10 });
  card(s, 3.55, 3.55, 2.9, 1.45, C.done, "EIG 主动追问（B-2 ~ B-5）", ["同 Top-1 下问询 −27%（2.79 → 2.04，p<0.0001）", "成本 0.85 → 0.26；Planner 接入 ask / call_tool / conclude"], { fontSize: 10 });
  card(s, 6.6, 3.55, 2.9, 1.45, C.teal, "关键消融", ["不校准的引擎跑同一 EIG 策略 Top-1 仅 0.383（校准后 0.660）", "结论：不校准就不能问"], { fontSize: 10 });
}

// ───────── 8 已完成：主线二 ─────────
{
  const s = pres.addSlide();
  header(s, "已完成 · 主线二：声明级证据验证", "已完成 · 已验收", 8);
  s.addImage({ path: FIG("fig_c5_pass_vs_cost.png"), x: 0.5, y: 1.1, w: 6.4, h: 2.2 });
  s.addText("D11 300 条故障注入：v1 整体评分 vs v2 声明核查 vs v2 核查 + 补证（零 LLM，Token = 0）", { x: 0.5, y: 3.32, w: 6.4, h: 0.3, fontFace: BF, fontSize: 9, color: C.muted, margin: 0 });
  card(s, 7.1, 1.1, 2.4, 2.5, C.done, "做了什么", ["Generator 先输出 claims JSON，每条带类型与 evidence[]", "ClaimChecker 逐条核对 DATA / EVIDENCE / APPLICABILITY / SAFETY", "Validator 三路路由：修订 / 补证重规划 / 弃答"], { fontSize: 9.5 });
  stat(s, 0.5, 3.75, 2.3, "100% → 3%", "错误通过率（干净样本误报 0）", C.done, 24);
  stat(s, 2.85, 3.75, 2.4, "36.4% → 7.2%", "无依据结论率（E2 缺证场景）", C.done, 24);
  stat(s, 5.3, 3.75, 2.1, "63% → 3%", "弃答率：只核查 vs 核查 + 补证", C.done, 24);
  stat(s, 7.45, 3.75, 2.05, "+1.6 次", "补证的代价：每样本额外工具调用", C.amber, 24);
}

// ───────── 9 已完成：主线三离线部分 + 五级模式 ─────────
{
  const s = pres.addSlide();
  header(s, "已完成 · 主线三离线部分与五级协议", "已完成", 9);
  card(s, 0.5, 1.1, 4.4, 2.0, C.done, "主线三（D）已就位的部分", [
    "训练数据：3482 种子 → 口语化改写 4538 条（守卫拒 961）→ 百炼 ChatML 7001 / 1301",
    "偏好打分函数：format / tool / schema / business / faith / ask 六维，6 类扰动排序方向全对",
    "D13-exec（只看工具执行）与 D13-full（+追问一致 + 声明忠实）两套信号，用于消融",
    "百炼 SFT / DPO 提交脚本与请求体（submit_job.py），dry-run 通过",
  ], { fontSize: 9.5 });
  card(s, 0.5, 3.25, 4.4, 1.75, C.done, "论文第 5 章素材", ["问题定义、训练路线表、消融设计、与 ToolRL / ARTIST / Deep-DxSearch 的差异", "数字留空待 D-4 回填"], { fontSize: 9.5 });
  // 五级模式表
  s.addText("五级增量模式（E-1，第 6 章主表的列）", { x: 5.1, y: 1.1, w: 4.4, h: 0.35, fontFace: HF, fontSize: 12, bold: true, color: C.navy, margin: 0 });
  const rows = [
    [{ text: "模式", options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: "相对上一级新增", options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: "对应", options: { bold: true, color: C.white, fill: { color: C.navy } } }],
    ["1 llm_only", "无工具，LLM 直答", "参照"],
    ["2 tool_base", "五工具 + 检索 + 图谱 + 反思；自由规划", "工程基座"],
    ["3 active_plan", "校准归因 + EIG；Planner 主动策略", "主线一"],
    ["4 claim_verify", "claims 输出 + 声明核查 + 补证路由", "主线二"],
    ["5 dpo_planner", "Planner 换为 DPO 模型", "主线三"],
  ];
  s.addTable(rows, { x: 5.1, y: 1.5, w: 4.4, colW: [1.15, 2.45, 0.8], fontFace: BF, fontSize: 9, color: C.text, border: { type: "solid", pt: 0.5, color: C.line }, rowH: 0.34, valign: "middle", margin: 0.05 });
  s.addText("相邻只差一组开关，单元测试断言；resolve_mode 可交叉对照（如 mode4 + 基座 Planner）。Oracle 上界 97.4% 已用该协议跑出。", { x: 5.1, y: 3.65, w: 4.4, h: 0.6, fontFace: BF, fontSize: 9.5, color: C.muted, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 4.35, w: 4.4, h: 0.65, fill: { color: C.ice }, line: { color: C.ice } });
  s.addText("顺带完成：Streamlit 前端支持五级切换与六项覆盖；技术报告 v2、速览卡、三条主线执行轨迹文档。", { x: 5.25, y: 4.35, w: 4.15, h: 0.65, fontFace: BF, fontSize: 9.5, color: C.navy, margin: 0, valign: "middle" });
}

// ───────── 10 未完成 ─────────
{
  const s = pres.addSlide();
  header(s, "未完成 · 34 项，按所需资源分为四类", "未完成", 10);
  const y = 1.15, h = 3.85, w = 2.15, gap = 0.13;
  card(s, 0.5, y, w, h, C.todo, "需百炼账号 · 15 项", [
    "D-1 上传训练集，efficient_sft 训 M1，部署",
    "D-1 M1 离线 7 指标（与 M0 并列）",
    "D-2 用 M1 / M0 采样正式候选 ≥1200 对",
    "D-3 dpo_lora 训 M4-exec / M4-full，部署",
    "D-4 四模型矩阵 × 2 seed，写 planner_dpo_eval.md",
  ], { fontSize: 9.5 });
  card(s, 0.5 + (w + gap), y, w, h, C.amber, "需 LLM 费用 · 8 项", [
    "E-2 正式评测：196 × 五级 × 2 次",
    "E-2 交叉对照、LLM 裁判 + 10% 人工",
    "E-2 写 system_modes_eval.md 与图表",
    "E-3 鲁棒性：D11 上 mode2 vs mode5、D12 重档 mode2 vs mode3",
    "C-5 语义层 LLM 判定人工一致率",
    "预估几十元，等 D 阶段模型就位后一次跑完",
  ], { fontSize: 9.5 });
  card(s, 0.5 + 2 * (w + gap), y, w, h, C.teal, "需本人动手 · 7 项", [
    "D9 反思 298 对人工评分，算 Kappa",
    "图谱 56 条三元组精度人工确认",
    "D10 196 条参考要点复核（E-2 前置）",
    "D-2 偏好对 100 对方向复核",
    "R8 数据来源核实或移除",
    "IEC TC 10 案例库外部检验（可选）",
    "历史泄漏 API Key 撤销重建",
  ], { fontSize: 9.5 });
  card(s, 0.5 + 3 * (w + gap), y, w, h, C.muted, "F 收尾 · 4 项", [
    "三份报告数字回填：baseline 已有，planner_dpo_eval / system_modes_eval 待",
    "架构图 PNG 重绘（加 EIG 与声明核查器）",
    "打 v2-final tag，对照 baseline-v0",
    "论文 3-6 章实验小节初稿",
  ], { fontSize: 9.5 });
}

// ───────── 11 下一步与时间线 ─────────
{
  const s = pres.addSlide();
  header(s, "下一步计划：一条依赖链，两个月", "计划", 11);
  const steps = [
    { t: "9 月下旬", h: "D-1 SFT", d: "上传 → 训练 → 部署 M1\n离线 7 指标 vs M0", c: C.todo },
    { t: "10 月", h: "D-2 / D-3 DPO", d: "M1 采样候选 → 打分配对\n100 对复核 → 训 M4 ×2", c: C.todo },
    { t: "11 月", h: "D-4 矩阵", d: "M0 / M1 / M4-exec / M4-full\n验证信号 vs 执行信号", c: C.todo },
    { t: "12 月", h: "E-2 正式评测", d: "196 × 五级 × 2 次\nLLM 裁判 + 人工复核", c: C.amber },
    { t: "2027-01", h: "F 固化", d: "v2-final tag\n3-6 章实验小节初稿", c: C.muted },
  ];
  const x0 = 0.5, wStep = 1.72, gap = 0.1, y = 1.3;
  s.addShape(pres.shapes.LINE, { x: x0, y: y + 0.42, w: 9.0, h: 0, line: { color: C.line, width: 2 } });
  steps.forEach((st, i) => {
    const x = x0 + i * (wStep + gap);
    s.addShape(pres.shapes.OVAL, { x: x + wStep / 2 - 0.16, y: y + 0.26, w: 0.32, h: 0.32, fill: { color: st.c }, line: { color: C.white, width: 2 } });
    s.addText(st.t, { x, y: y - 0.1, w: wStep, h: 0.3, fontFace: BF, fontSize: 10, color: C.muted, align: "center", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.75, w: wStep, h: 1.35, fill: { color: C.white }, line: { color: C.line, width: 0.75 }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.75, w: wStep, h: 0.06, fill: { color: st.c }, line: { color: st.c } });
    s.addText(st.h, { x: x + 0.1, y: y + 0.85, w: wStep - 0.2, h: 0.35, fontFace: HF, fontSize: 12, bold: true, color: C.navy, margin: 0, valign: "middle" });
    s.addText(st.d, { x: x + 0.1, y: y + 1.22, w: wStep - 0.2, h: 0.85, fontFace: BF, fontSize: 9.5, color: C.text, margin: 0, valign: "top" });
  });
  card(s, 0.5, 3.65, 4.4, 1.35, C.teal, "并行进行的人工项（不阻塞脚本）", ["D9 Kappa、图谱 56 条、D10 复核合计约 4 小时，可在等训练时完成", "D10 复核必须在 E-2 之前，否则忠实度分不可信"], { fontSize: 10 });
  card(s, 5.1, 3.65, 4.4, 1.35, C.done, "验收口径（每步过了再进下一步）", ["SFT：参数正确率 61% → ≥80% 即通过，不追求全指标", "DPO：看 M4-full 是否在「该问」「该停」场景优于 M4-exec"], { fontSize: 10 });
}

// ───────── 12 风险与所需支持 ─────────
{
  const s = pres.addSlide();
  header(s, "风险与需要的支持", "计划", 12);
  card(s, 0.5, 1.15, 4.4, 1.75, C.amber, "风险 1：主线三尚无任何模型数字", ["论文闭环点全压在 D 阶段；若百炼 DPO 对 qwen3-8b 不可用，切魔搭 ms-swift 备选路线（脚本已保留）", "若 M4-full 与 M4-exec 无显著差异，口径改为「DPO 有效但信号来源不敏感」"], { fontSize: 10 });
  card(s, 5.1, 1.15, 4.4, 1.75, C.amber, "风险 2：评测集问法与改写模型同源", ["训练问法由 qwen3.7-flash 改写，M0 基线也是它", "对策：D10 来自封存 test 且人工复核；E-2 加 10% 人工裁判"], { fontSize: 10 });
  card(s, 0.5, 3.05, 4.4, 1.95, C.amber, "风险 3：D12 只揭示气体征兆", ["EIG 目前推荐「问哪种气体」，不是「去做什么检测」", "对策：如时间允许加入油温 / 局放等非 DGA 征兆；否则在局限中如实写"], { fontSize: 10 });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 3.05, w: 4.4, h: 1.95, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("需要的支持", { x: 5.3, y: 3.15, w: 4.0, h: 0.35, fontFace: HF, fontSize: 13, bold: true, color: C.white, margin: 0 });
  s.addText([
    { text: "百炼账号与训练费用（SFT 3 epoch + DPO 2 epoch × 2 模型，按 Token 计费）", options: { bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "LLM 评测费用（E-2 约 2000 次调用，预估几十元）", options: { bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "第二位标注人：D9 反思评分与图谱抽检需双人标注算 Kappa", options: { bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "对第 8 章量化目标（mode5 成功率 ≥65% 等）的意见", options: { bullet: true } },
  ], { x: 5.3, y: 3.55, w: 4.0, h: 1.4, fontFace: BF, fontSize: 10, color: "E6F0F5", margin: 0, valign: "top" });
}

// ───────── 13 结尾 ─────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("小结", { x: 0.9, y: 0.7, w: 8, h: 0.5, fontFace: BF, fontSize: 14, color: "9FC3D6", margin: 0 });
  s.addText([
    { text: "中期之后完成了叙事升级：从「能诊断」到「诊断可信」。", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "主线一、主线二已有完整方法、数据与验收数字，较计划提前约 8 周；", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "主线三与系统级评测的脚本全部就位，剩余 34 项无一被代码阻塞，", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "接下来两个月的核心是拿到百炼账号、跑出 M1 / M4，把第 5、6 章的表填满。", options: {} },
  ], { x: 0.9, y: 1.3, w: 8.4, h: 2.4, fontFace: HF, fontSize: 18, color: C.white, margin: 0, valign: "top" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 4.0, w: 8.2, h: 0.95, fill: { color: C.deep }, line: { color: C.deep } });
  s.addText([
    { text: "附录（仓库 docs/）  ", options: { color: "9FC3D6", bold: true } },
    { text: "技术报告_v2.md 全文 · 速览卡.md 一页版 · walkthrough_traces.md 三条主线真实执行轨迹 · baseline.md / attribution_calibration.md / active_planning_eval.md / validator_eval.md 验收报告 · PLAN_优化计划清单.md §0.6 进度表", options: { color: "CADCFC" } },
  ], { x: 1.1, y: 4.0, w: 7.8, h: 0.95, fontFace: BF, fontSize: 10.5, margin: 0, valign: "middle" });
}

pres.writeFile({ fileName: path.join(ROOT, "docs/thesis/进度汇报_2026-09.pptx") }).then((f) => console.log("written", f));
