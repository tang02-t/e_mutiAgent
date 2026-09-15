#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【评测二】多智能体系统端到端评测框架。

用评测集（data/synthetic/eval/eval_set.jsonl）驱动完整工作流
Planner → Retriever → Generator → Validator，评估系统整体诊断效果。

为保证可运行性，采用「可注入工具」设计：
  - fault_attribution : 始终使用真实引擎（确定性，无外部依赖）
  - kg_search         : 始终使用真实图谱（data/kg/graph.json，离线）
  - rag_search        : --kb-mode local_kb（默认，BM25+锚点本地知识库）/ milvus / mock
  - timeseries_anomaly: 对用户提供的 signal 做 3σ（P0 后不再读取任何示例文件）
  - LLM               : 使用 config.yaml 的真实配置；不可用时各 Agent 自动降级为模板/规则
                        （此时仍可评测：流程连通性、安全合规规则、归因命中）

数据守卫：
  - 默认评测集 data/synthetic/eval/eval_set.jsonl 为合成数据（S4），只允许作为 dev_regression
    （开发回归）使用，报告写入 report_end2end_dev.md；
  - 只有传入 --purpose eval_set 且评测集不在合成目录下时，才生成正式评测报告；
    对合成数据传 --purpose eval_set 会被 assert_not_synthetic 直接拒绝。

评测指标：
  1. 流程完成率          : 成功跑完工作流的比例
  2. 主故障命中率        : 最终诊断文本是否命中 expected.primary_fault（中文名匹配）
  3. 关键要点覆盖率      : expected.key_points 在诊断文本中的平均覆盖比例
  4. 安全合规率          : must_mention_safety=true 的样本中，文本含安全/停电/保护等关键词的比例
  5. Validator 平均得分  : 最终验证评分均值（若 LLM 可用）
  6. 平均迭代轮次        : Generator 重生成次数，反映自我修正活跃度
  7. LLM 降级率          : 触发模板/规则降级的比例（反映对外部服务的依赖健康度）

用法：
  python3 scripts/eval_end2end.py --limit 20                   # 开发回归，先跑 20 条
  python3 scripts/eval_end2end.py --kb-mode milvus             # 使用真实 Milvus RAG
  python3 scripts/eval_end2end.py --no-llm                     # 强制不调用 LLM（仅评测流程+规则）
  python3 scripts/eval_end2end.py --eval data/real/eval/xxx.jsonl --purpose eval_set   # 正式评测
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.state import AgentState                       # noqa: E402
from src.graph.workflow import run_diagnosis_workflow        # noqa: E402
from src.tools.mcp_client import MCPClient                   # noqa: E402
from src.tools.fault_attribution import fault_attribution    # noqa: E402
from src.tools.kg_search import kg_search                    # noqa: E402
from src.utils.config import load_config                     # noqa: E402
from src.utils.data_guard import assert_not_synthetic, is_synthetic_path  # noqa: E402

SAFETY_KEYWORDS = ["安全", "停电", "保护", "注意", "断电", "隔离", "防护"]
TS_DIR = ROOT / "data/synthetic/timeseries"


# ──────────────────────────────────────────────────────────────
# Mock 工具（无外部依赖，便于离线/CI 评测）
# ──────────────────────────────────────────────────────────────
def _build_case_index() -> List[Dict[str, Any]]:
    """从故障案例库构造一个简单的 mock 知识库。"""
    path = ROOT / "data/synthetic/cases/fault_cases.jsonl"
    cases = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                cases.append(json.loads(line))
    return cases


_CASE_INDEX = _build_case_index()


def mock_rag_search(query: str) -> List[Dict[str, Any]]:
    """按 query 与案例的故障中文名/征兆做朴素关键词匹配，返回 top 知识片段。"""
    scored = []
    for c in _CASE_INDEX:
        score = 0
        text = c["fault_label_cn"] + " " + " ".join(c.get("symptoms", []))
        for ch in set(query):
            if ch in text:
                score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    results = []
    for score, c in scored[:3]:
        results.append({
            "text": f"案例{c['case_id']}：{c['fault_location']}。征兆：{'、'.join(c.get('symptoms', []))}。"
                    f"处理：{c['handling']}。参考 {c['reference']}。",
            "score": float(score),
            "metadata": {"doc_name": c["case_id"], "title": c["fault_label_cn"]},
        })
    return results


def mock_timeseries_anomaly(signal=None) -> Dict[str, Any]:
    """对传入 signal 做 3σ 检测；P0 修复后不再读取模拟设备文件。"""
    if not isinstance(signal, (list, tuple)) or len(signal) < 3:
        return {"status": "error", "message": "timeseries_anomaly 需要至少 3 个点的 signal 序列"}
    ot = [float(x) for x in signal]
    mean = sum(ot) / len(ot)
    var = sum((v - mean) ** 2 for v in ot) / max(len(ot) - 1, 1)
    std = var ** 0.5
    anomalies = [i for i, v in enumerate(ot) if std and abs(v - mean) > 3 * std]
    return {"status": "ok", "n": len(ot), "mean": mean, "std": std, "anomaly_indices": anomalies}


def build_mcp(kb_mode: str = "local_kb") -> MCPClient:
    """kb_mode: local_kb（默认）/ milvus / mock；失败时逐级回退 local_kb -> mock。"""
    mcp = MCPClient()
    mcp.register_tool("fault_attribution", fault_attribution)
    mcp.register_tool("timeseries_anomaly", mock_timeseries_anomaly)
    mcp.register_tool("kg_search", kg_search)

    if kb_mode == "milvus":
        try:
            from src.tools.rag_engine import RAGEngine
            cfg = load_config()
            mv = cfg["knowledge_base"]["milvus"]
            engine = RAGEngine(
                uri=mv.get("uri"), token=mv.get("token"), host=mv.get("host"), port=mv.get("port"),
                collection_name=mv["collection"], text_field=mv["text_field"],
                vector_field=mv["vector_field"], top_k=cfg["knowledge_base"].get("top_k", 5),
                metric_type=mv.get("metric_type", "L2"), nprobe=mv.get("nprobe", 10),
            )
            mcp.register_tool("rag_search", engine.search)
            return mcp
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Milvus RAG 初始化失败（{exc}），回退到 local_kb")
            kb_mode = "local_kb"

    if kb_mode == "local_kb":
        try:
            from src.tools.local_kb import local_kb_search
            local_kb_search("变压器", top_k=1)  # 触发索引加载，尽早暴露缺失
            mcp.register_tool("rag_search", local_kb_search)
            return mcp
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] local_kb 初始化失败（{exc}），回退到 mock")

    mcp.register_tool("rag_search", mock_rag_search)
    return mcp


# ──────────────────────────────────────────────────────────────
# 评测主流程
# ──────────────────────────────────────────────────────────────
def build_reflector(config: Dict[str, Any], mode: str):
    """
    mode: off | lexical | llm
    - lexical：离线词法评分器（不依赖接口）
    - llm    ：LLM 评分器，接口不可用时自动回退 lexical（ReflectionModule 内部处理）
    """
    if mode == "off":
        return None
    from src.agents.reflection import ReflectionModule, LexicalScorer, LLMScorer
    from src.tools.local_kb import get_local_kb
    kb = get_local_kb()
    scorer = LexicalScorer()
    if mode == "llm":
        try:
            from src.utils.llm import LLMClient, LLMConfig
            from src.utils.config import get_llm_config
            m = get_llm_config(config, "reflection")
            client = LLMClient(LLMConfig(
                provider=m.get("provider", "openai"), model_name=m.get("model_name", ""),
                temperature=0.0, max_tokens=int(m.get("max_tokens", 256)),
                base_url=m.get("base_url", "https://api.openai.com/v1"),
                api_key=m.get("api_key", ""), api_key_env=m.get("api_key_env", "OPENAI_API_KEY"),
                timeout=float(m.get("timeout", 60.0)), max_retries=int(m.get("max_retries", 1)),
            ))
            scorer = LLMScorer(client)
        except Exception as exc:  # noqa: BLE001
            print(f"[reflection] LLM 评分器不可用（{exc}），回退 lexical")
    return ReflectionModule(scorer=scorer, kb=kb)


def run_one(sample: Dict[str, Any], config: Dict[str, Any], mcp: MCPClient, reflector=None,
            planner_mode: str | None = None) -> Dict[str, Any]:
    from src.agents.planner import PlannerAgent
    from src.agents.retriever import RetrieverAgent
    from src.agents.generator import GeneratorAgent
    from src.agents.validator import ValidatorAgent

    state = AgentState(
        user_query=sample["user_query"],
        context=sample.get("context", {}),
        max_iterations=config.get("workflow", {}).get("max_iterations", 3),
    )

    planner = PlannerAgent(config, planner_mode=planner_mode)
    retriever = RetrieverAgent(config, mcp)
    generator = GeneratorAgent(config)
    validator = ValidatorAgent(config)

    t0 = time.time()
    try:
        final = run_diagnosis_workflow(state, planner, retriever, generator, validator, reflector=reflector)
        ok = True
    except Exception as exc:
        return {"completed": False, "error": str(exc), "elapsed": time.time() - t0}

    answer = final.final_answer or final.draft_answer or ""
    exp = sample["expected"]

    # 主故障命中（中文名出现在答案中）
    hit_primary = exp["primary_fault_cn"] in answer

    # 关键要点覆盖率
    kps = exp.get("key_points", [])
    covered = sum(1 for kp in kps if kp in answer)
    kp_coverage = covered / len(kps) if kps else 1.0

    # 安全合规
    need_safety = exp.get("must_mention_safety", False)
    has_safety = any(k in answer for k in SAFETY_KEYWORDS)
    safety_ok = (has_safety if need_safety else True)

    # 验证得分（从推理轨迹提取最后一次 evaluation）
    score = None
    for item in reversed(final.reasoning_trace):
        if item.get("agent") == "validator" and item.get("type") == "evaluation":
            score = item["content"].get("score")
            break

    # 降级判定：Planner 因 LLM 不可用/出错未产出计划，或 Generator/Validator 报 LLM 错误
    plan_status = getattr(final, "plan_status", None)
    fell_back = (
        final.fallback_mode
        or plan_status in ("llm_disabled", "llm_error", "parse_failed")
        or any(
            e.get("agent") in ("generator", "validator") and "LLM" in str(e.get("error", ""))
            for e in final.errors
        )
    )

    traj = getattr(final, "trajectory", []) or []
    n_calls = len(traj)
    n_ok = sum(1 for t in traj if t.get("exec_success") and t.get("business_success"))

    rlog = (getattr(final, "reflection_log", None) or [])
    rlog = rlog[-1] if rlog else {}

    return {
        "completed": ok,
        "hit_primary": hit_primary,
        "kp_coverage": kp_coverage,
        "need_safety": need_safety,
        "safety_ok": safety_ok,
        "score": score,
        "iterations": final.iteration,
        "verdict": final.validation_verdict,
        "fell_back": fell_back,
        "plan_status": getattr(final, "plan_status", None),
        "n_tool_calls": n_calls,
        "n_tool_ok": n_ok,
        "reflection": {
            "decision": rlog.get("decision"),
            "input_count": rlog.get("input_count", 0),
            "output_count": rlog.get("output_count", 0),
            "rewrite_triggered": bool(rlog.get("rewrite_triggered")),
            "scorer": rlog.get("scorer"),
        } if rlog else None,
        "elapsed": time.time() - t0,
        "answer_len": len(answer),
    }


def aggregate(results: List[Dict[str, Any]]) -> str:
    done = [r for r in results if r.get("completed")]
    n = len(results); nd = len(done)
    if nd == 0:
        return "# 端到端评测报告\n\n所有样本均执行失败，请检查依赖（LLM/工具）。\n"

    def rate(pred):
        s = [r for r in done if pred(r)]
        return len(s)

    primary = rate(lambda r: r["hit_primary"]) / nd
    avg_kp = sum(r["kp_coverage"] for r in done) / nd
    safety_samples = [r for r in done if r["need_safety"]]
    safety_rate = (sum(1 for r in safety_samples if r["safety_ok"]) / len(safety_samples)) if safety_samples else None
    scores = [r["score"] for r in done if isinstance(r["score"], (int, float))]
    avg_score = sum(scores) / len(scores) if scores else None
    avg_iter = sum(r["iterations"] for r in done) / nd
    fb_rate = sum(1 for r in done if r["fell_back"]) / nd
    avg_time = sum(r["elapsed"] for r in done) / nd

    L = ["# 多智能体系统端到端评测报告\n"]
    L.append(f"- 样本总数：{n}，成功完成：{nd}（**完成率 {nd/n:.1%}**）")
    L.append(f"- **主故障命中率**：{primary:.1%}")
    L.append(f"- **关键要点平均覆盖率**：{avg_kp:.1%}")
    if safety_rate is not None:
        L.append(f"- **安全合规率**（高危样本须含安全提示）：{safety_rate:.1%}（n={len(safety_samples)}）")
    if avg_score is not None:
        L.append(f"- **Validator 平均得分**：{avg_score:.2f}/10")
    L.append(f"- 平均迭代轮次：{avg_iter:.2f}")
    L.append(f"- LLM 降级率：{fb_rate:.1%}")
    L.append(f"- 平均单条耗时：{avg_time:.2f}s")
    ps: Dict[str, int] = {}
    for r in done:
        ps[str(r.get("plan_status"))] = ps.get(str(r.get("plan_status")), 0) + 1
    L.append(f"- Planner 计划状态分布：{ps}")
    tot_calls = sum(r.get("n_tool_calls", 0) for r in done)
    tot_ok = sum(r.get("n_tool_ok", 0) for r in done)
    L.append(f"- 工具调用：{tot_calls} 次，业务成功 {tot_ok} 次"
             + (f"（成功率 {tot_ok/tot_calls:.1%}）" if tot_calls else ""))
    refl = [r["reflection"] for r in done if r.get("reflection")]
    if refl:
        dec: Dict[str, int] = {}
        for x in refl:
            dec[str(x.get("decision"))] = dec.get(str(x.get("decision")), 0) + 1
        tin = sum(x["input_count"] for x in refl); tout = sum(x["output_count"] for x in refl)
        nrw = sum(1 for x in refl if x["rewrite_triggered"])
        scorers = sorted({str(x.get("scorer")) for x in refl})
        L.append(f"- 反思模块：评分器 {scorers}，决策分布 {dec}，输入块 {tin} → 输出块 {tout}"
                 f"（保留率 {tout/tin:.1%}），触发改写重检索 {nrw} 条" if tin else
                 f"- 反思模块：评分器 {scorers}，决策分布 {dec}，无检索输入")
    else:
        L.append("- 反思模块：未启用（--reflection off）")
    L.append("")
    L.append("## 指标说明")
    L.append("- **完成率**：流程连通性。低于 100% 说明存在崩溃，需查工具/LLM 配置。")
    L.append("- **主故障命中率 / 要点覆盖率**：诊断内容正确性的核心指标。")
    L.append("- **安全合规率**：安全底线。高危故障若未提示停电/保护，应视为严重缺陷。")
    L.append("- **Validator 平均得分 / 迭代轮次**：自我修正闭环的有效性。")
    L.append("- **LLM 降级率**：>0 时部分结果由模板/规则产生，命中率仅反映规则能力下限。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "data/synthetic/eval/eval_set.jsonl"))
    ap.add_argument("--out", default="", help="报告输出路径；留空则按 purpose 自动命名")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 条，0 表示全部")
    ap.add_argument("--kb-mode", default="local_kb", choices=["local_kb", "milvus", "mock"],
                    help="rag_search 后端")
    ap.add_argument("--real-rag", action="store_true", help="[兼容旧参数] 等价于 --kb-mode milvus")
    ap.add_argument("--no-llm", action="store_true", help="强制禁用 LLM（清空 api_key）")
    ap.add_argument("--purpose", default="dev_regression", choices=["dev_regression", "eval_set"],
                    help="dev_regression：开发回归（允许合成数据）；eval_set：正式评测（拒绝合成数据）")
    ap.add_argument("--reflection", default="off", choices=["off", "lexical", "llm"],
                    help="反思模块：off 关闭；lexical 离线词法评分；llm LLM 评分（不可用时回退 lexical）")
    ap.add_argument("--planner-mode", default=None, choices=["baseline", "finetuned"],
                    help="Planner 实验模式（P6 对比）：baseline 通用模型；finetuned 使用 llms.planner_finetuned；"
                         "留空则读 config.workflow.planner_mode")
    args = ap.parse_args()
    if args.real_rag:
        args.kb_mode = "milvus"

    # 数据守卫：合成数据不得用于正式评测
    assert_not_synthetic(args.eval, purpose=args.purpose)
    synthetic = is_synthetic_path(args.eval)
    if synthetic:
        print("[guard] 评测集为合成数据（S4），本次结果仅作开发回归，不得写入基线/论文报告。")

    if not args.out:
        suffix = "_dev" if args.purpose == "dev_regression" else ""
        args.out = str(Path(args.eval).parent / f"report_end2end{suffix}.md")

    config = load_config()
    if args.no_llm:
        for v in config.get("llms", {}).values():
            if isinstance(v, dict):
                v["api_key"] = ""
                v["api_key_env"] = "___DISABLED___"

    mcp = build_mcp(kb_mode=args.kb_mode)
    reflector = build_reflector(config, args.reflection)
    if args.reflection != "off" and not args.out.endswith(f"_refl_{args.reflection}.md"):
        args.out = args.out[:-3] + f"_refl_{args.reflection}.md" if args.out.endswith(".md") else args.out
    planner_mode = args.planner_mode or (config.get("workflow", {}) or {}).get("planner_mode") or "baseline"
    if planner_mode != "baseline" and args.out.endswith(".md") and f"_planner_{planner_mode}" not in args.out:
        args.out = args.out[:-3] + f"_planner_{planner_mode}.md"

    samples = []
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]

    print(f"开始端到端评测：{len(samples)} 条，kb_mode={args.kb_mode}, no_llm={args.no_llm}, "
          f"purpose={args.purpose}, reflection={args.reflection}, planner_mode={planner_mode}")
    results = []
    for i, s in enumerate(samples, 1):
        r = run_one(s, config, mcp, reflector=reflector, planner_mode=planner_mode)
        results.append(r)
        flag = "OK" if r.get("completed") else "ERR"
        print(f"  [{i}/{len(samples)}] {flag} {s['eval_id']} "
              f"hit={r.get('hit_primary')} kp={r.get('kp_coverage', 0):.0%} "
              f"verdict={r.get('verdict')} iter={r.get('iterations')}")

    report = aggregate(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
