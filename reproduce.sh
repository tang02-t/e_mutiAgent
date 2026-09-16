#!/usr/bin/env bash
# 一键复现（离线部分，不调用 LLM）：v1 工程基座（DGA / 知识库 / 图谱 / 反思 / 规划数据 / D10）
#                                   + v2 三条主线的离线阶段（B 主动规划 / C 声明验证 / D 数据构造 / E 五级模式冒烟）+ 回归测试
#
# 用法：
#   bash reproduce.sh                  # 全部离线步骤（约 10–20 分钟）
#   bash reproduce.sh kb kg            # 只跑指定阶段
#   SKIP_EXISTING=1 bash reproduce.sh  # 产物已存在的阶段跳过
#
# 阶段名（按依赖顺序）：
#   v1 基座：dga | kb | kg | reflection | planner_data | d10
#   v2 主线：B（主动规划）| C（声明验证）| D（偏好数据，离线部分）| E（五级模式冒烟 + Oracle 校验）
#   收尾：   test（回归测试）
#
# 需要 LLM / 百炼账号的步骤不在本脚本内：
#   - A-1 基线、E-2 正式评测、LLM 裁判：见 PLAN_优化计划清单.md 对应节
#   - D-1 / D-3 百炼 SFT / DPO 训练与部署：training/planner_bailian/submit_job.py（默认 dry-run，--execute 才发请求）与 README.md
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
export LOG_LEVEL="${LOG_LEVEL:-WARNING}"
FILTER='grep -vE "jieba|Prefix|Loading|Building|Dumping|INFO|RuntimeWarning" || true'

STAGES="${*:-dga kb kg reflection planner_data d10 B C D E test}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
skip_if() { [[ "${SKIP_EXISTING:-0}" == "1" && -e "$1" ]]; }
run() { echo; echo "▶ $*"; eval "$* 2>&1 | $FILTER"; }

# ───────────────────────────── v1 工程基座 ─────────────────────────────
if has dga; then
  echo "== 0. DGA 数据合并与贝叶斯参数学习（B-1 校准）=="
  skip_if data/real/dga/dga_records.jsonl || run python3 scripts/convert_real_dga.py
  skip_if data/real/dga/learned_params.json || run python3 scripts/learn_cpt.py --kfold 5   # 5 折校准，写 docs/attribution_calibration.md
fi

if has kb; then
  echo "== 1. 知识库 =="
  skip_if data/kb/units.jsonl || run python3 scripts/kb/build_units.py
  skip_if data/kb/chunks.jsonl || run python3 scripts/kb/build_chunks.py
  run python3 scripts/kb/build_index.py
  skip_if data/kb/eval/retrieval_seed.jsonl || run python3 scripts/kb/build_retrieval_evalset.py
  run python3 scripts/kb/eval_retrieval.py --ablation
  run python3 scripts/kb/ablation_chunking.py
fi

if has kg; then
  echo "== 2. 图谱 =="
  run python3 scripts/kg/extract_rules.py
  run python3 scripts/kg/build_graph.py
  run python3 scripts/kg/build_kg_eval.py
fi

if has reflection; then
  echo "== 3. 反思 =="
  run python3 scripts/kb/build_reflection_evalset.py --n 300
  run python3 scripts/kb/eval_reflection_recall.py
fi

if has planner_data; then
  echo "== 4. 规划数据 D8（含百炼 SFT 格式导出与 test 封存校验）=="
  run python3 scripts/planner_data/build_task_seeds.py --execute
  run python3 scripts/planner_data/build_multi_turn.py
  # A-2 口语化改写产物随库保存（LLM 生成，不可确定性重建）；存在则用它导出，否则退回模板种子
  if [ -f data/planner/seeds/task_seeds_rewritten.jsonl ]; then
    run python3 scripts/planner_data/export_sft.py --seeds data/planner/seeds/task_seeds_rewritten.jsonl
  else
    run python3 scripts/planner_data/export_sft.py
  fi
  run python3 scripts/planner_data/check_sealed.py
  run python3 training/planner_sft/build_domain_terms.py
fi

if has d10; then
  echo "== 5. 端到端评测集 D10 =="
  run python3 scripts/eval/build_d10.py
fi

# ───────────────────────────── v2 主线 ─────────────────────────────
if has B; then
  echo "== B. 主线一：不确定性驱动的主动规划（EIG 抽检 → D12 模拟集 → 五策略对照）=="
  run python3 scripts/eval/eig_sanity_check.py                       # B-2：EIG 领域一致性抽检，写 docs/eig_sanity_check.md
  skip_if data/eval/d12/partial_obs.jsonl || run python3 scripts/sim/partial_obs_sim.py --build   # B-3：D12 1500 条
  run python3 scripts/sim/partial_obs_sim.py --check
  run python3 scripts/eval/eval_active_planning.py                   # B-5：五策略 × 三档，写 docs/active_planning_eval.md
fi

if has C; then
  echo "== C. 主线二：声明级证据约束验证（声明输出 → 核查器 → 路由 → D11 → 三组对照）=="
  run python3 scripts/eval/eval_claims_c1.py                         # C-1：30 条声明 schema 验收
  run python3 scripts/eval/eval_claim_checker_c2.py                  # C-2：确定性层正反例
  run python3 scripts/eval/eval_route_c3.py                          # C-3：补证与重规划路由
  skip_if data/eval/d11/fault_injection_eval.jsonl || run python3 scripts/eval/build_d11_fault_injection.py   # C-4：D11 300 条
  run python3 scripts/eval/eval_validator.py                         # C-5：v1 / v2_check / v2_route，写 docs/validator_eval.md
fi

if has D; then
  echo "== D. 主线三：偏好对构造管线验收（金标扰动 demo，不用于训练）=="
  run python3 scripts/planner_data/score_candidates.py --synthetic-demo --n 60 --seed 20260915   # D-2：打分 / 配对 / 导出 / 泄漏检查
  run python3 training/planner_bailian/submit_job.py create --stage sft --training-file file-placeholder --validation-file file-placeholder      # D-1：dry-run 请求体核对
fi

if has E; then
  echo "== E. 五级模式冒烟 + OraclePlanner 校验 =="
  run python3 scripts/eval/eval_planner_offline.py --self-test
  run python3 scripts/eval/eval_system_modes.py --modes all --no-llm --limit 5 --tag smoke
  run python3 scripts/eval/eval_system_modes.py --modes mode2 --no-llm --oracle-planner --limit 50 --tag smoke   # 正式 196 条结果为 results/oracle_mode2.jsonl，勿覆盖
fi

# ───────────────────────────── 回归测试 ─────────────────────────────
if has test; then
  echo "== 7. 回归测试 =="
  for t in tests/test_*.py; do echo "-- $t"; python3 "$t" 2>&1 | grep -E "FAIL|失败|passed|结果：|通过" | tail -1 || true; done
fi

echo
echo "完成。主要报告："
echo "  基座：docs/kb_ablation_test.md  docs/reflection_eval.md  data/planner/sft/DATA_CARD.md  data/eval/d10/DATA_CARD.md"
echo "  B：  docs/attribution_calibration.md  docs/eig_sanity_check.md  docs/active_planning_eval.md  data/eval/d12/DATA_CARD.md"
echo "  C：  docs/claims_acceptance.md  docs/claim_checker_acceptance.md  docs/route_acceptance.md  docs/validator_eval.md  data/eval/d11/DATA_CARD.md"
echo "  D：  data/planner/dpo/DATA_CARD.md  training/planner_bailian/README.md"
echo "  E：  docs/end2end_eval.md"
