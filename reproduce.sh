#!/usr/bin/env bash
# 一键复现（离线部分，不调用 LLM）：清洗 → 分块 → 索引 → 评测集 → 图谱 → 造数 → 导出 → 评测 → 回归测试
#
# 用法：
#   bash reproduce.sh                # 全部离线步骤（约 5–10 分钟）
#   bash reproduce.sh kb kg          # 只跑指定阶段：dga | kb | kg | reflection | planner_data | d10 | eval | test
#   SKIP_EXISTING=1 bash reproduce.sh  # 产物已存在的阶段跳过
#
# 需要 LLM / GPU 的步骤不在本脚本内，见 docs/planner_training.md 与 PLAN_优化计划清单.md 进度表。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
export LOG_LEVEL="${LOG_LEVEL:-WARNING}"
FILTER='grep -vE "jieba|Prefix|Loading|Building|Dumping|INFO|RuntimeWarning" || true'

STAGES="${*:-dga kb kg reflection planner_data d10 eval test}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
skip_if() { [[ "${SKIP_EXISTING:-0}" == "1" && -e "$1" ]]; }
run() { echo; echo "▶ $*"; eval "$* 2>&1 | $FILTER"; }

if has dga; then
  echo "== 0. DGA 数据合并与贝叶斯参数学习 =="
  skip_if data/real/dga/dga_records.jsonl || run python3 scripts/convert_real_dga.py
  skip_if data/real/dga/learned_params.json || run python3 scripts/learn_cpt.py --kfold 5   # B-1：5 折校准，写 docs/attribution_calibration.md
  run python3 scripts/eval/eig_sanity_check.py   # B-2：EIG 领域一致性抽检，写 docs/eig_sanity_check.md
fi

if has kb; then
  echo "== 1. 知识库（P1）=="
  skip_if data/kb/units.jsonl || run python3 scripts/kb/build_units.py
  skip_if data/kb/chunks.jsonl || run python3 scripts/kb/build_chunks.py
  run python3 scripts/kb/build_index.py
  skip_if data/kb/eval/retrieval_seed.jsonl || run python3 scripts/kb/build_retrieval_evalset.py
  run python3 scripts/kb/eval_retrieval.py --ablation
  run python3 scripts/kb/ablation_chunking.py
fi

if has kg; then
  echo "== 2. 图谱（P2）=="
  run python3 scripts/kg/extract_rules.py
  run python3 scripts/kg/build_graph.py
  run python3 scripts/kg/build_kg_eval.py
fi

if has reflection; then
  echo "== 3. 反思（P3）=="
  run python3 scripts/kb/build_reflection_evalset.py --n 300
  run python3 scripts/kb/eval_reflection_recall.py
fi

if has planner_data; then
  echo "== 4. 规划数据（P4）=="
  run python3 scripts/planner_data/build_task_seeds.py --execute
  run python3 scripts/planner_data/build_multi_turn.py
  run python3 scripts/planner_data/export_sft.py
  run python3 training/planner_sft/build_domain_terms.py
fi

if has d10; then
  echo "== 5. 端到端评测集 D10（P6-1）=="
  run python3 scripts/eval/build_d10.py
fi

if has eval; then
  echo "== 6. 离线评测自检（P5-4 / P6-2 OraclePlanner）=="
  run python3 scripts/eval/eval_planner_offline.py --self-test
  run python3 scripts/eval/eval_system_modes.py --modes mode5 --no-llm --oracle-planner --limit 50
fi

if has test; then
  echo "== 7. 回归测试 =="
  for t in tests/test_*.py; do echo "-- $t"; python3 "$t" 2>&1 | grep -E "FAIL|结果：" || true; done
fi

echo; echo "完成。报告：docs/kb_ablation_test.md  docs/reflection_eval.md  docs/end2end_eval.md  data/planner/sft/DATA_CARD.md  data/eval/d10/DATA_CARD.md"
