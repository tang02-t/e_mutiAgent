#!/usr/bin/env bash
# [v1 路线，已降级为备选] P5-3：对照实验矩阵一键运行（M1/M2/M3 × 2 seeds）+ 推理 + 离线评测
# v2 计划（PLAN_优化计划清单.md D 阶段）主路线为百炼 SFT + DPO，M2/M3 加权 SFT 不再作为论文对照；
# 本脚本与 plugin_loss_scale.py 仅在百炼 DPO 不可用时作为魔搭本地训练备选。
#
# 用法：
#   bash training/planner_sft/run_matrix.sh                 # 全矩阵
#   VARIANTS="M2 M3" SEEDS="42" bash training/planner_sft/run_matrix.sh
#   SKIP_TRAIN=1 bash training/planner_sft/run_matrix.sh    # 已训练完成，只做推理与评测
#
# 流程：train.sh → predict.py（对 test 集推理，写 predictions jsonl）→ eval_planner_offline.py（写 docs/planner_eval.md）
# M0（未微调基座）只做推理与评测：predict.py --adapters 留空。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VARIANTS="${VARIANTS:-M1 M2 M3}"
SEEDS="${SEEDS:-42 2026}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/planner_sft}"
PRED_DIR="${PRED_DIR:-$ROOT/data/planner/predictions}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
mkdir -p "$PRED_DIR"

best_ckpt() {  # 取 trainer_state 记录的最佳 checkpoint，缺失则取最后一个
  local dir="$1"
  local best
  best="$(python3 - "$dir" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
cands = sorted(glob.glob(os.path.join(d, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
best = None
for c in cands:
    st = os.path.join(c, "trainer_state.json")
    if os.path.exists(st):
        best = json.load(open(st)).get("best_model_checkpoint") or best
print(best or (cands[-1] if cands else ""))
PY
)"
  echo "$best"
}

# M0 基座
if [[ "${SKIP_M0:-0}" != "1" ]]; then
  echo "== M0 基座推理 =="
  python3 "$ROOT/training/planner_sft/predict.py" --model "$MODEL" --run-name M0 --out-dir "$PRED_DIR"
fi

for v in $VARIANTS; do
  for s in $SEEDS; do
    run="${v}_seed${s}"
    if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
      echo "== 训练 $run =="
      bash "$ROOT/training/planner_sft/train.sh" "$v" "$s"
    fi
    ckpt="$(best_ckpt "$OUT_ROOT/$run")"
    if [[ -z "$ckpt" ]]; then echo "未找到 $run 的 checkpoint，跳过" >&2; continue; fi
    echo "== 推理 $run ($ckpt) =="
    python3 "$ROOT/training/planner_sft/predict.py" --model "$MODEL" --adapters "$ckpt" --run-name "$run" --out-dir "$PRED_DIR"
  done
done

echo "== 离线评测 → docs/planner_eval.md =="
python3 "$ROOT/scripts/eval/eval_planner_offline.py" --pred-dir "$PRED_DIR" --write-report
