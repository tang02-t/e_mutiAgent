#!/usr/bin/env bash
# P5-2 / P5-3：Planner LoRA-SFT 训练脚本（ms-swift，魔搭 A10 24GB 单卡）
#
# 用法：
#   bash training/planner_sft/train.sh <M1|M2|M3> [seed] [extra swift args...]
# 环境变量：
#   MODEL        基座（默认 Qwen/Qwen3-VL-8B-Instruct，与叶金涛论文一致；百炼导入要求训练基座与导入基座一致）
#   DATA_DIR     SFT 数据目录（默认 data/planner/sft）
#   OUT_ROOT     输出根目录（默认 output/planner_sft）
#   EPOCHS       训练轮数上限（默认 5；PLAN P5-2：3-5 轮按验证集早停，不照抄 20）
#   PILOT=1      50 条小闭环（P5-1）：只取 50 条训练样本、1 轮，用于「训练→导出→OSS→百炼→调用」流程验证
#
# 对照矩阵（PLAN P5-3）：
#   M0 未微调基座（无需训练，直接评测基座）
#   M1 普通 LoRA-SFT              --loss_scale default
#   M2 结构 Token 加权            --loss_scale planner_struct  --loss_type normalized_weighted_ce
#   M3 结构 + 领域 Token 加权     --loss_scale planner_domain  --loss_type normalized_weighted_ce
#
# 公共配置（PLAN P5-2）：LoRA rank 8 / alpha 32 / dropout 0.05；lr 5e-5；AdamW；warmup 20 步；cosine；bf16；
#   seq_len 2048；有效 batch 16（per_device 2 × grad_accum 8）；冻结 ViT（百炼导入约束）；不改词表与对话模板。
set -euo pipefail

VARIANT="${1:?用法: train.sh <M1|M2|M3> [seed]}"
SEED="${2:-42}"
shift $(( $# >= 2 ? 2 : $# ))

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-$ROOT/data/planner/sft}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/planner_sft}"
EPOCHS="${EPOCHS:-5}"
PLUGIN="$ROOT/training/planner_sft/plugin_loss_scale.py"

TRAIN_SETS=("$DATA_DIR/swift_train.jsonl" "$DATA_DIR/swift_multiturn_train.jsonl")
DEV_SETS=("$DATA_DIR/swift_dev.jsonl" "$DATA_DIR/swift_multiturn_dev.jsonl")

case "$VARIANT" in
  M1) LOSS_ARGS=(--loss_scale default) ;;
  M2) LOSS_ARGS=(--loss_scale planner_struct --loss_type normalized_weighted_ce --external_plugins "$PLUGIN") ;;
  M3) LOSS_ARGS=(--loss_scale planner_domain --loss_type normalized_weighted_ce --external_plugins "$PLUGIN") ;;
  *) echo "未知变体 $VARIANT（应为 M1/M2/M3）" >&2; exit 2 ;;
esac

RUN_NAME="${VARIANT}_seed${SEED}"
if [[ "${PILOT:-0}" == "1" ]]; then
  RUN_NAME="pilot_${RUN_NAME}"
  EPOCHS=1
  TRAIN_SETS=("$DATA_DIR/swift_train.jsonl#40" "$DATA_DIR/swift_multiturn_train.jsonl#10")
  DEV_SETS=("$DATA_DIR/swift_dev.jsonl#20")
fi
OUT_DIR="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "[train.sh] variant=$VARIANT seed=$SEED model=$MODEL epochs=$EPOCHS out=$OUT_DIR"
printf '%s\n' "variant=$VARIANT" "seed=$SEED" "model=$MODEL" "epochs=$EPOCHS" "date=$(date -Iseconds)" > "$OUT_DIR/run_meta.txt"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift sft \
  --model "$MODEL" \
  --train_type lora \
  --dataset "${TRAIN_SETS[@]}" \
  --val_dataset "${DEV_SETS[@]}" \
  --agent_template hermes \
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 \
  --learning_rate 5e-5 --optim adamw_torch \
  --lr_scheduler_type cosine --warmup_steps 20 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 --truncation_strategy delete \
  --gradient_checkpointing true \
  --eval_strategy epoch --save_strategy epoch \
  --save_total_limit 5 \
  --load_best_model_at_end true --metric_for_best_model eval_loss --greater_is_better false \
  --early_stop_interval 2 \
  --logging_steps 10 --report_to tensorboard \
  --dataloader_num_workers 4 \
  --seed "$SEED" \
  --output_dir "$OUT_DIR" \
  "${LOSS_ARGS[@]}" \
  "$@" 2>&1 | tee "$OUT_DIR/train.log"

echo "[train.sh] 完成。验证曲线：tensorboard --logdir $OUT_DIR ；最佳 checkpoint 见 $OUT_DIR/checkpoint-*"
