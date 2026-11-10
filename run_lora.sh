#!/usr/bin/env bash
# Copyright (c) 2026 Paulo Condado
# License: MIT
#
# Train Whisper with LoRA for every combination of model, rank (r) and
# learning rate. alpha is always 2 x r. Each run writes its own output folder
# (output_lora_<MODEL>_r<R>_a<ALPHA>_lr<LR>_...) and a log in sweep_logs/.
#
# Usage (from the repository root):
#     bash run_lora.sh

MODELS=(medium large-v3)          # medium first: it is faster
RANKS=(2 4 8 16)
LEARNING_RATES=(3e-5 4e-5 5e-5)

# Show Python output on screen immediately (it is also saved to the log).
export PYTHONUNBUFFERED=1

LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"

total=$(( ${#MODELS[@]} * ${#RANKS[@]} * ${#LEARNING_RATES[@]} ))
run=0
failed=0

for model in "${MODELS[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for r in "${RANKS[@]}"; do
      run=$((run + 1))
      alpha=$((r * 2))
      log="$LOG_DIR/${model}_r${r}_a${alpha}_lr${lr}.log"

      echo "[$run/$total] $(date '+%Y-%m-%d %H:%M:%S')  model=$model  r=$r  alpha=$alpha  lr=$lr  -> $log"

      python src/train/whisper_lora_finetune/train.py \
        --dataset data/train/data.csv \
        --audio-dir data/train/audio/ \
        --model "$model" \
        --val-ratio 0.15 \
        --seed 3 \
        --epochs 13 \
        --group-by-text \
        --early-stopping-patience 4 \
        --min-val-words 4 \
        --long-audio keep \
        --long-audio-train-only \
        --learning-rate "$lr" \
        --lora-r "$r" \
        --lora-alpha "$alpha" \
        2>&1 | tee "$log"

      # Exit code of python, not of tee.
      if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        failed=$((failed + 1))
        echo "        FAILED (see $log), continuing with the next run"
      fi
    done
  done
done

echo "Done: $total runs, $failed failed. Logs in $LOG_DIR/"
