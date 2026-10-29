#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output="models/short_experiment_$(date +%Y%m%d_%H%M%S).keras"
exec systemd-inhibit --what=idle:sleep --why="FDM short training experiment" \
  .venv/bin/python -u src/finetune_local.py \
  --model models/full_dataset_20261029_150330.keras \
  --split outputs/reports/local_20261029_150339/split.json \
  --output-model "$output" \
  --batch-size 4 --epochs 2 --fine-tune-epochs 4 \
  --learning-rate 0.00001 --fine-tune-lr 0.000001 \
  --class-weight-mode sqrt --max-per-class 0 --minutes 20 "$@"
