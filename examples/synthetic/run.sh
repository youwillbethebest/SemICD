#!/usr/bin/env bash
# Complete the README walkthrough, including checkpoint reload evaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."
output="${1:-artifacts/synthetic-walkthrough}"
if [[ "$output" != /* ]]; then
  output="$PWD/$output"
fi
python examples/synthetic/make_model.py --output-dir "$output/tiny-model"
semicd prepare-training --train-file examples/synthetic/train.jsonl \
  --profiles-file examples/synthetic/profiles.jsonl \
  --sid-bundle examples/synthetic/bundle --output-dir "$output/tasks"
semicd train --model-name-or-path "$output/tiny-model" \
  --train-file "$output/tasks/train.jsonl" \
  --validation-file examples/synthetic/validation.jsonl \
  --max-seq-length 256 --max-new-tokens 12 --gradient-accumulation-steps 8 \
  --epochs 2 --learning-rate 0.01 --no-gradient-checkpointing \
  --output-dir "$output/run"
semicd evaluate --checkpoint "$output/run" \
  --eval-file examples/synthetic/validation.jsonl --output-dir "$output/reloaded-validation"
python tests/check_synthetic_run.py "$output"
