#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
WORKERS="${WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
for key in BATCH_SIZE EPOCHS WORKERS VAL_EVERY; do
  value="${!key}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$key must be an integer" >&2
    exit 1
  fi
  if [[ "$key" != WORKERS && "$value" -eq 0 ]]; then
    echo "$key must be greater than zero" >&2
    exit 1
  fi
done
TAG="${1:-ricew3999_ep${EPOCHS}_bs${BATCH_SIZE}}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "TAG must contain only letters, numbers, dots, underscores and hyphens" >&2
  exit 1
fi
DATA_PATH="${DATA_PATH:-$PWD/data/RiceW3999}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$PWD/pretrained/vgg16_bn-6c64b313.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
for split in train val; do
  if [[ ! -s "$DATA_PATH/ImageSets/$split.txt" ]]; then
    echo "Missing or empty $DATA_PATH/ImageSets/$split.txt; see data/README.md" >&2
    exit 1
  fi
done
for folder in JPEGImages PointAnnotations; do
  if [[ ! -d "$DATA_PATH/$folder" ]]; then
    echo "Missing $DATA_PATH/$folder; see data/README.md" >&2
    exit 1
  fi
done
if [[ ! -f "$WEIGHTS_PATH" ]]; then
  echo "Missing VGG16-BN weights: $WEIGHTS_PATH (override with WEIGHTS_PATH=/path/to/weights.pth)" >&2
  exit 1
fi
if [[ -d "exp/$TAG/output" ]] && [[ -n "$(find "exp/$TAG/output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Experiment exp/$TAG/output is not empty. Use a new TAG; this trainer does not support full-state resume." >&2
  exit 1
fi
mkdir -p "exp/$TAG/train.log"
echo "Data=$DATA_PATH | batch=$BATCH_SIZE | epochs=$EPOCHS | workers=$WORKERS | validation interval=$VAL_EVERY | labels=100%"
"$PYTHON_BIN" -u main.py \
  --data-path "$DATA_PATH" \
  --label 100 --batch-size "$BATCH_SIZE" --tag "$TAG" \
  --opts DATA.DATASET ricew3999 TRAIN.EPOCHS "$EPOCHS" \
  DATA.NUM_WORKERS "$WORKERS" SAVE_FREQ "$VAL_EVERY" \
  MODEL.PRETRAINED "$WEIGHTS_PATH" \
  2>&1 | tee "exp/$TAG/train.log/running.log"
