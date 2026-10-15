#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-ricew3999_ep50_bs8}"
DATA_PATH="${DATA_PATH:-$PWD/data/RiceW3999}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$PWD/vgg16_bn-6c64b313.pth}"
if [[ ! -f "$DATA_PATH/ImageSets/train.txt" || ! -f "$DATA_PATH/ImageSets/val.txt" ]]; then
  echo "Missing ImageSets/train.txt or val.txt under $DATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS_PATH" ]]; then
  echo "Missing VGG16-BN weights: $WEIGHTS_PATH (override with WEIGHTS_PATH=/path/to/weights.pth)" >&2
  exit 1
fi
mkdir -p "exp/$TAG/train.log"
python -u main.py \
  --data-path "$DATA_PATH" \
  --label 100 --batch-size 8 --tag "$TAG" \
  --opts DATA.DATASET ricew3999 TRAIN.EPOCHS 50 MODEL.PRETRAINED "$WEIGHTS_PATH" \
  2>&1 | tee "exp/$TAG/train.log/running.log"
