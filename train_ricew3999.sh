#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-ricew3999_ep50_bs8}"
mkdir -p "exp/$TAG/train.log"
python -u main.py \
  --data-path /root/autodl-tmp/P2RLoss-main/data/RiceW3999 \
  --label 100 --batch-size 8 --tag "$TAG" \
  --opts DATA.DATASET ricew3999 TRAIN.EPOCHS 50 \
  2>&1 | tee "exp/$TAG/train.log/running.log"
