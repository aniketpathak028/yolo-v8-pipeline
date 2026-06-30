#!/bin/bash
# Entrypoint for manual runs and cron scheduling.
# Logs stdout/stderr to results/pipeline.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/results/pipeline.log"
mkdir -p "$SCRIPT_DIR/results"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting CV benchmark run" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
source venv/bin/activate

python evaluate.py \
  --model yolov8n.pt \
  --data coco128.yaml \
  --images datasets/coco128/images/train2017 \
  --output results \
  --device cpu 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Run complete" | tee -a "$LOG_FILE"