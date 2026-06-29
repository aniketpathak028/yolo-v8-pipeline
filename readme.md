# Yolo v8 Pipeline

A lightweight, automated evaluation pipeline for object detection models. Runs YOLOv8 inference on a dataset, computes standard CV metrics, and logs structured JSON reports per run. 

## Metrics computed

| Metric | Description |
|---|---|
| `mAP@0.5` | Mean Average Precision at IoU threshold 0.5 |
| `mAP@0.5:0.95` | mAP averaged across IoU thresholds 0.5–0.95 |
| `detection_rate_pct` | % of frames containing at least one detection |
| `temporal_jitter_px` | Mean centroid displacement between consecutive frames (px) |
| `avg_inference_ms` | Average per-frame inference time in milliseconds |

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/aniketpathak028/cv-benchmark-pipeline
cd cv-benchmark-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run evaluation (downloads COCO128 and yolov8n.pt automatically on first run)
python evaluate.py

# 3. Check results
cat results/run_<timestamp>.json
```

## Sample output

```json
{
  "run_id": "20250612_020001",
  "model": "yolov8n.pt",
  "device": "cpu",
  "images_evaluated": 100,
  "metrics": {
    "mAP@0.5": 0.6023,
    "mAP@0.5:0.95": 0.4471,
    "detection_rate_pct": 94.0,
    "temporal_jitter_px": 12.4,
    "avg_inference_ms": 187.3
  },
  "timestamp": "2025-06-12T02:00:01.234567"
}
```

## CLI options

```
--model     YOLOv8 weights file (default: yolov8n.pt)
--data      Dataset YAML in Ultralytics format (default: coco128.yaml)
--images    Path to images directory
--output    Output directory for JSON reports (default: results/)
--device    cpu or 0 for GPU (default: cpu)
```

## Docker

```bash
docker build -t cv-benchmark .
docker run -v $(pwd)/results:/app/results -v $(pwd)/datasets:/app/datasets cv-benchmark
```

## Automated scheduling

See `cron.example` for a ready-to-use crontab entry that runs the pipeline nightly at 02:00.

## Switching models

Any Ultralytics-compatible model works as a drop-in:

```bash
python evaluate.py --model yolov8s.pt  
python evaluate.py --model yolov8n.pt 
```