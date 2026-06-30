"""
Object Detection Benchmarking Pipeline
Runs YOLOv8 inference on COCO128 validation images, computes detection rate,
temporal jitter, and mAP, then logs results as a timestamped JSON report.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


def compute_jitter(detections_per_frame: list[list[tuple]]) -> float:
    """
    Compute mean centroid displacement between consecutive frames.
    For each frame pair, matches detections by class and computes
    centroid distance. Returns mean jitter in pixels.
    """
    jitters = []
    for i in range(1, len(detections_per_frame)):
        prev = detections_per_frame[i - 1]
        curr = detections_per_frame[i]
        if not prev or not curr:
            continue
        # Group by class, compare centroids of first matched detection
        prev_by_class = {cls: (cx, cy) for cx, cy, cls in prev}
        for cx, cy, cls in curr:
            if cls in prev_by_class:
                px, py = prev_by_class[cls]
                jitters.append(float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2)))
    return round(float(np.mean(jitters)), 2) if jitters else 0.0


def run_evaluation(model_name: str, data_yaml: str, images_dir: str,
                   output_dir: str, device: str = "cpu") -> dict:
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # --- mAP via official YOLO val ---
    print("Running validation (mAP)...")
    val_results = model.val(data=data_yaml, device=device, verbose=False)
    map50 = round(float(val_results.box.map50), 4)
    map50_95 = round(float(val_results.box.map), 4)

    # --- Per-image inference for detection rate + jitter ---
    image_paths = sorted(Path(images_dir).glob("*.jpg"))[:100]  # cap at 100 for speed
    if not image_paths:
        image_paths = sorted(Path(images_dir).glob("*.png"))[:100]

    print(f"Running inference on {len(image_paths)} images...")
    frames_with_detection = 0
    detections_per_frame = []
    inference_times = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        results = model.predict(img, device=device, verbose=False)
        inference_times.append(time.perf_counter() - t0)

        boxes = results[0].boxes
        frame_detections = []
        if boxes is not None and len(boxes):
            frames_with_detection += 1
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = round((x1 + x2) / 2, 1)
                cy = round((y1 + y2) / 2, 1)
                cls = int(box.cls[0].item())
                frame_detections.append((cx, cy, cls))
        detections_per_frame.append(frame_detections)

    detection_rate = round(frames_with_detection / len(image_paths) * 100, 2) if image_paths else 0.0
    jitter = compute_jitter(detections_per_frame)
    avg_inference_ms = round(float(np.mean(inference_times)) * 1000, 1) if inference_times else 0.0

    report = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "device": device,
        "images_evaluated": len(image_paths),
        "metrics": {
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map50_95,
            "detection_rate_pct": detection_rate,
            "temporal_jitter_px": jitter,
            "avg_inference_ms": avg_inference_ms,
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_path = Path(output_dir) / f"run_{report['run_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Results ===")
    print(f"  mAP@0.5:       {map50}")
    print(f"  mAP@0.5:0.95:  {map50_95}")
    print(f"  Detection rate: {detection_rate}%")
    print(f"  Temporal jitter: {jitter}px")
    print(f"  Avg inference:  {avg_inference_ms}ms")
    print(f"\nReport saved to: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Benchmarking Pipeline")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset YAML (ultralytics format)")
    parser.add_argument("--images", default="datasets/coco128/images/train2017", help="Path to images dir")
    parser.add_argument("--output", default="results", help="Output directory for JSON reports")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu or 0 (GPU)")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        data_yaml=args.data,
        images_dir=args.images,
        output_dir=args.output,
        device=args.device,
    )