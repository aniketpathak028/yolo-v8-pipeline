FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate.py .

# Results volume — mount externally to persist reports
VOLUME ["/app/results", "/app/datasets"]

ENTRYPOINT ["python", "evaluate.py"]
CMD ["--model", "yolov8n.pt", "--data", "coco128.yaml", \
     "--images", "datasets/coco128/images/train2017", \
     "--output", "results", "--device", "cpu"]