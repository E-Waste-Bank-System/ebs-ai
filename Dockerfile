FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p runs/detect

ENV MODEL_PATH=models/yolo11n.pt
ENV PORT=8080
ENV DEBUG=False

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/ || exit 1

CMD ["uvicorn", "src.core.app:app", "--host", "0.0.0.0", "--port", "8080"]
