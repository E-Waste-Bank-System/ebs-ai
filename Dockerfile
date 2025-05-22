FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models runs/detect knr_models

# Copy the model files
COPY models/v4.pt models/
COPY models/model_knr_best.joblib knr_models/
COPY models/encoder_target.joblib knr_models/

# Copy the rest of the application
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/models/v4.pt
ENV KNR_MODEL_PATH=/app/knr_models/model_knr_best.joblib
ENV KNR_ENCODER_PATH=/app/knr_models/encoder_target.joblib
ENV PORT=8080
ENV DEBUG=False
ENV PYTHONPATH=/app

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/ || exit 1

# Run the application
CMD ["uvicorn", "src.core.app:app", "--host", "0.0.0.0", "--port", "8080"]
