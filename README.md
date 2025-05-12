# Object Classification Model API

## Production Deployment (Google Cloud Run)

### 1. Requirements
- Python 3.8+
- Docker
- Google Cloud SDK (for local testing/deployment)
- Google Cloud Storage bucket (for image uploads)

### 2. Environment Variables
Set these in Cloud Run (or locally for testing):
- `GOOGLE_APPLICATION_CREDENTIALS` (path to GCP service account JSON, if needed)
- `MODEL_PATH` (optional, path to YOLO model weights)

### 3. Build & Run Locally (for testing)
```bash
# Build Docker image
DOCKER_BUILDKIT=1 docker build -t ewaste-api .

# Run locally
# (Set env vars as needed)
docker run -p 8080:8080 --env-file .env ewaste-api
```

### 4. Deploy to Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ewaste-api

gcloud run deploy ewaste-api \
  --image gcr.io/YOUR_PROJECT_ID/ewaste-api \
  --platform managed \
  --region YOUR_REGION \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_URL=... \
  --set-env-vars GEMINI_API_KEY=... \
  --set-env-vars MODEL_PATH=models/yolo11n.pt
```

### 5. API Usage
- **POST /predict**: Upload an image for prediction and validation
- **GET /**: Health check

### 6. Notes
- The app uses FastAPI and is served with Uvicorn (see Dockerfile).
- All model/data/config paths are relative to the container root.
- For best performance, use a GPU-enabled Cloud Run instance (if available).

---
