from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import logging
import tempfile
import time
import uuid
import requests
from datetime import datetime
from cloud_storage import CloudStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
SAVE_DIR = os.path.join(os.getcwd(), "runs/detect")
os.makedirs(SAVE_DIR, exist_ok=True)

cloud_storage = CloudStorage()

# Load model
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = YOLO("yolov8n.pt")
    logger.info("Loaded default YOLOv11n model")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "YOLOv8 E-waste Classification API is running",
        "model": MODEL_PATH,
        "version": "1.0.0",
        "classes": 10,
        "timestamp": datetime.now().isoformat(),
        "cloud_storage": cloud_storage.enabled
    })

@app.route("/infer", methods=["POST"])
def infer():
    start_time = time.time()
    
    if "image" not in request.files:
        logger.warning("No image uploaded in request")
        return jsonify({"error": "No image uploaded."}), 400
    
    image = request.files["image"]
    if not image.filename:
        logger.warning("Empty image file submitted")
        return jsonify({"error": "Empty image file."}), 400
    
    save_results = request.form.get("save", "false").lower() == "true"
    
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Processing inference request {run_id} for file: {image.filename}")
    
    try:
        results = model.predict(
            source=image.stream,
            conf=float(request.form.get("conf", "0.35")),
            iou=float(request.form.get("iou", "0.45")),
            save=save_results,
            project=SAVE_DIR,
            name=run_id
        )
        
        detections = []
        result_image_path = None
        cloud_url = None
        
        for result in results:
            if save_results and result.path:
                result_image_path = result.path
                
                if cloud_storage.enabled:
                    cloud_url = cloud_storage.upload_detection_result(result_image_path, run_id)
                
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                
                detection = {
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1),
                    }
                }
                
                detections.append(detection)
        
        process_time = round(time.time() - start_time, 3)
        logger.info(f"Inference {run_id} complete: {len(detections)} objects found in {process_time}s")
        
        response = {
            "run_id": run_id,
            "detections": detections,
            "processing_time": process_time,
            "image_saved": save_results and result_image_path is not None,
        }
        
        if cloud_url:
            response["result_url"] = cloud_url
        
        if request.form.get("predict_price", "false").lower() == "true" and detections:
            try:
                price_api_url = os.environ.get("PRICE_API_URL", "http://price-service/predict-price")
                price_response = requests.post(
                    price_api_url,
                    json={"detections": detections}
                )
                
                if price_response.status_code == 200:
                    response["price_estimation"] = price_response.json()
                    logger.info(f"Price estimation successful for {run_id}")
                else:
                    logger.warning(f"Price API returned status {price_response.status_code}")
            except Exception as price_error:
                logger.error(f"Failed to get price estimation: {price_error}")
                response["price_error"] = str(price_error)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/results/<run_id>", methods=["GET"])
def get_results(run_id):
    result_dir = os.path.join(SAVE_DIR, run_id)
    
    if not os.path.exists(result_dir):
        return jsonify({"error": "Results not found"}), 404
    
    image_files = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        return jsonify({"error": "No result images found"}), 404
    
    return send_file(os.path.join(result_dir, image_files[0]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    logger.info(f"Starting E-waste detection API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
