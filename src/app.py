from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os

app = Flask(__name__)

model = YOLO("")  

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "YOLOv11 E-waste Model API is running."})

@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image = request.files["image"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
        image.save(temp_img.name)
        results = model.predict(source=temp_img.name, save=False, conf=0.3)

    os.remove(temp_img.name)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": result.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                }
            })

    return jsonify({"detections": detections})
