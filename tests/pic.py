from ultralytics import YOLO
import os
import cv2
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO('models/v4.pt')

# Configure model prediction parameters
results = model.predict(
    source='/home/axldvd/Documents/projects/ebs-ai/tests/detection_0a017e45-5507-49b9-b99d-559f5b2739c2.jpg',
    show=True,
    conf=0.25,       # Increased confidence threshold for better accuracy
    iou=0.45,        # Increased IoU threshold to allow more overlapping detections
    max_det=50,      # Maximum number of detections
    verbose=True     # Print detection results
)

# Log raw results for debugging
logger.info("Raw Results:")
logger.info(f"Number of results: {len(results)}")
logger.info(f"Number of boxes in first result: {len(results[0].boxes)}")

# Log detected objects
logger.info(f"Total objects detected: {len(results[0].boxes)}")
logger.info("Detected Objects:")
for i, box in enumerate(results[0].boxes, 1):
    class_id = int(box.cls)
    confidence = float(box.conf)
    bbox = box.xyxy[0].tolist()  # Get bounding box coordinates
    logger.info(f"Object {i}:")
    logger.info(f"  - Class ID: {class_id}")
    logger.info(f"  - Confidence: {confidence:.2f}")
    logger.info(f"  - Bounding Box: {bbox}")