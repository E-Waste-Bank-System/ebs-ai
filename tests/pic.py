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
    source='/home/axldvd/Documents/projects/ebs-ai/src/data/w2.jpeg',
    show=True,
    conf=0.2,        # Confidence threshold (0-1)
    iou=0.2,         # NMS IoU threshold
    max_det=50,       # Maximum number of detections
    verbose=True      # Print detection results
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