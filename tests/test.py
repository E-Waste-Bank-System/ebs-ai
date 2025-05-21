from ultralytics import YOLO
import os
import cv2

# Load your trained model
model = YOLO('models/v4.pt')

# Evaluate the model on the test dataset
# This will generate confusion matrix, precision, recall, and other metrics
results = model.val(
    data='configs/data.yaml',  # Path to your data config file
    split='test',             # Use test split
    imgsz=640,               # Image size
    batch=16,                # Batch size
    plots=True,              # Generate plots
    save_json=True,          # Save results as JSON
    save_hybrid=True,        # Save hybrid labels
    conf=0.001,              # Confidence threshold
    iou=0.6,                 # IoU threshold
    max_det=300,             # Maximum detections per image
    device='0'               # Use GPU if available
)

# Print the evaluation metrics
print("\nEvaluation Results:")
print(f"mAP50: {results.box.map50:.3f}")
print(f"mAP50-95: {results.box.map:.3f}")
print(f"Mean Precision: {results.box.mp:.3f}")  # Changed from mp() to mp
print(f"Mean Recall: {results.box.mr:.3f}")     # Changed from mr() to mr

# Print per-class results
print("\nPer-class Results:")
for i in range(results.box.nc):  # nc is number of classes
    p, r, ap50, ap = results.box.class_result(i)
    print(f"Class {i}:")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall: {r:.3f}")
    print(f"  AP50: {ap50:.3f}")
    print(f"  AP50-95: {ap:.3f}")