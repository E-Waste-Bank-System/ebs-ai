from ultralytics import YOLO
import os
import cv2

model = YOLO('models/v4.pt')

model.predict(source='/home/axldvd/Documents/projects/ebs-ai/src/data/detection_36acd18f-1c77-4a87-b3ab-5b9681104eda.jpg', show=True)