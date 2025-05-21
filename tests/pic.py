from ultralytics import YOLO
import os
import cv2

model = YOLO('models/v4.pt')

model.predict(source='/home/axldvd/Documents/projects/ebs-ai/src/data/test/images/whirlpool_fwsl_61052_w_7_jpg.rf.0d7400536bcc59130fe569b224b26cc0.jpg', show=True)