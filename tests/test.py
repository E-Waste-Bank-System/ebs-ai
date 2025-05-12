from ultralytics import YOLO
import os

model = YOLO('runs/detect/train/weights/last.pt')

# model.predict(source='src/test/4.jpg', show=True, save=True, project='results', name='webcam_test')
model.predict(source=2, show=True)


output_dir = 'results/webcam_test'
for file in os.listdir(output_dir):
    if file.endswith('.avi'):
        os.rename(os.path.join(output_dir, file), os.path.join(output_dir, file.replace('.avi', '.mp4')))