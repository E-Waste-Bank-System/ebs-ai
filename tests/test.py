from ultralytics import YOLO
import os
import cv2

model = YOLO('models/v3.pt')

model.predict(source=4, show=True)  

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

output_dir = 'results/webcam_test'
for file in os.listdir(output_dir):
    if file.endswith('.avi'):
        os.rename(os.path.join(output_dir, file), os.path.join(output_dir, file.replace('.avi', '.mp4')))