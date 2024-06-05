import os
import time
from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_DIR = os.path.join(r'C:\Users\Ujjwal\Downloads', 'New folder')
video_path = os.path.join(VIDEOS_DIR, 'footage5.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'models', r'D:\2024\pythonProject1', 'runs', 'detect', 'train11', 'weights', 'last.pt')
model = YOLO(model_path)

threshold = 0.71
class_name_dict = {0: 'triangle'}

while ret:
    results = model(frame)

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # x1, y1, x2, y2
            conf = box.conf[0]  # confidence
            cls_id = box.cls[0]  # class id

            if conf > threshold:
                label = class_name_dict.get(int(cls_id), 'Unknown')
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,
                            2)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
