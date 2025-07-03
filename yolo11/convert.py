import os
from ultralytics import YOLO

model = YOLO("/scratch/ychen855/hand_gesture/yolo11/scripts/runs/detect/train2/weights/best.pt")

model.export(format="coreml")
