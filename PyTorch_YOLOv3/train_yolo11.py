import torch
from ultralytics import YOLO

yaml_path = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/coco_yolo.yaml'

model = YOLO("yolo11n.pt")
# validation_results = model.val(data=yaml_path, imgsz=416, batch=16, conf=0.25, iou=0.6, device="0")

results = model.train(data=yaml_path, epochs=100, imgsz=640)

