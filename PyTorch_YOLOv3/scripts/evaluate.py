import torch
from ultralytics import YOLO

yaml_path = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/coco_yolo.yaml'

model = YOLO("/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/runs/detect/train/weights/best.pt")
# validation_results = model.val(data=yaml_path, imgsz=416, batch=16, conf=0.25, iou=0.6, device="0")

metrics = model.val(data=yaml_path, imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
print(metrics.box.map)
