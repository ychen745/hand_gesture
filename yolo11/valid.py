from ultralytics import YOLO

yaml_path = 'config/hand_keypoints.yaml'
weight_path = 'runs/pose/train3/weights/best.pt'

# Load a model
model = YOLO(weight_path)

# Validate the model
metrics = model.val(data=yaml_path, imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
print(metrics.box.map)  # map50-95
