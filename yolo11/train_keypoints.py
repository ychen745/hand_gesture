from ultralytics import YOLO

yaml_path = '/scratch/ychen855/hand_gesture/yolo11/config/hand_keypoints.yaml'
weight_path = '/scratch/ychen855/hand_gesture/yolo11/weights/yolo11n-pose.pt'

# Load a model
model = YOLO(weight_path)  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=yaml_path, epochs=100, imgsz=640)

