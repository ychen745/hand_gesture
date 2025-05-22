from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/scratch/ychen855/hand_gesture/yolo11/config/hand.yaml', epochs=100, imgsz=640)