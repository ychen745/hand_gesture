PYTHON=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/test.py
MODEL=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.cfg
DATA=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.data
WEIGHTS=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/checkpoints/hand_yolo_lr0.0001_conf0.3/60.pth
# WEIGHTS=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/weights/cross-hands.weights

python ${PYTHON} --model ${MODEL} --data ${DATA} --weights ${WEIGHTS}