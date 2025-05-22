PYTHON=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/train.py
MODEL=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.cfg
DATA=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.data
PRETRAINED_WEIGHTS=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/weights/cross-hands.weights
EPOCHS=60

python ${PYTHON} --model ${MODEL} --epochs ${EPOCHS} --data ${DATA} --pretrained_weights ${PRETRAINED_WEIGHTS}