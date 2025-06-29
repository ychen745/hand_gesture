PYTHON=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/test.py
MODEL=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.cfg
DATA=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.data
WEIGHTS=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/checkpoints/crosshands_newaug/best.pth
# WEIGHTS=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/weights/cross-hands.weights

IOU_THRES=0.5
CONF_THRES=0.2
NMS_THRES=0.4

python ${PYTHON} --model ${MODEL} --data ${DATA} --weights ${WEIGHTS} --iou_thres ${IOU_THRES} --conf_thres ${CONF_THRES} --nms_thres ${NMS_THRES}
