PYTHON=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/train.py
MODEL=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.cfg
DATA=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.data
PRETRAINED_WEIGHTS=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/weights/cross-hands.weights
CHECKPOINTS_DIR=/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/checkpoints/crosshands_newaug
CHECKPOINT_INTERVAL=5
EVALUATION_INTERVAL=1
EARLYSTOPPING=40
IOU_THRES=0.5
CONF_THRES=0.3
NMS_THRES=0.4
EPOCHS=120

python ${PYTHON} \
    --model ${MODEL} \
    --epochs ${EPOCHS} \
    --data ${DATA} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL}\
    --evaluation_interval ${EVALUATION_INTERVAL} \
    --earlystopping ${EARLYSTOPPING} \
    --iou_thres ${IOU_THRES} \
    --conf_thres ${CONF_THRES} \
    --nms_thres ${NMS_THRES} \
    --verbose