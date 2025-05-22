PYTHON=/scratch/ychen855/hand_gesture/inference.py
YOLO_MODEL=/scratch/ychen855/hand_gesture/config/yolo.cfg
YOLO_WEIGHTS=/scratch/ychen855/hand_gesture/weights/yolo.weights
YOLO_CLASSES=/scratch/ychen855/hand_gesture/yolo.names
OUTPUT=/scratch/ychen855/hand_gesture/results/yolo
IMAGES=/scratch/ychen855/Data/hagrid/test_images
POSE_WEIGHTS=/scratch/ychen855/hand_gesture/weights/pose.pt


python ${PYTHON} \
    --yolo_model ${YOLO_MODEL} \
    --yolo_weights ${YOLO_WEIGHTS} \
    --yolo_classes ${YOLO_CLASSES} \
    --output ${OUTPUT} \
    --images ${IMAGES} \
    --pose_weights ${POSE_WEIGHTS} \
    --test_only