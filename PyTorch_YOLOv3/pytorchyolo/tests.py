import os
import numpy as np
import cv2
from PIL import Image
from utils.datasets import pad_to_square

image_root = '/scratch/ychen855/Data/hand_yolo/train/images'
label_root = '/scratch/ychen855/Data/hand_yolo/train/labels'
output_root = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/test_output'

counter = 0
for fname in os.listdir(image_root):
    counter += 1
    img_path = os.path.join(image_root, fname)
    label_path = os.path.join(label_root, fname[:-4] + '.txt')
    image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
    boxes = np.loadtxt(label_path).reshape(-1, 5)
    class_labels = boxes[:, 0]
    boxes = boxes[:, 1:]

    h, w, c = image.shape
    if h == w:
        counter -= 1
        continue

    x1s = (boxes[:, 0] - boxes[:, 2] / 2) * w
    x2s = (boxes[:, 0] + boxes[:, 2] / 2) * w
    y1s = (boxes[:, 1] - boxes[:, 3] / 2) * h
    y2s = (boxes[:, 1] + boxes[:, 3] / 2) * h

    cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(x1s.shape[0]):
        cv2.rectangle(cv_image, (int(x1s[i]), int(y1s[i])), (int(x2s[i]), int(y2s[i])), color=(0,255,255), thickness=2)
    
    # cv2.imwrite(os.path.join(output_root, 'raw_' + fname), cv_image)

    # new image
    image, boxes = pad_to_square(image, boxes)
    h, w, c = image.shape

    x1s = (boxes[:, 0] - boxes[:, 2] / 2) * w
    x2s = (boxes[:, 0] + boxes[:, 2] / 2) * w
    y1s = (boxes[:, 1] - boxes[:, 3] / 2) * h
    y2s = (boxes[:, 1] + boxes[:, 3] / 2) * h

    cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(x1s.shape[0]):
        cv2.rectangle(cv_image, (int(x1s[i]), int(y1s[i])), (int(x2s[i]), int(y2s[i])), color=(0,255,255), thickness=2)
    
    # cv2.imwrite(os.path.join(output_root, 'new_' + fname), cv_image)


    if counter > 10:
        break

    # print(out_img.shape)
    # h, w, c = out_img.shape
    # if h != w:
    #     print(fname)

