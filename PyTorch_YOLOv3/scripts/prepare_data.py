import os
import shutil
import json
import random

def prepare_image():
    src = '/scratch/ychen855/hand_gesture/data/hagrid/hagrid-sample-30k-384p/hagrid_30k'
    dst = '/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/images'
    gestures = []
    with open('/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/classes.names') as f:
        for line in f:
            gestures.append(line.strip())
    gestures = gestures[:-1]
    for gesture in gestures:
        for f in os.listdir(os.path.join(src, 'train_val_' + gesture)):
            shutil.move(os.path.join(src, 'train_val_' + gesture, f), os.path.join(dst, f))

def prepare_label():
    gestures = []
    with open('/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/gesture.names') as f:
        for line in f:
            gestures.append(line.strip())

    img_src = '/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/images'
    label_dst = '/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/labels'
    anno = '/scratch/ychen855/hand_gesture/data/hagrid/hagrid-sample-30k-384p/ann_train_val'

    image_set = set()
    for gesture in gestures:
        if gesture == 'no_gesture':
            continue
        for f in os.listdir(img_src):
            image_set.add(f[:-4])

    for gesture in gestures:
        if gesture == 'no_gesture':
            continue
        anno_file = os.path.join(anno, gesture + '.json')
        with open(anno_file, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                if k not in image_set:
                    continue
                bboxes = v['bboxes']
                labels = v['labels']
                bboxes_out = []
                labels_out = []
                for i in range(len(bboxes)):
                    x, y, width, height = [float(ele) for ele in bboxes[i]]
                    x_center = x + width / 2.0
                    y_center = y + height / 2.0
                    bboxes_out.append([str(ele) for ele in [x_center, y_center, width, height]])
                    # labels_out.append(str(gestures.index(labels[i])))
                    labels_out.append('0')
                with open(os.path.join(label_dst, k + '.txt'), 'w') as flabel:
                    for i in range(len(bboxes_out)):
                        flabel.write(labels_out[i] + ' ' + ' '.join(bboxes_out[i]) + '\n')
                    # exit()

def train_test_split():
    src = '/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/images'
    images = os.listdir(src)
    random.shuffle(images)
    train_images = []
    test_images = []
    for i in range(len(images) // 5):
        test_images.append(os.path.join(src, images[i]))
    for i in range(len(images) // 5, len(images)):
        train_images.append(os.path.join(src, images[i]))
    
    with open('/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/train.txt', 'w') as f:
        for img in train_images:
            f.write(img + '\n')
    with open('/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/valid.txt', 'w') as f:
        for img in test_images:
            f.write(img + '\n')

if __name__ == '__main__':
    # prepare_image()
    prepare_label()
    # train_test_split()