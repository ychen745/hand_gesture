import os
import shutil
import random

src = '/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/images'
dst = '/scratch/ychen855/hand_gesture/PyTorch-YOLOv3/data/custom/test_images'

images = os.listdir(src)
for i in range(10):
    idx = random.randint(0, len(images))
    shutil.copy(os.path.join(src, images[idx]), os.path.join(dst))