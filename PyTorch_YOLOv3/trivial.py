import os

src = '/scratch/ychen855/Data/hand_yolo'

with open(os.path.join(src, 'test.txt'), 'w') as f:
    for fname in sorted(os.listdir(os.path.join(src, 'test', 'images'))):
        f.write(os.path.join(src, 'test', 'images', fname) + '\n')