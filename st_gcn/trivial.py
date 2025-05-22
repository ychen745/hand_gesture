import os
import json

with open('/scratch/ychen855/hand_gesture/st_gcn/toyTestDataset/json_data/IMG_0086_point-right_4.json') as f:
    frames = json.load(f)
    print(len(frames))
    print(len(frames[0]['keypoints']))