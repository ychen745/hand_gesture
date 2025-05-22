import os
import torch
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO("/scratch/ychen855/hand_gesture/yolo11/runs/pose/train/weights/best.pt")  # pretrained YOLO11n model

val_root = '/scratch/ychen855/Data/hagrid/images'
# val_root = '/scratch/ychen855/Data/hand/val/images'
images = os.listdir(val_root)
# images = ['7f4e52dd-2d94-4746-859c-4aa37a9bc901.jpg']

output = '/scratch/ychen855/Data/hagrid/labels_pose'

for image in images:
    print(image)
    result = model([os.path.join(val_root, image)])[0]
    if len(result.boxes.conf) == 0:
        continue
    idx = torch.argmax(result.boxes.conf)

    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[idx]
        xyn = result.keypoints.xyn[idx]
        if torch.min(conf) > 0.9:
            xyn_arry = xyn.cpu().numpy()
            np.savetxt(os.path.join(output, image[:-4] + '.txt'), xyn_arry, fmt="%4f")

# Run batched inference on a list of images
# results = model([os.path.join(val_root, image) for image in images])  # return a list of Results objects

# Process results list
# for i in range(len(images)):
#     boxes = results[i].boxes  # Boxes object for bounding box outputs
#     masks = results[i].masks  # Masks object for segmentation masks outputs
#     keypoints = results[i].keypoints  # Keypoints object for pose outputs
#     probs = results[i].probs  # Probs object for classification outputs
#     obb = results[i].obb  # Oriented boxes object for OBB outputs
#     # result.show()  # display to screen
#     results[i].save(filename=images[i])  # save to disk
#     print(keypoints)