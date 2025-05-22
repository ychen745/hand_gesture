import os
import json
import cv2
import numpy as np
from collections import OrderedDict

# info, licenses, images, annotations, categories

# "info": {
#         "description": "COCO 2017 Dataset",
#         "url": "http://cocodataset.org",
#         "version": "1.0",
#         "year": 2017,
#         "contributor": "COCO Consortium",
#         "date_created": "2017/09/01"
#     }

# images (list): license, file_name, coco_url, height, width, date_captured, flickr_url, id
# annotations (list): segmentation, num_keypoints, area, iscrowd, keypoints, image_id, bbox, category_id, id


src_images = '/scratch/ychen855/Data/hand/train/images'
src_labels = '/scratch/ychen855/Data/hand/train/labels'
dst = '/scratch/ychen855/Data/hand/json/hand_train.json'

json_template = 'person_keypoints_val2017.json'

with open(json_template) as f:
    obj = json.load(f)
    new_obj = OrderedDict()
    new_obj["info"] = obj["info"]
    del new_obj["info"]["url"]
    del new_obj["info"]["contributor"]
    new_obj["info"]["description"] = "hand keypoint dataset"
    new_obj["info"]["year"] = 2024
    new_obj["info"]["date_created"] = "2024/12/05"

    new_obj["licenses"] = obj["licenses"]

    new_obj["images"] = []
    new_obj["annotations"] = []
    # new_obj["categories"] = {}

    image_id = 0
    anno_id = 0
    for fname in os.listdir(src_images):
        image = cv2.imread(os.path.join(src_images, fname))

        image_obj = OrderedDict()
        image_obj["license"] = 4
        image_obj["file_name"] = fname
        image_obj["height"] = image.shape[0]
        image_obj["width"] = image.shape[1]
        image_obj["id"] = image_id

        new_obj["images"].append(image_obj)

        anno_obj = OrderedDict()
        with open(os.path.join(src_labels, fname[:-4] + '.txt')) as flabel:
            for line in flabel:
                linelist = line.strip().split()
                category = linelist[0]
                x = float(linelist[1])
                y = float(linelist[2])
                w = float(linelist[3])
                h = float(linelist[4])
                keypoints = linelist[5:]

                anno_obj["num_keypoints"] = 21
                anno_obj["iscrowd"] = 0
                anno_obj["keypoints"] = [float(ele) for ele in keypoints]

                for idx in range(len(keypoints)):
                    if idx % 3 == 2:
                        anno_obj["keypoints"][idx] = int(anno_obj["keypoints"][idx])

                anno_obj["image_id"] = image_id
                anno_obj["bbox"] = [x, y, w, h]
                anno_obj["category_id"] = 0
                anno_obj["id"] = anno_id

                new_obj["annotations"].append(anno_obj)

                anno_id += 1

        image_id += 1

    cat_obj = OrderedDict()
    cat_obj["supercategory"] = "hand"
    cat_obj["id"] = 0
    cat_obj["name"] = "hand"
    cat_obj["keypoints"] = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                            "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
                            "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
                            "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
                            "pinky_finger_mcp", "pinky_finger_pip", "pinky_finger_dip", "pinky_finger_tip"]

    cat_obj["skeleton"] = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                            [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


    new_obj["categories"] = [cat_obj]

    with open(dst, "w") as fout:
        json.dump(new_obj, fout)

