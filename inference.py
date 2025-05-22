import argparse
import os
import numpy as np
import time
import torch
from PyTorch_YOLOv3 import detect
from PyTorch_YOLOv3.pytorchyolo.models import load_model as load_yolo_model
from PIL import Image
from PyTorch_YOLOv3.pytorchyolo.utils.utils import rescale_boxes
from ultralytics import YOLO


def yolo_inference_one(yolo_model, image, img_size, yolo_conf_thres, yolo_nms_thres):

    detection = detect.inference_one(yolo_model, image, img_size, yolo_conf_thres, yolo_nms_thres)

    if len(detection) == 0:
        return None
    
    x1, y1, x2, y2 = detection[0][:4]
    # x1 -= 5
    # y1 -= 5
    # x2 += 5
    # y2 += 5
    # x1, y1, x2, y2 = [int(ele) for ele in detection[0][:4]]

    cropped_image = Image.fromarray(image)
    cropped_image = cropped_image.crop((x1, y1, x2, y2))
    output_image = np.array(cropped_image)

    return output_image


def yolo_inference_test(args):
    yolo_model = load_yolo_model(args.yolo_model, args.yolo_weights)    

    temp_output = '/scratch/ychen855/hand_gesture/images'
    cropped_output = '/scratch/ychen855/hand_gesture/cropped_images'

    # image_name = '00005c9c-3548-4a8f-9d0b-2dd4aff37fc9.jpg'
    image_names = os.listdir(args.images)
    for i in range(10):
        # print(f'Start detecting {i}th image.')
        start = time.time()
        
        image_name = image_names[i]
        print(image_name)
        image = np.array(Image.open(os.path.join(args.images, image_name)).convert('RGB'), dtype=np.uint8)
        detection = detect.inference_one(yolo_model, image, args.img_size, args.yolo_conf_thres, args.yolo_nms_thres)

        if len(detection) == 0:
            continue
        
        x1, y1, x2, y2 = detection[0][:4]

        output_image = image[x1:x2, y1:y2, :]
        output = Image.fromarray(output_image)

        # cropped = output.crop((x1, y1, x2, y2))
        output.save(os.path.join(temp_output, image_name))
        # cropped.save(os.path.join(cropped_output, image_name))

        duration = time.time() - start
        print(f'Time taken: {duration} seconds.')


def pose_estimation_one(args):
    yolo_model = load_yolo_model(args.yolo_model, args.yolo_weights)  
    pose_model = YOLO(args.pose_weights)

    images = os.listdir(args.images)
    output_root = '/scratch/ychen855/hand_gesture/images'
    cropped_root = '/scratch/ychen855/hand_gesture/cropped_images'

    for i in range(len(images)):
        image = np.array(Image.open(os.path.join(args.images, images[i])).convert('RGB'), dtype=np.uint8)
        cropped_image = yolo_inference_one(yolo_model, image, args.img_size, args.yolo_conf_thres, args.yolo_nms_thres)
        if cropped_image is None:
            continue
        # cropped_output = Image.fromarray(cropped_image)
        # cropped_output.save(os.path.join(cropped_root, images[i]))

        # print('image shape:', cropped_image.shape)
        pose_result = pose_model(cropped_image)[0]
        # print(len(pose_result.boxes.conf))

        if len(pose_result.boxes.conf) == 0:
            continue
        idx = torch.argmax(pose_result.boxes.conf)

        if pose_result.keypoints.conf is not None:
            conf = pose_result.keypoints.conf[idx]
            xyn = pose_result.keypoints.xyn[idx]

            conf_arry = conf.cpu().numpy()
            xyn_arry = xyn.cpu().numpy()
            print('conf', conf)
            print('x', xyn_arry[:, 0])
            print('y', xyn_arry[:, 1])
        
        x_arry = xyn_arry[:, 0]
        y_arry = xyn_arry[:, 1]

        return x_arry, y_arry, conf_arry


def pose_estimation_test(args):
    yolo_model = load_yolo_model(args.yolo_model, args.yolo_weights)  
    pose_model = YOLO(args.pose_weights)

    images = os.listdir(args.images)
    output_root = '/scratch/ychen855/hand_gesture/images'
    cropped_root = '/scratch/ychen855/hand_gesture/cropped_images'

    for i in range(len(images)):
        image = np.array(Image.open(os.path.join(args.images, images[i])).convert('RGB'), dtype=np.uint8)
        cropped_image = yolo_inference_one(yolo_model, image, args.img_size, args.yolo_conf_thres, args.yolo_nms_thres)
        if cropped_image is None:
            continue
        # cropped_output = Image.fromarray(cropped_image)
        # cropped_output.save(os.path.join(cropped_root, images[i]))

        # print('image shape:', cropped_image.shape)
        pose_result = pose_model(cropped_image)[0]
        # print(len(pose_result.boxes.conf))

        if len(pose_result.boxes.conf) == 0:
            continue
        idx = torch.argmax(pose_result.boxes.conf)

        if pose_result.keypoints.conf is not None:
            conf = pose_result.keypoints.conf[idx]
            xyn = pose_result.keypoints.xyn[idx]

            conf_arry = conf.cpu().numpy()
            xyn_arry = xyn.cpu().numpy()
            print('conf', conf_arry)
            print('x', xyn_arry[:, 0])
            print('y', xyn_arry[:, 1])
            # exit()
                # np.savetxt(os.path.join(output, image[:-4] + '.txt'), xyn_arry, fmt="%4f")

        # boxes = pose_result.boxes  # Boxes object for bounding box outputs
        # masks = pose_result.masks  # Masks object for segmentation masks outputs
        # keypoints = pose_result.keypoints  # Keypoints object for pose outputs
        # probs = pose_result.probs  # Probs object for classification outputs
        # obb = pose_result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        # pose_result.save(filename=os.path.join(output_root, images[i]))  # save to disk
        # exit()

def inference(frames, model, device, args):

    model.eval() # this will turn off dropout

    for keypoints, targets, file_paths in data_loader:
        targets = targets.to(device)
        results = model(keypoints.to(device))
        confusion_meter.add(results.detach(), targets.detach())
    return confusion_meter, cal_top1(confusion_meter, args.num_actions), failed_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("--yolo_model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("--yolo_weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("--yolo_classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("--images", type=str, default="data/samples", help="Path to directory with images to inference")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--yolo_conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--yolo_nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")

    parser.add_argument("--pose_weights", type=str)

    args = parser.parse_args()

    pose_estimation_test(args)
    

