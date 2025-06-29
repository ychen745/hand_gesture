import albumentations as A
import numpy as np
import cv2

def get_transforms(size=416):
    train_transform = A.Compose([
        A.Resize(
            height=size,
            width=size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        A.ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True, filter_invalid_bboxes=True))

    return train_transform

DEFAULT_TRANSFORMS = get_transforms()


