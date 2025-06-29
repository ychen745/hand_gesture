import albumentations as A
import cv2

def get_transforms(size=416, exposure=1.2):
    aug_list = [
        A.Resize(
            height=size,
            width=size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        A.Sharpen(
            alpha=(0.0, 0.1),
            lightness=(0.8, 1.2)
        ),
        A.Affine(
            scale=(0.8, 1.5),
            translate_percent=(-0.1, 0.1),
            rotate=(-0, 0)
        ),
        A.RandomBrightnessContrast(
            brightness_limit=exposure - 1.0,
            p=0.2
        ),
        A.HueSaturationValue(
            hue_shift_limit=[-10, 10],
            sat_shift_limit=[-0, 0],
            val_shift_limit=[-0, 0]
        ),
        A.HorizontalFlip(p=0.5),
        A.ToTensorV2()
    ]

    return A.Compose(aug_list, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True, filter_invalid_bboxes=True))
