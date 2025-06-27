import albumentations as A
import cv2

def get_transforms(size=416):
    sizing_list = [
        A.LongestMaxSize(
            max_size=size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            position="center",
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0
        )
    ]

    aug_list = [
        A.Sharpen(
            alpha=(0.0, 0.1),
            lightness=(0.8, 1.2)
        ),
        A.Affine(
            scale=(0.8, 1.5),
            translate_percent=(-0.1, 0.1),
            rotate=(-0, 0)
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=[-10, 10],
            sat_shift_limit=[-0, 0],
            val_shift_limit=[-0, 0]
        ),
        A.HorizontalFlip(p=0.5)
    ]

    tensor_list = [A.ToTensorV2()]

    transform_list = sizing_list + aug_list + tensor_list

    return A.Compose(transform_list, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True, filter_invalid_bboxes=True))
