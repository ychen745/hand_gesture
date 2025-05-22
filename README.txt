Data used for training YOLO v3 consists of a mixture of two datasets:
    1. kaggle hand detection dataset: https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format/data.
        - 1,551 training
        - 510 testing
    2. Roboflow Universe dataset: https://universe.roboflow.com/handrecognizer/hand-detection-2r6df/dataset/8.
        - 1,908 training
        - 182 validation
        - 94 testing
        - preprocessed to 416 x 416

    - That makes a total of 3459 training images and 786 testing images.

Data used for training Litepose are from Yolo11 hand keypoints dataset: https://docs.ultralytics.com/datasets/pose/hand-keypoints/.
    - Total: 26,768 images
    - Training: 18,776 images
    - Validataion: 7,992 images
    - image size: 224 x 224