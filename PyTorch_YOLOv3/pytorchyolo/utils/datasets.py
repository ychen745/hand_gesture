from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_to_square(image, boxes, value=0):
    h, w, c = image.shape

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * w
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * w
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * h
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * h

    pad = [0] * 4

    if h == w:
        return image, boxes
    elif h > w:
        pad[0] = (h - w) // 2
        pad[1] = (h - w) // 2 if (h - w) % 2 == 0 else (h - w) // 2 + 1
    else:
        pad[2] = (w - h) // 2
        pad[3] = (w - h) // 2 if (w - h) % 2 == 0 else (w - h) // 2 + 1
    
    padded_w = w + pad[0] + pad[1]
    padded_h = h + pad[2] + pad[3]

    assert padded_w == h or padded_h == w

    if pad[0] > 0:
        left_padding = np.zeros((h, pad[0], 3), dtype=np.uint8)
        image = np.concatenate((left_padding, image), axis=1)

    if pad[1] > 0:
        right_padding = np.zeros((h, pad[1], 3), dtype=np.uint8)
        image = np.concatenate((image, right_padding), axis=1)

    if pad[2] > 0:
        top_padding = np.zeros((pad[2], w, 3), dtype=np.uint8)
        image = np.concatenate((top_padding, image), axis=0)

    if pad[3] > 0:
        bottom_padding = np.zeros((pad[3], w, 3), dtype=np.uint8)
        image = np.concatenate((image, bottom_padding), axis=0)

    x1 += pad[0]
    x2 += pad[0]
    y1 += pad[2]
    y2 += pad[2]

    # new x center, new y center, new width, new height
    boxes[:, 0] = ((x1 + x2) / 2) / padded_w
    boxes[:, 1] = ((y1 + y2) / 2) / padded_h
    boxes[:, 2] = (x2 - x1) / padded_w
    boxes[:, 3] = (y2 - y1) / padded_h

    return image, boxes


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image and label
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------

        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
                class_labels = boxes[:, 0]
                boxes = boxes[:, 1:]
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        
        try:
            img, boxes = pad_to_square(img, boxes)
        except Exception:
            print(f"Could not pad to square.")
            return

        if self.transform:
            try:
                augmented = self.transform(image=img, bboxes=boxes, class_labels=class_labels)
                img = augmented['image'].float()
                bb_targets = torch.cat((torch.from_numpy(np.array(augmented['class_labels'])).unsqueeze(-1), torch.from_numpy(augmented['bboxes'])), dim=1)
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))


        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        imgs = torch.stack(imgs)

        # Add sample index to targets
        idx_list = []
        for i, boxes in enumerate(bb_targets):
            idx_list += [i] * boxes.shape[0]
        
        bb_targets = torch.cat(bb_targets, 0)
        bb_targets = torch.cat((torch.tensor(idx_list).unsqueeze(0).reshape((-1, 1)), bb_targets), dim=1)


        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
