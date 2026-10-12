"""VOC-style rice/wheat leaf-tip data with 512px training crops."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import NormalSample


class RiceW3999(Dataset):
    def __init__(self, root_path, mode, label_prob=1, protc_path=''):
        if label_prob != 1:
            raise ValueError("RiceW3999 currently supports fully supervised training; use --label 100.")
        self.training = mode == 'train'
        root = Path(root_path)
        split = 'train' if self.training else 'val'
        split_path = root / 'ImageSets' / (split + '.txt')
        ids = split_path.read_text(encoding='utf-8-sig').split()
        self.samples = []
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        for image_id in ids:
            image = next((root / 'JPEGImages' / (image_id + ext)
                          for ext in extensions
                          if (root / 'JPEGImages' / (image_id + ext)).is_file()), None)
            annotation = root / 'PointAnnotations' / (image_id + '.txt')
            if image is None or not annotation.is_file():
                raise FileNotFoundError("Missing image or annotation for " + image_id)
            self.samples.append((image_id, image, annotation))
        if not self.samples:
            raise ValueError("Empty dataset split: " + str(split_path))
        self.norm_func = NormalSample(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            crop_size=(512, 512), train=self.training)
        print("RiceW3999 {}: {} images | train patch 512x512 | validation full image".format(
            split, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id, image_path, annotation_path = self.samples[index]
        with Image.open(image_path) as image:
            image = self.norm_func.im2tensor(image.convert('RGB'))
        points = []
        for line_number, line in enumerate(annotation_path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) < 2:
                raise ValueError("{}:{}: expected x y".format(annotation_path, line_number))
            points.append([float(values[0]), float(values[1])])
        points = torch.from_numpy(np.asarray(points, dtype=np.float32).reshape(-1, 2))
        # NormalSample transforms x,y pixel coordinates and returns y,x plus
        # the nearest-neighbor statistic expected by the existing loss.
        images, sequences = self.norm_func.process_lable(image, points)
        if self.training:
            # Keep the trainer's six-item interface. Fully supervised training
            # does not consume weak/strong images or cutout masks.
            return images, sequences, image_id, None, None, image_id
        return images, sequences, image_id

    @staticmethod
    def collate_fn(samples):
        if len(samples[0]) == 6:
            images, sequences, ids, _, _, uids = zip(*samples)
            return torch.cat(images), sum(sequences, []), ids, None, None, uids
        images, sequences, ids = zip(*samples)
        return torch.cat(images), sum(sequences, []), ids
