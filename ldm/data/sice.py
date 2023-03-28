import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
import glob
from pathlib import Path
import cv2

class SICE(Dataset):
    def __init__(self, dir_data, split, dataset_name='SICE', part='part1', crop_size=256, **kwargs):
        
        assert split in ['train', 'val'], f"Unknown split: {split}"

        current_dir = os.path.join(dir_data, dataset_name, part, split)
        self.image_paths = glob.glob(os.path.join(current_dir, 'low', '*.png'))
        self.target_paths = glob.glob(os.path.join(current_dir, 'high', '*.png'))
        self.crop_size = crop_size

        self.transforms = tf.Compose(
            [
                RandomCrop(crop_size),
                RandomHorizontalFlip(),
                ImageToLDMTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]), Image.open(self.target_paths[idx])
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }

class SICETrain(SICE):
    def __init__(self, dir_data, **kwargs):
        super().__init__(dir_data, split="train", **kwargs)

class SICEValidation(SICE):
    def __init__(self, dir_data, **kwargs):
        super().__init__(dir_data, split="val", **kwargs)

        self.transforms = tf.Compose(
            [
                CenterCrop(size=self.crop_size),
                ImageToLDMTensor(), # No random crop or flipping
            ]
        )

class CenterCrop:
    def __init__(self, **kwargs):
        self.transform = tf.CenterCrop(**kwargs)

    def __call__(self, sample):
        image, target = sample
        return self.transform(image), self.transform(target)
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, target = sample
            image = F.hflip(image) if isinstance(image, torch.Tensor) else image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            target = F.hflip(target) if isinstance(target, torch.Tensor) else target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return image, target
        else:
            return sample
        
class RandomCrop:
    """Transform (img, target) by randomly cropping. Compatable with both tensor and PIL image"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample

        if torch.is_tensor(image):
            h, w = image.shape[:2]
        else:
            w, h = image.size
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h, (1,)).item()
        left = torch.randint(0, w - new_w, (1,)).item()

        return F.crop(image, top, left, new_h, new_w), F.crop(
            target, top, left, new_h, new_w
        )

class ImageToLDMTensor:
    """Convert PIL samples of (img, target) to their tensor equivalents"""

    def __init__(self):
        self.transform = tf.ToTensor()

    def __call__(self, sample):
        image, target = sample
        image = self.transform(image) * 2.0 - 1.0 # LDM scaling
        target = self.transform(target) * 2.0 - 1.0
        return image, target