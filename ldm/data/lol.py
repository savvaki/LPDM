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

class LOL(Dataset):
    def __init__(self, dir_data, split, dataset_name='lol', crop_size=256, **kwargs):
        
        assert split in ['our485', 'eval15'], f"Unknown split: {split}"

        current_dir = os.path.join(dir_data, dataset_name, split)
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

class LOLTrain(LOL):
    def __init__(self, dir_data, **kwargs):
        super().__init__(dir_data, split="our485", **kwargs)

class LOLValidation(LOL):
    def __init__(self, dir_data, **kwargs):
        super().__init__(dir_data, split="eval15", **kwargs)

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
    
##################################  HE LOL
    
class LOLTrainHE(LOLTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                RandomCrop(self.crop_size),
                RandomHorizontalFlip(),
                ImageToHELDMTensor(),
            ]
        )

class LOLValidationHE(LOLValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                CenterCrop(size=self.crop_size),
                ImageToHELDMTensor(), # No random crop or flipping
            ]
        )

class LOLTrainHE_V(LOLTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                RandomCrop(self.crop_size),
                RandomHorizontalFlip(),
                ImageToHE_V_LDMTensor(),
            ]
        )

class LOLValidationHE_V(LOLValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                CenterCrop(size=self.crop_size),
                ImageToHE_V_LDMTensor(), # No random crop or flipping
            ]
        )

class ImageToHELDMTensor:
    """Convert PIL samples of (img, target) where img is converted to HE RGB, target remains the same"""

    def __init__(self):
        self.transform = tf.ToTensor()

    def __call__(self, sample):
        image, target = sample
        assert isinstance(image, Image.Image)
        rgb = np.asarray(image)
        r_equalized = cv2.equalizeHist(rgb[..., 0])
        g_equalized = cv2.equalizeHist(rgb[..., 1])
        b_equalized = cv2.equalizeHist(rgb[..., 2])
        rgb = np.stack([r_equalized, g_equalized, b_equalized], axis=-1)
        image = Image.fromarray(rgb)
        image = self.transform(image) * 2.0 - 1.0 # LDM scaling
        target = self.transform(target) * 2.0 - 1.0
        return image, target
    
class ImageToHE_V_LDMTensor:
    """Convert PIL samples of (img, target) where img is converted to HE RGB, target remains the same"""

    def __init__(self):
        self.transform = tf.ToTensor()

    def __call__(self, sample):
        image, target = sample
        assert isinstance(image, Image.Image)
        rgb = np.asarray(image)
        r_equalized = cv2.equalizeHist(rgb[..., 0])
        g_equalized = cv2.equalizeHist(rgb[..., 1])
        b_equalized = cv2.equalizeHist(rgb[..., 2])
        value_channel = np.asarray(image.convert('HSV'))[..., -1] # value channel
        rgb = np.stack([r_equalized, g_equalized, b_equalized, value_channel], axis=-1)

        image = Image.fromarray(rgb)
        image = self.transform(image) * 2.0 - 1.0 # LDM scaling
        target = self.transform(target) * 2.0 - 1.0
        return image, target
    
################################## ImageNet scaling 
    
class LOLTrainImageNet(LOLTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                RandomCrop(self.crop_size),
                RandomHorizontalFlip(),
                XToImageNetYToLDMTensor(),
            ]
        )

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]), Image.open(self.target_paths[idx])
        return self.transforms((x, t))

class LOLValidationImageNet(LOLValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                CenterCrop(size=self.crop_size),
                XToImageNetYToLDMTensor(), # No random crop or flipping
            ]
        )

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]), Image.open(self.target_paths[idx])
        return self.transforms((x, t))
    
class XToImageNetYToLDMTensor:
    """Convert PIL samples of (img, target) to their tensor equivalents"""

    def __init__(self):
        self.transform = tf.ToTensor()
        self.normalize = tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, sample):
        image, target = sample
        ret = {'x_ddpm' : self.transform(image) * 2.0 - 1.0}
        image = self.normalize(self.transform(image)) # image has 0 to 1 scaling normalised for imagenet
        ret['x'] = image
        target = self.transform(target) * 2.0 - 1.0
        ret['t'] = target 
        return ret
    
################################## HSV 

class LOLTrainHSVInputOnly(LOLTrain):
    """The low light image only is in HSV"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]).convert('HSV'), Image.open(self.target_paths[idx])
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }

class LOLValidationHSVInputOnly(LOLValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]).convert('HSV'), Image.open(self.target_paths[idx])
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }
        

class LOLTrainHSV(LOLTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]).convert('HSV'), Image.open(self.target_paths[idx]).convert('HSV')
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }

class LOLValidationHSV(LOLValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]).convert('HSV'), Image.open(self.target_paths[idx]).convert('HSV')
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }
    
################################## YCbCr

class LOLTrainYCbCr(LOLTrain):
    """"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]).convert('YCbCr'), Image.open(self.target_paths[idx]).convert('YCbCr')
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }

class LOLValidationYCbCr(LOLValidation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, t = Image.open(self.image_paths[idx]).convert('YCbCr'), Image.open(self.target_paths[idx]).convert('YCbCr')
        x, t = self.transforms((x, t))
        return {
            'x' : x, # image
            't' : t, # target
        }


################################## Noise dataset 

class LOLTrainNoise(LOLTrain):
    "Note only denoising the targets"
    def __init__(self, *args, noise_stds, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = tf.Compose(
            [
                RandomCrop(self.crop_size),
                RandomHorizontalFlip(),
                ImageToLDMNoisyTensor(noise_stds),
            ]
        )

class LOLValidationNoise(LOLValidation):
    "Note only denoising the targets"
    def __init__(self, *args, noise_stds, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = tf.Compose(
            [
                CenterCrop(size=self.crop_size),
                ImageToLDMNoisyTensor(noise_stds), # No random crop or flipping
            ]
        )

class ImageToLDMNoisyTensor:
    """Convert PIL samples of (img, target) to their tensor equivalents with img adding noise"""

    def __init__(self, noise_stds):
        self.transform = tf.ToTensor()
        self.noise_stds = noise_stds

    def __call__(self, sample):
        _, target = sample
        target = self.transform(target)
        x = target + torch.normal(0, random.choice(self.noise_stds), size=target.shape)
        x = torch.clamp(x, 0, 1) * 2.0 - 1.0 
        target = target * 2.0 - 1.0
        return x, target