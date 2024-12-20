# PyTorch
import torch

# Numpy
import numpy as np

# Torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
from scipy.ndimage import gaussian_filter, map_coordinates

class ToTensor(object):
    def __call__(self, image, target):
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image=torch.from_numpy(image)
        image = image.float()/255.0

        target = np.array(target)
        target = torch.from_numpy(target)
        target = target.unsqueeze(0)
        target = target.float()
        
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = transforms.functional.hflip(image)
            target = transforms.functional.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = transforms.functional.vflip(image)
            target = transforms.functional.vflip(target)
        return image, target


class RandomContrast(object):
    def __init__(self, p=0.5, brightness=0, contrast=(0.8,1.2), saturation=0, hue=0):
        self.contrast_range = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.hue = hue
        self.p = p
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            color_jitter = transforms.ColorJitter(
                brightness=self.brightness, 
                contrast=self.contrast_range, 
                saturation=self.saturation, 
                hue=self.hue)
            image = color_jitter(image)
        return image, target

class RandomBrightness(object):
    def __init__(self, p=0.5, brightness=(0.8, 1.2), contrast=0, saturation=0, hue=0):
        self.brightness_range = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            color_jitter = transforms.ColorJitter(
                brightness=self.brightness_range,
                contrast=self.contrast, 
                saturation=self.saturation, 
                hue=self.hue)
            image = color_jitter(image)
        return image, target

class RandomBlur(object):
    def __init__(self, p=0.5, sigma_range = (0.5, 1.5)):
        self.p = p
        self.sigma_range = sigma_range
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            blur_factor = np.random.uniform(*self.sigma_range)
            image = F.gaussian_blur(image, kernel_size=5, sigma=(blur_factor, blur_factor))
        return image, target