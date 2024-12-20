import torch
import numpy as np
from monai import transforms
import scipy.ndimage

class ToTensor_mt(object):
    def __call__(self, image, target):
        image = image.transpose((3, 1, 2, 0))
        image=torch.from_numpy(image)
        image = image.float() # division by 255.0 is already happening

        # IMAGES:
        #currently: time, widht, height, channel
        # goal: channel, widht, height, time

        # TARGETS:
        # currently: time, width, height, channel
        # goal: channel

        target = target.transpose((3, 1, 2, 0))
        target = torch.from_numpy(target)
        target = target.float()
        
        return image, target#{"image" : image, "target" : target}

class RandomHorizontalFlip_mt(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = image[:, :, ::-1, :].copy()
            target = target[:, :, ::-1, :].copy()
        return image, target

class RandomVerticalFlip_mt(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = image[:, ::-1, :, :].copy()
            target = target[:, ::-1, :, :].copy()
        return image, target

class RandomContrast_mt(object):
    def __init__(self, p=0.5, contrast_range = (0.8,1.2)):
        self.p = p
        self.contrast_range = contrast_range

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            contrast_factor = np.random.uniform(*self.contrast_range)
            mean = np.mean(image, axis=(1, 2), keepdims=True)
            image = (image - mean) * contrast_factor + mean
            image = np.clip(image, 0.0, 1.0)
        return image, target

class RandomBlur_mt(object):
    def __init__(self, p=0.5, sigma_range = (0.5, 1.5)):
        self.p = p
        self.sigma_range = sigma_range
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            blur_factor = np.random.uniform(*self.sigma_range)
            image = scipy.ndimage.gaussian_filter(image, sigma=blur_factor)
        return image, target

class RandomBrightness_mt(object):
    def __init__(self, p=0.5, brightness_range=(0.8, 1.2)):
        self.p = p
        self.brightness_range = brightness_range
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            brightness_factor = np.random.uniform(*self.brightness_range)
            image = image * brightness_factor
            image = np.clip(image, 0.0, 1.0)
        return image, target








