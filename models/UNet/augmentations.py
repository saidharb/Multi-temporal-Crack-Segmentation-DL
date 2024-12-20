# PyTorch
import torch

# Numpy
import numpy as np

# Torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
from scipy.ndimage import gaussian_filter, map_coordinates
class ShearTransform:
    def __init__(self, p, shear_range):
        self.p = p
        self.shear_range = shear_range

    def __call__(self, img, target):
        if np.random.rand() < self.p:
            shear_factor = np.random.uniform(*self.shear_range)
            img = F.affine(img, angle=0, translate=(0, 0), scale=1, shear=(shear_factor, 0))
            target = F.affine(target, angle=0, translate=(0, 0), scale=1, shear=(shear_factor, 0))

        return img, target

class ElasticTransform:
    def __init__(self, p, alpha, sigma):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img, target):
        if np.random.rand() < self.p:
            random_state = np.random.RandomState(None)

            shape = img.shape
            dx = gaussian_filter((random_state.rand(*shape[1:]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((random_state.rand(*shape[1:]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            transformed_img = np.zeros_like(img)
            for i in range(shape[0]):
                transformed_img[i] = map_coordinates(img[i], indices, order=1, mode='reflect').reshape(shape[1:])
            target_squeezed = target.squeeze(0)
            target_shape = target_squeezed.shape
            dx = gaussian_filter((random_state.rand(*target_shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((random_state.rand(*target_shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

            x, y = np.meshgrid(np.arange(target_shape[1]), np.arange(target_shape[0]), indexing='ij')
            target_indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            transformed_target = map_coordinates(target_squeezed, target_indices, order=1, mode='reflect').reshape(target_shape)
            transformed_target = transformed_target[np.newaxis, ...]

            return torch.from_numpy(transformed_img), torch.from_numpy(transformed_target)
        return img, target
class ToTensor(object):
    def __call__(self, image, target):
        #image = sample["image"]
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image=torch.from_numpy(image)
        image = image.float()/255.0
        

        #target = sample["target"]
        target = np.array(target)
        target = torch.from_numpy(target)
        target = target.unsqueeze(0)
        target = target.float()
        
        return image, target#{"image" : image, "target" : target}
        
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

class RandomColorJitter(object):
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p
    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = self.color_jitter(image)
        return image, target

class RandomRotation(object):
    def __init__(self, p = 0.5):
        self.angles = [0, 90, 180, 270]
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            degrees = np.random.choice(self.angles)
            image = transforms.functional.rotate(image, degrees)
            target = transforms.functional.rotate(target, degrees)
        return image, target

class AddGaussianNoise(object):
    def __init__(self, p = 0.5, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = image + torch.randn(image.size()) * self.std + self.mean
            image = torch.clamp(image, 0.0, 1.0)
        return image, target

class RandomBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if np.random.rand() < self.p:
            image = F.gaussian_blur(image, kernel_size=5, sigma=(10.0, 10.0))
        return image, target

class ToTensorOnlyImages(object):
    def __call__(self, image):
        #image = sample["image"]
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image=torch.from_numpy(image)
        image = image.float()/255.0
        
        return image