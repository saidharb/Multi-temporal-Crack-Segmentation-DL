# For handling .tif files
from PIL import Image

# For handling the operating system
import os

from glob import glob
from natsort import natsorted
import numpy as np
# PyTorch
from torch.utils.data import Dataset
import torch
# For plotting
import matplotlib.pyplot as plt


class ToTensorOnlyImages(object):
    def __call__(self, image):
        #image = sample["image"]
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image=torch.from_numpy(image)
        image = image.float()/255.0
        
        return image

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


class CrackDataset(Dataset):
    def __init__(self, image_paths, target_paths , transform = ToTensor()):
        
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return(len(self.image_paths))

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        target = Image.open(self.target_paths[idx])
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def get_image_path(self, idx):
        return(self.image_paths[idx])


    def get_target_path(self, idx):
        return(self.target_paths[idx])

    def visualize(self, idx):
        print(self.image_paths[idx])
        print(self.target_paths[idx])
        image = Image.open(self.image_paths[idx])
        target = Image.open(self.target_paths[idx])
        if self.transform:
            image, target = self.transform(image, target)
        image = image.permute(1, 2, 0).numpy()  # Change order to HxWxC
        target = target.squeeze(0).numpy()  # Remove the channel dimension for target
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(target, cmap='gray')
        axes[1].set_title('Target')
        axes[1].axis('off')  # Hide axes
        plt.show()


class ImageDataset(Dataset):
    def __init__(self, root, transform = ToTensorOnlyImages()):
        self.root = root
        images_paths = root
        
        list_images = glob(os.path.join(images_paths, "*.png"), recursive=True)
        list_images = natsorted(list_images, key=lambda y: y.lower())
        
        self.image_paths = list_images # dataframe["path_image"]
        self.transform = transform

    def __len__(self):
        return(len(self.image_paths))

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

    def get_image_path(self, idx):
        return(self.image_paths[idx])

    def visualize(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        
        image = image.permute(1, 2, 0).numpy()  # Change order to HxWxC
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image)
        ax.set_title('Image')
        ax.axis('off')
        plt.show()



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

class MtCrackDataset(Dataset):
    
    def __init__(self, image_paths, target_paths , transform = ToTensor_mt()):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return(len(self.image_paths))

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        target = np.load(self.target_paths[idx])
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def get_image_path(self, idx):
        return(self.image_paths[idx])

    def visualize(self, idx):
        image = np.load(self.image_paths[idx])
        target = np.load(self.target_paths[idx])
        if self.transform:
            image, target = self.transform(image, target)
        image = image.permute(3, 1, 2, 0).numpy()  # Change order to HxWxC
        
        target = target.permute(3, 1, 2, 0).numpy()  # Remove the channel dimension for target

        for slice in range(0, 32): 
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image[slice, :, :, :])
            axes[0].set_title('Image t = {}'.format(slice))
            axes[0].axis('off')
            axes[1].imshow(target[slice, :, :, :], cmap='gray')
            axes[1].set_title('Target t = {}'.format(slice))
            axes[1].axis('off')  # Hide axes
            plt.show()