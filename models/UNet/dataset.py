# For handling .tif files
from PIL import Image

# For handling the operating system
import os

from glob import glob
from natsort import natsorted

# PyTorch
from torch.utils.data import Dataset

# For plotting
import matplotlib.pyplot as plt
from augmentations import ToTensor, ToTensorOnlyImages

class EarlyStopping:
    def __init__(self, patience = 30, delta = 0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

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

    def visualize(self, idx):
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