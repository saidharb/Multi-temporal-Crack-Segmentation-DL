# For handling .tif files
from PIL import Image
import tifffile as tiff

# For handling the operating system
import os

from glob import glob
from natsort import natsorted
# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# To track training progress
import time

# Torchvision
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# For handling arrays
import numpy as np
import pandas as pd
import math

from dataset import *
from unet import UNet


# For plotting
import matplotlib.pyplot as plt


import wandb

# For handling the operating system
import os

# PyTorch
import torch

def save_checkpoint(model, optimizer, epoch, logs, sigmoid_threshold):
    trained_models_path = "Trained_Models"
    if not os.path.exists(trained_models_path):
        os.makedirs(trained_models_path)
    version = 0
    base_model_name = f"model_nrepochs_{epoch}_sig_{str(sigmoid_threshold).replace('.', '-')}"
    model_name = base_model_name

    while os.path.exists(os.path.join(trained_models_path, model_name + ".pth")):
        version += 1
        model_name = f"{base_model_name}_v{version}"
   
    path = os.path.join(trained_models_path, model_name + ".pth")
    print("Saving the best model to {}".format(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def save_logs(logs, epoch, sigmoid_threshold):
    trained_models_path = "Trained_Models"
    base_model_name = f"model_nrepochs_{epoch}_sig_{str(sigmoid_threshold).replace('.', '-')}"
    model_name = base_model_name
    version = 0
    while os.path.exists(os.path.join(trained_models_path, model_name) + '_training_logs.csv'):
        version += 1
        model_name = f"{base_model_name}_v{version}"
    path = os.path.join(trained_models_path, model_name) + '_training_logs.csv'
    print("Saving the training logs to {}".format(path))
    logs.to_csv(path, index=False)

def split_image(image_path, DATA_PATH, target_size, custom_heightXwidth=None):
    target_width=target_size[0]
    target_height=target_size[1]
    
    image=Image.open(image_path)
    image_width, image_height = image.size
    image_name=image_path.split("/")[-1]
    epoch_name = image_name.split(".")[0]
    epoch_name = epoch_name.split("-")[-1]
    print("The image {} is {} pixels wide and {} pixels high.".format(image_name, image_width, image_height))

    image_crop_width = (image_width//target_width)*target_width
    image_crop_height = (image_height//target_height)*target_height #2560#
    if custom_heightXwidth is not None:
        image_crop_width = custom_heightXwidth[1]
        image_crop_height = custom_heightXwidth[0]
    print("Cropped Image width: " + str(image_crop_width))
    print("Cropped Image height: " + str(image_crop_height))

    images_paths = os.path.join(DATA_PATH, "images")
    if not images_paths:
        os.makedirs(images_paths)
    tile_image_dir=os.path.join(images_paths, image_name.split(".")[0])
    if not os.path.exists(tile_image_dir):
        os.makedirs(tile_image_dir)

    counter = 0
    counter_file_exist = 0
    for i in range (0, image_crop_width, target_width):
        for j in range (0, image_crop_height, target_height):
            box = (i, j, i+target_width, j+target_height)
            tile = image.crop(box)
            tile_filename = os.path.join(tile_image_dir, f'image_{epoch_name}_{i}_{j}.png')
            if not os.path.exists(tile_filename):
                tile.save(tile_filename)
                counter += 1
            else:
                counter_file_exist += 1
                
    print(f"Created {counter} images ({counter_file_exist} already exist) from {image_name}")
    print(f"Saved to: {tile_image_dir}")

def split_target(target_path, DATA_PATH, target_size, custom_heightXwidth=None):
    target_width=target_size[0]
    target_height=target_size[1]
    
    target=Image.open(target_path)
    image_width, image_height = target.size

    target_name = target_path.split("/")[-1]
    epoch_name = target_name.split(".")[0]
    epoch_name = epoch_name.split("-")[-1]
    epoch_name = epoch_name.split("_")[0]
    print("The image {} is {} pixels wide and {} pixels high.".format(target_name, image_width, image_height))

    image_crop_width = (image_width//target_width)*target_width
    image_crop_height = (image_height//target_height)*target_height
    if custom_heightXwidth is not None:
        image_crop_width = custom_heightXwidth[1]
        image_crop_height = custom_heightXwidth[0]
    print("Cropped Target width: " + str(image_crop_width))
    print("Cropped Target height: " + str(image_crop_height))

    target_paths = os.path.join(DATA_PATH, "targets")
    if not target_paths:
        os.makedirs(target_paths)
    tile_target_dir = os.path.join(target_paths, target_name.split(".")[0])
    if not os.path.exists(tile_target_dir):
        os.makedirs(tile_target_dir)
    
    counter = 0
    counter_file_exist = 0
    for i in range (0, image_crop_width, target_width):
        for j in range (0, image_crop_height, target_height):
            box = (i, j, i+target_width, j+target_height)
            tile = target.crop(box)
            tile_filename = os.path.join(tile_target_dir, f'target_{epoch_name}_{i}_{j}.png')
            if not os.path.exists(tile_filename):
                tile.save(tile_filename)
                counter += 1
            else:
                counter_file_exist += 1
                
    print(f"Created {counter} images ({counter_file_exist} already exist) from {target_name}")
    print(f"Saved to: {tile_target_dir}")
    
def predict_image(PROJECT_DIR, image_name, model_name, target_size):
    
    #Create patches of non-annotated images
    path_image = os.path.join(PROJECT_DIR, "data", "images", image_name + ".tif")
    image_epoch = image_name.split("-")[-1].split(".")[0]
    DATA_PATH_SEG = os.path.join(PROJECT_DIR, "data", "images")
    create_image_patches(path_image, DATA_PATH_SEG, target_size)

    # Create dataset class for non-annotated image patches
    image_path = os.path.join(PROJECT_DIR, "data", "images", image_name + "_patches")
    image_dataset = ImageDataset(image_path)

    # Create prediction mask for images
    model = UNet(1)
    model_checkpoint = os.path.join(PROJECT_DIR, "models", "UNet", "Trained_Models", "Official_Models", model_name)
    checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_name = model_name.replace(".pth","")
    save_path = os.path.join(PROJECT_DIR, "data", "labels", "Predictions", image_epoch + "_" + model_name + ".tif")
    seg_mask_pred = create_prediction_mask(model, image_dataset, save_path, threshold = 0.5)

def log_wandb(project_name, run_name, csv_file):
    wandb.login(key='22efc7f5b70c0a0812fb73908cb58c659dfd8239')
    wandb.init(project = project_name, name = run_name)
    df_csv = pd.read_csv(csv_file)
    for index, row in df_csv.iterrows():
        wandb.log({
            'epoch': row['Epoch'],
            'learning rate': row["Learning Rate"],
            'train loss': row['Train Loss'],
            'train crack iou': row['Train Crack IoU'],
            'train background iou': row['Train Background IoU'],
            'train crack dice': row['Train Crack Dice'],
            'train background dice': row['Train Background Dice'],
            'val loss': row['Val Loss'],
            'val loss': row['Val Loss'],
            'val crack iou': row['Val Crack IoU'],
            'val background iou': row['Val Background IoU'],
            'val crack dice': row['Val Crack Dice'],
            'val background dice': row['Val Background Dice'],
            'time': row['Time in min']
        })
    wandb.finish()


def compare_model_predictions(model_dir):
    model_paths = []
  #  for _, _, files in os.walk(model_dir):
  #      for file in files:
    #        if file.endswith(".pth"):
    #            model_paths.append(file)
    model_paths = glob(os.path.join(model_dir, "**", "*.pth"), recursive=True)
    for path in model_paths:
        print(path)
    for model_path in model_paths:
        model = UNet(1)
        checkpoint = torch.load(os.path.join(model_dir, model_path), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(model_path)
        visualize_prediction(model, test_crack_dataset, 270)
        plt.show()

def visualize_prediction(model, dataset, index):
    image, target = dataset[index]
    image=image.unsqueeze(dim=0)
    prediction = model(image)
    prediction_mask = (torch.sigmoid(prediction) > 0.5).float()
    prediction_mask = prediction_mask.numpy().squeeze()  # Remove the singleton dimensions
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image.squeeze(dim=0).permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(prediction_mask, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    axes[2].imshow(target.permute(1, 2, 0), cmap='gray')
    axes[2].set_title('Target')
    axes[2].axis('off')

def show_training_results(path, custom_order):
    df = pd.read_csv(path)
    #df.loc[3, "Training Loss"] = 0.028 # Otherwise the plot is very scewed. Dont know why average loss was 20 in epoch 4
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    for i, column in enumerate(custom_order):
        df[column].plot(ax=axes[i], title=column)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(column)
    plt.tight_layout()
    plt.show()

def create_image_patches(path_image, path_data, target_size):
    target_width=target_size[0]
    target_height=target_size[1]
    
    image=Image.open(path_image)
    image_width, image_height = image.size
    image_name=path_image.split("/")[-1]
    epoch_name = image_name.split(".")[0]
    epoch_name = epoch_name.split("-")[-1]
    print("The image {} is {} pixels wide and {} pixels high.".format(image_name, image_width, image_height))
    image_tile_paths = os.path.join(path_data, image_name.split(".")[0] + "_patches")
    
    if not os.path.exists(image_tile_paths):
        os.makedirs(image_tile_paths)
    
    #Crop the image height and width so it is divisible by the target width and height
    image_crop_width = (image_width//target_width)*target_width
    image_crop_height = (image_height//target_height)*target_height #2560#
    print("Cropped image width: " + str(image_crop_width))
    print("Cropped image height: " + str(image_crop_height))

    counter = 0
    counter_file_exist = 0
    for i in range (0, image_crop_width, target_width):
        for j in range (0, image_crop_height, target_height):
            box = (i, j, i+target_width, j+target_height)
            tile = image.crop(box)
            tile_filename = os.path.join(image_tile_paths, f'image_{epoch_name}_{i}_{j}.png')
            if not os.path.exists(tile_filename):
                tile.save(tile_filename)
                counter += 1
            else:
                counter_file_exist += 1
    print(f"Created {counter} images ({counter_file_exist} already exist) from {image_name}")
    print(f"Saved to: {image_tile_paths}")

def create_prediction_mask(model, dataset, save_path, threshold = 0.5):
    start_time = time.time()

    x=[]
    y=[]
    prediction_list = []
    for i in range (len(dataset)):
        path = dataset.get_image_path(i)
        coordinates = path.split("/")[-1]
        coordinates = coordinates.split(".")[0]
        coordinates = coordinates.split("_")[-4:]
        x.append(coordinates[2])
        y.append(coordinates[3])

        image = dataset[i]
        image=image.unsqueeze(dim=0)
        prediction = model(image)
        prediction = (prediction > threshold).float()
        prediction = prediction.detach().squeeze().numpy().astype(np.uint8)
        prediction_list.append(prediction)

        print("\r{}/{}".format(i+1, len(dataset)), end = '', flush = True)

    unique_values=set(x)
    len_x = len(unique_values)
    unique_values=set(y)
    len_y = len(unique_values)

    column_list = []
    for i in range(len_x):
        column = np.concatenate(prediction_list[i*len_y:(i+1)*len_y], axis=0)
        column_list.append(column)

    complete_prediction = np.concatenate(column_list, axis = 1)
    duration = time.time() - start_time
    print(f"\nDuration: {round(duration/60)} minutes")

    tiff.imwrite(save_path, complete_prediction)
    print("Saved segmentation mask tif file to: {}".format(save_path))
    return complete_prediction

def reshape_data(prediction_tif_path, input_image_path):
    prediction = Image.open(prediction_tif_path)
    prediction = np.array(prediction)
    prediction_shape = prediction.shape
    print(f"Prediction shape: {prediction_shape}")

    image = Image.open(input_image_path)
    image = np.array(image)
    print(f"Original Image shape: {image.shape}")
    if image.ndim == 3:
        image = image[:prediction_shape[0], :prediction_shape[1], :]
    else:
        image = image[:prediction_shape[0], :prediction_shape[1]]
    print(f"Transformed Image shape: {image.shape}")
    
    transformed_input_image_path = input_image_path.split(".")[0]
    transformed_input_image_path = transformed_input_image_path + "_transformed.tif"
    tiff.imwrite(transformed_input_image_path, image)
    print("Saved transformed image tif file to: {}".format(transformed_input_image_path))

def analyse_df(df, target_size):
    sum_images_with_pixels = np.sum(df["crack"])
    print("Ratio of crack images to all images: " + str(round(100*sum_images_with_pixels/len(df),2)) + " %")
    sum_crack_pixels = np.sum(df["num_pixels"])
    print("Ratio of crack pixels to all pixels: " + str(round(100*sum_crack_pixels/(len(df)*target_size[0]*target_size[1]),2)) + " %")
    print("Number of samples: {}\n".format(len(df)))

def df_for_all_samples(root, testset_size = 0.2, valset_size = 0.2, SEED = 42):
    images_paths = os.path.join(root, "images")
    targets_paths = os.path.join(root, "targets")
    
    list_images = glob(os.path.join(images_paths, "**", "*.png"), recursive=True)
    list_images = natsorted(list_images, key=lambda y: y.lower())
    list_targets = glob(os.path.join(targets_paths, "**", "*.png"), recursive=True)
    list_targets = natsorted(list_targets, key=lambda y: y.lower())

    pixel_count = []
    target = Image.open(list_targets[0])
    target_size = target.size
    for target_path in list_targets:
        target = Image.open(target_path)
        target = np.array(target)
        num_crack_pxls = np.sum(target)
        pixel_count.append(num_crack_pxls)
    contains_cracks = [0 if x == 0 else 1 for x in pixel_count]

    df = pd.DataFrame(zip(list_images, list_targets, contains_cracks, pixel_count),
                      columns = ["path_image", "path_target","crack", "num_pixels"])

    return df, target_size

def create_splits(root, testset_size = 0.2, valset_size = 0.2, SEED = 42):

    df, target_size = df_for_all_samples(root, testset_size = 0.2, valset_size = 0.2, SEED = 42)
    
    bin_edges = [0, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, float('inf')]
    df['crack_pixel_bins'] = pd.cut(df['num_pixels'], bins=bin_edges, labels=False, include_lowest=True)

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=SEED)
    for train_val_idx, test_idx in strat_split.split(df, df['crack_pixel_bins']):
        train_val_set = df.iloc[train_val_idx]
        test_set = df.iloc[test_idx]
        
    strat_split_val = StratifiedShuffleSplit(n_splits=1, test_size=valset_size/(1-testset_size), random_state=SEED)  # 0.25 * 0.8 = 0.2
    for train_idx, val_idx in strat_split_val.split(train_val_set, train_val_set['crack_pixel_bins']):
        train_set = train_val_set.iloc[train_idx]
        val_set = train_val_set.iloc[val_idx]
    
    print("Training")
    analyse_df(train_set, target_size)
    print("Validation")
    analyse_df(val_set, target_size)
    print("Testing")
    analyse_df(test_set, target_size)

    return train_set, val_set, test_set

def clean_dataset(df, pixel_threshold = 5):
    low_pixel_targets_df = df[(df["num_pixels"] <= pixel_threshold) & (df["num_pixels"]>0)]
    for index, row in low_pixel_targets_df.iterrows():
        target = Image.open(row["path_target"])
        size = target.size
        target = np.array(target)
        target = np.zeros((size[1], size[0]), dtype = np.uint8)
        target_image = Image.fromarray(target)
        target_image.save(row["path_target"])
    if low_pixel_targets_df.empty:
        print("No annotations with less than {} pixels in the dataset.".format(pixel_threshold+1))
    else:
        print(low_pixel_targets_df)
        print("Removed annotations in {} targets.".format(len(low_pixel_targets_df)))

def create_target_from_patches(df, target_path):
    prefix = target_path.split("_")[-3].replace("labels","data_UNet/targets")
    print(prefix)
    target_df = df[df["path_target"].str.startswith(prefix)]
   
    x=[]
    y=[]
    path_target_list = target_df["path_target"].tolist()
    targets = []
    for path in path_target_list:
        coordinates = path.split("/")[-1]
        coordinates = coordinates.split(".")[0]
        coordinates = coordinates.split("_")[-2:]
        
        x.append(coordinates[0])
        y.append(coordinates[1])
        
        target = Image.open(path)
        target = np.array(target, dtype = np.uint8)
        targets.append(target)
              
    unique_values=set(x)
    len_x = len(unique_values)
    unique_values=set(y)
    len_y = len(unique_values)

    column_list = []
    
    for i in range(len_x):
        
        column = np.concatenate(targets[i*len_y:(i+1)*len_y], axis=0)
        column_list.append(column)
    
    print(len(column_list))
    new_target = np.concatenate(column_list, axis = 1)
    save_path = target_path.split(".")[0] + "_clean.tif"
    tiff.imwrite(save_path, new_target)
    print("Saved clean target tif file to: {}".format(save_path))