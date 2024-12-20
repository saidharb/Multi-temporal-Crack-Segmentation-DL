from PIL import Image
import os
from glob import glob
from natsort import natsorted
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
import torch
import matplotlib.pyplot as plt

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

def create_multitemporal_data(root):
    images_parent_dir = os.path.join(root, "images")
    targets_parent_dir = os.path.join(root, "targets")

    images_paths = [] # save sorted image dir paths (sorted by epoch)
    targets_paths = []
    patches_per_image = [] # list with lists per image
    patches_per_target = []
    
    for image_dir in os.listdir(images_parent_dir):
        path = os.path.join(images_parent_dir, image_dir)
        if not path.endswith(".DS_Store"):
            images_paths.append(path)

    images_paths = sorted(images_paths, key=lambda x: int(x.split('-')[-1].split('_')[0]))
    for image_path in images_paths:
        list_images = glob(os.path.join(image_path, "*.png"), recursive=True)
        list_images = natsorted(list_images, key=lambda y: y.lower())
        patches_per_image.append(list_images)

    for target_dir in os.listdir(targets_parent_dir):
        path = os.path.join(targets_parent_dir, target_dir)
        if not path.endswith(".DS_Store"):
            targets_paths.append(path)
    targets_paths = sorted(targets_paths, key=lambda x: int(x.split('-')[-1].split('_')[0]))
    for target_path in targets_paths:
        list_targets = glob(os.path.join(target_path, "*.png"), recursive=True)
        list_targets = natsorted(list_targets, key=lambda y: y.lower())
        patches_per_target.append(list_targets)

    mt_dir = os.path.join(root, "multi-temporal_data")
    mt_image_dir = os.path.join(mt_dir, "mt-images")
    mt_target_dir = os.path.join(mt_dir, "mt-targets")
    if not os.path.exists(mt_dir):
        os.mkdir(mt_dir)
        os.mkdir(mt_image_dir)
        os.mkdir(mt_target_dir)
        print(f"Created multi-temporal data directory in '{mt_dir}'.")
        

    if not all(len(sublist) == len(patches_per_image[0]) for sublist in patches_per_image):
        raise ValueError("Error: Missing multitemporal data for images.")
    
    counter = 0
    for i in range(len(patches_per_image[0])):
        stacked_image_paths = []
        stacked_images = []
        coordinates = []
        for patch_list in patches_per_image:
            stacked_image_paths.append(patch_list[i])
        for image_path in stacked_image_paths:
            coordinates = image_path.split(".")[0].split("_")[-2:]
            if not all(coordinate.isdigit() for coordinate in coordinates):
                raise ValueError("Error: Naming scheme wrong, coordinates of images are not digits.")
            image = Image.open(image_path)
            image = np.array(image)/255.0
            stacked_images.append(image)
        mt_image = np.stack(stacked_images, axis=0)
        save_path = os.path.join(mt_image_dir,
                                 "mt_image_{}_{}".format(coordinates[0], coordinates[1]))
        np.save(save_path, mt_image)
        counter += 1
    print("Saved {} multi-temporal images to '{}'.".format(counter, mt_image_dir)) 
    
    if not all(len(sublist) == len(patches_per_target[0]) for sublist in patches_per_target):
        raise ValueError("Error: Missing multitemporal data for targts.")
    
    counter = 0
    for i in range(len(patches_per_target[0])):
        stacked_target_paths = []
        stacked_targets = []
        coordinates = []
        for patch_list in patches_per_target:
            stacked_target_paths.append(patch_list[i])
        for target_path in stacked_target_paths:
            coordinates = target_path.split(".")[0].split("_")[-2:]
            if not all(coordinate.isdigit() for coordinate in coordinates):
                raise ValueError("Error: Naming scheme wrong, coordinates of targets are not digits.")
            target = Image.open(target_path)
            target = np.array(target)
            target = np.expand_dims(target, axis=-1)
            stacked_targets.append(target)
        mt_target = np.stack(stacked_targets, axis=0)
        save_path = os.path.join(mt_target_dir,
                                 "mt_targets_{}_{}".format(coordinates[0], coordinates[1]))
        np.save(save_path, mt_target)
        counter += 1

    print("Saved {} multi-temporal targets to '{}'.".format(counter, mt_target_dir)) 

def df_for_all_samples_mt(root, testset_size = 0.2, valset_size = 0.2, SEED = 42):
    images_paths = os.path.join(root, "multi-temporal_data_only_7-8", "mt-images")
    targets_paths = os.path.join(root, "multi-temporal_data_only_7-8", "mt-targets")

    list_images = glob(os.path.join(images_paths, "**", "*.npy"), recursive=True)
    list_images = natsorted(list_images, key=lambda y: y.lower())
    list_targets = glob(os.path.join(targets_paths, "**", "*.npy"), recursive=True)
    list_targets = natsorted(list_targets, key=lambda y: y.lower())

    pixel_count = []
    target = np.load(list_targets[0])
    target_size = target.shape[1:3]
    for target_path in list_targets:
        target = np.load(target_path)
        target = np.array(target)
        num_crack_pxls = np.sum(target)
        pixel_count.append(num_crack_pxls)
    contains_cracks = [0 if x == 0 else 1 for x in pixel_count]

    df = pd.DataFrame(zip(list_images, list_targets, contains_cracks, pixel_count),
                      columns = ["path_image", "path_target","crack", "num_pixels"])

    return df, target_size

def df_for_all_samples_mono(root, testset_size = 0.2, valset_size = 0.2, SEED = 42):
    images_paths = os.path.join(root, "images_patches")
    targets_paths = os.path.join(root, "targets_patches")

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

def get_only_78_mono_images_from_all_images(image_patch_dir, target_patch_dir):
    list_image_dir = os.listdir(image_patch_dir)
    list_target_dir = os.listdir(target_patch_dir)
    
    mono_image_directories = []
    mono_target_directories = []
    repeat_list = [7, 11, 15, 19, 23, 26, 31]
    
    for dir in list_image_dir:
        dir = os.path.join(image_patch_dir, dir)
        if os.path.isdir(dir):
            mono_image_directories.append(dir)
    
    for dir in list_target_dir:
        dir = os.path.join(target_patch_dir, dir)
        if os.path.isdir(dir):
            mono_target_directories.append(dir)

    mono_image_directories = natsorted(mono_image_directories, key=lambda y: y.lower())
    mono_target_directories = natsorted(mono_target_directories, key=lambda y: y.lower())

    mono_image_directories_only_78 = []
    mono_target_directories_only_78 = []
    
    for i in range(7, 32):
        mono_image_directories_only_78.append(mono_image_directories[i])
        mono_target_directories_only_78.append(mono_target_directories[i])
        if i in repeat_list:
            mono_image_directories_only_78.append(mono_image_directories[i])
            mono_target_directories_only_78.append(mono_target_directories[i])

    list_images = []
    list_targets = []
    for image_dir in mono_image_directories_only_78:
        images = glob(os.path.join(image_dir, "*.png"), recursive=True)
        list_images = list_images + images

    for target_dir in mono_target_directories_only_78:
        targets = glob(os.path.join(target_dir, "*.png"), recursive=True)
        list_targets = list_targets + targets

    list_images = natsorted(list_images, key=lambda y: y.lower())
    list_targets = natsorted(list_targets, key=lambda y: y.lower())


    return list_images, list_targets



    

def create_splits(df, testset_size = 0.2, valset_size = 0.2, SEED = 42, target_size = None):

    #df, target_size = df_for_all_samples(root, testset_size = 0.2, valset_size = 0.2, SEED = 42)
    
    bin_edges = [0, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000,
                 12000, 13000, 14000, float('inf')]
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
    analyse_df_mt(train_set, target_size)
    print("Validation")
    analyse_df_mt(val_set, target_size)
    print("Testing")
    analyse_df_mt(test_set, target_size)

    return train_set, val_set, test_set

def analyse_df_mono(df, target_size):
    sum_images_with_pixels = np.sum(df["crack"])
    print("Ratio of crack images to all images: " + str(round(100*sum_images_with_pixels/(len(df)),4)) + " %")
    sum_crack_pixels = np.sum(df["num_pixels"])
    print("Ratio of crack pixels to all pixels: " + str(round(100*sum_crack_pixels/(len(df)*target_size[0]*target_size[1]),4)) + " %")
    print("Number of samples: {}\n".format(len(df)))

def analyse_df_mt(df, target_size):
    sum_images_with_pixels = np.sum(df["crack"])
    print("Ratio of crack images to all images: " + str(round(100*sum_images_with_pixels/(len(df)),2)) + " %")
    sum_crack_pixels = np.sum(df["num_pixels"])
    print("Ratio of crack pixels to all pixels: " + str(round(100*sum_crack_pixels/(len(df)*target_size[0]*target_size[1]*target_size[2]),2)) + " %")
    print("Number of samples: {}\n".format(len(df)))


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

def find_matching_png(npy_path, base_directories):

    npy_filename = os.path.basename(npy_path)
    npy_coordinates = npy_filename.split('_')[-2] + '_' + npy_filename.split('_')[-1].replace('.npy', '')
    png_paths = []
    for base_dir in base_directories:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.png') and file.split('-')[-1].replace('.png', '') == npy_coordinates:
                    png_paths.append(os.path.join(root, file))
    return png_paths

def mono_temp_split_from_mt_split(image_patch_dir, target_patch_dir, mt_image_path_list, mt_target_path_list):
    list_image_dir = os.listdir(image_patch_dir)
    list_target_dir = os.listdir(target_patch_dir)
    
    mono_image_directories = []
    mono_target_directories = []
    repeat_list = [7, 11, 15, 19, 23, 26, 31]
    
    for dir in list_image_dir:
        dir = os.path.join(image_patch_dir, dir)
        if os.path.isdir(dir):
            mono_image_directories.append(dir)
    
    for dir in list_target_dir:
        dir = os.path.join(target_patch_dir, dir)
        if os.path.isdir(dir):
            mono_target_directories.append(dir)

    

    mono_image_directories = natsorted(mono_image_directories, key=lambda y: y.lower())
    mono_target_directories = natsorted(mono_target_directories, key=lambda y: y.lower())

    mono_image_directories_only_78 = []
    mono_target_directories_only_78 = []
    
    for i in range(7, 32):
        mono_image_directories_only_78.append(mono_image_directories[i])
        mono_target_directories_only_78.append(mono_target_directories[i])
        if i in repeat_list:
            mono_image_directories_only_78.append(mono_image_directories[i])
            mono_target_directories_only_78.append(mono_target_directories[i])

    mono_train_targets = []
    for mt_path in mt_target_path_list:
        new_list = find_matching_png(mt_path, mono_target_directories_only_78)
        mono_train_targets = mono_train_targets + new_list


    
    mono_train_images = []
    for mt_path in mt_image_path_list:
        new_list = find_matching_png(mt_path, mono_image_directories_only_78)
        mono_train_images = mono_train_images + new_list



    return mono_train_images, mono_train_targets

def df_from_path_lists(list_images,list_targets):
    pixel_count = []
    for target_path in list_targets:
        target = Image.open(target_path)
        target = np.array(target)
        num_crack_pxls = np.sum(target)
        pixel_count.append(num_crack_pxls)
    contains_cracks = [0 if x == 0 else 1 for x in pixel_count]

    df = pd.DataFrame(zip(list_images, list_targets, contains_cracks, pixel_count),
                      columns = ["path_image", "path_target","crack", "num_pixels"])
    return df


def find_match(npy_path, base_directories):

    npy_filename = os.path.basename(npy_path)
    npy_coordinates = npy_filename.split('_')[-2] + '_' + npy_filename.split('_')[-1].replace('.npy', '')
    png_paths = []
    for base_dir in base_directories:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.png') and file.split('-')[-1].replace('.png', '') == npy_coordinates:
                    png_paths.append(os.path.join(root, file))
    return png_paths


def get_mono_paths_from_mt_path(image_patch_dir, target_patch_dir, mt_image_path, mt_target_path):
    list_image_dir = os.listdir(image_patch_dir)
    list_target_dir = os.listdir(target_patch_dir)
    
    mono_image_directories = []
    mono_target_directories = []
    
    for dir in list_image_dir:
        dir = os.path.join(image_patch_dir, dir)
        if os.path.isdir(dir):
            mono_image_directories.append(dir)
    
    for dir in list_target_dir:
        dir = os.path.join(target_patch_dir, dir)
        if os.path.isdir(dir):
            mono_target_directories.append(dir)

    mono_image_directories = natsorted(mono_image_directories, key=lambda y: y.lower())
    mono_target_directories = natsorted(mono_target_directories, key=lambda y: y.lower())
    
    
    mono_train_targets = find_match(mt_target_path, mono_target_directories)
    mono_train_images = find_match(mt_image_path, mono_image_directories)



    return mono_train_images, mono_train_targets

def visualize_prediction_mt_2models(model_mt, model_mono, dataset_mt, dataset_mono, index, images_paths, targets_paths):
    '''
    model_mt: mt model
    model_mono: mono model
    dataset_mt: mt dataset
    datset_mono: mono_dataset
    index: index of image in dataset to infer
    images_paths: list of paths to mono image patches directories
    targets_paths: lists of paths to mono target patches directories
    '''
# mono inference
    image_mt_path = dataset_mt.get_image_path(index)
    target_mt_path = dataset_mt.get_target_path(index)

    mono_images, mono_targets = get_mono_paths_from_mt_path(images_paths, targets_paths, image_mt_path, target_mt_path)

    indices = []
    for path in mono_images:
        idx = dataset_mono.get_index_from_image_path(path)
        indices.append(idx)

    mono_predictions = []
    for idx in indices:
        image_mono, _ = dataset_mono[idx]
        image_mono = image_mono.unsqueeze(0)
        prediction_mono = model_mono(image_mono)
        prediction_mask_mono = (torch.sigmoid(prediction_mono) > 0.5).float()
        prediction_mono = prediction_mask_mono.squeeze(0).permute(1, 2, 0).detach().numpy()
        mono_predictions.append(prediction_mono)
# mt inference
    image_mt, target_mt = dataset_mt[index]
    image_mt=image_mt.unsqueeze(0)
    prediction_mt = model_mt(image_mt)
    image_mt = image_mt.squeeze().permute(3, 1, 2, 0).numpy()  # Change order to HxWxC
    target_mt = target_mt.permute(3, 1, 2, 0).numpy()  # Remove the channel dimension for target
    prediction_mask_mt = (torch.sigmoid(prediction_mt) > 0.5).float()
    prediction_mt = prediction_mask_mt.squeeze(0).permute(3, 1, 2, 0).detach().numpy()

    for slice in range(0, 32): 
        fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        axes[0].imshow(image_mt[slice, :, :, :])
        axes[0].set_title('Image t = {}'.format(slice))
        axes[0].axis('off')
        axes[1].imshow(target_mt[slice, :, :, :], cmap='gray')
        axes[1].set_title('Target t = {}'.format(slice))
        axes[1].axis('off')  
        axes[2].imshow(prediction_mt[slice, :, :, :], cmap='gray')
        axes[2].set_title('SwinUNetR t = {}'.format(slice))
        axes[2].axis('off')  
        axes[3].imshow(mono_predictions[slice], cmap='gray')
        axes[3].set_title('UNet t = {}'.format(slice))
        axes[3].axis('off')  
        
        plt.show()

def visualize_prediction_model_mt(model, dataset, index):
    image, target = dataset[index]
    image=image.unsqueeze(0)
    print(image.shape)
    print(type(image))
    prediction = model(image)
    image = image.squeeze().permute(3, 1, 2, 0).numpy()  # Change order to HxWxC
    target = target.permute(3, 1, 2, 0).numpy()  # Remove the channel dimension for target
    prediction_mask = (torch.sigmoid(prediction) > 0.5).float()
    prediction = prediction_mask.squeeze(0).permute(3, 1, 2, 0).detach().numpy()

    for slice in range(0, 32): 
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(image[slice, :, :, :])
        axes[0].set_title('Image t = {}'.format(slice))
        axes[0].axis('off')
        axes[1].imshow(target[slice, :, :, :], cmap='gray')
        axes[1].set_title('Target t = {}'.format(slice))
        axes[1].axis('off')  # Hide axes
        axes[2].imshow(prediction[slice, :, :, :], cmap='gray')
        axes[2].set_title('Prediction t = {}'.format(slice))
        axes[2].axis('off')  # Hide axes
        
        plt.show()
