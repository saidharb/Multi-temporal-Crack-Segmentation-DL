
# For handling the operating system
import os

# PyTorch
import torch

def save_checkpoint(model, optimizer, epoch, logs, sigmoid_threshold, final = True):
    trained_models_path = "Trained_Models"
    if not os.path.exists(trained_models_path):
        os.makedirs(trained_models_path)
    version = 0
    base_model_name = f"model_nrepochs_{epoch}_sig_{str(sigmoid_threshold).replace('.', '-')}"
    model_name = base_model_name

    if not final:
        model_name = f"checkpoint_model"
    else:
        while os.path.exists(os.path.join(trained_models_path, model_name + ".pth")):
            version += 1
            model_name = f"{base_model_name}_v{version}"
   
    logs.to_csv(os.path.join(trained_models_path, model_name) + '_training_logs.csv', index=False)
    path = os.path.join(trained_models_path, model_name + ".pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

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
    target_name=target_path.split("/")[-1]
    print("The image {} is {} pixels wide and {} pixels high.".format(target_name, image_width, image_height))

    image_crop_width = (image_width//target_width)*target_width
    image_crop_height = (image_height//target_height)*target_height
    if custom_heightXwidth is not None:
        image_crop_width = custom_heightXwidth[1]
        image_crop_height = custom_heightXwidth[0]
    print("Cropped Target width: " + str(image_crop_width))
    print("Cropped Target height: " + str(image_crop_height))

    target_paths = os.path.join(DATA_PATH, "target")
    print(target_paths)
    if not targets_paths:
        os.makedirs(targets_paths)
    tile_target_dir = os.path.join(targets_paths, target_name.split(".")[0])
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

