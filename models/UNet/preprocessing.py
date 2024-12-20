# For handling .tif files
from PIL import Image

# For handling the operating system
import os

target_size = (256, 256)
n_classes = 1 # Probability for each pixel to be a crack
target_width=target_size[0]
target_height=target_size[1]

DATA_PATH = "/Users/saidharb/Documents/LocalDocuments/Studienarbeit/Multitemporal-Crack-Segmentation-DL/data/data_UNet"

image_path="/Users/saidharb/Documents/LocalDocuments/Studienarbeit/Multitemporal-Crack-Segmentation-DL/data/images/2023-09-27-Weimar-Deformation-Epoch7to8-1026.tif"
image=Image.open(image_path)
image_width, image_height = image.size
image_name=image_path.split("/")[-1]
epoch_name = image_name.split(".")[0]
epoch_name = epoch_name.split("-")[-1]
print("The image {} is {} pixels wide and {} pixels high.".format(image_name, image_width, image_height))

images_paths = os.path.join(DATA_PATH, "images")
targets_paths = os.path.join(DATA_PATH, "targets")

if not images_paths:
    os.makedirs(images_paths)
if not targets_paths:
    os.makedirs(images_paths)


#Crop the image height and width so it is divisible by the target width and height
image_crop_width = (image_width//target_width)*target_width
image_crop_height = (image_height//target_height)*target_height #2560#
print("Image width: " + str(image_crop_width))
print("Image height: " + str(image_crop_height))

tile_image_dir=os.path.join(images_paths, image_name.split(".")[0])
tile_target_dir = os.path.join(targets_paths, image_name.split(".")[0])

print("tile_image_dir: " + str(tile_image_dir))
print("tile_target_dir: " + str(tile_target_dir))

if not os.path.exists(tile_image_dir):
    print("LOL")
    os.makedirs(tile_image_dir)
    os.makedirs(tile_target_dir)

counter = 0
for i in range (0, image_crop_width, target_width):
    for j in range (0, image_crop_height, target_height):
        box = (i, j, i+target_width, j+target_height)
        tile = image.crop(box)
        tile_filename = os.path.join(tile_image_dir, f'image_{epoch_name}_{i}_{j}.png')
        if not os.path.exists(tile_filename):
            tile.save(tile_filename)
            counter += 1
print(f"Created {counter} images from {image_name} and saved them to {tile_image_dir}")

target_path="/Users/saidharb/Documents/LocalDocuments/Studienarbeit/Multitemporal-Crack-Segmentation-DL/data/images/1026_labeling.tif"
target=Image.open(target_path)
image_width, image_height = target.size
target_name=target_path.split("/")[-1]
print("The image {} is {} pixels wide and {} pixels high.".format(target_name, image_width, image_height))

image_crop_width = (image_width//target_width)*target_width
image_crop_height = (image_height//target_height)*target_height #2560

counter = 0
for i in range (0, image_crop_width, target_width):
    for j in range (0, image_crop_height, target_height):
        box = (i, j, i+target_width, j+target_height)
        tile = target.crop(box)
        tile_filename = os.path.join(tile_target_dir, f'target_{epoch_name}_{i}_{j}.png')
        if not os.path.exists(tile_filename):
            tile.save(tile_filename)
            counter += 1
print(f"Created {counter} targets from {target_name} and saved them to {tile_target_dir}")