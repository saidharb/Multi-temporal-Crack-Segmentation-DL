from PIL import Image
import numpy as np
import os

def change_segmentation_to_orange(input_tif, output_tif): #input file.tif, output file.tif
    PROJECT_DIR=os.getcwd()
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')

    input_path=os.path.join(DATA_DIR, "labels",input_tif)
    output_path=os.path.join(DATA_DIR,"labels", output_tif)
    print(input_path)
    print(output_path)
    img = Image.open(input_path)
    img_array = np.array(img)

    orange = [255, 165, 0]

    rgba_image = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    # 4 channels for RGBA image

    mask = img_array > 0
    rgba_image[:,:,:3] = orange # Right now everything is transpartent
    # as the alpha value is zero
    rgba_image[:, :, 3] = mask * 255 #Set mask to be opaque

    output_img = Image.fromarray(rgba_image, 'RGBA')
    output_img.save(output_path)

input_tif = '1026_Full.tif'
output_tif = 'crack1_color.tif'
change_segmentation_to_orange(input_tif, output_tif)
