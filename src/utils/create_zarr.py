import os
import numpy as np
import pandas as pd
import rasterio
import zarr

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


BASE_PATH = "./input/"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
MASK_PATH = os.path.join(BASE_PATH, "train_mask")
ZARR_PATH = os.path.join(BASE_PATH, "train_zarr")
IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)

def main():
    df_train = pd.read_csv(
        os.path.join(BASE_PATH, "train.csv")
    )

    g_out = zarr.group(ZARR_PATH)
    for image_id in df_train.id.values:
        print(image_id)
        img_path = os.path.join(TRAIN_PATH, image_id+'.tiff')
        mask_path = os.path.join(MASK_PATH, image_id+'.tiff')

        with rasterio.open(img_path, transform=IDNT) as image:
            if image.count != 3:
                subdatasets = image.subdatasets
                layers = []
                if len(subdatasets) > 0:
                    for i, subdataset in enumerate(subdatasets, 0):
                        layers.append(rasterio.open(subdataset))

            width = image.width
            height = image.height
            
            if image.count != 3:
                crop_img = np.zeros((height, width, 3), np.uint8)
                for i,layer in enumerate(layers):
                    crop_img[:,:,i] = layer.read()
            else:
                crop_img = image.read([1,2,3])
                crop_img = np.moveaxis(crop_img, 0, -1)
           
        
        with rasterio.open(mask_path) as mask:
            crop_mask = mask.read()
            crop_mask = np.moveaxis(crop_mask, 0, -1)

        g_out[image_id] = crop_img
        g_out[image_id+'mask'] = crop_mask


if __name__ == "__main__":
    main()