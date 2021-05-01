import os
import numpy as np
import pandas as pd
import rasterio
import h5py

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


BASE_PATH = "./input/"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
MASK_PATH = os.path.join(BASE_PATH, "train_mask")
HDF5_PATH = os.path.join(BASE_PATH, "train_hdf5")
IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)

def main():
    df_train = pd.read_csv(
        os.path.join(BASE_PATH, "train.csv")
    )

    for image_id in df_train.id.values:
        print(image_id)
        img_path = os.path.join(TRAIN_PATH, image_id+'.tiff')
        mask_path = os.path.join(MASK_PATH, image_id+'.tiff')
        hdf5_path = os.path.join(HDF5_PATH, image_id+'.hdf5')

        if os.path.exists(hdf5_path):
            continue

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

        with h5py.File(hdf5_path, 'w') as f:
            f.attrs["width"] = width
            f.attrs["height"] = height
            f.create_dataset('img', data=crop_img, dtype='uint8')
            f.create_dataset('mask', data=crop_mask, dtype='uint8')


if __name__ == "__main__":
    main()