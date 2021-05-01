import os
import numpy as np
import pandas as pd
from PIL import Image
import rasterio


BASE_PATH = "../input/"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape).T


def read_mask(image_id, df_train):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

    img_path = os.path.join(TRAIN_PATH, image_id+'.tiff')
    image = rasterio.open(img_path, transform = identity,
                          num_threads='all_cpus')

    print(image_id, image.height, image.width)
    
    mask = rle2mask(
        df_train[df_train["id"] == image_id]["encoding"].values[0], 
        (image.width, image.height)
    )
    
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
        
    return mask
 

def main():
    df_train = pd.read_csv(
        os.path.join(BASE_PATH, "train.csv")
    )

    for image_id in df_train.id.values:
        mask = read_mask(image_id, df_train)
        mask.save(f'../input/train_mask/{image_id}.tiff')

if __name__ == "__main__":
    main()
