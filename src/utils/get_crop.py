import cv2
import numpy as np
import random
import skimage.measure  
import rasterio
from rasterio.windows import Window

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def make_crop_img_random(image, CROP_DIM, OUT_DIM, entropy_thr):
    if image.count != 3:
        subdatasets = image.subdatasets
        layers = []
        if len(subdatasets) > 0:
            for i, subdataset in enumerate(subdatasets, 0):
                layers.append(rasterio.open(subdataset))
            
    width = image.width
    height = image.height

    entropy = 0
    while entropy <= entropy_thr:
        x1 = random.randrange(height - CROP_DIM)
        y1 = random.randrange(width - CROP_DIM)

        if image.count != 3:
            crop_img = np.zeros((CROP_DIM, CROP_DIM, 3), np.uint8)
            for i,layer in enumerate(layers):
                crop_img[:,:,i] = layer.read(window=Window(y1, x1, CROP_DIM, CROP_DIM))
        else:
            crop_img = image.read([1,2,3], window=Window(y1, x1, CROP_DIM, CROP_DIM))
            crop_img = np.moveaxis(crop_img, 0, -1)

        #if OUT_DIM != CROP_DIM:
        #    crop_img = cv2.resize(crop_img, (OUT_DIM, OUT_DIM))
        
        if entropy_thr > 0:
            entropy = skimage.measure.shannon_entropy(crop_img)
        else:
            entropy = 1

    return crop_img, x1, y1


def make_crop_mask_random(mask, CROP_DIM, OUT_DIM):
    width = mask.width
    height = mask.height

    is_mask = False
    while not is_mask:
        x1 = random.randrange(height - CROP_DIM)
        y1 = random.randrange(width - CROP_DIM)

        crop_mask = mask.read(window=Window(y1, x1, CROP_DIM, CROP_DIM))
        if random.uniform(0, 1) < 0.3:
            is_mask = True
        else:
            is_mask = True if (crop_mask == 1).any() else False


    crop_mask = np.moveaxis(crop_mask, 0, -1)
    
    #if OUT_DIM != CROP_DIM:
    #    crop_mask = cv2.resize(crop_mask, (OUT_DIM, OUT_DIM))
    #    crop_mask = np.expand_dims(crop_mask, axis=2)
    
    return crop_mask, x1, y1



def make_crop_img(image, x1, y1, CROP_DIM, OUT_DIM):
    if image.count != 3:
        subdatasets = image.subdatasets
        layers = []
        if len(subdatasets) > 0:
            for i, subdataset in enumerate(subdatasets, 0):
                layers.append(rasterio.open(subdataset))


    if image.count != 3:
        crop_img = np.zeros((CROP_DIM, CROP_DIM, 3), np.uint8)
        for i,layer in enumerate(layers):
            crop_img[:,:,i] = layer.read(window=Window(y1, x1, CROP_DIM, CROP_DIM))
    else:
        crop_img = image.read([1,2,3], window=Window(y1, x1, CROP_DIM, CROP_DIM))
        crop_img = np.moveaxis(crop_img, 0, -1)
    
    #if OUT_DIM != CROP_DIM:
    #    crop_img = cv2.resize(crop_img, (OUT_DIM, OUT_DIM))

    return crop_img


def make_crop_mask(mask, x1, y1, CROP_DIM, OUT_DIM):
    crop_mask = mask.read(window=Window(y1, x1, CROP_DIM, CROP_DIM))
    crop_mask = np.moveaxis(crop_mask, 0, -1)

    #if OUT_DIM != CROP_DIM:
    #    crop_mask = cv2.resize(crop_mask, (OUT_DIM, OUT_DIM))
    #    crop_mask = np.expand_dims(crop_mask, axis=2)
    return crop_mask