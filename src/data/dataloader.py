import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
import skimage  
import rasterio
from rasterio.windows import Window

from src.utils.get_crop import *

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)


# classes for data loading and preprocessing
class Dataset_random_img:    
    def __init__(
            self, 
            ids,
            path_img,
            path_mask,
            crop_dim,
            out_dim,
            entropy_thr,
            augmentation=None, 
            preprocessing=None
    ):
        self.ids = ids
        self.images_path = [os.path.join(path_img, image_id+'.tiff') for image_id in self.ids]
        self.masks_path = [os.path.join(path_mask, image_id+'.tiff') for image_id in self.ids]
        self.crop_dim = crop_dim
        self.out_dim = out_dim 
        self.entropy_thr = entropy_thr
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):
        # read data
        with rasterio.open(self.images_path[i], transform=IDNT, num_threads='all_cpus') as src:
            crop_img, x1, y1 = make_crop_img_random(src, self.crop_dim, self.out_dim, self.entropy_thr)
        
        with rasterio.open(self.masks_path[i], num_threads='all_cpus') as src:
            crop_mask = make_crop_mask(src, x1, y1, self.crop_dim, self.out_dim)

        crop_img = cv2.resize(crop_img, (self.out_dim, self.out_dim))
        crop_mask = cv2.resize(crop_mask, (self.out_dim, self.out_dim))
        crop_mask = np.expand_dims(crop_mask, axis=2)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
            
        return torch.Tensor(crop_img).permute(2, 0, 1), torch.Tensor(crop_mask).permute(2, 0, 1)
        
    def __len__(self):
        return len(self.ids)





# classes for data loading and preprocessing
class Dataset_random_mask:    
    def __init__(
            self, 
            ids,
            path_img,
            path_mask,
            crop_dim,
            out_dim,
            entropy_thr,
            augmentation=None, 
            preprocessing=None
    ):
        self.ids = ids
        self.images_path = [os.path.join(path_img, image_id+'.tiff') for image_id in self.ids]
        self.masks_path = [os.path.join(path_mask, image_id+'.tiff') for image_id in self.ids]
        self.crop_dim = crop_dim
        self.out_dim = out_dim 
        self.entropy_thr = entropy_thr
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):
        # read data
        with rasterio.open(self.masks_path[i], num_threads='all_cpus') as src:
            crop_mask, x1, y1 = make_crop_mask_random(src, self.crop_dim, self.out_dim)

        with rasterio.open(self.images_path[i], transform=IDNT, num_threads='all_cpus') as src:
            crop_img = make_crop_img(src, x1, y1, self.crop_dim, self.out_dim)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
            
        return torch.Tensor(crop_img).permute(2, 0, 1), torch.Tensor(crop_mask).permute(2, 0, 1)
        
    def __len__(self):
        return len(self.ids)


class Dataset_all_imgs:    
    def __init__(
            self, 
            ids,
            path_img,
            path_mask,
            crop_dim,
            out_dim,
            entropy_thr,
            augmentation=None, 
            preprocessing=None
    ):
        self.ids = ids
        self.images_path = [os.path.join(path_img, image_id+'.tiff') for image_id in self.ids]
        self.masks_path = [os.path.join(path_mask, image_id+'.tiff') for image_id in self.ids]
        self.crop_dim = crop_dim
        self.out_dim = out_dim 
        self.entropy_thr = entropy_thr
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        width = self.image_size[self.ids[i]][1]
        height = self.image_size[self.ids[i]][0]

        valid_min = max(0, self.fold * height // 5 - self.crop_dim)
        valid_max = (self.fold + 1) * height // 5

        entropy = 0
        while entropy <= self.entropy_thr:
            x1 = valid_min
            while x1 >= valid_min and x1 <= valid_max:
                x1 = random.randrange(height - self.crop_dim)
            
            y1 = random.randrange(width - self.crop_dim)

            with rasterio.open(self.images_path[i], transform=IDNT, num_threads='all_cpus') as src:
                crop_img = make_crop_img(src, x1, y1, self.crop_dim, self.out_dim)

            if self.entropy_thr > 0:
                entropy = skimage.measure.shannon_entropy(crop_img)
            else:
                entropy = 1

        with rasterio.open(self.masks_path[i], num_threads='all_cpus') as src:
            crop_mask = make_crop_mask(src, x1, y1, self.crop_dim, self.out_dim)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
            
        return torch.Tensor(crop_img).permute(2, 0, 1), torch.Tensor(crop_mask).permute(2, 0, 1)
        
    def __len__(self):
        return len(self.ids)