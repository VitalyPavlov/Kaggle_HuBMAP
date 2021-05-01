import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
import zarr

from src.utils.utils import check_background
from src.utils import set_seed

set_seed.seed_everything()

IMG_SIZES = {
    "2f6ecfcdf": (31278, 25794),
    "8242609fa": (31299, 44066),
    "aaa6a05cc": (18484, 13013),
    "cb2d976f4": (34940, 49548),
    "b9a3865fc": (31295, 40429),
    "b2dc8411c": (14844, 31262),
    "0486052bb": (25784, 34937),
    "e79de561c": (16180, 27020),
    "095bf7a1f": (38160, 39000),
    "54f2eec69": (30440, 22240),
    "4ef6695ce": (39960, 50680),
    "26dc41664": (38160, 42360),
    "c68fe75ea": (26840, 49780),
    "afa5e8098": (36800, 43780),
    "1e2425f28": (26780, 32220),
    "aa05346ff": (30720, 47340),
    "2ec3f1bb9": (23990, 47723),
    "3589adb90": (29433, 22165),
    "d488c759a": (46660, 29020),
    "d488c759ahand": (46660, 29020),
    "57512b7f1": (33240, 43160),
}


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
        self.zarr = zarr.open(path_img, mode="r")
        self.crop_dim = crop_dim
        self.out_dim = out_dim 
        self.entropy_thr = entropy_thr
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):
        # read data
        height = IMG_SIZES[self.ids[i]][0]
        width = IMG_SIZES[self.ids[i]][1]

        is_background = True
        while is_background:
            x1 = random.randrange(height - self.crop_dim)
            y1 = random.randrange(width - self.crop_dim)
            crop_img = self.zarr[self.ids[i]][x1:x1+self.crop_dim, y1:y1+self.crop_dim,:]
            if self.crop_dim != self.out_dim:
                crop_img = cv2.resize(crop_img, (self.out_dim, self.out_dim))
            is_background = check_background(crop_img, self.out_dim)
            
        crop_mask = self.zarr[self.ids[i]+'mask'][x1:x1+self.crop_dim, y1:y1+self.crop_dim,:]
        if self.crop_dim != self.out_dim:
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



class Dataset_test:    
    def __init__(
            self, 
            slices,
            path_img,
            image_id,
            crop_dim,
            out_dim,
            preprocessing=None
    ):
        self.slices = slices
        self.zarr = zarr.open(path_img, mode="r")
        self.image_id = image_id
        self.crop_dim = crop_dim
        self.out_dim = out_dim 
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):
        # read data
        x1, y1 = self.slices[i]
        crop_img = self.zarr[self.image_id][x1:x1+self.crop_dim, y1:y1+self.crop_dim,:]
        crop_mask = self.zarr[self.image_id+'mask'][x1:x1+self.crop_dim, y1:y1+self.crop_dim,:]

        if self.crop_dim != self.out_dim:
            crop_img = cv2.resize(crop_img, (self.out_dim, self.out_dim))
            crop_mask = cv2.resize(crop_mask, (self.out_dim, self.out_dim))
            crop_mask = np.expand_dims(crop_mask, axis=2)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=crop_img, mask=crop_mask)
            crop_img, crop_mask = sample['image'], sample['mask']
            
        return torch.Tensor(crop_img).permute(2, 0, 1), torch.Tensor(crop_mask).permute(2, 0, 1), i
        
    def __len__(self):
        return len(self.slices)
