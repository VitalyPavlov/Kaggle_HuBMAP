import os
import cv2
import numpy as np
import pandas as pd
import torch
import random
import skimage  
import h5py

from src.utils.get_crop import *


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
        self.images_path = [os.path.join(path_img, image_id+'.hdf5') for image_id in self.ids]
        self.masks_path = [os.path.join(path_mask, image_id+'.hdf5') for image_id in self.ids]
        self.crop_dim = crop_dim
        self.out_dim = out_dim 
        self.entropy_thr = entropy_thr
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    
    def __getitem__(self, i):
        # read data
        with h5py.File(self.images_path[i], 'r') as f:
            entropy = 0
            while entropy <= self.entropy_thr:
                x1 = random.randrange(f.attrs['height'] - self.crop_dim)
                y1 = random.randrange(f.attrs['width'] - self.crop_dim)
                crop_img = f['img'][x1:x1+self.crop_dim, y1:y1+self.crop_dim,:]

                #if self.entropy_thr > 0:
                #    entropy = skimage.measure.shannon_entropy(crop_img)
                #else:
                entropy = 10
            
            crop_mask = f['mask'][x1:x1+self.crop_dim, y1:y1+self.crop_dim,:]

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
