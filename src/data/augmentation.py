import cv2
import numpy as np
import albumentations as A

# define heavy augmentations
def get_augmentation_v1_resize(size):
    train_transform = [
        #A.OneOf([
        #    A.Blur(blur_limit=[19,23], p=1),
        #    A.MedianBlur(blur_limit=[19,23], p=1)
        #], p=0.2),

        A.Resize(size, size, always_apply=True, p=1),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5, 
                           border_mode=cv2.BORDER_REFLECT),

        A.OneOf([
            A.OpticalDistortion(p=0.4),
            A.GridDistortion(p=0.2),
            A.IAAPiecewiseAffine(p=0.4),
        ], p=0.3),
        
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2, p=0.3),
            A.RandomBrightnessContrast(p=0.4),            
        ], p=0.3),
    ]
    return A.Compose(train_transform)


def get_augmentation_v1(size):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5, 
                           border_mode=cv2.BORDER_REFLECT),

        A.OneOf([
            A.OpticalDistortion(p=0.4),
            A.GridDistortion(p=0.2),
            A.IAAPiecewiseAffine(p=0.4),
        ], p=0.3),
        
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2, p=0.3),
            A.RandomBrightnessContrast(p=0.4),            
        ], p=0.3),
    ]
    return A.Compose(train_transform)


def get_augmentation_v2(size):
    train_transform = [
        A.OneOf([
            A.RandomBrightness(limit=.2, p=1), 
            A.RandomContrast(limit=.2, p=1), 
            A.RandomGamma(p=1)
        ], p=.5),
        A.OneOf([
            A.Blur(blur_limit=[5, 9], p=1),
            A.MedianBlur(blur_limit=[5, 9], p=1)
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(0.002, p=.5),
            A.IAAAffine(p=.5),
        ], p=.25),
        A.RandomRotate90(p=.5),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        A.Cutout(num_holes=10, 
                    max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
                    p=.25),
        A.ShiftScaleRotate(p=.25)
    ]
    return A.Compose(train_transform)


def get_augmentation_v3(size):
    train_transform = [
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5, 
                           border_mode=cv2.BORDER_REFLECT),
        
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1)
        ], p=0.1),
        
        A.OneOf([
            A.OpticalDistortion(p=0.4),
            A.GridDistortion(p=0.2),
            A.IAAPiecewiseAffine(p=0.4),
            A.GaussNoise(0.002, p=.5),
            A.IAAAffine(p=.5),
        ], p=0.3),
        
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2, p=0.3),
            A.RandomBrightnessContrast(p=0.4), 
            A.RandomBrightness(limit=.2, p=0.3), 
            A.RandomContrast(limit=.2, p=0.3), 
            A.RandomGamma(p=0.3)           
        ], p=0.3),

        A.Cutout(num_holes=10, 
                    max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
                    p=.2),
    ]
    return A.Compose(train_transform)


def get_preprocessing():
    _transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225))
    ]
    return A.Compose(_transform)

