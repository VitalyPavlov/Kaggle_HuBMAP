
import os
import torch
import rasterio
import skimage 
import cv2
import numpy as np
from src.data import augmentation
from src.utils.get_grid import make_grid
from src.utils import get_crop

IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)

def evaluate_model(model, criterion, metrics, entropy_thr, test_ids, fold,
                   path_img, path_mask, crop_dim, out_dim, device):
    model.eval()
    running_loss = 0.0
    valid_size = 0
    dice_pos, dice_neg = [], []
    for i_img, img_file in enumerate(test_ids):
        images_path = os.path.join(path_img, img_file+'.tiff')
        mask_path = os.path.join(path_mask, img_file+'.tiff')
    
        with rasterio.open(images_path, transform=IDNT, num_threads='all_cpus') as img_data, \
             rasterio.open(mask_path, num_threads='all_cpus') as mask_data:

            slices = make_grid(img_data.shape, window=crop_dim, min_overlap=0)
            for x1, y1 in slices:
                crop_img = get_crop.make_crop_img(img_data, x1, y1, crop_dim, out_dim)
                crop_mask = get_crop.make_crop_mask(mask_data, x1, y1, crop_dim, out_dim)

                if out_dim != crop_dim:
                    crop_img = cv2.resize(crop_img, (out_dim, out_dim))
                    crop_mask = cv2.resize(crop_mask, (out_dim, out_dim))
                    crop_mask = np.expand_dims(crop_mask, axis=2)

                if entropy_thr > 0:
                    entropy = skimage.measure.shannon_entropy(crop_img)
                    if entropy <= entropy_thr:
                        continue

                sample = augmentation.get_preprocessing()(image=crop_img)
                crop_img = sample['image']

                inputs = torch.Tensor(crop_img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                labels = torch.Tensor(crop_mask).unsqueeze(0).permute(0, 3, 1, 2).to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)

                    outputs = torch.sigmoid(outputs)
                    preds = (outputs > 0.5).long()
                    _dice_pos, _dice_neg = metrics(preds.data.cpu().numpy(), 
                                                  labels.data.cpu().numpy())

                    dice_pos.extend(_dice_pos)
                    dice_neg.extend(_dice_neg)
                    valid_size += 1
                    
    loss_val= running_loss / valid_size
    metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
    return loss_val, metrics_val, np.nanmean(dice_pos), np.nanmean(dice_neg)


def evaluate_model_all_imgs(model, criterion, metrics, test_ids, fold,
                            path_img, path_mask, crop_dim, out_dim, device):
    model.eval()
    running_loss = 0.0
    valid_size = 0
    dice_pos, dice_neg = [], []
    for i_img, img_file in enumerate(test_ids):
        images_path = os.path.join(path_img, img_file+'.tiff')
        mask_path = os.path.join(path_mask, img_file+'.tiff')

        with rasterio.open(images_path, transform=IDNT, num_threads='all_cpus') as img_data, \
             rasterio.open(mask_path, num_threads='all_cpus') as mask_data:
            
            slices = make_grid((img_data.height // 5, img_data.width), window=crop_dim, min_overlap=0)
            offset = fold * img_data.height // 5
            
            for x1, y1 in slices:
                x1 += offset
                crop_img = get_crop.make_crop_img(img_data, x1, y1, crop_dim, out_dim)
                crop_mask = get_crop.make_crop_mask(mask_data, x1, y1, crop_dim, out_dim)

                entropy = skimage.measure.shannon_entropy(crop_img)
                if entropy < 5:
                    continue

                sample = augmentation.get_preprocessing()(image=crop_img)
                crop_img = sample['image']

                inputs = torch.Tensor(crop_img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                labels = torch.Tensor(crop_mask).unsqueeze(0).permute(0, 3, 1, 2).to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)

                    outputs = torch.sigmoid(outputs)
                    preds = (outputs > 0.5).long()

                    _dice_pos, _dice_neg = metrics(preds.data.cpu().numpy(),
                                                   labels.data.cpu().numpy())

                    dice_pos.extend(_dice_pos)
                    dice_neg.extend(_dice_neg)
                    valid_size += 1

    loss_val= running_loss / valid_size
    metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
    return loss_val, metrics_val, np.nanmean(dice_pos), np.nanmean(dice_neg)



def evaluate_model_full_img(model, criterion, metrics, test_ids, fold,
                            path_img, path_mask, crop_dim, out_dim, device):
    model.eval()
    running_loss = 0.0
    valid_size = 0
    dice_pos, dice_neg = [], []
    for i_img, img_file in enumerate(test_ids):
        images_path = os.path.join(path_img, img_file+'.tiff')
        mask_path = os.path.join(path_mask, img_file+'.tiff')
        
        with rasterio.open(images_path, transform=IDNT, num_threads='all_cpus') as img_data, \
             rasterio.open(mask_path, num_threads='all_cpus') as mask_data:

            img_preds = np.zeros(img_data.shape, dtype=np.uint8)
            slices = make_grid(img_data.shape, window=crop_dim, min_overlap=0)
            for x1, y1 in slices:
                crop_img = get_crop.make_crop_img(img_data, x1, y1, crop_dim, out_dim)
                crop_mask = get_crop.make_crop_mask(mask_data, x1, y1, crop_dim, out_dim)

                entropy = skimage.measure.shannon_entropy(crop_img)
                if entropy < 5:
                    continue

                sample = augmentation.get_preprocessing()(image=crop_img)
                crop_img = sample['image']

                inputs = torch.Tensor(crop_img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                labels = torch.Tensor(crop_mask).unsqueeze(0).permute(0, 3, 1, 2).to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)

                    outputs = torch.sigmoid(outputs)
                    preds = (outputs > 0.5).long()

                preds = preds.data.cpu().numpy().astype(np.uint8)
                preds = np.moveaxis(preds[0], 0, -1)
                preds = cv2.resize(preds, (crop_dim, crop_dim))
                
                img_preds[x1:x1+crop_dim, y1:y1+crop_dim] += preds.astype(np.uint8)
                valid_size += 1
            
        with rasterio.open(mask_path, num_threads='all_cpus') as mask_data:
            labels = mask_data.read()
        
        img_preds = np.expand_dims(img_preds, axis=0)
        _dice_pos, _dice_neg = metrics(img_preds, labels)

        dice_pos.extend(_dice_pos)
        dice_neg.extend(_dice_neg)

    loss_val= running_loss / valid_size
    metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
    return loss_val, metrics_val, np.nanmean(dice_pos), np.nanmean(dice_neg)



def evaluate_model_tta(model, test_ids):
    model.eval()
    dice_pos, dice_neg = [], []
    for i_img, img_file in enumerate(test_ids):
        images_path = os.path.join(BASE_DIR, 'train', img_file+'.tiff')
        mask_path = os.path.join(BASE_DIR, 'train_mask', img_file+'.tiff')
        with rasterio.open(images_path, transform=IDNT, num_threads='all_cpus') as img_data, rasterio.open(mask_path, num_threads='all_cpus') as mask_data:

            slices = make_grid(img_data.shape, window=CROP_DIM, min_overlap=0)
            for x1, y1 in slices:
                
                crop_img = make_crop_img(img_data, x1, y1, CROP_DIM, OUT_DIM)
                crop_mask = make_crop_mask(mask_data, x1, y1, CROP_DIM, OUT_DIM)

                entropy = skimage.measure.shannon_entropy(crop_img)

                if entropy < 5:
                    continue

                preds = None
                for j, tta_aug in enumerate(tta_augs):
                    # Augmentation
                    aug_img = tta_aug(image = crop_img)['image']
                    aug_img =  get_preprocessing()(image=aug_img)['image']
                    aug_img = np.moveaxis(aug_img, -1, 0)
                    aug_img = torch.from_numpy(aug_img)

                    with torch.no_grad():
                        score = model(aug_img.float().to(DEVICE)[None])
                        score = torch.sigmoid(score)
                        score = (score > 0.5).long()
                        score = score.cpu().numpy()[0][0]

                        # Deaugmentation
                        if tta_deaugs[j] is not None:
                            score = tta_deaugs[j](image = crop_img, 
                                                mask = score)['mask'] 

                        if preds is None:
                            preds = score / len(tta_augs) 
                        else:       
                            preds += score / len(tta_augs) 

                preds = (preds > 0).astype(np.uint8)
                #preds = cv2.resize(preds, (CROP_DIM, CROP_DIM))
                
                preds = np.expand_dims(preds, axis=0)
                crop_mask = np.moveaxis(crop_mask, -1, 0)

                _dice_pos, _dice_neg = get_metrics(crop_mask, 
                                                   preds)


                dice_pos.extend(_dice_pos)
                dice_neg.extend(_dice_neg)

    metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
    
    return metrics_val, dice_pos, dice_neg
