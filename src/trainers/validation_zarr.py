import os
import torch
import skimage 
import cv2
import numpy as np
import zarr
from pydoc import locate
from src.data import augmentation
from src.utils.utils import make_grid
from src.data.dataloader_zarr import IMG_SIZES


def evaluate_model(model, test_ids, criterion, metrics, 
                   dataloader, path_img, crop_dim, out_dim, 
                   preprocessing, device, batch_size):
    model.eval()
    running_loss = 0.0
    valid_size = 0
    dice_pos, dice_neg = [], []
    for i_img, img_file in enumerate(test_ids):
        slices = make_grid(
            IMG_SIZES[img_file], 
            window=crop_dim, 
            min_overlap=0
         )
        valid_size += len(slices)

        test_dataset = dataloader(
                            slices,
                            path_img=path_img,
                            image_id=img_file,
                            crop_dim=crop_dim,
                            out_dim=out_dim,
                            preprocessing=preprocessing
                        )

        dataloaders_test = torch.utils.data.DataLoader(test_dataset, shuffle=False, 
                                                       num_workers=min(10, batch_size), 
                                                       batch_size=batch_size)

        for inputs, labels, ind in dataloaders_test:
            inputs = inputs.to(device)
            labels = labels.to(device)

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
                        
    loss_val = running_loss / valid_size
    metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
    return loss_val, metrics_val, np.nanmean(dice_pos), np.nanmean(dice_neg)



def evaluate_model_full(model, test_ids, criterion, metrics, 
                        dataloader, path_img, crop_dim, out_dim, 
                        preprocessing, device, batch_size):
    model.eval()
    running_loss = 0.0
    valid_size = 0
    dice_pos, dice_neg = [], []
    for i_img, img_file in enumerate(test_ids):
        slices = make_grid(
            IMG_SIZES[img_file], 
            window=crop_dim, 
            min_overlap=0
         )
        valid_size += len(slices)

        test_dataset = dataloader(
                            slices,
                            path_img=path_img,
                            image_id=img_file,
                            crop_dim=crop_dim,
                            out_dim=out_dim,
                            preprocessing=preprocessing
                        )

        dataloaders_test = torch.utils.data.DataLoader(test_dataset, shuffle=False, 
                                                       num_workers=min(10, batch_size), 
                                                       batch_size=batch_size)

        img_preds = np.zeros(IMG_SIZES[img_file], dtype=np.uint8)
        for inputs, labels, ind in dataloaders_test:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).long()

            for i in range(len(ind)):
                pred = preds[i].data.cpu().numpy().astype(np.uint8)
                pred = np.moveaxis(pred, 0, -1)
                pred = cv2.resize(pred, (crop_dim, crop_dim))

                x1, y1 = slices[ind[i]]
                img_preds[x1:x1+crop_dim, y1:y1+crop_dim] = pred

        
        mask = zarr.open(path_img, mode="r")[img_file+'mask'][:,:,0]

        _dice_pos, _dice_neg = metrics(img_preds, mask)        
        dice_pos.extend(_dice_pos)
        dice_neg.extend(_dice_neg)
                        
    loss_val = running_loss / valid_size
    metrics_val = 0.5 * np.nanmean(dice_pos) + 0.5 *  np.nanmean(dice_neg)
    return loss_val, metrics_val, np.nanmean(dice_pos), np.nanmean(dice_neg)

