
import logging
from omegaconf import DictConfig, OmegaConf
import hydra

import numpy as np
import pandas as pd
import functools
import time
import copy
from collections import deque
from pydoc import locate
from torch.utils.tensorboard import SummaryWriter


import torch
from torch.cuda.amp import GradScaler
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L

from src.data import augmentation, dataloader
from src.utils import set_seed
from src.trainers import train_loop

import rasterio
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


FOLD_IMGS = {
    0: ["4ef6695ce", "0486052bb", "2f6ecfcdf"],
    1: ["c68fe75ea", "095bf7a1f", "aaa6a05cc"],
    2: ["afa5e8098", "1e2425f28", "b2dc8411c"],
    3: ["cb2d976f4", "8242609fa", "54f2eec69"],
    4: ["26dc41664", "b9a3865fc", "e79de561c"],
}


@hydra.main(config_path='./config', config_name="default")
def train(cfg: DictConfig) -> None:    
    # logger
    logger = logging.getLogger(__name__)
    if not cfg.info.debug_mode:
        logger.critical("epoch, lr, train_loss, valid_loss, dice, dice_pos, dice_neg, best_score")

    # seed
    set_seed.seed_everything(cfg.train.seed)
    
    # fold
    test_ids = FOLD_IMGS[cfg.dataset.fold]
    train_ids = functools.reduce(lambda x,y: x + y, [v for k,v in FOLD_IMGS.items() if k != cfg.dataset.fold])

    if cfg.dataset.pseudo:
        train_ids = ['aa05346ff', '2ec3f1bb9', '3589adb90', 'd488c759a', '57512b7f1']

    if cfg.dataset.hand:
        train_ids += ['d488c759ahand']

    if cfg.dataset.crazy:
        train_ids = functools.reduce(lambda x,y: x + y, [v for k,v in FOLD_IMGS.items()]) + ['d488c759ahand']
        test_ids = functools.reduce(lambda x,y: x + y, [v for k,v in FOLD_IMGS.items() if k in [0,2,4]]) + ['d488c759ahand']

    print('train_size', len(train_ids), 'test_size', len(test_ids))

    train_ids *= cfg.dataset.train_size

    if cfg.info.debug_mode:
        test_ids = test_ids[:1]
        train_ids = train_ids[:2]
    
    # Dataset for train images
    train_dataset = locate(cfg.dataset.loader)(
        ids=train_ids,
        path_img=cfg.path.train_img,
        path_mask=cfg.path.train_mask,
        crop_dim = cfg.dataset.crop_dim,
        out_dim = cfg.dataset.out_dim,
        entropy_thr=cfg.train.entropy_thr,
        augmentation=locate(cfg.dataset.augs)(cfg.dataset.out_dim),
        preprocessing=locate(cfg.dataset.preprocessing)()
    )

    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, shuffle=True, 
                                                        num_workers=min(10,cfg.train.batch_size), 
                                                        batch_size=cfg.train.batch_size)}

    device = torch.device('cuda:0')
    model = smp.Unet(cfg.train.model, classes=1, activation=None, encoder_weights='imagenet')
    model = model.to(device)
    
    scaler = GradScaler()
    criterion = locate(cfg.train.loss)()
    optimizer = locate(cfg.train.optimizer)(model.parameters(), lr=cfg.train.lr)
    scheduler = locate(cfg.train.scheduler)(optimizer, mode='max', 
                                            factor=cfg.train.reduce_lr_factor,
                                            patience=cfg.train.reduce_lr_patience,  
                                            min_lr=cfg.train.reduce_lr_min,
                                            verbose=True)
    
    if cfg.path.pretrained_weights:
        model.load_state_dict(torch.load(cfg.path.pretrained_weights))

    if not cfg.info.debug_mode:
        writer = SummaryWriter(log_dir=cfg.path.tb_logger)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = 0.0
    early_stoping = 0
    checkpoints = deque(maxlen=5)  
    loss_updated = False
    
    for epoch in range(cfg.train.epochs):
        lr = optimizer.param_groups[0]['lr']
        if lr < 1e-4 and not loss_updated and cfg.train.loss_ft:
            criterion = locate(cfg.train.loss_ft)()
            loss_updated = True

        loss_train = train_loop.one_epoch(model, 
                                          criterion=criterion, 
                                          optimizer=optimizer, 
                                          dataloaders=dataloaders,
                                          scaler=scaler, 
                                          device=device)

        if cfg.train.zarr:
            results = locate(cfg.validation.type)(model, 
                                                test_ids=test_ids, 
                                                criterion=criterion,
                                                metrics=locate(cfg.validation.metrics),
                                                dataloader=locate(cfg.dataset.loader_test),
                                                path_img=cfg.path.train_img,
                                                crop_dim=cfg.dataset.crop_dim,
                                                out_dim=cfg.dataset.out_dim,
                                                preprocessing=locate(cfg.dataset.preprocessing)(),
                                                device=device,
                                                batch_size=cfg.train.batch_size)
        else:
            results = locate(cfg.validation.type)(model, 
                                                criterion=criterion, 
                                                metrics=locate(cfg.validation.metrics),
                                                entropy_thr=cfg.validation.entropy_thr,
                                                test_ids=test_ids, 
                                                fold=cfg.dataset.fold,
                                                path_img=cfg.path.train_img,
                                                path_mask=cfg.path.train_mask,
                                                crop_dim = cfg.dataset.crop_dim,
                                                out_dim = cfg.dataset.out_dim, 
                                                device=device)

        loss_val, metrics_val, dice_pos, dice_neg = results
        scheduler.step(metrics_val)

        if not cfg.info.debug_mode:
            writer.add_scalar("Loss/train", loss_train, epoch)
            writer.add_scalar("Loss/valid", loss_val, epoch)
            writer.add_scalar("Dice/mean_valid", metrics_val, epoch)
            writer.add_scalar("Dice/pos_valid", dice_pos, epoch)
            writer.add_scalar("Dice/neg_valid", dice_neg, epoch)
            writer.add_scalar("Learning rate", lr, epoch)


        if metrics_val > best_metrics:
            if not cfg.info.debug_mode:
                logger.critical(f"{epoch}, {lr}, {loss_train:.3f}, {loss_val:.3f}, {metrics_val:.3f}, {dice_pos:.3f}, {dice_neg:.3f}, 1")
            best_metrics = metrics_val
            best_model_wts = copy.deepcopy(model.state_dict())
            #checkpoints.append(best_model_wts)
            torch.save(best_model_wts, cfg.path.weights)
            early_stoping = 0
        else:
            if not cfg.info.debug_mode:
                logger.critical(f"{epoch}, {lr}, {loss_train:.3f}, {loss_val:.3f}, {metrics_val:.3f}, {dice_pos:.3f}, {dice_neg:.3f}, 0")
            early_stoping += 1
        
        if early_stoping > cfg.train.early_stop_patience:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val metrics: {:4f}'.format(best_metrics))
    

    #model, checkpoints = train_model(model, test_ids, criterion, optimizer_ft, dataloaders,
    #                                    dataset_sizes, lr_scheduler,
    #                                    num_epochs=EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE)

    #best_weights = find_best_weights(model, train_ids[:8], checkpoints)
    #best_weight = average_weights(best_weights)
    #torch.save(best_weight, PATH_WEIGTS)

if __name__ == "__main__":
    train()