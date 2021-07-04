import torch

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import .models.vit as vit
import .models.deit as deit
from .DataLoader import get_dataloader

batch_size = 16

def train_K_splits(model_class,train_path,train_df,test_path,test_df,K, wandb_name,ckpt_path,seed=55, wandb_project = "cassava-leaf-disease"):
    max_epochs = 10
    test_loader = get_dataloader(test_path, test_df, batch_size,num_workers=8,train=False)
    n_samples = len(train_df)
    kf = StratifiedKFold(n_splits=K,random_state=seed)
    splits = kf.split(train_df,train_df['label'])
    for i, (train_split, val_split) in enumerate(splits):
        print(f"Spit number {i}")
        train_loader  = get_dataloader(train_path, train_df.iloc[train_split], batch_size,num_workers=8,train=True)
        val_loader  = get_dataloader(train_path, train_df.iloc[val_split], batch_size,num_workers=8,train=False)
        wandb.init(project=wandb_project, name=wandb_name+f" split {i}")
        model =model_class(max_epochs)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=ckpt_path,
            filename=wandb_name+f"-split-{i}",
            save_top_k=1,
            mode='max')

        lr_logger_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

        wandb_logger = WandbLogger(project=wandb_project)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                    checkpoint_callback,
                    lr_logger_callback
                    ],
            gpus=1,
            logger = wandb_logger,
            precision =16
            )
        trainer.fit(model, train_loader,val_loader)
        wandb.save(trainer.checkpoint_callback.best_model_path)
        test_acc = trainer.test(model,test_loader)

        del model, trainer
        torch.cuda.empty_cache()

def __main__():
    base_path = 'data'

    train_path =base_path + '/train_images/'
    train_csv = pd.read_csv(base_path + '/train.csv')

    np.random.seed(42)
    msk = np.random.rand(len(train_csv)) < 0.9
    train_df = train_csv[msk]
    test_df = train_csv[~msk]
    
    train_K_splits(deit.HardDistilledDeitNet,train_path,train_df,train_path,test_df,5, "hard-distiled-deit",'./models_ckpt/',seed=55, wandb_project = "cassava-leaf-disease")
