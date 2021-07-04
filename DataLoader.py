import torch    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import PIL
from PIL import Image, ImageOps, ImageEnhance
import cv2
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as AF

class ADataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            dataframe,
            transform= None,
            augmix = False):
        self.df = dataframe
        self.transform = transform
        self.img_path = path
        self.augmix = augmix
        

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        label = self.df.iloc[index].label
        p = self.df.iloc[index].image_id
        p_path = self.img_path + p
        
        image =  Image.open(p_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

    def __len__(self):
        return len(self.df)


data_transforms_train = A.Compose([
        A.CoarseDropout(),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(p=0.3),
        A.RandomCrop(384, 384,p=0.5),
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],max_pixel_value=255.0, p=1),

        ToTensorV2()                              
])

data_transforms_val = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],max_pixel_value=255.0, p=1),
        ToTensorV2()
                                  
])


def get_dataloader(img_path, labels_df, batch_size,num_workers=8,train=True):
    if train:
        transform = data_transforms_train
        shuffle = True
    else: 
        transform = data_transforms_val
        shuffle = False
    dataset = ADataset(
        img_path,
        labels_df,
        transform = transform
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
        )
    return loader

