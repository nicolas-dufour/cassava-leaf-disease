import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, models
from torch.autograd import Variable

import numpy as np
import timm
import wandb

import pytorch_lightning as pl


nclasses = 5 

class VitNet(pl.LightningModule):
    def __init__(self, max_epochs=10):
        super(VitNet, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        num_ftrs = list(self.vit.children())[-1].out_features

        self.vit.classifier = nn.Identity()
        self.fc = nn.Linear(num_ftrs,nclasses)

        self.lr = 1.5e-5
        self.max_epochs = max_epochs

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.vit(x)
        return self.fc(x)

    def embedding(self,x):
        return self.vit(x)   

    def training_step(self,batch, batch_idx):
        data,target = batch
        logits = self(data.float())

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, target)

        self.log("training_loss_step", loss)
        acc = self.train_acc(logits,target)
        return {'loss': loss, "n_pred": len(target)}

    def training_epoch_end(self, outputs):
        n_features =  torch.Tensor([x['n_pred'] for x in outputs]).sum()
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum()/n_features
        self.log("train_loss",avg_loss,logger=True)
        self.log("train_acc",self.train_acc.compute(),logger=True)

    def validation_step(self, batch, batch_idx):
        data,target = batch
        logits = self(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, target)
        acc = self.val_acc(logits, target)
        return {'val_loss': loss, "n_pred": len(target)}

    def validation_epoch_end(self, outputs):
        n_features =  torch.Tensor([x['n_pred'] for x in outputs]).sum()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()/n_features
        self.log("val_loss", avg_loss,logger=True)
        self.log("val_acc", self.val_acc.compute(),logger=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr,weight_decay=0.01)
        scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=1.5e-5, epochs=self.max_epochs, steps_per_epoch=1338)
        return {'optimizer': optimizer,'interval': 'step','lr_scheduler':{'scheduler':scheduler,'interval': 'step'}}