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

class DeitNet(pl.LightningModule):
    def __init__(self, max_epochs=10):
        super(DeitNet, self).__init__()
        self.deit = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        num_ftrs = list(self.deit.children())[-1].in_features

        self.deit.head = nn.Identity()
        self.deit.head_dist = nn.Identity()
        self.fc = nn.Linear(num_ftrs,nclasses)

        self.lr = 1e-5
        self.max_epochs = max_epochs

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        if self.training:
            x, _ = self.deit(x)
        else:
            x= self.deit(x)
        return self.fc(x)

    def embedding(self,x):
        return self.deit(x)

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
        scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=1e-5, epochs=self.max_epochs, steps_per_epoch=1338)
        return {'optimizer': optimizer,'interval': 'step','lr_scheduler':{'scheduler':scheduler,'interval': 'step'}}



class SoftCrossEntropy(nn.Module):
    def __init__(self, dim=-1):
        super(SoftCrossEntropy, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(-torch.sum(target * pred, dim=self.dim))


class SoftDistilledDeitNet(pl.LightningModule):
    def __init__(self, max_epochs=10,loss_weights=None):
        super(SoftDistilledDeitNet, self).__init__()
        self.deit = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        num_ftrs = list(self.deit.children())[-1].in_features

        self.deit.head = nn.Identity()
        self.deit.head_dist = nn.Identity()
        self.fc = nn.Linear(num_ftrs,nclasses)
        self.fc_distil = nn.Linear(num_ftrs,nclasses)

        self.lr = 1e-5
        self.max_epochs = max_epochs
        self.loss_weights = loss_weights

        self.train_acc_teacher = pl.metrics.Accuracy()
        self.train_acc_student = pl.metrics.Accuracy()
        self.train_acc = pl.metrics.Accuracy()

        self.val_acc_teacher = pl.metrics.Accuracy()
        self.val_acc_student = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

        self.test_acc_teacher = pl.metrics.Accuracy()
        self.test_acc_student = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
    def forward(self, x):
        if self.training:
            x, _ = self.deit(x)
        else:
            x= self.deit(x)
        return self.fc(x), self.fc_distil(x)

    def embedding(self,x):
        return self.deit(x)
  

    def training_step(self,batch, batch_idx):
        data,target = batch
        logits_teacher, logits_student = self(data.float())

        criterion_student = SoftCrossEntropy()
        criterion_teacher = torch.nn.CrossEntropyLoss(reduction='mean', weight = self.loss_weights)
        loss_teacher = criterion_teacher(logits_teacher, target)
        loss_student = criterion_student(logits_student, F.softmax(logits_teacher,dim=-1))
        loss = (loss_teacher+loss_student)/2

        self.log("training_loss_step", loss)
        self.train_acc_teacher(logits_teacher,target)
        self.train_acc_student(logits_student,target)
        self.train_acc((logits_teacher+logits_teacher)/2,target)
        return {'loss': loss,'loss_teacher':loss_teacher,'loss_student':loss_student, "n_pred": len(target)}

    def training_epoch_end(self, outputs):

        n_features =  torch.Tensor([x['n_pred'] for x in outputs]).sum()
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum()/n_features
        avg_loss_student = torch.stack([x['loss_student'] for x in outputs]).sum()/n_features
        avg_loss_teacher = torch.stack([x['loss_teacher'] for x in outputs]).sum()/n_features

        self.log("train_loss",avg_loss,logger=True)
        self.log("train_loss_teacher",avg_loss_teacher,logger=True)
        self.log("train_loss_student",avg_loss_student,logger=True)

        self.log("train_acc",self.train_acc_teacher.compute(),logger=True)
        self.log("train_acc_teacher",self.train_acc_teacher.compute(),logger=True)
        self.log("train_acc_student",self.train_acc_student.compute(),logger=True)

    def validation_step(self, batch, batch_idx):
        data,target = batch
        logits_teacher, logits_student = self(data.float())

        criterion_student = SoftCrossEntropy()
        criterion_teacher = torch.nn.CrossEntropyLoss(reduction='mean', weight = self.loss_weights)
        loss_teacher = criterion_teacher(logits_teacher, target)
        loss_student = criterion_student(logits_student, F.softmax(logits_teacher,dim=-1))
        loss = (loss_teacher+loss_student)/2

        self.val_acc_teacher(logits_teacher,target)
        self.val_acc_student(logits_student,target)
        self.val_acc((logits_teacher+logits_teacher)/2,target)
        return {'loss': loss,'loss_teacher':loss_teacher,'loss_student':loss_student, "n_pred": len(target)}

    def validation_epoch_end(self, outputs):
        n_features =  torch.Tensor([x['n_pred'] for x in outputs]).sum()
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum()/n_features
        avg_loss_student = torch.stack([x['loss_student'] for x in outputs]).sum()/n_features
        avg_loss_teacher = torch.stack([x['loss_teacher'] for x in outputs]).sum()/n_features

        self.log("val_loss",avg_loss,logger=True)
        self.log("val_loss_teacher",avg_loss_teacher,logger=True)
        self.log("val_loss_student",avg_loss_student,logger=True)

        self.log("val_acc",self.val_acc_teacher.compute(),logger=True)
        self.log("val_acc_teacher",self.val_acc_teacher.compute(),logger=True)
        self.log("val_acc_student",self.val_acc_student.compute(),logger=True)

    def test_step(self, batch, batch_idx):
        data,target = batch
        logits_teacher, logits_student = self(data.float())

        self.test_acc_teacher(logits_teacher,target)
        self.test_acc_student(logits_student,target)
        self.test_acc((logits_teacher+logits_teacher)/2,target)

    def test_epoch_end(self, outputs):
        test_acc = self.log("test_acc",self.test_acc_teacher.compute(),logger=True)
        self.log("test_acc_teacher",self.test_acc_teacher.compute(),logger=True)
        self.log("test_acc_student",self.test_acc_student.compute(),logger=True)
        return test_acc
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr,weight_decay=0.01)
        scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=1e-5, epochs=self.max_epochs, steps_per_epoch=1338)
        return {'optimizer': optimizer,'interval': 'step','lr_scheduler':{'scheduler':scheduler,'interval': 'step'}}


class HardDistilledDeitNet(pl.LightningModule):
    def __init__(self, max_epochs=10,loss_weights=None):
        super(HardDistilledDeitNet, self).__init__()
        self.deit = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        num_ftrs = list(self.deit.children())[-1].in_features

        self.deit.head = nn.Identity()
        self.deit.head_dist = nn.Identity()
        self.fc = nn.Linear(num_ftrs,nclasses)
        self.fc_distil = nn.Linear(num_ftrs,nclasses)

        self.lr = 1e-5
        self.max_epochs = max_epochs
        self.loss_weights = loss_weights

        self.train_acc_teacher = pl.metrics.Accuracy()
        self.train_acc_student = pl.metrics.Accuracy()
        self.train_acc = pl.metrics.Accuracy()

        self.val_acc_teacher = pl.metrics.Accuracy()
        self.val_acc_student = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

        self.test_acc_teacher = pl.metrics.Accuracy()
        self.test_acc_student = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
    def forward(self, x):
        if self.training:
            x, _ = self.deit(x)
        else:
            x= self.deit(x)
        return self.fc(x), self.fc_distil(x)

    def embedding(self,x):
        return self.deit(x)
  

    def training_step(self,batch, batch_idx):
        data,target = batch
        logits_teacher, logits_student = self(data.float())

        criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = self.loss_weights)
        loss_teacher = criterion(logits_teacher, target)
        logits_teacher_hard = logits_teacher.argmax(dim=1)
        loss_student = criterion(logits_student, F.softmax(logits_teacher,dim=-1))
        loss = (loss_teacher+loss_student)/2

        self.log("training_loss_step", loss)
        self.train_acc_teacher(logits_teacher,target)
        self.train_acc_student(logits_student,target)
        self.train_acc((logits_teacher+logits_teacher)/2,target)
        return {'loss': loss,'loss_teacher':loss_teacher,'loss_student':loss_student, "n_pred": len(target)}

    def training_epoch_end(self, outputs):

        n_features =  torch.Tensor([x['n_pred'] for x in outputs]).sum()
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum()/n_features
        avg_loss_student = torch.stack([x['loss_student'] for x in outputs]).sum()/n_features
        avg_loss_teacher = torch.stack([x['loss_teacher'] for x in outputs]).sum()/n_features

        self.log("train_loss",avg_loss,logger=True)
        self.log("train_loss_teacher",avg_loss_teacher,logger=True)
        self.log("train_loss_student",avg_loss_student,logger=True)

        self.log("train_acc",self.train_acc_teacher.compute(),logger=True)
        self.log("train_acc_teacher",self.train_acc_teacher.compute(),logger=True)
        self.log("train_acc_student",self.train_acc_student.compute(),logger=True)

    def validation_step(self, batch, batch_idx):
        data,target = batch
        logits_teacher, logits_student = self(data.float())

        criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = self.loss_weights)
        loss_teacher = criterion(logits_teacher, target)
        logits_teacher_hard = logits_teacher.argmax(dim=1)
        loss_student = criterion(logits_student, F.softmax(logits_teacher,dim=-1))
        loss = (loss_teacher+loss_student)/2

        self.val_acc_teacher(logits_teacher,target)
        self.val_acc_student(logits_student,target)
        self.val_acc((logits_teacher+logits_teacher)/2,target)
        return {'loss': loss,'loss_teacher':loss_teacher,'loss_student':loss_student, "n_pred": len(target)}

    def validation_epoch_end(self, outputs):
        n_features =  torch.Tensor([x['n_pred'] for x in outputs]).sum()
        avg_loss = torch.stack([x['loss'] for x in outputs]).sum()/n_features
        avg_loss_student = torch.stack([x['loss_student'] for x in outputs]).sum()/n_features
        avg_loss_teacher = torch.stack([x['loss_teacher'] for x in outputs]).sum()/n_features

        self.log("val_loss",avg_loss,logger=True)
        self.log("val_loss_teacher",avg_loss_teacher,logger=True)
        self.log("val_loss_student",avg_loss_student,logger=True)

        self.log("val_acc",self.val_acc_teacher.compute(),logger=True)
        self.log("val_acc_teacher",self.val_acc_teacher.compute(),logger=True)
        self.log("val_acc_student",self.val_acc_student.compute(),logger=True)

    def test_step(self, batch, batch_idx):
        data,target = batch
        logits_teacher, logits_student = self(data.float())

        self.test_acc_teacher(logits_teacher,target)
        self.test_acc_student(logits_student,target)
        self.test_acc((logits_teacher+logits_teacher)/2,target)

    def test_epoch_end(self, outputs):
        test_acc = self.log("test_acc",self.test_acc_teacher.compute(),logger=True)
        self.log("test_acc_teacher",self.test_acc_teacher.compute(),logger=True)
        self.log("test_acc_student",self.test_acc_student.compute(),logger=True)
        return test_acc
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr,weight_decay=0.01)
        scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=1e-5, epochs=self.max_epochs, steps_per_epoch=1338)
        return {'optimizer': optimizer,'interval': 'step','lr_scheduler':{'scheduler':scheduler,'interval': 'step'}}