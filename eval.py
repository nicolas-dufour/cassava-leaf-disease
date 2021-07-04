import torch

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import .models.vit as vit
import .models.deit as deit
from .DataLoader import get_dataloader

n_classes = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cv_predictions(model_class,models_paths,image_path,df):
    loader = get_dataloader(image_path,df,batch_size=16,train=False)
    predictions = torch.zeros((len(dataset),len(models_paths),n_classes),device = device)
    for i, model_path in enumerate(models_paths):
        print(f"Fold n {i}")
        model = model_class.load_from_checkpoint(model_path,max_epochs=0).to(device)
        model.eval()
        with torch.no_grad():
            last_batch = 0
            for j, batch in enumerate(tqdm(loader)):
                images= batch[0].to(device)
                batch_len = images.size(0)
                logits = F.softmax(model(images),dim=1)
                predictions[last_batch:last_batch+batch_len,i] = logits
                last_batch = last_batch+batch_len
        del model
        torch.cuda.empty_cache()
    return predictions

def get_cv_predictions_distil(model_class,models_paths,image_path,df):
    loader = get_dataloader(image_path,df,batch_size=16,train=False)
    prediction_teacher = torch.zeros((len(dataset),len(models_paths),5),device = device)
    prediction_student = torch.zeros((len(dataset),len(models_paths),5),device = device)
    prediction = torch.zeros((len(dataset),len(models_paths),5),device = device)
    for i, model_path in enumerate(models_paths):
        print(f"Fold n {i}")
        model = model_class.load_from_checkpoint(model_path,max_epochs=0).to(device)
        model.eval()
        with torch.no_grad():
            last_batch = 0
            for j, batch in enumerate(tqdm(loader)):
                images= batch[0].to(device)
                batch_len = images.size(0)
                logits_teacher, logits_student = model(images)
                logits  = (logits_teacher+logits_student)/2

                logits_teacher = F.softmax(logits_teacher,dim=1)
                logits_student = F.softmax(logits_student,dim=1)
                logits = F.softmax(logits,dim=1)

                prediction[last_batch:last_batch+batch_len,i] = logits
                prediction_teacher[last_batch:last_batch+batch_len,i] = logits_teacher
                prediction_student[last_batch:last_batch+batch_len,i] = logits_student
                last_batch = last_batch+batch_len
        del model
        torch.cuda.empty_cache()
    return prediction_teacher, prediction_student, prediction

def mean_aggregate_prediction(predictions):
    return predictions.mean(dim=1).argmax(dim=1).cpu().numpy()

def __main__():
    train_path =base_path + '/train_images/'
    train_csv = pd.read_csv(base_path + '/train.csv')

    np.random.seed(42)
    msk = np.random.rand(len(train_csv)) < 0.9

    test_df = train_csv[~msk]
    models_paths = [f"./models_ckpt/hard-distiled-deit-split-{i}" for i in range(5)]

    prediction_teacher, prediction_student, prediction = get_cv_predictions_distil(deit.HardDistilledDeitNet,models_paths,train_path,test_df)

    prediction_teacher, prediction_student, prediction = mean_aggregate_prediction(prediction_teacher), mean_aggregate_prediction(prediction_student), mean_aggregate_prediction(prediction)

    print(f"Teacher accuracy: {(prediction_teacher==test_df['label']).mean()}")
    print(f"Student accuracy: {(prediction_student==test_df['label']).mean()}")
    print(f"Both accuracy: {(prediction==test_df['label']).mean()}")
