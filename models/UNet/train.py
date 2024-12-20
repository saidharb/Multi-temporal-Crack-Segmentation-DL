# PyTorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# To track training progress
import time

# Torchvision
import torchvision.transforms.v2 as transforms
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For handling arrays
import numpy as np
import pandas as pd

# From other files
from dataset import CrackDataset, EarlyStopping
from unet import UNet
from metrics import SegmentationRunningScore
from utils import save_checkpoint, save_logs
from augmentations import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomColorJitter, RandomRotation, AddGaussianNoise, RandomBlur

def get_model(train_set, 
              val_set,
              train = False, 
              n_classes = 1, 
              batch_size = 16, 
              learning_rate = 0.001, 
              num_epochs = 100, 
              sigmoid_threshold = 0.5, 
              verbose = True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        
    model = UNet(n_classes).to(device)
    if train: 
        time_in_mins = 0.0
        if verbose:
            print("Training started")
        model.train()
        trainloader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
        valloader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='min', 
                                      factor=0.1, 
                                      patience=10, 
                                      threshold=0.0001, 
                                      cooldown=0, 
                                      min_lr=1e-6)
        early_stopping = EarlyStopping(patience = 30, delta = 0.0)
        logs = pd.DataFrame({"Epoch": pd.Series(dtype='int'),
                             "Learning Rate": pd.Series(dtype='float'),
                             "Train Loss": pd.Series(dtype='float'),
                             "Train Crack IoU": pd.Series(dtype='float'),
                             "Train Background IoU": pd.Series(dtype='float'),
                             "Train Crack Dice": pd.Series(dtype='float'),
                             "Train Background Dice": pd.Series(dtype='float'),
                             "Val Loss": pd.Series(dtype='float'),
                             "Val Crack IoU": pd.Series(dtype='float'),
                             "Val Background IoU": pd.Series(dtype='float'),
                             "Val Crack Dice": pd.Series(dtype='float'),
                             "Val Background Dice": pd.Series(dtype='float')
                            })
        
        score_train = SegmentationRunningScore(2)
        score_val = SegmentationRunningScore(2)
        best_val_crack_iou = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            train_running_loss = 0.0
            score_train.reset()
            for i, batch in enumerate(trainloader):
                images, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss_train = criterion(outputs, targets)
                loss_train.backward()
                optimizer.step()
                train_running_loss += loss_train.item()
                
                targets = targets.cpu().numpy().astype(np.uint8)
                outputs = torch.sigmoid(outputs) > sigmoid_threshold
                outputs = outputs.int().cpu().numpy().astype(np.uint8)

                score_train.update(targets, outputs)
               
                if verbose:
                    batch_iou, batch_dice = score_train.batch_metrics(targets, outputs)
                    batch_iou_no_crack = batch_iou[0]
                    batch_iou_crack = batch_iou[1]
                    batch_dice_no_crack = batch_dice[0]
                    batch_dice_crack = batch_dice[1]
                    
                    print("Epoch {}, Batch {}/{}, loss: {}, iou_1: {}, iou_0: {}, dice_1: {}, dice_0: {}".format(epoch+1, 
                                                                                i+1, 
                                                                                len(trainloader), 
                                                                                round(train_running_loss,2),
                                                                                batch_iou_crack,
                                                                                batch_iou_no_crack,
                                                                                batch_dice_crack,
                                                                                batch_dice_no_crack),
                          end = '\n',
                          flush = True)
        
                



            model.eval()
            val_running_loss = 0.0
            score_val.reset()
            with torch.no_grad():
                for i, batch in enumerate(valloader):
                    images, targets = batch[0].to(device), batch[1].to(device)
                    outputs = model(images)
                    loss_val = criterion(outputs, targets)
                    val_running_loss += loss_val.item()

                    targets = targets.cpu().numpy().astype(np.uint8)
                    outputs = torch.sigmoid(outputs) > sigmoid_threshold
                    outputs = outputs.int().cpu().numpy().astype(np.uint8)

                    score_val.update(targets, outputs)
                    
                if verbose:
                    batch_iou, _, batch_dice, _ = score_val.get_scores()
                    batch_iou_no_crack = batch_iou[0]
                    batch_iou_crack = batch_iou[1]
                    batch_dice_no_crack = batch_dice[0]
                    batch_dice_crack = batch_dice[1]
                    
                    print("Epoch {} FINNISHED, loss: {}, val_iou_1: {}, val_iou_0: {}, val_dice_1: {}, val_dice_0: {}".format(
                        epoch+1, 
                        round(val_running_loss,2),
                        batch_iou_crack,
                        batch_iou_no_crack,
                        batch_dice_crack,
                        batch_dice_no_crack),
                        end = '\n',
                        flush = True)
                
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = train_running_loss/len(trainloader)
            val_loss = train_running_loss/len(valloader)

            train_iou, _, train_dice, _ =score_train.get_scores()
            val_iou, _, val_dice, _ =score_val.get_scores()
            end_time = time.time()
            duration = (end_time-start_time)/60
            time_in_mins += duration
            
            new_epoch = pd.DataFrame({"Epoch": [epoch + 1],
                                    "Learning Rate": [current_lr],
                                    "Train Loss": [train_loss],
                                    "Train Crack IoU": [train_iou[1]],
                                    "Train Background IoU": [train_iou[0]],
                                    "Train Crack Dice": [train_dice[1]],
                                    "Train Background Dice": [train_dice[0]],
                                    "Val Loss": [val_loss],
                                    "Val Crack IoU": [val_iou[1]],
                                    "Val Background IoU": [val_iou[0]],
                                    "Val Crack Dice": [val_dice[1]],
                                    "Val Background Dice": [val_dice[0]],
                                    "Time in min": [duration]})

            logs = pd.concat([logs, new_epoch], ignore_index=True)
            save_logs(logs, epoch, sigmoid_threshold)

            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Saving the best model
            print(val_iou[1])
            print(best_val_crack_iou)
            if val_iou[1] > best_val_crack_iou:
                if verbose:
                    print("Validation mIoU of class crack increased from {} to {}.".format(round(best_val_crack_iou, 5), round(val_iou[1], 5)))
                best_val_crack_iou = val_iou[1]
                save_checkpoint(model, optimizer, epoch, logs, sigmoid_threshold)
            else:
                print("Validation mIoU of class crack did not increased from {}.".format(round(best_val_crack_iou, 5)))
            # Early Stopping
            early_stopping(val_iou[1])
            if early_stopping.early_stop:
                print("Early stopping, val crack iou  did not decrease more than {} from {} in epoch {}.".format(early_stopping.delta,
                                                                                                          early_stopping.best_iou,
                                                                                                          epoch + 1 - early_stopping.counter))
                break
            
            
        
        
        if verbose:
            print("Training time: {} minutes".format(round(time_in_mins)))
    else:
        model.eval()
        
    return model
