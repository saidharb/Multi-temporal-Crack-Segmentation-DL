# For handling arrays
import numpy as np
import torch
import os
from torch.utils.data import DataLoader


class SegmentationRunningScore(object): 
    def __init__(self, n_classes): 
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
    def iou(self):
        hist = self.confusion_matrix 
        true_positives = np.diag(hist) 
        false_positives=[]
        false_negatives=[] 
        false_negative_sum=np.sum(hist,axis=1) 
        false_positive_sum=np.sum(hist,axis=0) 
        for i in range (self.n_classes):
            false_negatives.append(false_negative_sum[i]-hist[i,i]) 
        for i in range (self.n_classes):
           false_positives.append(false_positive_sum[i]-hist[i,i])
        class_iou = np.zeros(2,)
        class_dice = np.zeros(2,)
        #print(f"TP:{true_positives}")
        #print(f"FP:{false_positives}")
        #print(f"FN:{false_negatives}")
        if true_positives[1] == 0: #Case if there are no cracks in the image
            class_iou[0] = true_positives[0] / (true_positives[0] + false_positives[0] + false_negatives[0])
            class_iou[1] = 0
            class_dice[0] = 2 * true_positives[0] / (2 * true_positives[0] + false_positives[0] + false_negatives[0])
            class_dice[1] = 0
        else:
            class_iou = true_positives / (true_positives + false_positives + false_negatives)
            class_dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
            
        class_iou = np.nan_to_num(class_iou, nan=-1) # Convert NaN to -1
        valid_indices = np.where(class_iou >= 0)[0]
        valid_iou = class_iou[valid_indices]
        mean_iou = np.mean(valid_iou) if valid_iou.size > 0 else 0.0
        class_iou_dict = {trainid: iou for trainid, iou in zip(valid_indices, valid_iou)}

        class_dice = np.nan_to_num(class_dice, nan=-1) # Convert NaN to -1
        valid_indices = np.where(class_dice >= 0)[0]
        valid_dice = class_dice[valid_indices]
        mean_dice = np.mean(valid_dice) if valid_dice.size > 0 else 0.0
        class_dice_dict = {trainid: dice for trainid, dice in zip(valid_indices, valid_dice)}
        return class_iou_dict, mean_iou, class_dice_dict, mean_dice
        
    def fast_hist(self, label_true, label_pred): 
        mask=(label_true>=0)&(label_true<self.n_classes) 
        hist=np.bincount(
            self.n_classes*label_true[mask].astype(int)+label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes) 
        return hist

    def batch_metrics(self, labels_true, labels_pred):
        label_true = labels_true.flatten()
        label_pred = labels_pred.flatten()
        mask=(label_true>=0)&(label_true<self.n_classes)
        hist=np.bincount(
            self.n_classes*label_true[mask].astype(int)+label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes) 
        true_positives = np.diag(hist) 
        false_positives=[]
        false_negatives=[] 
        false_negative_sum=np.sum(hist,axis=1) 
        false_positive_sum=np.sum(hist,axis=0) 
        for i in range (self.n_classes):
            false_negatives.append(false_negative_sum[i]-hist[i,i]) 
        for i in range (self.n_classes):
           false_positives.append(false_positive_sum[i]-hist[i,i])
        class_iou = np.zeros(2,)
        class_dice = np.zeros(2,)
        if true_positives[1] == 0: #Case if there are no cracks in the image or the crack did not get detected
            class_iou[0] = true_positives[0] / (true_positives[0] + false_positives[0] + false_negatives[0])
            class_iou[1] = 0
            class_dice[0] = 2 * true_positives[0] / (2 * true_positives[0] + false_positives[0] + false_negatives[0])
            class_dice[1] = 0
        else:
            class_iou = true_positives / (true_positives + false_positives + false_negatives)
            class_dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        return class_iou, class_dice
        
    def update(self, label_trues, label_preds): 
        self.confusion_matrix += self.fast_hist(label_trues.flatten(),label_preds.flatten())
    def get_scores(self):
        iou, mean_iou, dice, mean_dice = self.iou() 
        return iou, mean_iou, dice, mean_dice
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def calculate_metrics(target_path, prediction_path):
    
    target = Image.open(target_path)
    target = np.array(target, dtype = np.uint8)
    prediction = Image.open(prediction_path)
    prediction = np.array(prediction, dtype = np.uint8)
    
    start = time.time()
    precision = precision_score(target.flatten(), prediction.flatten())
    recall = recall_score(target.flatten(), prediction.flatten())
    f1 = f1_score(target.flatten(), prediction.flatten())
    
    target_inverted = 1 - target
    prediction_inverted = 1 - prediction
    
    precision_bg = precision_score(target_inverted.flatten(), prediction_inverted.flatten())
    recall_bg = recall_score(target_inverted.flatten(), prediction_inverted.flatten())
    f1_bg = f1_score(target_inverted.flatten(), prediction_inverted.flatten())

    model_score = SegmentationRunningScore(2)
    model_score.update(target, prediction)
    iou, _, dice, _ = model_score.get_scores()

    data = {
        'Precision': [precision_bg, precision],
        'Recall': [recall_bg, recall],
        'F1-Score': [f1_bg, f1],
        'IoU': [iou[0], iou[1]],
        'Dice': [dice[0], dice[1]]
    }
    df = pd.DataFrame(data, index=['Background', 'Crack'])
    duration = time.time() - start
    print("Duration: {}s".format(round(duration)))
    return df

    
def calculate_batch_metrics(predictions, targets):

    predictions = predictions.view(-1)
    targets = targets.view(-1)
    TP = (predictions * targets).sum().item()
    FP = (predictions * (1 - targets)).sum().item()
    FN = ((1 - predictions) * targets).sum().item()
    
    return TP, FP, FN
