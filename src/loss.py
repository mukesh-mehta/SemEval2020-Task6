#https://stackoverflow.com/questions/53354176/how-to-use-f-score-as-error-function-to-train-neural-networks
import torch
import torch.nn as nn
import torch.nn.functional as F

def f1_loss(predict, target, weights=None):
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean() + nn.BCEWithLogitsLoss(weight=weights)(predict, target)