#https://stackoverflow.com/questions/53354176/how-to-use-f-score-as-error-function-to-train-neural-networks
import torch
import torch.nn as nn
import torch.nn.functional as F

def f1_loss(predict, target):
    loss = 0
    # lack_cls = target.sum(dim=0) == 0
    # if lack_cls.any():
    #     loss += F.binary_cross_entropy_with_logits(
    #         predict[:, lack_cls], target[:, lack_cls])
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean() + nn.BCEWithLogitsLoss()(predict, target)#+ loss