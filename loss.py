import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma, loss_weights):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = loss_weights

    def forward(self, pred, target):
        pred = nn.Softmax(pred)