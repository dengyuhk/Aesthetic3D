import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import torch.nn.functional as F
import math


class DatasetPair(Dataset):
    def __init__(self, pairs, data):
        self.data = data
        self.pairs = pairs

    def __getitem__(self, index):
        idx1 = self.pairs[index][0]
        idx2 = self.pairs[index][1]
        shape1 = self.data[idx1]
        shape2 = self.data[idx2]
        margin_w = torch.tensor([self.pairs[index][2]])
        return shape1, shape2, margin_w

    def __len__(self):
        return len(self.pairs)


class DatasetSingle(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def weighted_loss(s1, s2, margin_w, margin):
    temp = margin_w * margin - (s1 - s2)
    loss = torch.clamp(temp, min=0)
    return torch.mean(loss)


class ModelImg1(nn.Module):
    def __init__(self):
        super(ModelImg1, self).__init__()

        # input 128*128
        self.viewCNN = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )

        self.shapeCNN = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        B, _, _, W, H = x.shape
        x = x.view(B * 12, 1, W, H)
        x = self.viewCNN(x)
        _, C, W, H = x.shape
        x = x.view(B, 12, C, W, H)
        x = torch.max(x, 1)[0]
        x = self.shapeCNN(x)
        return x


class ModelImg2(nn.Module):
    def __init__(self):
        super(ModelImg2, self).__init__()

        # input 128*128
        self.viewCNN = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64)
        )

        self.shapeCNN = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        B, _, _, W, H = x.shape
        x = x.view(B * 12, 1, W, H)
        x = self.viewCNN(x)
        _, C, W, H = x.shape
        x = x.view(B, 12, C, W, H)
        x = torch.max(x, 1)[0]
        x = self.shapeCNN(x)
        return x


class ModelImg3(nn.Module):
    def __init__(self, cfg):
        super(ModelImg3, self).__init__()

        self.viewCNN = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64)
        )

        self.shapeCNN = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256 * (cfg.input_dims // 128)**2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        B, _, _, W, H = x.shape
        x = x.view(B * 12, 1, W, H)
        x = self.viewCNN(x)
        _, C, W, H = x.shape
        x = x.view(B, 12, C, W, H)
        x = torch.max(x, 1)[0]
        x = self.shapeCNN(x)
        return x
