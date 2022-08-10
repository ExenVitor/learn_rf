#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv2d_size_out

__author__ = 'Vitor Chen'
__email__ = "exen3995@gmail.com"
__version__ = "0.1.0"


class DQN(nn.Module):

    def __init__(self, c: int, h: int, w: int, outputs: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        conv_w = conv2d_size_out(size=w, kernel_size=8, stride=4)
        conv_h = conv2d_size_out(size=h, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        conv_w = conv2d_size_out(size=conv_w, kernel_size=4, stride=2)
        conv_h = conv2d_size_out(size=conv_h, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        conv_w = conv2d_size_out(size=conv_w, kernel_size=3, stride=1)
        conv_h = conv2d_size_out(size=conv_h, kernel_size=3, stride=1)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5)
        # self.bn3 = nn.BatchNorm2d(32)
        # conv_w = conv2d_size_out(size=conv_w, kernel_size=5)
        # conv_h = conv2d_size_out(size=conv_h, kernel_size=5)

        linear_input_size = conv_w * conv_h * 64

        self.fc1 = nn.Linear(in_features=linear_input_size, out_features=512)
        # self.dropout1 = nn.Dropout()
        self.head = nn.Linear(in_features=512, out_features=outputs)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)
