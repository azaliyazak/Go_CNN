import torch.nn.functional as F
import torch
from const import *
import numpy as np


class GOCNN(torch.nn.Module):
    def __init__(self):
        super(GOCNN, self).__init__()
        
        self.conv_1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm_1 = torch.nn.BatchNorm2d(16)
        
        self.conv_3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm_2 = torch.nn.BatchNorm2d(64)
        
        self.conv_5 = torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.norm_3 = torch.nn.BatchNorm2d(96)
        
        self.conv_6 = torch.nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.norm_4 = torch.nn.BatchNorm1d(128 * 4 * 4)
        
        self.lin_1 = torch.nn.Linear(128 * 4 * 4, 30 * 30)
        self.norm_5 = torch.nn.BatchNorm1d(30 * 30)
        self.lin_2 = torch.nn.Linear(30 * 30, 15 * 15)

    def forward(self, x):
        x = x.view(-1, 3, 16, 16)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.pool(x)
        x = self.norm_1(x)
        x = F.relu(self.conv_3(x))
        x = self.pool(x)
        x = F.relu(self.conv_4(x))
        x = self.norm_2(x)
        x = F.relu(self.conv_5(x))
        x = self.norm_3(x)
        x = F.relu(self.conv_6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.norm_4(x)
        x = F.relu(self.lin_1(x))
        x = self.norm_5(x)
        x = self.lin_2(x)
        return (x)


def decode(x):
    number = int(x / 15)+1
    letter = (x % 15)+1
    return f'{dig_to_let[letter]}{number}'


def predict(net, x):
    return net(torch.from_numpy(np.array([x])).float()).data[0]
