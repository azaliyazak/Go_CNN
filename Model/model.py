import torch.nn.functional as F
import torch
from const import *
import numpy as np


class GOCNN(torch.nn.Module):
    def __init__(self):
        super(GOCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm1 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.norm3 = torch.nn.BatchNorm2d(96)
        self.conv6 = torch.nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.norm4 = torch.nn.BatchNorm1d(128 * 4 * 4)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 30 * 30)
        self.norm5 = torch.nn.BatchNorm1d(30 * 30)
        self.fc2 = torch.nn.Linear(30 * 30, 15 * 15)

    def forward(self, x):
        x = x.view(-1, 3, 16, 16)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.norm2(x)
        x = F.relu(self.conv5(x))
        x = self.norm3(x)
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.norm4(x)
        x = F.relu(self.fc1(x))
        x = self.norm5(x)
        x = self.fc2(x)
        return (x)


def decode(x):
    number = int(x / 15)+1
    letter = (x % 15)+1
    return f'{dig_to_let[letter]}{number}'


def predict(net, x):
    return net(torch.from_numpy(np.array([x])).float()).data[0]
