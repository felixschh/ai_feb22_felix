import torch
import torch.nn.functional as F
from torch import nn, softmax
from torchsummary import summary

class Network(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.fc1 = nn.Linear(input, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        layer1 = self.fc1(x)
        act1 = F.relu(layer1)
        layer2 = self.fc2(act1)
        act2 = F.relu(layer2)
        layer3 = self.fc3(act2)
        act3 = F.relu(layer3)
        layer4 = self.fc4(act3)
        return layer4








        