import torch
import torch.nn.functional as F
from torch import nn

class Classify(nn.Module):
    def __init__(self, input):
        super(Classify, self).__init__()
        self.input_layer = nn.Linear(input, 400)  
        self.hidden1 = nn.Linear(400, 200) 
        self.hidden2 = nn.Linear(200, 100) 
        self.hidden3 = nn.Linear(100, 75)
        self.hidden4 = nn.Linear(75, 50)
        self.output = nn.Linear(50, 10) 

    
    def forward(self, x):
        first_layer = self.input_layer(x)
        act1 = F.relu(first_layer)
        second_layer = self.hidden1(act1)
        act2 = F.relu(second_layer)
        third_layer = self.hidden2(act2)
        act3 = F.relu(third_layer)
        fourth_layer = self.hidden3(act3)
        act4 = F.relu(fourth_layer)
        third_layer = self.hidden4(act4)
        act5 = F.relu(third_layer)
        out_layer = self.output(act5)
        x = F.softmax(out_layer, dim=1)
        return out_layer
