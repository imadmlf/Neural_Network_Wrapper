import os
import torch
from torch import nn
from torch.utils.data import dataloader 
from torchvision import datasets, transforms
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from DataPreprocessing import *

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_features,out_features):
        super(NeuralNetwork, self).__init__()
        # Define the architecture of the network
        self.fc1 = nn.Linear(input_features, 30) 
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x





# class NeuralNetwork(nn.Module , DataPreprocessing):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(TensorizeData.train_data.shape[1], 30)
#         self.fc2 = nn.Linear(30, 15)
#         self.fc3 = nn.Linear(15, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.sigmoid(self.fc3(x))
#         return x
