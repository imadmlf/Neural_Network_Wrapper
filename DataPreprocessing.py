import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

# This class is used to preprocess the data before it is fed into the neural network
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

class DataPreprocessing():
    def __init__(self, data):
        self.data = data

    def split_data(self, test_size=0.2, random_state=42):
        # Assuming the last column is the target variable
        x = self.data.drop(self.data.columns[-1], axis=1)
        y = self.data[self.data.columns[-1]]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def normalize_data(self):
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        return self.x_train, self.x_test

    def standardize_data(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        return self.x_train, self.x_test

    def tensorize_data(self):
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float32).view(-1, 1) 
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32).view(-1, 1) 
        return self.x_train, self.x_test, self.y_train, self.y_test














# class DataPreprocessing():
#     def __init__(self, data):
#         self.data = data
        
#     def SplitData(self,data,test_size=0.2, random_state=42):    
#         self.x = data.drop(data.columns[-1], axis=1)
#         self.y = data[data.columns[-1]]
#         self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
#         return self.x_train, self.x_test, self.y_train, self.y_test
    
#     def NormalizeData(self, x_train, x_test):
#         self.scaler = MinMaxScaler()
#         self.train_data = self.scaler.fit_transform(x_train)
#         self.test_data = self.scaler.transform(x_test)
#         return self.train_data, self.test_data
    
#     def StandardizeData(self, x_train, x_test):
#         self.scaler = StandardScaler()
#         self.train_data = self.scaler.fit_transform(x_train)
#         self.test_data = self.scaler.transform(x_test)
#         return self.train_data, self.test_data
    
#     def TensorizeData(self,train_data, test_data, y_train, y_test):
#         self.train_data = torch.tensor(train_data,dtype=torch.float32) 
#         self.test_data = torch.tensor(test_data,dtype=torch.float32)
#         self.y_train = torch.tensor(y_train.values,dtype=torch.float32)
#         self.y_test = torch.tensor(y_test.values,dtype=torch.float32)
#         return self.train_data, self.test_data, self.y_train, self.y_test
    
#     '''NOTE:
#     The following were not implemented
#     view(-1,1) is used to reshape the tensor to a 2D tensor.
#     device is used to move the tensor to the GPU if available.'''