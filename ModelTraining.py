import torch
from torch import nn
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from DataPreprocessing import *


class ModelTraining:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    
    def train_model(self, train_loader, test_loader, epochs=100):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for data, targets in train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                outputs = output.squeeze(1) 
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            test_loss = self.test_model(test_loader)
            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {test_loss:.4f}')
        
        self.plot_loss(train_losses, test_losses)
    
    def test_model(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, targets in data_loader:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)
        print(f'Model saved as {file_name}')
    
    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.model(data)
    
    def plot_loss(self, train_losses, test_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()














# model = NeuralNetwork()
# criterion = nn.BCELoss()  # Utilise la perte binary cross-entropy pour la classification binaire
# optimizer = torch.optim.Adam(model.parameters())  # Utilise l'optimiseur Adam









# class ModelTraining(NeuralNetwork):
#     def __init__(self, model, criterion, optimizer):
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
'''     
     def TrainModel(self, train_data, y_train,y_test, epochs=100):
         train_losses = []
         test_losses = []
         for epoch in range(epochs):
             self.model.train()
             self.optimizer.zero_grad()
             outputs = self.model(train_data)
             loss = self.criterion(outputs, y_train)
             #Effectue la rétropropagation pour calculer les gradients de la perte par rapport aux paramètres du modèle.
             loss.backward()
             self.optimizer.step()
             self.model.eval()
             with torch.no_grad():
                 test_outputs = self.model(train_data)
                 test_loss = self.criterion(test_outputs, y_test)
             train_losses.append(loss.item())
             test_losses.append(test_loss.item())

             print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {test_loss.item():.4f}')

    '''
#     def TestModel(self, test_data, y_test):
#         with torch.no_grad():
#             y_pred = self.model(test_data)
#             loss = self.criterion(y_pred, y_test)
#         return loss.item()
    
#     def SaveModel(self, model, file_name):
#         torch.save(model.state_dict(), file_name)
#         return f'Model saved as {file_name}'
    
#     def LoadModel(self, file_name):
#         self.model.load_state_dict(torch.load(file_name))
#         return self.model
    
#     def Predict(self, model, test_data):
#         with torch.no_grad():
#             y_pred = model(test_data)
#         return y_pred
    
#     def PlotLoss(self, train_losses, test_losses):
#         plt.plot(train_losses, label='training loss')
#         plt.plot(test_losses,label='testing loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.show()    
    
