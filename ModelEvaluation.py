import os
import torch
from torch import nn
from torch.utils.data import dataloader 
from torchvision import datasets, transforms
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from ModelTraining import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from ModelTraining import *

class ModelEvaluation(ModelTraining):
    def confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy())
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def classification_report(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))

    def accuracy_score(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        return accuracy

    def plot_loss(self, loss):
        plt.plot(loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.show()

    def plot_prediction(self, y_test, y_pred):
        plt.scatter(y_test, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. True Values')
        plt.show()

    def plot_prediction_error(self, y_test, y_pred):
        error = y_test - y_pred
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Prediction Error Distribution')
        plt.show()




# class ModelEvaluation(ModelTraining):
#     def __init__(self):
#         pass
#     def ConfusionMatrix(self, y_test, y_pred):
#         cm = confusion_matrix(y_test, y_pred)
#         sns.heatmap(cm, annot=True)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.show()
    
#     def ClassificationReport(self, y_test, y_pred):
#         print(classification_report(y_test, y_pred))

#     def AccuracyScore(self, y_test, y_pred):
#         print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

#     def PlotLoss(self, loss):
#         plt.plot(loss)
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.show()
        
#     def PlotPrediction(self, y_test, y_pred):
#         plt.scatter(y_test, y_pred)
#         plt.xlabel('True Values')
#         plt.ylabel('Predictions')
#         plt.show()
        
#     def PlotPredictionError(self, y_test, y_pred):
#         error = y_test - y_pred
#         plt.hist(error, bins=25)
#         plt.xlabel('Prediction Error')
#         plt.ylabel('Count')
#         plt.show()
        
#     def PlotPredictionError(self, y_test, y_pred):
#         error = y_test - y_pred
#         plt.hist(error, bins=25)
#         plt.xlabel('Prediction Error')
#         plt.ylabel('Count')
#         plt.show()
        
#     def PlotPredictionError(self, y_test, y_pred):
#         error = y_test - y_pred
#         plt.hist(error, bins=25)
#         plt.xlabel('Prediction Error')
#         plt.ylabel('Count')
#         plt.show()
        
#     def PlotPredictionError(self, y_test, y_pred):
#         error = y_test - y_pred