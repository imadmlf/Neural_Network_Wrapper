# this module contains all data exploration, analysis and visualization methods

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataExploration():
    def __init__(self, data):
        self.data = data
        
    def DisplayData(self):
        return self.data.head()
    
    def DisplayDataTypes(self):
        return self.data.dtypes
    
    def DisplayDataInfo(self):
        self.data.info()
    
    def DisplayDataDescription(self):
        return self.data.describe()
    
    def DisplayDataShape(self):
        return self.data.shape
    
    def DisplayMissingValues(self):
        return self.data.isnull().sum()
    
    def DisplayCorrelationMatrix(self):
        return self.data.corr()
    #correletion with a specific column
    def DisplayCorrelationWithColumn(self, column):
        return self.data.corrwith(self.data[column])
    
    def DisplayHeatMap(self):
        sns.heatmap(self.data.corr(), annot=True)
        plt.show()
    
    def DisplayPairPlot(self):
        sns.pairplot(self.data)
        plt.show()
    
    def DisplayCountPlot(self, column):
        sns.countplot(self.data[column])
        plt.show()
    
    def DisplayBoxPlot(self, column):
        sns.boxplot(self.data[column])
        plt.show()
       
    def DisplayScatterPlot(self, x, y):
        plt.scatter(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
    
    def DisplayHistogram(self, column):
        plt.hist(self.data[column], bins=25)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()