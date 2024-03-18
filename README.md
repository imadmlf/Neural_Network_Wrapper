# taskes
# Testing Classes on a DataFrame

```python
import pandas as pd 
from DataPreprocessing import DataPreprocessing
from DataExploration import DataExploration
from ModelEvaluation import ModelEvaluation
from ModelTraining import ModelTraining
from neural_network import NeuralNetwork
import torch
```
# Read the dataset
```python
df = pd.read_csv('cancer_classification.csv')
```
# Test the DataPreprocessing class
```python
preprocessor = DataPreprocessing(df)
x_train, x_test, y_train, y_test = preprocessor.split_data(test_size=0.2, random_state=42)
x_train, x_test = preprocessor.normalize_data()
x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = preprocessor.tensorize_data()
```
# Test the DataExploration class
```python
explorer = DataExploration(df)
print("DataFrame Head")
explorer.DisplayData()
print("\nData Types")
explorer.DisplayDataTypes()
print("\nData Info")
explorer.DisplayDataInfo()
print("\nData Description")
explorer.DisplayDataDescription()
print("\nData Shape")
explorer.DisplayDataShape()
print("\nMissing Values")
explorer.DisplayMissingValues()
print("\nCorrelation Matrix")
explorer.DisplayCorrelationMatrix()
print("\nCorrelation with 'target' column:")
explorer.DisplayCorrelationWithColumn('benign_0__mal_1')
print("\nHeatMap")
explorer.DisplayHeatMap()
```
# Testing the NeuralNetwork class
```python
input_features = len(df.columns) - 1
neural_net = NeuralNetwork(input_features)
print("Neural Network Architecture:")
print(neural_net)
```
# Testing the ModelTraining class
```python
model = NeuralNetwork(input_features)
criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
trainer = ModelTraining(model, criterion, optimizer)
train_losses, test_losses = trainer.train(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, epochs=600)
```
# Testing the ModelEvaluation class
```python
evaluator = ModelEvaluation(model, criterion, optimizer)
model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor)
    y_pred = (y_pred > 0.5).float()
print("\nConfusion Matrix")
evaluator.confusion_matrix(y_test_tensor, y_pred)
print("\nClassification Report")
evaluator.classification_report(y_test_tensor, y_pred)
print("\nAccuracy Score")
accuracy = evaluator.accuracy_score(y_test_tensor, y_pred)
print("Accuracy:", accuracy)```

# Visualization
print("\nPrediction Visualization")
evaluator.plot_prediction(y_test_tensor, y_pred)
print("\nPrediction Error Distribution")
evaluator.plot_prediction_error(y_test_tensor, y_pred)

