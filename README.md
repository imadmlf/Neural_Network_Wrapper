# Neural Network Wrapper [dataset](https://github.com/imadmlf/taskes/blob/main/cancer_classification.csv)
## Training on a simple dataset


1. **[DataPreprocessing](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py)**: This module likely contains functions or classes for preparing your raw data for analysis. This can include tasks such as handling missing values, encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets.

2. **[DataExploration](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataExploration.py)**: This part of your pipeline focuses on understanding the structure and characteristics of your dataset. It might include functions or classes for displaying basic statistics (like mean, median, standard deviation), visualizations (like histograms, scatter plots, or correlation matrices), and checking for any anomalies or inconsistencies in the data.

3. **[ModelTraining](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/modeltrainer.py)**: Here, you're training a machine learning model on your preprocessed data. This typically involves selecting an appropriate algorithm (like a neural network), defining a loss function, and optimizing model parameters using an optimization algorithm (like stochastic gradient descent).

4. **[ModelEvaluation](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/ModelEvaluation.py)**: After training your model, you need to evaluate its performance. This module likely contains functions or classes for computing various evaluation metrics (like accuracy, precision, recall, or F1-score), generating confusion matrices, and visualizing prediction results.

5. **[NeuralNetwork](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py)**: This appears to be a class for defining a neural network architecture using the PyTorch library. It specifies the layers, activation functions, and connections between neurons in the network.


```python
import pandas as pd 
from DataPreprocessing import DataPreprocessing
from DataExploration import DataExploration
from ModelEvaluation import ModelEvaluation
from ModelTraining import ModelTraining
from neural_network import NeuralNetwork
import torch
```
## Read the dataset
```python
df = pd.read_csv('cancer_classification.csv')
```
## Test the [DataPreprocessing](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py) class
The `preprocessor` object is created using the [DataPreprocessing](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py) class, which prepares the data for training a machine learning model. After splitting the data into training and testing sets using the [split_data()](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py#LX) 
method, it normalizes the data with [normalize_data()](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py#LX). Finally, it converts the data into tensors with [tensorize_data()](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py#LX), ready for model training and evaluation.
```python
preprocessor = DataPreprocessing(df)
x_train, x_test, y_train, y_test = preprocessor.split_data(test_size=0.2, random_state=42)
x_train, x_test = preprocessor.normalize_data()
x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = preprocessor.tensorize_data()
```
## Test the [DataExploration](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataExploration.py) class


-  [DisplayData()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays the head of the DataFrame.
```python
explorer = DataExploration(df)
print("DataFrame Head")
explorer.DisplayData()
```
- [DisplayDataTypes()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays the data types of columns in the DataFrame.
```python
print("\nData Types")
explorer.DisplayDataTypes()
```
- [DisplayDataInfo()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays general information about the DataFrame.
```python
print("\nData Info")
explorer.DisplayDataInfo()
```
- [DisplayDataDescription()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays statistical descriptions of the data.
```python
print("\nData Description")
explorer.DisplayDataDescription()
```
- [DisplayDataShape()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays the shape of the DataFrame.
```python
print("\nData Shape")
explorer.DisplayDataShape()
```
- [DisplayMissingValues()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays information about missing values in the DataFrame.
```python
print("\nMissing Values")
explorer.DisplayMissingValues()
```
- [DisplayCorrelationMatrix()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays the correlation matrix of numerical features in the DataFrame.
```python
print("\nCorrelation Matrix")
explorer.DisplayCorrelationMatrix()
```
- [DisplayCorrelationWithColumn('benign_0__mal_1')](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays the correlation of all features with the target column named `'benign_0__mal_1'`.
```python

print("\nCorrelation with 'target' column:")
explorer.DisplayCorrelationWithColumn('benign_0__mal_1')
```

- [DisplayHeatMap()](https://github.com/imadmlf/taskes/Neural_Network_Wrapper/main/DataExploration.py#LX): Displays a heatmap of the correlation matrix.
```python
print("\nHeatMap")
explorer.DisplayHeatMap()
```
## Testing the [NeuralNetwork](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py) class

This code snippet tests the [NeuralNetwork](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py) class. It calculates the number of input features by subtracting 1 from the total number of columns in the DataFrame (`df`). Then, it instantiates a `neural_net` object using the `NeuralNetwork` class, passing the calculated number of input features. Finally, it prints the architecture of the neural network by displaying the `neural_net` object.
```python
input_features = len(df.columns) - 1
out_features=df['benign_0__mal_1'].unique().sum()
neural_net = NeuralNetwork(input_features,out_features)
print("Neural Network Architecture:")
print(neural_net)
```
## Testing the [ModelTraining](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/modeltrainer.py) class

This code snippet instantiates a neural network model (`model`) using the [NeuralNetwork](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py) class, specifying the number of input features. It defines a binary cross-entropy loss function (`criterion`) and a stochastic gradient descent optimizer (`optimizer`) to train the model. 

Then, it creates a `trainer` object using the [ModelTraining](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/modeltrainer.py) class, passing the model, criterion, and optimizer as arguments. Finally, it trains the model using the `train()` method of the `trainer` object, passing the training and testing data tensors (`x_train_tensor`, `y_train_tensor`, `x_test_tensor`, `y_test_tensor`) and specifying the number of epochs (600).
```python
from torch import nn
model = neural_net
criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
```
```python
from modeltrainer import ModelTrainer
trainer = ModelTrainer(model, criterion, optimizer)
train_losses, test_losses = trainer.train(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, epochs=600)
```

## Testing the [ModelEvaluation](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/ModelEvaluation.py) class
This code snippet evaluates the trained neural network model (`model`) using the [ModelEvaluation](https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/ModelEvaluation.py) class. First, it sets the model to evaluation mode using `model.eval()`. Then, it generates predictions (`y_pred`) for the test data (`x_test_tensor`) using the trained model. The predictions are thresholded at 0.5 to convert probabilities to binary predictions.

Next, it prints the confusion matrix, classification report, and accuracy score using the `confusion_matrix()`, `classification_report()`, and `accuracy_score()` methods of the `evaluator` object, respectively.

After that, it performs visualization by plotting prediction results and prediction error distribution using the `plot_prediction()` and `plot_prediction_error()` methods of the `evaluator` object.
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

