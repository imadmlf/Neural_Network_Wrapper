import torch
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, epochs=600):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(x_train_tensor)
            outputs = outputs.squeeze(1)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(x_test_tensor)
                test_outputs = test_outputs.squeeze(1)
                test_loss = self.criterion(test_outputs, y_test_tensor)

            # Logging
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {test_loss.item():.4f}')

        return train_losses, test_losses
   
    
    def plot_loss(self, train_losses, test_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
