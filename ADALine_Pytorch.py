import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('data/perceptron.csv')

X = df.drop(columns = ['Label']).values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

class ADALinePytorch():
    def __init__(self, num_features, lr):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float64, device=device)
        self.bias = torch.zeros(1, dtype=torch.float64, device=device)
        self.lr = torch.tensor(lr, dtype=torch.float64, device=device)
        self.ones = torch.ones(1, dtype=torch.float64, device=device)
        self.zeros = torch.zeros(1, dtype=torch.float64, device=device)

    def forward(self, x):
        linear = torch.mm(x, self.weights) + self.bias
        return linear.view(-1)
    
    def backward(self, x, y):
        error  = y - self.forward(x)  
        # Compute gradients
        grad_w = -2 * torch.matmul(x.T, error.reshape(-1, 1)) / x.shape[0]
        grad_b = -2 * torch.sum(error) / x.shape[0]
        return grad_w, grad_b
    
    def train(self, X, y, epochs, batch_size=10):
        num_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle indices
            indices = torch.randperm(num_samples, device=device)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                xi = X_shuffled[start:end]
                yi = y_shuffled[start:end]

                grad_w, grad_b = self.backward(xi, yi)

                # Update weights and bias
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

    def accuracy(self, X, y):
        predicted_values = torch.where(self.forward(X).reshape(-1) > 0, self.ones, self.zeros)
        accuracy = torch.sum(predicted_values == y)/ y.shape[0]
        return accuracy * 100        

    def visualization(self, X, y, data_name):
        weights_cpu = self.weights.cpu().numpy()
        bias_cpu = self.bias.cpu().numpy()
        X_cpu = X.cpu().numpy()
        y_cpu = y.cpu().numpy()
        fig, ax = plt.subplots()
        x0_min = X_cpu[:, 0].min()
        x1_min = ( (-(weights_cpu[0] * x0_min) - bias_cpu) 
                / weights_cpu[1] )

        x0_max = X_cpu[:, 0].max()
        x1_max = ( (-(weights_cpu[0] * x0_max) - bias_cpu) 
                / weights_cpu[1] )
        
        ax.scatter(X_cpu[y_cpu == 1, 0], X_cpu[y_cpu == 1, 1], label = 'class 1', marker = 's')
        ax.scatter(X_cpu[y_cpu ==0, 0], X_cpu[y_cpu == 0, 1], label = 'class 0', marker = 'o')
        ax.plot([x0_min, x0_max], [x1_min, x1_max])
        ax.set_title(f'{data_name}')
        ax.set_xlabel('feature 1')
        ax.set_ylabel('feature 2')
        plt.show()

X_train_tensor =  torch.tensor(X_train, dtype=torch.float64, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=device)

obj = ADALinePytorch(2, 0.01)
obj.train(X_train_tensor, y_train_tensor, 10, batch_size=10)

print(obj.accuracy(X_test_tensor, y_test_tensor))

obj.visualization(X_test_tensor, y_test_tensor, 'Test Dataset')

# ----------------- Analytical Solution -------------------

# Add bias term to X matrices
X_train_aug = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_aug = np.c_[np.ones(X_test.shape[0]), X_test]

# Closed-form solution
w_analytical = np.linalg.inv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

bias_analytical = w_analytical[0]
weights_analytical = w_analytical[1:]

# Predictions
y_pred_train = X_train @ weights_analytical + bias_analytical
y_pred_class_train = np.where(y_pred_train > 0, 1, 0)

y_pred_test = X_test @ weights_analytical + bias_analytical
y_pred_class_test = np.where(y_pred_test > 0, 1, 0)

# Accuracy
train_acc_analytical = (y_pred_class_train == y_train).mean() * 100
test_acc_analytical = (y_pred_class_test == y_test).mean() * 100

print(f"Analytical Solution Train Accuracy: {train_acc_analytical:.2f}%")
print(f"Analytical Solution Test Accuracy: {test_acc_analytical:.2f}%")
