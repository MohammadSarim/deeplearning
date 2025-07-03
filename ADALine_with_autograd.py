import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- Data Preparation -------------------

df = pd.read_csv('data/perceptron.csv')

X = df.drop(columns=['Label']).values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train_tensor = torch.tensor(X_train, dtype=torch.float64, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=device)

# ----------------- Model Definition -------------------

class ADALine(torch.nn.Module):
    def __init__(self, num_features):
        super(ADALine, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1, dtype=torch.float64)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, X):
        net_inputs = self.linear(X)
        return net_inputs.view(-1)

# ----------------- Loss Function -------------------

def loss_func(yhat, y):
    return torch.mean((yhat - y) ** 2)

# ----------------- Training Function -------------------

def train(model, X, y, num_epochs, learning_rate=0.01, seed=123, minibatch_size=10):
    cost = []
    num_samples = X.shape[0]

    torch.manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        indices = torch.randperm(num_samples, device=device)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, num_samples, minibatch_size):
            end = start + minibatch_size
            xi = X_shuffled[start:end]
            yi = y_shuffled[start:end]

            yhat = model.forward(xi)
            loss = F.mse_loss(yhat, yi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            yhat = model.forward(X)
            curr_loss = loss_func(yhat, y)
            print(f'Epoch: {epoch + 1:03d} | MSE: {curr_loss:.5f}')
            cost.append(curr_loss.item())

    return cost

# ----------------- Model Training -------------------

model = ADALine(num_features=X_train_tensor.shape[1]).to(device)
cost = train(model, X_train_tensor, y_train_tensor, num_epochs=20, learning_rate=0.01, seed=123, minibatch_size=10)

# ----------------- Accuracy Calculation -------------------

with torch.no_grad():
    yhat_train = model.forward(X_train_tensor)
    yhat_test = model.forward(X_test_tensor)

    y_pred_train = torch.where(yhat_train > 0, 1.0, 0.0)
    y_pred_test = torch.where(yhat_test > 0, 1.0, 0.0)

    train_acc = torch.sum(y_pred_train == y_train_tensor) / y_train_tensor.shape[0] * 100
    test_acc = torch.sum(y_pred_test == y_test_tensor) / y_test_tensor.shape[0] * 100

print(f"\nTrain Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
