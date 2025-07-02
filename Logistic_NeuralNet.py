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

class LogisticRegression():
    def __init__(self, num_features, lr):
        self.num_features= num_features
        self.lr = lr
        self.weights = torch.zeros(num_features, 1, dtype = torch.float64, device = device)
        self.bias = torch.zeros(1, dtype = torch.float64, device=device)
        self.ones = torch.ones(1, dtype=torch.float64, device=device)
        self.zeros = torch.zeros(1, dtype=torch.float64, device=device)

    def sigmoid(self, net_input):
        return 1/(1+torch.exp(-net_input))
    
    def loss(self, X, y):
        y_pred = self.forward(X)
        # Adding epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        cost = -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        return cost

    def forward(self, x):
        net_input = torch.mm(x, self.weights) + self.bias
        sigmoid = self.sigmoid(net_input)
        return sigmoid.view(-1)

    def backward(self, x, y):
        error = self.forward(x) - y
        grad_w = torch.mm(x.T, error.view(-1, 1)) / x.shape[0]
        grad_b = torch.sum(error) / x.shape[0]
        return (-1)*grad_w, (-1)*grad_b
    
    def train(self, X, y, epochs, batch_size):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = torch.randperm(num_samples, device=device)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                xi = X_shuffled[start:end]
                yi = y_shuffled[start:end]

                negative_grad_w, negative_grade_b = self.backward(xi, yi)
                self.weights += self.lr * negative_grad_w
                self.bias += self.lr * negative_grade_b
            # Cost after each epoch
            cost = self.loss(X, y)
            print(f"Epoch {epoch+1}, Cost: {cost.item():.4f}")

    def accuracy(self, X, y):
        predicted_values = torch.where(self.forward(X).reshape(-1) > 0.5, self.ones, self.zeros)
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

obj = LogisticRegression(2, 0.01)
obj.train(X_train_tensor, y_train_tensor, 5, 10)                    

print(obj.accuracy(X_test_tensor, y_test_tensor))
obj.visualization(X_test_tensor, y_test_tensor, "Test dataset")