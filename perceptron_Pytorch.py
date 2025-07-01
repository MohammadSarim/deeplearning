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

class PerceptronPytorch():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float64, device=device)
        self.bias = torch.zeros(1, dtype=torch.float64, device=device)
        self.ones = torch.ones(1, dtype=torch.float64, device=device)
        self.zeros = torch.zeros(1, dtype=torch.float64, device=device)

    def forward(self, x):
        net_input = torch.mm(x, self.weights) + self.bias
        predicted_value = torch.where(net_input > 0, self.ones, self.zeros)
        return  predicted_value
    
    def backward(self, x, y):
        error = y - self.forward(x)
        return error
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for xi, yi in zip(X,y):
                error = self.backward(xi.reshape(1, self.num_features), yi).reshape(-1)
                self.weights += (xi * error).reshape(self.num_features, 1)
                self.bias += error
        print(self.weights, self.bias)

    def accuracy(self, X, y):
        predicted_values = self.forward(X).reshape(-1)
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

obj = PerceptronPytorch(2)
obj.train(X_train_tensor, y_train_tensor, 2)

print(obj.accuracy(X_test_tensor, y_test_tensor))

obj.visualization(X_test_tensor, y_test_tensor, 'Test Dataset')