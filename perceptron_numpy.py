import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.read_csv('data/perceptron.csv')

X = df.drop(columns = ['Label']).values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

class PerceptronNumpy():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.bias = 0

    def forward(self, x):
        net_input = np.dot(x, self.weights.T) + self.bias
        prediction = np.where(net_input > 0, 1, 0)
        return prediction

    def backward(self, x, y):
        error = y - self.forward(x)
        return error

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                error = self.backward(xi, yi)
                self.weights += error * xi
                self.bias += error
        print(self.bias, self.weights)

    def accuracy(self, X, y):
        predicted_values = self.forward(X)
        accuracy = np.sum(predicted_values == y)/ y.shape[0]
        return accuracy * 100
    
    def visualization(self, X, y, data_name):
        fig, ax = plt.subplots()
        x0_min = X[:, 0].min()
        x1_min = ( (-(self.weights[0] * x0_min) - self.bias) 
                / self.weights[1] )

        x0_max = X[:, 0].max()
        x1_max = ( (-(self.weights[0] * x0_max) - self.bias) 
                / self.weights[1] )
        ax.scatter(X[y == 1, 0], X[y == 1, 1], label = 'class 1', marker = 's')
        ax.scatter(X[y ==0, 0], X[y == 0, 1], label = 'class 0', marker = 'o')
        ax.plot([x0_min, x0_max], [x1_min, x1_max])
        ax.set_title(f'{data_name}')
        ax.set_xlabel('feature 1')
        ax.set_ylabel('feature 2')
        plt.show()
        
obj = PerceptronNumpy(2)
obj.train(X_train, y_train, 2)

obj.visualization(X_test, y_test, "Test dataset")
print(f"The accuracy of the model on the test set is: {obj.accuracy(X_test, y_test)}%")