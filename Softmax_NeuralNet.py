import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------- Data Preparation ---------------------
df = pd.read_csv('Data/Diabetes.csv')
X = df.iloc[:, 2:-1]
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
num_features = len(X.columns)
X = X.astype(float).values
y = df.iloc[:, -1].str.strip()
unique_classes = y.unique()
y = pd.get_dummies(y).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# --------------------- PyTorch Softmax Classifier ---------------------
class SoftmaxClassifier():
    def __init__(self, num_features, lr, num_classes):
        self.num_features = num_features
        self.lr = lr
        self.num_classes = num_classes
        self.weights = torch.zeros(num_features, num_classes, dtype=torch.float64, device=device)
        self.bias = torch.zeros(1, num_classes, dtype=torch.float64, device=device)
    
    def softmax(self, net_input):
        exp = torch.exp(net_input)
        return exp / torch.sum(exp, dim=1, keepdim=True)
    
    def forward(self, X):
        net_input = torch.mm(X, self.weights) + self.bias 
        return self.softmax(net_input)   

    def backward(self, X, y):
        error = self.forward(X) - y
        grad_w = torch.mm(X.T, error) / X.shape[0]
        grad_b = torch.sum(error, dim=0, keepdim=True) / X.shape[0]
        return (-1) * grad_w, (-1) * grad_b

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        loss = -torch.sum(y * torch.log(y_pred + 1e-10)) / X.shape[0]
        return loss.item()

    def predict(self, X):
        y_prob = self.forward(X)
        return torch.argmax(y_prob, dim=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_true = torch.argmax(y, dim=1)
        acc = torch.sum(y_pred == y_true).item() / X.shape[0]
        return acc * 100

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

                neg_grad_w, neg_grad_b = self.backward(xi, yi)
                self.weights += self.lr * neg_grad_w
                self.bias += self.lr * neg_grad_b

            loss = self.compute_loss(X, y)
            acc = self.accuracy(X, y)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

# --------------------- PyTorch Training ---------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float64, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=device)

model_torch = SoftmaxClassifier(num_features, 0.01, len(unique_classes))
model_torch.train(X_train_tensor, y_train_tensor, epochs=100, batch_size=16)

train_acc_torch = model_torch.accuracy(X_train_tensor, y_train_tensor)
test_acc_torch = model_torch.accuracy(X_test_tensor, y_test_tensor)

print(f"\n[PyTorch] Train Accuracy: {train_acc_torch:.2f}%")
print(f"[PyTorch] Test Accuracy: {test_acc_torch:.2f}%")

# --------------------- Scikit-learn Logistic Regression ---------------------
# Convert one-hot targets to class indices
y_train_class = np.argmax(y_train, axis=1)
y_test_class = np.argmax(y_test, axis=1)

model_sklearn = LogisticRegression(max_iter=1000, solver='lbfgs')
model_sklearn.fit(X_train, y_train_class)

train_preds_sklearn = model_sklearn.predict(X_train)
test_preds_sklearn = model_sklearn.predict(X_test)

train_acc_sklearn = accuracy_score(y_train_class, train_preds_sklearn) * 100
test_acc_sklearn = accuracy_score(y_test_class, test_preds_sklearn) * 100

print(f"\n[Scikit-learn] Train Accuracy: {train_acc_sklearn:.2f}%")
print(f"[Scikit-learn] Test Accuracy: {test_acc_sklearn:.2f}%")
