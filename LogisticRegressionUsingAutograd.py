import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

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

X_train_tensor = torch.tensor(X_train, dtype=torch.float64, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=device)

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1, dtype = torch.float64)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, X):
        logits = self.linear(X)
        probas = torch.sigmoid(logits)
        return probas
    
def comp_accuracy(label_var, pred_probas):
    pred_labels = torch.where((pred_probas > 0.5), 1, 0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
    return acc

model = LogisticRegression(num_features=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 5

for epoch in range(num_epochs):
    
    #### Compute outputs ####
    out = model(X_train_tensor)
    
    #### Compute gradients ####
    loss = F.binary_cross_entropy(out, y_train_tensor.view(-1, 1), reduction='sum')
    optimizer.zero_grad()
    loss.backward()
    
    #### Update weights ####  
    optimizer.step()
    
    #### Logging ####      
    pred_probas = model(X_train_tensor)
    acc = comp_accuracy(y_train_tensor, pred_probas)
    print('Epoch: %03d' % (epoch + 1), end="")
    print(' | Train ACC: %.3f' % acc, end="")
    print(' | Cost: %.3f' % F.binary_cross_entropy(pred_probas, y_train_tensor.view(-1, 1)))


    
print('\nModel parameters:')
print('  Weights: %s' % model.linear.weight)
print('  Bias: %s' % model.linear.bias)

with torch.no_grad():  
    pred_probas_test = model(X_test_tensor)
    test_acc = comp_accuracy(y_test_tensor, pred_probas_test)
    print(f"\nTest Accuracy: {test_acc:.3f}")