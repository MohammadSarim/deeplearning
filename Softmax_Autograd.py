import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# --------------------- Setup ---------------------
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------- Data Preparation ---------------------
df = pd.read_csv('Data/Diabetes.csv')

X = df.iloc[:, 2:-1]
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
num_features = X.shape[1]
X = X.astype(float).values

y = df.iloc[:, -1].str.strip()
y = y.astype('category').cat.codes.values  # Convert to 0,1,2,...
num_classes = len(np.unique(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# --------------------- Model ---------------------
class Softmax(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Softmax, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, X):
        logits = self.linear(X)
        return logits
    
    def accuracy(self, X, y):
        logits = self.forward(X)
        y_pred = torch.argmax(logits, dim=1)
        acc = torch.sum(y_pred == y).item() / X.shape[0]
        return acc * 100

# --------------------- Training with Minibatches ---------------------
model = Softmax(num_features, num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
batch_size = 16
num_samples = X_train_tensor.shape[0]

for epoch in range(num_epochs):

    indices = torch.randperm(num_samples, device=device)
    X_shuffled = X_train_tensor[indices]
    y_shuffled = y_train_tensor[indices]

    epoch_loss = 0.0
    correct = 0

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.shape[0]
        y_pred = torch.argmax(logits, dim=1)
        correct += torch.sum(y_pred == y_batch).item()

    avg_loss = epoch_loss / num_samples
    acc = correct / num_samples * 100

    print(f"Epoch: {epoch + 1:03d} | Train ACC: {acc:.3f}% | Loss: {avg_loss:.4f}")

# --------------------- Test Accuracy ---------------------
with torch.no_grad():
    test_acc = model.accuracy(X_test_tensor, y_test_tensor)
    print(f"\nTest Accuracy: {test_acc:.3f}%")
