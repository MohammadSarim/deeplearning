import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------ Hyperparameters ------------------------
random_seed = 123
learning_rate = 0.1
num_epochs = 25
batch_size = 256

num_features = 784  # 28x28 flattened image
num_classes = 10

torch.manual_seed(random_seed)

# ------------------------ Dataset ------------------------
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# ------------------------ Checking Batch Dimensions ------------------------
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)  # NCHW format
    print('Label batch dimensions:', labels.shape)
    break

# ------------------------ Model Definition ------------------------

class MLP(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(MLP, self).__init__()
        
        ### First Hidden Layer
        self.num_classes = num_classes
        self.linear1 = torch.nn.Linear(num_features, num_hidden)
        self.linear1.weight.data.detach.normal(0, 0.1)
        self.linear1.bias.data.zero_()

        ### Output Layer
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.data.detach.normal(0, 0.1)
        self.linear1.bias.data.zero_()

    def forward(self, X):
        out = self.linear1(X)
        out = torch.sigmoid(out)

        logits = self.linear_out(out)
        probas = torch.softmax(logits, dim = 1)
        return logits, probas
    
model = MLP(num_features, num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ------------------------ Accuracy Function ------------------------
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    
    for features, targets in data_loader:
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100

# ------------------------ Training ------------------------

