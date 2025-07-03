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
class SoftmaxClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, X):
        logits = self.linear(X)
        probas = torch.softmax(logits, dim=1)
        return logits, probas

model = SoftmaxClassifier(num_features, num_classes).to(device)
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
start_time = time.time()
epoch_costs = []

for epoch in range(num_epochs):
    avg_cost = 0.

    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost.item()

        if not batch_idx % 50:
            print(f'Epoch: {epoch+1:03d}/{num_epochs} | Batch: {batch_idx:03d}/{len(train_loader)} | Cost: {cost.item():.4f}')
    
    avg_cost /= len(train_loader)
    epoch_costs.append(avg_cost)

    train_acc = compute_accuracy(model, train_loader)
    print(f'Epoch: {epoch+1:03d}/{num_epochs} | Train Accuracy: {train_acc:.2f}% | Avg Cost: {avg_cost:.4f}')
    print(f'Time Elapsed: {(time.time() - start_time)/60:.2f} min')

# ------------------------ Plotting Cost ------------------------
plt.plot(epoch_costs)
plt.ylabel('Avg Cross Entropy Loss (per minibatch)')
plt.xlabel('Epoch')
plt.title('Training Loss Curve')
plt.show()

# ------------------------ Final Test Accuracy ------------------------
test_acc = compute_accuracy(model, test_loader)
print(f'Test Accuracy: {test_acc:.2f}%')

# ------------------------ Visualize Predictions ------------------------
for features, targets in test_loader:
    break

fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(features[i].view(28, 28).cpu(), cmap=matplotlib.cm.binary)
    ax[i].axis('off')

plt.show()

features_subset = features[:4].view(-1, 28*28).to(device)
_, probas = model(features_subset)
predictions = torch.argmax(probas, dim=1).cpu()

print('Predicted Labels:', predictions.numpy())
print('Actual Labels   :', targets[:4].numpy())
