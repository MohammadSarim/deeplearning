import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import PIL

##########################
### SETTINGS
##########################

RANDOM_SEED = 1
BATCH_SIZE = 100
NUM_EPOCHS = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

############################
### Transforms with Standardization
############################

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),           #===========>>>>>> Data Augmentation(To handle Overfitting)
    transforms.RandomCrop(size = (48, 48)),
    transforms.RandomRotation(degrees = 30, interpolation = PIL.Image.BILINEAR),  #========>>>>>>>>>>>>>>> Data Augmentation(To handle Overfitting)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # [-1, 1] normalization
])

validation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.CenterCrop(size = (48, 48)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

############################
### Dataset and DataLoader
############################

train_dataset = datasets.ImageFolder(
    root="C:/Deep Learning Projects/Face Expression Recognition/images/train",
    transform=  train_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = datasets.ImageFolder(
    root="C:/Deep Learning Projects/Face Expression Recognition/images/validation",
    transform=validation_transform
)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

############################
### MLP Model Definition
############################

class FaceExpressionRecognition(torch.nn.Module):
    def __init__(self, num_features, num_hidden, drop_proba, num_classes):
        super(FaceExpressionRecognition, self).__init__()
        self.num_classes = num_classes

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, num_hidden, bias = True),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(num_hidden), #===================>>>>>>> BatchNorm
            torch.nn.Dropout(drop_proba),  #====================>>>>> Dropout
            torch.nn.Linear(num_hidden, num_classes)
        )

    def forward(self, X):
        return self.model(X)

############################
### Model Setup
############################

torch.manual_seed(RANDOM_SEED)
model = FaceExpressionRecognition(
    num_features=48*48,
    num_hidden=100,
    drop_proba=0.2,
    num_classes=num_classes
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

############################
### Accuracy Function
############################

def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = net(features)
            predicted_labels = torch.argmax(logits, 1)
            correct_pred += (predicted_labels == targets).sum()
            num_examples += targets.size(0)
    return correct_pred.float() / num_examples * 100

############################
### Reuse Your compute_loss Function
############################

def compute_loss(model, data_loader):
    total_loss = 0.
    with torch.no_grad():
        for cnt, (features, targets) in enumerate(data_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss
        return float(total_loss) / cnt

############################
### Training Loop
############################

start_time = time.time()
minibatch_cost = []
epoch_cost = []
train_accuracies = []     # <<< For Accuracy Chart
val_accuracies = []       # <<< For Accuracy Chart

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(features)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        minibatch_cost.append(loss.item())

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS} | Batch: {batch_idx:03d}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    model.eval()
    train_loss = compute_loss(model, train_loader)
    val_loss = compute_loss(model, validation_loader)
    epoch_cost.append(train_loss)

    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, validation_loader)
    train_accuracies.append(train_acc.item())     # <<< Track Train Accuracy
    val_accuracies.append(val_acc.item())         # <<< Track Validation Accuracy

    print(f'Epoch {epoch+1:03d}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

############################
### Plotting
############################

plt.plot(minibatch_cost)
plt.ylabel('Minibatch Loss')
plt.xlabel('Iteration')
plt.title('Minibatch Loss Curve')
plt.show()

plt.plot(epoch_cost)
plt.ylabel('Epoch Loss')
plt.xlabel('Epoch')
plt.title('Epoch Loss Curve')
plt.show()

# <<< Accuracy vs Epoch >>>
plt.plot(range(1, NUM_EPOCHS+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, NUM_EPOCHS+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

############################
### Final Accuracy
############################

print('Final Training Accuracy: %.2f%%' % compute_accuracy(model, train_loader))
print('Final Validation Accuracy: %.2f%%' % compute_accuracy(model, validation_loader))
