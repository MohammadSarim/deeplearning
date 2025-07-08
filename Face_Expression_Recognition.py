import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

##########################
### SETTINGS
##########################

RANDOM_SEED = 1
BATCH_SIZE = 100
NUM_EPOCHS = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


############################
### Face Expression Dataset
############################

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root="C:/Deep Learning Projects/Face Expression Recognition/images/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

validation_dataset = datasets.ImageFolder(root="C:/Deep Learning Projects/Face Expression Recognition/images/validation", transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = True)

# ------------------------ Checking Batch Dimensions ------------------------
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

# ------------------------ Model Definition ---------------------------------

class FaceExpressionRecognition(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(FaceExpressionRecognition, self).__init__()

        self.num_classes = num_classes

        ### First Hidden Layer
        self.linear_in = torch.nn.Linear(num_features, num_hidden)
        self.linear_in.weight.detach().normal_(0, 0.1)
        self.linear_in.bias.detach().zero_()

        ### Output Layer
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.detach().normal_(0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, X):
        out = self.linear_in(X)
        out = torch.relu(out)

        logits = self.linear_out(out)
        #probas = torch.softmax(logits, dim = 1)
        return logits
    
#################################
### Model Initialization
#################################

torch.manual_seed(RANDOM_SEED)
model = FaceExpressionRecognition(
    num_features= 48*48,
    num_hidden=100,
    num_classes=num_classes
)

model = model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

#################################
### Training
#################################

def compute_loss(model, data_loader):
    curr_loss = 0.
    with torch.no_grad():
        for cnt, (features, targets) in enumerate(data_loader):
            features = features.view(-1, 48*48).to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            curr_loss += loss
        return float(curr_loss)/cnt
 

start_time = time.time()
minibatch_cost = []
epoch_cost = []
for epoch in range(NUM_EPOCHS):
    for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.view(-1, 48*48).to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)

            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            optimizer.step()

            ### LOGGING
            minibatch_cost.append(cost.item())
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), cost.item()))
        
    cost = compute_loss(model, train_loader)
    epoch_cost.append(cost)
    print('Epoch: %03d/%03d Train Cost: %.4f' % (
            epoch+1, NUM_EPOCHS, cost))
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

plt.plot(range(len(epoch_cost)), epoch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.show()

def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 48*48).to(DEVICE)
            targets = targets.to(DEVICE)
            logits = net.forward(features)
            predicted_labels = torch.argmax(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100
    
print('Training Accuracy: %.2f' % compute_accuracy(model, train_loader))
print('Test Accuracy: %.2f' % compute_accuracy(model, validation_loader))