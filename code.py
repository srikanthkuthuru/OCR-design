import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.utils import save_image
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import os
import time
import copy

#Read Data
rootdir= '/Users/srikanthkuthuru/Downloads/DevanagariHandwrittenCharacterDataset/'
a = os.listdir(rootdir+'train/')
data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ])
dev_dataset = {x: datasets.ImageFolder(root=rootdir+x+'/',transform=data_transform)
for x in ['train', 'test']}

dataset_loader = {x: DataLoader(dev_dataset[x], batch_size=256, shuffle=True,
                                             num_workers=1)
for x in ['train', 'test']}

#---------------------------------------------------#
# Train model function
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        #scheduler.step()
        model.train(True) 
        
        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for data in dataset_loader['train']:
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print([preds, labels.data])
            #input("Press Enter to continue...")
        
        print(zip(preds.tolist(), labels.data.tolist()))
        epoch_loss = running_loss / len(dev_dataset['train'])
        epoch_acc = running_corrects / len(dev_dataset['train'])
        
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

def test_model(model):
    running_corrects = 0.0
    for data in dataset_loader['test']:
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects / len(dev_dataset['test'])
    print('Test Acc: {:.4f}'.format(acc))
    return

        
#-----------------------------------------------------------#
# LeNet-5
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 2)
        self.fc1   = nn.Linear(16*8*8, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 46)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out
    
    
model_lenet = LeNet()
criterion = nn.CrossEntropyLoss()
#optimizer_lenet = optim.SGD(model_lenet.parameters(), lr=0.001, momentum=0.9)
optimizer_lenet = optim.Adam(model_lenet.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_lenet, step_size=7, gamma=0.1)

model_lenet = train_model(model_lenet, criterion, optimizer_lenet, num_epochs=5)
    
test_model(model_lenet)    
    
 #%%   
#--------------------------------------------------------#    
# Transfer Learning
model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
    
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 46) #we have 46 classes
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
