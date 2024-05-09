import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
class AddGaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
     AddGaussianBlur(radius=3), # WHY 224x224? I BELIEVE THE IMAGES ARE SMALLER (128x128). STILL, IF THEY HAVE VARIABLE SIZE, KEEP THIS RESIZE TO 224x224
    transforms.ToTensor(), 
    # WHERE DID YOU GET THIS NUMBERS? IF THESE ARE NOT THE GOOD VALUES FOR THE MSTAR DATASET, REMOVE THE NORMALIZATION
])
MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_TargetData/';  
dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transform)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataset.class_to_idx)
total_size = len(dataset)
train_size = int(total_size * 0.8)  
test_size = total_size - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = models.resnet18(pretrained=False)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for name, param in model.named_parameters():
    if "fc" in name:  
        param.requires_grad = True
    else:
        param.requires_grad = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 
num_epochs = 50 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # running_corrects = 0
    for images, labels in train_loader: #WHY DON'T YOU USE HERE THE DATALOADER AS YOU DO FOR THE TESTING LOOP? I WOULD COPY WHAT YOU HAVE BELOW TO GET IMAGES/LABEL/OUTPUTS
        images = images.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels) 
        labels = labels.to(device)  
        optimizer.zero_grad()
        outputs = model(images)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print(images)
        #print('Loss: {:.4f}'.format(loss.item()))        
        # running_corrects += torch.sum(preds == labels.data)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total}%')
    print(f'Average loss on the test dataset: {test_loss / len(DataLoader(test_dataset))}')
