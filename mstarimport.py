import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])
MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_DATASET/';  
dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transform)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataset.class_to_idx)

# def imshow(img, mean, std):
#     img = img.numpy().transpose((1, 2, 0))
#     img = std * img + mean  
#     img = np.clip(img, 0, 1) 
#     plt.imshow(img)
#     plt.show()
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# for images, labels in dataloader:
#     imshow(images[0], mean, std)
#     print(labels)
total_size = len(dataset)
train_size = int(total_size * 0.8)  
test_size = total_size - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
model = models.resnet18(pretrained=True)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for name, param in model.named_parameters():
    if "fc" in name:  
        param.requires_grad = True
    else:
        param.requires_grad = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9) 
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # running_corrects = 0
    for images, labels in train_dataset:
        images = images.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels) 
        labels = labels.to(device)  
        optimizer.zero_grad()
        outputs = model(images.unsqueeze(0))
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
        for images, labels in DataLoader(test_dataset, batch_size=32, shuffle=False):
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
    

