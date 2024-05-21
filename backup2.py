import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchhd
import torchmetrics
from tqdm import tqdm
from torchhd.models import Centroid
from torchhd import embeddings
from PIL import Image, ImageFilter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMENSIONS = 10000
IMG_SIZE = 128
NUM_LEVELS = 1000

BATCH_SIZE = 1
class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.id = embeddings.Random(size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

    def forward(self, x):
        sample_hv = torchhd.bind(self.id.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

class AddGaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    AddGaussianBlur(radius=3), 
    transforms.ToTensor(), 
])
input_features = IMG_SIZE * IMG_SIZE  
#encoder = Encoder(input_features, DIMENSIONS).to(device)
MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_TargetData/';  
dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transform)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataset.class_to_idx)
total_size = len(dataset)
train_size = int(total_size * 0.8)  
test_size = total_size - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
encode = Encoder(DIMENSIONS, train_dataset[0][0].numel()).to(device)
model = Centroid(DIMENSIONS, 8)
model = model.to(device)
with torch.no_grad():
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        hypervectors = encode(images)
        model.add(hypervectors, labels)
with torch.no_grad():
    model.normalize()
accuracy = torchmetrics.Accuracy("multiclass", num_classes=8).to(device)

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        hypervectors = encode(images)
        outputs = model(hypervectors, dot=True)
        print("outputs")
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy: {accuracy.compute().item() * 100:.2f}%")











# model = models.resnet18(pretrained=False)    
# model.to(device)
# for name, param in model.named_parameters():
#     if "fc" in name:  
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001) 
# num_epochs = 50 
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     # running_corrects = 0
#     for hypervector, labels in train_loader: 
#         hypervector = hypervector.to(device)
#         if not isinstance(labels, torch.Tensor):
#             labels = torch.tensor(labels) 
#         labels = labels.to(device)  
#         optimizer.zero_grad()
#         outputs = model(hypervector)
#         outputs = torch.squeeze(outputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         #print(images)
#         #print('Loss: {:.4f}'.format(loss.item()))        
#         # running_corrects += torch.sum(preds == labels.data)
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for hypervector, labels in test_loader:
#             hypervector = hypervector.to(device)
#             labels = labels.to(device)
#             outputs = model(hypervector)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print(f'Accuracy of the model on the test images: {100 * correct / total}%')
#     print(f'Average loss on the test dataset: {test_loss / len(DataLoader(test_dataset))}')
