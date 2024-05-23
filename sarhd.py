import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchhd
import torchvision
import torchmetrics
from tqdm import tqdm
from torchhd.models import Centroid
from torchhd import embeddings
from PIL import Image, ImageFilter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIMENSIONS = 1000
IMG_SIZE = 128
NUM_LEVELS = 100

BATCH_SIZE = 32
class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        y = self.value(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])
input_features = IMG_SIZE * IMG_SIZE  
MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_TargetData/';  
dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transform)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(dataset.class_to_idx)
total_size = len(dataset)
train_size = int(total_size * 0.8)  
test_size = total_size - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)
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
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy: {accuracy.compute().item() * 100:.2f}%")