import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchmetrics
from tqdm import tqdm
from torchvision import transforms, datasets, models
import torchhd
from torchhd.models import Centroid, IntRVFL
from torchhd import embeddings
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os
import argparse as ap
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
IMG_SIZE = 128
NUM_LEVELS = 1000
BATCH_SIZE = 50 
PATCH_SIZE = 32

def get_position_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return torch.tensor(P, dtype=torch.float32)

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels, patch_size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.patch_size = patch_size
        self.size = size
        self.num_patches = (size // patch_size) ** 2 
        self.position_encoding = get_position_encoding(self.num_patches, out_features)
        self.value_embedding = embeddings.Level(levels, out_features).to(device) 
        self.x_position_embedding = embeddings.Level(size, out_features).to(device) 
        self.y_position_embedding = embeddings.Level(size, out_features).to(device) 

    def pixel_encoding(self, image):
        batch_size, channels, height, width = image.shape
        print(batch_size)
        image_flat = image.view(batch_size, -1).long()  
        image_hv = torch.zeros((batch_size, DIMENSIONS), device=device)
        for y in range(height):
            for x in range(width):
                pixel_value = image_flat[:, y * width + x]  
                value_hv = self.value_embedding(pixel_value) 
                x_hv = self.x_position_embedding(torch.tensor(x, device=device))  
                y_hv = self.y_position_embedding(torch.tensor(y, device=device)) 
                pixel_hv = torchhd.bind(torchhd.bind(value_hv, x_hv), y_hv) 
                image_hv += pixel_hv 

        return image_hv

        # if args.type == 'linear':
        #     self.position = embeddings.Random(patch_size * patch_size, out_features)
        #     self.value = embeddings.Level(levels, out_features)
        # else:
        #     self.nonlinear_projection = embeddings.Sinusoid(patch_size * patch_size, out_features)
    def forward(self, x):
        sample_hv = self.pixel_encoding(x) 
        return torchhd.hard_quantize(sample_hv)
    # def forward(self, x):
    #     batch_size, channels, _, _ = x.size()
    #     x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    #     x = x.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
    #     x = x.view(-1, self.patch_size * self.patch_size)

    #     if args.type == 'linear':
    #         patches_hv = torchhd.bind(self.position.weight, self.value(x))
    #     else:
    #         patches_hv = self.nonlinear_projection(x)
    #     patches_hv += self.position_encoding
    #     patches_hv = patches_hv.view(batch_size, -1, patches_hv.size(-1))
    #     sample_hv = torchhd.multiset(patches_hv)
    #     return torchhd.hard_quantize(sample_hv)

# transform = torchvision.transforms.ToTensor()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

parser = ap.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--type', type=str, default='linear')
args = parser.parse_args()

if args.dataset == 'mstar':
    MSTAR_dir = r'mstar'
    dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_size = len(dataset)
    train_size = int(total_size * 0.8)  
    test_size = total_size - train_size 
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    num_classes = len(dataset.classes)
else:
    train_ds = MNIST("../data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = MNIST("../data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    num_classes = len(train_ds.classes)

encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS, PATCH_SIZE)
encode = encode.to(device)

model = Centroid(in_features=DIMENSIONS, out_features=num_classes)
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)
#check channels in the tensor
#try projection level encoding
        samples_hv = encode(samples)
        model.add(samples_hv, labels)

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

with torch.no_grad():
    model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")