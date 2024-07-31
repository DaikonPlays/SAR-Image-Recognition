import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchmetrics
from tqdm import tqdm
from torchvision import transforms, datasets, models
import torchhd
from torchhd.models import Centroid
from torchhd import embeddings
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
IMG_SIZE = 58
NUM_LEVELS = 8
BATCH_SIZE = 1  
class CustomResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img = img.convert("L")  
        img = transforms.ToTensor()(img)
        return img
transform = CustomResizeTransform(IMG_SIZE)
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     torchvision.transforms.ToTensor()
# ])

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.projection = nn.Linear(size * size, out_features, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0)
        # self.position = embeddings.Random(size * size, out_features)
        # self.value = embeddings.Level(l   evels, out_features)

    # def forward(self, x):
    #     x = self.flatten(x)
    #     sample_hv = torchhd.bind(self.position.weight, self.value(x))
    #     sample_hv = torchhd.multiset(sample_hv)   
    #     return torchhd.hard_quantize(sample_hv)
    def forward(self, x):
        x = self.flatten(x)
        x = self.projection(x)
        return torchhd.hard_quantize(x)

transformee = torchvision.transforms.ToTensor()
MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_TargetData/';  
dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transformee)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
total_size = len(dataset)
train_size = int(total_size * 0.8)  
test_size = total_size - train_size 
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
# train_ds = MNIST("../data", train=True, transform=transform, download=True)
# train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# test_ds = MNIST("../data", train=False, transform=transform, download=True)
# test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

for i, (sample, label) in enumerate(train_ld):
    print(f"Sample {i}: Image size: {sample.shape}")
    if i == 8:  # Show 3 images
        break
encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = 8
model = Centroid(DIMENSIONS, num_classes)
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