import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_PUBLIC_MIXED_TARGETS_CD1/';  
dataset = datasets.ImageFolder(root=MSTAR_dir, transform=transform)
datasets_list = []
degree_dirs = ['15_DEG', '16_DEG', '29_DEG', '31_DEG', '43_DEG', '44_DEG', '45_DEG']
for deg_val in degree_dirs :
    dir_path = os.path.join(MSTAR_dir, deg_val)
    dataset = datasets.ImageFolder(root=dir_path, transform=transform)
    datasets_list.append(dataset)
concatenated_dataset = ConcatDataset(datasets_list)
dataloader = DataLoader(concatenated_dataset, batch_size=32, shuffle=True)
print(len(concatenated_dataset))
def imshow(img, mean, std):
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean  
    img = np.clip(img, 0, 1) 
    plt.imshow(img)
    plt.show()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
for images, labels in dataloader:
    imshow(images[0], mean, std)
    print(labels)
    break
    

