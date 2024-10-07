import torch
from PIL import Image
from torchvision import transforms

MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_TargetData/D7/HB20013.JPG';  
image = Image.open(MSTAR_dir)
transform = transforms.Compose([
    transforms.ToTensor(),  
])
x = transform(image)

# x = torch.randn(1, 1, 128, 128)
print(x)
unfolded_x = x.unfold(2, 32, 32).unfold(3, 32, 32)

print(unfolded_x)
